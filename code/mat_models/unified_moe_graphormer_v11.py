# -*- coding: utf-8 -*-
"""
unified_moe_graphormer_v11.py

v11 模型结构：
- ConvLayer: 基于你提供的 CGCNN 卷积，但用 LayerNorm 替代 BatchNorm，适配 batch_size=1
- Graphormer 编码模块：CentralityEncoding + MultiHeadAttention + Encoder
- StructEncoderV11：Graphormer + CGCNN 结构编码器，输出 graph embedding (embed_dim)
- CompEncoderV11：多层 MLP 成分编码器，输出 embed_dim
- SingleBranchModelV11：单分支多任务模型（只用成分或只用结构），用于 Stage1 分组预训练
- MoEUnifiedModelV11：MoE 总模型，支持成分 / 结构 MoE 专家 + gating + 每任务独立 head，
  用于 Stage2 统一 13 任务训练
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import degree


# ============================================================
# 1. 基础 ConvLayer（改为 LayerNorm，避免 batch_size=1 问题）
# ============================================================

class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs (改写版，使用 LayerNorm)
    """

    def __init__(self, atom_fea_len: int, nbr_fea_len: int):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len

        self.fc_full = nn.Linear(
            2 * self.atom_fea_len + self.nbr_fea_len,
            2 * self.atom_fea_len,
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        # 用 LayerNorm 替代 BatchNorm，更适合 N=1 场景
        self.ln1 = nn.LayerNorm(2 * self.atom_fea_len)
        self.ln2 = nn.LayerNorm(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea: Tensor, nbr_fea: Tensor, nbr_fea_idx: Tensor) -> Tensor:
        """
        atom_in_fea: (N, atom_fea_len)
        nbr_fea:     (N, M, nbr_fea_len)
        nbr_fea_idx: (N, M)
        """
        N, M = nbr_fea_idx.shape
        # 邻居原子特征
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]  # (N, M, atom_fea_len)
        total_nbr_fea = torch.cat(
            [
                atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
                atom_nbr_fea,
                nbr_fea,
            ],
            dim=2,
        )  # (N, M, 2*atom_fea_len + nbr_fea_len)

        total_gated_fea = self.fc_full(total_nbr_fea)  # (N, M, 2*atom_fea_len)
        total_gated_fea = self.ln1(total_gated_fea)    # LN over last dim

        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)  # (N, M, atom_fea_len)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)

        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)  # (N, atom_fea_len)
        nbr_sumed = self.ln2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


# ============================================================
# 2. Graphormer 组件
# ============================================================

class CentralityEncoding(nn.Module):
    def __init__(self, max_in_degree: int, max_out_degree: int, node_dim: int):
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.node_dim = node_dim
        self.z_in = nn.Parameter(torch.randn((max_in_degree, node_dim)))
        self.z_out = nn.Parameter(torch.randn((max_out_degree, node_dim)))

    @staticmethod
    def _clamp_degree(x: Tensor, max_value: int) -> Tensor:
        return torch.clamp(x, max=max_value)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        x: (N, node_dim)
        edge_index: (2, E)
        """
        num_nodes = x.shape[0]
        in_deg = degree(edge_index[1], num_nodes=num_nodes).long()
        out_deg = degree(edge_index[0], num_nodes=num_nodes).long()

        in_deg = self._clamp_degree(in_deg, self.max_in_degree - 1)
        out_deg = self._clamp_degree(out_deg, self.max_out_degree - 1)

        x = x + self.z_in[in_deg] + self.z_out[out_deg]
        return x


class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, x: Tensor, edge_index: Tensor, ptr: Optional[Tensor] = None) -> Tensor:
        """
        x: (N, dim_in)
        edge_index: (2, E)
        ptr: (num_graphs+1,) graph pointer
        """
        query = self.q(x)  # (N, dim_q)
        key = self.k(x)    # (N, dim_k)
        value = self.v(x)  # (N, dim_k)

        N = x.size(0)
        # adjacency matrix (允许自环)
        adjacency = torch.zeros(N, N, device=x.device)
        adjacency[edge_index[0], edge_index[1]] = 1.0
        adjacency = adjacency + torch.eye(N, device=x.device)

        if ptr is None:
            attn_logits = query @ key.transpose(0, 1) / math.sqrt(query.size(-1))
        else:
            attn_logits = torch.zeros((N, N), device=x.device)
            for i in range(len(ptr) - 1):
                s, e = ptr[i].item(), ptr[i + 1].item()
                q_i = query[s:e]
                k_i = key[s:e]
                attn_logits[s:e, s:e] = q_i @ k_i.transpose(0, 1) / math.sqrt(q_i.size(-1))

        # mask 非邻接元素
        attn_logits = attn_logits.masked_fill(adjacency == 0, -1e6)
        attn = torch.softmax(attn_logits, dim=-1)
        out = attn @ value
        return out


class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.heads = nn.ModuleList([
            GraphormerAttentionHead(dim_in, dim_q, dim_k)
            for _ in range(num_heads)
        ])
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, x: Tensor, edge_index: Tensor, ptr: Optional[Tensor]) -> Tensor:
        head_outs = [head(x, edge_index, ptr) for head in self.heads]
        concatenated = torch.cat(head_outs, dim=-1)
        out = self.linear(concatenated)
        return out


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim: int, num_heads: int, max_path_distance: int = 5):
        super().__init__()
        self.attn = GraphormerMultiHeadAttention(
            num_heads=num_heads,
            dim_in=node_dim,
            dim_q=node_dim,
            dim_k=node_dim,
        )
        self.ln1 = nn.LayerNorm(node_dim)
        self.ln2 = nn.LayerNorm(node_dim)
        self.ff = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )

    def forward(self, x: Tensor, edge_index: Tensor, ptr: Optional[Tensor]) -> Tensor:
        x_1 = self.attn(self.ln1(x), edge_index, ptr) + x
        x_2 = self.ff(self.ln2(x_1)) + x_1
        return x_2


class GraphormerEncoder(nn.Module):
    def __init__(self, layers: int, node_dim: int, num_heads: int, max_path_distance: int = 5):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(node_dim, num_heads, max_path_distance)
            for _ in range(layers)
        ])

    def forward(self, x: Tensor, edge_index: Tensor, ptr: Optional[Tensor]) -> Tensor:
        for layer in self.layers:
            x = layer(x, edge_index, ptr)
        return x


# ============================================================
# 3. 结构编码器：Graphormer + CGCNN backbone
# ============================================================

class StructEncoderV11(nn.Module):
    """
    结构编码器：
    - 输入：原子特征 atom_fea (N, atom_dim), 邻接特征 nbr_fea (N, M, edge_dim),
            邻接索引 nbr_idx (N, M), crystal_atom_idx (list[tensor] or tensor)
    - 输出：每个晶体的 embedding (num_crystals, embed_dim)
    """

    def __init__(
        self,
        orig_atom_fea_len: int,
        nbr_fea_len: int,
        atom_fea_len: int = 64,
        n_conv: int = 3,
        graphormer_layers: int = 1,
        num_heads: int = 4,
        max_path_distance: int = 5,
        h_fea_len: int = 256,
        n_h: int = 1,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([
            ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
            for _ in range(n_conv)
        ])
        self.centrality_encoding = CentralityEncoding(
            max_in_degree=10,
            max_out_degree=10,
            node_dim=atom_fea_len,
        )
        self.graphormer_encoder = GraphormerEncoder(
            layers=graphormer_layers,
            node_dim=atom_fea_len,
            num_heads=num_heads,
            max_path_distance=max_path_distance,
        )
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.act = nn.Softplus()
        self.n_h = n_h
        if n_h > 1:
            self.fcs = nn.ModuleList([
                nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)
            ])
            self.acts = nn.ModuleList([
                nn.ReLU() for _ in range(n_h - 1)
            ])
        else:
            self.fcs = None
            self.acts = None

        # 最终将 h_fea_len 映射到 embed_dim
        if h_fea_len != embed_dim:
            self.proj = nn.Linear(h_fea_len, embed_dim)
        else:
            self.proj = nn.Identity()

    @staticmethod
    def _build_edge_index(nbr_idx: Tensor) -> Tensor:
        """
        nbr_idx: (N, M)  neighbor indices
        return: edge_index (2, E)
        """
        device = nbr_idx.device
        N, M = nbr_idx.shape
        src = torch.arange(N, device=device).repeat_interleave(M)  # (N*M,)
        dst = nbr_idx.view(-1)
        edge_index = torch.stack([src, dst], dim=0)
        return edge_index

    @staticmethod
    def _build_ptr_from_crystal_idx(crystal_atom_idx) -> Tensor:
        """
        crystal_atom_idx: list[tensor] or tensor
        返回 ptr: (num_graphs+1,)
        """
        if isinstance(crystal_atom_idx, torch.Tensor):
            # 如果是 1D tensor，认为只有一个 crystal
            n = crystal_atom_idx.numel()
            return torch.tensor([0, n], dtype=torch.long, device=crystal_atom_idx.device)

        # list[LongTensor]
        ptr = [0]
        total = 0
        for idx_map in crystal_atom_idx:
            total += len(idx_map)
            ptr.append(total)
        return torch.tensor(ptr, dtype=torch.long, device=crystal_atom_idx[0].device)

    @staticmethod
    def _pooling(atom_fea: Tensor, crystal_atom_idx) -> Tensor:
        """
        atom_fea: (N, atom_fea_len)
        crystal_atom_idx: list[LongTensor] or 1D Tensor
        """
        if isinstance(crystal_atom_idx, torch.Tensor):
            # assume single crystal: indices [0..N-1]
            return atom_fea.mean(dim=0, keepdim=True)

        # list of tensors
        feats = [
            atom_fea[idx_map].mean(dim=0) for idx_map in crystal_atom_idx
        ]
        return torch.stack(feats, dim=0)

    def forward(
        self,
        atom_fea: Tensor,
        nbr_fea: Tensor,
        nbr_fea_idx: Tensor,
        crystal_atom_idx,
    ) -> Tensor:
        """
        返回: (num_crystals, embed_dim)
        """
        atom_fea = self.embedding(atom_fea)  # (N, atom_fea_len)

        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)

        edge_index = self._build_edge_index(nbr_fea_idx)

        atom_fea = self.centrality_encoding(atom_fea, edge_index)

        ptr = self._build_ptr_from_crystal_idx(crystal_atom_idx)

        x = self.graphormer_encoder(atom_fea, edge_index, ptr)

        crys_fea = self._pooling(x, crystal_atom_idx)  # (N_crystal, atom_fea_len)

        h = self.act(self.conv_to_fc(crys_fea))
        if self.fcs is not None:
            for fc, act in zip(self.fcs, self.acts):
                h = act(fc(h))

        emb = self.proj(h)  # (N_crystal, embed_dim)
        return emb


# ============================================================
# 4. 成分编码器：简单 MLP
# ============================================================

class CompEncoderV11(nn.Module):
    """
    成分编码器：MLP -> embed_dim
    """

    def __init__(
        self,
        comp_dim: int,
        embed_dim: int = 256,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 256]

        dims = [comp_dim] + hidden_dims + [embed_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.mlp = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        h = self.mlp(x)
        h = self.ln(h)
        return h


# ============================================================
# 5. Stage1 用的单分支多任务模型
# ============================================================

class SingleBranchModelV11(nn.Module):
    """
    只用一个 encoder（成分/结构）+ 每任务独立 head，用于 Stage1 分组预训练。
    """

    def __init__(
        self,
        encoder_type: str,
        task_names: List[str],
        comp_dim: Optional[int],
        atom_dim: Optional[int],
        edge_dim: Optional[int],
        embed_dim: int = 256,
    ):
        super().__init__()
        assert encoder_type in ["composition", "structure"]
        self.encoder_type = encoder_type
        self.task_names = task_names
        self.embed_dim = embed_dim

        if encoder_type == "composition":
            assert comp_dim is not None
            self.encoder = CompEncoderV11(comp_dim=comp_dim, embed_dim=embed_dim)
        else:
            assert atom_dim is not None and edge_dim is not None
            self.encoder = StructEncoderV11(
                orig_atom_fea_len=atom_dim,
                nbr_fea_len=edge_dim,
                atom_fea_len=64,
                n_conv=3,
                graphormer_layers=1,
                num_heads=4,
                max_path_distance=5,
                h_fea_len=embed_dim,
                n_h=1,
                embed_dim=embed_dim,
            )

        self.heads = nn.ModuleDict({
            t: nn.Linear(embed_dim, 1) for t in task_names
        })

    def forward(self, inputs: Dict[str, Tensor], task_name: str) -> Tensor:
        if self.encoder_type == "composition":
            x = inputs["composition_vec"]  # (B, comp_dim)
            emb = self.encoder(x)          # (B, embed_dim)
        else:
            atom = inputs["atom_fea"]
            nbr = inputs["nbr_fea"]
            idx = inputs["nbr_idx"]
            crystal_idx = inputs.get("crystal_atom_idx", None)
            emb = self.encoder(atom, nbr, idx, crystal_idx)  # (N_crystal, embed_dim)

        out = self.heads[task_name](emb).view(-1)
        return out

    def get_encoder_state(self):
        return self.encoder.state_dict()

    def load_encoder_state(self, state_dict):
        self.encoder.load_state_dict(state_dict)


# ============================================================
# 6. Stage2 用的 MoE 统一模型
# ============================================================

class MoEUnifiedModelV11(nn.Module):
    """
    MoE 统一模型：
    - 成分 MoE 专家：CompEncoderV11 列表
    - 结构 MoE 专家：StructEncoderV11 列表
    - gating：基于 task embedding 的 softmax
    - 每任务独立 head
    """

    def __init__(
        self,
        comp_dim: int,
        atom_dim: int,
        edge_dim: int,
        task_names: List[str],
        embed_dim: int = 256,
        n_comp_experts: int = 3,
        n_struct_experts: int = 3,
        task_embed_dim: int = 64,
    ):
        super().__init__()

        self.comp_dim = comp_dim
        self.atom_dim = atom_dim
        self.edge_dim = edge_dim
        self.embed_dim = embed_dim
        self.n_comp_experts = n_comp_experts
        self.n_struct_experts = n_struct_experts

        self.task_names = task_names
        self.task_to_id = {t: i for i, t in enumerate(task_names)}
        self.task_embed = nn.Embedding(len(task_names), task_embed_dim)

        # 成分专家
        self.comp_experts = nn.ModuleList([
            CompEncoderV11(comp_dim=comp_dim, embed_dim=embed_dim)
            for _ in range(n_comp_experts)
        ])
        # 结构专家
        self.struct_experts = nn.ModuleList([
            StructEncoderV11(
                orig_atom_fea_len=atom_dim,
                nbr_fea_len=edge_dim,
                atom_fea_len=64,
                n_conv=3,
                graphormer_layers=1,
                num_heads=4,
                max_path_distance=5,
                h_fea_len=embed_dim,
                n_h=1,
                embed_dim=embed_dim,
            )
            for _ in range(n_struct_experts)
        ])

        # gating 网络
        self.comp_gate = nn.Sequential(
            nn.Linear(task_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_comp_experts),
        )
        self.struct_gate = nn.Sequential(
            nn.Linear(task_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_struct_experts),
        )

        # 每任务 head
        self.heads = nn.ModuleDict({
            t: nn.Linear(embed_dim, 1) for t in task_names
        })

    # ------------ 工具函数：MoE 编码 ------------ #

    def _get_task_emb(self, task_name: str) -> Tensor:
        tid = self.task_to_id[task_name]
        te = self.task_embed.weight[tid:tid+1]  # (1, task_embed_dim)
        return te

    def _encode_comp(self, x: Tensor, task_name: str) -> Tensor:
        """
        x: (B, comp_dim)
        return: (B, embed_dim)
        """
        te = self._get_task_emb(task_name)   # (1, d)
        gate_logits = self.comp_gate(te)    # (1, n_comp_experts)
        gate = torch.softmax(gate_logits, dim=-1)[0]  # (n_comp_experts,)

        expert_outs = []
        for e in self.comp_experts:
            expert_outs.append(e(x))  # (B, D)
        # stack: (E, B, D)
        expert_outs = torch.stack(expert_outs, dim=0)

        # gate (E,) -> (E, 1, 1)
        g = gate.view(-1, 1, 1)
        emb = (g * expert_outs).sum(dim=0)  # (B, D)
        return emb

    def _encode_struct(
        self,
        atom: Tensor,
        nbr: Tensor,
        idx: Tensor,
        crystal_atom_idx,
        task_name: str,
    ) -> Tensor:
        """
        atom: (N, atom_dim)
        nbr:  (N, M, edge_dim)
        idx:  (N, M)
        crystal_atom_idx: list/tensor  -> encoder 内部处理
        return: (N_crystal, embed_dim)
        """
        te = self._get_task_emb(task_name)
        gate_logits = self.struct_gate(te)   # (1, n_struct_experts)
        gate = torch.softmax(gate_logits, dim=-1)[0]  # (E,)

        expert_outs = []
        for e in self.struct_experts:
            expert_outs.append(e(atom, nbr, idx, crystal_atom_idx))  # (Nc, D)
        expert_outs = torch.stack(expert_outs, dim=0)  # (E, Nc, D)

        g = gate.view(-1, 1, 1)
        emb = (g * expert_outs).sum(dim=0)  # (Nc, D)
        return emb

    # ------------ 前向 ------------ #

    def forward(self, inputs: Dict[str, Tensor], task_name: str) -> Tensor:
        """
        inputs:
          若为成分任务：{"composition_vec": (B, comp_dim)}
          若为结构任务：{
              "atom_fea": (N, atom_dim),
              "nbr_fea": (N, M, edge_dim),
              "nbr_idx": (N, M),
              "crystal_atom_idx": list[tensor] or tensor
          }
        """
        if "composition_vec" in inputs:
            x = inputs["composition_vec"]
            emb = self._encode_comp(x, task_name)
        else:
            atom = inputs["atom_fea"]
            nbr = inputs["nbr_fea"]
            idx = inputs["nbr_idx"]
            crystal = inputs.get("crystal_atom_idx", None)
            emb = self._encode_struct(atom, nbr, idx, crystal, task_name)

        out = self.heads[task_name](emb).view(-1)
        return out

    # ------------ encoder 冻结 / 解冻 ------------ #

    def set_encoders_trainable(self, flag: bool = True):
        for p in self.comp_experts.parameters():
            p.requires_grad = flag
        for p in self.struct_experts.parameters():
            p.requires_grad = flag

    # ------------ 从 Stage1 encoder 初始化专家 ------------ #

    def init_comp_experts_from_state(self, state_dict, expert_indices: Optional[List[int]] = None):
        if expert_indices is None:
            expert_indices = list(range(self.n_comp_experts))
        for idx in expert_indices:
            self.comp_experts[idx].load_state_dict(state_dict, strict=False)

    def init_struct_experts_from_state(self, state_dict, expert_indices: Optional[List[int]] = None):
        if expert_indices is None:
            expert_indices = list(range(self.n_struct_experts))
        for idx in expert_indices:
            self.struct_experts[idx].load_state_dict(state_dict, strict=False)