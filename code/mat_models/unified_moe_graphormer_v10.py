# -*- coding: utf-8 -*-
"""
unified_moe_graphormer_v10.py

MoE v10: 高性能版统一模型
- 成分侧：多专家 MLP 编码器 + gating
- 结构侧：CGCNN + Graphormer 组合编码器，多专家 + gating
- 每个 task 对应一个独立 head（回归 / 二分类）
- 支持 MoE 专家使用统计，方便分析 gating 行为
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree


# ============================================================
# 1. 基础组件：ConvLayer（CGCNN 卷积层，做了 BN 稳定性修复）
# ============================================================

class ConvLayer(nn.Module):
    """
    CGCNN 风格的图卷积层，做了轻微修改：
    - 保留 fc_full + gating 结构
    - bn1 使用 BatchNorm1d
    - bn2 替换为 LayerNorm，避免 batch_size 很小时数值不稳定
    """

    def __init__(self, atom_fea_len: int, nbr_fea_len: int):
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len

        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len,
                                 2 * self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()

        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        # 这里用了 LayerNorm，避免 N 很小时 BN 崩掉
        self.ln2 = nn.LayerNorm(self.atom_fea_len)

        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        N: 原子数
        M: 每个原子的邻居数
        atom_in_fea: (N, atom_fea_len)
        nbr_fea:     (N, M, nbr_fea_len)
        nbr_fea_idx: (N, M)
        """
        N, M = nbr_fea_idx.shape

        # 邻居原子特征
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]  # (N, M, atom_fea_len)

        # 拼接中心原子 / 邻居原子 / 边特征
        total_nbr_fea = torch.cat(
            [
                atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
                atom_nbr_fea,
                nbr_fea,
            ],
            dim=2,
        )  # (N, M, 2*atom_fea_len + nbr_fea_len)

        # 线性 + BN
        total_gated_fea = self.fc_full(total_nbr_fea)  # (N, M, 2*atom_fea_len)
        total_gated_fea = self.bn1(
            total_gated_fea.view(-1, self.atom_fea_len * 2)
        ).view(N, M, self.atom_fea_len * 2)

        # 拆成 filter & core
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)  # (N, M, atom_fea_len)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)

        # 邻居聚合
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)  # (N, atom_fea_len)
        nbr_sumed = self.ln2(nbr_sumed)  # LayerNorm，避免 N 很小 BN 不稳定

        out = self.softplus2(atom_in_fea + nbr_sumed)  # (N, atom_fea_len)
        return out


# ============================================================
# 2. Graphormer 组件（Centrality + Multi-Head Attention）
# ============================================================

class CentralityEncoding(nn.Module):
    """
    Graphormer 中的度编码：根据 in-degree / out-degree 加上可学习向量
    """

    def __init__(self, max_in_degree: int, max_out_degree: int, node_dim: int):
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.node_dim = node_dim

        self.z_in = nn.Parameter(torch.randn((max_in_degree, node_dim)))
        self.z_out = nn.Parameter(torch.randn((max_out_degree, node_dim)))

    @staticmethod
    def _clamp_degree(x, max_value: int):
        return torch.clamp(x, max=max_value)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """
        x: (N, node_dim)
        edge_index: (2, E)
        """
        num_nodes = x.shape[0]
        row, col = edge_index

        in_degree = degree(col, num_nodes=num_nodes).long()
        out_degree = degree(row, num_nodes=num_nodes).long()

        in_degree = self._clamp_degree(in_degree, self.max_in_degree - 1)
        out_degree = self._clamp_degree(out_degree, self.max_out_degree - 1)

        x = x + self.z_in[in_degree] + self.z_out[out_degree]
        return x


class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        ptr: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        x: (N, d)
        edge_index: (2, E)
        ptr: (n_graph+1,) graph 边界（batch 情况下用）
        """
        query = self.q(x)  # (N, d_q)
        key = self.k(x)    # (N, d_k)
        value = self.v(x)  # (N, d_k)

        N = x.size(0)
        # adjacency 矩阵（无向或有向都可以，取 edge_index）
        adjacency = torch.zeros(N, N, device=x.device)
        adjacency[edge_index[0], edge_index[1]] = 1.0

        if ptr is None:
            # 单图：全局 attention + 邻接 mask
            a = query @ key.transpose(0, 1) / math.sqrt(query.size(-1))
        else:
            # 多图：分块计算 attention，避免图间互相看
            a = torch.zeros((N, N), device=x.device)
            for i in range(len(ptr) - 1):
                s, e = int(ptr[i]), int(ptr[i + 1])
                q_i = query[s:e]
                k_i = key[s:e]
                a[s:e, s:e] = q_i @ k_i.transpose(0, 1) / math.sqrt(query.size(-1))

        # 仅在有边的位置做 attention
        a = a * adjacency + (1.0 - adjacency) * (-1e6)
        softmax = torch.softmax(a, dim=-1)
        out = softmax @ value  # (N, d_k)
        return out


class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [GraphormerAttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        ptr: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        head_outs = [head(x, edge_index, ptr) for head in self.heads]
        concatenated = torch.cat(head_outs, dim=-1)
        out = self.linear(concatenated)
        return out


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim: int, num_heads: int, max_path_distance: int = 5):
        super().__init__()
        self.attention = GraphormerMultiHeadAttention(
            num_heads=num_heads,
            dim_in=node_dim,
            dim_q=node_dim,
            dim_k=node_dim,
        )
        self.ln1 = nn.LayerNorm(node_dim)
        self.ln2 = nn.LayerNorm(node_dim)
        self.ff = nn.Sequential(
            nn.Linear(node_dim, node_dim * 4),
            nn.ReLU(),
            nn.Linear(node_dim * 4, node_dim),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.LongTensor, ptr: Optional[torch.LongTensor]
    ) -> torch.Tensor:
        # Pre-LN + 残差 Attention
        x1 = self.attention(self.ln1(x), edge_index, ptr) + x
        # Pre-LN + 残差 FFN
        x2 = self.ff(self.ln2(x1)) + x1
        return x2


class GraphormerEncoder(nn.Module):
    def __init__(self, layers: int, node_dim: int, num_heads: int, max_path_distance: int = 5):
        super().__init__()
        self.layers = nn.ModuleList(
            [GraphormerEncoderLayer(node_dim, num_heads, max_path_distance) for _ in range(layers)]
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.LongTensor, ptr: Optional[torch.LongTensor]
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index, ptr)
        return x


# ============================================================
# 3. 结构专家：CGCNN + Graphormer backbone（输出 embedding）
# ============================================================

class StructBackboneCGCNNGraphormer(nn.Module):
    """
    CGCNN + Graphormer backbone
    - 输入: atom_fea, nbr_fea, nbr_idx
    - 假定当前 batch 是单晶体（如 DataLoader batch_size=1）
    - 内部构造 crystal_atom_idx = [0..N-1]
    - 输出: (1, embed_dim)
    """

    def __init__(
        self,
        orig_atom_fea_len: int,
        nbr_fea_len: int,
        atom_fea_len: int = 64,
        n_conv: int = 3,
        h_fea_len: int = 128,
        graphormer_layers: int = 2,
        num_heads: int = 4,
        max_path_distance: int = 5,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList(
            [ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len) for _ in range(n_conv)]
        )

        self.centrality_encoding = CentralityEncoding(
            max_in_degree=10, max_out_degree=10, node_dim=atom_fea_len
        )
        self.graphormer_encoder = GraphormerEncoder(
            layers=graphormer_layers,
            node_dim=atom_fea_len,
            num_heads=num_heads,
            max_path_distance=max_path_distance,
        )

        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.act = nn.Softplus()
        self.fc_proj = nn.Linear(h_fea_len, embed_dim)

    @staticmethod
    def build_edge_index(nbr_fea_idx: torch.LongTensor) -> torch.LongTensor:
        """
        nbr_fea_idx: (N, M)
        返回 edge_index: (2, E)
        """
        N, M = nbr_fea_idx.size()
        device = nbr_fea_idx.device

        src = torch.repeat_interleave(torch.arange(N, device=device), M)  # (N*M,)
        dst = nbr_fea_idx.view(-1)  # (N*M,)
        edge_index = torch.stack([src, dst], dim=0)  # (2, N*M)
        return edge_index

    @staticmethod
    def build_crystal_indices(num_atoms: int, device: torch.device) -> List[torch.Tensor]:
        """
        单晶体场景：crystal_atom_idx = [0..N-1]
        """
        idx = torch.arange(num_atoms, device=device, dtype=torch.long)
        return [idx]

    @staticmethod
    def pooling(atom_fea: torch.Tensor, crystal_atom_idx: List[torch.Tensor]) -> torch.Tensor:
        """
        简单平均池化：
        atom_fea: (N, D)
        crystal_atom_idx: list of index tensors，每个对应一个晶体
        返回 (N_crystal, D)
        """
        feats = []
        for idx in crystal_atom_idx:
            feats.append(atom_fea[idx].mean(dim=0))
        return torch.stack(feats, dim=0)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx):
        """
        atom_fea: (N, orig_atom_fea_len)
        nbr_fea:  (N, M, nbr_fea_len)
        nbr_fea_idx: (N, M)
        返回: (1, embed_dim)
        """
        device = atom_fea.device
        N = atom_fea.size(0)

        # 1) 原子特征线性映射
        atom_fea = self.embedding(atom_fea)  # (N, atom_fea_len)

        # 2) CGCNN 卷积堆叠
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)

        # 3) 构建 edge_index & crystal_atom_idx
        edge_index = self.build_edge_index(nbr_fea_idx)  # (2, E)
        crystal_atom_idx = self.build_crystal_indices(N, device=device)

        # 4) Centrality Encoding + Graphormer Encoder
        atom_fea = self.centrality_encoding(atom_fea, edge_index)
        # 构造 ptr: [0, N]  单晶体情形
        ptr = torch.tensor([0, N], dtype=torch.long, device=device)
        x = self.graphormer_encoder(atom_fea, edge_index, ptr)  # (N, atom_fea_len)

        # 5) 池化到晶体级别
        crys_fea = self.pooling(x, crystal_atom_idx)  # (1, atom_fea_len)

        # 6) 投射到 embedding
        crys_fea = self.act(self.conv_to_fc(crys_fea))  # (1, h_fea_len)
        emb = self.fc_proj(crys_fea)  # (1, embed_dim)
        return emb


# ============================================================
# 4. 成分专家：MLP 编码器
# ============================================================

class CompositionExpert(nn.Module):
    """
    成分侧专家：简单的 3 层 MLP + LayerNorm
    """

    def __init__(self, comp_dim: int, embed_dim: int):
        super().__init__()
        hidden = max(embed_dim, comp_dim)
        self.net = nn.Sequential(
            nn.Linear(comp_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.net(x)
        emb = self.ln(emb)
        return emb


# ============================================================
# 5. MoE 统一模型（Composition + Structure）
# ============================================================

class MoEUnifiedModel(nn.Module):
    """
    v10 统一 MoE 模型：
    - 成分：多个 MLP 专家 + gating
    - 结构：多个 CGCNN + Graphormer 专家 + gating
    - 每个任务一个线性 head 输出标量（回归/二分类）

    前向接口：
        forward(inputs: dict, task_name: str) -> (B,) 或 (1,)
    其中：
        composition: {"composition_vec": (B, comp_dim)}
        structure:   {
            "atom_fea": (N, atom_dim),
            "nbr_fea":  (N, M, edge_dim),
            "nbr_idx":  (N, M)
        }
    """

    def __init__(
        self,
        comp_dim: int,
        atom_dim: int,
        edge_dim: int,
        embed_dim: int = 256,
        n_comp_experts: int = 3,
        n_struct_experts: int = 4,
        graphormer_layers: int = 2,
        graphormer_heads: int = 4,
        task_config: Optional[Dict[str, Dict]] = None,
    ):
        super().__init__()
        self.comp_dim = comp_dim
        self.atom_dim = atom_dim
        self.edge_dim = edge_dim
        self.embed_dim = embed_dim
        self.n_comp_experts = n_comp_experts
        self.n_struct_experts = n_struct_experts

        # --------- 成分专家 ---------
        self.comp_experts = nn.ModuleList(
            [CompositionExpert(comp_dim, embed_dim) for _ in range(n_comp_experts)]
        )
        self.comp_gating = nn.Linear(comp_dim, n_comp_experts)

        # --------- 结构专家 ---------
        self.struct_experts = nn.ModuleList(
            [
                StructBackboneCGCNNGraphormer(
                    orig_atom_fea_len=atom_dim,
                    nbr_fea_len=edge_dim,
                    atom_fea_len=64,
                    n_conv=3,
                    h_fea_len=128,
                    graphormer_layers=graphormer_layers,
                    num_heads=graphormer_heads,
                    max_path_distance=5,
                    embed_dim=embed_dim,
                )
                for _ in range(n_struct_experts)
            ]
        )
        # gating: 用平均原子特征作为输入
        self.struct_gating = nn.Linear(atom_dim, n_struct_experts)

        # --------- Task heads ---------
        self.task_heads = nn.ModuleDict()
        if task_config is not None:
            for tname, cfg in task_config.items():
                self.add_task_head(tname, cfg["type"])
        else:
            # 如果训练脚本自己 add_task_head，这里可以为空
            pass

        # --------- MoE 使用统计 ---------
        # 记录每个专家在训练中的平均 gating 权重（指数滑动）
        self.register_buffer("comp_usage_ema", torch.zeros(n_comp_experts))
        self.register_buffer("struct_usage_ema", torch.zeros(n_struct_experts))
        self.usage_momentum = 0.95  # EMA 动量

    # ---------------- Task Head 管理 ---------------- #

    def add_task_head(self, task_name: str, task_type: str):
        """
        task_type: "regression" or "classification"
        都输出一个标量，区别仅在 loss 函数
        """
        self.task_heads[task_name] = nn.Linear(self.embed_dim, 1)

    # ---------------- MoE usage 统计 ---------------- #

    def _update_usage_ema(self, is_struct: bool, weights: torch.Tensor):
        """
        weights: (B, n_experts) 的 softmax 权重
        """
        with torch.no_grad():
            mean_w = weights.mean(dim=0)  # (n_experts,)
            if is_struct:
                self.struct_usage_ema.mul_(self.usage_momentum).add_(
                    mean_w * (1.0 - self.usage_momentum)
                )
            else:
                self.comp_usage_ema.mul_(self.usage_momentum).add_(
                    mean_w * (1.0 - self.usage_momentum)
                )

    def get_moe_usage(self):
        """
        返回当前 EMA 的专家使用情况（用于日志打印）
        """
        return {
            "comp_usage": self.comp_usage_ema.detach().cpu().tolist(),
            "struct_usage": self.struct_usage_ema.detach().cpu().tolist(),
        }

    # ---------------- 成分侧 MoE 前向 ---------------- #

    def _forward_comp(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, comp_dim)
        返回: (B, embed_dim)
        """
        gate_logits = self.comp_gating(x)          # (B, n_comp_experts)
        gate_weights = torch.softmax(gate_logits, dim=-1)  # (B, n_comp_experts)

        # 更新 MoE 使用统计
        self._update_usage_ema(is_struct=False, weights=gate_weights)

        expert_outs = []
        for expert in self.comp_experts:
            emb = expert(x)  # (B, embed_dim)
            expert_outs.append(emb.unsqueeze(1))  # (B, 1, D)

        # (B, nE, D)
        expert_stack = torch.cat(expert_outs, dim=1)
        # gate_weights: (B, nE) -> (B, nE, 1)
        weights = gate_weights.unsqueeze(-1)
        emb = torch.sum(weights * expert_stack, dim=1)  # (B, D)
        return emb

    # ---------------- 结构侧 MoE 前向 ---------------- #

    def _forward_struct(self, atom_fea, nbr_fea, nbr_idx) -> torch.Tensor:
        """
        输入为单晶体 batch：
            atom_fea: (N, atom_dim)
            nbr_fea:  (N, M, edge_dim)
            nbr_idx:  (N, M)
        返回: (1, embed_dim)
        """
        # gating 使用平均原子特征
        atom_mean = atom_fea.mean(dim=0, keepdim=True)   # (1, atom_dim)
        gate_logits = self.struct_gating(atom_mean)      # (1, n_struct_experts)
        gate_weights = torch.softmax(gate_logits, dim=-1)  # (1, n_struct_experts)

        # MoE 使用统计
        self._update_usage_ema(is_struct=True, weights=gate_weights)

        expert_outs = []
        for expert in self.struct_experts:
            emb = expert(atom_fea, nbr_fea, nbr_idx)     # (1, embed_dim)
            expert_outs.append(emb.unsqueeze(1))         # (1, 1, D)

        # (1, nE, D)
        expert_stack = torch.cat(expert_outs, dim=1)
        weights = gate_weights.unsqueeze(-1)             # (1, nE, 1)
        emb = torch.sum(weights * expert_stack, dim=1)   # (1, D)
        return emb

    # ---------------- 统一前向接口 ---------------- #

    def forward(self, inputs: Dict[str, torch.Tensor], task_name: str) -> torch.Tensor:
        """
        inputs:
            成分: {"composition_vec": (B, comp_dim)}
            结构: {
                "atom_fea": (N, atom_dim),
                "nbr_fea":  (N, M, edge_dim),
                "nbr_idx":  (N, M)
            }
        返回:
            logits: (B,) 或 (1,)
        """
        if "composition_vec" in inputs:
            x = inputs["composition_vec"]
            emb = self._forward_comp(x)  # (B, D)
        else:
            atom = inputs["atom_fea"]
            nbr = inputs["nbr_fea"]
            idx = inputs["nbr_idx"]
            emb = self._forward_struct(atom, nbr, idx)  # (1, D)

        if task_name not in self.task_heads:
            raise ValueError(f"Task head for '{task_name}' not found, "
                             f"you need to call add_task_head for this task.")

        head = self.task_heads[task_name]
        logits = head(emb).view(-1)
        return logits