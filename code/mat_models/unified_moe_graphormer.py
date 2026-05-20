# -*- coding: utf-8 -*-
"""
MoE + Graphormer v7 (BatchNorm -> LayerNorm 修正版)

- 成分侧：多个 MLP Expert + MoE 动态 gating
- 结构侧：多个 Graphormer-CGCNN Expert + MoE 动态 gating
- 输出：统一的 embed_dim，再接 13 个任务的线性 head
- 接口兼容之前的 MoE 训练脚本：
    class MoEUnifiedModel(nn.Module):
        def forward(self, inputs, task_name):  # inputs 里包含 composition_vec 或 atom_fea/nbr_fea/nbr_idx
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree


# ----------------------------------------------------------------------
# 1. ConvLayer（CGCNN 卷积层）—— 使用 LayerNorm 替代 BatchNorm
# ----------------------------------------------------------------------


class ConvLayer(nn.Module):
    """
    CGCNN-style graph convolution layer (no DGL, pure PyTorch).
    使用 LayerNorm 以支持 batch_size=1 的情况。
    """

    def __init__(self, atom_fea_len: int, nbr_fea_len: int):
        """
        Parameters
        ----------
        atom_fea_len: int
            Hidden atom feature dimension.
        nbr_fea_len: int
            Edge (bond) feature dimension.
        """
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len

        self.fc_full = nn.Linear(
            2 * self.atom_fea_len + self.nbr_fea_len,
            2 * self.atom_fea_len,
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        # 用 LayerNorm 代替 BatchNorm1d，直接作用在最后一维
        self.ln1 = nn.LayerNorm(2 * self.atom_fea_len)
        self.ln2 = nn.LayerNorm(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Parameters
        ----------
        atom_in_fea: (N, atom_fea_len)
        nbr_fea:     (N, M, nbr_fea_len)
        nbr_fea_idx: (N, M)  long

        Returns
        -------
        out: (N, atom_fea_len)
        """
        N, M = nbr_fea_idx.shape

        # 邻居原子特征
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]  # (N, M, atom_fea_len)

        # 拼接中心原子、邻居原子、边特征
        total_nbr_fea = torch.cat(
            [
                atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
                atom_nbr_fea,
                nbr_fea,
            ],
            dim=2,
        )  # (N, M, 2*atom_fea_len + nbr_fea_len)

        total_gated_fea = self.fc_full(total_nbr_fea)  # (N, M, 2*atom_fea_len)
        total_gated_fea = self.ln1(total_gated_fea)    # LayerNorm on last dim

        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)  # (N, M, atom_fea_len)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)

        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)  # (N, atom_fea_len)
        nbr_sumed = self.ln2(nbr_sumed)                      # (N, atom_fea_len)

        out = self.softplus2(atom_in_fea + nbr_sumed)        # (N, atom_fea_len)
        return out


# ----------------------------------------------------------------------
# 2. Graphormer 组件（CentralityEncoding + Multi-Head Attention）
# ----------------------------------------------------------------------


class CentralityEncoding(nn.Module):
    def __init__(self, max_in_degree: int, max_out_degree: int, node_dim: int):
        """
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max out degree of nodes
        :param node_dim: hidden dimensions of node features
        """
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.node_dim = node_dim
        self.z_in = nn.Parameter(torch.randn((max_in_degree, node_dim)))
        self.z_out = nn.Parameter(torch.randn((max_out_degree, node_dim)))

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """
        :param x: node feature matrix, shape (N, node_dim)
        :param edge_index: edge_index of graph (2, E)
        """
        num_nodes = x.shape[0]

        in_degree = self._clamp_degree(
            degree(index=edge_index[1], num_nodes=num_nodes).long(),
            self.max_in_degree - 1,
        )
        out_degree = self._clamp_degree(
            degree(index=edge_index[0], num_nodes=num_nodes).long(),
            self.max_out_degree - 1,
        )

        x = x + self.z_in[in_degree] + self.z_out[out_degree]
        return x

    @staticmethod
    def _clamp_degree(x, max_value):
        return torch.clamp(x, max=max_value)


class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor, ptr=None) -> torch.Tensor:
        """
        x: (N, dim_in)
        edge_index: (2, E)
        ptr: (num_graphs+1,) batch pointers or None
        """
        query = self.q(x)  # (N, dim_q)
        key = self.k(x)    # (N, dim_k)
        value = self.v(x)  # (N, dim_k)

        N = x.size(0)
        adjacency = torch.zeros(N, N, device=x.device)
        adjacency[edge_index[0], edge_index[1]] = 1.0

        if ptr is None:
            a = query @ key.t() / (query.size(-1) ** 0.5)
        else:
            a = torch.zeros((N, N), device=x.device)
            for i in range(len(ptr) - 1):
                s, e = ptr[i], ptr[i + 1]
                q_i = query[s:e]
                k_i = key[s:e]
                a[s:e, s:e] = q_i @ k_i.t() / (query.size(-1) ** 0.5)

        # mask 非边位置
        a = a * adjacency + (1.0 - adjacency) * (-1e6)
        softmax = torch.softmax(a, dim=-1)
        out = softmax @ value  # (N, dim_k)
        return out


class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [GraphormerAttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor, ptr) -> torch.Tensor:
        outs = []
        for head in self.heads:
            outs.append(head(x, edge_index, ptr))  # (N, dim_k)
        h = torch.cat(outs, dim=-1)  # (N, num_heads*dim_k)
        out = self.linear(h)
        return out


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim: int, num_heads: int):
        super().__init__()
        self.attn = GraphormerMultiHeadAttention(
            num_heads=num_heads,
            dim_in=node_dim,
            dim_q=node_dim,
            dim_k=node_dim,
        )
        self.ln_1 = nn.LayerNorm(node_dim)
        self.ln_2 = nn.LayerNorm(node_dim)
        self.ff = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor, ptr) -> torch.Tensor:
        # pre-norm
        x_attn = self.attn(self.ln_1(x), edge_index, ptr) + x
        x_out = self.ff(self.ln_2(x_attn)) + x_attn
        return x_out


class GraphormerEncoder(nn.Module):
    def __init__(self, layers: int, node_dim: int, num_heads: int):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(node_dim=node_dim, num_heads=num_heads)
            for _ in range(layers)
        ])

    def forward(self, x, edge_index, ptr):
        for layer in self.layers:
            x = layer(x, edge_index, ptr)
        return x


# ----------------------------------------------------------------------
# 3. FCrystalGraphConvNet: Graphormer + CGCNN + Pooling，作为结构 Expert backbone
# ----------------------------------------------------------------------


class FCrystalGraphConvNet(nn.Module):
    """
    Graph encoder:
    - CGCNN ConvLayers (ConvLayer)
    - CentralityEncoding + Graphormer Encoder
    - Global pooling to graph embedding
    - MLP 映射到 out_dim

    注意：这里不直接做任务预测，只输出图级 embedding，供 MoE 统一 head 使用。
    """

    def __init__(
        self,
        orig_atom_fea_len: int,
        nbr_fea_len: int,
        atom_fea_len: int = 64,
        n_conv: int = 3,
        h_fea_len: int = 128,
        n_h: int = 1,
        out_dim: int = 128,
        graphormer_layers: int = 1,
        num_heads: int = 4,
        max_in_degree: int = 10,
        max_out_degree: int = 10,
    ):
        super().__init__()

        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        # CGCNN conv layers
        self.convs = nn.ModuleList([
            ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
            for _ in range(n_conv)
        ])

        # Centrality + Graphormer
        self.centrality_encoding = CentralityEncoding(
            max_in_degree=max_in_degree,
            max_out_degree=max_out_degree,
            node_dim=atom_fea_len,
        )
        self.graphormer_encoder = GraphormerEncoder(
            layers=graphormer_layers,
            node_dim=atom_fea_len,
            num_heads=num_heads,
        )

        # Post-pooling MLP
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_act = nn.Softplus()

        self.fcs = nn.ModuleList()
        self.acts = nn.ModuleList()
        in_dim = h_fea_len
        for _ in range(max(0, n_h - 1)):
            self.fcs.append(nn.Linear(in_dim, h_fea_len))
            self.acts.append(nn.ReLU())
            in_dim = h_fea_len

        self.fc_out = nn.Linear(in_dim, out_dim)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        atom_fea: (N, orig_atom_fea_len)
        nbr_fea:  (N, M, nbr_fea_len)
        nbr_fea_idx: (N, M)
        crystal_atom_idx: list[LongTensor], len = n_crystals
        """
        # 原子特征嵌入
        atom_fea = self.embedding(atom_fea)

        # CGCNN 多层卷积
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)

        # 构造 edge_index
        N = nbr_fea_idx.size(0)
        M = nbr_fea_idx.size(1)
        device = atom_fea.device

        src = torch.repeat_interleave(torch.arange(N, device=device), M)  # (N*M,)
        dst = nbr_fea_idx.view(-1)
        edge_index = torch.stack([src, dst], dim=0)  # (2, N*M)

        # Centrality Encoding
        atom_fea = self.centrality_encoding(atom_fea, edge_index)

        # 构造 ptr
        ptr = [0]
        for idx_map in crystal_atom_idx:
            ptr.append(ptr[-1] + len(idx_map))
        ptr = torch.tensor(ptr, dtype=torch.long, device=device)

        # Graphormer Encoder
        x = self.graphormer_encoder(atom_fea, edge_index, ptr)  # (N, atom_fea_len)

        # 图级 pooling
        crys_fea = self.pooling(x, crystal_atom_idx)  # (n_crystals, atom_fea_len)

        # MLP -> out_dim
        h = self.conv_to_fc_act(self.conv_to_fc(crys_fea))
        for fc, act in zip(self.fcs, self.acts):
            h = act(fc(h))
        out = self.fc_out(h)  # (n_crystals, out_dim)
        return out

    @staticmethod
    def pooling(atom_fea, crystal_atom_idx):
        """
        atom_fea: (N, atom_fea_len)
        crystal_atom_idx: list[LongTensor], each gives indices of atoms in that crystal
        """
        assert sum(len(idx) for idx in crystal_atom_idx) == atom_fea.size(0)
        # 简单 mean pooling
        crys_fea = torch.stack(
            [atom_fea[idx_map].mean(dim=0) for idx_map in crystal_atom_idx],
            dim=0,
        )
        return crys_fea


# ----------------------------------------------------------------------
# 4. 简单的成分 Expert（MLP encoder）
# ----------------------------------------------------------------------


class CompositionExpert(nn.Module):
    """
    一个简单的 MLP 成分 Expert：
    输入: composition_vec (B, comp_dim)
    输出: embedding (B, embed_dim)
    """

    def __init__(self, comp_dim: int, embed_dim: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = comp_dim
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else embed_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.ReLU())
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------
# 5. 结构 Expert：用 FCrystalGraphConvNet 输出 embed_dim
# ----------------------------------------------------------------------


class StructureExpert(nn.Module):
    """
    封装 FCrystalGraphConvNet，使其输出统一的 embed_dim 向量。
    """

    def __init__(
        self,
        atom_dim: int,
        edge_dim: int,
        embed_dim: int,
        atom_fea_len: int = 64,
        n_conv: int = 3,
        h_fea_len: int = 128,
        n_h: int = 1,
        graphormer_layers: int = 1,
        num_heads: int = 4,
    ):
        super().__init__()
        self.backbone = FCrystalGraphConvNet(
            orig_atom_fea_len=atom_dim,
            nbr_fea_len=edge_dim,
            atom_fea_len=atom_fea_len,
            n_conv=n_conv,
            h_fea_len=h_fea_len,
            n_h=n_h,
            out_dim=embed_dim,
            graphormer_layers=graphormer_layers,
            num_heads=num_heads,
        )

    def forward(self, atom_fea, nbr_fea, nbr_idx):
        """
        atom_fea: (N, atom_dim)
        nbr_fea:  (N, M, edge_dim)
        nbr_idx:  (N, M)
        """
        device = atom_fea.device
        N = atom_fea.size(0)
        # 当前 DataLoader 设计：每个 batch 只有一个 crystal
        crystal_atom_idx = [torch.arange(N, device=device, dtype=torch.long)]
        emb = self.backbone(atom_fea, nbr_fea, nbr_idx, crystal_atom_idx)  # (1, embed_dim)
        return emb  # 返回 (1, embed_dim)


# ----------------------------------------------------------------------
# 6. MoEUnifiedModel v7：成分 MoE + 结构 MoE + 13 任务 head
# ----------------------------------------------------------------------


class MoEUnifiedModel(nn.Module):
    """
    MoE + Graphormer v7 统一材料模型：

    - 成分侧：多个 CompositionExpert + gating (softmax over experts)
    - 结构侧：多个 StructureExpert(FCrystalGraphConvNet+Graphormer) + gating
    - 任务 head：13 个线性 head，每个任务一个
    """

    def __init__(
        self,
        comp_dim: int | None,
        atom_dim: int | None,
        edge_dim: int | None,
        embed_dim: int = 256,
        n_comp_experts: int = 3,
        n_struct_experts: int = 2,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.comp_dim = comp_dim
        self.atom_dim = atom_dim
        self.edge_dim = edge_dim
        self.n_comp_experts = n_comp_experts
        self.n_struct_experts = n_struct_experts

        # ----------------- Composition MoE -----------------
        if comp_dim is not None and n_comp_experts > 0:
            self.comp_experts = nn.ModuleList([
                CompositionExpert(comp_dim=comp_dim, embed_dim=embed_dim, hidden_dim=embed_dim)
                for _ in range(n_comp_experts)
            ])
            self.comp_gating = nn.Linear(comp_dim, n_comp_experts)
        else:
            self.comp_experts = None
            self.comp_gating = None

        # ----------------- Structure MoE -------------------
        if atom_dim is not None and edge_dim is not None and n_struct_experts > 0:
            self.struct_experts = nn.ModuleList([
                StructureExpert(atom_dim=atom_dim, edge_dim=edge_dim, embed_dim=embed_dim)
                for _ in range(n_struct_experts)
            ])
            # gating 基于简单的 mean(atom_fea) 全局特征
            self.struct_gating = nn.Linear(atom_dim, n_struct_experts)
        else:
            self.struct_experts = None
            self.struct_gating = None

        # ----------------- Task Heads (13 任务) -------------
        # 全部回归/二分类，统一用 Linear(embed_dim,1)
        self.task_heads = nn.ModuleDict({
            # composition regression
            "steels_yield": nn.Linear(embed_dim, 1),
            "expt_gap": nn.Linear(embed_dim, 1),

            # composition classification
            "glass": nn.Linear(embed_dim, 1),
            "expt_is_metal": nn.Linear(embed_dim, 1),

            # structure regression
            "phonons": nn.Linear(embed_dim, 1),
            "mp_gap": nn.Linear(embed_dim, 1),
            "mp_e_form": nn.Linear(embed_dim, 1),
            "log_kvrh": nn.Linear(embed_dim, 1),
            "log_gvrh": nn.Linear(embed_dim, 1),
            "perovskites": nn.Linear(embed_dim, 1),
            "jdft2d": nn.Linear(embed_dim, 1),
            "dielectric": nn.Linear(embed_dim, 1),

            # structure classification
            "mp_is_metal": nn.Linear(embed_dim, 1),
        })

        # 任务模态划分
        self.composition_tasks = {
            "steels_yield",
            "expt_gap",
            "glass",
            "expt_is_metal",
        }
        self.structure_tasks = {
            "phonons",
            "mp_gap",
            "mp_e_form",
            "log_kvrh",
            "log_gvrh",
            "perovskites",
            "jdft2d",
            "dielectric",
            "mp_is_metal",
        }

    # ----------------- 内部：成分 MoE 前向 -----------------
    def _forward_comp(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, comp_dim)
        return: (B, embed_dim)
        """
        if self.comp_experts is None:
            raise RuntimeError("Composition experts not initialized.")

        # gating
        logits = self.comp_gating(x)             # (B, n_comp_experts)
        weights = F.softmax(logits, dim=-1)      # (B, n_comp_experts)

        # 每个 expert 输出一个 embedding
        expert_outs = []
        for expert in self.comp_experts:
            emb_i = expert(x)                    # (B, embed_dim)
            expert_outs.append(emb_i.unsqueeze(1))  # (B,1,D)

        expert_outs = torch.cat(expert_outs, dim=1)  # (B, n_comp_experts, D)

        # 加权求和
        weights = weights.unsqueeze(-1)          # (B, n_comp_experts, 1)
        out = (weights * expert_outs).sum(dim=1) # (B, D)
        return out

    # ----------------- 内部：结构 MoE 前向 -----------------
    def _forward_struct(self, atom_fea, nbr_fea, nbr_idx) -> torch.Tensor:
        """
        atom_fea: (N, atom_dim)
        nbr_fea:  (N, M, edge_dim)
        nbr_idx:  (N, M)
        return: (1, embed_dim)   # 对应当前 DataLoader 的 batch_size=1
        """
        if self.struct_experts is None:
            raise RuntimeError("Structure experts not initialized.")

        # gating 输入：全局 mean atom 特征
        global_fea = atom_fea.mean(dim=0, keepdim=True)  # (1, atom_dim)
        logits = self.struct_gating(global_fea)          # (1, n_struct_experts)
        weights = F.softmax(logits, dim=-1).view(-1)     # (n_struct_experts,)

        expert_outs = []
        for expert in self.struct_experts:
            emb_i = expert(atom_fea, nbr_fea, nbr_idx)   # (1, embed_dim)
            expert_outs.append(emb_i)

        expert_outs = torch.stack(expert_outs, dim=0)    # (n_struct_experts, 1, D)
        weights = weights.view(-1, 1, 1)                 # (n_struct_experts,1,1)
        out = (weights * expert_outs).sum(dim=0)         # (1, D)
        return out

    # ----------------- 统一 forward -----------------
    def forward(self, inputs: dict, task_name: str) -> torch.Tensor:
        """
        inputs:
          - 对成分任务: {"composition_vec": (B, comp_dim)}
          - 对结构任务: {"atom_fea": (N, atom_dim),
                         "nbr_fea": (N, M, edge_dim),
                         "nbr_idx": (N, M)}
        返回:
          - (B,1) 或 (1,1) logits（任务 head 后的输出）
        """
        if task_name not in self.task_heads:
            raise ValueError(
                f"Task '{task_name}' not found in task_heads. "
                f"Available: {list(self.task_heads.keys())}"
            )

        if task_name in self.composition_tasks:
            if "composition_vec" not in inputs:
                raise ValueError(f"Task {task_name} expects 'composition_vec' in inputs.")
            x = inputs["composition_vec"]
            emb = self._forward_comp(x)  # (B, D)

        elif task_name in self.structure_tasks:
            for k in ("atom_fea", "nbr_fea", "nbr_idx"):
                if k not in inputs:
                    raise ValueError(f"Task {task_name} expects '{k}' in inputs.")
            atom = inputs["atom_fea"]
            nbr = inputs["nbr_fea"]
            idx = inputs["nbr_idx"]
            emb = self._forward_struct(atom, nbr, idx)  # (1, D)

        else:
            raise ValueError(f"Unknown task_name '{task_name}'")

        logits = self.task_heads[task_name](emb)  # (B,1) or (1,1)
        return logits