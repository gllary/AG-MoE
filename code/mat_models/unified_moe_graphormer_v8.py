# -*- coding: utf-8 -*-
"""
Unified MoE + CGCNN + Graphormer Encoder (v8)
用于 13 任务统一多专家结构编码器。

- Composition MoE Encoders
- Structure MoE Encoders (CGCNN + Graphormer)
- Dynamic Gating
- Unified Forward Interface

本文件作为 mat_models/unified_moe_graphormer_v8.py 使用。
"""

import torch
import torch.nn as nn
from torch_geometric.utils import degree
import torch.nn.functional as F


# ============================================================
# Part 1 — 基础模块：CGCNN ConvLayer
# ============================================================

class ConvLayer(nn.Module):
    """
    Crystal Graph Convolution Layer
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len

        self.fc_full = nn.Linear(
            2 * self.atom_fea_len + self.nbr_fea_len,
            2 * self.atom_fea_len
        )

        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()

        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)

        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        atom_in_fea: (N, F)
        nbr_fea: (N, M, B)
        nbr_idx: (N, M)
        """
        N, M = nbr_fea_idx.shape

        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]  # (N, M, F)

        total = torch.cat(
            [
                atom_in_fea.unsqueeze(1).expand(N, M, -1),
                atom_nbr_fea,
                nbr_fea,
            ],
            dim=2,
        )  # (N, M, 2F + B)

        total = self.fc_full(total)
        total = self.bn1(total.view(-1, 2 * self.atom_fea_len))
        total = total.view(N, M, 2 * self.atom_fea_len)

        nbr_filter, nbr_core = total.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)

        nbr_sum = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sum = self.bn2(nbr_sum)

        out = self.softplus2(atom_in_fea + nbr_sum)
        return out


# ============================================================
# Part 2 — Graphormer 组件：Centrality Encoding
# ============================================================

class CentralityEncoding(nn.Module):

    def __init__(self, max_in_degree, max_out_degree, node_dim):
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree

        self.z_in = nn.Parameter(torch.randn(max_in_degree, node_dim))
        self.z_out = nn.Parameter(torch.randn(max_out_degree, node_dim))

    def forward(self, x, edge_index):
        """
        x: (N, d)
        edge_index: (2, E)
        """
        num_nodes = x.shape[0]

        deg_in = degree(edge_index[1], num_nodes=num_nodes).long()
        deg_out = degree(edge_index[0], num_nodes=num_nodes).long()

        deg_in = torch.clamp(deg_in, max=self.max_in_degree - 1)
        deg_out = torch.clamp(deg_out, max=self.max_out_degree - 1)

        return x + self.z_in[deg_in] + self.z_out[deg_out]


# ============================================================
# Part 3 — Graphormer Attention Head
# ============================================================

class GraphormerAttentionHead(nn.Module):

    def __init__(self, dim_in, dim_q, dim_k):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, x, edge_index, ptr=None):
        """
        x: (N, d)
        """
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        N = x.size(0)
        adj = torch.zeros(N, N, device=x.device)
        adj[edge_index[0], edge_index[1]] = 1.0

        if ptr is None:
            att = query @ key.t() / (query.size(-1) ** 0.5)
        else:
            att = torch.zeros(N, N, device=x.device)
            for bi in range(len(ptr) - 1):
                s, e = ptr[bi], ptr[bi + 1]
                att[s:e, s:e] = (
                    query[s:e] @ key[s:e].t() / (query.size(-1) ** 0.5)
                )

        att = att * adj + (1 - adj) * (-1e6)
        alpha = torch.softmax(att, dim=-1)
        return alpha @ value


# ============================================================
# Part 4 — Multi-head Graphormer Attention
# ============================================================

class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_in, dim_q, dim_k):
        super().__init__()
        self.heads = nn.ModuleList(
            [GraphormerAttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, x, edge_index, ptr):
        outs = [h(x, edge_index, ptr) for h in self.heads]
        cat = torch.cat(outs, dim=-1)
        return self.linear(cat)


# ============================================================
# Part 5 — GraphormerEncoderLayer
# ============================================================

class GraphormerEncoderLayer(nn.Module):

    def __init__(self, node_dim, num_heads):
        super().__init__()
        self.att = GraphormerMultiHeadAttention(
            num_heads=num_heads,
            dim_in=node_dim,
            dim_q=node_dim,
            dim_k=node_dim,
        )
        self.ln1 = nn.LayerNorm(node_dim)
        self.ff = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.ln2 = nn.LayerNorm(node_dim)

    def forward(self, x, edge_index, ptr):
        x2 = self.att(self.ln1(x), edge_index, ptr) + x
        x3 = self.ff(self.ln2(x2)) + x2
        return x3


# ============================================================
# Part 6 — GraphormerEncoder Stack
# ============================================================

class GraphormerEncoder(nn.Module):
    def __init__(self, layers, node_dim, num_heads):
        super().__init__()
        self.layers = nn.ModuleList(
            [GraphormerEncoderLayer(node_dim, num_heads) for _ in range(layers)]
        )

    def forward(self, x, edge_index, ptr):
        for L in self.layers:
            x = L(x, edge_index, ptr)
        return x

# ============================================================
# Part 7 — FCrystalGraphConvNet（结构专家骨干）
#   CGCNN Conv + Graphormer Encoder + Mean Pooling
# ============================================================

class FCrystalGraphConvNet(nn.Module):
    """
    Full CGCNN + Graphormer pipeline used as the main expert backbone.

    Input:
        atom_fea: (N, atom_dim)
        nbr_fea: (N, M, edge_dim)
        nbr_fea_idx: (N, M)
        crystal_atom_idx: list of index lists, defining graphs

    Output:
        (num_graphs, embed_dim)
    """

    def __init__(
        self,
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        n_h=1,
        graphormer_layers=2,
        num_heads=4,
        max_deg=10,
        embed_dim=256,
    ):
        super().__init__()

        self.atom_fea_len = atom_fea_len
        self.embed_dim = embed_dim

        # ---- Embedding ----
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        # ---- CGCNN Convolution layers ----
        self.convs = nn.ModuleList([
            ConvLayer(atom_fea_len, nbr_fea_len) for _ in range(n_conv)
        ])

        # ---- Centrality Encoding ----
        self.centrality = CentralityEncoding(
            max_in_degree=max_deg,
            max_out_degree=max_deg,
            node_dim=atom_fea_len,
        )

        # ---- Graphormer Encoder ----
        self.graphormer = GraphormerEncoder(
            layers=graphormer_layers,
            node_dim=atom_fea_len,
            num_heads=num_heads,
        )

        # ---- Fully connected after pooling ----
        self.fc_pre = nn.Linear(atom_fea_len, h_fea_len)
        self.fc_pre_act = nn.ReLU()

        self.fcs = nn.ModuleList()
        for _ in range(n_h - 1):
            self.fcs.append(nn.Linear(h_fea_len, h_fea_len))

        self.fc_out = nn.Linear(h_fea_len, embed_dim)

    # --------------------------------------------------------
    # 辅助：构造 edge_index 和 ptr
    # --------------------------------------------------------
    def _build_graph_data(self, nbr_fea_idx, crystal_atom_idx, device):
        N, M = nbr_fea_idx.shape
        src = torch.repeat_interleave(torch.arange(N, device=device), M)
        dst = nbr_fea_idx.reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)

        ptr = [0]
        for idx in crystal_atom_idx:
            ptr.append(ptr[-1] + len(idx))
        ptr = torch.tensor(ptr, dtype=torch.long, device=device)
        return edge_index, ptr

    # --------------------------------------------------------
    # Mean Pooling
    # --------------------------------------------------------
    def pooling(self, atom_fea, crystal_atom_idx):
        pooled = []
        for idx in crystal_atom_idx:
            pooled.append(atom_fea[idx].mean(dim=0))
        return torch.stack(pooled, dim=0)

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Return:
            (num_graphs, embed_dim)
        """
        device = atom_fea.device

        # --- embedding ---
        atom_fea = self.embedding(atom_fea)

        # --- CGCNN layers ---
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)

        # --- Build graph ---
        edge_index, ptr = self._build_graph_data(nbr_fea_idx, crystal_atom_idx, device)

        # --- Centrality Encoding ---
        atom_fea = self.centrality(atom_fea, edge_index)

        # --- Graphormer Encoder ---
        atom_fea = self.graphormer(atom_fea, edge_index, ptr)

        # --- Pooling ---
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)

        # --- FC projection ---
        crys_fea = self.fc_pre_act(self.fc_pre(crys_fea))

        for fc in self.fcs:
            crys_fea = F.relu(fc(crys_fea))

        return self.fc_out(crys_fea)

# ============================================================
# Part 8 — Composition Expert（成分侧专家）
#   简单但宽的 MLP，用于从 composition_vec → embed_dim
# ============================================================

class CompositionExpert(nn.Module):
    """
    Expert for composition-only inputs.

    Input:
        x: (B, comp_dim)

    Output:
        (B, embed_dim)
    """

    def __init__(
        self,
        comp_dim: int,
        embed_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_dim = comp_dim
        for i in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # 最后一层映射到 embed_dim
        layers.append(nn.Linear(in_dim, embed_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, comp_dim)
        return self.mlp(x)



# ============================================================
# Part 8 — Composition Expert（成分侧专家）
#   简单但宽的 MLP，用于从 composition_vec → embed_dim
# ============================================================

class CompositionExpert(nn.Module):
    """
    Expert for composition-only inputs.

    Input:
        x: (B, comp_dim)

    Output:
        (B, embed_dim)
    """

    def __init__(
        self,
        comp_dim: int,
        embed_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_dim = comp_dim
        for i in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # 最后一层映射到 embed_dim
        layers.append(nn.Linear(in_dim, embed_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, comp_dim)
        return self.mlp(x)


# ============================================================
# Part 9 — Structure Expert Wrapper（结构侧专家）
#   封装 FCrystalGraphConvNet，输入是 (atom_fea, nbr_fea, nbr_idx)
# ============================================================

class StructureExpert(nn.Module):
    """
    Wrap FCrystalGraphConvNet into a simple interface:
        forward(atom_fea, nbr_fea, nbr_idx) -> (B, embed_dim)
    在当前数据管线中，每个 batch 的结构任务 B=1，因此输出通常是 (1, embed_dim)。
    """

    def __init__(
        self,
        atom_dim: int,
        edge_dim: int,
        embed_dim: int = 256,
        atom_fea_len: int = 64,
        n_conv: int = 3,
        h_fea_len: int = 128,
        n_h: int = 1,
        graphormer_layers: int = 2,
        num_heads: int = 4,
        max_deg: int = 10,
    ):
        super().__init__()

        self.backbone = FCrystalGraphConvNet(
            orig_atom_fea_len=atom_dim,
            nbr_fea_len=edge_dim,
            atom_fea_len=atom_fea_len,
            n_conv=n_conv,
            h_fea_len=h_fea_len,
            n_h=n_h,
            graphormer_layers=graphormer_layers,
            num_heads=num_heads,
            max_deg=max_deg,
            embed_dim=embed_dim,
        )

    def forward(
        self,
        atom_fea: torch.Tensor,    # (N_atoms, atom_dim)
        nbr_fea: torch.Tensor,     # (N_atoms, M, edge_dim)
        nbr_idx: torch.Tensor,     # (N_atoms, M)
    ) -> torch.Tensor:
        # 现有数据管线：每次只传入一个 crystal（batch_size=1），
        # 所以 crystal_atom_idx 就是 [0..N_atoms-1]
        device = atom_fea.device
        N = atom_fea.size(0)
        crystal_atom_idx = [torch.arange(N, device=device)]

        emb = self.backbone(atom_fea, nbr_fea, nbr_idx, crystal_atom_idx)
        # emb: (1, embed_dim)
        return emb


# ============================================================
# Part 10 — Gating Networks（动态路由）
#   - CompositionGating:  x -> softmax over comp experts
#   - StructureGating:    pooled node feature -> softmax over struct experts
# ============================================================

class CompositionGating(nn.Module):
    """
    Gating network for composition experts.

    Input:
        x: (B, comp_dim)
    Output:
        gate: (B, n_experts)  — softmax over experts (per sample)
    """

    def __init__(self, comp_dim: int, n_experts: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(comp_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        gate = F.softmax(logits, dim=-1)
        return gate


class StructureGating(nn.Module):
    """
    Gating network for structure experts.

    We first mean-pool raw atom features:  pooled = mean(atom_fea, dim=0)
    Then:
        pooled: (atom_dim,) -> gating MLP -> (n_experts,)
    """

    def __init__(self, atom_dim: int, n_experts: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_experts),
        )

    def forward(self, atom_fea: torch.Tensor) -> torch.Tensor:
        # atom_fea: (N_atoms, atom_dim)
        pooled = atom_fea.mean(dim=0, keepdim=True)  # (1, atom_dim)
        logits = self.net(pooled)                    # (1, n_experts)
        gate = F.softmax(logits, dim=-1)             # (1, n_experts)
        return gate

# ============================================================
# Part 11 — MoEUnifiedGraphormerModel（总 MoE 模型 v8）
#   - 支持 13 个任务（任意数量也可以，只要传 task_config）
#   - 成分输入 & 结构输入分别走各自的 MoE
#   - 每个任务有独立 head（回归 / 分类）
# ============================================================

class MoEUnifiedGraphormerModel(nn.Module):
    """
    Unified MoE model for 13 Matbench tasks (or more):

    - Composition branch:
        * n_comp_experts 个 CompositionExpert
        * CompositionGating 动态路由
    - Structure branch:
        * n_struct_experts 个 StructureExpert (Graphormer + CGCNN)
        * StructureGating 动态路由

    task_config 示例（来自训练脚本）:
        {
            "steels_yield": {"modality": "composition", "type": "regression", ...},
            "mp_gap":       {"modality": "structure",   "type": "regression", ...},
            "mp_is_metal":  {"modality": "structure",   "type": "classification", ...},
            ...
        }
    """

    def __init__(
        self,
        comp_dim: int,
        atom_dim: int,
        edge_dim: int,
        embed_dim: int,
        task_config: dict,
        n_comp_experts: int = 3,
        n_struct_experts: int = 2,
        # composition expert config
        comp_hidden_dim: int = 512,
        comp_n_layers: int = 3,
        comp_dropout: float = 0.1,
        # structure expert config
        atom_fea_len: int = 64,
        n_conv: int = 3,
        h_fea_len: int = 128,
        n_h: int = 1,
        graphormer_layers: int = 2,
        num_heads: int = 4,
        max_deg: int = 10,
    ):
        super().__init__()

        self.comp_dim = comp_dim
        self.atom_dim = atom_dim
        self.edge_dim = edge_dim
        self.embed_dim = embed_dim
        self.task_config = task_config
        self.n_comp_experts = n_comp_experts
        self.n_struct_experts = n_struct_experts

        # -----------------------------
        # 1) Composition MoE Branch
        # -----------------------------
        self.comp_experts = nn.ModuleList([
            CompositionExpert(
                comp_dim=comp_dim,
                embed_dim=embed_dim,
                hidden_dim=comp_hidden_dim,
                n_layers=comp_n_layers,
                dropout=comp_dropout,
            )
            for _ in range(n_comp_experts)
        ])
        self.comp_gating = CompositionGating(
            comp_dim=comp_dim,
            n_experts=n_comp_experts,
            hidden_dim=256,
        )

        # -----------------------------
        # 2) Structure MoE Branch
        # -----------------------------
        self.struct_experts = nn.ModuleList([
            StructureExpert(
                atom_dim=atom_dim,
                edge_dim=edge_dim,
                embed_dim=embed_dim,
                atom_fea_len=atom_fea_len,
                n_conv=n_conv,
                h_fea_len=h_fea_len,
                n_h=n_h,
                graphormer_layers=graphormer_layers,
                num_heads=num_heads,
                max_deg=max_deg,
            )
            for _ in range(n_struct_experts)
        ])
        self.struct_gating = StructureGating(
            atom_dim=atom_dim,
            n_experts=n_struct_experts,
            hidden_dim=256,
        )

        # -----------------------------
        # 3) Task-specific heads
        # -----------------------------
        self.task_heads = nn.ModuleDict()
        for t, cfg in task_config.items():
            ttype = cfg["type"]
            if ttype == "regression":
                head = nn.Linear(embed_dim, 1)
            elif ttype == "classification":
                # 二分类：直接用一个 logit，配合 BCEWithLogitsLoss
                head = nn.Linear(embed_dim, 1)
            else:
                raise ValueError(f"Unknown task type {ttype} for task {t}")
            self.task_heads[t] = head

    # --------------------------------------------------------
    # Composition forward
    # --------------------------------------------------------
    def _forward_comp(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, comp_dim)
        return: (B, embed_dim)
        """
        gate = self.comp_gating(x)  # (B, n_comp_experts)

        expert_outputs = []
        for expert in self.comp_experts:
            h = expert(x)           # (B, embed_dim)
            expert_outputs.append(h.unsqueeze(1))  # (B, 1, E)

        expert_outputs = torch.cat(expert_outputs, dim=1)  # (B, n_exp, E)
        gate = gate.unsqueeze(-1)                          # (B, n_exp, 1)

        emb = (gate * expert_outputs).sum(dim=1)           # (B, E)
        return emb

    # --------------------------------------------------------
    # Structure forward
    # --------------------------------------------------------
    def _forward_struct(
        self,
        atom_fea: torch.Tensor,   # (N_atoms, atom_dim)
        nbr_fea: torch.Tensor,    # (N_atoms, M, edge_dim)
        nbr_idx: torch.Tensor,    # (N_atoms, M)
    ) -> torch.Tensor:
        """
        Return:
            emb: (1, embed_dim)
        当前数据管线：每次只传单个 crystal（batch_size=1）。
        """
        gate = self.struct_gating(atom_fea)  # (1, n_struct_experts)

        expert_outs = []
        for expert in self.struct_experts:
            h = expert(atom_fea, nbr_fea, nbr_idx)  # (1, embed_dim)
            expert_outs.append(h.unsqueeze(1))      # (1, 1, E)

        expert_outs = torch.cat(expert_outs, dim=1)  # (1, n_exp, E)
        gate = gate.unsqueeze(-1)                    # (1, n_exp, 1)

        emb = (gate * expert_outs).sum(dim=1)        # (1, E)
        return emb

    # --------------------------------------------------------
    # Unified forward
    #   inputs:
    #       composition-only:
    #           {"composition_vec": tensor(B, comp_dim)}
    #       structure-only (batch_size=1 in current pipeline):
    #           {
    #               "atom_fea": tensor(N_atoms, atom_dim),
    #               "nbr_fea":  tensor(N_atoms, M, edge_dim),
    #               "nbr_idx":  tensor(N_atoms, M),
    #           }
    # --------------------------------------------------------
    def forward(self, inputs: dict, task_name: str) -> torch.Tensor:
        if task_name not in self.task_heads:
            raise KeyError(f"Unknown task_name: {task_name}")

        cfg = self.task_config[task_name]
        modality = cfg["modality"]

        if modality == "composition":
            x = inputs["composition_vec"]          # (B, comp_dim)
            emb = self._forward_comp(x)           # (B, E)
        elif modality == "structure":
            atom = inputs["atom_fea"]
            nbr = inputs["nbr_fea"]
            idx = inputs["nbr_idx"]
            emb = self._forward_struct(atom, nbr, idx)  # (1, E)
        else:
            raise ValueError(f"Unknown modality {modality} for task {task_name}")

        logits = self.task_heads[task_name](emb)  # (B, 1) or (1,1)
        return logits