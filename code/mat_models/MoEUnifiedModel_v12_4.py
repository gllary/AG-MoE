# mat_models/MoEUnifiedModel_v12_4.py
# -----------------------------------
# v12.4: Steels-special MoE
# - 成分侧: steels 使用更深的 MLP expert + dropout + gating route bias
# - 其他任务使用标准 2 层 MLP expert
# - 结构侧: 沿用 Graphormer+CGCNN backbone
#
# 接口:
#   model = MoEUnifiedModelV12_4(...)
#   out = model(inputs, task_name)

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree


# ----------------------------------------------------------------------
# 1. CGCNN ConvLayer (LayerNorm 版)
# ----------------------------------------------------------------------

class ConvLayer(nn.Module):
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
        self.ln1 = nn.LayerNorm(2 * self.atom_fea_len)
        self.ln2 = nn.LayerNorm(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]  # (N, M, atom_fea_len)

        total_nbr_fea = torch.cat(
            [
                atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
                atom_nbr_fea,
                nbr_fea,
            ],
            dim=2,
        )  # (N,M,2*atom_fea_len+nbr_fea_len)

        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.ln1(total_gated_fea)

        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)

        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.ln2(nbr_sumed)

        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


# ----------------------------------------------------------------------
# 2. Graphormer block
# ----------------------------------------------------------------------

class CentralityEncoding(nn.Module):
    def __init__(self, max_in_degree: int, max_out_degree: int, node_dim: int):
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.z_in = nn.Parameter(torch.randn(max_in_degree, node_dim))
        self.z_out = nn.Parameter(torch.randn(max_out_degree, node_dim))

    @staticmethod
    def _clamp_degree(x, max_value):
        return torch.clamp(x, max=max_value)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        num_nodes = x.size(0)
        in_degree = self._clamp_degree(
            degree(edge_index[1], num_nodes=num_nodes).long(),
            self.max_in_degree - 1,
        )
        out_degree = self._clamp_degree(
            degree(edge_index[0], num_nodes=num_nodes).long(),
            self.max_out_degree - 1,
        )
        x = x + self.z_in[in_degree] + self.z_out[out_degree]
        return x


class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, x, edge_index, ptr=None):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        N = x.size(0)

        # 构造邻接矩阵
        adj = torch.zeros(N, N, device=x.device)
        adj[edge_index[0], edge_index[1]] = 1.0

        if ptr is None:
            attn = q @ k.t() / (q.size(-1) ** 0.5)
        else:
            attn = torch.zeros(N, N, device=x.device)
            for i in range(len(ptr) - 1):
                s, e = ptr[i], ptr[i + 1]
                qi = q[s:e]
                ki = k[s:e]
                attn[s:e, s:e] = qi @ ki.t() / (q.size(-1) ** 0.5)

        attn = attn * adj + (1.0 - adj) * (-1e6)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        return out


class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [GraphormerAttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, x, edge_index, ptr):
        outs = [h(x, edge_index, ptr) for h in self.heads]
        h = torch.cat(outs, dim=-1)
        return self.linear(h)


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim: int, num_heads: int):
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

    def forward(self, x, edge_index, ptr):
        x_attn = self.attn(self.ln1(x), edge_index, ptr) + x
        x_out = self.ff(self.ln2(x_attn)) + x_attn
        return x_out


class GraphormerEncoder(nn.Module):
    def __init__(self, layers: int, node_dim: int, num_heads: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [GraphormerEncoderLayer(node_dim, num_heads) for _ in range(layers)]
        )

    def forward(self, x, edge_index, ptr):
        for layer in self.layers:
            x = layer(x, edge_index, ptr)
        return x


# ----------------------------------------------------------------------
# 3. Graph backbone: FCrystalGraphConvNet
# ----------------------------------------------------------------------

class FCrystalGraphConvNet(nn.Module):
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
        self.convs = nn.ModuleList(
            [ConvLayer(atom_fea_len, nbr_fea_len) for _ in range(n_conv)]
        )
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

    @staticmethod
    def pooling(atom_fea, crystal_atom_idx):
        assert sum(len(idx) for idx in crystal_atom_idx) == atom_fea.size(0)
        crys_fea = torch.stack(
            [atom_fea[idx].mean(dim=0) for idx in crystal_atom_idx],
            dim=0,
        )
        return crys_fea

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        atom_fea = self.embedding(atom_fea)
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)

        N, M = nbr_fea_idx.size(0), nbr_fea_idx.size(1)
        device = atom_fea.device
        src = torch.repeat_interleave(torch.arange(N, device=device), M)
        dst = nbr_fea_idx.reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)

        atom_fea = self.centrality_encoding(atom_fea, edge_index)

        ptr = [0]
        for idx in crystal_atom_idx:
            ptr.append(ptr[-1] + len(idx))
        ptr = torch.tensor(ptr, dtype=torch.long, device=device)

        x = self.graphormer_encoder(atom_fea, edge_index, ptr)
        crys = self.pooling(x, crystal_atom_idx)

        h = self.conv_to_fc_act(self.conv_to_fc(crys))
        for fc, act in zip(self.fcs, self.acts):
            h = act(fc(h))
        out = self.fc_out(h)
        return out


# ----------------------------------------------------------------------
# 4. CompositionExpert (支持多层 + dropout)
# ----------------------------------------------------------------------

class CompositionExpert(nn.Module):
    def __init__(
        self,
        comp_dim: int,
        embed_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        in_dim = comp_dim
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else embed_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------
# 5. StructureExpert 封装
# ----------------------------------------------------------------------

class StructureExpert(nn.Module):
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
        device = atom_fea.device
        N = atom_fea.size(0)
        crystal_atom_idx = [torch.arange(N, device=device, dtype=torch.long)]
        emb = self.backbone(atom_fea, nbr_fea, nbr_idx, crystal_atom_idx)  # (1,D)
        return emb


# ----------------------------------------------------------------------
# 6. MoEUnifiedModelV12_4
# ----------------------------------------------------------------------

class MoEUnifiedModelV12_4(nn.Module):
    def __init__(
        self,
        comp_input_dims: Dict[str, int],
        atom_dim: Optional[int],
        edge_dim: Optional[int],
        embed_dim: int = 512,
        adapter_dim: int = 256,
        n_comp_experts: int = 8,
        n_struct_experts: int = 2,
        target_task: Optional[str] = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.comp_input_dims = comp_input_dims
        self.atom_dim = atom_dim
        self.edge_dim = edge_dim
        self.n_comp_experts = n_comp_experts
        self.n_struct_experts = n_struct_experts
        self.target_task = target_task

        # ---------------- Composition MoE ----------------
        self.comp_experts: Optional[nn.ModuleList]
        self.comp_gating: Optional[nn.Linear]
        self.comp_route_bias: Optional[nn.Parameter]

        if len(comp_input_dims) > 0 and n_comp_experts > 0:
            # 单任务训练时只会有一个 key
            comp_dim = list(comp_input_dims.values())[0]

            if target_task == "steels_yield":
                # steels 专用: 深层 + dropout
                n_layers = 4
                dropout = 0.1
            else:
                n_layers = 2
                dropout = 0.0

            self.comp_experts = nn.ModuleList(
                [
                    CompositionExpert(
                        comp_dim=comp_dim,
                        embed_dim=embed_dim,
                        hidden_dim=embed_dim,
                        n_layers=n_layers,
                        dropout=dropout,
                    )
                    for _ in range(n_comp_experts)
                ]
            )
            self.comp_gating = nn.Linear(comp_dim, n_comp_experts)

            if target_task == "steels_yield":
                # steels 的路由 bias（独立可学习）
                self.comp_route_bias = nn.Parameter(torch.zeros(n_comp_experts))
            else:
                self.comp_route_bias = None
        else:
            self.comp_experts = None
            self.comp_gating = None
            self.comp_route_bias = None

        # ---------------- Structure MoE ----------------
        if atom_dim is not None and edge_dim is not None and n_struct_experts > 0:
            self.struct_experts = nn.ModuleList(
                [
                    StructureExpert(atom_dim=atom_dim, edge_dim=edge_dim, embed_dim=embed_dim)
                    for _ in range(n_struct_experts)
                ]
            )
            self.struct_gating = nn.Linear(atom_dim, n_struct_experts)
        else:
            self.struct_experts = None
            self.struct_gating = None

        # ---------------- Task Heads ----------------
        self.task_heads = nn.ModuleDict(
            {
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
            }
        )

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

    # --------- internal: composition MoE ----------
    def _forward_comp(self, x: torch.Tensor) -> torch.Tensor:
        if self.comp_experts is None or self.comp_gating is None:
            raise RuntimeError("Composition experts not initialized.")

        logits = self.comp_gating(x)  # (B, n_experts)
        if self.comp_route_bias is not None:
            logits = logits + self.comp_route_bias  # steels route bias
        weights = F.softmax(logits, dim=-1)  # (B, n_experts)

        outs = []
        for expert in self.comp_experts:
            outs.append(expert(x).unsqueeze(1))  # (B,1,D)
        outs = torch.cat(outs, dim=1)  # (B, n_experts, D)

        weights = weights.unsqueeze(-1)  # (B, n_experts, 1)
        emb = (weights * outs).sum(dim=1)  # (B,D)
        return emb

    # --------- internal: structure MoE ----------
    def _forward_struct(self, atom_fea, nbr_fea, nbr_idx) -> torch.Tensor:
        if self.struct_experts is None or self.struct_gating is None:
            raise RuntimeError("Structure experts not initialized.")

        global_fea = atom_fea.mean(dim=0, keepdim=True)  # (1, atom_dim)
        logits = self.struct_gating(global_fea)  # (1, n_experts)
        weights = F.softmax(logits, dim=-1).view(-1)  # (n_experts,)

        outs = []
        for expert in self.struct_experts:
            outs.append(expert(atom_fea, nbr_fea, nbr_idx))  # (1,D)
        outs = torch.stack(outs, dim=0)  # (n_experts,1,D)
        weights = weights.view(-1, 1, 1)
        emb = (weights * outs).sum(dim=0)  # (1,D)
        return emb

    # --------- public forward ----------
    def forward(self, inputs: dict, task_name: str) -> torch.Tensor:
        if task_name not in self.task_heads:
            raise ValueError(f"Unknown task {task_name}")

        if task_name in self.composition_tasks:
            if "composition_vec" not in inputs:
                raise ValueError(f"{task_name} expects 'composition_vec'")
            x = inputs["composition_vec"]
            emb = self._forward_comp(x)
        elif task_name in self.structure_tasks:
            for k in ("atom_fea", "nbr_fea", "nbr_idx"):
                if k not in inputs:
                    raise ValueError(f"{task_name} expects '{k}' in inputs.")
            atom = inputs["atom_fea"]
            nbr = inputs["nbr_fea"]
            idx = inputs["nbr_idx"]
            emb = self._forward_struct(atom, nbr, idx)
        else:
            raise ValueError(f"Unknown task {task_name}")

        out = self.task_heads[task_name](emb)
        return out