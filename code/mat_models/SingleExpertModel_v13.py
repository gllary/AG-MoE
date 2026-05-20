# mat_models/SingleExpertModel_v13.py
# -----------------------------------
from __future__ import annotations
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree


# --------- 1. CGCNN & Graphormer 模块（与之前基本一致） ---------

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
        )  # (N, M, 2*atom_fea_len + nbr_fea_len)

        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.ln1(total_gated_fea)

        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)

        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.ln2(nbr_sumed)

        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


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

    def forward(self, x, edge_index):
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
            num_heads=num_heads, dim_in=node_dim, dim_q=node_dim, dim_k=node_dim
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


class FCrystalGraphConvNet(nn.Module):
    def __init__(
        self,
        orig_atom_fea_len: int,
        nbr_fea_len: int,
        atom_fea_len: int = 64,
        n_conv: int = 3,
        h_fea_len: int = 128,
        n_h: int = 1,
        out_dim: int = 512,
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
    def pooling(atom_fea, crystal_atom_idx: List[torch.Tensor]):
        assert sum(len(idx) for idx in crystal_atom_idx) == atom_fea.size(0)
        return torch.stack(
            [atom_fea[idx].mean(dim=0) for idx in crystal_atom_idx], dim=0
        )

    def forward(self, atom_fea, nbr_fea, nbr_idx, crystal_atom_idx):
        atom_fea = self.embedding(atom_fea)
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_idx)

        N, M = nbr_idx.size()
        device = atom_fea.device
        src = torch.repeat_interleave(torch.arange(N, device=device), M)
        dst = nbr_idx.reshape(-1)
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
        return out  # (n_crystals, out_dim)


# --------- 2. 成分 Encoder（原版 + 深层残差版） ---------

class CompositionEncoderV13(nn.Module):
    """
    通用成分 Encoder：简单 MLP（其他任务继续使用）
    """
    def __init__(
        self,
        comp_dim: int,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        depth: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        in_dim = comp_dim
        for i in range(depth):
            out_dim = hidden_dim if i < depth - 1 else embed_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < depth - 1:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CompositionEncoderV13DeepResidual(nn.Module):
    """
    深层残差成分 Encoder：
    - 6 层 Linear + LayerNorm + SiLU + Dropout
    - 每层有 residual 连接
    - 主要用于 expt_gap 任务
    """
    def __init__(
        self,
        comp_dim: int,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = comp_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        blocks = []
        for i in range(num_layers):
            in_dim = self.input_dim if i == 0 else hidden_dim
            blocks.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.final = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        h = x
        for block in self.blocks:
            h = h + block(h)  # residual
        return self.final(h)


# --------- 3. 结构 Encoder ---------

class StructureEncoderV13(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        edge_dim: int,
        embed_dim: int = 512,
        atom_fea_len: int = 64,
        n_conv: int = 3,
        h_fea_len: int = 128,
        depth: int = 1,   # 控制 FC 层数
    ):
        super().__init__()
        self.backbone = FCrystalGraphConvNet(
            orig_atom_fea_len=atom_dim,
            nbr_fea_len=edge_dim,
            atom_fea_len=atom_fea_len,
            n_conv=n_conv,
            h_fea_len=h_fea_len,
            n_h=depth,
            out_dim=embed_dim,
        )

    def forward(self, atom_fea, nbr_fea, nbr_idx):
        device = atom_fea.device
        N = atom_fea.size(0)
        crystal_atom_idx = [torch.arange(N, device=device, dtype=torch.long)]
        emb = self.backbone(atom_fea, nbr_fea, nbr_idx, crystal_atom_idx)  # (1,D)
        return emb  # (1, D)


# --------- 4. 单任务专属专家模型（可选 encoder_type） ---------

class SingleExpertModelV13(nn.Module):
    """
    Stage1 单任务专属专家:
      - mode='composition': 使用 CompositionEncoderV13 或 深层残差版
      - mode='structure':   使用 StructureEncoderV13
    head 统一是 Linear(embed_dim, 1)

    encoder_type:
      - 'default': 原来的浅 MLP（其他任务默认）
      - 'deep':    深层残差 MLP（expt_gap 专用）
    """

    def __init__(
        self,
        mode: str,
        comp_dim: int | None = None,
        atom_dim: int | None = None,
        edge_dim: int | None = None,
        embed_dim: int = 512,
        depth: int = 2,
        dropout: float = 0.0,
        encoder_type: str = "default",  # 新增参数
    ):
        super().__init__()
        self.mode = mode
        self.embed_dim = embed_dim
        self.encoder_type = encoder_type

        if mode == "composition":
            assert comp_dim is not None
            if encoder_type == "deep":
                # 深层残差版（主要给 expt_gap 用）
                self.encoder = CompositionEncoderV13DeepResidual(
                    comp_dim=comp_dim,
                    embed_dim=embed_dim,
                    hidden_dim=embed_dim,
                    num_layers=6,
                    dropout=dropout,
                )
            else:
                # 默认浅 MLP（其他任务不变）
                self.encoder = CompositionEncoderV13(
                    comp_dim=comp_dim,
                    embed_dim=embed_dim,
                    hidden_dim=embed_dim,
                    depth=depth,
                    dropout=dropout,
                )
        elif mode == "structure":
            assert atom_dim is not None and edge_dim is not None
            self.encoder = StructureEncoderV13(
                atom_dim=atom_dim,
                edge_dim=edge_dim,
                embed_dim=embed_dim,
                depth=depth,
            )
        else:
            raise ValueError(f"Unknown mode {mode}")

        self.head = nn.Linear(embed_dim, 1)

    def forward(self, inputs: dict) -> torch.Tensor:
        if self.mode == "composition":
            x = inputs["composition_vec"]
            emb = self.encoder(x)           # (B, D)
        else:
            atom = inputs["atom_fea"]
            nbr = inputs["nbr_fea"]
            idx = inputs["nbr_idx"]
            emb = self.encoder(atom, nbr, idx)  # (1, D)

        out = self.head(emb)
        return out