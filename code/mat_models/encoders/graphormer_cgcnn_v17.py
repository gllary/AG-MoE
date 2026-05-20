# mat_models/encoders/graphormer_cgcnn_v17.py
# v17 FINAL:
#   - CGCNN (PyG message passing, AMP-safe)
#   - Optional Graphormer + CLS
#   - FlashAttention via torch SDPA
#   - Fast path for large datasets (graphormer_layers = 0)

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree

try:
    from torch_scatter import scatter_mean
    _HAS_SCATTER = True
except Exception:
    _HAS_SCATTER = False
    from torch_geometric.utils import scatter as pyg_scatter


# ---------------------------------------------------------
# CGCNN Conv (AMP safe, PyG)
# ---------------------------------------------------------

class CGCNNConvV17(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        self.fc = nn.Linear(2 * node_dim + edge_dim, 2 * node_dim)
        self.ln_msg = nn.LayerNorm(2 * node_dim)
        self.ln_agg = nn.LayerNorm(node_dim)

        self.sigmoid = nn.Sigmoid()
        self.sp1 = nn.Softplus()
        self.sp2 = nn.Softplus()

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        x_src = x[src]
        x_dst = x[dst]

        msg = torch.cat([x_dst, x_src, edge_attr], dim=-1)
        msg = self.fc(msg)
        msg = self.ln_msg(msg)

        gate, core = msg.chunk(2, dim=-1)
        msg = self.sigmoid(gate) * self.sp1(core)

        # AMP-safe aggregation
        with torch.amp.autocast("cuda", enabled=False):
            x32 = x.float()
            msg32 = msg.float()
            agg = x32.new_zeros(x32.size(0), x32.size(1))
            agg.index_add_(0, dst, msg32)
            agg = self.ln_agg(agg)
            out = self.sp2(x32 + agg)

        return out.to(x.dtype)


# ---------------------------------------------------------
# Centrality Encoding
# ---------------------------------------------------------

class CentralityEncodingV17(nn.Module):
    def __init__(self, max_in_degree, max_out_degree, node_dim):
        super().__init__()
        self.z_in = nn.Parameter(torch.randn(max_in_degree, node_dim))
        self.z_out = nn.Parameter(torch.randn(max_out_degree, node_dim))

    def forward(self, x, edge_index):
        N = x.size(0)
        in_deg = degree(edge_index[1], N, dtype=torch.long)
        out_deg = degree(edge_index[0], N, dtype=torch.long)

        in_deg = torch.clamp(in_deg, max=self.z_in.size(0) - 1)
        out_deg = torch.clamp(out_deg, max=self.z_out.size(0) - 1)

        return x + self.z_in[in_deg].to(x.dtype) + self.z_out[out_deg].to(x.dtype)


# ---------------------------------------------------------
# Graphormer Attention (FlashAttention via SDPA)
# ---------------------------------------------------------

class GraphormerAttentionHeadV17(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x, batch):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        out = torch.zeros_like(q)
        num_graphs = int(batch.max()) + 1

        for g in range(num_graphs):
            idx = (batch == g).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue

            qg = q[idx].unsqueeze(0)
            kg = k[idx].unsqueeze(0)
            vg = v[idx].unsqueeze(0)

            yg = F.scaled_dot_product_attention(qg, kg, vg)
            out[idx] = yg.squeeze(0).to(out.dtype)

        return out


class GraphormerLayerV17(nn.Module):
    def __init__(self, dim, num_heads, ff_hidden, dropout):
        super().__init__()
        self.heads = nn.ModuleList([GraphormerAttentionHeadV17(dim) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * dim, dim)

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch):
        h = torch.cat([h(x, batch) for h in self.heads], dim=-1)
        h = self.proj(h)
        x = x + self.dropout(h)
        x = self.ln1(x)

        h2 = self.ff(x)
        x = x + self.dropout(h2)
        return self.ln2(x)


class GraphormerEncoderV17(nn.Module):
    def __init__(self, layers, dim, num_heads, ff_hidden, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphormerLayerV17(dim, num_heads, ff_hidden, dropout)
            for _ in range(layers)
        ])

    def forward(self, x, batch):
        for layer in self.layers:
            x = layer(x, batch)
        return x


# ---------------------------------------------------------
# Encoder v17 (FAST + FULL)
# ---------------------------------------------------------

class GraphormerCGCNNEncoderV17(nn.Module):
    def __init__(
        self,
        atom_dim,
        edge_dim,
        embed_dim=512,
        node_dim=128,
        conv_layers=3,
        graphormer_layers=2,
        num_heads=4,
        ff_hidden=256,
        dropout=0.1,
    ):
        super().__init__()
        self.graphormer_layers = graphormer_layers

        self.atom_embed = nn.Linear(atom_dim, node_dim)

        self.convs = nn.ModuleList([
            CGCNNConvV17(node_dim, edge_dim) for _ in range(conv_layers)
        ])

        self.centrality = CentralityEncodingV17(10, 10, node_dim)

        self.graphormer = GraphormerEncoderV17(
            graphormer_layers, node_dim, num_heads, ff_hidden, dropout
        ) if graphormer_layers > 0 else None

        self.cls_token = nn.Parameter(torch.zeros(1, node_dim))

        self.head = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, embed_dim),
        )

    def _pool_mean(self, x, batch):
        if _HAS_SCATTER:
            return scatter_mean(x, batch, dim=0)
        return pyg_scatter(x, batch, dim=0, reduce="mean")

    def forward_batch(self, x, edge_index, edge_attr, batch):
        x = self.atom_embed(x)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)

        # ---------------- FAST PATH ----------------
        if self.graphormer_layers <= 0:
            g = self._pool_mean(x, batch)
            return self.head(g)
        # -------------------------------------------

        x = self.centrality(x, edge_index)

        num_graphs = int(batch.max()) + 1
        cls = self.cls_token.to(x.dtype).expand(num_graphs, -1)
        cls_batch = torch.arange(num_graphs, device=x.device)

        x_all = torch.cat([x, cls], dim=0)
        batch_all = torch.cat([batch, cls_batch], dim=0)

        x_all = self.graphormer(x_all, batch_all)
        cls_emb = x_all[-num_graphs:]

        return self.head(cls_emb)

    def forward_single(self, atom_fea, nbr_fea, nbr_idx):
        N, M = nbr_idx.shape
        src = torch.arange(N, device=atom_fea.device).repeat_interleave(M)
        dst = nbr_idx.reshape(-1)

        edge_index = torch.stack([src, dst])
        edge_attr = nbr_fea.reshape(-1, nbr_fea.size(-1))
        batch = torch.zeros(N, dtype=torch.long, device=atom_fea.device)

        return self.forward_batch(atom_fea, edge_index, edge_attr, batch)

    def forward(self, atom_fea, nbr_fea, nbr_idx):
        return self.forward_single(atom_fea, nbr_fea, nbr_idx)