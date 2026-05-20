# mat_models/encoders/graphormer_cgcnn_v15.py
# v15: CGCNN + Graphormer + CLS token pooling
# 支持“单个晶体”的编码；批处理在 StructureExpertV15 里做循环堆叠

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree


# -------------------- CGCNN 卷积层 v15（LayerNorm 版） --------------------


class CGCNNConvV15(nn.Module):
    """
    简化版 CGCNN 卷积层，使用 LayerNorm 支持小 batch。
    输入:
      atom_fea: (N, atom_dim)
      nbr_fea:  (N, M, edge_dim)
      nbr_idx:  (N, M)
    """

    def __init__(self, atom_dim: int, edge_dim: int):
        super().__init__()
        self.atom_dim = atom_dim
        self.edge_dim = edge_dim

        self.fc_full = nn.Linear(2 * atom_dim + edge_dim, 2 * atom_dim)
        self.ln1 = nn.LayerNorm(2 * atom_dim)
        self.ln2 = nn.LayerNorm(atom_dim)

        self.softplus1 = nn.Softplus()
        self.softplus2 = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, atom_fea, nbr_fea, nbr_idx):
        """
        atom_fea: (N, atom_dim)
        nbr_fea:  (N, M, edge_dim)
        nbr_idx:  (N, M)
        """
        N, M = nbr_idx.shape
        # 邻居原子特征
        atom_nbr = atom_fea[nbr_idx, :]  # (N, M, atom_dim)

        # 拼接中心原子 + 邻居原子 + 边特征
        total = torch.cat(
            [
                atom_fea.unsqueeze(1).expand(N, M, self.atom_dim),  # (N, M, atom_dim)
                atom_nbr,                                          # (N, M, atom_dim)
                nbr_fea,                                           # (N, M, edge_dim)
            ],
            dim=-1,
        )  # (N, M, 2*atom_dim + edge_dim)

        total = self.fc_full(total)            # (N, M, 2*atom_dim)
        total = self.ln1(total)                # LN

        # gate
        nbr_filter, nbr_core = total.chunk(2, dim=-1)   # 各 (N, M, atom_dim)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)

        # 聚合
        nbr_sum = torch.sum(nbr_filter * nbr_core, dim=1)  # (N, atom_dim)
        nbr_sum = self.ln2(nbr_sum)

        out = self.softplus2(atom_fea + nbr_sum)
        return out


# -------------------- Graphormer 组件（带 ptr 支持） --------------------


class CentralityEncodingV15(nn.Module):
    """
    按入度 / 出度做 centrality embedding，和之前 v14 类似。
    """

    def __init__(self, max_in_degree: int, max_out_degree: int, node_dim: int):
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.node_dim = node_dim

        self.z_in = nn.Parameter(torch.randn(max_in_degree, node_dim))
        self.z_out = nn.Parameter(torch.randn(max_out_degree, node_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """
        x: (N, node_dim)
        edge_index: (2, E)
        """
        num_nodes = x.shape[0]
        in_deg = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.long)
        out_deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.long)

        in_deg = torch.clamp(in_deg, max=self.max_in_degree - 1)
        out_deg = torch.clamp(out_deg, max=self.max_out_degree - 1)

        return x + self.z_in[in_deg] + self.z_out[out_deg]


class GraphormerAttentionHeadV15(nn.Module):
    def __init__(self, dim_in: int, dim_qk: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_qk)
        self.k = nn.Linear(dim_in, dim_qk)
        self.v = nn.Linear(dim_in, dim_qk)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor, ptr: torch.LongTensor) -> torch.Tensor:
        """
        x: (N, dim_in)
        edge_index: (2, E)
        ptr: (num_graphs+1,)  graph 边界
        """
        q = self.q(x)  # (N, d)
        k = self.k(x)  # (N, d)
        v = self.v(x)  # (N, d)

        N = x.size(0)
        device = x.device
        d = q.size(-1)

        # block-diag attention: 每个 graph 单独算
        attn = torch.zeros(N, N, device=device)

        for i in range(ptr.size(0) - 1):
            s = ptr[i].item()
            e = ptr[i + 1].item()
            if e <= s:
                continue
            qi = q[s:e]                        # (n_i, d)
            ki = k[s:e]                        # (n_i, d)
            attn[s:e, s:e] = qi @ ki.t() / (d ** 0.5)

        # adjacency 掩码：仅保留真实边
        adj = torch.zeros(N, N, device=device)
        adj[edge_index[0], edge_index[1]] = 1.0
        # 自环也给 1
        idx = torch.arange(N, device=device)
        adj[idx, idx] = 1.0

        attn = attn * adj + (1.0 - adj) * (-1e6)
        alpha = torch.softmax(attn, dim=-1)   # (N, N)
        out = alpha @ v                       # (N, d)
        return out


class GraphormerMultiHeadV15(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_qk: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [GraphormerAttentionHeadV15(dim_in, dim_qk) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(num_heads * dim_qk, dim_in)

    def forward(self, x, edge_index, ptr):
        outs = [h(x, edge_index, ptr) for h in self.heads]  # 每个 (N, d)
        h_cat = torch.cat(outs, dim=-1)
        return self.proj(h_cat)


class GraphormerEncoderLayerV15(nn.Module):
    def __init__(self, node_dim: int, num_heads: int, ff_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.attn = GraphormerMultiHeadV15(num_heads=num_heads, dim_in=node_dim, dim_qk=node_dim)
        self.ln1 = nn.LayerNorm(node_dim)
        self.ln2 = nn.LayerNorm(node_dim)
        self.ff = nn.Sequential(
            nn.Linear(node_dim, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, node_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, ptr):
        # pre-norm
        h = self.attn(self.ln1(x), edge_index, ptr)
        x = x + self.dropout(h)
        h2 = self.ff(self.ln2(x))
        x = x + self.dropout(h2)
        return x


class GraphormerEncoderV15(nn.Module):
    def __init__(self, layers: int, node_dim: int, num_heads: int, ff_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphormerEncoderLayerV15(
                node_dim=node_dim,
                num_heads=num_heads,
                ff_hidden=ff_hidden,
                dropout=dropout,
            )
            for _ in range(layers)
        ])

    def forward(self, x, edge_index, ptr):
        for lyr in self.layers:
            x = lyr(x, edge_index, ptr)
        return x


# -------------------- Graphormer + CGCNN + CLS pooling 编码器 v15 --------------------


class GraphormerCGCNNEncoderV15(nn.Module):
    """
    单晶体编码器：
      输入: atom_fea (N, atom_dim), nbr_fea (N, M, edge_dim), nbr_idx (N, M)
      输出: (1, embed_dim)   —— 使用 CLS token pooling
    """

    def __init__(
        self,
        atom_dim: int,
        edge_dim: int,
        embed_dim: int = 512,
        node_dim: int = 128,
        conv_layers: int = 3,
        graphormer_layers: int = 2,
        num_heads: int = 4,
        ff_hidden: int = 256,
        max_in_degree: int = 10,
        max_out_degree: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.atom_dim = atom_dim
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.embed_dim = embed_dim

        # 原子特征线性投影
        self.atom_embed = nn.Linear(atom_dim, node_dim)

        # 多层 CGCNN 卷积
        self.convs = nn.ModuleList([
            CGCNNConvV15(atom_dim=node_dim, edge_dim=edge_dim)
            for _ in range(conv_layers)
        ])

        # centrality + graphormer
        self.centrality = CentralityEncodingV15(
            max_in_degree=max_in_degree,
            max_out_degree=max_out_degree,
            node_dim=node_dim,
        )
        self.graphormer = GraphormerEncoderV15(
            layers=graphormer_layers,
            node_dim=node_dim,
            num_heads=num_heads,
            ff_hidden=ff_hidden,
            dropout=dropout,
        )

        # CLS token（每个图一个，参数共享）
        self.cls_token = nn.Parameter(torch.zeros(1, node_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 最后投影到 embed_dim
        self.fc_out = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, embed_dim),
        )

    def forward(self, atom_fea, nbr_fea, nbr_idx):
        """
        单个晶体:
          atom_fea: (N, atom_dim)
          nbr_fea:  (N, M, edge_dim)
          nbr_idx:  (N, M)
        返回:
          emb: (1, embed_dim)
        """
        device = atom_fea.device
        N, M = nbr_idx.shape

        # 1) 原子特征投影
        x = self.atom_embed(atom_fea)  # (N, node_dim)

        # 2) CGCNN 多层卷积
        for conv in self.convs:
            x = conv(x, nbr_fea, nbr_idx)  # (N, node_dim)

        # 3) 构造 edge_index（局部图）
        src = torch.repeat_interleave(torch.arange(N, device=device), M)  # (N*M,)
        dst = nbr_idx.reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)  # (2, E)

        # 4) centrality 编码
        x = self.centrality(x, edge_index)  # (N, node_dim)

        # 5) CLS token 接到每个图末尾
        cls = self.cls_token.to(device)  # (1, node_dim)
        x_all = torch.cat([x, cls], dim=0)  # (N+1, node_dim)

        # CLS 索引
        cls_idx = N

        # 6) 补充 CLS 与所有原子双向连边
        cls_src = torch.full((N,), cls_idx, device=device, dtype=torch.long)
        cls_dst = torch.arange(N, device=device, dtype=torch.long)

        # CLS -> atom, atom -> CLS
        cls_edges1 = torch.stack([cls_src, cls_dst], dim=0)   # (2, N)
        cls_edges2 = torch.stack([cls_dst, cls_src], dim=0)   # (2, N)

        edge_index_all = torch.cat(
            [edge_index, cls_edges1, cls_edges2],
            dim=1
        )  # (2, E + 2N)

        # 7) ptr: 单图 [0, N+1]
        ptr = torch.tensor([0, N + 1], dtype=torch.long, device=device)

        # 8) Graphormer 编码
        x_encoded = self.graphormer(x_all, edge_index_all, ptr)  # (N+1, node_dim)

        # 9) 取 CLS 对应的 embedding
        cls_emb = x_encoded[cls_idx:cls_idx + 1, :]  # (1, node_dim)

        # 10) 投影到统一 embed_dim
        emb = self.fc_out(cls_emb)  # (1, embed_dim)
        return emb