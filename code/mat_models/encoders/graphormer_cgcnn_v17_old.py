# mat_models/encoders/graphormer_cgcnn_v17.py
# v17: CGCNN (message passing) + Graphormer + CLS pooling
# 使用 PyTorch 2.x 的 scaled_dot_product_attention，
# 在支持的环境下会自动调用 FlashAttention 核心实现。
#
# 支持：
#   - 单晶体：forward_single(atom_fea, nbr_fea, nbr_idx)
#   - mini-batch：forward_batch(x, edge_index, edge_attr, batch)

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree


# ---------------------------------------------------------
# CGCNN 卷积层 v17（基于 edge_index 的 message passing）
# ---------------------------------------------------------

class CGCNNConvV17(nn.Module):
    """
    简化版 CGCNN 卷积层（支持 PyG 图表示）：
      - 输入:
          x:         (N, node_dim)    所有原子特征
          edge_index:(2, E)           [src, dst]
          edge_attr: (E, edge_dim)    边特征（原 matbench 中的 nbr_fea 展平）
      - 输出:
          x_out:     (N, node_dim)
    """

    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        self.fc_full = nn.Linear(2 * node_dim + edge_dim, 2 * node_dim)
        self.ln1 = nn.LayerNorm(2 * node_dim)
        self.ln2 = nn.LayerNorm(node_dim)

        self.softplus1 = nn.Softplus()
        self.softplus2 = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr):
        """
        x:         (N, node_dim)
        edge_index:(2, E)
        edge_attr: (E, edge_dim)
        """
        src, dst = edge_index  # src -> dst
        x_src = x[src]         # (E, node_dim)
        x_dst = x[dst]         # (E, node_dim)

        # 拼接中心原子 + 邻居原子 + 边特征
        total = torch.cat([x_dst, x_src, edge_attr], dim=-1)  # (E, 2*node_dim + edge_dim)
        total = self.fc_full(total)                          # (E, 2*node_dim)
        total = self.ln1(total)

        # gate
        nbr_filter, nbr_core = total.chunk(2, dim=-1)  # (E, node_dim) each
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)

        msg = nbr_filter * nbr_core  # (E, node_dim)

        # 🔧 关键改动：在 AMP 环境下，index_add 聚合 & 更新用 FP32 计算，避免 Half/Float 冲突和数值不稳
        orig_dtype = x.dtype
        N = x.size(0)

        # 这里显式关闭 autocast，并切换到 float32 做聚合与更新
        with torch.amp.autocast("cuda", enabled=False):
            x_fp32 = x.to(torch.float32)           # (N, D)
            msg_fp32 = msg.to(torch.float32)       # (E, D)

            agg_fp32 = x_fp32.new_zeros(N, self.node_dim)  # (N, D), float32
            agg_fp32.index_add_(0, dst, msg_fp32)          # 按 dst 索引累加

            agg_fp32 = self.ln2(agg_fp32)
            out_fp32 = self.softplus2(x_fp32 + agg_fp32)   # 残差 + 激活

        # 再转回原始 dtype（fp16/bf16/fp32），后续层继续走 AMP 流程
        out = out_fp32.to(orig_dtype)
        return out


# ---------------------------------------------------------
# Graphormer 组件（按 batch 分图做 self-attention，使用 SDPA/FlashAttention）
# ---------------------------------------------------------

class CentralityEncodingV17(nn.Module):
    """
    按入度 / 出度做 centrality embedding，
    在前向中会根据 x 的 dtype 自动 cast，支持 AMP。
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
        num_nodes = x.size(0)
        in_deg = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.long)
        out_deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.long)

        in_deg = torch.clamp(in_deg, max=self.max_in_degree - 1)
        out_deg = torch.clamp(out_deg, max=self.max_out_degree - 1)

        # 保证 dtype 一致（AMP 下 x 可能是 half/bfloat16）
        z_in = self.z_in.to(dtype=x.dtype, device=x.device)
        z_out = self.z_out.to(dtype=x.dtype, device=x.device)

        return x + z_in[in_deg] + z_out[out_deg]


class GraphormerAttentionHeadV17(nn.Module):
    """
    单头 self-attention，按图（batch）分块计算。
    使用 torch.nn.functional.scaled_dot_product_attention，
    在 CUDA + FP16/BF16 下会自动调用 FlashAttention kernel。
    """

    def __init__(self, dim_in: int, dim_qk: int):
        super().__init__()
        self.q_proj = nn.Linear(dim_in, dim_qk)
        self.k_proj = nn.Linear(dim_in, dim_qk)
        self.v_proj = nn.Linear(dim_in, dim_qk)

    def forward(self, x: torch.Tensor, batch: torch.LongTensor) -> torch.Tensor:
        """
        x:     (N, dim_in)
        batch: (N,)  每个节点所属图 id
        """
        device = x.device
        N, _ = x.size()
        d = self.q_proj.out_features

        # 保持与输入 dtype 一致（可能是 fp16/bf16）
        orig_dtype = x.dtype  

        # 投影也在 AMP dtype 下执行
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # out 的 dtype 与 x 一致
        out = x.new_zeros(N, d)  # dtype = x.dtype

        num_graphs = int(batch.max().item()) + 1

        for g in range(num_graphs):
            mask = (batch == g)
            idx = mask.nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue

            qg = q[idx].unsqueeze(0)  # (1, L, d)
            kg = k[idx].unsqueeze(0)
            vg = v[idx].unsqueeze(0)

            # --- FlashAttention / SDPA ---
            # 输出 dtype 可能不是原 dtype（某些实现会输出 fp32）
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                qg, kg, vg,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )  # (1, L, d)

            # ⭐ 关键解决方案：确保 dtype 一致
            attn_out = attn_out.to(orig_dtype)

            # 写回
            out[idx] = attn_out.squeeze(0)

        return out



class GraphormerMultiHeadV17(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_qk: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [GraphormerAttentionHeadV17(dim_in, dim_qk) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(num_heads * dim_qk, dim_in)

    def forward(self, x, batch):
        outs = [h(x, batch) for h in self.heads]   # 每个 (N, d)
        h_cat = torch.cat(outs, dim=-1)           # (N, num_heads * d)
        return self.proj(h_cat)


class GraphormerEncoderLayerV17(nn.Module):
    def __init__(self, node_dim: int, num_heads: int, ff_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.attn = GraphormerMultiHeadV17(num_heads=num_heads, dim_in=node_dim, dim_qk=node_dim)
        self.ln1 = nn.LayerNorm(node_dim)
        self.ln2 = nn.LayerNorm(node_dim)
        self.ff = nn.Sequential(
            nn.Linear(node_dim, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, node_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch):
        # pre-norm
        h = self.attn(self.ln1(x), batch)
        x = x + self.dropout(h)
        h2 = self.ff(self.ln2(x))
        x = x + self.dropout(h2)
        return x


class GraphormerEncoderV17(nn.Module):
    def __init__(self, layers: int, node_dim: int, num_heads: int, ff_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphormerEncoderLayerV17(
                node_dim=node_dim,
                num_heads=num_heads,
                ff_hidden=ff_hidden,
                dropout=dropout,
            )
            for _ in range(layers)
        ])

    def forward(self, x, batch):
        for layer in self.layers:
            x = layer(x, batch)
        return x


# ---------------------------------------------------------
# Graphormer + CGCNN + CLS pooling 编码器 v17
# ---------------------------------------------------------

class GraphormerCGCNNEncoderV17(nn.Module):
    """
    v17 结构编码器：
      - CGCNNConvV17 做基于边的 message passing
      - CentralityEncodingV17 注入度中心性
      - GraphormerEncoderV17 做图内 self-attention（SDPA/FlashAttention）
      - 每个图一个 CLS token，最后用 CLS embedding 作为图表示

    支持两种入口：
      1) forward_single(atom_fea, nbr_fea, nbr_idx)
      2) forward_batch(x, edge_index, edge_attr, batch)
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

        # 原子特征 → node_dim
        self.atom_embed = nn.Linear(atom_dim, node_dim)

        # 多层 CGCNNConv
        self.convs = nn.ModuleList([
            CGCNNConvV17(node_dim=node_dim, edge_dim=edge_dim)
            for _ in range(conv_layers)
        ])

        # centrality + graphormer
        self.centrality = CentralityEncodingV17(
            max_in_degree=max_in_degree,
            max_out_degree=max_out_degree,
            node_dim=node_dim,
        )
        self.graphormer = GraphormerEncoderV17(
            layers=graphormer_layers,
            node_dim=node_dim,
            num_heads=num_heads,
            ff_hidden=ff_hidden,
            dropout=dropout,
        )

        # CLS token（每个图一个，但参数共享）
        self.cls_token = nn.Parameter(torch.zeros(1, node_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 最后图级投影到 embed_dim
        self.fc_out = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, embed_dim),
        )

    # -------------------- 核心：batched forward --------------------

    def forward_batch(self, x, edge_index, edge_attr, batch):
        """
        x:         (N, atom_dim)   —— 所有节点的原子特征
        edge_index:(2, E)          —— 所有边
        edge_attr: (E, edge_dim)
        batch:     (N,) long       —— 每个节点所属图 id

        返回:
          embs: (B, embed_dim)，B 为图的数量（batch_size）
        """
        device = x.device
        # 1) 原子特征投影
        x = self.atom_embed(x)  # (N, node_dim)

        # 2) CGCNN 多层 message passing（内部已处理 AMP/FP32 聚合）
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)  # (N, node_dim)

        # 3) centrality 编码（依然只对原始节点做）
        x = self.centrality(x, edge_index)  # (N, node_dim)

        # 4) 为每个图加一个 CLS 节点
        num_graphs = int(batch.max().item()) + 1
        cls_token = self.cls_token.to(device=device, dtype=x.dtype)
        cls = cls_token.expand(num_graphs, -1)         # (B, node_dim)

        # CLS 节点的 batch 索引
        cls_batch = torch.arange(num_graphs, device=device, dtype=torch.long)  # (B,)

        # 拼接到所有节点末尾
        x_all = torch.cat([x, cls], dim=0)                # (N+B, node_dim)
        batch_all = torch.cat([batch, cls_batch], dim=0)  # (N+B,)

        # 5) Graphormer 编码（图内 self-attention，内部用 FlashAttention/SDPA）
        x_enc = self.graphormer(x_all, batch_all)         # (N+B, node_dim)

        # 6) 取 CLS embedding：最后的 B 个节点
        cls_emb = x_enc[-num_graphs:, :]                  # (B, node_dim)

        # 7) 投影到统一 embed_dim
        embs = self.fc_out(cls_emb)                       # (B, embed_dim)
        return embs

    # -------------------- 单晶体 forward（兼容旧接口） --------------------

    def forward_single(self, atom_fea, nbr_fea, nbr_idx):
        """
        单个晶体（兼容你旧的数据结构）：
          atom_fea: (N, atom_dim)
          nbr_fea:  (N, M, edge_dim)
          nbr_idx:  (N, M)

        内部构造：
          x          = atom_fea                   → (N, atom_dim)
          edge_index = 使用 nbr_idx 展平         → (2, N*M)
          edge_attr  = nbr_fea 展平              → (N*M, edge_dim)
          batch      = 全 0（只有一个图）        → (N,)
        """
        device = atom_fea.device
        N, M = nbr_idx.shape

        # 展平邻居为 PyG 风格边
        src = torch.arange(N, device=device, dtype=torch.long).repeat_interleave(M)  # (N*M,)
        dst = nbr_idx.reshape(-1).long()                                            # (N*M,)

        edge_index = torch.stack([src, dst], dim=0)                                 # (2, E)
        edge_attr = nbr_fea.reshape(-1, self.edge_dim)                              # (E, edge_dim)

        batch = torch.zeros(N, dtype=torch.long, device=device)                     # (N,)

        embs = self.forward_batch(atom_fea, edge_index, edge_attr, batch)
        return embs  # (1, embed_dim)

    # 兼容原来 encoder(...) 的写法
    def forward(self, atom_fea, nbr_fea, nbr_idx):
        return self.forward_single(atom_fea, nbr_fea, nbr_idx)