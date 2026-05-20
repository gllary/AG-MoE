# mat_models/encoders/structure_expert_v17.py

from __future__ import annotations

import torch
import torch.nn as nn

from .graphormer_cgcnn_v17 import GraphormerCGCNNEncoderV17


class StructureExpertV17(nn.Module):
    """
    结构专家 v17:
      - backbone: GraphormerCGCNNEncoderV17 (CGCNN + Graphormer + CLS + FlashAttention)
      - 支持:
          * 单个晶体 forward(atom_fea, nbr_fea, nbr_idx) -> (1, D)
          * mini-batch forward_batch(list_of_graphs) -> (B, D)
          * 以及直接 forward_batch(x, edge_index, edge_attr, batch)（训练用）
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
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = GraphormerCGCNNEncoderV17(
            atom_dim=atom_dim,
            edge_dim=edge_dim,
            embed_dim=embed_dim,
            node_dim=node_dim,
            conv_layers=conv_layers,
            graphormer_layers=graphormer_layers,
            num_heads=num_heads,
            ff_hidden=ff_hidden,
            dropout=dropout,
        )

    # -------- 单个晶体（兼容老接口） --------
    def forward(self, atom_fea, nbr_fea, nbr_idx):
        """
        单个晶体:
          atom_fea: (N, atom_dim)
          nbr_fea:  (N, M, edge_dim)
          nbr_idx:  (N, M)
        输出:
          (1, embed_dim)
        """
        return self.encoder.forward_single(atom_fea, nbr_fea, nbr_idx)

    # -------- 一批晶体（mini-batch, PyG 风格）--------
    def forward_batch(self, x, edge_index, edge_attr, batch):
        """
        PyG 风格批处理：
          x:         (N, atom_dim)
          edge_index:(2, E)
          edge_attr: (E, edge_dim)
          batch:     (N,)
        返回:
          embs: (B, embed_dim)
        """
        return self.encoder.forward_batch(x, edge_index, edge_attr, batch)