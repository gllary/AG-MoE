# mat_models/encoders/structure_expert_v16.py

from __future__ import annotations
import torch
import torch.nn as nn

from .graphormer_cgcnn_v16 import GraphormerCGCNNEncoderV16


class StructureExpertV16(nn.Module):
    """
    结构专家 v16:
      - backbone: GraphormerCGCNNEncoderV16（CGCNN + Graphormer + CLS）
      - 支持:
          forward(atom_fea, nbr_fea, nbr_idx)
          forward_batch(x, edge_index, edge_attr, batch)
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

        self.encoder = GraphormerCGCNNEncoderV16(
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

    def forward(self, atom_fea, nbr_fea, nbr_idx):
        return self.encoder(atom_fea, nbr_fea, nbr_idx)  # (1, D)

    def forward_batch(self, x, edge_index, edge_attr, batch):
        return self.encoder.forward_batch(x, edge_index, edge_attr, batch)  # (B, D)