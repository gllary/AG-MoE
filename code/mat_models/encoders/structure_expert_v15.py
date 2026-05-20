# mat_models/encoders/structure_expert_v15.py

from __future__ import annotations

import torch
import torch.nn as nn

from .graphormer_cgcnn_v15 import GraphormerCGCNNEncoderV15


class StructureExpertV15(nn.Module):
    """
    结构专家 v15:
      - backbone: GraphormerCGCNNEncoderV15 (CGCNN + Graphormer + CLS)
      - 支持:
          * 单个晶体 forward(atom_fea, nbr_fea, nbr_idx) -> (1, D)
          * 一批晶体 forward_batch(list_of_graphs) -> (B, D)
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

        self.encoder = GraphormerCGCNNEncoderV15(
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

    # -------- 单个晶体 --------
    def forward(self, atom_fea, nbr_fea, nbr_idx):
        """
        单个晶体:
          atom_fea: (N, atom_dim)
          nbr_fea:  (N, M, edge_dim)
          nbr_idx:  (N, M)
        输出:
          (1, embed_dim)
        """
        return self.encoder(atom_fea, nbr_fea, nbr_idx)

    # -------- 一批晶体（mini-batch）--------
    def forward_batch(self, graphs: list[tuple]):
        """
        graphs: list of (atom_fea, nbr_fea, nbr_idx)，长度 = batch_size
          atom_fea: (N_i, atom_dim)
          nbr_fea:  (N_i, M, edge_dim)
          nbr_idx:  (N_i, M)

        返回:
          embs: (B, embed_dim)
        """
        embs = []
        for (atom_fea, nbr_fea, nbr_idx) in graphs:
            # 保证都在同一设备上
            atom_fea = atom_fea.to(next(self.parameters()).device)
            nbr_fea = nbr_fea.to(next(self.parameters()).device)
            nbr_idx = nbr_idx.to(next(self.parameters()).device)
            emb_i = self.encoder(atom_fea, nbr_fea, nbr_idx)  # (1, D)
            embs.append(emb_i)

        embs = torch.cat(embs, dim=0)  # (B, embed_dim)
        return embs