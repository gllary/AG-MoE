# mat_models/encoders/structure_expert_v14.py

import torch
import torch.nn as nn

from .graphormer_cgcnn_v2 import GraphormerCGCNNEncoder


class StructureExpertV14(nn.Module):
    """
    更强结构专家：
        - Graphormer + CGCNN 混合结构编码器
        - 输出固定 embed_dim (默认 512)
        - 支持丰富超参数配置（v14.1）
    """

    def __init__(
        self,
        atom_dim: int,
        edge_dim: int,
        embed_dim: int = 512,
        n_conv: int = 3,
        graphormer_layers: int = 2,
        node_dim: int = 128,
        num_heads: int = 4,
        h_fea_len: int = 128,
        n_h: int = 1,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
        pooling: str = "mean",
        ffn_hidden_dim: int | None = None,
    ):
        super().__init__()

        self.encoder = GraphormerCGCNNEncoder(
            orig_atom_fea_len=atom_dim,
            nbr_fea_len=edge_dim,
            embed_dim=embed_dim,
            atom_fea_len=node_dim,
            n_conv=n_conv,
            h_fea_len=h_fea_len,
            n_h=n_h,
            graphormer_layers=graphormer_layers,
            num_heads=num_heads,
            dropout=dropout,
            ffn_hidden_dim=ffn_hidden_dim,
            norm_type=norm_type,
            pooling=pooling,
        )

    def forward(self, atom_fea, nbr_fea, nbr_idx):
        """
        输入：
            atom_fea : (N, atom_dim)
            nbr_fea  : (N, M, edge_dim)
            nbr_idx  : (N, M)
        返回：
            (1, embed_dim) 表示
        """
        return self.encoder(atom_fea, nbr_fea, nbr_idx)