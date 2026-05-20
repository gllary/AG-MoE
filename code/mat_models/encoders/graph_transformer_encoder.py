import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeMLP(nn.Module):
    def __init__(self, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, nbr_fea):
        """
        nbr_fea: (N, max_nbr, edge_dim)
        返回标量权重: (N, max_nbr)
        """
        N, max_nbr, D = nbr_fea.shape
        x = nbr_fea.reshape(N * max_nbr, D)
        w = self.net(x)  # (N*max_nbr, 1)
        return w.reshape(N, max_nbr)


class GraphTransformerEncoder(nn.Module):
    """
    简化版 GraphTransformer：
    - 先用 edge MLP 聚合邻居（GNN）
    - 再用 TransformerEncoder 在节点维度上做全局 attention
    - 最后对节点做池化得到 graph embedding
    """

    def __init__(
        self,
        atom_dim: int,
        edge_dim: int,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.atom_embedding = nn.Linear(atom_dim, embed_dim)
        self.edge_mlp = EdgeMLP(edge_dim=edge_dim, hidden_dim=embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=False,  # (S, B, E)
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.readout = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, atom_fea, nbr_fea, nbr_idx):
        """
        atom_fea: (N, atom_dim)
        nbr_fea:  (N, max_nbr, edge_dim)
        nbr_idx:  (N, max_nbr)
        """
        N, max_nbr, _ = nbr_fea.shape

        h = self.atom_embedding(atom_fea)  # (N, E)

        # 基于 edge 的一阶邻居聚合（GNN 风格）
        weights = self.edge_mlp(nbr_fea)  # (N, max_nbr)
        weights = torch.sigmoid(weights)  # (0,1)
        nbr_h = h[nbr_idx]               # (N, max_nbr, E)
        msg = (weights.unsqueeze(-1) * nbr_h).sum(dim=1)  # (N, E)

        h = h + msg  # (N, E)

        # TransformerEncoder 期望 (S, B, E)
        h_seq = h.unsqueeze(1)          # (N, 1, E)
        h_seq = self.transformer(h_seq) # (N, 1, E)
        h = h_seq.squeeze(1)            # (N, E)

        graph_emb = h.mean(dim=0)       # (E,)
        return self.readout(graph_emb)