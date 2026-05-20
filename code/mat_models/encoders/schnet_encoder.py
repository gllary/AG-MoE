import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousFilterNetwork(nn.Module):
    """
    SchNet 的 continuous filter：
    - 对距离做 RBF 展开
    - 通过 MLP 生成 filter
    """

    def __init__(self, num_rbf: int = 50, hidden_dim: int = 128, cutoff: float = 5.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.register_buffer("centers", torch.linspace(0.0, cutoff, num_rbf))
        self.register_buffer("gamma", torch.tensor(10.0))

        self.mlp = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, distances: torch.Tensor):
        # distances: (E,)
        diff = distances.unsqueeze(-1) - self.centers  # (E, num_rbf)
        rbf = torch.exp(-self.gamma * diff ** 2)
        return self.mlp(rbf)  # (E, hidden_dim)


class SchNetInteraction(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.filter_net = ContinuousFilterNetwork(num_rbf=50, hidden_dim=hidden_dim)

        self.atom_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, atom_fea, nbr_fea, nbr_idx):
        """
        atom_fea: (N, D)
        nbr_fea:  (N, max_nbr, edge_dim)  约定 nbr_fea[..., 0] 是距离
        nbr_idx:  (N, max_nbr)
        """
        N, max_nbr, _ = nbr_fea.shape

        distances = nbr_fea[:, :, 0]  # (N, max_nbr)
        dist_flat = distances.reshape(-1)  # (N*max_nbr,)

        # 生成连续 filter
        W = self.filter_net(dist_flat)  # (N*max_nbr, D)

        # 邻居原子特征
        nbr_atom = atom_fea[nbr_idx.reshape(-1)]  # (N*max_nbr, D)

        msg = W * nbr_atom
        msg = msg.reshape(N, max_nbr, -1).sum(dim=1)  # (N, D)

        atom_fea = atom_fea + self.atom_net(msg)
        return atom_fea


class SchNetEncoder(nn.Module):
    """
    结构侧 SchNet 编码器：
    输入原子 & 邻居特征，输出一个 graph-level embedding
    """

    def __init__(
        self,
        atom_dim: int,
        edge_dim: int,         # 目前只用到 nbr_fea[..., 0] 作为距离
        embed_dim: int = 256,
        num_interactions: int = 4,
    ):
        super().__init__()

        self.atom_embedding = nn.Linear(atom_dim, embed_dim)

        self.interactions = nn.ModuleList(
            [SchNetInteraction(hidden_dim=embed_dim) for _ in range(num_interactions)]
        )

        self.readout = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, atom_fea, nbr_fea, nbr_idx):
        """
        atom_fea: (N, atom_dim)
        nbr_fea:  (N, max_nbr, edge_dim)
        nbr_idx:  (N, max_nbr)
        """
        h = self.atom_embedding(atom_fea)

        for interaction in self.interactions:
            h = interaction(h, nbr_fea, nbr_idx)

        graph_emb = h.mean(dim=0)  # (embed_dim,)
        return self.readout(graph_emb)