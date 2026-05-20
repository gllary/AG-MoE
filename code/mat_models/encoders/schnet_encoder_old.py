import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousFilterNetwork(nn.Module):
    """
    Continuous filter network in SchNet:
    - Expand distances with RBF basis
    - Apply MLP to produce continuous filter
    """

    def __init__(self, num_rbf=50, hidden_dim=128):
        super().__init__()

        self.num_rbf = num_rbf
        self.centers = nn.Parameter(torch.linspace(0, 5, num_rbf), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(10.0), requires_grad=False)

        self.mlp = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, distances):
        # distances: (num_edges,)
        # RBF embedding
        rbf = torch.exp(-self.gamma * (distances.unsqueeze(1) - self.centers) ** 2)
        return self.mlp(rbf)  # (num_edges, hidden_dim)


class SchNetInteraction(nn.Module):
    """
    SchNet interaction block:
    - Uses continuous filter to modulate neighbor messages
    """

    def __init__(self, hidden_dim=128):
        super().__init__()

        self.filter_net = ContinuousFilterNetwork(num_rbf=50, hidden_dim=hidden_dim)

        self.atom_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, atom_fea, nbr_fea, nbr_idx):
        """
        atom_fea: (N, atom_dim)
        nbr_fea: (N, max_nbr, edge_dim)
            - assume first dimension of edge features includes distance
        nbr_idx: (N, max_nbr)
        """

        N, max_nbr, edge_dim = nbr_fea.shape

        # Split edge features
        distances = nbr_fea[:, :, 0]  # assume first dim = distance

        # Flatten neighbors
        nbr_fea_flat = distances.reshape(-1)  # (N * max_nbr)

        # Continuous filter output: (N * max_nbr, hidden_dim)
        W = self.filter_net(nbr_fea_flat)

        # Neighbor atom features
        nbr_atoms_flat = atom_fea[nbr_idx.reshape(-1)]  # (N * max_nbr, hidden_dim)

        # Weighted message
        m = W * nbr_atoms_flat  # (N * max_nbr, hidden_dim)

        # Sum aggregation
        m = m.reshape(N, max_nbr, -1).sum(dim=1)  # (N, hidden_dim)

        # Update atom features
        atom_fea = atom_fea + self.atom_net(m)
        return atom_fea


class SchNetEncoder(nn.Module):
    """
    Full SchNet encoder for structure-based material tasks.
    """

    def __init__(
        self,
        atom_dim,
        edge_dim,
        embed_dim=128,
        num_interactions=3,
    ):
        super().__init__()

        # Initial atom embedding
        self.atom_embedding = nn.Linear(atom_dim, embed_dim)

        # Interaction blocks
        self.interactions = nn.ModuleList(
            [SchNetInteraction(hidden_dim=embed_dim) for _ in range(num_interactions)]
        )

        # Final output transform
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

        h = self.atom_embedding(atom_fea)  # (N, embed_dim)

        for interaction in self.interactions:
            h = interaction(h, nbr_fea, nbr_idx)

        # Pool (mean)
        graph_emb = h.mean(dim=0)

        return self.readout(graph_emb)