import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphEncoder(nn.Module):
    """
    一个非常简化的 GNN encoder：
    - 先把 atom_fea / nbr_fea 映射到同一维度
    - 多轮 message passing（邻居信息 mean 聚合）
    - 最后对所有原子做 mean pooling 得到图级表示
    """

    def __init__(self, atom_dim: int, edge_dim: int, embed_dim: int = 256, n_conv: int = 3):
        super().__init__()
        self.atom_fc = nn.Linear(atom_dim, embed_dim)
        self.edge_fc = nn.Linear(edge_dim, embed_dim)
        self.updates = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(n_conv)])

    def forward(self, atom_fea, nbr_fea, nbr_idx):
        """
        atom_fea: [N_atom, F_atom]
        nbr_fea:  [N_atom, N_nbr, F_edge]
        nbr_idx:  [N_atom, N_nbr]  (目前没用到，可后续增强)
        """
        h = self.atom_fc(atom_fea)      # [N, H]
        e = self.edge_fc(nbr_fea)       # [N, N_nbr, H]

        for upd in self.updates:
            # 简单 mean 聚合所有邻居的信息
            nbr_msg = e.mean(dim=1)     # [N, H]
            h = F.relu(upd(h + nbr_msg))

        # graph-level pooling
        g = h.mean(dim=0, keepdim=True)  # [1, H]
        return g