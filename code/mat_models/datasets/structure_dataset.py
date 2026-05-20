import torch
from torch.utils.data import Dataset

class StructureDataset(Dataset):
    """
    适配你的 .pt 数据结构：
    item = (
        (atom_fea, nbr_fea, nbr_idx),
        target,
        extra
    )
    """
    def __init__(self, pt_path: str):
        self.data = torch.load(pt_path, weights_only=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 解包三层 tuple
        atom_fea = item[0][0].float()      # [N_atom, F_atom]
        nbr_fea  = item[0][1].float()      # [N_atom, N_nbr, F_edge]
        nbr_idx  = item[0][2].long()       # [N_atom, N_nbr]
        target   = item[1].float().view(1) # [1]

        return {
            "atom_fea": atom_fea,
            "nbr_fea": nbr_fea,
            "nbr_idx": nbr_idx,
            "target": target,
        }