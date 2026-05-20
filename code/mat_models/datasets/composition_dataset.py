import numpy as np
import torch
from torch.utils.data import Dataset


class CompositionDataset(Dataset):
    """
    通用的成分特征数据集，适用于 steels / expt_gap / dielectric 等 npz 文件：
    npz 内必须有键 "X" 和 "y"：
      - X: [N_samples, feat_dim]
      - y: [N_samples,]
    """
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32).reshape(-1, 1)

        self.X = torch.from_numpy(X)           # [N, feat_dim]
        self.y = torch.from_numpy(y)           # [N, 1]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {
            "composition_vec": self.X[idx],    # [feat_dim]
            "target": self.y[idx],             # [1]
        }