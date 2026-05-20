import torch.nn as nn


class CompositionEncoder(nn.Module):
    """
    输入: [batch, in_dim] 的 composition 特征
    输出: [batch, embed_dim] 的材料表征
    """
    def __init__(self, in_dim: int, embed_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)