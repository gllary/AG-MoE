import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvancedCompositionEncoder(nn.Module):
    """
    更强的成分编码器（MODNet 风格多层 MLP + 残差 + Dropout）：
    - 使用 LayerNorm 而不是 BatchNorm，避免 batch_size=1 报错
    输入：预先提取好的 matminer 特征 (batch, comp_dim)
    输出：材料嵌入向量 (batch, embed_dim)
    """

    def __init__(self, in_dim: int, embed_dim: int, hidden_factor: int = 4, dropout: float = 0.3):
        super().__init__()

        hdim = embed_dim * hidden_factor

        self.block1 = nn.Sequential(
            nn.Linear(in_dim, hdim),
            nn.LayerNorm(hdim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.block2 = nn.Sequential(
            nn.Linear(hdim, hdim),
            nn.LayerNorm(hdim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.proj_out = nn.Linear(hdim, embed_dim)
        self.res_proj = nn.Linear(in_dim, embed_dim)

    def forward(self, x):
        """
        x: (B, in_dim)
        """
        h = self.block1(x)
        h = self.block2(h)
        h = self.proj_out(h)

        res = self.res_proj(x)
        return F.silu(h + res)