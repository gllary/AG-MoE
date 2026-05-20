# mat_models/FeatureNetExptGap_v1.py
# ----------------------------------------------------
# 专用于 expt_gap 的 FeatureNet 组合特征网络
# 输入: {"composition_vec": (B, feat_dim)}
# 输出: (B, 1) 实验带隙回归值
# ----------------------------------------------------

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureCrossLayer(nn.Module):
    """
    Deep & Cross Network 风格的特征交叉层:
      x_{l+1} = x0 * (w^T x_l) + b + x_l
    其中 x0 为初始输入，x_l 为当前层特征。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x0, x: (B, D)
        xw = torch.matmul(x, self.w)  # (B,)
        xw = xw.unsqueeze(-1)         # (B, 1)
        return x0 * xw + self.b + x   # (B, D)


class FeatureBlock(nn.Module):
    """
    标准 LN + MLP block，带残差。
    """

    def __init__(self, dim: int, hidden_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * hidden_mult)
        self.fc2 = nn.Linear(dim * hidden_mult, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h


class FeatureNetExptGapV1(nn.Module):
    """
    专用于 expt_gap 的 FeatureNet:
      - Linear 投影到 embed_dim
      - 若干 CrossLayer 做显式特征交叉
      - 若干 Residual MLP Block 做高阶非线性拟合
      - 回归 head 输出带隙

    设计目标:
      1) 维度保持 embed_dim=512，方便后续 MoE 对接
      2) 明确建模特征交叉 (CrossLayer)
      3) 深度适中，便于在 400 epoch 内收敛
    """

    def __init__(
        self,
        comp_dim: int,
        embed_dim: int = 512,
        n_cross: int = 3,
        n_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.comp_dim = comp_dim
        self.embed_dim = embed_dim

        # 输入投影
        self.in_proj = nn.Sequential(
            nn.Linear(comp_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 显式特征交叉层
        self.cross_layers = nn.ModuleList(
            [FeatureCrossLayer(embed_dim) for _ in range(n_cross)]
        )

        # 深层 MLP Block
        self.blocks = nn.ModuleList(
            [FeatureBlock(embed_dim, hidden_mult=2, dropout=dropout) for _ in range(n_blocks)]
        )

        # 输出 head
        self.out_ln = nn.LayerNorm(embed_dim)
        self.out_fc = nn.Linear(embed_dim, 1)

    def forward(self, inputs: dict) -> torch.Tensor:
        """
        inputs: {"composition_vec": (B, comp_dim)}
        return: (B, 1)
        """
        x = inputs["composition_vec"]  # (B, comp_dim)

        h0 = self.in_proj(x)          # (B, D)
        h = h0

        # Cross Layers: 显式二阶交叉
        for layer in self.cross_layers:
            h = layer(h0, h)

        # Residual MLP Blocks: 高阶非线性
        for blk in self.blocks:
            h = blk(h)

        h = self.out_ln(h)
        out = self.out_fc(h)
        return out