# mat_models/encoders/composition_expert_v16.py
# v16 成分专家：稍强一点的 MLP Encoder

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositionExpertV16(nn.Module):
    """
    成分专家 v16:
      输入: composition_vec (B, comp_dim)
      输出: (B, embed_dim)
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 512,
        hidden_dims=(512, 512),
        dropout: float = 0.1,
    ):
        super().__init__()

        dims = [input_dim] + list(hidden_dims) + [embed_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        # 最后一层不加 dropout
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.mlp = nn.Sequential(*layers)

        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, input_dim)
        """
        h = self.mlp(x)
        h = self.ln(h)
        return h  # (B, embed_dim)