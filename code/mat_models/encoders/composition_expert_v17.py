# mat_models/encoders/composition_expert_v17.py

from __future__ import annotations

import torch
import torch.nn as nn


class CompositionExpertV17(nn.Module):
    """
    成分专家 v17（Stage-1 单任务用）

    设计目标：
      - 稳定、强基线（不引入 MoE / attention）
      - 对 matbench / steels / expt_gap 这类 tabular-like 成分任务非常友好
      - 参数接口与 structure v17 对齐，避免 v16/v15 混乱

    输入：
      comp_vec: (B, comp_dim)

    输出：
      embedding: (B, embed_dim)
    """

    def __init__(
        self,
        comp_dim: int,
        embed_dim: int = 512,
        hidden_dims: tuple[int, ...] = (512, 512),
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()

        self.comp_dim = comp_dim
        self.embed_dim = embed_dim

        if activation == "relu":
            act = nn.ReLU
        elif activation == "gelu":
            act = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        in_dim = comp_dim

        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(act())
            layers.append(nn.Dropout(dropout))
            in_dim = h

        self.mlp = nn.Sequential(*layers)

        # 输出统一映射到 embed_dim
        self.out_proj = nn.Linear(in_dim, embed_dim)

    def forward(self, comp_vec: torch.Tensor) -> torch.Tensor:
        """
        comp_vec: (B, comp_dim)
        return:   (B, embed_dim)
        """
        x = self.mlp(comp_vec)
        emb = self.out_proj(x)
        return emb