# mat_models/encoders/composition_expert_v15.py
# ---------------------------------------------
# v15 Composition Expert (更强 MLP Encoder)
# 支持：
#   - 深层 MLP
#   - LayerNorm
#   - Dropout
#   - 残差 FFN
#   - 可配置 hidden_dim / ff_expansion
# ---------------------------------------------

import torch
import torch.nn as nn


class FFNBlock(nn.Module):
    """Transformer FFN 风格的 MLP block（可选残差）"""

    def __init__(self, dim, hidden_dim, dropout=0.0, use_residual=True):
        super().__init__()
        self.use_residual = use_residual

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        out = self.net(self.ln(x))
        return x + out if self.use_residual else out


class CompositionExpertV15(nn.Module):
    """
    v15 composition expert:
    - 多层 FFNBlock
    - 最终输出 embed_dim
    """

    def __init__(
        self,
        input_dim,
        embed_dim=512,
        depth=4,
        dropout=0.0,
        hidden_dim=None,
        ff_expansion=4,
        use_layernorm=True,
    ):
        super().__init__()

        self.input_dim = input_dim

        # 默认 hidden_dim = embed_dim
        hidden_dim = hidden_dim or embed_dim

        # 输入线性层
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # FFN blocks
        ffn_hidden = hidden_dim * ff_expansion

        layers = []
        for _ in range(depth):
            layers.append(
                FFNBlock(
                    dim=embed_dim,
                    hidden_dim=ffn_hidden,
                    dropout=dropout,
                    use_residual=True,
                )
            )
        self.blocks = nn.ModuleList(layers)

        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.final_ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, input_dim)
        """
        h = self.input_proj(x)

        for blk in self.blocks:
            h = blk(h)

        if self.use_layernorm:
            h = self.final_ln(h)

        return h   # (B, embed_dim)