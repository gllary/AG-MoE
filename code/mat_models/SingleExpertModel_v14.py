# mat_models/SingleExpertModel_v14.py
# ------------------------------------------------------
# Single Expert Model (V14)
# - Composition Expert: MLP encoder
# - Structure Expert: Graphormer-CGCNN (V14)
# ------------------------------------------------------

import torch
import torch.nn as nn

from mat_models.encoders.structure_expert_v14 import StructureExpertV14


# ================================================================
#  Composition Expert (更稳定的 MLP encoder)
# ================================================================
class CompositionExpertV14(nn.Module):
    """
    更强版本的 composition encoder：
    - 多层 MLP（可调 depth）
    - LayerNorm + GELU 激活
    """

    def __init__(self, comp_dim, embed_dim=512, depth=3, dropout=0.1):
        super().__init__()

        layers = []
        in_dim = comp_dim

        for i in range(depth - 1):
            layers += [
                nn.Linear(in_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ]
            in_dim = embed_dim

        # 最后一层输出 embed_dim，不激活
        layers.append(nn.Linear(in_dim, embed_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        输入: (B, comp_dim)
        输出: (B, embed_dim)
        """
        return self.encoder(x)


# ================================================================
#  SingleExpertModelV14 — 统一成分/结构专家模型
# ================================================================
class SingleExpertModelV14(nn.Module):
    """
    单专家模型（Stage-1 训练用）
    支持:
      - mode="composition": 使用 MLP expert
      - mode="structure":   使用 Graphormer-CGCNN expert

    输出为 (B,1) 或 (1,1)
    训练脚本将自己决定使用的 loss（MSE/BCE）。
    """

    def __init__(
        self,
        mode: str,
        comp_dim=None,
        atom_dim=None,
        edge_dim=None,
        embed_dim=512,
        depth=3,
        dropout=0.1,
    ):
        super().__init__()

        assert mode in ("composition", "structure"), f"Invalid mode={mode}"
        self.mode = mode
        self.embed_dim = embed_dim

        # ---------------- Composition Expert ----------------
        if mode == "composition":
            if comp_dim is None:
                raise ValueError("comp_dim must be provided for composition mode")

            self.encoder = CompositionExpertV14(
                comp_dim=comp_dim,
                embed_dim=embed_dim,
                depth=depth,
                dropout=dropout,
            )

        # ---------------- Structure Expert (Graphormer-CGCNN) ----------------
        elif mode == "structure":
            if atom_dim is None or edge_dim is None:
                raise ValueError("atom_dim and edge_dim must be provided for structure mode")

            self.encoder = StructureExpertV14(
                atom_dim=atom_dim,
                edge_dim=edge_dim,
                embed_dim=embed_dim,
            )

        # ---------------- Prediction Head ----------------
        self.head = nn.Linear(embed_dim, 1)

    # ===============================================================
    # forward
    # ===============================================================
    def forward(self, inputs: dict):
        """
        inputs:
          - composition tasks: {"composition_vec": (B, comp_dim)}
          - structure tasks: {"atom_fea":(N,D), "nbr_fea":(N,M,E), "nbr_idx":(N,M)}

        return:
            (B,1) or (1,1)
        """

        if self.mode == "composition":
            x = inputs["composition_vec"]  # (B, comp_dim)
            emb = self.encoder(x)          # (B, embed_dim)

        else:  # structure
            atom = inputs["atom_fea"]
            nbr = inputs["nbr_fea"]
            idx = inputs["nbr_idx"]
            emb = self.encoder(atom, nbr, idx)  # (1, embed_dim)

        return self.head(emb)