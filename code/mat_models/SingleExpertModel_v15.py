# mat_models/SingleExpertModel_v15.py
# ----------------------------------
# 通用单任务专家模型（v15）
# 支持：composition / structure
# encoder 全可配置（comp_config / struct_config）
# 用于 Stage1 单专家预训练 + Stage2 MoE 微调
# ----------------------------------

from __future__ import annotations
import torch
import torch.nn as nn


class SingleExpertModelV15(nn.Module):
    """
    单任务专家模型：
    - composition: CompositionExpertV15 编码器
    - structure: StructureExpertV15 编码器
    - head：线性层（embedding → 1）
    """

    def __init__(
        self,
        mode: str,               # "composition" | "structure"
        comp_dim: int = None,
        atom_dim: int = None,
        edge_dim: int = None,
        embed_dim: int = 512,
        depth: int = 2,
        dropout: float = 0.0,
        comp_config: dict = None,
        struct_config: dict = None,
    ):
        super().__init__()

        self.mode = mode
        self.embed_dim = embed_dim

        comp_config = comp_config or {}
        struct_config = struct_config or {}

        # =============================
        # Composition expert
        # =============================
        if mode == "composition":
            from .encoders.composition_expert_v15 import CompositionExpertV15

            self.encoder = CompositionExpertV15(
                input_dim=comp_dim,
                embed_dim=embed_dim,
                depth=depth,
                dropout=dropout,
                **comp_config,    # 🔥自动接收 YAML extra 参数
            )

            self.head = nn.Linear(embed_dim, 1)

        # =============================
        # Structure expert
        # =============================
        elif mode == "structure":
            from .encoders.structure_expert_v15 import StructureExpertV15

            self.encoder = StructureExpertV15(
                atom_dim=atom_dim,
                edge_dim=edge_dim,
                embed_dim=embed_dim,
                **struct_config,   # 🔥支持 graphormer_cgcnn v15 参数配置
            )

            self.head = nn.Linear(embed_dim, 1)

        else:
            raise ValueError(f"Unknown mode={mode}")

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, batch):
        """
        batch 字典格式：
        composition:
            {"composition_vec": tensor(B, comp_dim)}

        structure:
            {
                "atom_fea": (N, atom_dim),
                "nbr_fea":  (N, M, edge_dim),
                "nbr_idx":  (N, M)
            }
        """
        if self.mode == "composition":
            x = batch["composition_vec"]
            h = self.encoder(x)           # (B, embed_dim)
            out = self.head(h)            # (B,1)
            return out

        else:  # structure
            atom = batch["atom_fea"]
            nbr = batch["nbr_fea"]
            idx = batch["nbr_idx"]
            h = self.encoder(atom, nbr, idx)  # (1, embed_dim)
            out = self.head(h)                # (1,1)
            return out