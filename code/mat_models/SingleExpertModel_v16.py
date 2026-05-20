# mat_models/SingleExpertModel_v16.py
# v16: 单任务单专家模型，支持 composition / structure + PyG mini-batch

from __future__ import annotations

import torch
import torch.nn as nn

from .encoders.composition_expert_v16 import CompositionExpertV16
from .encoders.structure_expert_v16 import StructureExpertV16


class SingleExpertModelV16(nn.Module):
    """
    单任务单专家模型：
      - mode='composition': 使用 CompositionExpertV16
      - mode='structure':   使用 StructureExpertV16 (PyG mini-batch)
      - 统一输出: (B, 1)
    """

    def __init__(
        self,
        mode: str,
        embed_dim: int = 512,
        # composition
        comp_dim: int | None = None,
        comp_hidden_dims=(512, 512),
        comp_dropout: float = 0.1,
        # structure
        atom_dim: int | None = None,
        edge_dim: int | None = None,
        struct_node_dim: int = 128,
        struct_conv_layers: int = 3,
        struct_graphormer_layers: int = 2,
        struct_num_heads: int = 4,
        struct_ff_hidden: int = 256,
        struct_dropout: float = 0.1,
    ):
        super().__init__()
        assert mode in ("composition", "structure")
        self.mode = mode
        self.embed_dim = embed_dim

        if mode == "composition":
            assert comp_dim is not None, "comp_dim must be provided for composition mode"
            self.encoder = CompositionExpertV16(
                input_dim=comp_dim,
                embed_dim=embed_dim,
                hidden_dims=comp_hidden_dims,
                dropout=comp_dropout,
            )
        else:
            assert atom_dim is not None and edge_dim is not None, \
                "atom_dim & edge_dim must be provided for structure mode"
            self.encoder = StructureExpertV16(
                atom_dim=atom_dim,
                edge_dim=edge_dim,
                embed_dim=embed_dim,
                node_dim=struct_node_dim,
                conv_layers=struct_conv_layers,
                graphormer_layers=struct_graphormer_layers,
                num_heads=struct_num_heads,
                ff_hidden=struct_ff_hidden,
                dropout=struct_dropout,
            )

        # 统一回归 / 二分类 head
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        对外统一接口：
          - composition: batch["composition_vec"] -> (B,1)
          - structure mini-batch:
              batch["x"], batch["edge_index"], batch["edge_attr"], batch["batch"]
        """
        if self.mode == "composition":
            x = batch["composition_vec"]  # (B, comp_dim)
            emb = self.encoder(x)         # (B, embed_dim)
        else:
            if "x" in batch:
                emb = self.encoder.forward_batch(
                    batch["x"],
                    batch["edge_index"],
                    batch["edge_attr"],
                    batch["batch"],
                )  # (B, embed_dim)
            else:
                # 兼容旧接口：单晶体
                emb = self.encoder(
                    batch["atom_fea"],
                    batch["nbr_fea"],
                    batch["nbr_idx"],
                )  # (1, embed_dim)

        out = self.head(emb)  # (B, 1)
        return out