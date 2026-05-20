# mat_models/SingleExpertModel_v17.py

from __future__ import annotations

import torch
import torch.nn as nn


from .encoders.composition_expert_v17 import CompositionExpertV17
# 结构侧：升级为 v17（FlashAttention 版 Graphormer）
from .encoders.structure_expert_v17 import StructureExpertV17


class SingleExpertModelV17(nn.Module):
    """
    v17 单任务专家模型：
      - mode = "composition": 使用 CompositionExpertV17
      - mode = "structure":   使用 StructureExpertV17 (CGCNN + Graphormer + FlashAttention)
      - 输出统一为标量预测 (回归 or 二分类 logits)
    """

    def __init__(
        self,
        mode: str,
        embed_dim: int = 512,
        # ---- composition ----
        comp_dim: int | None = None,
        comp_hidden_dims: tuple[int, ...] = (512, 512, 512),
        comp_dropout: float = 0.1,
        # ---- structure ----
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
            if comp_dim is None:
                raise ValueError("comp_dim must be provided for composition mode.")
            self.encoder = CompositionExpertV17(
                comp_dim=comp_dim,               # ✅ 改成 comp_dim
                embed_dim=embed_dim,             # ✅ 显式对齐结构侧
                hidden_dims=comp_hidden_dims,
                dropout=comp_dropout,
            )
        else:
            if atom_dim is None or edge_dim is None:
                raise ValueError("atom_dim and edge_dim must be provided for structure mode.")
            self.encoder = StructureExpertV17(
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

        # 统一预测头：标量
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, batch_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        输入：
          - composition 模式:
              batch_dict = {"composition_vec": X}
          - structure 模式:
              batch_dict = {
                  "x": x,
                  "edge_index": edge_index,
                  "edge_attr": edge_attr,
                  "batch": batch_idx,
              }
        输出：
          y_pred: (B, 1)
        """
        if self.mode == "composition":
            x = batch_dict["composition_vec"]        # (B, comp_dim)
            emb = self.encoder(x)                   # (B, embed_dim)
        else:
            x = batch_dict["x"]                     # (N, atom_dim)
            edge_index = batch_dict["edge_index"]   # (2, E)
            edge_attr = batch_dict["edge_attr"]     # (E, edge_dim)
            batch_idx = batch_dict["batch"]         # (N,)
            emb = self.encoder.forward_batch(
                x, edge_index, edge_attr, batch_idx
            )                                       # (B, embed_dim)

        out = self.head(emb)                        # (B, 1)
        return out
        
