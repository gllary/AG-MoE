# mat_models/unified_multi_encoder.py
# -*- coding: utf-8 -*-
"""
MultiEncoderUnifiedModel

多分支 Encoder 的统一材料模型：
- 支持多个 composition encoder + 多个 structure encoder
- 每个任务通过 task_to_group 路由到某个 encoder group
- 每个任务仍然有自己的 TaskAdapter + Head

使用方式：
    group_configs = {
        "C1": {"modality": "composition"},
        "C2": {"modality": "composition"},
        "C3": {"modality": "composition"},
        "S1": {"modality": "structure"},
        "S2": {"modality": "structure"},
    }

    task_to_group = {
        "glass": "C1",
        "expt_is_metal": "C1",
        "expt_gap": "C2",
        "steels_yield": "C3",
        "mp_gap": "S1",
        "mp_e_form": "S1",
        "log_kvrh": "S1",
        "log_gvrh": "S1",
        "perovskites": "S1",
        "dielectric": "S1",
        "phonons": "S2",
        "jdft2d": "S2",
        "mp_is_metal": "S2",
    }

    model = MultiEncoderUnifiedModel(
        comp_dim=comp_dim,
        atom_dim=atom_dim,
        edge_dim=edge_dim,
        embed_dim=embed_dim,
        group_configs=group_configs,
        task_to_group=task_to_group,
        comp_encoder_type="advanced",
        graph_encoder_type="schnet",
    )

然后在训练脚本中保持原来调用方式：
    logits = model(inputs_dict, task_name=task)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoders.composition_encoder import CompositionEncoder
from .encoders.advanced_composition_encoder import AdvancedCompositionEncoder
from .encoders.schnet_encoder import SchNetEncoder
from .encoders.graph_transformer_encoder import GraphTransformerEncoder
from .encoders.task_adapter import TaskAdapter


class MultiEncoderUnifiedModel(nn.Module):
    """
    多分支 Encoder 版本的统一材料模型。

    参数
    ----
    comp_dim : int | None
        成分特征维度（npz 中 X 的维度）。如果没有成分任务可以为 None。
    atom_dim : int | None
        原子特征维度（atom_fea 的最后一维）。
    edge_dim : int | None
        边特征维度（nbr_fea 的最后一维）。
    embed_dim : int
        每个 encoder 输出的 embedding 维度，也是 task head 的输入维度。
    group_configs : dict
        形如：
        {
            "C1": {"modality": "composition"},
            "S1": {"modality": "structure"},
            ...
        }
        每个 group 名对应一个 encoder 和一个 modality。
    task_to_group : dict
        形如：
        {
            "steels_yield": "C3",
            "mp_gap": "S1",
            ...
        }
        标明每个任务应该走哪个 encoder group。
    comp_encoder_type : {"mlp", "advanced"}
        成分 encoder 类型，和你原来的 UnifiedMaterialModelV4 保持一致。
    graph_encoder_type : {"schnet", "graph_transformer"}
        结构 encoder 类型。
    """

    def __init__(
        self,
        comp_dim: int | None,
        atom_dim: int | None,
        edge_dim: int | None,
        embed_dim: int = 256,
        group_configs: dict | None = None,
        task_to_group: dict | None = None,
        comp_encoder_type: str = "advanced",
        graph_encoder_type: str = "schnet",
    ):
        super().__init__()

        if group_configs is None or task_to_group is None:
            raise ValueError(
                "MultiEncoderUnifiedModel requires group_configs and task_to_group."
            )

        self.comp_dim = comp_dim
        self.atom_dim = atom_dim
        self.edge_dim = edge_dim
        self.embed_dim = embed_dim

        self.group_configs = group_configs
        self.task_to_group = task_to_group
        self.comp_encoder_type = comp_encoder_type
        self.graph_encoder_type = graph_encoder_type

        # ------------ 构建各个 encoder group -------------
        encoders = {}
        group_modalities = {}

        for gname, gcfg in group_configs.items():
            modality = gcfg.get("modality", None)
            if modality not in {"composition", "structure"}:
                raise ValueError(
                    f"group {gname} must have modality in ['composition', 'structure'], "
                    f"got {modality}"
                )
            group_modalities[gname] = modality

            # ---- composition encoder ----
            if modality == "composition":
                if comp_dim is None:
                    raise ValueError(
                        f"group {gname} is composition but comp_dim is None"
                    )
                if comp_encoder_type == "mlp":
                    encoder = CompositionEncoder(comp_dim, embed_dim)
                elif comp_encoder_type == "advanced":
                    encoder = AdvancedCompositionEncoder(comp_dim, embed_dim)
                else:
                    raise ValueError(
                        f"Unknown comp_encoder_type: {comp_encoder_type}"
                    )

            # ---- structure encoder ----
            else:  # "structure"
                if atom_dim is None or edge_dim is None:
                    raise ValueError(
                        f"group {gname} is structure but atom_dim/edge_dim is None"
                    )
                if graph_encoder_type == "schnet":
                    encoder = SchNetEncoder(
                        atom_dim=atom_dim,
                        edge_dim=edge_dim,
                        embed_dim=embed_dim,
                        num_interactions=4,
                    )
                elif graph_encoder_type == "graph_transformer":
                    encoder = GraphTransformerEncoder(
                        atom_dim=atom_dim,
                        edge_dim=edge_dim,
                        embed_dim=embed_dim,
                        num_layers=4,
                        num_heads=4,
                    )
                else:
                    raise ValueError(
                        f"Unknown graph_encoder_type: {graph_encoder_type}"
                    )

            encoders[gname] = encoder

        self.group_modalities = group_modalities
        self.group_encoders = nn.ModuleDict(encoders)

        # ------------ Task Heads & Adapters ------------

        task_names = list(task_to_group.keys())

        # 每个任务一个 adapter + head（1 维输出，回归/分类由 loss 决定）
        self.task_adapters = nn.ModuleDict(
            {
                t: TaskAdapter(embed_dim, bottleneck=64)
                for t in task_names
            }
        )

        self.task_heads = nn.ModuleDict(
            {
                t: nn.Linear(embed_dim, 1)
                for t in task_names
            }
        )

    # -------------------------------------------------
    #  forward 接口：与原 UnifiedMaterialModel 保持一致
    # -------------------------------------------------
    def forward(self, inputs: dict, task_name: str) -> torch.Tensor:
        """
        inputs:
            - 对于 composition 任务：
                {"composition_vec": Tensor[B, comp_dim]}
            - 对于 structure 任务：
                {
                    "atom_fea": Tensor[N, atom_dim],
                    "nbr_fea":  Tensor[N, max_nbr, edge_dim] or [E, edge_dim],
                    "nbr_idx":  Tensor[N, max_nbr]           or [E, 2]
                }

        返回：
            Tensor[B, 1] 或 [N, 1]，由上层 loss 自行处理 (view(-1) 等)。
        """
        if task_name not in self.task_to_group:
            raise ValueError(
                f"Task '{task_name}' not found in task_to_group mapping. "
                f"Available tasks: {list(self.task_to_group.keys())}"
            )

        group_name = self.task_to_group[task_name]
        if group_name not in self.group_encoders:
            raise ValueError(
                f"Group '{group_name}' for task '{task_name}' not found in group_encoders. "
                f"Available groups: {list(self.group_encoders.keys())}"
            )

        modality = self.group_modalities[group_name]
        encoder = self.group_encoders[group_name]

        # ----- Composition -----
        if modality == "composition":
            if "composition_vec" not in inputs:
                raise ValueError(
                    f"Task '{task_name}' in group '{group_name}' "
                    f"expects 'composition_vec' in inputs."
                )
            x = inputs["composition_vec"]  # [B, comp_dim]
            emb = encoder(x)               # [B, embed_dim]

        # ----- Structure -----
        elif modality == "structure":
            required = {"atom_fea", "nbr_fea", "nbr_idx"}
            if not required.issubset(inputs.keys()):
                raise ValueError(
                    f"Task '{task_name}' in group '{group_name}' "
                    f"expects keys {required} in inputs, but got {inputs.keys()}."
                )
            emb = encoder(
                inputs["atom_fea"],  # [N, atom_dim]
                inputs["nbr_fea"],   # ...
                inputs["nbr_idx"],   # ...
            )                       # [*, embed_dim]

        else:
            raise ValueError(f"Unknown modality '{modality}' for group '{group_name}'.")

        # ----- Task-specific adapter + head -----
        emb = self.task_adapters[task_name](emb)
        out = self.task_heads[task_name](emb)  # [..., 1]
        return out

    # 方便调试：打印 encoder 分组信息
    def describe_groups(self) -> None:
        print("=== Encoder Groups ===")
        for g, m in self.group_modalities.items():
            print(f"  Group {g}: modality={m}")
        print("=== Task → Group mapping ===")
        for t, g in self.task_to_group.items():
            print(f"  Task {t:15s} -> Group {g}")