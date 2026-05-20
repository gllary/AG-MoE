# -*- coding: utf-8 -*-
"""
Mixture-of-Experts (MoE) Unified Material Model.

- 成分任务：多个 CompositionEncoder 做 expert，输入为 composition_vec
- 结构任务：多个 GraphEncoder 做 expert，输入为 (atom_fea, nbr_fea, nbr_idx)
- gating：
    * 成分：根据 composition_vec 做 softmax，得到各 expert 权重
    * 结构：根据 atom_fea 的原子特征均值做 softmax，得到各 expert 权重
- 输出：task-specific head（回归 / 二分类）

使用方式与之前的 UnifiedMaterialModel / MultiEncoderUnifiedModel 一样：
    logits = model(inputs, task_name="mp_gap")
"""

from __future__ import annotations
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# 这里假设你原来就有这两个 encoder；
# 如果文件名略有不同，只需要改 import 即可。
from .encoders.composition_encoder import CompositionEncoder
from .encoders.graph_encoder import GraphEncoder


class MoEUnifiedModel(nn.Module):
    """
    MoE 版统一材料模型：
    - n_comp_experts 个成分 Encoder
    - n_struct_experts 个结构 Encoder
    - gating 网络根据输入自动决定专家权重（Soft MoE）
    - head：每个任务一层 Linear(embed_dim → 1)

    任务名字与你当前 TASK_CONFIG 保持一致：
      steels_yield, expt_gap, glass, expt_is_metal,
      phonons, mp_gap, mp_e_form, log_kvrh, log_gvrh,
      perovskites, jdft2d, dielectric, mp_is_metal
    """

    def __init__(
        self,
        comp_dim: int | None,
        atom_dim: int | None,
        edge_dim: int | None,
        embed_dim: int = 256,
        n_comp_experts: int = 3,
        n_struct_experts: int = 2,
    ):
        super().__init__()

        self.comp_dim = comp_dim
        self.atom_dim = atom_dim
        self.edge_dim = edge_dim
        self.embed_dim = embed_dim
        self.n_comp_experts = n_comp_experts
        self.n_struct_experts = n_struct_experts

        # -----------------------
        # 1. 定义“哪些任务是成分 / 结构”
        # -----------------------
        self.composition_tasks = {
            "steels_yield",
            "expt_gap",
            "glass",
            "expt_is_metal",
        }
        self.structure_tasks = {
            "phonons",
            "mp_gap",
            "mp_e_form",
            "log_kvrh",
            "log_gvrh",
            "perovskites",
            "jdft2d",
            "dielectric",
            "mp_is_metal",
        }

        # -----------------------
        # 2. Expert：多个 Encoder
        # -----------------------
        self.comp_experts = None
        if comp_dim is not None:
            self.comp_experts = nn.ModuleList([
                CompositionEncoder(comp_dim, embed_dim)
                for _ in range(n_comp_experts)
            ])

        self.struct_experts = None
        if atom_dim is not None and edge_dim is not None:
            self.struct_experts = nn.ModuleList([
                GraphEncoder(atom_dim, edge_dim, embed_dim)
                for _ in range(n_struct_experts)
            ])

        # -----------------------
        # 3. Gating 网络
        # -----------------------
        # 成分 gating：输入 composition_vec [B, comp_dim]
        if comp_dim is not None:
            self.comp_gate = nn.Sequential(
                nn.Linear(comp_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, n_comp_experts),
            )
        else:
            self.comp_gate = None

        # 结构 gating：输入为 atom_fea 的均值向量 [atom_dim]
        #   做法：对每个结构样本，先对 atom_fea 做 mean pooling -> [atom_dim]
        if atom_dim is not None:
            self.struct_gate = nn.Sequential(
                nn.Linear(atom_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, n_struct_experts),
            )
        else:
            self.struct_gate = None

        # -----------------------
        # 4. Task heads
        # -----------------------
        self.task_heads = nn.ModuleDict({
            # composition + regression
            "steels_yield": nn.Linear(embed_dim, 1),
            "expt_gap": nn.Linear(embed_dim, 1),

            # composition + classification
            "glass": nn.Linear(embed_dim, 1),
            "expt_is_metal": nn.Linear(embed_dim, 1),

            # structure + regression
            "phonons": nn.Linear(embed_dim, 1),
            "mp_gap": nn.Linear(embed_dim, 1),
            "mp_e_form": nn.Linear(embed_dim, 1),
            "log_kvrh": nn.Linear(embed_dim, 1),
            "log_gvrh": nn.Linear(embed_dim, 1),
            "perovskites": nn.Linear(embed_dim, 1),
            "jdft2d": nn.Linear(embed_dim, 1),
            "dielectric": nn.Linear(embed_dim, 1),

            # structure + classification
            "mp_is_metal": nn.Linear(embed_dim, 1),
        })

    # -----------------------------
    #   成分 MoE forward
    # -----------------------------
    def _forward_composition(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, comp_dim]
        返回：MoE 聚合后的 embedding [B, embed_dim]
        """
        assert self.comp_experts is not None, "comp_experts is None but composition task was used."
        assert self.comp_gate is not None, "comp_gate is None but composition task was used."

        B = x.size(0)

        # gating: [B, n_comp_experts] -> softmax
        gate_logits = self.comp_gate(x)  # [B, n_comp_experts]
        gate = F.softmax(gate_logits, dim=-1)  # [B, n_comp_experts]

        # 每个 expert 计算 embedding: list of [B, D]
        expert_outputs: List[torch.Tensor] = []
        for expert in self.comp_experts:
            h = expert(x)  # [B, D]
            expert_outputs.append(h)

        # [B, n_comp_experts, D]
        stack_h = torch.stack(expert_outputs, dim=1)
        # gate: [B, n_comp_experts] -> [B, n_comp_experts, 1]
        gate_expanded = gate.unsqueeze(-1)

        # 加权求和 -> [B, D]
        moe_emb = (stack_h * gate_expanded).sum(dim=1)
        return moe_emb

    # -----------------------------
    #   结构 MoE forward
    # -----------------------------
    def _forward_structure(self, atom_fea, nbr_fea, nbr_idx) -> torch.Tensor:
        """
        当前版本假设 batch_size=1（与你的 StructureDataset 用法一致）
        atom_fea: [N_atoms, atom_dim]
        nbr_fea : [N_atoms, max_nbr, edge_dim]
        nbr_idx : [N_atoms, max_nbr]
        返回：MoE 聚合后的 embedding [1, embed_dim]
        """
        assert self.struct_experts is not None, "struct_experts is None but structure task was used."
        assert self.struct_gate is not None, "struct_gate is None but structure task was used."

        # gating 输入：对 atom_fea 做 mean pooling
        # atom_fea: [N_atoms, atom_dim]
        atom_mean = atom_fea.mean(dim=0, keepdim=True)  # [1, atom_dim]
        gate_logits = self.struct_gate(atom_mean)       # [1, n_struct_experts]
        gate = F.softmax(gate_logits, dim=-1)           # [1, n_struct_experts]

        # 每个结构 expert 得到 embedding: list of [1, D]
        expert_outputs: List[torch.Tensor] = []
        for expert in self.struct_experts:
            h = expert(atom_fea, nbr_fea, nbr_idx)  # 通常输出 [1, D] 或 [D]
            if h.dim() == 1:
                h = h.unsqueeze(0)  # [D] -> [1, D]
            expert_outputs.append(h)

        # [1, n_struct_experts, D]
        stack_h = torch.stack(expert_outputs, dim=1)   # dim=1: expert 维
        gate_expanded = gate.unsqueeze(-1)              # [1, n_struct_experts, 1]

        # [1, D]
        moe_emb = (stack_h * gate_expanded).sum(dim=1)
        return moe_emb

    # -----------------------------
    #   统一 forward
    # -----------------------------
    def forward(self, inputs: Dict[str, Any], task_name: str) -> torch.Tensor:
        if task_name not in self.task_heads:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Available tasks: {list(self.task_heads.keys())}"
            )

        # ---- Composition tasks ----
        if task_name in self.composition_tasks:
            if "composition_vec" not in inputs:
                raise ValueError(f"Task {task_name} expects 'composition_vec' in inputs.")
            x = inputs["composition_vec"]
            emb = self._forward_composition(x)

        # ---- Structure tasks ----
        elif task_name in self.structure_tasks:
            for k in ("atom_fea", "nbr_fea", "nbr_idx"):
                if k not in inputs:
                    raise ValueError(f"Task {task_name} expects '{k}' in inputs.")
            atom_fea = inputs["atom_fea"]
            nbr_fea = inputs["nbr_fea"]
            nbr_idx = inputs["nbr_idx"]
            emb = self._forward_structure(atom_fea, nbr_fea, nbr_idx)

        else:
            raise ValueError(f"Task {task_name} is neither composition nor structure task.")

        # task-specific head
        out = self.task_heads[task_name](emb)  # [B,1] 或 [1,1]
        return out