# mat_models/moe/Stage2MoEModel_v25.py
# V25: Adaptive Sharing Gate MoE
#
# h = (1 - g) * h_self + g * h_moe
# g = sigmoid( GateMLP( concat(q, task_emb) ) )
#
# No manual task rules. Fully adaptive.

from __future__ import annotations
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from mat_models.moe.Stage2MoEModel_v18 import (
    ExpertSpec,
    FrozenExpert,
    TaskAwareTopKRouter,
)


class AdaptiveSharingGate(nn.Module):
    """
    g = sigmoid( MLP([q, task_emb]) )
    """
    def __init__(self, moe_dim: int, num_tasks: int, hidden: int = 256):
        super().__init__()
        self.task_emb = nn.Embedding(num_tasks, moe_dim)

        self.net = nn.Sequential(
            nn.Linear(2 * moe_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, q: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        t_emb = self.task_emb(task_id)
        x = torch.cat([q, t_emb], dim=-1)
        g = torch.sigmoid(self.net(x))  # (B,1)
        return g


class Stage2MoEModelV25(nn.Module):
    """
    V25 Stage2 MoE with Adaptive Sharing Gate
    """

    def __init__(
        self,
        experts: List[ExpertSpec],
        moe_dim: int = 512,
        top_k: int = 2,
        router_hidden: int = 256,
        router_dropout: float = 0.1,
        gate_hidden: int = 256,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.moe_dim = moe_dim

        # ---------- experts ----------
        self.expert_specs = experts
        self.experts = nn.ModuleList(
            [FrozenExpert(s, self.device) for s in experts]
        ).to(self.device)
        self.num_experts = len(self.experts)

        # ---------- task indexing ----------
        self.task_names = [s.name for s in experts]
        self.task2id = {t: i for i, t in enumerate(self.task_names)}
        self.name2expert_idx = {s.name: i for i, s in enumerate(experts)}
        self.struct_expert_indices = [
            i for i, s in enumerate(experts) if s.mode == "structure"
        ]

        # ---------- heads ----------
        self.task_heads = nn.ModuleDict(
            {s.name: nn.Linear(moe_dim, 1) for s in experts}
        )

        # ---------- router ----------
        self.struct_router = TaskAwareTopKRouter(
            moe_dim=moe_dim,
            num_tasks=len(self.task_names),
            num_experts=max(1, len(self.struct_expert_indices)),
            hidden=router_hidden,
            dropout=router_dropout,
            k=min(top_k, max(1, len(self.struct_expert_indices))),
        )

        # ---------- adaptive sharing gate ----------
        self.share_gate = AdaptiveSharingGate(
            moe_dim=moe_dim,
            num_tasks=len(self.task_names),
            hidden=gate_hidden,
        )

        # logs (for training script compatibility)
        self._last_router_w = None
        self._last_router_q = None
        self._last_gate_g = None

    # --------------------------------------------------
    def _infer_batch_size(self, batch_dict: Dict[str, torch.Tensor]) -> int:
        if "composition_vec" in batch_dict:
            return int(batch_dict["composition_vec"].size(0))
        if "batch" in batch_dict:
            return int(batch_dict["batch"].max().item() + 1)
        raise RuntimeError("[Stage2] Cannot infer batch size")

    # --------------------------------------------------
    def forward(self, task_name: str, batch_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert task_name in self.task2id, f"Unknown task: {task_name}"

        B = self._infer_batch_size(batch_dict)
        task_id = self.task2id[task_name]
        idx = self.name2expert_idx[task_name]
        spec = self.expert_specs[idx]

        # ================================
        # composition: one-hot (same as v18)
        # ================================
        if spec.mode == "composition":
            with torch.no_grad():
                h = self.experts[idx].encode(batch_dict)

            self._last_router_q = h.detach()
            w_full = torch.zeros(B, self.num_experts, device=h.device)
            w_full[:, idx] = 1.0
            self._last_router_w = w_full.detach()
            self._last_gate_g = None

            return self.task_heads[task_name](h)

        # ================================
        # structure tasks
        # ================================
        struct_ids = self.struct_expert_indices
        z_list = []
        with torch.no_grad():
            for ei in struct_ids:
                z_list.append(self.experts[ei].encode(batch_dict))
        z = torch.stack(z_list, dim=1)  # (B, E, D)

        # router
        q = z.mean(dim=1)
        self._last_router_q = q.detach()

        task_id_tensor = torch.full((B,), task_id, device=q.device, dtype=torch.long)
        w_struct = self.struct_router(q, task_id_tensor)

        w_full = torch.zeros(B, self.num_experts, device=q.device)
        w_full[:, struct_ids] = w_struct
        self._last_router_w = w_full.detach()

        h_moe = (z * w_struct.unsqueeze(-1)).sum(dim=1)

        # self expert
        self_pos = struct_ids.index(idx)
        h_self = z[:, self_pos]

        # adaptive gate
        g = self.share_gate(q, task_id_tensor)  # (B,1)
        self._last_gate_g = g.detach()

        h = (1.0 - g) * h_self + g * h_moe
        return self.task_heads[task_name](h)