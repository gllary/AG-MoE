# mat_models/moe/Stage2MoEModel_v20.py
# v20 = v18 + Learnable Task-wise Residual Gate (structure only)
#
# Goal:
# - Keep v18 behavior for the already-strong tasks (alpha -> 1 => EXACT v18 behavior)
# - Automatically learn which structure tasks should rely more on self expert (alpha -> 0)
#
# For STRUCTURE tasks:
#   z: (B, E_struct, D) frozen expert embeddings
#   h_moe  = sum_e w_e * z_e         (same as v18)
#   h_self = z_{self_task_expert}
#   alpha_t in (0,1): learnable per-task gate
#   h = alpha_t * h_moe + (1-alpha_t) * h_self
#
# For COMPOSITION tasks:
#   EXACTLY the same as v18: strict one-hot route to its own expert.

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from mat_models.moe.Stage2MoEModel_v18 import (
    ExpertSpec,
    FrozenExpert,
    TaskAwareTopKRouter,
)


class Stage2MoEModelV20(nn.Module):
    """
    v20 routing policy:

    - composition task:
        * strict one-hot routing (IDENTICAL to v18)

    - structure task:
        * router: IDENTICAL to v18 (task-aware top-k among structure experts)
        * NEW: learnable task-wise gate alpha_t in (0,1):
              h = alpha_t * h_moe + (1-alpha_t) * h_self
          where alpha_t is learned automatically per task.

    Notes:
    - When alpha_t -> 1, behavior is EXACTLY v18 (h == h_moe).
    - When alpha_t -> 0, behavior becomes pure self-expert (h == h_self).
    """

    def __init__(
        self,
        experts: List[ExpertSpec],
        moe_dim: int = 512,
        top_k: int = 2,
        router_hidden: int = 256,
        router_dropout: float = 0.1,
        device: Optional[torch.device] = None,
        # gate init: larger => alpha closer to 1.0 (v18-like).
        gate_init_logit: float = 6.0,  # sigmoid(6)=0.9975 ~ almost v18
    ):
        super().__init__()
        self.moe_dim = moe_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------- experts (same as v18) ----------
        self.expert_specs = experts
        self.experts = nn.ModuleList([FrozenExpert(s, self.device) for s in experts]).to(self.device)
        self.num_experts = len(self.experts)

        # ---------- task indexing ----------
        self.task_names = [s.name for s in experts]
        self.task2id = {t: i for i, t in enumerate(self.task_names)}
        self.num_tasks = len(self.task_names)

        self.name2expert_idx = {s.name: i for i, s in enumerate(experts)}
        self.struct_expert_indices = [i for i, s in enumerate(experts) if s.mode == "structure"]

        # ---------- heads (same) ----------
        self.task_heads = nn.ModuleDict({s.name: nn.Linear(moe_dim, 1) for s in experts})

        # ---------- router (IDENTICAL to v18) ----------
        self.struct_router = TaskAwareTopKRouter(
            moe_dim=moe_dim,
            num_tasks=self.num_tasks,
            num_experts=len(self.struct_expert_indices) if len(self.struct_expert_indices) > 0 else 1,
            hidden=router_hidden,
            dropout=router_dropout,
            k=min(top_k, max(1, len(self.struct_expert_indices))),
        )

        # ---------- NEW: task-wise residual gate ----------
        # store logits, apply sigmoid in forward to get alpha in (0,1)
        # we learn alpha for ALL tasks, but only apply it for STRUCTURE tasks.
        self.gate_logit = nn.Parameter(torch.full((self.num_tasks,), float(gate_init_logit)))

        # ---------- debug ----------
        self._last_router_w = None
        self._last_router_q = None
        self._last_alpha = None

    # --------------------------------------------------
    # utils (same as v18)
    # --------------------------------------------------
    def _infer_batch_size(self, batch_dict: Dict[str, torch.Tensor]) -> int:
        if "composition_vec" in batch_dict:
            return int(batch_dict["composition_vec"].size(0))
        if "batch" in batch_dict:
            return int(batch_dict["batch"].max().item() + 1)
        raise RuntimeError("[Stage2] Cannot infer batch size from batch_dict")

    @torch.no_grad()
    def get_task_alpha(self, task_name: str) -> float:
        """Return current alpha_t (after sigmoid) for logging/analysis."""
        tid = self.task2id[task_name]
        return float(torch.sigmoid(self.gate_logit[tid]).item())

    def gate_regularizer(self, target: float = 1.0) -> torch.Tensor:
        """
        Optional: encourage alpha_t close to target (default 1.0 to preserve v18).
        Trainer can add: loss += lambda_gate * model.gate_regularizer()
        """
        alpha = torch.sigmoid(self.gate_logit)
        return ((alpha - target) ** 2).mean()

    # --------------------------------------------------
    # forward
    # --------------------------------------------------
    def forward(self, task_name: str, batch_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert task_name in self.task2id, f"Unknown task_name={task_name}"
        B = self._infer_batch_size(batch_dict)

        task_id = self.task2id[task_name]
        idx = self.name2expert_idx[task_name]
        spec_mode = self.expert_specs[idx].mode

        # ==================================================
        # composition: IDENTICAL to v18 (one-hot)
        # ==================================================
        if spec_mode == "composition":
            with torch.no_grad():
                h = self.experts[idx].encode(batch_dict)

            self._last_router_q = h.detach()
            w_full = torch.zeros(B, self.num_experts, device=h.device, dtype=torch.float32)
            w_full[:, idx] = 1.0
            self._last_router_w = w_full.detach()
            self._last_alpha = None

            return self.task_heads[task_name](h)

        # ==================================================
        # structure: v18 router + learnable residual gate
        # ==================================================
        struct_ids = self.struct_expert_indices
        if len(struct_ids) == 0:
            raise RuntimeError("[Stage2] No structure experts found but got structure task.")

        # encode all structure experts (frozen) - same as v18
        z_list = []
        with torch.no_grad():
            for ei in struct_ids:
                z_list.append(self.experts[ei].encode(batch_dict))
        z = torch.stack(z_list, dim=1)  # (B, E_struct, D)

        q = z.mean(dim=1)
        self._last_router_q = q.detach()

        task_id_tensor = torch.full((B,), task_id, device=q.device, dtype=torch.long)
        w_struct = self.struct_router(q, task_id_tensor)  # (B, E_struct)

        # expand to full expert space for logging (same shape as v18)
        w_full = torch.zeros(B, self.num_experts, device=q.device, dtype=torch.float32)
        w_full[:, struct_ids] = w_struct
        self._last_router_w = w_full.detach()

        h_moe = (z * w_struct.unsqueeze(-1)).sum(dim=1)  # (B, D)

        # self expert embedding
        self_pos = struct_ids.index(idx)  # position inside z
        h_self = z[:, self_pos]           # (B, D)

        # learnable alpha_t in (0,1) ; alpha->1 => EXACT v18 (h==h_moe)
        alpha_t = torch.sigmoid(self.gate_logit[task_id]).to(h_moe.dtype)  # scalar
        self._last_alpha = float(alpha_t.item())

        h = alpha_t * h_moe + (1.0 - alpha_t) * h_self
        return self.task_heads[task_name](h)