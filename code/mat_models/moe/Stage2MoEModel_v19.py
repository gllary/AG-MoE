# mat_models/moe/Stage2MoEModel_v19.py
# v19 = v18 + selective GATED residual MoE
#
# Only for fragile structure tasks:
#   mp_e_form / log_kvrh / log_gvrh
# we apply:
#   h = h_self + alpha(task) * h_moe
# where alpha is learnable and INIT=0  (safe, no performance collapse at start)
#
# All other tasks behave EXACTLY the same as v18.

from __future__ import annotations

from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn

from mat_models.moe.Stage2MoEModel_v18 import (
    ExpertSpec,
    FrozenExpert,
    TaskAwareTopKRouter,
)


class Stage2MoEModelV19(nn.Module):
    """
    v19 routing policy:

    - composition task:
        * strict one-hot routing (IDENTICAL to v18)

    - structure task:
        * default: IDENTICAL to v18 (pure task-aware MoE)
        * ONLY for fragile tasks:
              mp_e_form / log_kvrh / log_gvrh
          apply GATED residual:
              h = h_self + alpha(task) * h_moe
          alpha(task) is learnable scalar, initialized to 0.
    """

    RESIDUAL_TASKS: Set[str] = {
        "mp_e_form",
        "log_kvrh",
        "log_gvrh",
    }

    def __init__(
        self,
        experts: List[ExpertSpec],
        moe_dim: int = 512,
        top_k: int = 2,
        router_hidden: int = 256,
        router_dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.moe_dim = moe_dim

        # ---------- experts ----------
        self.expert_specs = experts
        self.experts = nn.ModuleList([FrozenExpert(s, self.device) for s in experts]).to(self.device)
        self.num_experts = len(self.experts)

        # ---------- task indexing ----------
        self.task_names = [s.name for s in experts]
        self.task2id = {t: i for i, t in enumerate(self.task_names)}
        self.name2expert_idx = {s.name: i for i, s in enumerate(experts)}
        self.struct_expert_indices = [i for i, s in enumerate(experts) if s.mode == "structure"]

        # ---------- heads ----------
        self.task_heads = nn.ModuleDict({s.name: nn.Linear(moe_dim, 1) for s in experts})

        # ---------- router (IDENTICAL to v18) ----------
        self.struct_router = TaskAwareTopKRouter(
            moe_dim=moe_dim,
            num_tasks=len(self.task_names),
            num_experts=max(1, len(self.struct_expert_indices)),
            hidden=router_hidden,
            dropout=router_dropout,
            k=min(top_k, max(1, len(self.struct_expert_indices))),
        )

        # ---------- GATED residual alpha (per task scalar) ----------
        # We only USE it for RESIDUAL_TASKS, but easiest is to allocate for all tasks.
        self.task_alpha = nn.Embedding(len(self.task_names), 1)
        nn.init.zeros_(self.task_alpha.weight)  # IMPORTANT: start with alpha=0 (safe)

        # logs
        self._last_router_w = None
        self._last_router_q = None

    # --------------------------------------------------
    # utils
    # --------------------------------------------------
    def _infer_batch_size(self, batch_dict: Dict[str, torch.Tensor]) -> int:
        if "composition_vec" in batch_dict:
            return int(batch_dict["composition_vec"].size(0))
        if "batch" in batch_dict:
            return int(batch_dict["batch"].max().item() + 1)
        raise RuntimeError("[Stage2] Cannot infer batch size")

    # --------------------------------------------------
    # forward
    # --------------------------------------------------
    def forward(self, task_name: str, batch_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert task_name in self.task2id, f"Unknown task: {task_name}"

        B = self._infer_batch_size(batch_dict)
        task_id = self.task2id[task_name]

        idx = self.name2expert_idx[task_name]
        spec = self.expert_specs[idx]

        # ==================================================
        # composition task: IDENTICAL to v18
        # ==================================================
        if spec.mode == "composition":
            with torch.no_grad():
                h = self.experts[idx].encode(batch_dict)

            self._last_router_q = h.detach()
            w_full = torch.zeros(B, self.num_experts, device=h.device, dtype=torch.float32)
            w_full[:, idx] = 1.0
            self._last_router_w = w_full.detach()

            return self.task_heads[task_name](h)

        # ==================================================
        # structure task
        # ==================================================
        struct_ids = self.struct_expert_indices
        assert len(struct_ids) > 0, "No structure experts found"

        # ---- encode all structure experts (frozen) ----
        z_list = []
        with torch.no_grad():
            for ei in struct_ids:
                z_list.append(self.experts[ei].encode(batch_dict))
        z = torch.stack(z_list, dim=1)  # (B, E_struct, D)

        q = z.mean(dim=1)
        self._last_router_q = q.detach()

        task_id_tensor = torch.full((B,), task_id, device=q.device, dtype=torch.long)
        w_struct = self.struct_router(q, task_id_tensor)  # (B, E_struct)

        # expand to full expert space (for logging only, MUST be length = num_experts)
        w_full = torch.zeros(B, self.num_experts, device=q.device, dtype=torch.float32)
        w_full[:, struct_ids] = w_struct
        self._last_router_w = w_full.detach()

        h_moe = (z * w_struct.unsqueeze(-1)).sum(dim=1)  # (B, D)

        # ==================================================
        # ⭐ selective GATED residual (ONLY 3 tasks)
        # ==================================================
        if task_name in self.RESIDUAL_TASKS:
            # self expert position inside struct list
            try:
                self_struct_pos = struct_ids.index(idx)
            except ValueError as e:
                raise RuntimeError(
                    f"[Stage2] task {task_name} is structure but its expert idx={idx} not in struct_ids"
                ) from e

            h_self = z[:, self_struct_pos]  # (B, D)

            # alpha(task): shape (B,1), init=0, learnable
            alpha = self.task_alpha(task_id_tensor).to(h_moe.dtype)  # (B,1)

            # optional safety clamp to avoid explosion (can comment out)
            alpha = torch.clamp(alpha, -1.0, 1.0)

            h = h_self + alpha * h_moe
        else:
            # EXACTLY v18 behavior
            h = h_moe

        return self.task_heads[task_name](h)