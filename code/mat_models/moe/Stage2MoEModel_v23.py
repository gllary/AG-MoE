# mat_models/moe/Stage2MoEModel_v23.py
# v23 Stage2 = v22 -> "Continuous Task-wise Sharing" (paper-ready)
#
# Key idea:
# - Replace hard Expert Isolation (NO_MOE_TASKS) + selective residual with ONE unified mechanism:
#       h = (1 - beta(task)) * h_self + beta(task) * h_moe
#   where beta(task) ∈ [0,1] is learnable (per-task scalar).
#
# Properties:
# - beta≈1  => identical to v18 MoE aggregation (full sharing)
# - beta≈0  => identical to "expert isolation" (self-only)
# - intermediate => soft sharing continuum (learned)
#
# Initialization (safe, do-no-harm):
# - For tasks in NO_MOE_TASKS (e.g., phonons): init beta ~ 0
# - For other structure tasks: init beta ~ 1 (keeps v18-like behavior at start)
#
# Router logging shapes (_last_router_w/_last_router_q) remain compatible with v18 training script.

from __future__ import annotations

from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn

from mat_models.moe.Stage2MoEModel_v18 import (
    ExpertSpec,
    FrozenExpert,
    TaskAwareTopKRouter,
)


class Stage2MoEModelV23(nn.Module):
    """
    v23 routing policy:

    - composition tasks:
        * strict one-hot routing to their own expert (IDENTICAL to v18)

    - structure tasks:
        * ALWAYS compute task-aware top-k MoE among structure experts (same router as v18)
        * output uses continuous sharing gate:
              h = (1-beta(task)) * h_self + beta(task) * h_moe
          beta(task): learnable scalar in [0,1], init:
              - NO_MOE_TASKS: beta ~ 0  (soft isolation)
              - else:         beta ~ 1  (start identical to v18)
    """

    # 你在 v22 里“想隔离”的任务，在 v23 里只用于初始化（不再硬分支）
    NO_MOE_TASKS: Set[str] = {
        "phonons",
    }

    def __init__(
        self,
        experts: List[ExpertSpec],
        moe_dim: int = 512,
        top_k: int = 2,
        router_hidden: int = 256,
        router_dropout: float = 0.1,
        # beta init settings
        beta_init_on: float = 0.98,   # for normal structure tasks (≈1 => v18)
        beta_init_off: float = 0.02,  # for NO_MOE_TASKS (≈0 => isolation)
        beta_temperature: float = 1.0,  # sigmoid temperature (>=1 softer; <1 sharper)
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.moe_dim = int(moe_dim)

        self.beta_temperature = float(beta_temperature)
        if self.beta_temperature <= 0:
            raise ValueError("beta_temperature must be > 0")

        # ---------- experts (IDENTICAL to v18/v21/v22) ----------
        self.expert_specs = experts
        self.experts = nn.ModuleList([FrozenExpert(s, self.device) for s in experts]).to(self.device)
        self.num_experts = len(self.experts)

        # ---------- task indexing ----------
        self.task_names = [s.name for s in experts]
        self.task2id = {t: i for i, t in enumerate(self.task_names)}
        self.name2expert_idx = {s.name: i for i, s in enumerate(experts)}
        self.struct_expert_indices = [i for i, s in enumerate(experts) if s.mode == "structure"]

        # ---------- heads ----------
        self.task_heads = nn.ModuleDict({s.name: nn.Linear(self.moe_dim, 1) for s in experts})

        # ---------- router (IDENTICAL to v18) ----------
        self.struct_router = TaskAwareTopKRouter(
            moe_dim=self.moe_dim,
            num_tasks=len(self.task_names),
            num_experts=max(1, len(self.struct_expert_indices)),
            hidden=router_hidden,
            dropout=router_dropout,
            k=min(top_k, max(1, len(self.struct_expert_indices))),
        )

        # ---------- task-wise beta gate ----------
        # We store raw logits, beta = sigmoid(raw / temperature)
        self.raw_task_beta = nn.Embedding(len(self.task_names), 1)

        # Safe init:
        # - for NO_MOE_TASKS: beta ~ beta_init_off
        # - else:            beta ~ beta_init_on
        self._init_task_beta(beta_init_on=beta_init_on, beta_init_off=beta_init_off)

        # logs for v18 training script
        self._last_router_w = None
        self._last_router_q = None

    # --------------------------------------------------
    # init helpers
    # --------------------------------------------------
    def _inv_sigmoid(self, p: float) -> float:
        eps = 1e-6
        p = float(max(eps, min(1.0 - eps, p)))
        return float(torch.log(torch.tensor(p / (1.0 - p))).item())

    def _init_task_beta(self, beta_init_on: float, beta_init_off: float):
        # raw = logit(beta) * temperature
        raw_on = self._inv_sigmoid(beta_init_on) * self.beta_temperature
        raw_off = self._inv_sigmoid(beta_init_off) * self.beta_temperature

        with torch.no_grad():
            self.raw_task_beta.weight.fill_(raw_on)

            for t in self.NO_MOE_TASKS:
                if t in self.task2id:
                    tid = self.task2id[t]
                    self.raw_task_beta.weight[tid].fill_(raw_off)

    # --------------------------------------------------
    # utils
    # --------------------------------------------------
    def _infer_batch_size(self, batch_dict: Dict[str, torch.Tensor]) -> int:
        if "composition_vec" in batch_dict:
            return int(batch_dict["composition_vec"].size(0))
        if "batch" in batch_dict:
            return int(batch_dict["batch"].max().item() + 1)
        raise RuntimeError("[Stage2] Cannot infer batch size from batch_dict")

    def _beta(self, task_id_tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        beta(task) in (0,1), shape (B,1)
        """
        raw = self.raw_task_beta(task_id_tensor).to(dtype)  # (B,1)
        return torch.sigmoid(raw / self.beta_temperature)

    @torch.no_grad()
    def get_task_beta(self) -> Dict[str, float]:
        """
        For analysis / paper: current beta(task) after sigmoid+temperature.
        """
        raw = self.raw_task_beta.weight.squeeze(1)  # (T,)
        beta = torch.sigmoid(raw / self.beta_temperature)
        return {t: float(beta[self.task2id[t]].item()) for t in self.task2id.keys()}

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
        # composition: IDENTICAL to v18 (one-hot)
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
        # structure: continuous task-wise sharing
        # ==================================================
        struct_ids = self.struct_expert_indices
        if len(struct_ids) == 0:
            raise RuntimeError("[Stage2] No structure experts found but got structure task.")

        # encode all structure experts (frozen)
        z_list = []
        with torch.no_grad():
            for ei in struct_ids:
                z_list.append(self.experts[ei].encode(batch_dict))
        z = torch.stack(z_list, dim=1)  # (B, E_struct, D)

        # router query (same as v18)
        q = z.mean(dim=1)
        self._last_router_q = q.detach()

        task_id_tensor = torch.full((B,), task_id, device=q.device, dtype=torch.long)
        w_struct = self.struct_router(q, task_id_tensor)  # (B, E_struct)

        # expand to full expert space (for logging; MUST match num_experts)
        w_full = torch.zeros(B, self.num_experts, device=q.device, dtype=torch.float32)
        w_full[:, struct_ids] = w_struct
        self._last_router_w = w_full.detach()

        # MoE aggregation (v18)
        h_moe = (z * w_struct.unsqueeze(-1)).sum(dim=1)  # (B, D)

        # self expert embedding
        try:
            self_struct_pos = struct_ids.index(idx)
        except ValueError as e:
            raise RuntimeError(
                f"[Stage2] task={task_name} is structure but expert idx={idx} not in struct_ids"
            ) from e
        h_self = z[:, self_struct_pos]  # (B, D)

        # task-wise beta gate
        beta = self._beta(task_id_tensor, dtype=h_moe.dtype)  # (B,1)

        # convex combination: (1-beta)*self + beta*moe
        h = (1.0 - beta) * h_self + beta * h_moe

        return self.task_heads[task_name](h)