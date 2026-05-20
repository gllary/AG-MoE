# mat_models/moe/Stage2MoEModel_v24.py
# v24 Stage2 = v22 + task-wise continuous sharing + beta warmup
#
# Key upgrade over v23:
# - Introduce beta warmup:
#       for first N epochs, force beta(task) = 1 (full MoE sharing)
# - Prevents large-data tasks (e.g. mp_gap) from being harmed
#   by early isolation or under-sharing
#
# Paper-ready interpretation:
# - Stage2 learns a continuous spectrum of task-wise sharing
# - Warmup ensures router + experts are first co-adapted globally
# - Later epochs allow task-specific specialization

from __future__ import annotations

from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn

from mat_models.moe.Stage2MoEModel_v18 import (
    ExpertSpec,
    FrozenExpert,
    TaskAwareTopKRouter,
)


class Stage2MoEModelV24(nn.Module):
    """
    v24 routing policy:

    - composition tasks:
        * strict one-hot routing (IDENTICAL to v18)

    - structure tasks:
        * if task in NO_MOE_TASKS AND epoch >= beta_warmup_epochs:
              h = h_self
        * else:
              h = h_self + beta_eff(task, epoch) * h_moe

    where:
        beta_eff(task, epoch) =
            1                              if epoch < beta_warmup_epochs
            tanh(raw_beta(task)) * clip    otherwise
    """

    # -------- task groups --------

    # gated residual candidates
    RESIDUAL_TASKS: Set[str] = {
        "mp_e_form",
        "log_kvrh",
        "log_gvrh",
        "mp_gap",          # ✅ 现在允许 mp_gap 自己学共享强度
    }

    # expert isolation targets (small/noisy tasks)
    NO_MOE_TASKS: Set[str] = {
        "phonons",
    }

    # --------------------------------------------------

    def __init__(
        self,
        experts: List[ExpertSpec],
        moe_dim: int = 512,
        top_k: int = 2,
        router_hidden: int = 256,
        router_dropout: float = 0.1,
        alpha_clip: float = 1.0,
        beta_warmup_epochs: int = 10,   # ⭐ 新增：beta warmup
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.moe_dim = int(moe_dim)
        self.alpha_clip = float(alpha_clip)
        self.beta_warmup_epochs = int(beta_warmup_epochs)

        # 当前 epoch（由训练脚本注入）
        self.current_epoch: int = 0

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
            {s.name: nn.Linear(self.moe_dim, 1) for s in experts}
        )

        # ---------- router ----------
        self.struct_router = TaskAwareTopKRouter(
            moe_dim=self.moe_dim,
            num_tasks=len(self.task_names),
            num_experts=max(1, len(self.struct_expert_indices)),
            hidden=router_hidden,
            dropout=router_dropout,
            k=min(top_k, max(1, len(self.struct_expert_indices))),
        )

        # ---------- task-wise beta ----------
        # raw_beta = 0 => tanh(0)=0 (no residual)
        self.raw_task_beta = nn.Embedding(len(self.task_names), 1)
        nn.init.zeros_(self.raw_task_beta.weight)

        # ---------- logging hooks ----------
        self._last_router_w = None
        self._last_router_q = None

    # --------------------------------------------------
    # epoch control (IMPORTANT)
    # --------------------------------------------------
    def set_epoch(self, epoch: int):
        """
        Called by training loop at beginning of each epoch.
        """
        self.current_epoch = int(epoch)

    # --------------------------------------------------
    # utils
    # --------------------------------------------------
    def _infer_batch_size(self, batch_dict: Dict[str, torch.Tensor]) -> int:
        if "composition_vec" in batch_dict:
            return int(batch_dict["composition_vec"].size(0))
        if "batch" in batch_dict:
            return int(batch_dict["batch"].max().item() + 1)
        raise RuntimeError("[Stage2] Cannot infer batch size")

    @torch.no_grad()
    def get_task_beta(self) -> Dict[str, float]:
        """
        Return effective beta(task) at current epoch (for logging / paper).
        """
        raw = self.raw_task_beta.weight.squeeze(1)
        if self.current_epoch < self.beta_warmup_epochs:
            beta = torch.ones_like(raw)
        else:
            beta = torch.tanh(raw) * self.alpha_clip

        return {t: float(beta[self.task2id[t]].item()) for t in self.task2id}

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
        # composition task (v18)
        # ==================================================
        if spec.mode == "composition":
            with torch.no_grad():
                h = self.experts[idx].encode(batch_dict)

            self._last_router_q = h.detach()
            w_full = torch.zeros(B, self.num_experts, device=h.device)
            w_full[:, idx] = 1.0
            self._last_router_w = w_full.detach()

            return self.task_heads[task_name](h)

        # ==================================================
        # structure tasks
        # ==================================================
        struct_ids = self.struct_expert_indices
        if len(struct_ids) == 0:
            raise RuntimeError("[Stage2] No structure experts found")

        # ---------- expert isolation ----------
        if (
            task_name in self.NO_MOE_TASKS
            and self.current_epoch >= self.beta_warmup_epochs
        ):
            with torch.no_grad():
                h_self = self.experts[idx].encode(batch_dict)

            self._last_router_q = h_self.detach()
            w_full = torch.zeros(B, self.num_experts, device=h_self.device)
            w_full[:, idx] = 1.0
            self._last_router_w = w_full.detach()

            return self.task_heads[task_name](h_self)

        # ---------- normal MoE path ----------
        z_list = []
        with torch.no_grad():
            for ei in struct_ids:
                z_list.append(self.experts[ei].encode(batch_dict))
        z = torch.stack(z_list, dim=1)  # (B, E, D)

        q = z.mean(dim=1)
        self._last_router_q = q.detach()

        task_id_tensor = torch.full(
            (B,), task_id, device=q.device, dtype=torch.long
        )
        w_struct = self.struct_router(q, task_id_tensor)

        w_full = torch.zeros(B, self.num_experts, device=q.device)
        w_full[:, struct_ids] = w_struct
        self._last_router_w = w_full.detach()

        h_moe = (z * w_struct.unsqueeze(-1)).sum(dim=1)

        # ---------- beta logic ----------
        if task_name in self.RESIDUAL_TASKS:
            self_pos = struct_ids.index(idx)
            h_self = z[:, self_pos]

            if self.current_epoch < self.beta_warmup_epochs:
                beta = torch.ones(B, 1, device=h_moe.device, dtype=h_moe.dtype)
            else:
                raw = self.raw_task_beta(task_id_tensor).to(h_moe.dtype)
                beta = torch.tanh(raw) * self.alpha_clip

            h = h_self + beta * h_moe
        else:
            h = h_moe

        return self.task_heads[task_name](h)