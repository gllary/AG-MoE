# mat_models/moe/Stage2MoEModel_v21.py
# v21 Stage2 = v18 (stable) + selective task-wise gated residual
#
# Motivation (paper-ready):
# - v18 is strong and stable for most tasks, but a few "fragile" structure tasks suffer from negative transfer.
# - For those tasks only, we add a safe residual bypass:
#       h = h_self + alpha(task) * h_moe
#   where alpha(task) is a learnable scalar initialized to 0 (no harm at start).
# - For ALL other tasks, behavior is IDENTICAL to v18 (Do-No-Harm principle).
#
# Key properties:
# - Reuses v18 experts/router/load logic (no changes)
# - Router logging (_last_router_w/_last_router_q) matches v18 shapes for training stats
# - alpha(task) is stable via tanh() squash (optional but recommended)

from __future__ import annotations

from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn

from mat_models.moe.Stage2MoEModel_v18 import (
    ExpertSpec,
    FrozenExpert,
    TaskAwareTopKRouter,
)


class Stage2MoEModelV21(nn.Module):
    """
    v21 routing policy:

    - composition tasks:
        * strict one-hot routing to their own expert (IDENTICAL to v18)

    - structure tasks:
        * default: IDENTICAL to v18 (task-aware top-k MoE among structure experts)
        * ONLY for RESIDUAL_TASKS:
              h = h_self + alpha(task) * h_moe
          alpha(task): learnable scalar, init=0 (safe warm start)

    Notes:
    - alpha is squashed by tanh to keep it bounded in [-alpha_clip, alpha_clip].
      This makes training more stable and paper-friendly.
    """

    # ✅ 只对这些“脆弱结构任务”启用 gated residual
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
        alpha_clip: float = 1.0,   # tanh 后再乘 clip，默认范围 [-1, 1]
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.moe_dim = int(moe_dim)
        self.alpha_clip = float(alpha_clip)

        # ---------- experts (IDENTICAL to v18) ----------
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

        # ---------- task-wise gated residual alpha ----------
        # raw_alpha init=0 => tanh(0)=0 => alpha=0 (safe, no effect at start)
        self.raw_task_alpha = nn.Embedding(len(self.task_names), 1)
        nn.init.zeros_(self.raw_task_alpha.weight)

        # logs (to keep train_v18 router stats code working)
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
        raise RuntimeError("[Stage2] Cannot infer batch size from batch_dict")

    @torch.no_grad()
    def get_task_alpha(self) -> Dict[str, float]:
        """
        For analysis / paper: returns current alpha(task) after squash+clip.
        """
        raw = self.raw_task_alpha.weight.squeeze(1)  # (T,)
        alpha = torch.tanh(raw) * self.alpha_clip
        out = {}
        for t, tid in self.task2id.items():
            out[t] = float(alpha[tid].item())
        return out

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
        # structure: IDENTICAL to v18 unless task in RESIDUAL_TASKS
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

        # router query
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

        # ---------- selective gated residual ----------
        if task_name in self.RESIDUAL_TASKS:
            # find own expert position within struct expert list
            try:
                self_struct_pos = struct_ids.index(idx)
            except ValueError as e:
                raise RuntimeError(
                    f"[Stage2] task={task_name} is structure but expert idx={idx} not in struct_ids"
                ) from e

            h_self = z[:, self_struct_pos]  # (B, D)

            # alpha(task): tanh-squashed, bounded
            raw_alpha = self.raw_task_alpha(task_id_tensor).to(h_moe.dtype)  # (B,1)
            alpha = torch.tanh(raw_alpha) * self.alpha_clip                 # (B,1)

            h = h_self + alpha * h_moe
        else:
            # EXACTLY v18 behavior
            h = h_moe

        return self.task_heads[task_name](h)