# mat_models/moe/JointAGMoEModel.py
#
# Fully Joint-from-Scratch AG-MoE baseline.
#
# Architecture is IDENTICAL to Stage2MoEModelV25 (same router, same adaptive gate,
# same task heads, same composition/structure routing policy), with ONE key difference:
#
#   - FrozenExpert  (Stage2) : loads Stage-1 checkpoint, freezes ALL encoder weights.
#   - TrainableExpert (here) : initialises encoder from scratch, ALL weights trainable.
#
# This baseline answers: "Is Stage-1 specialist pre-training necessary, or can the
# same AG-MoE architecture be trained end-to-end from random initialisation?"
#
# Usage:
#   from mat_models.moe.JointAGMoEModel import JointAGMoEModel, FreshExpertSpec, probe_dims

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from mat_models.encoders.composition_expert_v17 import CompositionExpertV17
from mat_models.encoders.structure_expert_v17 import StructureExpertV17
from mat_models.moe.Stage2MoEModel_v25 import AdaptiveSharingGate
from mat_models.moe.Stage2MoEModel_v18 import TaskAwareTopKRouter


# ---------------------------------------------------------------------------
# FreshExpertSpec – all hyperparams explicit (no ckpt, no yaml inference)
# ---------------------------------------------------------------------------

@dataclass
class FreshExpertSpec:
    name: str
    mode: str                          # "composition" | "structure"
    moe_dim: int = 512                 # projected MoE dim

    # ---- composition branch ----
    comp_dim: int = 0                  # input feature size (inferred from data)
    comp_hidden_dims: Tuple[int, ...] = (512, 512, 256)
    comp_embed_dim: int = 512          # encoder output dim

    # ---- structure branch ----
    atom_dim: int = 92                 # inferred from data
    edge_dim: int = 41                 # inferred from data
    struct_embed_dim: int = 512        # encoder output dim
    node_dim: int = 128
    conv_layers: int = 3
    graphormer_layers: int = 2
    num_heads: int = 4
    ff_hidden: int = 256
    dropout: float = 0.1

    def embed_dim(self) -> int:
        return self.comp_embed_dim if self.mode == "composition" else self.struct_embed_dim

    def to_dict(self) -> dict:
        d = asdict(self)
        d["comp_hidden_dims"] = list(d["comp_hidden_dims"])
        return d

    @staticmethod
    def from_dict(d: dict) -> "FreshExpertSpec":
        d = dict(d)
        d["comp_hidden_dims"] = tuple(d["comp_hidden_dims"])
        return FreshExpertSpec(**d)


# ---------------------------------------------------------------------------
# Dimension probing helpers
# ---------------------------------------------------------------------------

def probe_comp_dim(npz_path: str) -> int:
    """Return number of composition features from a .npz file."""
    data = np.load(npz_path)
    return int(data["X"].shape[1])


def probe_struct_dims(pt_path: str) -> Tuple[int, int]:
    """Return (atom_dim, edge_dim) by inspecting one sample from a .pt file."""
    raw = torch.load(pt_path, map_location="cpu")
    # raw is a list of ((atom, nbr, idx), target, ...)
    sample = raw[0]
    atom, nbr, idx = sample[0]          # atom: (N, atom_dim), nbr: (N, M, edge_dim)
    return int(atom.shape[1]), int(nbr.shape[2])


# ---------------------------------------------------------------------------
# TrainableExpert – fully trainable encoder, no checkpoint loading
# ---------------------------------------------------------------------------

class TrainableExpert(nn.Module):
    """
    Expert module with freshly-initialised encoder. No Stage-1 weights are loaded
    and no parameters are frozen; the full encoder participates in Stage-2 training.
    """

    def __init__(self, spec: FreshExpertSpec, device: torch.device):
        super().__init__()
        self.spec = spec
        self.device = device

        if spec.mode == "composition":
            self.encoder = CompositionExpertV17(
                comp_dim=spec.comp_dim,
                embed_dim=spec.comp_embed_dim,
                hidden_dims=spec.comp_hidden_dims,
                dropout=spec.dropout,
            )
            expert_out_dim = spec.comp_embed_dim

        else:  # structure
            self.encoder = StructureExpertV17(
                atom_dim=spec.atom_dim,
                edge_dim=spec.edge_dim,
                embed_dim=spec.struct_embed_dim,
                node_dim=spec.node_dim,
                conv_layers=spec.conv_layers,
                graphormer_layers=spec.graphormer_layers,
                num_heads=spec.num_heads,
                ff_hidden=spec.ff_hidden,
                dropout=spec.dropout,
            )
            expert_out_dim = spec.struct_embed_dim

        # Project to shared MoE dim (identity if already matching)
        if expert_out_dim != spec.moe_dim:
            self.adapter = nn.Linear(expert_out_dim, spec.moe_dim)
        else:
            self.adapter = nn.Identity()

        print(
            f"[JointScratch] Init expert '{spec.name}' | mode={spec.mode} "
            f"| embed={expert_out_dim} -> moe={spec.moe_dim} | ALL weights trainable",
            flush=True,
        )

    def encode(self, batch_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward through encoder + adapter.  NO torch.no_grad() – fully trainable."""
        if self.spec.mode == "composition":
            emb = self.encoder(batch_dict["composition_vec"])
        else:
            emb = self.encoder.forward_batch(
                batch_dict["x"],
                batch_dict["edge_index"],
                batch_dict["edge_attr"],
                batch_dict["batch"],
            )
        return self.adapter(emb)


# ---------------------------------------------------------------------------
# JointAGMoEModel
# ---------------------------------------------------------------------------

class JointAGMoEModel(nn.Module):
    """
    Fully joint-from-scratch AG-MoE.

    Forward-pass logic is byte-for-byte identical to Stage2MoEModelV25; the
    only architectural difference is that every encoder weight is trainable.

    Composition routing policy  : one-hot to own expert (same as Stage-2).
    Structure routing policy    : TaskAwareTopKRouter + AdaptiveSharingGate.
    """

    def __init__(
        self,
        experts: List[FreshExpertSpec],
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

        # ---- experts ----
        self.expert_specs = experts
        self.experts = nn.ModuleList(
            [TrainableExpert(s, self.device) for s in experts]
        ).to(self.device)
        self.num_experts = len(self.experts)

        # ---- task indexing ----
        self.task_names = [s.name for s in experts]
        self.task2id = {t: i for i, t in enumerate(self.task_names)}
        self.name2expert_idx = {s.name: i for i, s in enumerate(experts)}
        self.struct_expert_indices = [
            i for i, s in enumerate(experts) if s.mode == "structure"
        ]

        # ---- task heads ----
        self.task_heads = nn.ModuleDict(
            {s.name: nn.Linear(moe_dim, 1) for s in experts}
        )

        # ---- router (structure tasks only) ----
        self.struct_router = TaskAwareTopKRouter(
            moe_dim=moe_dim,
            num_tasks=len(self.task_names),
            num_experts=max(1, len(self.struct_expert_indices)),
            hidden=router_hidden,
            dropout=router_dropout,
            k=min(top_k, max(1, len(self.struct_expert_indices))),
        )

        # ---- adaptive sharing gate ----
        self.share_gate = AdaptiveSharingGate(
            moe_dim=moe_dim,
            num_tasks=len(self.task_names),
            hidden=gate_hidden,
        )

        # monitoring buffers (compatible with training-script log code)
        self._last_router_w: Optional[torch.Tensor] = None
        self._last_router_q: Optional[torch.Tensor] = None
        self._last_gate_g: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def _infer_batch_size(self, batch_dict: Dict[str, torch.Tensor]) -> int:
        if "composition_vec" in batch_dict:
            return int(batch_dict["composition_vec"].size(0))
        if "batch" in batch_dict:
            return int(batch_dict["batch"].max().item() + 1)
        raise RuntimeError("[JointAGMoE] Cannot infer batch size from batch_dict")

    # ------------------------------------------------------------------
    def forward(self, task_name: str, batch_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert task_name in self.task2id, f"[JointAGMoE] Unknown task: {task_name}"

        B = self._infer_batch_size(batch_dict)
        task_id = self.task2id[task_name]
        idx = self.name2expert_idx[task_name]
        spec = self.expert_specs[idx]

        # ============================================================
        # Composition branch  (one-hot: own expert only, no cross-task
        # sharing – mirrors Stage-2 policy exactly)
        # ============================================================
        if spec.mode == "composition":
            h = self.experts[idx].encode(batch_dict)   # fully trainable

            self._last_router_q = h.detach()
            w_full = torch.zeros(B, self.num_experts, device=h.device)
            w_full[:, idx] = 1.0
            self._last_router_w = w_full.detach()
            self._last_gate_g = None

            return self.task_heads[task_name](h)

        # ============================================================
        # Structure branch  (TaskAwareTopKRouter + AdaptiveSharingGate)
        # ============================================================
        struct_ids = self.struct_expert_indices
        z_list = [self.experts[ei].encode(batch_dict) for ei in struct_ids]
        z = torch.stack(z_list, dim=1)          # (B, E_struct, D)

        # router
        q = z.mean(dim=1)
        self._last_router_q = q.detach()

        task_id_t = torch.full((B,), task_id, device=q.device, dtype=torch.long)
        w_struct = self.struct_router(q, task_id_t)     # (B, E_struct)

        w_full = torch.zeros(B, self.num_experts, device=q.device)
        w_full[:, struct_ids] = w_struct
        self._last_router_w = w_full.detach()

        h_moe = (z * w_struct.unsqueeze(-1)).sum(dim=1)

        # self-expert residual
        self_pos = struct_ids.index(idx)
        h_self = z[:, self_pos]

        # adaptive gate
        g = self.share_gate(q, task_id_t)       # (B, 1)
        self._last_gate_g = g.detach()

        h = (1.0 - g) * h_self + g * h_moe
        return self.task_heads[task_name](h)
