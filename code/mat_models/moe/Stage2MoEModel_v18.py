# mat_models/moe/Stage2MoEModel_v18.py
# v18 Stage2: Frozen experts + adapters + Task-Aware Router
# Fixes:
# - robust key search supporting both "encoder.xxx" and "xxx"
# - Stage2 trusts ExpertSpec.mode, never asserts yaml["mode"]
# - composition tasks forced one-hot to their own expert
# - structure tasks route among structure experts only, task-aware top-k

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from mat_models.encoders.composition_expert_v17 import CompositionExpertV17
from mat_models.encoders.structure_expert_v17 import StructureExpertV17


# -------------------------
# ExpertSpec
# -------------------------

@dataclass
class ExpertSpec:
    name: str
    mode: str                 # "composition" | "structure"
    ckpt_path: str            # stage1 best ckpt
    stage1_cfg: str           # stage1 yaml path
    moe_dim: int = 512        # shared dim after adapter


# -------------------------
# Structure expert OLD wrapper
# -------------------------

class _StructureExpertV17Old(nn.Module):
    """
    Wrapper for graphormer_cgcnn_v17_old.py encoder.
    Exposes forward_batch(x, edge_index, edge_attr, batch) -> (B, D)
    """
    def __init__(
        self,
        atom_dim: int,
        edge_dim: int,
        embed_dim: int = 512,
        node_dim: int = 128,
        conv_layers: int = 3,
        graphormer_layers: int = 2,
        num_heads: int = 4,
        ff_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        from mat_models.encoders.graphormer_cgcnn_v17_old import GraphormerCGCNNEncoderV17 as OldEnc

        self.encoder = OldEnc(
            atom_dim=atom_dim,
            edge_dim=edge_dim,
            embed_dim=embed_dim,
            node_dim=node_dim,
            conv_layers=conv_layers,
            graphormer_layers=graphormer_layers,
            num_heads=num_heads,
            ff_hidden=ff_hidden,
            dropout=dropout,
        )
        self.embed_dim = embed_dim

    def forward_batch(self, x, edge_index, edge_attr, batch):
        return self.encoder.forward_batch(x, edge_index, edge_attr, batch)


# -------------------------
# YAML utils
# -------------------------

def _read_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _extract_encoder_sd(stage1_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Stage1 checkpoint is from SingleExpertModelV17.
    We keep BOTH forms:
      - raw "encoder.xxx"
      - stripped "xxx"
    This guarantees downstream inference is robust.
    """
    enc_sd = {}
    for k, v in stage1_sd.items():
        if k.startswith("encoder."):
            enc_sd[k] = v
            enc_sd[k[len("encoder."):]] = v
    return enc_sd


def _is_struct_old_by_keys(enc_sd: Dict[str, torch.Tensor]) -> bool:
    keys = list(enc_sd.keys())
    has_fc_full = any("convs." in k and "fc_full.weight" in k for k in keys)
    has_attn_qproj = any(".attn.heads." in k and ".q_proj.weight" in k for k in keys)
    return bool(has_fc_full or has_attn_qproj)


def _pick_first_key(enc_sd: Dict[str, torch.Tensor], candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in enc_sd:
            return k
    return None


# -------------------------
# Infer STRUCTURE hyperparams from checkpoint shapes (robust)
# -------------------------

def _infer_struct_hparams_from_ckpt(enc_sd: Dict[str, torch.Tensor], is_old: bool) -> Dict[str, int]:
    """
    Infer (atom_dim, edge_dim, node_dim, conv_layers, graphormer_layers, num_heads, ff_hidden, embed_dim)
    from checkpoint tensor shapes.
    Robust to naming differences and presence/absence of "encoder." prefix.
    """

    # ---- node_dim + atom_dim ----
    atom_embed_key = _pick_first_key(
        enc_sd,
        [
            "atom_embed.weight",
            "encoder.atom_embed.weight",
            "node_embed.weight",
            "encoder.node_embed.weight",
            "atom_embedding.weight",
            "encoder.atom_embedding.weight",
        ],
    )

    if atom_embed_key is None:
        keys_preview = list(enc_sd.keys())[:30]
        raise RuntimeError(
            "[Stage2] Cannot find atom/node embed weight in structure enc_sd keys. "
            f"Preview keys: {keys_preview}"
        )

    W = enc_sd[atom_embed_key]  # (node_dim, atom_dim)
    node_dim = int(W.shape[0])
    atom_dim = int(W.shape[1])

    # ---- conv fc weight for edge_dim ----
    if is_old:
        conv_w_key = _pick_first_key(
            enc_sd, ["convs.0.fc_full.weight", "encoder.convs.0.fc_full.weight"]
        )
    else:
        conv_w_key = _pick_first_key(
            enc_sd, ["convs.0.fc.weight", "encoder.convs.0.fc.weight"]
        )

    if conv_w_key is None:
        keys_preview = [k for k in enc_sd.keys() if "convs.0" in k][:50]
        raise RuntimeError(
            "[Stage2] Cannot find conv fc weight in structure enc_sd keys. "
            f"Preview conv keys: {keys_preview}"
        )

    in_dim = int(enc_sd[conv_w_key].shape[1])
    edge_dim = int(in_dim - 2 * node_dim)

    # count conv layers by scanning indices
    conv_idx = set()
    for k in enc_sd.keys():
        if ".convs." in k:
            try:
                part = k.split(".convs.")[1]
                i = int(part.split(".")[0])
                conv_idx.add(i)
            except Exception:
                pass
    conv_layers = int(max(conv_idx) + 1) if conv_idx else 3

    # ---- graphormer_layers ----
    layer_idx = set()
    for k in enc_sd.keys():
        if ".graphormer.layers." in k:
            try:
                part = k.split(".graphormer.layers.")[1]
                i = int(part.split(".")[0])
                layer_idx.add(i)
            except Exception:
                pass
    graphormer_layers = int(max(layer_idx) + 1) if layer_idx else 0

    # ---- num_heads + ff_hidden ----
    num_heads = 0
    ff_hidden = 0

    if graphormer_layers > 0:
        if is_old:
            proj_key = _pick_first_key(
                enc_sd,
                [
                    "graphormer.layers.0.attn.proj.weight",
                    "encoder.graphormer.layers.0.attn.proj.weight",
                ],
            )
            if proj_key is not None:
                proj_w = enc_sd[proj_key]
                num_heads = int(proj_w.shape[1] // node_dim)

            ff0_key = _pick_first_key(
                enc_sd,
                [
                    "graphormer.layers.0.ff.0.weight",
                    "encoder.graphormer.layers.0.ff.0.weight",
                ],
            )
            if ff0_key is not None:
                ff_hidden = int(enc_sd[ff0_key].shape[0])
        else:
            proj_key = _pick_first_key(
                enc_sd,
                [
                    "graphormer.layers.0.proj.weight",
                    "encoder.graphormer.layers.0.proj.weight",
                ],
            )
            if proj_key is not None:
                proj_w = enc_sd[proj_key]
                num_heads = int(proj_w.shape[1] // node_dim)

            ff0_key = _pick_first_key(
                enc_sd,
                [
                    "graphormer.layers.0.ff.0.weight",
                    "encoder.graphormer.layers.0.ff.0.weight",
                ],
            )
            if ff0_key is not None:
                ff_hidden = int(enc_sd[ff0_key].shape[0])

    if num_heads <= 0:
        num_heads = 4
    if ff_hidden <= 0:
        ff_hidden = 256

    # ---- embed_dim ----
    embed_key = _pick_first_key(
        enc_sd,
        [
            "fc_out.2.weight",
            "encoder.fc_out.2.weight",
            "head.3.weight",
            "encoder.head.3.weight",
            "head.2.weight",
            "encoder.head.2.weight",
        ],
    )
    embed_dim = int(enc_sd[embed_key].shape[0]) if embed_key is not None else 512

    return dict(
        atom_dim=atom_dim,
        edge_dim=edge_dim,
        node_dim=node_dim,
        conv_layers=conv_layers,
        graphormer_layers=graphormer_layers,
        num_heads=num_heads,
        ff_hidden=ff_hidden,
        embed_dim=embed_dim,
    )


# -------------------------
# Build encoder from stage1 cfg + ckpt hint
# IMPORTANT: Stage2 trusts spec.mode, not yaml mode.
# -------------------------

def build_expert_encoder_from_stage1_cfg(
    stage1_cfg_path: str,
    mode: str,
    enc_sd_hint: Optional[Dict[str, torch.Tensor]] = None,
) -> nn.Module:
    cfg = _read_yaml(stage1_cfg_path)

    if mode == "composition":
        embed_dim = int(cfg.get("embed_dim", 512))
        hidden_dims = tuple(cfg.get("comp_hidden_dims", [512, 512, 512]))
        dropout = float(cfg.get("comp_dropout", cfg.get("dropout", 0.1)))

        cand = None
        for k, v in (enc_sd_hint or {}).items():
            if k.endswith("mlp.0.weight"):
                cand = int(v.shape[1])
                break
        if cand is None:
            raise RuntimeError(f"[Stage2] Cannot infer comp_dim from ckpt for {stage1_cfg_path}")

        comp_dim = cand
        print(f"[Stage2][comp_dim] {stage1_cfg_path}: comp_dim={comp_dim}", flush=True)

        extra_kwargs = {}
        for k in ["use_magpie", "use_deep_encoder"]:
            if k in cfg:
                extra_kwargs[k] = cfg[k]

        try:
            enc = CompositionExpertV17(
                comp_dim=comp_dim,
                embed_dim=embed_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
                **extra_kwargs,
            )
        except TypeError:
            enc = CompositionExpertV17(
                comp_dim=comp_dim,
                embed_dim=embed_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )

        enc.embed_dim = embed_dim
        return enc

    # ---- structure ----
    if enc_sd_hint is None:
        raise RuntimeError("[Stage2] structure expert requires enc_sd_hint to infer hyperparams safely.")

    is_old = _is_struct_old_by_keys(enc_sd_hint)
    hp = _infer_struct_hparams_from_ckpt(enc_sd_hint, is_old=is_old)
    dropout = float(cfg.get("struct_dropout", cfg.get("dropout", 0.1)))

    if is_old:
        enc = _StructureExpertV17Old(
            atom_dim=hp["atom_dim"],
            edge_dim=hp["edge_dim"],
            embed_dim=hp["embed_dim"],
            node_dim=hp["node_dim"],
            conv_layers=hp["conv_layers"],
            graphormer_layers=hp["graphormer_layers"],
            num_heads=hp["num_heads"],
            ff_hidden=hp["ff_hidden"],
            dropout=dropout,
        )
        enc.embed_dim = hp["embed_dim"]
        return enc

    enc = StructureExpertV17(
        atom_dim=hp["atom_dim"],
        edge_dim=hp["edge_dim"],
        embed_dim=hp["embed_dim"],
        node_dim=hp["node_dim"],
        conv_layers=hp["conv_layers"],
        graphormer_layers=hp["graphormer_layers"],
        num_heads=hp["num_heads"],
        ff_hidden=hp["ff_hidden"],
        dropout=dropout,
    )
    enc.embed_dim = hp["embed_dim"]
    return enc


# -------------------------
# Safe load helper
# -------------------------

def _filter_state_dict_by_shape(
    model: nn.Module,
    sd: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
    msd = model.state_dict()
    filtered = {}
    skipped_not_found = []
    skipped_shape = []

    for k, v in sd.items():
        if k not in msd:
            skipped_not_found.append(k)
            continue
        if tuple(msd[k].shape) != tuple(v.shape):
            skipped_shape.append(k)
            continue
        filtered[k] = v

    return filtered, skipped_not_found, skipped_shape


# -------------------------
# FrozenExpert + Adapter
# -------------------------

class FrozenExpert(nn.Module):
    def __init__(self, spec: ExpertSpec, device: torch.device):
        super().__init__()
        assert spec.mode in ("composition", "structure")
        self.spec = spec
        self.device = device

        raw = torch.load(spec.ckpt_path, map_location="cpu")
        sd = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw

        # ✅ v18: keep both "encoder.xxx" and stripped "xxx" in enc_sd
        enc_sd = _extract_encoder_sd(sd)

        self.encoder = build_expert_encoder_from_stage1_cfg(
            spec.stage1_cfg,
            mode=spec.mode,
            enc_sd_hint=enc_sd,
        ).to(device)

        print(
            f"[Loading expert] {spec.name} | mode={spec.mode} | stage1_cfg={spec.stage1_cfg} | ckpt={spec.ckpt_path}",
            flush=True,
        )

        # safe-load: keys must exist in encoder.state_dict (encoder itself uses stripped keys)
        # so we filter by encoder keys only
        filtered_sd, _, skipped_shape = _filter_state_dict_by_shape(self.encoder, enc_sd)
        missing, unexpected = self.encoder.load_state_dict(filtered_sd, strict=False)

        if missing:
            print(f"[Stage2][WARN] Missing keys for {spec.name}: {missing[:8]} ...", flush=True)
        if unexpected:
            print(f"[Stage2][WARN] Unexpected keys for {spec.name}: {unexpected[:8]} ...", flush=True)
        if skipped_shape:
            print(f"[Stage2][WARN] Shape-mismatch skipped for {spec.name}: {skipped_shape[:8]} ...", flush=True)

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        expert_dim = int(getattr(self.encoder, "embed_dim", spec.moe_dim))
        self.embed_dim = expert_dim

        if expert_dim == spec.moe_dim:
            self.adapter = nn.Identity()
        else:
            self.adapter = nn.Linear(expert_dim, spec.moe_dim)

    def encode(self, batch_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
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


# -------------------------
# Task-Aware Router (structure only)
# -------------------------

class TaskAwareTopKRouter(nn.Module):
    def __init__(
        self,
        moe_dim: int,
        num_tasks: int,
        num_experts: int,
        hidden: int = 256,
        dropout: float = 0.1,
        k: int = 2,
    ):
        super().__init__()
        self.k = k
        self.num_experts = num_experts
        self.task_emb = nn.Embedding(num_tasks, moe_dim)

        self.net = nn.Sequential(
            nn.Linear(moe_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, q: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        tq = q + self.task_emb(task_id)
        logits = self.net(tq)

        logits_fp32 = logits.float()

        if self.k >= self.num_experts:
            return torch.softmax(logits_fp32, dim=-1)

        topk_idx = torch.topk(logits_fp32, k=self.k, dim=-1).indices
        mask = torch.full_like(logits_fp32, -1e9)
        mask.scatter_(dim=-1, index=topk_idx, value=0.0)
        return torch.softmax(logits_fp32 + mask, dim=-1)


# -------------------------
# Stage2 MoE v18
# -------------------------

class Stage2MoEModelV18(nn.Module):
    """
    v18 routing policy:
    - composition task: one-hot route to its own expert
    - structure task: route among structure experts only, task-aware top-k
    """
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
        self.moe_dim = moe_dim
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.expert_specs = experts
        self.experts = nn.ModuleList([FrozenExpert(s, self.device) for s in experts]).to(self.device)
        self.num_experts = len(self.experts)

        self.task_names = [s.name for s in experts]
        self.task2id = {t: i for i, t in enumerate(self.task_names)}
        self.num_tasks = len(self.task_names)

        self.name2expert_idx = {s.name: i for i, s in enumerate(experts)}
        self.struct_expert_indices = [i for i, s in enumerate(experts) if s.mode == "structure"]

        self.task_heads = nn.ModuleDict({s.name: nn.Linear(moe_dim, 1) for s in experts})

        self.struct_router = TaskAwareTopKRouter(
            moe_dim=moe_dim,
            num_tasks=self.num_tasks,
            num_experts=len(self.struct_expert_indices) if len(self.struct_expert_indices) > 0 else 1,
            hidden=router_hidden,
            dropout=router_dropout,
            k=min(top_k, max(1, len(self.struct_expert_indices))),
        )

        self._last_router_w = None
        self._last_router_q = None

    def _infer_batch_size(self, batch_dict: Dict[str, torch.Tensor]) -> int:
        if "composition_vec" in batch_dict:
            return int(batch_dict["composition_vec"].size(0))
        if "batch" in batch_dict:
            return int(batch_dict["batch"].max().item() + 1)
        raise RuntimeError("[Stage2] Cannot infer batch size from batch_dict")

    def forward(self, task_name: str, batch_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert task_name in self.task2id, f"Unknown task_name={task_name}"
        B = self._infer_batch_size(batch_dict)

        idx = self.name2expert_idx[task_name]
        spec_mode = self.expert_specs[idx].mode

        # composition: one-hot
        if spec_mode == "composition":
            with torch.no_grad():
                h = self.experts[idx].encode(batch_dict)
            self._last_router_q = h.detach()
            w_full = torch.zeros(B, self.num_experts, device=h.device, dtype=torch.float32)
            w_full[:, idx] = 1.0
            self._last_router_w = w_full.detach()
            return self.task_heads[task_name](h)

        # structure: route among structure experts only
        struct_ids = self.struct_expert_indices
        if len(struct_ids) == 0:
            raise RuntimeError("[Stage2] No structure experts found but got structure task.")

        z_list = []
        with torch.no_grad():
            for ei in struct_ids:
                z_list.append(self.experts[ei].encode(batch_dict))
        z = torch.stack(z_list, dim=1)  # (B, E_struct, D)

        q = z.mean(dim=1)
        self._last_router_q = q.detach()

        task_id = torch.full((B,), self.task2id[task_name], device=q.device, dtype=torch.long)
        w_struct = self.struct_router(q, task_id)  # (B, E_struct)

        w_full = torch.zeros(B, self.num_experts, device=q.device, dtype=torch.float32)
        w_full[:, struct_ids] = w_struct
        self._last_router_w = w_full.detach()

        h = (z * w_struct.unsqueeze(-1)).sum(dim=1)
        return self.task_heads[task_name](h)