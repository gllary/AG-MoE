# mat_models/moe/Stage2MoEModel_v17.py
# v17 Stage2: Frozen experts + adapters + router + task heads
# - Experts reconstructed from Stage1 yaml, but STRUCTURE hyperparams are auto-inferred from ckpt shapes
# - Safe loading: only load keys that exist AND match shape (avoids size mismatch crash)
# - Stage2 trains only: adapters + router + task heads

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
    Stage1 checkpoint is from SingleExpertModelV17. We only want encoder.* weights.
    Strip leading 'encoder.' exactly once.
    """
    enc_sd = {}
    for k, v in stage1_sd.items():
        if k.startswith("encoder."):
            enc_sd[k[len("encoder."):]] = v
    return enc_sd


def _is_struct_old_by_keys(enc_sd: Dict[str, torch.Tensor]) -> bool:
    keys = list(enc_sd.keys())
    has_fc_full = any("convs." in k and "fc_full.weight" in k for k in keys)
    has_attn_qproj = any(".attn.heads." in k and ".q_proj.weight" in k for k in keys)
    # old encoder uses fc_full + ln1/ln2 + attn.q_proj
    return bool(has_fc_full or has_attn_qproj)


# -------------------------
# Infer STRUCTURE hyperparams from checkpoint shapes
# -------------------------

def _infer_struct_hparams_from_ckpt(enc_sd: Dict[str, torch.Tensor], is_old: bool) -> Dict[str, int]:
    """
    Infer (atom_dim, edge_dim, node_dim, conv_layers, graphormer_layers, num_heads, ff_hidden, embed_dim)
    from checkpoint tensor shapes, so yaml mismatch won't crash.
    """
    # ---- node_dim + atom_dim ----
    # Prefer encoder.atom_embed.weight if exists
    atom_embed_key = None
    for k in ["encoder.atom_embed.weight", "atom_embed.weight"]:
        if k in enc_sd:
            atom_embed_key = k
            break
    if atom_embed_key is None:
        raise RuntimeError("[Stage2] Cannot find atom_embed.weight in structure enc_sd keys.")

    W = enc_sd[atom_embed_key]          # (node_dim, atom_dim)
    node_dim = int(W.shape[0])
    atom_dim = int(W.shape[1])

    # ---- conv_layers + edge_dim ----
    # old: encoder.convs.0.fc_full.weight: (2*node_dim, 2*node_dim + edge_dim)
    # new: encoder.convs.0.fc.weight:      (2*node_dim, 2*node_dim + edge_dim)
    conv_w_key = None
    if is_old:
        cand = ["encoder.convs.0.fc_full.weight", "convs.0.fc_full.weight"]
    else:
        cand = ["encoder.convs.0.fc.weight", "convs.0.fc.weight"]
    for k in cand:
        if k in enc_sd:
            conv_w_key = k
            break
    if conv_w_key is None:
        raise RuntimeError("[Stage2] Cannot find conv fc weight in structure enc_sd keys.")

    in_dim = int(enc_sd[conv_w_key].shape[1])
    edge_dim = int(in_dim - 2 * node_dim)

    # count conv layers by scanning indices
    conv_idx = set()
    for k in enc_sd.keys():
        if ".convs." in k:
            # e.g. encoder.convs.2.fc.weight
            try:
                part = k.split(".convs.")[1]
                i = int(part.split(".")[0])
                conv_idx.add(i)
            except Exception:
                pass
    conv_layers = int(max(conv_idx) + 1) if conv_idx else 3

    # ---- graphormer_layers + num_heads + ff_hidden ----
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

    num_heads = 0
    ff_hidden = 0
    if graphormer_layers > 0:
        if is_old:
            # old: encoder.graphormer.layers.0.attn.proj.weight: (node_dim, num_heads*node_dim)
            proj_key = None
            for k in ["encoder.graphormer.layers.0.attn.proj.weight", "graphormer.layers.0.attn.proj.weight"]:
                if k in enc_sd:
                    proj_key = k
                    break
            if proj_key is not None:
                proj_w = enc_sd[proj_key]
                # proj_w shape: (node_dim, num_heads*node_dim)
                num_heads = int(proj_w.shape[1] // node_dim)

            # old ff: encoder.graphormer.layers.0.ff.0.weight: (ff_hidden, node_dim)
            ff0_key = None
            for k in ["encoder.graphormer.layers.0.ff.0.weight", "graphormer.layers.0.ff.0.weight"]:
                if k in enc_sd:
                    ff0_key = k
                    break
            if ff0_key is not None:
                ff_hidden = int(enc_sd[ff0_key].shape[0])
        else:
            # final: encoder.graphormer.layers.0.proj.weight: (node_dim, num_heads*node_dim)
            proj_key = None
            for k in ["encoder.graphormer.layers.0.proj.weight", "graphormer.layers.0.proj.weight"]:
                if k in enc_sd:
                    proj_key = k
                    break
            if proj_key is not None:
                proj_w = enc_sd[proj_key]
                num_heads = int(proj_w.shape[1] // node_dim)

            # final ff: encoder.graphormer.layers.0.ff.0.weight: (ff_hidden, node_dim)
            ff0_key = None
            for k in ["encoder.graphormer.layers.0.ff.0.weight", "graphormer.layers.0.ff.0.weight"]:
                if k in enc_sd:
                    ff0_key = k
                    break
            if ff0_key is not None:
                ff_hidden = int(enc_sd[ff0_key].shape[0])

    # fallbacks
    if num_heads <= 0:
        num_heads = 4
    if ff_hidden <= 0:
        ff_hidden = 256

    # ---- embed_dim from head/fc_out output ----
    # old: encoder.fc_out.2.weight: (embed_dim, node_dim)
    # new: encoder.head.3.weight:   (embed_dim, node_dim) (because Dropout exists)
    embed_dim = 512
    for k in [
        "encoder.fc_out.2.weight", "fc_out.2.weight",
        "encoder.head.3.weight", "head.3.weight",
        "encoder.head.2.weight", "head.2.weight",
    ]:
        if k in enc_sd:
            embed_dim = int(enc_sd[k].shape[0])
            break

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
# Build encoder from cfg + ckpt hint
# -------------------------

def build_expert_encoder_from_stage1_cfg(
    stage1_cfg_path: str,
    mode: str,
    enc_sd_hint: Optional[Dict[str, torch.Tensor]] = None,
) -> nn.Module:
    cfg = _read_yaml(stage1_cfg_path)
    if "mode" in cfg:
        assert cfg["mode"] == mode, f"stage1_cfg mode {cfg['mode']} != task mode {mode}"

    if mode == "composition":
        embed_dim = int(cfg.get("embed_dim", 512))
        hidden_dims = tuple(cfg.get("comp_hidden_dims", [512, 512, 512]))
        dropout = float(cfg.get("comp_dropout", cfg.get("dropout", 0.1)))

        # comp_dim 永远以 ckpt 为准（最稳）
        # enc_sd_hint 里的 key 形如 "mlp.0.weight" 或 "encoder.mlp.0.weight"（取决于你 strip 的方式）
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

    # if mode == "composition":
    #     # --- infer comp_dim if missing ---
    #     embed_dim = int(cfg.get("embed_dim", 512))
    #     hidden_dims = tuple(cfg.get("comp_hidden_dims", [512, 512, 512]))
    #     dropout = float(cfg.get("comp_dropout", cfg.get("dropout", 0.1)))

    #     if "comp_dim" in cfg:
    #         comp_dim = int(cfg["comp_dim"])
    #     else:
    #         cand = None
    #         if enc_sd_hint is not None:
    #             for k, v in enc_sd_hint.items():
    #                 if k.endswith("mlp.0.weight"):
    #                     cand = int(v.shape[1])
    #                     break
    #         if cand is None:
    #             raise RuntimeError("[Stage2] composition comp_dim missing and cannot be inferred from ckpt.")
    #         comp_dim = cand

    #     try:
    #         enc = CompositionExpertV17(
    #             comp_dim=comp_dim,
    #             embed_dim=embed_dim,
    #             hidden_dims=hidden_dims,
    #             dropout=dropout,
    #         )
    #     except TypeError:
    #         enc = CompositionExpertV17(comp_dim=comp_dim, embed_dim=embed_dim, hidden_dims=hidden_dims)

    #     enc.embed_dim = embed_dim
    #     return enc

    # ---- structure: ALWAYS infer key dims from ckpt to avoid yaml mismatch ----
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
# Safe load helper (avoid size mismatch crash)
# -------------------------

def _filter_state_dict_by_shape(model: nn.Module, sd: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
    """
    Keep only keys that exist in model.state_dict AND have identical shape.
    Return: (filtered_sd, skipped_missing_or_unexpected_keys, skipped_shape_mismatch_keys)
    """
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
# FrozenExpert (Encoder only) + Adapter
# -------------------------

class FrozenExpert(nn.Module):
    """
    - reconstruct encoder (structure hyperparams inferred from ckpt)
    - safe-load encoder weights (match by key+shape)
    - freeze encoder
    - trainable adapter projects to moe_dim
    """

    def __init__(self, spec: ExpertSpec, device: torch.device):
        super().__init__()
        assert spec.mode in ("composition", "structure")
        self.spec = spec
        self.device = device

        raw = torch.load(spec.ckpt_path, map_location="cpu")
        sd = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
        enc_sd = _extract_encoder_sd(sd)

        # build encoder
        self.encoder = build_expert_encoder_from_stage1_cfg(
            spec.stage1_cfg,
            mode=spec.mode,
            enc_sd_hint=enc_sd,
        ).to(device)

        print(
            f"[Loading expert] {spec.name} | mode={spec.mode} | stage1_cfg={spec.stage1_cfg} | ckpt={spec.ckpt_path}",
            flush=True,
        )

        # ---- SAFE LOAD (key exists + shape match) ----
        filtered_sd, skipped_not_found, skipped_shape = _filter_state_dict_by_shape(self.encoder, enc_sd)
        missing, unexpected = self.encoder.load_state_dict(filtered_sd, strict=False)

        if missing:
            print(f"[Stage2][WARN] Missing keys for {spec.name}: {missing[:8]} ...", flush=True)
        if unexpected:
            print(f"[Stage2][WARN] Unexpected keys for {spec.name}: {unexpected[:8]} ...", flush=True)
        if skipped_shape:
            print(f"[Stage2][WARN] Shape-mismatch skipped for {spec.name}: {skipped_shape[:8]} ...", flush=True)

        # freeze encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        expert_dim = int(getattr(self.encoder, "embed_dim", spec.moe_dim))
        if expert_dim == spec.moe_dim:
            self.adapter = nn.Identity()
        else:
            self.adapter = nn.Linear(expert_dim, spec.moe_dim)

    @torch.no_grad()
    def _encode(self, batch_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.spec.mode == "composition":
            return self.encoder(batch_dict["composition_vec"])
        return self.encoder.forward_batch(
            batch_dict["x"],
            batch_dict["edge_index"],
            batch_dict["edge_attr"],
            batch_dict["batch"],
        )
    
    def forward(self, batch_dict: Dict[str, torch.Tensor], task_name: str) -> torch.Tensor:
        """
        Only encode when task_name matches this expert.
        Otherwise return zero embedding.
        """

        # ---------- 关键修复点 ----------
        # batch size 必须由“当前 task 的 batch_dict”决定
        if "composition_vec" in batch_dict:
            B = batch_dict["composition_vec"].size(0)
        elif "batch" in batch_dict:
            B = batch_dict["batch"].max().item() + 1
        else:
            raise RuntimeError("[Stage2] Cannot infer batch size from batch_dict")
        # --------------------------------

        # 非当前 task 的 expert：直接返回 0
        if task_name != self.spec.name:
            return torch.zeros(B, self.spec.moe_dim, device=self.device)

        # 当前 task 的 expert：正常 encode
        with torch.no_grad():
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
# Router (TopK)
# -------------------------
class TopKRouter(nn.Module):
    def __init__(self, moe_dim: int, num_experts: int, hidden: int = 256, dropout: float = 0.1, k: int = 2):
        super().__init__()
        self.k = k
        self.num_experts = num_experts
        self.net = nn.Sequential(
            nn.Linear(moe_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        q: (B, moe_dim)
        return weights: (B, E) sparse top-k softmax
        """
        logits = self.net(q)  # (B, E)

        # 保险：mask + softmax 用 fp32，避免 fp16 overflow
        logits_fp32 = logits.float()

        if self.k >= self.num_experts:
            w = torch.softmax(logits_fp32, dim=-1)
            # return w.to(logits.dtype)
            return w

        topk_idx = torch.topk(logits_fp32, k=self.k, dim=-1).indices  # (B, k)

        # fp32 的 -1e9 没问题
        mask = torch.full_like(logits_fp32, -1e9)
        mask.scatter_(dim=-1, index=topk_idx, value=0.0)

        w = torch.softmax(logits_fp32 + mask, dim=-1)
        # return w.to(logits.dtype)
        return w



# -------------------------
# Stage2 MoE
# -------------------------

class Stage2MoEModelV17(nn.Module):
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

        self.router = TopKRouter(moe_dim, self.num_experts, hidden=router_hidden, dropout=router_dropout, k=top_k)
        self.task_heads = nn.ModuleDict({s.name: nn.Linear(moe_dim, 1) for s in experts})

    def forward(self, task_name: str, batch_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        embs = [exp(batch_dict, task_name) for exp in self.experts]  # list[(B,D)]
        z = torch.stack(embs, dim=1)                      # (B, E, D)

        q = z.mean(dim=1)                                 # (B, D)
        self._last_router_q = q.detach()   # ✅ 新增这一行
        w_raw = self.router(q)                 # (B, E)
        self._last_router_w = w_raw.detach()   # ✅ 保存一份
        w = w_raw.unsqueeze(-1)                # (B, E, 1)
        h = (z * w).sum(dim=1)    
        
        assert z.dim() == 3 and z.size(1) == self.num_experts                        # (B, D)

        return self.task_heads[task_name](h)              # (B, 1)