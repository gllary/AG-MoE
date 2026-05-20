# mat_models/moe/Stage2MoEModel_v17.py
# Stage2: Adapter-based MoE over frozen single-task experts (v17)
# - Auto-detects Stage1 expert architecture from checkpoint state_dict
# - Supports heterogeneous expert embedding dims (e.g., 256/512)
# - Each expert is frozen; trainable parts: adapters + router + per-task heads

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mat_models.encoders.composition_expert_v17 import CompositionExpertV17


# -------------------------
# Spec
# -------------------------

@dataclass
class ExpertSpec:
    name: str                 # task name
    mode: str                 # "composition" or "structure"
    ckpt_path: str            # path to stage1 checkpoint (.pt)
    expert_dim: int           # expert embedding dim (may be 256/512)
    moe_dim: int              # shared MoE dim (usually 512)


# -------------------------
# Utilities: infer from state_dict
# -------------------------

def _strip_prefix_if_needed(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    # e.g. sometimes saved with "module."
    if not any(k.startswith(prefix) for k in sd.keys()):
        return sd
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out


def _infer_embed_dim_from_sd(sd: Dict[str, torch.Tensor]) -> int:
    # head.weight: (1, embed_dim)
    if "head.weight" in sd:
        return int(sd["head.weight"].shape[1])
    # fallback: try common last linear names
    for k in ["encoder.encoder.fc_out.2.weight", "encoder.encoder.head.3.weight", "encoder.encoder.fc_out.3.weight"]:
        if k in sd and sd[k].ndim == 2:
            return int(sd[k].shape[0])
    raise RuntimeError("Cannot infer embed_dim from checkpoint (missing head.weight and known tails).")


def _infer_comp_arch_from_sd(sd: Dict[str, torch.Tensor]) -> Tuple[int, Tuple[int, ...], int]:
    """
    Infer:
      comp_dim, comp_hidden_dims, embed_dim
    by scanning encoder MLP linear weights.

    Supports keys like:
      encoder.mlp.0.weight
      encoder.net.0.weight
      encoder.layers.0.weight
    """
    linear_keys = []
    for k, v in sd.items():
        if not k.endswith(".weight"):
            continue
        if "encoder" not in k:
            continue
        if "head" in k:
            continue
        if v.ndim != 2:
            continue
        # composition MLP usually under encoder.mlp / encoder.net / encoder.layers
        if re.search(r"encoder\.(mlp|net|layers)\.\d+\.weight$", k):
            linear_keys.append(k)

    if len(linear_keys) == 0:
        # fallback: pick any encoder.*.weight 2D that looks like MLP chain
        for k, v in sd.items():
            if k.startswith("encoder.") and k.endswith(".weight") and v.ndim == 2 and "head" not in k:
                linear_keys.append(k)

    if len(linear_keys) == 0:
        raise RuntimeError("Cannot infer composition architecture: no encoder MLP linear weights found.")

    def layer_index(key: str) -> int:
        m = re.search(r"encoder\.(?:mlp|net|layers)\.(\d+)\.weight$", key)
        return int(m.group(1)) if m else 10**9

    linear_keys.sort(key=lambda x: (layer_index(x), x))

    first_w = sd[linear_keys[0]]
    comp_dim = int(first_w.shape[1])

    last_w = sd[linear_keys[-1]]
    embed_dim = int(last_w.shape[0])

    hidden_dims: List[int] = []
    for k in linear_keys[:-1]:
        w = sd[k]
        hidden_dims.append(int(w.shape[0]))

    # If hidden_dims empty, keep default-ish
    comp_hidden_dims = tuple(hidden_dims) if len(hidden_dims) > 0 else (512, 512, 512)
    return comp_dim, comp_hidden_dims, embed_dim


def _detect_struct_variant(sd: Dict[str, torch.Tensor]) -> str:
    # old has convs.*.fc_full + ln1/ln2
    if any("convs.0.fc_full.weight" in k for k in sd.keys()):
        return "old"
    # new has convs.*.fc + ln_msg/ln_agg
    if any("convs.0.fc.weight" in k for k in sd.keys()) or any("ln_msg" in k for k in sd.keys()):
        return "new"
    # also tolerate "encoder.encoder.convs.0.fc_full.weight" exactly
    if "encoder.encoder.convs.0.fc_full.weight" in sd:
        return "old"
    if "encoder.encoder.convs.0.fc.weight" in sd:
        return "new"
    raise RuntimeError("Cannot detect structure encoder variant (old/new) from checkpoint keys.")


def _infer_struct_arch_from_sd(sd: Dict[str, torch.Tensor], variant: str) -> Dict[str, int]:
    """
    Infer atom_dim, edge_dim, node_dim, conv_layers, graphormer_layers, num_heads, ff_hidden, embed_dim
    """
    embed_dim = _infer_embed_dim_from_sd(sd)

    # atom_embed is always Linear(atom_dim -> node_dim)
    atom_key = None
    for cand in [
        "encoder.encoder.atom_embed.weight",
        "encoder.atom_embed.weight",
    ]:
        if cand in sd:
            atom_key = cand
            break
    if atom_key is None:
        # fuzzy
        for k in sd.keys():
            if k.endswith("atom_embed.weight"):
                atom_key = k
                break
    if atom_key is None:
        raise RuntimeError("Cannot infer structure dims: atom_embed.weight not found.")

    atom_w = sd[atom_key]  # (node_dim, atom_dim)
    node_dim = int(atom_w.shape[0])
    atom_dim = int(atom_w.shape[1])

    # edge_dim from conv first layer weight
    if variant == "old":
        # fc_full: (2*node_dim, 2*node_dim + edge_dim)
        k_fc = None
        for k in sd.keys():
            if k.endswith("convs.0.fc_full.weight"):
                k_fc = k
                break
        if k_fc is None:
            raise RuntimeError("Cannot infer edge_dim for old variant: convs.0.fc_full.weight not found.")
        w = sd[k_fc]
        edge_dim = int(w.shape[1] - 2 * node_dim)
    else:
        # fc: (2*node_dim, 2*node_dim + edge_dim)
        k_fc = None
        for k in sd.keys():
            if k.endswith("convs.0.fc.weight"):
                k_fc = k
                break
        if k_fc is None:
            raise RuntimeError("Cannot infer edge_dim for new variant: convs.0.fc.weight not found.")
        w = sd[k_fc]
        edge_dim = int(w.shape[1] - 2 * node_dim)

    # conv_layers count
    conv_ids = set()
    for k in sd.keys():
        m = re.search(r"convs\.(\d+)\.", k)
        if m:
            conv_ids.add(int(m.group(1)))
    conv_layers = (max(conv_ids) + 1) if conv_ids else 3

    # graphormer layers count
    g_ids = set()
    for k in sd.keys():
        m = re.search(r"graphormer\.layers\.(\d+)\.", k)
        if m:
            g_ids.add(int(m.group(1)))
    graphormer_layers = (max(g_ids) + 1) if g_ids else 0

    # num_heads: depends on variant key pattern
    head_ids = set()
    if variant == "old":
        # encoder.encoder.graphormer.layers.0.attn.heads.<h>.q_proj.weight
        for k in sd.keys():
            m = re.search(r"attn\.heads\.(\d+)\.", k)
            if m:
                head_ids.add(int(m.group(1)))
    else:
        # encoder.encoder.graphormer.layers.0.heads.<h>.q.weight
        for k in sd.keys():
            m = re.search(r"graphormer\.layers\.\d+\.heads\.(\d+)\.", k)
            if m:
                head_ids.add(int(m.group(1)))
    num_heads = (max(head_ids) + 1) if head_ids else 0

    # ff_hidden from ff.0.weight: (ff_hidden, node_dim)
    ff_hidden = 256
    for k in sd.keys():
        if re.search(r"graphormer\.layers\.0\.ff\.0\.weight$", k):
            ff_hidden = int(sd[k].shape[0])
            break

    return dict(
        atom_dim=atom_dim,
        edge_dim=edge_dim,
        node_dim=node_dim,
        conv_layers=conv_layers,
        graphormer_layers=graphormer_layers,
        num_heads=num_heads if num_heads > 0 else 4,
        ff_hidden=ff_hidden,
        embed_dim=embed_dim,
    )


# -------------------------
# Frozen expert (auto-loader) + adapter
# -------------------------

class _ExpertCore(nn.Module):
    """
    Minimal expert wrapper that matches checkpoint exactly:
      - encoder: CompositionExpertV17 OR GraphormerCGCNNEncoderV17(_old/_new) inside StructureExpertV17 style
      - head: Linear(embed_dim -> 1)
    Provides:
      - encode(batch_dict) -> (B, embed_dim)
    """

    def __init__(self, mode: str, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.mode = mode
        self.encoder = encoder
        self.head = head

    @torch.no_grad()
    def encode(self, batch_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.mode == "composition":
            x = batch_dict["composition_vec"]
            return self.encoder(x)
        x = batch_dict["x"]
        edge_index = batch_dict["edge_index"]
        edge_attr = batch_dict["edge_attr"]
        batch = batch_dict["batch"]
        return self.encoder.forward_batch(x, edge_index, edge_attr, batch)


class FrozenExpert(nn.Module):
    """
    Load a Stage1 checkpoint (composition or structure), auto-detect its exact architecture,
    freeze it, and expose:
      emb_expert: (B, expert_dim)
      emb_moe:    (B, moe_dim) after adapter
    """

    def __init__(self, spec: ExpertSpec, device: torch.device):
        super().__init__()
        assert spec.mode in ("composition", "structure")
        self.spec = spec
        self.device = device

        raw = torch.load(spec.ckpt_path, map_location="cpu")
        sd = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
        sd = _strip_prefix_if_needed(sd, "module.")

        print(
            f"[Loading expert] {spec.name} | expert_dim={spec.expert_dim} | ckpt={spec.ckpt_path}",
            flush=True
        )

        if spec.mode == "composition":
            comp_dim, comp_hidden_dims, embed_dim = _infer_comp_arch_from_sd(sd)
            # build exact encoder
            encoder = CompositionExpertV17(
                comp_dim=comp_dim,
                embed_dim=embed_dim,
                hidden_dims=comp_hidden_dims,
                dropout=0.1,  # dropout doesn't affect weights loading; keep default
            )
            head = nn.Linear(embed_dim, 1)

            model = _ExpertCore(mode="composition", encoder=encoder, head=head).to(device)
            model.load_state_dict(sd, strict=True)

        else:
            variant = _detect_struct_variant(sd)
            arch = _infer_struct_arch_from_sd(sd, variant=variant)

            if variant == "old":
                from mat_models.encoders.graphormer_cgcnn_v17_old import GraphormerCGCNNEncoderV17 as Enc
            else:
                from mat_models.encoders.graphormer_cgcnn_v17 import GraphormerCGCNNEncoderV17 as Enc

            # build exact encoder
            encoder = Enc(
                atom_dim=arch["atom_dim"],
                edge_dim=arch["edge_dim"],
                embed_dim=arch["embed_dim"],
                node_dim=arch["node_dim"],
                conv_layers=arch["conv_layers"],
                graphormer_layers=arch["graphormer_layers"],
                num_heads=arch["num_heads"],
                ff_hidden=arch["ff_hidden"],
                dropout=0.1,
            )

            head = nn.Linear(arch["embed_dim"], 1)

            model = _ExpertCore(mode="structure", encoder=encoder, head=head).to(device)
            model.load_state_dict(sd, strict=True)

        # freeze
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self.model = model

        # adapter to unify dims
        # IMPORTANT: actual expert embedding dim is inferred from checkpoint head.weight
        inferred_dim = _infer_embed_dim_from_sd(sd)
        self.expert_dim = inferred_dim

        if self.expert_dim == spec.moe_dim:
            self.adapter = nn.Identity()
        else:
            self.adapter = nn.Linear(self.expert_dim, spec.moe_dim)

    def forward(self, batch_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            emb = self.model.encode(batch_dict)  # (B, expert_dim inferred)
        emb = self.adapter(emb)  # (B, moe_dim)
        return emb


# -------------------------
# Router
# -------------------------

class TopKRouter(nn.Module):
    def __init__(self, moe_dim: int, num_experts: int, k: int = 2, temperature: float = 1.0):
        super().__init__()
        self.k = k
        self.temperature = temperature
        self.scorer = nn.Linear(moe_dim, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.scorer(x) / max(1e-6, float(self.temperature))  # (B, E)

        if self.k >= logits.size(-1):
            w = torch.softmax(logits, dim=-1)
            topk = torch.topk(logits, k=logits.size(-1), dim=-1).indices
            return w, topk

        topk = torch.topk(logits, k=self.k, dim=-1).indices  # (B, k)
        mask = torch.full_like(logits, fill_value=-1e9)
        mask.scatter_(dim=-1, index=topk, value=0.0)
        w = torch.softmax(logits + mask, dim=-1)
        return w, topk


# -------------------------
# Stage2 MoE
# -------------------------

class Stage2MoEModelV17(nn.Module):
    """
    Multi-task MoE with frozen experts + trainable adapters + router + per-task heads.
    """

    def __init__(
        self,
        experts: List[ExpertSpec],
        moe_dim: int = 512,
        top_k: int = 2,
        router_temperature: float = 1.0,
        share_router: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.moe_dim = moe_dim
        self.top_k = top_k
        self.router_temperature = router_temperature
        self.share_router = share_router
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.expert_specs = experts
        self.experts = nn.ModuleList([FrozenExpert(s, self.device) for s in experts]).to(self.device)
        self.num_experts = len(self.experts)

        # Routers
        if share_router:
            self.router = TopKRouter(moe_dim, self.num_experts, k=top_k, temperature=router_temperature)
            self.task_routers = None
        else:
            self.router = None
            self.task_routers = nn.ModuleDict({
                s.name: TopKRouter(moe_dim, self.num_experts, k=top_k, temperature=router_temperature)
                for s in experts
            })

        # Per-task heads
        self.task_heads = nn.ModuleDict({s.name: nn.Linear(moe_dim, 1) for s in experts})

    def _get_router(self, task_name: str) -> TopKRouter:
        return self.router if self.share_router else self.task_routers[task_name]

    def forward(self, task_name: str, batch_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        # (B, E, D)
        z = torch.stack([exp(batch_dict) for exp in self.experts], dim=1)

        # routing query
        q = z.mean(dim=1)  # (B, D)

        w, _ = self._get_router(task_name)(q)  # (B, E)
        h = (z * w.unsqueeze(-1)).sum(dim=1)   # (B, D)

        return self.task_heads[task_name](h)   # (B, 1)