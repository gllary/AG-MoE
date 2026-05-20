# -*- coding: utf-8 -*-
"""
unified_moe_v12.py

MoE v12:
- Composition side: stronger MLP experts with residual blocks
- Structure side: Graphormer-style CGCNN backbone experts (FCrystalBackbone)
- Task + input aware gating (per modality), Top-k routing
- Shared MoE backbone + per-task heads (regression / classification)

This file does NOT contain any training loop; see train/multitask_13tasks_moe_v12.py
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree


# ---------------------------------------------------------------------------
# 1. Graphormer + CGCNN BackBone (for structure experts)
# ---------------------------------------------------------------------------


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs (CGCNN style).
    """

    def __init__(self, atom_fea_len: int, nbr_fea_len: int):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len, 2 * self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        atom_in_fea: (N, atom_fea_len)
        nbr_fea:     (N, M, nbr_fea_len)
        nbr_fea_idx: (N, M)
        """
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]  # (N, M, atom_fea_len)
        total_nbr_fea = torch.cat(
            [
                atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
                atom_nbr_fea,
                nbr_fea,
            ],
            dim=2,
        )  # (N, M, 2*atom_fea_len + nbr_fea_len)

        total_gated_fea = self.fc_full(total_nbr_fea)  # (N, M, 2*atom_fea_len)
        total_gated_fea = self.bn1(
            total_gated_fea.view(-1, self.atom_fea_len * 2)
        ).view(N, M, self.atom_fea_len * 2)

        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)  # (N, atom_fea_len)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CentralityEncoding(nn.Module):
    """
    Centrality encoding from Graphormer.
    """

    def __init__(self, max_in_degree: int, max_out_degree: int, node_dim: int):
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.node_dim = node_dim

        self.z_in = nn.Parameter(torch.randn((max_in_degree, node_dim)))
        self.z_out = nn.Parameter(torch.randn((max_out_degree, node_dim)))

    @staticmethod
    def _clamp_deg(x: torch.Tensor, max_value: int) -> torch.Tensor:
        return torch.clamp(x, max=max_value)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """
        x: (N, node_dim)
        edge_index: (2, E)
        """
        num_nodes = x.size(0)
        in_deg = degree(index=edge_index[1], num_nodes=num_nodes).long()
        out_deg = degree(index=edge_index[0], num_nodes=num_nodes).long()
        in_deg = self._clamp_deg(in_deg, self.max_in_degree - 1)
        out_deg = self._clamp_deg(out_deg, self.max_out_degree - 1)

        return x + self.z_in[in_deg] + self.z_out[out_deg]


class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor, ptr=None) -> torch.Tensor:
        """
        x: (N, d)
        edge_index: (2, E)
        ptr: (B+1,) graph pointer, or None when single graph
        """
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        N = x.size(0)
        adjacency = torch.zeros(N, N, device=x.device)
        adjacency[edge_index[0], edge_index[1]] = 1.0

        if ptr is None:
            att = q @ k.transpose(0, 1) / math.sqrt(q.size(-1))
        else:
            att = torch.zeros((N, N), device=x.device)
            for i in range(len(ptr) - 1):
                s, e = ptr[i].item(), ptr[i + 1].item()
                att[s:e, s:e] = q[s:e] @ k[s:e].transpose(0, 1) / math.sqrt(q.size(-1))

        att = att * adjacency + (1 - adjacency) * (-1e6)
        att = torch.softmax(att, dim=-1)
        out = att @ v
        return out


class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [GraphormerAttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, x, edge_index, ptr):
        head_outs = [h(x, edge_index, ptr) for h in self.heads]
        concat = torch.cat(head_outs, dim=-1)
        return self.linear(concat)


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim: int, num_heads: int, max_path_distance: int = 5):
        super().__init__()
        self.attn = GraphormerMultiHeadAttention(
            num_heads=num_heads,
            dim_in=node_dim,
            dim_q=node_dim,
            dim_k=node_dim,
        )
        self.ln1 = nn.LayerNorm(node_dim)
        self.ln2 = nn.LayerNorm(node_dim)
        self.ff = nn.Sequential(
            nn.Linear(node_dim, node_dim * 4),
            nn.ReLU(),
            nn.Linear(node_dim * 4, node_dim),
        )

    def forward(self, x, edge_index, ptr):
        x_prime = self.attn(self.ln1(x), edge_index, ptr) + x
        x_new = self.ff(self.ln2(x_prime)) + x_prime
        return x_new


class GraphormerEncoder(nn.Module):
    def __init__(self, layers: int, node_dim: int, num_heads: int, max_path_distance: int = 5):
        super().__init__()
        self.layers = nn.ModuleList(
            [GraphormerEncoderLayer(node_dim, num_heads, max_path_distance) for _ in range(layers)]
        )

    def forward(self, x, edge_index, ptr):
        for layer in self.layers:
            x = layer(x, edge_index, ptr)
        return x


class FCrystalBackbone(nn.Module):
    """
    Graphormer + CGCNN backbone that outputs crystal-level embeddings
    (no final regression / classification head).
    """

    def __init__(
        self,
        orig_atom_fea_len: int,
        nbr_fea_len: int,
        atom_fea_len: int = 64,
        n_conv: int = 3,
        h_fea_len: int = 128,
        graphormer_layers: int = 1,
        num_heads: int = 4,
        max_path_distance: int = 5,
        out_embed_dim: int = 256,
    ):
        super().__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList(
            [ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len) for _ in range(n_conv)]
        )

        self.centrality_encoding = CentralityEncoding(
            max_in_degree=10, max_out_degree=10, node_dim=atom_fea_len
        )
        self.graphormer_encoder = GraphormerEncoder(
            layers=graphormer_layers,
            node_dim=atom_fea_len,
            num_heads=num_heads,
            max_path_distance=max_path_distance,
        )

        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.act = nn.Softplus()
        self.out_proj = nn.Linear(h_fea_len, out_embed_dim)

    def _pooling_single_crystal(self, atom_fea: torch.Tensor) -> torch.Tensor:
        """
        atom_fea: (N, atom_fea_len), single crystal in current usage.
        """
        return atom_fea.mean(dim=0, keepdim=True)  # (1, atom_fea_len)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx) -> torch.Tensor:
        """
        atom_fea: (N, atom_dim)
        nbr_fea:  (N, M, edge_dim)
        nbr_fea_idx: (N, M)
        Returns:
            crystal embedding: (1, out_embed_dim)
        """
        device = atom_fea.device
        atom_fea = self.embedding(atom_fea)
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)

        # build edge_index (2, N*M)
        N, M = nbr_fea_idx.shape
        src = torch.repeat_interleave(torch.arange(N, device=device), M)
        dst = nbr_fea_idx.view(-1)
        edge_index = torch.stack([src, dst], dim=0)

        # centrality + graphormer encoder
        atom_fea = self.centrality_encoding(atom_fea, edge_index)

        # ptr for single graph
        ptr = torch.tensor([0, N], dtype=torch.long, device=device)

        x = self.graphormer_encoder(atom_fea, edge_index, ptr)

        crys_fea = self._pooling_single_crystal(x)  # (1, atom_fea_len)
        crys_fea = self.act(self.conv_to_fc(crys_fea))
        crys_fea = self.act(crys_fea)
        out = self.out_proj(crys_fea)  # (1, out_embed_dim)
        return out


# ---------------------------------------------------------------------------
# 2. Composition-side Expert (strong tabular encoder)
# ---------------------------------------------------------------------------


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * hidden_mult)
        self.fc2 = nn.Linear(dim * hidden_mult, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h


class CompositionExpertV12(nn.Module):
    """
    Stronger MLP-based expert for composition vectors.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_blocks: int = 4,
        hidden_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(embed_dim, hidden_mult=hidden_mult, dropout=dropout) for _ in range(num_blocks)]
        )
        self.out_ln = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, input_dim)
        return: (B, embed_dim)
        """
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.out_ln(h)


# ---------------------------------------------------------------------------
# 3. MoE Core with Task + Input aware Gating
# ---------------------------------------------------------------------------


class ModalityGate(nn.Module):
    """
    Modality-specific gating network:
    - input embedding (B, D)
    - task embedding  (B, D)
    -> gate logits (B, n_experts)
    """

    def __init__(self, embed_dim: int, n_experts: int, hidden_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim * 2)
        hidden_dim = embed_dim * hidden_mult
        self.fc1 = nn.Linear(embed_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_emb: torch.Tensor, task_emb: torch.Tensor) -> torch.Tensor:
        """
        x_emb: (B, D)
        task_emb: (B, D)
        returns: gate_logits (B, n_experts)
        """
        h = torch.cat([x_emb, task_emb], dim=-1)
        h = self.ln(h)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        logits = self.fc2(h)
        return logits


class MoEUnifiedModelV12(nn.Module):
    """
    Unified MoE model for 13 Matbench tasks.

    - MoE is separated by modality (composition / structure)
    - Task-aware + input-aware gating
    - Per-task prediction heads (regression / classification)
    """

    def __init__(
        self,
        comp_dim: int,
        atom_dim: int,
        edge_dim: int,
        embed_dim: int,
        task_names: List[str],
        task_modalities: Dict[str, str],
        task_types: Dict[str, str],
        n_comp_experts: int = 4,
        n_struct_experts: int = 3,
        topk: int = 2,
    ):
        super().__init__()

        self.comp_dim = comp_dim
        self.atom_dim = atom_dim
        self.edge_dim = edge_dim
        self.embed_dim = embed_dim
        self.n_comp_experts = n_comp_experts
        self.n_struct_experts = n_struct_experts
        self.topk = topk

        # ----- task meta -----
        self.task_names = list(task_names)
        self.task_to_id = {t: i for i, t in enumerate(self.task_names)}
        self.task_modalities = dict(task_modalities)
        self.task_types = dict(task_types)

        self.task_embedding = nn.Embedding(len(self.task_names), embed_dim)

        # ----- composition experts -----
        self.comp_experts = nn.ModuleList(
            [
                CompositionExpertV12(
                    input_dim=comp_dim,
                    embed_dim=embed_dim,
                    num_blocks=4,
                    hidden_mult=4,
                    dropout=0.1,
                )
                for _ in range(n_comp_experts)
            ]
        )
        # encoder used only for gating input
        self.comp_gate_encoder = CompositionExpertV12(
            input_dim=comp_dim,
            embed_dim=embed_dim,
            num_blocks=2,
            hidden_mult=2,
            dropout=0.1,
        )
        self.comp_gate = ModalityGate(embed_dim=embed_dim, n_experts=n_comp_experts)

        # ----- structure experts -----
        self.struct_experts = nn.ModuleList(
            [
                FCrystalBackbone(
                    orig_atom_fea_len=atom_dim,
                    nbr_fea_len=edge_dim,
                    atom_fea_len=64,
                    n_conv=3,
                    h_fea_len=128,
                    graphormer_layers=1,
                    num_heads=4,
                    max_path_distance=5,
                    out_embed_dim=embed_dim,
                )
                for _ in range(n_struct_experts)
            ]
        )
        # gate encoder for structure: simple mean-pooling + linear
        self.struct_gate_encoder = nn.Sequential(
            nn.Linear(atom_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.struct_gate = ModalityGate(embed_dim=embed_dim, n_experts=n_struct_experts)

        # ----- per-task prediction heads -----
        heads = {}
        for t in self.task_names:
            heads[t] = nn.Linear(embed_dim, 1)
        self.task_heads = nn.ModuleDict(heads)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _get_task_id(self, task_name: str) -> int:
        if task_name not in self.task_to_id:
            raise KeyError(f"Unknown task name: {task_name}")
        return self.task_to_id[task_name]

    def _topk_softmax(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """
        logits: (B, n_experts)
        returns weights: (B, n_experts) with only top-k non-zero
        """
        if k >= logits.size(-1):
            return torch.softmax(logits, dim=-1)

        topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)
        mask = torch.zeros_like(logits).scatter_(-1, topk_idx, 1.0)
        masked_logits = logits.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(masked_logits, dim=-1)
        return weights

    # ------------------------------------------------------------------
    # forward by modality
    # ------------------------------------------------------------------
    def _forward_comp(self, x: torch.Tensor, task_name: str) -> torch.Tensor:
        """
        x: (B, comp_dim)
        return pooled embedding: (B, embed_dim)
        """
        B = x.size(0)
        task_id = self._get_task_id(task_name)
        t_emb = self.task_embedding.weight[task_id].unsqueeze(0).expand(B, -1)

        # input embedding for gating
        gate_input = self.comp_gate_encoder(x)  # (B, D)
        gate_logits = self.comp_gate(gate_input, t_emb)
        gate_weights = self._topk_softmax(gate_logits, self.topk)  # (B, n_comp_experts)

        # run all experts
        expert_outputs = []
        for expert in self.comp_experts:
            e = expert(x)  # (B, D)
            expert_outputs.append(e.unsqueeze(1))  # (B, 1, D)
        expert_outputs = torch.cat(expert_outputs, dim=1)  # (B, n_comp_experts, D)

        gate_weights = gate_weights.unsqueeze(-1)  # (B, n_comp_experts, 1)
        emb = torch.sum(gate_weights * expert_outputs, dim=1)  # (B, D)
        return emb

    def _forward_struct(self, atom_fea, nbr_fea, nbr_idx, task_name: str) -> torch.Tensor:
        """
        atom_fea: (N, atom_dim) single-graph
        nbr_fea:  (N, M, edge_dim)
        nbr_idx:  (N, M)
        return: (1, embed_dim)
        """
        N = atom_fea.size(0)
        device = atom_fea.device

        # gate input: mean-pooling over atoms
        mean_atom = atom_fea.mean(dim=0, keepdim=True)  # (1, atom_dim)
        gate_input = self.struct_gate_encoder(mean_atom)  # (1, D)

        task_id = self._get_task_id(task_name)
        t_emb = self.task_embedding.weight[task_id].unsqueeze(0)  # (1, D)

        gate_logits = self.struct_gate(gate_input, t_emb)  # (1, n_struct_experts)
        gate_weights = self._topk_softmax(gate_logits, self.topk)  # (1, n_struct_experts)

        expert_outputs = []
        for expert in self.struct_experts:
            e = expert(atom_fea, nbr_fea, nbr_idx)  # (1, D)
            expert_outputs.append(e.unsqueeze(1))  # (1, 1, D)
        expert_outputs = torch.cat(expert_outputs, dim=1)  # (1, n_struct_experts, D)

        gate_weights = gate_weights.unsqueeze(-1)  # (1, n_struct_experts, 1)
        emb = torch.sum(gate_weights * expert_outputs, dim=1)  # (1, D)
        return emb

    # ------------------------------------------------------------------
    # unified forward
    # ------------------------------------------------------------------
    def forward(self, inputs: Dict[str, torch.Tensor], task_name: str) -> torch.Tensor:
        """
        inputs:
          - for composition: {"composition_vec": (B, comp_dim)}
          - for structure:   {"atom_fea": (N, atom_dim),
                              "nbr_fea": (N, M, edge_dim),
                              "nbr_idx": (N, M)}
        """
        modality = self.task_modalities[task_name]
        if modality == "composition":
            x = inputs["composition_vec"]
            emb = self._forward_comp(x, task_name)  # (B, D)
        elif modality == "structure":
            atom = inputs["atom_fea"]
            nbr = inputs["nbr_fea"]
            idx = inputs["nbr_idx"]
            emb = self._forward_struct(atom, nbr, idx, task_name)  # (1, D)
        else:
            raise ValueError(f"Unknown modality {modality} for task {task_name}")

        head = self.task_heads[task_name]
        out = head(emb)  # (B, 1) or (1, 1)
        return out.view(-1)