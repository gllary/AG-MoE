import torch.nn as nn
from .encoders.composition_encoder import CompositionEncoder
from .encoders.graph_encoder import GraphEncoder


class UnifiedMaterialModel(nn.Module):
    """
    Unified model supporting all 13 Matbench tasks.
    Automatically dispatches composition or structure encoder.
    """

    def __init__(
        self,
        comp_dim: int | None,
        atom_dim: int | None,
        edge_dim: int | None,
        embed_dim: int = 256,
    ):
        super().__init__()

        # -------- Composition Encoder --------
        self.comp_encoder = (
            CompositionEncoder(comp_dim, embed_dim) if comp_dim is not None else None
        )

        # -------- Graph Encoder --------
        self.graph_encoder = (
            GraphEncoder(atom_dim, edge_dim, embed_dim) if atom_dim is not None else None
        )

        # -------- Task Heads (13 tasks total) --------
        # IMPORTANT: steels task kept as "steels_yield"
        self.task_heads = nn.ModuleDict({

            # Composition regression
            "steels_yield": nn.Linear(embed_dim, 1),
            "expt_gap": nn.Linear(embed_dim, 1),

            # Composition classification
            "glass": nn.Linear(embed_dim, 1),
            "expt_is_metal": nn.Linear(embed_dim, 1),

            # Structure regression
            "phonons": nn.Linear(embed_dim, 1),
            "mp_gap": nn.Linear(embed_dim, 1),
            "mp_e_form": nn.Linear(embed_dim, 1),
            "log_kvrh": nn.Linear(embed_dim, 1),
            "log_gvrh": nn.Linear(embed_dim, 1),
            "perovskites": nn.Linear(embed_dim, 1),
            "jdft2d": nn.Linear(embed_dim, 1),
            "dielectric": nn.Linear(embed_dim, 1),

            # Structure classification
            "mp_is_metal": nn.Linear(embed_dim, 1),
        })


    def forward(self, inputs, task_name):
        """
        Unified forward for 13 tasks.
        Automatically selects correct encoder and correct task head.
        """

        if task_name not in self.task_heads:
            raise ValueError(
                f"Task '{task_name}' not implemented. Available: {list(self.task_heads.keys())}"
            )

        # -------- Task groups --------

        composition_tasks = {
            "steels_yield",
            "glass",
            "expt_is_metal",
            "expt_gap",
        }

        structure_tasks = {
            "phonons",
            "mp_gap",
            "mp_e_form",
            "mp_is_metal",
            "log_kvrh",
            "log_gvrh",
            "perovskites",
            "jdft2d",
            "dielectric",
        }

        # -------- Composition task --------
        if task_name in composition_tasks:

            if self.comp_encoder is None:
                raise ValueError(f"Task {task_name} requires comp_encoder but comp_dim=None")

            if "composition_vec" not in inputs:
                raise ValueError(f"Task {task_name} requires composition_vec in inputs")

            x = inputs["composition_vec"]
            emb = self.comp_encoder(x)

        # -------- Structure task --------
        elif task_name in structure_tasks:

            if self.graph_encoder is None:
                raise ValueError(f"Task {task_name} requires graph_encoder but atom_dim=None")

            required = {"atom_fea", "nbr_fea", "nbr_idx"}
            if not required.issubset(inputs.keys()):
                raise ValueError(f"Task {task_name} missing graph inputs: {required}")

            emb = self.graph_encoder(
                inputs["atom_fea"], inputs["nbr_fea"], inputs["nbr_idx"]
            )

        else:
            raise ValueError(f"Unknown task {task_name}")

        # -------- Task head --------
        return self.task_heads[task_name](emb)