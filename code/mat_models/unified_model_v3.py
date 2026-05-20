import torch
import torch.nn as nn

from .encoders.composition_encoder import CompositionEncoder
from .encoders.schnet_encoder import SchNetEncoder
from .heads import RegressionHead


class UnifiedMaterialModelV3(nn.Module):
    """
    Unified model with SchNet as structure encoder
    """

    def __init__(
        self,
        comp_dim,
        atom_dim,
        edge_dim,
        embed_dim=256,
        graph_type="schnet",
    ):
        super().__init__()

        # Composition encoder
        self.comp_encoder = (
            CompositionEncoder(comp_dim, embed_dim) if comp_dim is not None else None
        )

        # Graph encoder (SchNet)
        if graph_type == "schnet":
            self.graph_encoder = SchNetEncoder(
                atom_dim=atom_dim,
                edge_dim=edge_dim,
                embed_dim=embed_dim,
                num_interactions=4,  # larger = better
            )
        else:
            raise NotImplementedError

        # Task heads
        self.task_heads = nn.ModuleDict({
            "steels_yield": nn.Linear(embed_dim, 1),
            "expt_gap": nn.Linear(embed_dim, 1),

            "glass": nn.Linear(embed_dim, 1),
            "expt_is_metal": nn.Linear(embed_dim, 1),

            "phonons": nn.Linear(embed_dim, 1),
            "mp_gap": nn.Linear(embed_dim, 1),
            "mp_e_form": nn.Linear(embed_dim, 1),
            "log_kvrh": nn.Linear(embed_dim, 1),
            "log_gvrh": nn.Linear(embed_dim, 1),
            "perovskites": nn.Linear(embed_dim, 1),
            "jdft2d": nn.Linear(embed_dim, 1),
            "dielectric": nn.Linear(embed_dim, 1),

            "mp_is_metal": nn.Linear(embed_dim, 1),
        })

    def forward(self, inputs, task_name):
        if "composition_vec" in inputs:
            emb = self.comp_encoder(inputs["composition_vec"])
        else:
            emb = self.graph_encoder(
                inputs["atom_fea"],
                inputs["nbr_fea"],
                inputs["nbr_idx"],
            )
        return self.task_heads[task_name](emb)