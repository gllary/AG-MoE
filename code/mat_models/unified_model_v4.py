import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders.composition_encoder import CompositionEncoder
from .encoders.advanced_composition_encoder import AdvancedCompositionEncoder
from .encoders.schnet_encoder import SchNetEncoder
from .encoders.graph_transformer_encoder import GraphTransformerEncoder
from .encoders.task_adapter import TaskAdapter


class UnifiedMaterialModelV4(nn.Module):
    """
    统一材料模型 v4：
    支持：
    - 图结构 encoder: "schnet" / "graph_transformer"
    - 成分 encoder: "mlp" / "advanced"
    - Task Adapter 防止任务冲突
    """

    def __init__(
        self,
        comp_dim,
        atom_dim,
        edge_dim,
        embed_dim: int = 256,
        comp_encoder_type: str = "mlp",
        graph_encoder_type: str = "schnet",
    ):
        super().__init__()

        # -------- Composition Encoder --------
        if comp_dim is not None:
            if comp_encoder_type == "mlp":
                self.comp_encoder = CompositionEncoder(comp_dim, embed_dim)
            elif comp_encoder_type == "advanced":
                self.comp_encoder = AdvancedCompositionEncoder(comp_dim, embed_dim)
            else:
                raise ValueError(f"Unknown comp_encoder_type={comp_encoder_type}")
        else:
            self.comp_encoder = None

        # -------- Graph Encoder --------
        if atom_dim is not None and edge_dim is not None:
            if graph_encoder_type == "schnet":
                self.graph_encoder = SchNetEncoder(
                    atom_dim=atom_dim,
                    edge_dim=edge_dim,
                    embed_dim=embed_dim,
                    num_interactions=4,
                )
            elif graph_encoder_type == "graph_transformer":
                self.graph_encoder = GraphTransformerEncoder(
                    atom_dim=atom_dim,
                    edge_dim=edge_dim,
                    embed_dim=embed_dim,
                    num_layers=4,
                    num_heads=4,
                )
            else:
                raise ValueError(f"Unknown graph_encoder_type={graph_encoder_type}")
        else:
            self.graph_encoder = None

        self.comp_encoder_type = comp_encoder_type
        self.graph_encoder_type = graph_encoder_type

        # -------- 所有任务名称 --------
        self.comp_tasks = {"steels_yield", "expt_gap", "glass", "expt_is_metal"}
        self.struct_tasks = {
            "phonons",
            "mp_gap",
            "mp_e_form",
            "log_kvrh",
            "log_gvrh",
            "perovskites",
            "jdft2d",
            "dielectric",
            "mp_is_metal",
        }

        # -------- Task Heads (必须先定义) --------
        # （注意：TaskAdapter 会引用 task_heads，因此必须先定义 task_heads）
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

        # -------- Add Task Adapters (必须在 task_heads 之后定义) --------
        self.task_adapters = nn.ModuleDict({
            t: TaskAdapter(embed_dim, bottleneck=64)
            for t in self.task_heads.keys()
        })

    # ------------ forward ------------

    def forward(self, inputs, task_name: str):
        # ----- Composition tasks -----
        if task_name in self.comp_tasks:
            if "composition_vec" not in inputs:
                raise ValueError(f"Task {task_name} needs composition_vec")
            emb = self.comp_encoder(inputs["composition_vec"])

        # ----- Structure tasks -----
        elif task_name in self.struct_tasks:
            if not {"atom_fea", "nbr_fea", "nbr_idx"}.issubset(inputs.keys()):
                raise ValueError(f"Task {task_name} needs atom_fea, nbr_fea, nbr_idx")
            emb = self.graph_encoder(
                inputs["atom_fea"],
                inputs["nbr_fea"],
                inputs["nbr_idx"],
            )

        else:
            raise ValueError(f"Unknown task name: {task_name}")

        # ----- Apply task-specific adapter -----
        emb = self.task_adapters[task_name](emb)

        # ----- Final head -----
        return self.task_heads[task_name](emb)