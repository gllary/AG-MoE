import torch
import torch.nn as nn

from .your_code_import_here import (
    ConvLayer, CentralityEncoding,
    GraphormerEncoder, FCrystalGraphConvNet
)

class GraphormerCGCNNEncoder(nn.Module):
    """
    将你提供的 FCrystalGraphConvNet 封装成统一接口：
    输入:
        atom_fea: (N, atom_dim)
        nbr_fea: (N, M, edge_dim)
        nbr_idx: (N, M)
    输出:
        (1, embed_dim) 图表示
    """

    def __init__(
        self,
        orig_atom_fea_len,
        nbr_fea_len,
        embed_dim=128,
        conv_layers=3,
        graphormer_layers=1,
        node_dim=128,
        num_heads=4,
        h_fea_len=128,
        n_h=1,
    ):
        super().__init__()

        # 使用你给的模型框架（structure encoder）
        self.model = FCrystalGraphConvNet(
            orig_atom_fea_len=orig_atom_fea_len,
            nbr_fea_len=nbr_fea_len,
            atom_fea_len=node_dim,
            n_conv=conv_layers,
            graphormer_layers=graphormer_layers,
            num_heads=num_heads,
            h_fea_len=h_fea_len,
            n_h=n_h,
            classification=False
        )

        # 输出维度变成统一的 embed_dim
        self.fc_out = nn.Linear(h_fea_len, embed_dim)

    def forward(self, atom_fea, nbr_fea, nbr_idx):
        """
        输入来自 StructureDataset:
        atom_fea: (N, atom_dim)
        nbr_fea:  (N, M, edge_dim)
        nbr_idx:  (N, M)
        """

        # batch size = 1 (StructureDataset)
        # 构造 crystal_atom_idx
        N = atom_fea.shape[0]
        crystal_atom_idx = [torch.arange(N, device=atom_fea.device)]

        # 原模型 forward
        out = self.model(
            atom_fea,
            nbr_fea,
            nbr_idx,
            crystal_atom_idx
        )   # shape (1, 1)

        # 得到图 embedding（1, h_fea_len）
        if out.dim() == 2:
            graph_emb = out  # (1, h_fea_len)
        else:
            graph_emb = out.unsqueeze(0)

        # 映射到 unified embed_dim
        return self.fc_out(graph_emb)   # (1, embed_dim)