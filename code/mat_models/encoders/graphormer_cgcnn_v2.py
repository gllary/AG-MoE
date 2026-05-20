# mat_models/encoders/graphormer_cgcnn_v2.py

from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.utils import degree


# ------------------- 原始/升级 CGCNN 卷积层 -------------------


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    支持两种归一化：
      - norm_type = "batchnorm"  (默认，兼容旧实现)
      - norm_type = "layernorm"  (对小 batch 更稳)
    """

    def __init__(self, atom_fea_len, nbr_fea_len, norm_type: str = "batchnorm"):
        """
        Parameters
        ----------
        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        norm_type: "batchnorm" or "layernorm"
        """
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.norm_type = norm_type.lower()

        self.fc_full = nn.Linear(
            2 * self.atom_fea_len + self.nbr_fea_len,
            2 * self.atom_fea_len,
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.softplus2 = nn.Softplus()

        if self.norm_type == "batchnorm":
            self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
            self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
            self.use_batchnorm = True
            self.ln1 = None
            self.ln2 = None
        else:
            # 使用 LayerNorm 作用在最后一维
            self.ln1 = nn.LayerNorm(2 * self.atom_fea_len)
            self.ln2 = nn.LayerNorm(self.atom_fea_len)
            self.use_batchnorm = False
            self.bn1 = None
            self.bn2 = None

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        N: Total number of atoms in the batch
        M: Max number of neighbors
        """
        N, M = nbr_fea_idx.shape

        # 邻居特征
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

        if self.use_batchnorm:
            total_gated_fea = self.bn1(
                total_gated_fea.view(-1, self.atom_fea_len * 2)
            ).view(N, M, self.atom_fea_len * 2)
        else:
            total_gated_fea = self.ln1(total_gated_fea)

        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)  # (N, M, atom_fea_len)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)

        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)  # (N, atom_fea_len)

        if self.use_batchnorm:
            nbr_sumed = self.bn2(nbr_sumed)
        else:
            nbr_sumed = self.ln2(nbr_sumed)

        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


# ------------------- Graphormer 组件 -------------------


class CentralityEncoding(nn.Module):
    def __init__(self, max_in_degree: int, max_out_degree: int, node_dim: int):
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.node_dim = node_dim
        self.z_in = nn.Parameter(torch.randn((max_in_degree, node_dim)))
        self.z_out = nn.Parameter(torch.randn((max_out_degree, node_dim)))

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        num_nodes = x.shape[0]

        in_degree = self.decrease_to_max_value(
            degree(index=edge_index[1], num_nodes=num_nodes).long(),
            self.max_in_degree - 1,
        )
        out_degree = self.decrease_to_max_value(
            degree(index=edge_index[0], num_nodes=num_nodes).long(),
            self.max_out_degree - 1,
        )

        x = x + self.z_in[in_degree] + self.z_out[out_degree]
        return x

    @staticmethod
    def decrease_to_max_value(x, max_value):
        return torch.clamp(x, max=max_value)


class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor, ptr=None) -> torch.Tensor:
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        N = x.size(0)
        adjacency = torch.zeros(N, N, device=x.device)
        adjacency[edge_index[0], edge_index[1]] = 1.0

        if ptr is None:
            a = query @ key.t() / (query.size(-1) ** 0.5)
        else:
            a = torch.zeros((N, N), device=x.device)
            for i in range(len(ptr) - 1):
                s, e = ptr[i].item(), ptr[i + 1].item()
                q_i = query[s:e]
                k_i = key[s:e]
                a[s:e, s:e] = q_i @ k_i.t() / (query.size(-1) ** 0.5)

        # mask 非边
        a = a * adjacency + (1 - adjacency) * (-1e6)
        softmax = torch.softmax(a, dim=-1)
        out = softmax @ value
        return out


class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [GraphormerAttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor, ptr) -> torch.Tensor:
        outs = [head(x, edge_index, ptr) for head in self.heads]
        concatenated = torch.cat(outs, dim=-1)
        return self.linear(concatenated)


class GraphormerEncoderLayer(nn.Module):
    def __init__(
        self,
        node_dim: int,
        num_heads: int,
        max_path_distance: int = 5,
        dropout: float = 0.0,
        ffn_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.attention = GraphormerMultiHeadAttention(
            num_heads=num_heads,
            dim_in=node_dim,
            dim_q=node_dim,
            dim_k=node_dim,
        )
        self.ln_1 = nn.LayerNorm(node_dim)
        self.ln_2 = nn.LayerNorm(node_dim)

        hidden = ffn_hidden_dim or (node_dim * 4)
        self.ff = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, node_dim),
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor, ptr) -> torch.Tensor:
        # pre-norm + 残差
        attn_out = self.attention(self.ln_1(x), edge_index, ptr)
        x_prime = x + self.dropout1(attn_out)

        ff_out = self.ff(self.ln_2(x_prime))
        x_new = x_prime + self.dropout2(ff_out)
        return x_new


class GraphormerEncoder(nn.Module):
    def __init__(
        self,
        layers: int,
        node_dim: int,
        num_heads: int,
        max_path_distance: int = 5,
        dropout: float = 0.0,
        ffn_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GraphormerEncoderLayer(
                    node_dim=node_dim,
                    num_heads=num_heads,
                    max_path_distance=max_path_distance,
                    dropout=dropout,
                    ffn_hidden_dim=ffn_hidden_dim,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x, edge_index, ptr):
        for layer in self.layers:
            x = layer(x, edge_index, ptr)
        return x


# ------------------- FCrystalGraphConvNet (带 Graphormer) -------------------


class FCrystalGraphConvNet(nn.Module):
    """
    CGCNN + Graphormer 一体化结构 encoder，最后输出晶体级别表征。

    这里保留原 forward 接口：
        - 如果 classification=False，则 forward 返回 (batch_crystal, 1)
        - 我们额外提供 _encode_graph(...) 返回 (batch_crystal, h_fea_len)
          供 GraphormerCGCNNEncoder/结构专家使用。
    """

    def __init__(
        self,
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        n_h=1,
        classification=False,
        graphormer_layers=1,
        num_heads=4,
        max_path_distance=5,
        node_dim=128,
        dropout: float = 0.0,
        ffn_hidden_dim: int | None = None,
        norm_type: str = "batchnorm",
    ):
        super().__init__()
        self.classification = classification
        self.atom_fea_len = atom_fea_len

        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList(
            [
                ConvLayer(
                    atom_fea_len=atom_fea_len,
                    nbr_fea_len=nbr_fea_len,
                    norm_type=norm_type,
                )
                for _ in range(n_conv)
            ]
        )

        self.centrality_encoding = CentralityEncoding(
            max_in_degree=10,
            max_out_degree=10,
            node_dim=atom_fea_len,
        )

        self.graphormer_encoder = GraphormerEncoder(
            layers=graphormer_layers,
            node_dim=atom_fea_len,
            num_heads=num_heads,
            max_path_distance=max_path_distance,
            dropout=dropout,
            ffn_hidden_dim=ffn_hidden_dim,
        )

        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()

        if n_h > 1:
            self.fcs = nn.ModuleList(
                [nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)]
            )
            self.acts = nn.ModuleList([nn.ReLU() for _ in range(n_h - 1)])
        else:
            self.fcs = None
            self.acts = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)

    def _encode_graph(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        返回经过 CGCNN + Graphormer + FC/MLP 后的晶体 embedding:
            shape: (batch_crystal, h_fea_len)
        """
        atom_fea = self.embedding(atom_fea)

        # CGCNN 卷积
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)

        # 构造 edge_index
        N, M = nbr_fea_idx.size()
        device = atom_fea.device
        src = torch.repeat_interleave(torch.arange(N, device=device), M)
        dst = nbr_fea_idx.view(-1)
        edge_index = torch.stack([src, dst], dim=0)

        # Centrality + Graphormer
        atom_fea = self.centrality_encoding(atom_fea, edge_index)

        ptr = [0]
        for idx_map in crystal_atom_idx:
            ptr.append(ptr[-1] + len(idx_map))
        ptr = torch.tensor(ptr, dtype=torch.long, device=device)

        x = self.graphormer_encoder(atom_fea, edge_index, ptr)

        # pooling
        crys_fea = self.pooling(x, crystal_atom_idx)  # (n_crystal, atom_fea_len)

        # FC & MLP
        h = self.conv_to_fc_softplus(self.conv_to_fc(crys_fea))
        h = self.dropout(h)

        if self.fcs is not None:
            for fc, act in zip(self.fcs, self.acts):
                h = act(fc(h))

        return h  # (n_crystal, h_fea_len)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        h = self._encode_graph(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        out = self.fc_out(h)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    @staticmethod
    def pooling(atom_fea, crystal_atom_idx):
        """
        atom_fea: (N, atom_fea_len)
        crystal_atom_idx: list[Tensor]，每个 tensor 是该晶体的原子索引
        """
        assert sum(len(idx_map) for idx_map in crystal_atom_idx) == atom_fea.size(0)
        crys_fea = torch.stack(
            [atom_fea[idx_map].mean(dim=0) for idx_map in crystal_atom_idx],
            dim=0,
        )
        return crys_fea


# ------------------- MoE / v14 用统一 Encoder 封装 -------------------


class GraphormerCGCNNEncoder(nn.Module):
    """
    封装 FCrystalGraphConvNet，为单专家/多专家提供统一接口：
        forward(atom_fea, nbr_fea, nbr_idx) -> (1, embed_dim)

    可调超参：
        - n_conv: CGCNN 层数
        - graphormer_layers: Graphormer 层数
        - node_dim / atom_fea_len
        - num_heads
        - dropout
        - ffn_hidden_dim
        - norm_type: "batchnorm" / "layernorm"
        - pooling: 目前内部仍使用 mean pooling，"sum" 与 "mean" 等价，
                   "cls" 预留接口，暂未启用。
    """

    def __init__(
        self,
        orig_atom_fea_len: int,
        nbr_fea_len: int,
        embed_dim: int = 256,
        atom_fea_len: int = 64,
        n_conv: int = 3,
        h_fea_len: int = 128,
        n_h: int = 1,
        graphormer_layers: int = 1,
        num_heads: int = 4,
        max_path_distance: int = 5,
        dropout: float = 0.0,
        ffn_hidden_dim: int | None = None,
        norm_type: str = "layernorm",
        pooling: str = "mean",
    ):
        super().__init__()

        self.pooling = pooling.lower()
        if self.pooling not in {"mean", "sum", "cls"}:
            raise ValueError(f"Unsupported pooling: {pooling}")

        self.backbone = FCrystalGraphConvNet(
            orig_atom_fea_len=orig_atom_fea_len,
            nbr_fea_len=nbr_fea_len,
            atom_fea_len=atom_fea_len,
            n_conv=n_conv,
            h_fea_len=h_fea_len,
            n_h=n_h,
            classification=False,
            graphormer_layers=graphormer_layers,
            num_heads=num_heads,
            max_path_distance=max_path_distance,
            node_dim=atom_fea_len,
            dropout=dropout,
            ffn_hidden_dim=ffn_hidden_dim,
            norm_type=norm_type,
        )

        # 将 h_fea_len 投到统一的 embed_dim
        self.proj = nn.Linear(h_fea_len, embed_dim)

    def forward(self, atom_fea, nbr_fea, nbr_idx):
        """
        假定一个样本 = 一个晶体（StructureDataset 当前就是 batch_size=1）
        atom_fea: (N, atom_dim)
        nbr_fea:  (N, M, edge_dim)
        nbr_idx:  (N, M)
        """
        N = atom_fea.shape[0]
        crystal_atom_idx = [torch.arange(N, device=atom_fea.device)]

        # 使用 backbone 的 _encode_graph 得到 (1, h_fea_len)
        h = self.backbone._encode_graph(atom_fea, nbr_fea, nbr_idx, crystal_atom_idx)

        # pooling 参数目前只影响 future 扩展，这里 mean/sum 等价
        if self.pooling == "cls":
            # 这里预留 CLS pooling 接口，如需严格 CLS，我们再单独重构一版。
            raise NotImplementedError("CLS pooling is not implemented yet.")
        # "mean" / "sum" 对单晶体来说等价（只有一个 crystal）
        emb = self.proj(h)  # (1, embed_dim)
        return emb