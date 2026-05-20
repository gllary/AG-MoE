# mat_models/encoders/graphormer_cgcnn.py

from __future__ import print_function, division
import torch
import torch.nn as nn
from torch_geometric.utils import degree


# ------------------- 原始 CGCNN 卷积层 -------------------

class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Parameters
        ----------
        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len,
                                 2 * self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        N: Total number of atoms in the batch
        M: Max number of neighbors
        """
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]  # (N, M, atom_fea_len)
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2
        )  # (N, M, 2*atom_fea_len + nbr_fea_len)

        total_gated_fea = self.fc_full(total_nbr_fea)  # (N, M, 2*atom_fea_len)
        total_gated_fea = self.bn1(
            total_gated_fea.view(-1, self.atom_fea_len * 2)
        ).view(N, M, self.atom_fea_len * 2)

        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)  # Each (N,M,atom_fea_len)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)  # (N, atom_fea_len)
        nbr_sumed = self.bn2(nbr_sumed)
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
            self.max_in_degree - 1
        )
        out_degree = self.decrease_to_max_value(
            degree(index=edge_index[0], num_nodes=num_nodes).long(),
            self.max_out_degree - 1
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
            a = query.mm(key.transpose(0, 1)) / (query.size(-1) ** 0.5)
        else:
            a = torch.zeros((N, N), device=x.device)
            for i in range(len(ptr) - 1):
                s, e = ptr[i].item(), ptr[i + 1].item()
                a[s:e, s:e] = query[s:e].mm(key[s:e].transpose(0, 1)) / (query.size(-1) ** 0.5)

        # mask 非边
        a = a * adjacency + (1 - adjacency) * (-1e6)
        softmax = torch.softmax(a, dim=-1)
        out = softmax.mm(value)
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
    def __init__(self, node_dim, num_heads, max_path_distance: int = 5):
        super().__init__()
        self.attention = GraphormerMultiHeadAttention(
            num_heads=num_heads,
            dim_in=node_dim,
            dim_q=node_dim,
            dim_k=node_dim,
        )
        self.ln_1 = nn.LayerNorm(node_dim)
        self.ln_2 = nn.LayerNorm(node_dim)
        self.ff = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor, ptr) -> torch.Tensor:
        x_prime = self.attention(self.ln_1(x), edge_index, ptr) + x
        x_new = self.ff(self.ln_2(x_prime)) + x_prime
        return x_new


class GraphormerEncoder(nn.Module):
    def __init__(self, layers, node_dim, num_heads, max_path_distance: int = 5):
        super().__init__()
        self.layers = nn.ModuleList(
            [GraphormerEncoderLayer(node_dim, num_heads, max_path_distance)
             for _ in range(layers)]
        )

    def forward(self, x, edge_index, ptr):
        for layer in self.layers:
            x = layer(x, edge_index, ptr)
        return x


# ------------------- FCrystalGraphConvNet (带 Graphormer) -------------------

class FCrystalGraphConvNet(nn.Module):
    """
    CGCNN + Graphormer 一体化结构 encoder，最后输出晶体级别表征。
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
    ):
        super(FCrystalGraphConvNet, self).__init__()
        self.classification = classification

        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([
            ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
            for _ in range(n_conv)
        ])

        self.centrality_encoding = CentralityEncoding(
            max_in_degree=10,
            max_out_degree=10,
            node_dim=atom_fea_len
        )

        self.graphormer_encoder = GraphormerEncoder(
            layers=graphormer_layers,
            node_dim=atom_fea_len,
            num_heads=num_heads,
            max_path_distance=max_path_distance,
        )

        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()

        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.ReLU()
                                             for _ in range(n_h - 1)])
        else:
            self.fcs = None
            self.softpluses = None

        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        atom_fea: (N, orig_atom_fea_len)
        nbr_fea:  (N, M, nbr_fea_len)
        nbr_fea_idx: (N, M)
        crystal_atom_idx: list of idx tensors (每个晶体的原子索引)
        """
        atom_fea = self.embedding(atom_fea)
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)

        # 构造 edge_index
        N, M = nbr_fea_idx.size()
        src = torch.repeat_interleave(torch.arange(N, device=atom_fea.device), M)
        dst = nbr_fea_idx.view(-1)
        edge_index = torch.stack([src, dst], dim=0)

        # Centrality + Graphormer
        atom_fea = self.centrality_encoding(atom_fea, edge_index)

        ptr = [0]
        for idx_map in crystal_atom_idx:
            ptr.append(ptr[-1] + len(idx_map))
        ptr = torch.tensor(ptr, dtype=torch.long, device=atom_fea.device)

        x = self.graphormer_encoder(atom_fea, edge_index, ptr)

        crys_fea = self.pooling(x, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)

        if self.classification:
            crys_fea = self.dropout(crys_fea)

        if self.fcs is not None:
            for fc, act in zip(self.fcs, self.softpluses):
                crys_fea = act(fc(crys_fea))

        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out  # (batch_crystal, 1) for regression

    @staticmethod
    def pooling(atom_fea, crystal_atom_idx):
        """
        atom_fea: (N, atom_fea_len)
        crystal_atom_idx: list[Tensor]，每个 tensor 是该晶体的原子索引
        """
        assert sum(len(idx_map) for idx_map in crystal_atom_idx) == atom_fea.size(0)
        crys_fea = torch.stack(
            [atom_fea[idx_map].mean(dim=0) for idx_map in crystal_atom_idx], dim=0
        )
        return crys_fea


# ------------------- MoE 用的统一 Encoder 封装 -------------------

class GraphormerCGCNNEncoder(nn.Module):
    """
    封装 FCrystalGraphConvNet，为多任务 / MoE 提供统一接口：
    forward(atom_fea, nbr_fea, nbr_idx) -> (1, embed_dim)
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
    ):
        super().__init__()

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

        # backbone 输出 (1, 1)，但中间 conv_to_fc 输出是 (1, h_fea_len)
        # 简单做法：把 fc_out 删掉，或者我们这里“截取”倒数第二层。
        # 为简单起见，我们重用 backbone 的 pooling + conv_to_fc 部分：

        # 复用 backbone 逻辑：我们通过修改一点调用方式：
        # 这里直接用 backbone.forward，然后从中间层拿不到，
        # 所以更好的方式是用 pooling + conv_to_fc 重写一遍。
        # 为了不改 backbone，这里简单 hack：
        with torch.no_grad():
            # 复制 backbone 的 early stage：
            atom_fea2 = self.backbone.embedding(atom_fea)
            for conv in self.backbone.convs:
                atom_fea2 = conv(atom_fea2, nbr_fea, nbr_idx)

            N, M = nbr_idx.size()
            src = torch.repeat_interleave(torch.arange(N, device=atom_fea.device), M)
            dst = nbr_idx.view(-1)
            edge_index = torch.stack([src, dst], dim=0)

            atom_fea2 = self.backbone.centrality_encoding(atom_fea2, edge_index)

            ptr = torch.tensor([0, N], dtype=torch.long, device=atom_fea.device)
            x = self.backbone.graphormer_encoder(atom_fea2, edge_index, ptr)
            crys_fea = self.backbone.pooling(x, crystal_atom_idx)

        # 通过线性层映射到 embed_dim
        emb = self.proj(self.backbone.conv_to_fc_softplus(
            self.backbone.conv_to_fc(self.backbone.conv_to_fc_softplus(crys_fea))
        ))
        return emb  # (1, embed_dim)