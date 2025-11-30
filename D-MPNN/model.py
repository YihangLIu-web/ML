# model.py

import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph, global_mean_pool
from torch_geometric.utils import scatter  # PyG 自带的 scatter


class EGNNLayer(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        # 边特征: h_i, h_j, ||x_i - x_j||^2
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels + 1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
        )
        # 节点更新: concat(h_i, m_i)
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        # 坐标更新用的标量系数 g(m_ij)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, h, x, edge_index):
        """
        h: [N, H]
        x: [N, 3]
        edge_index: [2, E]
        """
        row, col = edge_index  # i = row, j = col

        # 相对位移 & 距离平方（保证旋转不变）
        diff = x[row] - x[col]  # [E, 3]
        dist2 = (diff ** 2).sum(-1, keepdim=True)  # [E, 1]

        # 边消息 m_ij
        edge_input = torch.cat([h[row], h[col], dist2], dim=-1)  # [E, 2H+1]
        m_ij = self.edge_mlp(edge_input)  # [E, H]

        # 坐标更新: x_i' = x_i + Σ_j g(m_ij) * (x_i - x_j)
        coord_coef = self.coord_mlp(m_ij)  # [E, 1]
        delta_x = diff * coord_coef  # [E, 3]
        x = x + scatter(delta_x, row, dim=0, dim_size=x.size(0), reduce="sum")

        # 节点特征更新: h_i' = φ(h_i, Σ_j m_ij)
        m_i = scatter(m_ij, row, dim=0, dim_size=h.size(0), reduce="sum")  # [N, H]
        h = self.node_mlp(torch.cat([h, m_i], dim=-1))  # [N, H]

        return h, x


class SimpleEGNN(nn.Module):
    def __init__(
            self,
            hidden_channels: int,
            num_filters: int = 128,  # 这几个参数只是为了接口兼容 SchNet
            num_interactions: int = 4,  # 用作 EGNN 层数
            num_gaussians: int = 50,  # 不使用，只是保留接口
            cutoff: float = 5.0,
            max_num_neighbors: int = 64,
            max_z: int = 100,  # 最大原子序数（可以根据体系调大/调小）
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_interactions
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

        # 用原子序数 z 做 embedding -> 初始节点特征 h
        self.embedding = nn.Embedding(max_z + 1, hidden_channels)

        # 多层 EGNN
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_channels) for _ in range(self.num_layers)
        ])

        # 图级读出 MLP: h_graph -> 能量
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),  # 输出 [num_graphs, 1]
        )

    def forward(self, z, pos, batch):
        """
        z: [N_total]   原子序数
        pos: [N_total, 3]
        batch: [N_total]  每个原子属于第几个分子
        """
        # 节点初始化
        h = self.embedding(z)  # [N, H]

        # 用坐标 + batch 构图，相当于“自动建邻接”
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            max_num_neighbors=self.max_num_neighbors,
        )  # [2, E]

        # 多层 EGNN message passing
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index)

        # 图级 pooling —— 把节点特征汇总到分子级别
        g = global_mean_pool(h, batch)  # [num_graphs, H]

        # 输出能量
        out = self.out_mlp(g)  # [num_graphs, 1]
        return out
