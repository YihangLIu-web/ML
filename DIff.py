"""
高性能SchNet优化版本 + 图扩散预训练
基于原有SchNet架构，添加图扩散预训练功能，用于更好的分子表征学习

主要增强：
1. 图扩散预训练模块
2. 噪声调度器
3. 四阶段训练流程：扩散预训练 → 监督微调
4. 保持原有的高性能优化策略
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import radius_graph, knn_graph
from torch_geometric.datasets import QM9
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
from torch_scatter import scatter_mean, scatter_max, scatter_add
import time
import threading
from collections import deque
import seaborn as sns
from datetime import datetime, timedelta
import math
import warnings

warnings.filterwarnings('ignore')

# 设置matplotlib样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ########################
# # 高性能配置 + 扩散预训练配置
# ########################
CONFIG = {
    # 数据集配置
    "dataset_path": '/home/user/ljrFiles/1/Q9',
    "target_index": 11,
    "split_ratio": [0.8, 0.1, 0.1],
    "dataset_size": 30000,
    "use_full_dataset": False,

    # 高性能模型配置
    "output_dim": 1,
    "hidden_dim": 128,
    "num_filters": 64,
    "num_interactions": 6,
    "num_gaussians": 25,
    "cutoff": 6.0,
    "dropout_rate": 0.15,

    # 高级架构组件
    "use_attention": False,
    "use_residual_gates": True,
    "use_layer_norm": True,
    "use_multi_scale": True,
    "attention_heads": 4,

    # 智能邻居管理
    "base_max_neighbors": 16,
    "large_mol_threshold": 20,
    "max_mol_size": 100,
    "adaptive_neighbors": True,
    "use_radius_graph": False,
    "radius_cutoff": 6.0,

    # 图扩散预训练配置 (NEW)
    "diffusion_timesteps": 1000,
    "diffusion_beta_start": 0.0001,
    "diffusion_beta_end": 0.02,
    "diffusion_pretrain_epochs": 150,  # 扩散预训练轮数
    "max_atomic_num": 20,  # QM9数据集中的最大原子序数

    # 高性能训练配置
    "batch_size": 64,
    "accumulation_steps": 8,
    "learning_rate": 0.001,
    "epochs": 800,  # 减少监督训练轮数（因为有预训练）
    "early_stop_patience": 100,
    "scheduler_patience": 30,
    "warmup_epochs": 50,

    # 混合精度训练
    "use_amp": True,
    "amp_opt_level": "O1",

    # 数据加载优化
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,

    # 可视化配置
    "update_interval": 5,
    "history_length": 200,
    "save_plots": True,

    "device": "cuda",
    "random_seed": 42,
}

torch.manual_seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])

# 启用混合精度
if CONFIG['use_amp']:
    from torch.cuda.amp import autocast, GradScaler

    scaler = GradScaler()


# ########################
# # 图扩散预训练组件 (NEW)
# ########################

class NoiseScheduler:
    """噪声调度器用于扩散过程"""

    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.timesteps = timesteps
        self.device = device

        # 线性beta调度
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # 扩散过程计算所需的值
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def to(self, device):
        """移动调度器到指定设备"""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

    def add_noise(self, x_start, noise, timesteps):
        """根据噪声调度为干净数据添加噪声"""
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def _extract(self, a, t, x_shape):
        """为时间步t提取适当的值"""
        batch_size = t.shape[0]
        # 确保所有张量在同一设备上
        if a.device != t.device:
            a = a.to(t.device)
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class GraphDiffusionPretrainer(nn.Module):
    """图扩散预训练器，用于学习分子表征"""

    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model  # SchNet模型
        self.config = config

        # 噪声调度器
        self.noise_scheduler = NoiseScheduler(
            timesteps=config["diffusion_timesteps"],
            beta_start=config["diffusion_beta_start"],
            beta_end=config["diffusion_beta_end"],
            device=config["device"]
        )

        # 原子特征重建头
        self.atomic_denoising_head = nn.Sequential(
            nn.Linear(config["hidden_dim"], config["hidden_dim"]),
            nn.SiLU(),
            nn.Dropout(config["dropout_rate"]),
            nn.Linear(config["hidden_dim"], config["max_atomic_num"] + 1)  # 预测原子序数
        )

        # 坐标重建头
        self.coord_denoising_head = nn.Sequential(
            nn.Linear(config["hidden_dim"], config["hidden_dim"] // 2),
            nn.SiLU(),
            nn.Dropout(config["dropout_rate"]),
            nn.Linear(config["hidden_dim"] // 2, 3)  # 预测3D坐标
        )

        # 时间步嵌入
        self.time_embedding = nn.Sequential(
            nn.Linear(1, config["hidden_dim"] // 4),
            nn.SiLU(),
            nn.Linear(config["hidden_dim"] // 4, config["hidden_dim"] // 2)
        )

    def to(self, device):
        """重写to方法以同时移动噪声调度器"""
        super().to(device)
        self.noise_scheduler = self.noise_scheduler.to(device)
        return self

    def get_timestep_embedding(self, timesteps):
        """获取时间步嵌入"""
        # 归一化时间步到[0,1]
        normalized_t = timesteps.float() / self.config["diffusion_timesteps"]
        normalized_t = normalized_t.unsqueeze(-1)  # [batch_size, 1]
        return self.time_embedding(normalized_t)

    def forward(self, data, timesteps=None):
        """扩散预训练前向传播"""
        x, pos, batch = data.x, data.pos, data.batch
        device = x.device
        batch_size = batch.max().item() + 1

        # 确保噪声调度器在正确的设备上
        if self.noise_scheduler.device != device:
            self.noise_scheduler = self.noise_scheduler.to(device)

        if timesteps is None:
            # 为训练采样随机时间步
            timesteps = torch.randint(0, self.noise_scheduler.timesteps, (batch_size,), device=device)

        # 为原子特征和坐标添加噪声
        # 原子特征噪声（连续化）
        x_float = x.float()  # [num_nodes, num_features]
        noise_x = torch.randn_like(x_float)

        # 坐标噪声
        noise_pos = torch.randn_like(pos)

        # 获取每个节点的时间步
        node_timesteps = timesteps[batch]

        # 根据调度添加噪声
        noisy_x = self.noise_scheduler.add_noise(x_float, noise_x, node_timesteps)
        noisy_pos = self.noise_scheduler.add_noise(pos, noise_pos, node_timesteps)

        # 创建带噪声的数据对象
        noisy_data = type(data)()
        noisy_data.x = noisy_x
        noisy_data.pos = noisy_pos
        noisy_data.batch = batch

        # 通过基础模型获取图表征
        # 注意：这里我们需要修改base_model的forward方法来返回中间特征
        # 我们将使用模型的embedding和interaction layers

        # 手动前向传播以获取节点特征
        node_features = self._forward_to_node_features(noisy_data)

        # 时间步嵌入
        time_emb = self.get_timestep_embedding(timesteps)  # [batch_size, hidden_dim//2]

        # 将时间嵌入广播到每个节点
        time_emb_nodes = time_emb[batch]  # [num_nodes, hidden_dim//2]

        # 拼接节点特征和时间嵌入
        enhanced_features = torch.cat([
            node_features,
            time_emb_nodes
        ], dim=-1)  # [num_nodes, hidden_dim + hidden_dim//2]

        # 使用线性层调整维度
        if not hasattr(self, 'feature_adapter'):
            self.feature_adapter = nn.Linear(
                node_features.size(-1) + time_emb_nodes.size(-1),
                self.config["hidden_dim"]
            ).to(device)

        adapted_features = self.feature_adapter(enhanced_features)

        # 预测去噪后的特征
        pred_x_logits = self.atomic_denoising_head(adapted_features)
        pred_pos = self.coord_denoising_head(adapted_features)

        return {
            'pred_x_logits': pred_x_logits,
            'pred_pos': pred_pos,
            'target_x': x,
            'target_pos': pos,
            'noise_x': noise_x,
            'noise_pos': noise_pos,
            'timesteps': node_timesteps,
            'noisy_x': noisy_x,
            'noisy_pos': noisy_pos
        }

    def _forward_to_node_features(self, data):
        """通过基础模型获取节点级特征"""
        # 构建图结构
        edge_index = self.base_model._build_advanced_graph(data.pos, data.batch)

        if edge_index.size(1) == 0:
            # 如果没有边，返回嵌入特征
            return self.base_model.embedding(data.x)

        row, col = edge_index
        edge_weight = (data.pos[row] - data.pos[col]).norm(dim=-1)

        # 嵌入
        x = self.base_model.embedding(data.x)

        # 通过交互层
        for interaction in self.base_model.interactions:
            x = interaction(x, edge_index, edge_weight, None)

        return x

    def compute_loss(self, outputs):
        """计算扩散预训练损失"""
        # 原子特征预测损失（分类损失）
        x_loss = F.cross_entropy(
            outputs['pred_x_logits'].view(-1, self.config["max_atomic_num"] + 1),
            outputs['target_x'][:, 0].long()  # 假设第一个特征是原子序数
        )

        # 坐标预测损失（回归损失 - 预测噪声）
        pos_loss = F.mse_loss(outputs['pred_pos'], outputs['noise_pos'])

        # 组合损失
        total_loss = x_loss + pos_loss

        return {
            'total_loss': total_loss,
            'x_loss': x_loss,
            'pos_loss': pos_loss
        }


# ########################
# # 增强的SchNet模型组件（保持原有）
# ########################

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index):
        batch_size = x.size(0)

        # 计算Q, K, V
        Q = self.query(x).view(batch_size, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, self.num_heads, self.head_dim)

        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # 应用注意力
        out = torch.matmul(attention, V)
        out = out.view(batch_size, self.hidden_dim)

        return self.output(out)


class ResidualGate(nn.Module):
    """残差门控机制"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x, residual):
        gate_input = torch.cat([x, residual], dim=-1)
        gate_value = self.gate(gate_input)
        return gate_value * x + (1 - gate_value) * residual


class EnhancedGaussianSmearing(nn.Module):
    """增强的高斯展开"""

    def __init__(self, start=0.0, stop=8.0, num_gaussians=50):
        super().__init__()
        # 使用更精细的高斯分布
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

        # 学习的缩放因子
        self.scale = nn.Parameter(torch.ones(num_gaussians))

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return self.scale * torch.exp(self.coeff * torch.pow(dist, 2))


class AdvancedCFConv(nn.Module):
    """高级连续滤波卷积"""

    def __init__(self, in_channels, out_channels, num_filters, cutoff):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 增强的距离展开
        self.distance_expansion = EnhancedGaussianSmearing(0.0, cutoff, num_filters)

        # 更深的滤波网络
        self.filter_network = nn.Sequential(
            nn.Linear(num_filters, num_filters * 2),
            nn.SiLU(),
            nn.LayerNorm(num_filters * 2),
            nn.Dropout(0.1),
            nn.Linear(num_filters * 2, num_filters),
            nn.SiLU(),
            nn.Linear(num_filters, in_channels * out_channels)
        )

        # 添加注意力权重
        self.attention_network = nn.Sequential(
            nn.Linear(num_filters, num_filters // 2),
            nn.SiLU(),
            nn.Linear(num_filters // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_weight, edge_attr):
        row, col = edge_index

        # 距离展开
        edge_weight_expanded = self.distance_expansion(edge_weight)

        # 注意力权重
        attention_weights = self.attention_network(edge_weight_expanded)
        edge_weight_expanded = edge_weight_expanded * attention_weights

        # 滤波器
        W = self.filter_network(edge_weight_expanded)
        W = W.view(-1, self.in_channels, self.out_channels)

        # 卷积操作
        x_j = x[col].unsqueeze(-1)
        x_j = torch.bmm(W.transpose(1, 2), x_j).squeeze(-1)

        return scatter_mean(x_j, row, dim=0, dim_size=x.size(0))


class AdvancedInteractionBlock(nn.Module):
    """高级交互块"""

    def __init__(self, hidden_dim, num_filters, cutoff, use_attention=True, use_residual_gates=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        self.use_residual_gates = use_residual_gates

        # 主要卷积
        self.cfconv = AdvancedCFConv(hidden_dim, hidden_dim, num_filters, cutoff)

        # 增强的原子级网络
        self.atom_wise_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 注意力机制
        if use_attention:
            self.attention = MultiHeadAttention(hidden_dim, CONFIG['attention_heads'])

        # 残差门控
        if use_residual_gates:
            self.residual_gate = ResidualGate(hidden_dim)

        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        residual = x

        # 卷积操作
        x_conv = self.cfconv(x, edge_index, edge_weight, edge_attr)
        x_conv = self.atom_wise_net(x_conv)

        # 注意力机制
        if self.use_attention:
            x_conv = x_conv + self.attention(x_conv, edge_index)

        # 残差连接
        if self.use_residual_gates:
            x = self.residual_gate(x_conv, residual)
        else:
            x = x + x_conv

        # 层归一化
        x = self.layer_norm(x)

        return x


class MultiScalePooling(nn.Module):
    """多尺度池化"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.mean_pool = global_mean_pool
        self.max_pool = global_max_pool
        self.add_pool = global_add_pool

        # 融合网络
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x, batch):
        mean_pooled = self.mean_pool(x, batch)
        max_pooled = self.max_pool(x, batch)
        add_pooled = self.add_pool(x, batch)

        # 融合多种池化结果
        combined = torch.cat([mean_pooled, max_pooled, add_pooled], dim=-1)
        return self.fusion(combined)


class OptimizedSchNetModel(nn.Module):
    """优化的SchNet模型（带扩散预训练支持）"""

    def __init__(self, node_dim, hidden_dim, output_dim, num_filters,
                 num_interactions, num_gaussians, cutoff, dropout):
        super().__init__()
        self.cutoff = cutoff

        # 更深的嵌入层
        self.embedding = nn.Sequential(
            nn.Linear(node_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 高级交互层
        self.interactions = nn.ModuleList([
            AdvancedInteractionBlock(
                hidden_dim, num_filters, cutoff,
                use_attention=CONFIG['use_attention'],
                use_residual_gates=CONFIG['use_residual_gates']
            )
            for _ in range(num_interactions)
        ])

        # 多尺度池化
        if CONFIG['use_multi_scale']:
            self.pooling = MultiScalePooling(hidden_dim)
        else:
            self.pooling = global_mean_pool

        # 更复杂的输出网络
        self.output_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def _build_advanced_graph(self, pos, batch):
        """构建高级图结构"""
        edges = []

        # KNN图
        knn_edges = knn_graph(pos, k=CONFIG['base_max_neighbors'], batch=batch)
        edges.append(knn_edges)

        # 径向图（如果启用）
        if CONFIG['use_radius_graph']:
            radius_edges = radius_graph(pos, r=CONFIG['radius_cutoff'], batch=batch)
            edges.append(radius_edges)

        # 合并边
        if len(edges) > 1:
            edge_index = torch.cat(edges, dim=1)
            # 去重
            edge_index = torch.unique(edge_index, dim=1)
        else:
            edge_index = edges[0]

        return edge_index

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch

        # 构建高级图
        edge_index = self._build_advanced_graph(pos, batch)

        if edge_index.size(1) == 0:
            return torch.zeros(batch.max().item() + 1, device=x.device)

        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)

        # 嵌入
        x = self.embedding(x)

        # 交互层
        for interaction in self.interactions:
            x = interaction(x, edge_index, edge_weight, None)

        # 池化
        if CONFIG['use_multi_scale']:
            x = self.pooling(x, batch)
        else:
            x = global_mean_pool(x, batch)

        # 输出
        return self.output_network(x).squeeze()


# ########################
# # 数据处理（保持原有）
# ########################
class OptimizedQM9DataProcessor:
    def __init__(self, config):
        print("Loading QM9 dataset...")
        # 加载数据集并限制大小
        self.dataset = QM9(root=config['dataset_path'])

        # 限制数据集大小
        dataset_size = config.get('dataset_size', 30000)
        if dataset_size and dataset_size < len(self.dataset):
            self.dataset = self.dataset[:dataset_size]
            print(f"Using subset of {dataset_size} molecules")

        self._preprocess_target(config['target_index'])
        print(f"Final dataset size: {len(self.dataset)}")

    def _preprocess_target(self, target_idx):
        self.target = self.dataset.data.y[:, target_idx]

    def get_data_loaders(self):
        # 确保分割比例正确
        total_size = len(self.dataset)
        ratios = CONFIG['split_ratio']

        # 计算每个集合的确切大小
        train_size = int(ratios[0] * total_size)
        val_size = int(ratios[1] * total_size)
        test_size = total_size - train_size - val_size  # 剩余的分配给测试集

        print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")

        # 确保大小加起来等于总大小
        assert train_size + val_size + test_size == total_size, f"Split sizes don't match: {train_size + val_size + test_size} != {total_size}"

        train_set, val_set, test_set = random_split(
            self.dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(CONFIG['random_seed']))

        # 高性能数据加载器
        loader_kwargs = {
            'batch_size': CONFIG['batch_size'],
            'num_workers': CONFIG['num_workers'],
            'pin_memory': CONFIG['pin_memory'],
            'persistent_workers': CONFIG['persistent_workers'],
            'prefetch_factor': CONFIG['prefetch_factor']
        }

        return (
            DataLoader(train_set, shuffle=True, **loader_kwargs),
            DataLoader(val_set, shuffle=False, **loader_kwargs),
            DataLoader(test_set, shuffle=False, **loader_kwargs)
        )


# ########################
# # 增强的训练器（带扩散预训练）
# ########################
class DiffusionPretrainedTrainer:
    def __init__(self, model, config, train_loader=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.config = config

        # 图扩散预训练器
        self.diffusion_pretrainer = GraphDiffusionPretrainer(
            base_model=self.model,
            config=config
        ).to(self.device)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # 扩散预训练优化器
        self.diffusion_optimizer = torch.optim.AdamW(
            self.diffusion_pretrainer.parameters(),
            lr=config['learning_rate'],
            weight_decay=1e-5
        )

        # 计算每个epoch的步数（如果提供了train_loader）
        if train_loader is not None:
            steps_per_epoch = len(train_loader) // config['accumulation_steps']
            # 使用OneCycleLR
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config['learning_rate'],
                epochs=config['epochs'],
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1,
                anneal_strategy='cos'
            )

            # 扩散预训练调度器
            self.diffusion_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.diffusion_optimizer,
                max_lr=config['learning_rate'],
                epochs=config['diffusion_pretrain_epochs'],
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1,
                anneal_strategy='cos'
            )
        else:
            # 备用调度器
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.8,
                patience=config['scheduler_patience'],
                min_lr=1e-7,
                verbose=True
            )

            self.diffusion_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.diffusion_optimizer,
                mode='min',
                factor=0.8,
                patience=config['scheduler_patience'],
                min_lr=1e-7
            )

        self.criterion = nn.L1Loss()

        # 混合精度
        if config['use_amp']:
            self.scaler = GradScaler()
            self.diffusion_scaler = GradScaler()

        self.best_val_loss = float('inf')
        self.early_stop_counter = 0

    def get_gpu_memory(self):
        """获取GPU内存使用情况"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 ** 3
        return 0

    def train_diffusion_epoch(self, loader):
        """图扩散预训练单个epoch"""
        self.diffusion_pretrainer.train()
        epoch_loss = 0
        epoch_x_loss = 0
        epoch_pos_loss = 0
        num_batches = 0
        accumulated_loss = 0

        torch.cuda.empty_cache()

        for batch_idx, batch in enumerate(loader):
            try:
                batch = batch.to(self.device, non_blocking=True)

                # 检查批次大小
                if hasattr(batch, 'num_nodes') and batch.num_nodes > 5000:
                    print(f"⚠️ 跳过大批次: {batch.num_nodes} 节点")
                    continue

                if CONFIG['use_amp']:
                    with autocast():
                        # 扩散前向传播
                        diffusion_outputs = self.diffusion_pretrainer(batch)

                        # 计算扩散损失
                        loss_dict = self.diffusion_pretrainer.compute_loss(diffusion_outputs)
                        loss = loss_dict['total_loss'] / CONFIG['accumulation_steps']

                    self.diffusion_scaler.scale(loss).backward()
                    accumulated_loss += loss.item()

                    # 梯度累积
                    if (batch_idx + 1) % CONFIG['accumulation_steps'] == 0:
                        self.diffusion_scaler.unscale_(self.diffusion_optimizer)
                        nn.utils.clip_grad_norm_(self.diffusion_pretrainer.parameters(), max_norm=1.0)
                        self.diffusion_scaler.step(self.diffusion_optimizer)
                        self.diffusion_scaler.update()
                        self.diffusion_optimizer.zero_grad()

                        # 调度器步进
                        if hasattr(self.diffusion_scheduler, 'step') and not isinstance(
                                self.diffusion_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.diffusion_scheduler.step()

                        epoch_loss += accumulated_loss * CONFIG['accumulation_steps']
                        epoch_x_loss += loss_dict['x_loss'].item()
                        epoch_pos_loss += loss_dict['pos_loss'].item()
                        accumulated_loss = 0
                        num_batches += 1

                        if num_batches % 10 == 0:
                            torch.cuda.empty_cache()
                else:
                    # 常规训练
                    diffusion_outputs = self.diffusion_pretrainer(batch)
                    loss_dict = self.diffusion_pretrainer.compute_loss(diffusion_outputs)
                    loss = loss_dict['total_loss'] / CONFIG['accumulation_steps']

                    loss.backward()
                    accumulated_loss += loss.item()

                    if (batch_idx + 1) % CONFIG['accumulation_steps'] == 0:
                        nn.utils.clip_grad_norm_(self.diffusion_pretrainer.parameters(), max_norm=1.0)
                        self.diffusion_optimizer.step()
                        self.diffusion_optimizer.zero_grad()

                        # 调度器步进
                        if hasattr(self.diffusion_scheduler, 'step') and not isinstance(
                                self.diffusion_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.diffusion_scheduler.step()

                        epoch_loss += accumulated_loss * CONFIG['accumulation_steps']
                        epoch_x_loss += loss_dict['x_loss'].item()
                        epoch_pos_loss += loss_dict['pos_loss'].item()
                        accumulated_loss = 0
                        num_batches += 1

                        if num_batches % 10 == 0:
                            torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️ 扩散预训练显存不足，跳过批次 {batch_idx}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    print(f"⚠️ 扩散预训练批次错误: {e}")
                    continue
            except Exception as e:
                print(f"⚠️ 扩散预训练批次错误: {e}")
                continue

        return {
            'total_loss': epoch_loss / num_batches if num_batches > 0 else float('inf'),
            'x_loss': epoch_x_loss / num_batches if num_batches > 0 else float('inf'),
            'pos_loss': epoch_pos_loss / num_batches if num_batches > 0 else float('inf')
        }

    def train_epoch(self, loader):
        """监督训练单个epoch"""
        self.model.train()
        epoch_loss = 0
        num_batches = 0
        accumulated_loss = 0

        torch.cuda.empty_cache()

        for batch_idx, batch in enumerate(loader):
            try:
                batch = batch.to(self.device, non_blocking=True)

                if hasattr(batch, 'num_nodes') and batch.num_nodes > 5000:
                    print(f"⚠️ 跳过大批次: {batch.num_nodes} 节点")
                    continue

                if CONFIG['use_amp']:
                    with autocast():
                        pred = self.model(batch)
                        loss = self.criterion(pred, batch.y[:, CONFIG['target_index']])
                        loss = loss / CONFIG['accumulation_steps']

                    self.scaler.scale(loss).backward()
                    accumulated_loss += loss.item()

                    if (batch_idx + 1) % CONFIG['accumulation_steps'] == 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()

                        if hasattr(self.scheduler, 'step') and not isinstance(self.scheduler,
                                                                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step()

                        epoch_loss += accumulated_loss
                        accumulated_loss = 0
                        num_batches += 1

                        if num_batches % 10 == 0:
                            torch.cuda.empty_cache()
                else:
                    pred = self.model(batch)
                    loss = self.criterion(pred, batch.y[:, CONFIG['target_index']])
                    loss = loss / CONFIG['accumulation_steps']

                    loss.backward()
                    accumulated_loss += loss.item()

                    if (batch_idx + 1) % CONFIG['accumulation_steps'] == 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        if hasattr(self.scheduler, 'step') and not isinstance(self.scheduler,
                                                                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step()

                        epoch_loss += accumulated_loss
                        accumulated_loss = 0
                        num_batches += 1

                        if num_batches % 10 == 0:
                            torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️ 显存不足，跳过批次 {batch_idx}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    print(f"⚠️ 训练批次错误: {e}")
                    continue
            except Exception as e:
                print(f"⚠️ 训练批次错误: {e}")
                continue

        return epoch_loss / num_batches if num_batches > 0 else float('inf')

    def evaluate(self, loader, collect_predictions=False):
        """评估模型"""
        self.model.eval()
        total_mae = 0.0
        total_mse = 0.0
        total_samples = 0
        predictions, targets = [], []

        with torch.no_grad():
            for batch in loader:
                try:
                    batch = batch.to(self.device, non_blocking=True)

                    if CONFIG['use_amp']:
                        with autocast():
                            pred = self.model(batch)
                    else:
                        pred = self.model(batch)

                    target = batch.y[:, CONFIG['target_index']]

                    total_mae += F.l1_loss(pred, target, reduction='sum').item()
                    total_mse += F.mse_loss(pred, target, reduction='sum').item()
                    total_samples += target.size(0)

                    if collect_predictions:
                        predictions.extend(pred.cpu().numpy())
                        targets.extend(target.cpu().numpy())

                except Exception as e:
                    continue

        avg_mae = total_mae / total_samples if total_samples > 0 else float('inf')
        avg_mse = total_mse / total_samples if total_samples > 0 else float('inf')

        metrics = {'mae': avg_mae, 'mse': avg_mse}

        if len(predictions) > 0 and len(targets) > 0:
            predictions = np.array(predictions)
            targets = np.array(targets)

            metrics['rmse'] = np.sqrt(avg_mse)

            # R²
            ss_res = np.sum((targets - predictions) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # MAPE
            non_zero_mask = targets != 0
            if np.any(non_zero_mask):
                metrics['mape'] = np.mean(
                    np.abs((targets[non_zero_mask] - predictions[non_zero_mask]) / targets[non_zero_mask])) * 100
            else:
                metrics['mape'] = float('inf')

            metrics['max_error'] = np.max(np.abs(targets - predictions))
            metrics['mean_bias'] = np.mean(predictions - targets)

        return metrics, predictions, targets

    def transfer_pretrained_weights(self):
        """将扩散预训练的权重转移到主模型"""
        print("正在转移扩散预训练的权重到主模型...")

        # 转移SchNet编码器的权重
        pretrained_state = self.diffusion_pretrainer.base_model.state_dict()

        # 只转移匹配的权重
        model_state = self.model.state_dict()
        transferred_keys = []

        for key in pretrained_state:
            if key in model_state and pretrained_state[key].shape == model_state[key].shape:
                model_state[key] = pretrained_state[key].clone()
                transferred_keys.append(key)

        self.model.load_state_dict(model_state)
        print(f"成功转移 {len(transferred_keys)} 个参数层")

        # 冻结部分预训练权重（可选）
        # for name, param in self.model.named_parameters():
        #     if any(key in name for key in transferred_keys[:len(transferred_keys)//2]):
        #         param.requires_grad = False

    def run(self, train_loader, val_loader):
        """运行完整的训练流程：扩散预训练 + 监督微调"""
        print("开始图扩散预训练 + SchNet监督训练...")
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"扩散预训练器参数: {sum(p.numel() for p in self.diffusion_pretrainer.parameters()):,}")

        # ===============================
        # 阶段1: 图扩散预训练
        # ===============================
        print("\n" + "=" * 80)
        print("阶段1: 图扩散预训练")
        print("=" * 80)

        diffusion_history = {
            'total_loss': [],
            'x_loss': [],
            'pos_loss': []
        }

        for epoch in range(CONFIG['diffusion_pretrain_epochs']):
            epoch_start_time = time.time()

            # 扩散预训练
            diffusion_losses = self.train_diffusion_epoch(train_loader)

            # 记录历史
            diffusion_history['total_loss'].append(diffusion_losses['total_loss'])
            diffusion_history['x_loss'].append(diffusion_losses['x_loss'])
            diffusion_history['pos_loss'].append(diffusion_losses['pos_loss'])

            epoch_time = time.time() - epoch_start_time
            current_lr = self.diffusion_optimizer.param_groups[0]['lr']
            gpu_mem = self.get_gpu_memory()

            # ReduceLROnPlateau调度器步进
            if isinstance(self.diffusion_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.diffusion_scheduler.step(diffusion_losses['total_loss'])

            if (epoch + 1) % 10 == 0:
                print(f"扩散预训练 Epoch {epoch + 1:3d}/{CONFIG['diffusion_pretrain_epochs']} | "
                      f"Total Loss: {diffusion_losses['total_loss']:.6f} | "
                      f"X Loss: {diffusion_losses['x_loss']:.6f} | "
                      f"Pos Loss: {diffusion_losses['pos_loss']:.6f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Time: {epoch_time:.1f}s | "
                      f"GPU: {gpu_mem:.2f}GB")

        # 保存扩散预训练模型
        torch.save({
            'diffusion_pretrainer_state_dict': self.diffusion_pretrainer.state_dict(),
            'diffusion_optimizer_state_dict': self.diffusion_optimizer.state_dict(),
            'diffusion_history': diffusion_history,
            'config': CONFIG
        }, 'diffusion_pretrained_model.pth')

        # 转移预训练权重
        self.transfer_pretrained_weights()

        # ===============================
        # 阶段2: 监督微调
        # ===============================
        print("\n" + "=" * 80)
        print("阶段2: 监督微调（使用扩散预训练权重）")
        print("=" * 80)

        # 重新初始化优化器和调度器（可选择使用更小的学习率）
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=CONFIG['learning_rate'] * 0.1,  # 使用更小的学习率进行微调
            weight_decay=1e-5
        )

        if hasattr(self, 'scheduler'):
            steps_per_epoch = len(train_loader) // CONFIG['accumulation_steps']
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=CONFIG['learning_rate'] * 0.1,
                epochs=CONFIG['epochs'],
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1,
                anneal_strategy='cos'
            )

        # 监督训练表头
        print(f"{'Epoch':>5} | {'Train MAE':>10} | {'Train MSE':>10} | {'Val MAE':>10} | {'Val MSE':>10} | "
              f"{'Val RMSE':>10} | {'Val R²':>8} | {'LR':>10} | {'Time':>8} | {'GPU':>8}")
        print("=" * 120)

        for epoch in range(CONFIG['epochs']):
            epoch_start_time = time.time()

            # 监督训练
            train_loss = self.train_epoch(train_loader)

            # 验证
            collect_pred = (epoch % 10 == 0)
            val_metrics, predictions, targets = self.evaluate(val_loader, collect_predictions=collect_pred)

            # 计算训练集指标
            if epoch % 10 == 0:
                train_metrics, _, _ = self.evaluate(train_loader, collect_predictions=False)
            else:
                train_metrics = {'mae': train_loss, 'mse': train_loss ** 2}

            # 学习率调度
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['mae'])

            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            gpu_mem = self.get_gpu_memory()

            # Early stopping
            val_loss = val_metrics['mae']
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                # 保存最佳模型
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_metrics': val_metrics,
                    'train_metrics': train_metrics,
                    'diffusion_history': diffusion_history
                }, 'best_diffusion_finetuned_model.pth')
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= CONFIG['early_stop_patience']:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

            # 输出信息
            if (epoch + 1) % 5 == 0:
                print(f"{epoch + 1:5d} | {train_metrics['mae']:10.6f} | {train_metrics['mse']:10.6f} | "
                      f"{val_metrics['mae']:10.6f} | {val_metrics['mse']:10.6f} | {val_metrics.get('rmse', 0):10.6f} | "
                      f"{val_metrics.get('r2', 0):8.4f} | {current_lr:10.2e} | {epoch_time:8.1f}s | {gpu_mem:8.2f}GB")

                if (epoch + 1) % 20 == 0 and 'mape' in val_metrics:
                    print(f"      Additional metrics - MAPE: {val_metrics['mape']:.2f}%, "
                          f"Max Error: {val_metrics['max_error']:.6f}, "
                          f"Mean Bias: {val_metrics['mean_bias']:+.6f}")

        return self.best_val_loss


# ########################
# # 可视化函数
# ########################
def plot_diffusion_training_history(diffusion_history, save_path='diffusion_training_history.png'):
    """绘制扩散预训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 总损失
    axes[0, 0].plot(diffusion_history['total_loss'], label='Total Diffusion Loss', color='blue')
    axes[0, 0].set_title('Diffusion Pretraining - Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 原子特征损失
    axes[0, 1].plot(diffusion_history['x_loss'], label='Atomic Feature Loss', color='red')
    axes[0, 1].set_title('Diffusion Pretraining - Atomic Feature Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 坐标损失
    axes[1, 0].plot(diffusion_history['pos_loss'], label='Coordinate Loss', color='green')
    axes[1, 0].set_title('Diffusion Pretraining - Coordinate Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 组合视图
    axes[1, 1].plot(diffusion_history['total_loss'], label='Total Loss', alpha=0.8)
    axes[1, 1].plot(diffusion_history['x_loss'], label='Atomic Loss', alpha=0.8)
    axes[1, 1].plot(diffusion_history['pos_loss'], label='Coordinate Loss', alpha=0.8)
    axes[1, 1].set_title('Diffusion Pretraining - All Losses')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"扩散预训练历史图保存至: {save_path}")


# ########################
# # 主程序
# ########################
if __name__ == "__main__":
    print("=" * 80)
    print("SchNet + Graph Diffusion Pretraining System")
    print(f"PyTorch: {torch.__version__} | Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name()} ({torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB)")
        torch.cuda.empty_cache()
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")
        print(f"Current GPU memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB")
    print("=" * 80)

    try:
        # 数据准备
        processor = OptimizedQM9DataProcessor(CONFIG)
        train_loader, val_loader, test_loader = processor.get_data_loaders()

        # 创建模型
        model = OptimizedSchNetModel(
            node_dim=11,
            hidden_dim=CONFIG['hidden_dim'],
            output_dim=CONFIG['output_dim'],
            num_filters=CONFIG['num_filters'],
            num_interactions=CONFIG['num_interactions'],
            num_gaussians=CONFIG['num_gaussians'],
            cutoff=CONFIG['cutoff'],
            dropout=CONFIG['dropout_rate']
        )

        print(f"\n模型配置:")
        print(f"  Hidden dim: {CONFIG['hidden_dim']}")
        print(f"  Interactions: {CONFIG['num_interactions']}")
        print(f"  扩散预训练轮数: {CONFIG['diffusion_pretrain_epochs']}")
        print(f"  监督训练轮数: {CONFIG['epochs']}")
        print(f"  扩散时间步: {CONFIG['diffusion_timesteps']}")

        # 开始训练（扩散预训练 + 监督微调）
        trainer = DiffusionPretrainedTrainer(model, CONFIG, train_loader)
        best_loss = trainer.run(train_loader, val_loader)

        # 加载最佳模型进行测试
        print("\n加载最佳模型进行最终评估...")
        checkpoint = torch.load('best_diffusion_finetuned_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])

        # 完整评估
        print("\n最终模型评估:")
        print("=" * 80)

        # 训练集评估
        train_metrics, _, _ = trainer.evaluate(train_loader, collect_predictions=True)
        print("训练集指标:")
        print(f"  MAE:        {train_metrics['mae']:.6f}")
        print(f"  MSE:        {train_metrics['mse']:.6f}")
        print(f"  RMSE:       {train_metrics.get('rmse', 0):.6f}")
        print(f"  R²:         {train_metrics.get('r2', 0):.6f}")
        print(f"  MAPE:       {train_metrics.get('mape', 0):.2f}%")
        print(f"  Max Error:  {train_metrics.get('max_error', 0):.6f}")

        # 验证集评估
        val_metrics, _, _ = trainer.evaluate(val_loader, collect_predictions=True)
        print("\n验证集指标:")
        print(f"  MAE:        {val_metrics['mae']:.6f}")
        print(f"  MSE:        {val_metrics['mse']:.6f}")
        print(f"  RMSE:       {val_metrics.get('rmse', 0):.6f}")
        print(f"  R²:         {val_metrics.get('r2', 0):.6f}")
        print(f"  MAPE:       {val_metrics.get('mape', 0):.2f}%")
        print(f"  Max Error:  {val_metrics.get('max_error', 0):.6f}")

        # 测试集评估
        test_metrics, _, _ = trainer.evaluate(test_loader, collect_predictions=True)
        print("\n测试集指标:")
        print(f"  MAE:        {test_metrics['mae']:.6f}")
        print(f"  MSE:        {test_metrics['mse']:.6f}")
        print(f"  RMSE:       {test_metrics.get('rmse', 0):.6f}")
        print(f"  R²:         {test_metrics.get('r2', 0):.6f}")
        print(f"  MAPE:       {test_metrics.get('mape', 0):.2f}%")
        print(f"  Max Error:  {test_metrics.get('max_error', 0):.6f}")

        # 绘制扩散预训练历史
        if 'diffusion_history' in checkpoint:
            plot_diffusion_training_history(checkpoint['diffusion_history'])

        print(f"\n最佳epoch: {checkpoint['epoch'] + 1}")
        print(f"扩散预训练模型保存为: diffusion_pretrained_model.pth")
        print(f"最佳微调模型保存为: best_diffusion_finetuned_model.pth")
        print("\n🎉 图扩散预训练 + SchNet微调训练完成!")
        print("=" * 80)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()