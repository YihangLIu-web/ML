"""
高性能SchNet优化版本 - 充分利用22GB显存
针对大显存优化的SchNet实现，包含多项性能提升策略

主要优化：
1. 大幅增加模型容量和复杂度
2. 混合精度训练（FP16）
3. 更大的批处理大小
4. 高级邻居图构建策略
5. 更复杂的网络架构
6. 数据预取和内存优化
7. 学习率调度优化
8. 梯度累积
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
# # 高性能配置
# ########################
CONFIG = {
    # 数据集配置
    "dataset_path": '/home/user/ljrFiles/1/Q9',
    "target_index": 11,
    "split_ratio": [0.8, 0.1, 0.1],  # 修改为整数比例
    "dataset_size": 30000,  # 限制数据集大小
    "use_full_dataset": False,  # 不使用完整数据集

    # 高性能模型配置 - 显存优化版本
    "output_dim": 1,
    "hidden_dim": 128,  # 降低维度 256 -> 128
    "num_filters": 64,  # 降低滤波器 128 -> 64
    "num_interactions": 6,  # 降低交互层 8 -> 6
    "num_gaussians": 25,  # 降低高斯基 50 -> 25
    "cutoff": 6.0,  # 降低截断距离 8.0 -> 6.0
    "dropout_rate": 0.15,

    # 高级架构组件 - 部分禁用
    "use_attention": False,  # 暂时禁用注意力机制（显存密集）
    "use_residual_gates": True,
    "use_layer_norm": True,
    "use_multi_scale": True,  # 暂时禁用多尺度（显存密集）
    "attention_heads": 4,  # 减少注意力头 8 -> 4

    # 智能邻居管理 - 减少邻居数
    "base_max_neighbors": 16,  # 降低邻居数 32 -> 16
    "large_mol_threshold": 20,
    "max_mol_size": 100,
    "adaptive_neighbors": True,
    "use_radius_graph": False,  # 禁用径向图减少边数
    "radius_cutoff": 6.0,

    # 高性能训练配置 - 显存优化版本
    "batch_size": 64,  # 降低批处理 128 -> 64
    "accumulation_steps": 8,  # 增加梯度累积，等效batch_size=512
    "learning_rate": 0.001,
    "epochs": 1000,
    "early_stop_patience": 100,
    "scheduler_patience": 30,
    "warmup_epochs": 50,

    # 混合精度训练
    "use_amp": True,  # 自动混合精度
    "amp_opt_level": "O1",

    # 数据加载优化 - 减少worker数量
    "num_workers": 4,  # 减少进程 8 -> 4
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,  # 减少预取 4 -> 2

    # 可视化配置
    "update_interval": 5,  # 每5个epoch更新图表
    "history_length": 200,  # 更长历史
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
# # 高级模型组件
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
    """优化的SchNet模型"""

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
# # 数据处理优化
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


class MetricsCalculator:
    """完整的回归评估指标计算器"""

    @staticmethod
    def calculate_all_metrics(predictions, targets):
        """计算所有回归指标"""
        predictions = np.array(predictions)
        targets = np.array(targets)

        metrics = {}

        # 基础指标
        metrics['mae'] = np.mean(np.abs(targets - predictions))
        metrics['mse'] = np.mean((targets - predictions) ** 2)
        metrics['rmse'] = np.sqrt(metrics['mse'])

        # R² 决定系数
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # 调整R²
        n = len(targets)
        p = 1  # 假设单变量回归
        metrics['adj_r2'] = 1 - (1 - metrics['r2']) * (n - 1) / (n - p - 1) if n > p + 1 else 0

        # 平均绝对百分比误差 MAPE
        non_zero_mask = targets != 0
        if np.any(non_zero_mask):
            metrics['mape'] = np.mean(
                np.abs((targets[non_zero_mask] - predictions[non_zero_mask]) / targets[non_zero_mask])) * 100
        else:
            metrics['mape'] = float('inf')

        # 最大绝对误差
        metrics['max_error'] = np.max(np.abs(targets - predictions))

        # 平均偏差
        metrics['mean_bias'] = np.mean(predictions - targets)

        # 平均绝对偏差
        metrics['mad'] = np.mean(np.abs(predictions - targets))

        # 相关系数
        correlation_matrix = np.corrcoef(targets, predictions)
        metrics['correlation'] = correlation_matrix[0, 1] if correlation_matrix.shape == (2, 2) else 0

        # 标准化均方根误差
        metrics['nrmse'] = metrics['rmse'] / (np.max(targets) - np.min(targets)) if np.max(targets) != np.min(
            targets) else 0

        # 平均绝对标准化误差
        mean_target = np.mean(targets)
        metrics['mase'] = metrics['mae'] / mean_target if mean_target != 0 else float('inf')

        return metrics


class OptimizedTrainer:
    def __init__(self, model, config, train_loader=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.config = config

        # 优化器 - 使用AdamW
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8
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
                pct_start=0.1,  # 10%用于预热
                anneal_strategy='cos'
            )
        else:
            # 备用调度器 - ReduceLROnPlateau
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.8,
                patience=config['scheduler_patience'],
                min_lr=1e-7,
                verbose=True
            )

        self.criterion = nn.L1Loss()

        # 混合精度
        if config['use_amp']:
            self.scaler = GradScaler()

        self.best_val_loss = float('inf')
        self.early_stop_counter = 0

    def get_gpu_memory(self):
        """获取GPU内存使用情况"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 ** 3
        return 0

    def train_epoch(self, loader):
        self.model.train()
        epoch_loss = 0
        num_batches = 0
        accumulated_loss = 0

        # 清理显存
        torch.cuda.empty_cache()

        for batch_idx, batch in enumerate(loader):
            try:
                batch = batch.to(self.device, non_blocking=True)

                # 检查批次大小，如果太大就跳过
                if hasattr(batch, 'num_nodes') and batch.num_nodes > 5000:
                    print(f"⚠️ 跳过大批次: {batch.num_nodes} 节点")
                    continue

                if CONFIG['use_amp']:
                    with autocast():
                        pred = self.model(batch)
                        loss = self.criterion(pred, batch.y[:, CONFIG['target_index']])
                        loss = loss / CONFIG['accumulation_steps']  # 梯度累积

                    self.scaler.scale(loss).backward()
                    accumulated_loss += loss.item()

                    # 梯度累积步骤
                    if (batch_idx + 1) % CONFIG['accumulation_steps'] == 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()

                        # OneCycleLR步进（每个有效batch后）
                        if hasattr(self.scheduler, 'step') and not isinstance(self.scheduler,
                                                                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step()

                        epoch_loss += accumulated_loss
                        accumulated_loss = 0
                        num_batches += 1

                        # 定期清理显存
                        if num_batches % 10 == 0:
                            torch.cuda.empty_cache()
                else:
                    # 常规训练
                    pred = self.model(batch)
                    loss = self.criterion(pred, batch.y[:, CONFIG['target_index']])
                    loss = loss / CONFIG['accumulation_steps']

                    loss.backward()
                    accumulated_loss += loss.item()

                    if (batch_idx + 1) % CONFIG['accumulation_steps'] == 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        # OneCycleLR步进（每个有效batch后）
                        if hasattr(self.scheduler, 'step') and not isinstance(self.scheduler,
                                                                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step()

                        epoch_loss += accumulated_loss
                        accumulated_loss = 0
                        num_batches += 1

                        # 定期清理显存
                        if num_batches % 10 == 0:
                            torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️ 显存不足，跳过批次 {batch_idx}, 错误: {e}")
                    torch.cuda.empty_cache()  # 清理显存
                    continue
                else:
                    print(f"⚠️ 训练批次错误: {e}")
                    continue
            except Exception as e:
                print(f"⚠️ 训练批次错误: {e}")
                continue

        return epoch_loss / num_batches if num_batches > 0 else float('inf')

    def evaluate(self, loader, collect_predictions=False):
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

                    # 计算各种损失
                    total_mae += F.l1_loss(pred, target, reduction='sum').item()
                    total_mse += F.mse_loss(pred, target, reduction='sum').item()
                    total_samples += target.size(0)

                    if collect_predictions:
                        predictions.extend(pred.cpu().numpy())
                        targets.extend(target.cpu().numpy())

                except Exception as e:
                    continue

        # 计算平均指标
        avg_mae = total_mae / total_samples if total_samples > 0 else float('inf')
        avg_mse = total_mse / total_samples if total_samples > 0 else float('inf')

        # 计算额外指标
        metrics = {'mae': avg_mae, 'mse': avg_mse}

        if len(predictions) > 0 and len(targets) > 0:
            predictions = np.array(predictions)
            targets = np.array(targets)

            # RMSE
            metrics['rmse'] = np.sqrt(avg_mse)

            # R² 决定系数
            ss_res = np.sum((targets - predictions) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # 平均绝对百分比误差 MAPE
            non_zero_mask = targets != 0
            if np.any(non_zero_mask):
                metrics['mape'] = np.mean(
                    np.abs((targets[non_zero_mask] - predictions[non_zero_mask]) / targets[non_zero_mask])) * 100
            else:
                metrics['mape'] = float('inf')

            # 最大绝对误差
            metrics['max_error'] = np.max(np.abs(targets - predictions))

            # 平均偏差
            metrics['mean_bias'] = np.mean(predictions - targets)

        return metrics, predictions, targets

    def run(self, train_loader, val_loader):
        print("Starting optimized SchNet training...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # 如果使用了ReduceLROnPlateau，重新初始化为OneCycleLR
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            steps_per_epoch = len(train_loader) // CONFIG['accumulation_steps']
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=CONFIG['learning_rate'],
                epochs=CONFIG['epochs'],
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1,
                anneal_strategy='cos'
            )
            print(f"Initialized OneCycleLR with {steps_per_epoch} steps per epoch")

        # 添加表头
        print("\n" + "=" * 120)
        print(
            f"{'Epoch':>5} | {'Train MAE':>10} | {'Train MSE':>10} | {'Val MAE':>10} | {'Val MSE':>10} | {'Val RMSE':>10} | {'Val R²':>8} | {'LR':>10} | {'Time':>8} | {'GPU':>8}")
        print("=" * 120)

        for epoch in range(CONFIG['epochs']):
            epoch_start_time = time.time()

            # 训练
            train_loss = self.train_epoch(train_loader)

            # 验证
            collect_pred = (epoch % 10 == 0)
            val_metrics, predictions, targets = self.evaluate(val_loader, collect_predictions=collect_pred)

            # 计算训练集指标（每10个epoch一次，避免影响训练速度）
            if epoch % 10 == 0:
                train_metrics, _, _ = self.evaluate(train_loader, collect_predictions=False)
            else:
                train_metrics = {'mae': train_loss, 'mse': train_loss ** 2}  # 近似值

            # 学习率调度
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['mae'])
            else:
                # OneCycleLR在每个batch后更新，这里不需要step
                pass

            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            gpu_mem = self.get_gpu_memory()

            # Early stopping (基于验证MAE)
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
                    'train_metrics': train_metrics
                }, 'best_model.pth')
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= CONFIG['early_stop_patience']:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

            # 输出详细信息
            if (epoch + 1) % 5 == 0:
                print(f"{epoch + 1:5d} | {train_metrics['mae']:10.6f} | {train_metrics['mse']:10.6f} | "
                      f"{val_metrics['mae']:10.6f} | {val_metrics['mse']:10.6f} | {val_metrics.get('rmse', 0):10.6f} | "
                      f"{val_metrics.get('r2', 0):8.4f} | {current_lr:10.2e} | {epoch_time:8.1f}s | {gpu_mem:8.2f}GB")

                # 每20个epoch输出额外指标
                if (epoch + 1) % 20 == 0 and 'mape' in val_metrics:
                    print(f"      Additional metrics - MAPE: {val_metrics['mape']:.2f}%, "
                          f"Max Error: {val_metrics['max_error']:.6f}, "
                          f"Mean Bias: {val_metrics['mean_bias']:+.6f}")

        return self.best_val_loss


# ########################
# # 主程序
# ########################
if __name__ == "__main__":
    print("=" * 80)
    print("High-Performance SchNet Training System (Memory Optimized)")
    print(f"PyTorch: {torch.__version__} | Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name()} ({torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB)")
        # 清理显存
        torch.cuda.empty_cache()
        # 显示当前显存使用
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")
        print(f"Current GPU memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB")
    print("=" * 80)

    try:
        # 数据准备
        processor = OptimizedQM9DataProcessor(CONFIG)
        train_loader, val_loader, test_loader = processor.get_data_loaders()

        # 优化模型
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

        print(f"\nModel Configuration:")
        print(f"  Hidden dim: {CONFIG['hidden_dim']}")
        print(f"  Interactions: {CONFIG['num_interactions']}")
        print(f"  Max neighbors: {CONFIG['base_max_neighbors']}")
        print(f"  Batch size: {CONFIG['batch_size']}")
        print(f"  Accumulation steps: {CONFIG['accumulation_steps']}")
        print(f"  Effective batch size: {CONFIG['batch_size'] * CONFIG['accumulation_steps']}")

        # 开始训练
        trainer = OptimizedTrainer(model, CONFIG, train_loader)  # 传入train_loader
        best_loss = trainer.run(train_loader, val_loader)

        # 加载最佳模型进行测试
        print("\nLoading best model for final evaluation...")
        checkpoint = torch.load('best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])

        # 完整评估
        print("\nFinal Model Evaluation:")
        print("=" * 80)

        # 训练集评估
        train_metrics, _, _ = trainer.evaluate(train_loader, collect_predictions=True)
        print("Training Set Metrics:")
        print(f"  MAE:        {train_metrics['mae']:.6f}")
        print(f"  MSE:        {train_metrics['mse']:.6f}")
        print(f"  RMSE:       {train_metrics.get('rmse', 0):.6f}")
        print(f"  R²:         {train_metrics.get('r2', 0):.6f}")
        print(f"  MAPE:       {train_metrics.get('mape', 0):.2f}%")
        print(f"  Max Error:  {train_metrics.get('max_error', 0):.6f}")
        print(f"  Mean Bias:  {train_metrics.get('mean_bias', 0):+.6f}")

        # 验证集评估
        val_metrics, _, _ = trainer.evaluate(val_loader, collect_predictions=True)
        print("\nValidation Set Metrics:")
        print(f"  MAE:        {val_metrics['mae']:.6f}")
        print(f"  MSE:        {val_metrics['mse']:.6f}")
        print(f"  RMSE:       {val_metrics.get('rmse', 0):.6f}")
        print(f"  R²:         {val_metrics.get('r2', 0):.6f}")
        print(f"  MAPE:       {val_metrics.get('mape', 0):.2f}%")
        print(f"  Max Error:  {val_metrics.get('max_error', 0):.6f}")
        print(f"  Mean Bias:  {val_metrics.get('mean_bias', 0):+.6f}")

        # 测试集评估
        test_metrics, _, _ = trainer.evaluate(test_loader, collect_predictions=True)
        print("\nTest Set Metrics:")
        print(f"  MAE:        {test_metrics['mae']:.6f}")
        print(f"  MSE:        {test_metrics['mse']:.6f}")
        print(f"  RMSE:       {test_metrics.get('rmse', 0):.6f}")
        print(f"  R²:         {test_metrics.get('r2', 0):.6f}")
        print(f"  MAPE:       {test_metrics.get('mape', 0):.2f}%")
        print(f"  Max Error:  {test_metrics.get('max_error', 0):.6f}")
        print(f"  Mean Bias:  {test_metrics.get('mean_bias', 0):+.6f}")

        print(f"\nBest epoch: {checkpoint['epoch'] + 1}")
        print(f"Model saved as: best_model.pth")
        print("=" * 80)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()