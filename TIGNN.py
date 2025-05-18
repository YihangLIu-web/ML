import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.datasets import QM9
from torch_geometric.data import Batch, Data, DataLoader as PyGDataLoader
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import os
import urllib.request
import shutil
from sklearn.manifold import TSNE

# ########################
# # Configuration Section
# ########################

# 检查并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    print(f"可用GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU索引: {torch.cuda.current_device()}")
else:
    print("使用CPU进行训练")

CONFIG = {
    # Dataset configuration
    "dataset_path": '/home/user/ljrFiles/1/Q9',  # 使用已下载的QM9数据集路径
    "target_index": 12,  # 预测吉布斯自由能 (Gibbs free energy)
    "split_ratio": [0.9, 0.05, 0.05],  # train/val/test split
    "max_smiles_length": 100,  # Max length for SMILES strings

    # Model architecture
    "embedding_dim": 128,  # SMILES embedding dimension
    "graph_hidden_dim": 256,  # Graph model hidden dimension
    "hidden_dim": 256,  # Hidden layer dimension
    "fusion_dim": 384,  # Fusion dimension
    "smiles_num_heads": 8,  # SMILES Transformer heads
    "graph_num_heads": 16,  # Graph GAT heads
    "smiles_num_layers": 8,  # SMILES Transformer layers
    "graph_num_layers": 8,  # Graph GAT layers
    "dropout_rate": 0.2,  # Dropout probability
    "contrastive_temp": 0.07,  # Temperature for contrastive loss
    "projection_dim": 128,  # Projection dimension for contrastive learning

    # Training parameters
    "batch_size": 1024,
    "learning_rate": 0.001,  # 初始学习率
    "min_lr": 1e-6,         # 最小学习率
    "lr_patience": 30,      # 学习率调整的耐心值
    "lr_factor": 0.5,       # 学习率调整因子
    "warmup_epochs": 200,  # Contrastive learning warmup epochs
    "prediction_epochs": 400,  # Main task prediction epochs
    "early_stop_patience": 50,  # Early stopping patience
    "alpha": 0.5,  # Weight for contrastive loss during joint training

    # Device configuration
    "device": device,  # 使用检测到的设备
    "random_seed": 42  # Random seed
}

# Set random seeds for reproducibility
torch.manual_seed(CONFIG["random_seed"])
np.random.seed(CONFIG["random_seed"])


# ########################
# SMILES Processing Classes
# ########################
class SMILESVocabulary:
    """Vocabulary for SMILES tokenization"""

    def __init__(self, smiles_list):
        # Special tokens
        self.special_tokens = ['<PAD>', '<CLS>', '<UNK>', '<MASK>']  # 添加MASK token

        # Collect all unique characters
        all_chars = set()
        for smiles in smiles_list:
            all_chars.update(list(smiles))
        all_chars = sorted(list(all_chars))  # Sort for consistency

        # Create mapping dictionaries
        self.char_to_index = {token: i for i, token in enumerate(self.special_tokens)}
        self.char_to_index.update({char: i + len(self.special_tokens) for i, char in enumerate(all_chars)})

        # Create reverse mapping
        self.index_to_char = {v: k for k, v in self.char_to_index.items()}
        self.vocab_size = len(self.char_to_index)

        # Constants
        self.pad_idx = self.char_to_index['<PAD>']
        self.cls_idx = self.char_to_index['<CLS>']
        self.unk_idx = self.char_to_index['<UNK>']
        self.mask_idx = self.char_to_index['<MASK>']  # 添加MASK token索引

        print(f"Vocabulary size: {self.vocab_size}")

    def __len__(self):
        return self.vocab_size

    def tokenize(self, smiles):
        """Convert SMILES string to token indices"""
        return [self.char_to_index.get(c, self.unk_idx) for c in smiles]

    def get_padding_mask(self, indices):
        """Create padding mask for transformer attention"""
        return indices == self.pad_idx


# ########################
# # Multimodal Dataset
# ########################
class QM9MultimodalDataset(Dataset):
    """Dataset that provides both graph and SMILES representations"""

    def __init__(self, dataset, indices, vocabulary, target_idx=0, max_length=100, device=None):
        self.dataset = dataset
        self.indices = indices
        self.vocab = vocabulary
        self.target_idx = target_idx
        self.max_length = max_length
        self.device = device

        # 添加数据检查
        sample_data = dataset[0]
        if not hasattr(sample_data, 'y'):
            raise ValueError("Dataset does not have 'y' attribute")
        if isinstance(sample_data.y, torch.Tensor):
            print(f"Target shape: {sample_data.y.shape}")
            # 检查目标索引是否在范围内
            if sample_data.y.dim() == 2:  # 如果是2D张量 [1, 19]
                if target_idx >= sample_data.y.shape[1]:
                    raise ValueError(f"target_idx {target_idx} is out of bounds for target shape {sample_data.y.shape}")
            else:  # 如果是1D张量 [19]
                if target_idx >= sample_data.y.shape[0]:
                    raise ValueError(f"target_idx {target_idx} is out of bounds for target shape {sample_data.y.shape}")
            print(f"Using target property at index {target_idx}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the QM9 data point
        real_idx = self.indices[idx]
        data_point = self.dataset[real_idx]
        smiles = self.dataset.data.smiles[real_idx]

        # 安全地获取目标值
        if isinstance(data_point.y, torch.Tensor):
            if data_point.y.dim() == 0:
                target = data_point.y.item()
            elif data_point.y.dim() == 2:  # 如果是2D张量 [1, 19]
                target = data_point.y[0, self.target_idx].item()
            else:  # 如果是1D张量 [19]
                target = data_point.y[self.target_idx].item()
        else:
            target = float(data_point.y)  # 如果是标量值

        # Process SMILES
        token_ids = self.vocab.tokenize(smiles)

        # Truncate/pad as needed
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            padding = [self.vocab.pad_idx] * (self.max_length - len(token_ids))
            token_ids = token_ids + padding

        # Create padding mask for transformer
        padding_mask = [1 if tid == self.vocab.pad_idx else 0 for tid in token_ids]

        return {
            'smiles': smiles,
            'smiles_ids': torch.LongTensor(token_ids),
            'padding_mask': torch.BoolTensor(padding_mask),
            'graph': data_point,
            'target': torch.FloatTensor([target])
        }


# ########################
# # SMILES Transformer Components
# ########################
class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ########################
# # Model Architecture Components
# ########################
class SMILESEncoder(nn.Module):
    """SMILES sequence encoder using Transformer"""

    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=4,
                 dim_feedforward=512, dropout=0.1, max_seq_len=100):
        super().__init__()
        self.d_model = d_model

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)  # 输出维度等于词表大小
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_key_padding_mask=None, output_hidden_states=False):
        # src: [batch_size, seq_len]
        # src_key_padding_mask: [batch_size, seq_len], True for positions to mask

        # Embedding and positional encoding
        src = self.embedding(src) * math.sqrt(self.d_model)  # [batch_size, seq_len, d_model]
        src = self.pos_encoder(src)

        # Transformer encoding
        if src_key_padding_mask is None:
            # Create padding mask if not provided
            src_key_padding_mask = (src == 0)[:, :, 0]  # [batch_size, seq_len]

        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

        # Global pooling (sequence average)
        # Ensure padding positions are not counted
        mask = ~src_key_padding_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        mask_expanded = mask.expand(-1, -1, self.d_model).float()  # [batch_size, seq_len, d_model]
        sum_embeddings = torch.sum(output * mask_expanded, dim=1)  # [batch_size, d_model]
        sum_mask = torch.sum(mask_expanded, dim=1)  # [batch_size, d_model]
        pooled_output = sum_embeddings / (sum_mask + 1e-9)  # [batch_size, d_model]

        if output_hidden_states:
            return self.layer_norm(pooled_output), output
        else:
            return self.layer_norm(pooled_output)

    def mlm_forward(self, src, src_key_padding_mask=None):
        """前向传播并获取MLM预测"""
        pooled_output, sequence_output = self.forward(src, src_key_padding_mask, output_hidden_states=True)
        mlm_logits = self.mlm_head(sequence_output)
        return pooled_output, mlm_logits


class ResidualGATBlock(nn.Module):
    """Residual GAT block with layer normalization"""

    def __init__(self, in_dim, hidden_dim, heads, dropout):
        super().__init__()
        self.gat = GATv2Conv(in_dim, hidden_dim // heads, heads=heads)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # Residual linear projection if dimensions don't match
        self.residual = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else None

    def forward(self, x, edge_index):
        # GAT layer with residual connection
        residual = x
        x = self.gat(x, edge_index)
        x = F.gelu(x)  # Using GELU activation
        x = self.dropout(x)

        # Apply residual connection with dim matching if needed
        if self.residual is not None:
            residual = self.residual(residual)
        x = x + residual

        # Layer normalization
        return self.norm(x)


class GraphEncoder(nn.Module):
    """Graph encoder using stacked GAT layers with residual connections"""

    def __init__(self, node_dim, hidden_dim, num_layers, heads, dropout):
        super().__init__()
        self.input_projection = nn.Linear(node_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Stack of GAT blocks
        self.gat_blocks = nn.ModuleList()
        for i in range(num_layers):
            layer_in_dim = hidden_dim
            self.gat_blocks.append(ResidualGATBlock(layer_in_dim, hidden_dim, heads, dropout))

        # Final layer normalization
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, batch):
        # Initial projection
        x = self.input_norm(F.gelu(self.input_projection(x)))

        # Process through GAT blocks
        for block in self.gat_blocks:
            x = block(x, edge_index)

        # Global mean pooling
        pooled = global_mean_pool(x, batch)

        return self.final_norm(pooled)


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)


class MultimodalFusion(nn.Module):
    """Fusion module for combining graph and SMILES representations"""

    def __init__(self, graph_dim, smiles_dim, fusion_dim, dropout=0.1):
        super().__init__()
        self.graph_projection = nn.Linear(graph_dim, fusion_dim)
        self.smiles_projection = nn.Linear(smiles_dim, fusion_dim)

        # Cross-attention weights
        self.graph_attn = nn.Parameter(torch.ones(1))
        self.smiles_attn = nn.Parameter(torch.ones(1))

        # Gate mechanism
        self.gate = nn.Sequential(
            nn.Linear(graph_dim + smiles_dim, fusion_dim),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_feat, smiles_feat):
        # Project to common space
        graph_proj = self.graph_projection(graph_feat)
        smiles_proj = self.smiles_projection(smiles_feat)

        # Compute dynamic weights
        graph_weight = F.softplus(self.graph_attn)
        smiles_weight = F.softplus(self.smiles_attn)

        # Gate mechanism
        gate_input = torch.cat([graph_feat, smiles_feat], dim=1)
        gate = self.gate(gate_input)

        # Weighted fusion with gate
        fused = gate * (graph_weight * graph_proj) + (1 - gate) * (smiles_weight * smiles_proj)

        return self.dropout(self.norm(fused))


class QM9MultimodalPredictor(nn.Module):
    """Full multimodal model for QM9 property prediction"""

    def __init__(self, config, vocab_size, node_dim):
        super().__init__()
        # SMILES encoder
        self.smiles_encoder = SMILESEncoder(
            vocab_size=vocab_size,
            d_model=config["embedding_dim"],
            nhead=config["smiles_num_heads"],
            num_layers=config["smiles_num_layers"],
            dim_feedforward=config["hidden_dim"],
            dropout=config["dropout_rate"],
            max_seq_len=config["max_smiles_length"]
        )

        # Graph encoder
        self.graph_encoder = GraphEncoder(
            node_dim=node_dim,
            hidden_dim=config["graph_hidden_dim"],
            num_layers=config["graph_num_layers"],
            heads=config["graph_num_heads"],
            dropout=config["dropout_rate"]
        )

        # Projection heads for contrastive learning
        self.smiles_projector = ProjectionHead(
            input_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            output_dim=config["projection_dim"]
        )

        self.graph_projector = ProjectionHead(
            input_dim=config["graph_hidden_dim"],
            hidden_dim=config["hidden_dim"],
            output_dim=config["projection_dim"]
        )

        # Fusion module
        self.fusion = MultimodalFusion(
            graph_dim=config["graph_hidden_dim"],
            smiles_dim=config["embedding_dim"],
            fusion_dim=config["fusion_dim"],
            dropout=config["dropout_rate"]
        )

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(config["fusion_dim"], config["hidden_dim"]),
            nn.LayerNorm(config["hidden_dim"]),
            nn.GELU(),
            nn.Dropout(config["dropout_rate"]),
            nn.Linear(config["hidden_dim"], 1)
        )

        # Single modality prediction heads (auxiliary)
        self.graph_predictor = nn.Linear(config["graph_hidden_dim"], 1)
        self.smiles_predictor = nn.Linear(config["embedding_dim"], 1)

    def encode_smiles(self, smiles_ids, padding_mask):
        """Encode SMILES sequence"""
        return self.smiles_encoder(smiles_ids, padding_mask)

    def encode_graph(self, x, edge_index, batch):
        """Encode molecular graph"""
        return self.graph_encoder(x, edge_index, batch)

    def project_embeddings(self, smiles_emb, graph_emb):
        """Project embeddings for contrastive learning"""
        smiles_proj = F.normalize(self.smiles_projector(smiles_emb), dim=1)
        graph_proj = F.normalize(self.graph_projector(graph_emb), dim=1)
        return smiles_proj, graph_proj

    def forward(self, smiles_ids, padding_mask, x, edge_index, batch):
        """Full forward pass"""
        # Encode both modalities
        smiles_feat = self.encode_smiles(smiles_ids, padding_mask)
        graph_feat = self.encode_graph(x, edge_index, batch)

        # Get contrastive projections
        smiles_proj, graph_proj = self.project_embeddings(smiles_feat, graph_feat)

        # Fuse representations
        fused = self.fusion(graph_feat, smiles_feat)

        # Get predictions from each branch
        main_pred = self.predictor(fused).squeeze(-1)
        graph_pred = self.graph_predictor(graph_feat).squeeze(-1)
        smiles_pred = self.smiles_predictor(smiles_feat).squeeze(-1)

        return {
            'main_pred': main_pred,
            'graph_pred': graph_pred,
            'smiles_pred': smiles_pred,
            'smiles_proj': smiles_proj,
            'graph_proj': graph_proj,
            'smiles_feat': smiles_feat,
            'graph_feat': graph_feat,
            'fused_feat': fused
        }


# ########################
# # Loss Functions
# ########################
class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, proj_1, proj_2):
        """
        proj_1, proj_2: normalized projection vectors [batch_size, projection_dim]
        """
        batch_size = proj_1.shape[0]
        device = proj_1.device

        # Compute cosine similarity
        # (equivalent to matrix multiplication since vectors are normalized)
        sim_matrix = torch.mm(proj_1, proj_2.T) / self.temperature  # [batch_size, batch_size]

        # Labels: diagonal elements (matching pairs)
        labels = torch.arange(batch_size, device=device)

        # Loss computation - symmetric across views
        loss_i = F.cross_entropy(sim_matrix, labels)
        loss_j = F.cross_entropy(sim_matrix.T, labels)

        return (loss_i + loss_j) / 2.0


class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning"""

    def __init__(self, contrastive_temp=0.07, alpha=0.5, warmup_epochs=100):
        super().__init__()
        self.contrastive_loss = ContrastiveLoss(temperature=contrastive_temp)
        self.prediction_loss = nn.MSELoss()
        self.alpha = alpha
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        # 添加可学习的损失权重参数
        self.main_weight = nn.Parameter(torch.ones(1))
        self.graph_weight = nn.Parameter(torch.ones(1))
        self.smiles_weight = nn.Parameter(torch.ones(1))

        # 初始化权重
        with torch.no_grad():
            self.main_weight.fill_(1.0)
            self.graph_weight.fill_(0.3)
            self.smiles_weight.fill_(0.3)

    def to(self, device):
        """重写to方法，确保权重参数被移动到正确的设备上"""
        super().to(device)
        self.main_weight = self.main_weight.to(device)
        self.graph_weight = self.graph_weight.to(device)
        self.smiles_weight = self.smiles_weight.to(device)
        return self

    def set_epoch(self, epoch):
        """设置当前epoch，用于调整对比损失权重"""
        self.current_epoch = epoch

    def get_contrastive_weight(self):
        """根据训练阶段调整对比损失权重"""
        if self.current_epoch < self.warmup_epochs:
            # 对比学习阶段，权重为1
            return 1.0
        else:
            # 正常训练阶段，权重逐渐减小
            progress = (self.current_epoch - self.warmup_epochs) / self.warmup_epochs
            return max(0.0, 1.0 - progress)

    def forward(self, outputs, targets):
        # 获取当前对比损失权重
        contrastive_weight = self.get_contrastive_weight()

        # 对比损失
        contrast_loss = self.contrastive_loss(outputs['smiles_proj'], outputs['graph_proj'])

        # 预测损失
        main_pred_loss = self.prediction_loss(outputs['main_pred'], targets)
        graph_pred_loss = self.prediction_loss(outputs['graph_pred'], targets)
        smiles_pred_loss = self.prediction_loss(outputs['smiles_pred'], targets)

        # 使用可学习的权重组合预测损失
        # 使用softplus确保权重为正
        main_w = F.softplus(self.main_weight)
        graph_w = F.softplus(self.graph_weight)
        smiles_w = F.softplus(self.smiles_weight)

        # 归一化权重
        total_w = main_w + graph_w + smiles_w
        main_w = main_w / total_w
        graph_w = graph_w / total_w
        smiles_w = smiles_w / total_w

        # 组合预测损失
        pred_loss = main_w * main_pred_loss + graph_w * graph_pred_loss + smiles_w * smiles_pred_loss

        # 根据训练阶段调整总损失
        total_loss = (1 - contrastive_weight) * pred_loss + contrastive_weight * contrast_loss

        return {
            'total': total_loss,
            'contrast': contrast_loss,
            'main_pred': main_pred_loss,
            'graph_pred': graph_pred_loss,
            'smiles_pred': smiles_pred_loss,
            'pred_total': pred_loss,
            'contrastive_weight': contrastive_weight,
            'main_weight': main_w.item(),
            'graph_weight': graph_w.item(),
            'smiles_weight': smiles_w.item()
        }


# ########################
# # Training & Evaluation Functions
# ########################
def prepare_batch_for_model(batch, device):
    """Prepare batch data for model input"""
    # Prepare SMILES data
    smiles_ids = batch['smiles_ids'].to(device)
    padding_mask = batch['padding_mask'].to(device)

    # Prepare graph data - build PyG batch
    graphs = batch['graph']
    target = batch['target'].to(device).squeeze()

    # Move graph data to device
    x = graphs.x.to(device)
    edge_index = graphs.edge_index.to(device)
    batch_idx = graphs.batch.to(device)

    return smiles_ids, padding_mask, x, edge_index, batch_idx, target


def train_epoch(model, loader, criterion, optimizer, device, mode='joint', epoch=None):
    """Train for one epoch"""
    model.train()
    epoch_losses = {
        'total': 0.0,
        'contrast': 0.0,
        'main_pred': 0.0,
        'graph_pred': 0.0,
        'smiles_pred': 0.0,
        'contrastive_weight': 0.0,
        'main_weight': 0.0,
        'graph_weight': 0.0,
        'smiles_weight': 0.0
    }

    # 设置当前epoch
    if epoch is not None:
        criterion.set_epoch(epoch)

    for batch in tqdm(loader, desc=f"Training ({mode})"):
        # 验证batch对齐
        assert len(batch['smiles']) == batch['graph'].num_graphs, "Batch size mismatch!"

        # Prepare data
        smiles_ids, padding_mask, x, edge_index, batch_idx, targets = prepare_batch_for_model(batch, device)

        # Forward pass
        outputs = model(smiles_ids, padding_mask, x, edge_index, batch_idx)

        # Compute loss based on training mode
        if mode == 'contrastive':
            # Only contrastive loss
            loss = criterion.contrastive_loss(outputs['smiles_proj'], outputs['graph_proj'])
            losses = {'total': loss, 'contrast': loss}
        else:
            # Full loss
            losses = criterion(outputs, targets)
            loss = losses['total']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Update losses
        for k in losses:
            if k in epoch_losses:
                # 处理不同类型的损失值
                if isinstance(losses[k], torch.Tensor):
                    epoch_losses[k] += losses[k].item()
                else:
                    epoch_losses[k] += float(losses[k])

    # Average losses
    for k in epoch_losses:
        epoch_losses[k] /= len(loader)

    return epoch_losses


def plot_parity(y_true, y_pred, title="Parity Plot", save_path="parity_plot.png"):
    """
    绘制Parity plot
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        title: 图表标题
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 10))
    
    # 计算相关系数
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    
    # 计算RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # 绘制散点图
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    
    # 绘制对角线
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    
    # 添加标签和标题
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{title}\nCorrelation: {correlation:.4f}, RMSE: {rmse:.4f}')
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_distribution(y_true, y_pred, title="Error Distribution", save_path="error_distribution.png"):
    """
    绘制误差分布图（小提琴图）
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        title: 图表标题
        save_path: 保存路径
    """
    # 计算相对误差
    relative_errors = (y_pred - y_true) / y_true * 100  # 转换为百分比
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 创建小提琴图
    violin = plt.violinplot(relative_errors, showmeans=True, showmedians=True)
    
    # 设置小提琴图颜色
    for pc in violin['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_alpha(0.7)
    
    # 添加均值点和中位数线
    plt.plot([1], [np.mean(relative_errors)], 'ko', label='Mean')
    plt.plot([1], [np.median(relative_errors)], 'wo', label='Median')
    
    # 计算统计信息
    mean_error = np.mean(relative_errors)
    median_error = np.median(relative_errors)
    std_error = np.std(relative_errors)
    q1 = np.percentile(relative_errors, 25)
    q3 = np.percentile(relative_errors, 75)
    iqr = q3 - q1
    
    # 添加统计信息文本
    stats_text = f'Mean: {mean_error:.2f}%\n'
    stats_text += f'Median: {median_error:.2f}%\n'
    stats_text += f'Std: {std_error:.2f}%\n'
    stats_text += f'IQR: {iqr:.2f}%'
    
    plt.text(1.1, plt.ylim()[1]*0.9, stats_text, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # 添加零线
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 设置标签和标题
    plt.xlabel('Relative Error Distribution')
    plt.ylabel('Relative Error (%)')
    plt.title(f'{title}\nDistribution of Prediction Errors')
    
    # 添加图例
    plt.legend()
    
    # 设置y轴范围，去除极端值的影响
    plt.ylim([np.percentile(relative_errors, 1), np.percentile(relative_errors, 99)])
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0  # 添加MAE累计值
    all_preds = []
    all_targets = []
    contrastive_similarities = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            # Prepare data
            smiles_ids, padding_mask, x, edge_index, batch_idx, targets = prepare_batch_for_model(batch, device)

            # Forward pass
            outputs = model(smiles_ids, padding_mask, x, edge_index, batch_idx)

            # Main prediction loss
            loss = criterion(outputs, targets)['total']
            total_loss += loss.item()

            # MSE for RMSE calculation
            mse = F.mse_loss(outputs['main_pred'], targets)
            total_mse += mse.item()

            # MAE calculation
            mae = F.l1_loss(outputs['main_pred'], targets)
            total_mae += mae.item()

            # 计算对比学习相似度
            similarity = F.cosine_similarity(outputs['smiles_proj'], outputs['graph_proj'])
            contrastive_similarities.extend(similarity.cpu().numpy())

            # Save predictions and targets for correlation metrics
            all_preds.extend(outputs['main_pred'].cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Average loss and calculate RMSE
    avg_loss = total_loss / len(loader)
    rmse = np.sqrt(total_mse / len(loader))
    mae = total_mae / len(loader)  # 计算平均MAE

    # Calculate correlation
    correlation = np.corrcoef(all_preds, all_targets)[0, 1]

    # 计算R²
    ss_total = np.sum((all_targets - np.mean(all_targets)) ** 2)
    ss_residual = np.sum((all_targets - all_preds) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    # 计算平均对比学习相似度
    avg_similarity = np.mean(contrastive_similarities)

    # 绘制Parity plot
    plot_parity(
        all_targets,
        all_preds,
        title="Model Predictions vs True Values",
        save_path="parity_plot.png"
    )

    # 绘制误差分布图
    plot_error_distribution(
        all_targets,
        all_preds,
        title="Model Prediction Error Distribution",
        save_path="error_distribution.png"
    )

    return {
        'loss': avg_loss,
        'rmse': rmse,
        'mae': mae,  # 添加MAE
        'r2': r2,    # 添加R²
        'correlation': correlation,
        'contrastive_similarity': avg_similarity
    }


# ########################
# # Training Pipeline
# ########################
def visualize_training(history):
    """Visualize training metrics"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))

    # Loss plot
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train')
    ax.plot(history['val_loss'], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSE plot
    ax = axes[0, 1]
    ax.plot(history['train_rmse'], label='Train')
    ax.plot(history['val_rmse'], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Contrastive loss
    ax = axes[1, 0]
    ax.plot(history['contrast_loss'], label='Contrastive Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Contrastive Loss')
    ax.grid(True, alpha=0.3)

    # MLM loss
    ax = axes[1, 1]
    ax.plot(history['mlm_loss'], label='MLM Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('SMILES MLM Loss')
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[2, 0]
    ax.plot(history['learning_rate'], label='Learning Rate')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.show()


def custom_collate(batch):
    # 验证数据对齐
    for item in batch:
        assert item['graph'].smiles == item[
            'smiles'], f"Graph and SMILES data mismatch! Graph: {item['graph'].smiles}, SMILES: {item['smiles']}"

    graphs = Batch.from_data_list([item['graph'] for item in batch])

    return {
        'smiles': [item['smiles'] for item in batch],
        'smiles_ids': torch.stack([item['smiles_ids'] for item in batch]),
        'padding_mask': torch.stack([item['padding_mask'] for item in batch]),
        'graph': graphs,
        'target': torch.stack([item['target'] for item in batch])
    }


class SMILESMasking:
    """用于SMILES序列掩码的类"""

    def __init__(self, vocab, mask_prob=0.15, random_token_prob=0.1, keep_prob=0.1):
        """
        初始化SMILES掩码器

        参数:
            vocab: SMILESVocabulary对象
            mask_prob: 总的掩码比例 (默认15%)
            random_token_prob: 用随机token替换的比例 (占mask_prob的10%)
            keep_prob: 保持原token不变的比例 (占mask_prob的10%)
        """
        self.vocab = vocab
        self.mask_prob = mask_prob
        self.random_token_prob = random_token_prob
        self.keep_prob = keep_prob

        # 获取特殊token索引
        self.pad_idx = vocab.pad_idx
        self.cls_idx = vocab.cls_idx
        self.mask_idx = vocab.vocab_size - 1  # 使用词表最后一个位置作为MASK token

    def apply_masking(self, token_ids):
        """
        对SMILES token序列应用掩码

        参数:
            token_ids: 原始token序列 [batch_size, seq_len]

        返回:
            masked_ids: 掩码后的token序列
            masked_positions: 掩码位置 (1表示被掩码)
            original_ids: 原始token序列 (用于计算loss)
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        # 初始化掩码位置和掩码后的序列
        masked_ids = token_ids.clone()
        masked_positions = torch.zeros_like(token_ids, dtype=torch.bool)

        # 只对非padding和非特殊token应用掩码
        valid_positions = (token_ids != self.pad_idx) & (token_ids != self.cls_idx)

        # 为每个位置生成随机概率
        rand = torch.rand(batch_size, seq_len, device=device)

        # 决定哪些位置需要掩码 (随机选择15%的token)
        mask_candidates = (rand < self.mask_prob) & valid_positions

        # 掩码策略1 (80%): 使用[MASK] token替换
        mask_pos = mask_candidates & (rand < self.mask_prob * 0.8)
        masked_ids[mask_pos] = self.mask_idx

        # 掩码策略2 (10%): 使用随机token替换
        random_pos = mask_candidates & (rand >= self.mask_prob * 0.8) & (rand < self.mask_prob * 0.9)
        random_tokens = torch.randint(low=0, high=self.vocab.vocab_size - 1,  # 减1是为了避免使用MASK token
                                    size=(random_pos.sum(),), device=device)
        masked_ids[random_pos] = random_tokens

        # 掩码策略3 (10%): 保持不变
        # 这部分不需要修改token，但需要预测
        keep_pos = mask_candidates & (rand >= self.mask_prob * 0.9)

        # 合并所有被掩码的位置
        masked_positions = mask_pos | random_pos | keep_pos

        return masked_ids, masked_positions, token_ids


class SMILESMLMHead(nn.Module):
    """SMILES掩码语言模型预测头"""

    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, hidden_states):
        """
        预测被掩码位置的原始token

        参数:
            hidden_states: 序列的隐藏状态 [batch_size, seq_len, hidden_dim]

        返回:
            logits: 预测的token分布 [batch_size, seq_len, vocab_size]
        """
        return self.mlm_head(hidden_states)


def smiles_mlm_loss(logits, original_ids, masked_positions):
    """
    计算SMILES掩码语言模型的损失

    参数:
        logits: 预测的token分布 [batch_size, seq_len, vocab_size]
        original_ids: 原始token序列 [batch_size, seq_len]
        masked_positions: 掩码位置 (1表示被掩码) [batch_size, seq_len]

    返回:
        loss: 交叉熵损失
    """
    # 提取被掩码位置的预测和原始值
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = original_ids.view(-1)
    mask_flat = masked_positions.view(-1)

    # 只计算被掩码位置的损失
    active_logits = logits_flat[mask_flat]
    active_targets = targets_flat[mask_flat]

    loss = F.cross_entropy(active_logits, active_targets)
    return loss


# 扩展SMILES编码器，增加掩码预训练功能
class SMILESEncoderWithMLM(SMILESEncoder):
    """带掩码语言模型的SMILES编码器"""

    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=4,
                 dim_feedforward=512, dropout=0.1, max_seq_len=100, with_mlm=True):
        super().__init__(vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout, max_seq_len)

        # 添加MLM预测头
        if with_mlm:
            self.mlm_head = SMILESMLMHead(d_model, vocab_size)

    def forward(self, src, src_key_padding_mask=None, output_hidden_states=False):
        # 继承原始forward逻辑获取hidden states
        embeddings = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(embeddings)

        if src_key_padding_mask is None:
            src_key_padding_mask = (src == 0)[:, :, 0]

        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

        # 全局平均池化
        mask = ~src_key_padding_mask.unsqueeze(-1)
        mask_expanded = mask.expand(-1, -1, self.d_model).float()
        sum_embeddings = torch.sum(output * mask_expanded, dim=1)
        sum_mask = torch.sum(mask_expanded, dim=1)
        pooled_output = sum_embeddings / (sum_mask + 1e-9)

        if output_hidden_states:
            return self.layer_norm(pooled_output), output
        else:
            return self.layer_norm(pooled_output)

    def mlm_forward(self, src, src_key_padding_mask=None):
        """前向传播并获取MLM预测"""
        pooled_output, sequence_output = self.forward(src, src_key_padding_mask, output_hidden_states=True)
        mlm_logits = self.mlm_head(sequence_output)
        return pooled_output, mlm_logits

def visualize_tsne(model, loader, device, save_path="tsne_visualization.png"):
    """
    使用t-SNE对两个模态的特征进行降维可视化
    
    参数:
        model: 模型
        loader: 数据加载器
        device: 设备
        save_path: 保存路径
    """
    model.eval()
    smiles_features = []
    graph_features = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting features for t-SNE"):
            # 准备数据
            smiles_ids, padding_mask, x, edge_index, batch_idx, _ = prepare_batch_for_model(batch, device)
            
            # 获取特征
            outputs = model(smiles_ids, padding_mask, x, edge_index, batch_idx)
            
            # 收集特征
            smiles_features.append(outputs['smiles_proj'].cpu().numpy())
            graph_features.append(outputs['graph_proj'].cpu().numpy())
    
    # 合并所有批次的特征
    smiles_features = np.vstack(smiles_features)
    graph_features = np.vstack(graph_features)
    
    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=42)
    smiles_2d = tsne.fit_transform(smiles_features)
    graph_2d = tsne.fit_transform(graph_features)
    
    # 创建可视化
    plt.figure(figsize=(12, 6))
    
    # 绘制SMILES特征
    plt.subplot(1, 2, 1)
    plt.scatter(smiles_2d[:, 0], smiles_2d[:, 1], alpha=0.6, label='SMILES')
    plt.title('SMILES Features t-SNE')
    plt.legend()
    
    # 绘制图特征
    plt.subplot(1, 2, 2)
    plt.scatter(graph_2d[:, 0], graph_2d[:, 1], alpha=0.6, label='Graph')
    plt.title('Graph Features t-SNE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 计算特征之间的平均余弦相似度
    similarity = F.cosine_similarity(
        torch.from_numpy(smiles_features).to(device),
        torch.from_numpy(graph_features).to(device)
    ).mean().item()
    
    print(f"Average cosine similarity between modalities: {similarity:.4f}")
    
    return similarity

def train_model(config):
    """Full training pipeline with three-phase training:
    1. SMILES pre-training
    2. Contrastive pre-training
    3. Supervised fine-tuning with property prediction"""
    device = config["device"]

    # 数据集目录存在确保数
    os.makedirs(config["dataset_path"], exist_ok=True)
    os.makedirs(os.path.join(config["dataset_path"], "raw"), exist_ok=True)

    print("Loading QM9 dataset...")
    try:
        dataset = QM9(root=config["dataset_path"])
        print(f"Dataset loaded: {len(dataset)} molecules")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Attempting to clean and redownload dataset...")
        # 清理可能损坏的文件
        dataset_dir = config["dataset_path"]
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
        os.makedirs(dataset_dir)
        os.makedirs(os.path.join(dataset_dir, "raw"))
        # 重新下载
        dataset = QM9(root=config["dataset_path"])
        print(f"Dataset reloaded: {len(dataset)} molecules")

    # Create indices for train/val/test split
    # 仅使用前1000个分子
    max_molecules = 10000
    indices = list(range(min(max_molecules, len(dataset))))
    np.random.shuffle(indices)

    # 用实际要用的分子数来划分
    total = len(indices)
    train_size = int(config["split_ratio"][0] * total)
    val_size = int(config["split_ratio"][1] * total)
    test_size = total - train_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    print(f"Split sizes: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    # Create vocabulary from training SMILES
    train_smiles = [dataset.data.smiles[i] for i in train_indices]
    vocab = SMILESVocabulary(train_smiles)

    # Create datasets
    train_dataset = QM9MultimodalDataset(
        dataset, train_indices, vocab,
        target_idx=config["target_index"],
        max_length=config["max_smiles_length"]
    )

    val_dataset = QM9MultimodalDataset(
        dataset, val_indices, vocab,
        target_idx=config["target_index"],
        max_length=config["max_smiles_length"]
    )

    test_dataset = QM9MultimodalDataset(
        dataset, test_indices, vocab,
        target_idx=config["target_index"],
        max_length=config["max_smiles_length"]
    )

    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=custom_collate
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=2
    )

    # Initialize model with MLM capability
    model = QM9MultimodalPredictor(
        config=config,
        vocab_size=len(vocab),
        node_dim=dataset.num_node_features
    ).to(device)

    # Initialize SMILES masking
    smiles_masker = SMILESMasking(vocab)

    # Initialize criterion and optimizer
    criterion = MultiTaskLoss(
        contrastive_temp=config["contrastive_temp"],
        alpha=config["alpha"]
    ).to(device)  # 确保criterion在正确的设备上

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=1e-5
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_rmse': [],
        'val_rmse': [],
        'contrast_loss': [],
        'mlm_loss': [],
        'learning_rate': []
    }

    # Early stopping parameters
    best_val_rmse = float('inf')
    best_model_state = None
    patience_counter = 0

    # Validation function
    def validate(phase='val'):
        loader = val_loader if phase == 'val' else test_loader
        metrics = evaluate(model, loader, criterion, device)
        print(
            f"{phase.capitalize()} metrics: Loss={metrics['loss']:.6f}, RMSE={metrics['rmse']:.6f}, "
            f"MAE={metrics['mae']:.6f}, R²={metrics['r2']:.4f}, Correlation={metrics['correlation']:.4f}")
        return metrics

    print("Starting three-phase training...")
    total_epochs = config["warmup_epochs"] + config["prediction_epochs"]

    # Phase 0: SMILES Pre-training
    print("\n=== Phase 0: SMILES Pre-training ===")
    mlm_epochs = 100  # SMILES预训练的轮数
    for epoch in range(mlm_epochs):
        model.train()
        total_mlm_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"SMILES Pre-training Epoch {epoch + 1}/{mlm_epochs}"):
            # 准备SMILES数据
            smiles_ids = batch['smiles_ids'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            
            # 应用掩码
            masked_ids, masked_positions, original_ids = smiles_masker.apply_masking(smiles_ids)
            
            # 前向传播
            pooled_output, mlm_logits = model.smiles_encoder.mlm_forward(masked_ids, padding_mask)
            
            # 计算MLM损失
            mlm_loss = smiles_mlm_loss(mlm_logits, original_ids, masked_positions)
            
            # 反向传播
            optimizer.zero_grad()
            mlm_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_mlm_loss += mlm_loss.item()
        
        avg_mlm_loss = total_mlm_loss / len(train_loader)
        history['mlm_loss'].append(avg_mlm_loss)
        
        # 验证但不调整学习率
        val_metrics = validate()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"SMILES Pre-training Epoch {epoch + 1}/{mlm_epochs} - MLM Loss: {avg_mlm_loss:.6f}, LR: {current_lr:.6f}")

    # Phase 1: Contrastive pre-training
    print("\n=== Phase 1: Contrastive Pre-training ===")
    for epoch in range(config["warmup_epochs"]):
        # Training
        epoch_losses = train_epoch(model, train_loader, criterion, optimizer, device, mode='contrastive', epoch=epoch)
        current_lr = optimizer.param_groups[0]['lr']

        # Evaluation
        train_metrics = evaluate(model, train_loader, criterion, device)
        val_metrics = validate()

        # Update history
        history['train_loss'].append(epoch_losses['total'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_rmse'].append(train_metrics['rmse'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['contrast_loss'].append(epoch_losses['contrast'])
        history['learning_rate'].append(current_lr)

        # Print progress
        print(f"Epoch {epoch + 1}/{total_epochs} - LR: {current_lr:.6f}")
        print(f"Train: Loss={epoch_losses['total']:.6f}, RMSE={train_metrics['rmse']:.6f}")
        print(
            f"Valid: Loss={val_metrics['loss']:.6f}, RMSE={val_metrics['rmse']:.6f}, Corr={val_metrics['correlation']:.4f}")

        # Early stopping check
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"New best model: RMSE={best_val_rmse:.6f}")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{config['early_stop_patience']}")

    # 在对比学习阶段结束后立即进行t-SNE可视化
    print("\n=== Visualizing Contrastive Learning Results ===")
    similarity = visualize_tsne(model, val_loader, device, save_path="contrastive_tsne.png")
    print(f"Contrastive learning visualization saved to 'contrastive_tsne.png'")
    print(f"Average cosine similarity: {similarity:.4f}")

    # Phase 2: Supervised fine-tuning
    print("\n=== Phase 2: Supervised Fine-tuning ===")

    # 在监督学习阶段开始时初始化学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config["lr_factor"],
        patience=config["lr_patience"],
        min_lr=config["min_lr"],
        verbose=True
    )

    # Reset early stopping
    best_val_rmse = float('inf')
    patience_counter = 0

    for epoch in range(config["warmup_epochs"], total_epochs):
        # Training
        epoch_losses = train_epoch(model, train_loader, criterion, optimizer, device, mode='joint', epoch=epoch)
        current_lr = optimizer.param_groups[0]['lr']

        # Evaluation
        train_metrics = evaluate(model, train_loader, criterion, device)
        val_metrics = validate()

        # 只在监督学习阶段更新学习率
        scheduler.step(val_metrics['rmse'])

        # Update history
        history['train_loss'].append(epoch_losses['total'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_rmse'].append(train_metrics['rmse'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['contrast_loss'].append(epoch_losses['contrast'])
        history['learning_rate'].append(current_lr)

        # Print progress
        epoch_adjusted = epoch - config["warmup_epochs"] + 1
        print(
            f"Epoch {epoch + 1}/{total_epochs} (Phase 2: {epoch_adjusted}/{config['prediction_epochs']}) - LR: {current_lr:.6f}")
        print(f"Train: Loss={epoch_losses['total']:.6f}, RMSE={train_metrics['rmse']:.6f}")
        print(
            f"Valid: Loss={val_metrics['loss']:.6f}, RMSE={val_metrics['rmse']:.6f}, Corr={val_metrics['correlation']:.4f}")

        # Early stopping check
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"New best model: RMSE={best_val_rmse:.6f}")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{config['early_stop_patience']}")

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation on test set
    print("\n=== Final Evaluation ===")
    test_metrics = validate(phase='test')
    print(
        f"Test metrics: Loss={test_metrics['loss']:.6f}, RMSE={test_metrics['rmse']:.6f}, Correlation={test_metrics['correlation']:.4f}")

    # Visualize training history
    visualize_training(history)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'vocabulary': vocab,
        'test_metrics': test_metrics,
        'history': history
    }, 'qm9_multimodal_model.pt')

    return model, history, test_metrics


# Add a function to make predictions on new molecules
def predict_property(model, smiles, graph, device, vocabulary, max_length=100):
    """Make prediction for a single molecule given its SMILES and graph"""
    model.eval()

    # Process SMILES
    token_ids = vocabulary.tokenize(smiles)

    # Truncate/pad as needed
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    else:
        padding = [vocabulary.pad_idx] * (max_length - len(token_ids))
        token_ids = token_ids + padding

    padding_mask = [1 if tid == vocabulary.pad_idx else 0 for tid in token_ids]

    # Prepare tensor inputs
    smiles_ids = torch.LongTensor([token_ids]).to(device)
    padding_mask = torch.BoolTensor([padding_mask]).to(device)

    # Prepare graph inputs
    x = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    batch = torch.zeros(x.size(0), dtype=torch.long).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(smiles_ids, padding_mask, x, edge_index, batch)
        prediction = outputs['main_pred'].item()

    return prediction


# Main execution
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["random_seed"])

    # Train model
    model, history, test_metrics = train_model(CONFIG)

    print(f"Training completed. Final test RMSE: {test_metrics['rmse']:.6f}")
    print(f"Training visualization saved to 'training_history.png'")
