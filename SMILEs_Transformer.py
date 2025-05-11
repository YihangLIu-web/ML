import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from torch_geometric.datasets import QM9
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm

CONFIG = {
    # 数据集配置
    "dataset_path": '/Users/liuyihang/Desktop/Q9',  # QM9数据集路径
    "target_index": 11,  # 目标属性索引 (G: 吉布斯自由能)
    "split_ratio": [0.8, 0.1, 0.1],  # 训练/验证/测试划分
    "max_length": 100,  # SMILES最大长度

    # 模型架构
    "embedding_dim": 128,  # 嵌入维度
    "hidden_dim": 256,  # 隐藏层维度
    "num_heads": 8,  # Transformer头数
    "num_layers": 4,  # Transformer层数
    "dropout_rate": 0.1,  # Dropout概率

    # 训练参数
    "batch_size": 64,
    "learning_rate": 0.001,
    "epochs": 50,
    "early_stop_patience": 10,  # 早停耐心值

    # 设备配置
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "random_seed": 42  # 随机种子
}

# 设置随机种子
torch.manual_seed(CONFIG["random_seed"])
np.random.seed(CONFIG["random_seed"])

print(f"使用设备: {CONFIG['device']}")

# 1. 构建词汇表
class SMILESVocabulary:
    def __init__(self, smiles_list):
        # 特殊标记
        self.special_tokens = ['<PAD>', '<CLS>', '<UNK>']
        
        # 收集所有唯一字符
        all_chars = set()
        for smiles in smiles_list:
            all_chars.update(list(smiles))
        all_chars = sorted(list(all_chars))  # 排序以保持一致性
        
        # 创建映射字典
        self.char_to_index = {token: i for i, token in enumerate(self.special_tokens)}
        self.char_to_index.update({char: i + len(self.special_tokens) for i, char in enumerate(all_chars)})
        
        # 创建反向映射
        self.index_to_char = {v: k for k, v in self.char_to_index.items()}
        self.vocab_size = len(self.char_to_index)
        
        # 常量
        self.pad_idx = self.char_to_index['<PAD>']
        self.cls_idx = self.char_to_index['<CLS>']
        self.unk_idx = self.char_to_index['<UNK>']
        
        print(f"词汇表大小: {self.vocab_size}")
    
    def __len__(self):
        return self.vocab_size
    
    def tokenize(self, smiles):
        """将SMILES字符串转换为token索引列表"""
        return [self.char_to_index.get(c, self.unk_idx) for c in smiles]
    
    def get_padding_mask(self, indices):
        """创建padding mask，用于transformer注意力机制"""
        return indices == self.pad_idx


# 2. 数据集处理类
class QM9SMILESDataset(Dataset):
    def __init__(self, smiles_list, targets, vocab, max_length=100):
        self.smiles = smiles_list
        self.targets = targets
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        # 获取SMILES字符串和目标值
        smiles = self.smiles[idx]
        target = self.targets[idx]
        
        # 将SMILES转换为索引序列
        indices = self.vocab.tokenize(smiles)
        
        # 截断/填充处理
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            padding = [self.vocab.pad_idx] * (self.max_length - len(indices))
            indices = indices + padding
        
        return {
            'smiles': smiles,
            'input_ids': torch.LongTensor(indices),
            'target': torch.FloatTensor([target])
        }


# 3. 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# 4. Transformer模型
class SMILESTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.1, max_seq_len=100):
        super().__init__()
        self.d_model = d_model
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)  # 0为<PAD>的索引
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 回归头
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        for layer in self.regression_head:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()
    
    def forward(self, src, src_key_padding_mask=None):
        # src: [batch_size, seq_len]
        # src_key_padding_mask: [batch_size, seq_len], True表示需要mask的位置
        
        # 嵌入和位置编码
        src = self.embedding(src) * math.sqrt(self.d_model)  # [batch_size, seq_len, d_model]
        src = self.pos_encoder(src)
        
        # Transformer编码
        if src_key_padding_mask is None:
            # 创建padding mask
            src_key_padding_mask = (src == 0)[:, :, 0]  # [batch_size, seq_len]
        
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        
        # 全局池化 (取序列的平均值)
        # 确保不计算padding位置
        mask = ~src_key_padding_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        mask_expanded = mask.expand(-1, -1, self.d_model).float()  # [batch_size, seq_len, d_model]
        sum_embeddings = torch.sum(output * mask_expanded, dim=1)  # [batch_size, d_model]
        sum_mask = torch.sum(mask_expanded, dim=1)  # [batch_size, d_model]
        pooled_output = sum_embeddings / (sum_mask + 1e-9)  # [batch_size, d_model]
        
        # 回归预测
        output = self.regression_head(pooled_output)
        
        return output.squeeze(-1)  # [batch_size]


# 5. 训练和评估函数
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        targets = batch['target'].to(device)
        
        # 创建padding mask
        padding_mask = (input_ids == 0)  # [batch_size, seq_len]
        
        # 前向传播
        outputs = model(input_ids, padding_mask)
        loss = criterion(outputs, targets.squeeze(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            targets = batch['target'].to(device)
            
            # 创建padding mask
            padding_mask = (input_ids == 0)  # [batch_size, seq_len]
            
            # 前向传播
            outputs = model(input_ids, padding_mask)
            loss = criterion(outputs, targets.squeeze(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


# 6. 主函数
def main():
    device = CONFIG["device"]
    
    # 加载QM9数据集
    print("加载QM9数据集...")
    dataset = QM9(root=CONFIG["dataset_path"])
    print(f"加载完成! 数据集大小: {len(dataset)}")
    
    # 提取SMILES和目标属性
    smiles_list = dataset.data.smiles
    # G: 吉布斯自由能, 索引11
    targets = dataset.data.y[:, CONFIG["target_index"]].numpy()
    
    # 标准化目标值
    scaler = StandardScaler()
    targets = scaler.fit_transform(targets.reshape(-1, 1)).flatten()
    
    # 划分数据集
    indices = np.arange(len(smiles_list))
    train_idx, temp_idx = train_test_split(
        indices, test_size=(CONFIG["split_ratio"][1] + CONFIG["split_ratio"][2]), 
        random_state=CONFIG["random_seed"]
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=CONFIG["split_ratio"][2] / (CONFIG["split_ratio"][1] + CONFIG["split_ratio"][2]), 
        random_state=CONFIG["random_seed"]
    )
    
    # 创建词汇表
    print("创建SMILES词汇表...")
    vocab = SMILESVocabulary([smiles_list[i] for i in train_idx])
    
    # 创建数据集
    train_dataset = QM9SMILESDataset(
        [smiles_list[i] for i in train_idx], 
        targets[train_idx], 
        vocab, 
        max_length=CONFIG["max_length"]
    )
    
    val_dataset = QM9SMILESDataset(
        [smiles_list[i] for i in val_idx], 
        targets[val_idx], 
        vocab, 
        max_length=CONFIG["max_length"]
    )
    
    test_dataset = QM9SMILESDataset(
        [smiles_list[i] for i in test_idx], 
        targets[test_idx], 
        vocab, 
        max_length=CONFIG["max_length"]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False
    )
    
    # 创建模型
    model = SMILESTransformer(
        vocab_size=len(vocab),
        d_model=CONFIG["embedding_dim"],
        nhead=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        dim_feedforward=CONFIG["hidden_dim"],
        dropout=CONFIG["dropout_rate"],
        max_seq_len=CONFIG["max_length"]
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练循环
    print("开始训练...")
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(CONFIG["epochs"]):
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss = evaluate(model, val_loader, criterion, device)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler,
                'vocab': vocab,
                'config': CONFIG
            }, '../神经网络训练/best_model.pth')
            print(f"保存最佳模型，验证损失：{best_val_loss:.6f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= CONFIG["early_stop_patience"]:
                print(f"早停：验证损失 {CONFIG['early_stop_patience']} 个epoch未改善")
                break
    
    # 加载最佳模型
    checkpoint = torch.load('../神经网络训练/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 在测试集上评估
    print("\n在测试集上评估最佳模型...")
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"测试集损失: {test_loss:.6f}")
    
    # 计算RMSE（将标准化的值转换回原始尺度）
    model.eval()
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['target'].numpy()
            
            padding_mask = (input_ids == 0)
            outputs = model(input_ids, padding_mask).cpu().numpy()
            
            predictions.extend(outputs)
            true_values.extend(targets.squeeze(-1))
    
    # 转换回原始尺度
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    true_values = scaler.inverse_transform(np.array(true_values).reshape(-1, 1)).flatten()
    
    # 计算RMSE
    rmse = np.sqrt(np.mean((predictions - true_values) ** 2))
    mae = np.mean(np.abs(predictions - true_values))
    
    print(f"测试集RMSE: {rmse:.6f}")
    print(f"测试集MAE: {mae:.6f}")


# 7. 推理函数（用于预测新的SMILES）
def predict_property(smiles, model_path='best_model.pth', device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    vocab = checkpoint['vocab']
    scaler = checkpoint['scaler']
    
    # 重建模型
    model = SMILESTransformer(
        vocab_size=len(vocab),
        d_model=config["embedding_dim"],
        nhead=config["num_heads"],
        num_layers=config["num_layers"],
        dim_feedforward=config["hidden_dim"],
        dropout=config["dropout_rate"],
        max_seq_len=config["max_length"]
    ).to(device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 处理SMILES
    indices = vocab.tokenize(smiles)
    
    # 填充处理
    if len(indices) > config["max_length"]:
        indices = indices[:config["max_length"]]
    else:
        padding = [vocab.pad_idx] * (config["max_length"] - len(indices))
        indices = indices + padding
    
    # 转换为张量
    input_ids = torch.LongTensor([indices]).to(device)
    padding_mask = (input_ids == 0)
    
    # 预测
    with torch.no_grad():
        output = model(input_ids, padding_mask).cpu().numpy()
    
    # 转换回原始尺度
    prediction = scaler.inverse_transform(output.reshape(-1, 1)).flatten()[0]
    
    return prediction


if __name__ == "__main__":
    main()
