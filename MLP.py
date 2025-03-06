import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

# 检测可用设备
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_device()
# 数据准备
def prepare_data():

    # 读取CSV文件
    df_data = pd.read_csv('/Users/liuyihang/Desktop/ML_features.csv', nrows=90)
    # 对数值列保留 4 位有效数字
    df_data = df_data.round(4)
    # 去掉第一列
    df_data = df_data.drop(columns=['Name'])
    # 得到G1 与 G2 数据集，每个df_data_Gi都包括完整的features and tagert
    df_data_G1 = df_data.drop(columns=['G2'])
    df_data_G2 = df_data.drop(columns=['G1'])
    # 定制配位规则映射，使其配位作为特征参与机器学习训练
    d = {'Pyridine G1': 1, 'Pyridine G2': 2, 'pyrrole G3': 3}
    # 使用列名索引
    df_data_G1['Type'] = df_data_G1['Type'].map(d)
    df_data_G2['Type'] = df_data_G2['Type'].map(d)
    df_data['Type'] = df_data['Type'].map(d)
    # 做出X_Gi 与 y_Gi目标进行特征提取与目标分割
    X_G1 = df_data_G1.drop('G1', axis=1)
    y_G1 = df_data_G1['G1']
    X_G2 = df_data_G2.drop('G2', axis=1)
    y_G2 = df_data_G2['G2']

    # 初始化标准化器
    scaler = StandardScaler()
    # 对特征数据进行标准化
    X_G1_std = pd.DataFrame(scaler.fit_transform(X_G1), columns=X_G1.columns)
    X_G2_std = pd.DataFrame(scaler.fit_transform(X_G2), columns=X_G2.columns)
    df_standardized_data = pd.DataFrame(scaler.fit_transform(df_data), columns=df_data.columns)
    Metal_features = X_G1_std  ## 归一化，且保留所有特征（包括金属与非金属配体及过渡金属与配体之间联系）

    # 转换为numpy数组
    Metal_features_np = np.array(Metal_features)
    y_G1_np = np.array(y_G1)
    y_G2_np = np.array(y_G2)

    # 转换为PyTorch Tensor

    X_tensor = torch.tensor(Metal_features_np, dtype=torch.float32)
    y_G1_tensor = torch.tensor(y_G1_np, dtype=torch.float32).reshape(-1, 1)
    y_G2_tensor = torch.tensor(y_G2_np, dtype=torch.float32).reshape(-1, 1)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_G1_tensor, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# 定义5层MLP模型
class MLPRegressor(nn.Module):
    def __init__(self, input_size=48):
        super(MLPRegressor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.LeakyReLU(),

            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.layers(x)


# 训练配置
def train_model(X_train, y_train, X_test, y_test, epochs=100, batch_size=16):
    # 转换数据到当前设备
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    # TensorDataset（可以将“特征”与“标签”封装一个pytorch可以识别的数据格式；
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # train_loader：数据加载器，可以把刚刚封装的数据格式按批次划分，用于训练数据；
    # shuffle=True：在每个训练周期（epoch）开始时候打乱数据格式，避免因为机器记住了数据的顺序而造成过拟合

    # 初始化模型、损失函数和优化器
    model = MLPRegressor().to(device)
    # 使用均方误差MES作为回归任务的损失函数
    criterion = nn.MSELoss()
    # 使用Adam求解器进行参数更新
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 初始化记录容器
    train_losses = []
    # 初始化记录容器
    val_losses = []

    # 创建进度条对象
    progress_bar = tqdm(range(epochs), desc="Training", unit="epoch")


    # 训练循环：
    #for epoch in range(epochs):
    for epoch in progress_bar:
        model.train()
        epoch_train_loss = 0.0
        # 将模型调整为训练模式：在训练模式可以随机屏蔽神经源（Dropout）防止过拟合；但是在评估模式会保持所有神经元激活状态；确保自动梯度计算为开启模拟
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播：将数据送入编译好的model模型函数当中，得到最后输出结果
            outputs = model(batch_x)
            # 计算损失
            loss = criterion(outputs, batch_y)
            # 根据损失，进行反向传播，调整梯度更新
            loss.backward()
            # 根据调整后的梯度，进行参数更新——前面制定了优化器Adam
            optimizer.step()
            epoch_train_loss += loss.item() * batch_x.size(0)

        # 计算平均训练损失
        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test).item()
            val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            progress_bar.write(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

            # 实时更新进度条描述
            progress_bar.set_postfix({
                "Train Loss": f"{avg_train_loss:.4f}",
                "Val Loss": f"{val_loss:.4f}"
            })



    return model , train_losses, val_losses

# 评估函数
def evaluate_model(model, X_test, y_test):
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        mse = nn.MSELoss()(predictions, y_test)
        mae = nn.L1Loss()(predictions, y_test)
        print(f'MSE: {mse.item():.4f}, MAE: {mae.item():.4f}')

def plot_training_curve(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss', linestyle='--')
    plt.title('Training and Validation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(False)
    plt.show()

# 主流程
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()

    # 训练模型
    model, train_losses, val_losses = train_model(X_train, y_train, X_test, y_test, epochs=100)

    # 评估模型
    evaluate_model(model, X_test, y_test)

    # 绘制训练曲线
    plot_training_curve(train_losses, val_losses)


print(f"Using device: {device}",'MLP is runing on GPU(M1 pro) of MacBook Pro ' )
