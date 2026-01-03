import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
from tqdm import tqdm


class LightCNN(nn.Module):
    """
    轻量级CNN模型，适合CPU训练
    参数量约100K，比ViT小很多
    """

    def __init__(self, num_classes=2):
        super(LightCNN, self).__init__()
        # 通道扩展层：将3通道扩展到32通道，以便后续的CGHalfConv处理
        self.channel_expand = nn.Conv2d(3, 32, kernel_size=1, bias=False)
        # 特征提取层
        self.features = nn.Sequential(

            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),  # 自适应平均池化
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 扩展通道数到32
        x = self.channel_expand(x)  # 3->32通道

        # 经过常规CNN特征提取
        x = self.features(x)

        # 分类
        x = self.classifier(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 迟化后通道数保持不变化，尺寸减少一半，从原来的224->112
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  # 32*112*112
        self.fc1 = nn.Linear(in_features=32 * 112 * 112, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = self.activation(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
