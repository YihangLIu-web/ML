import torch
import os
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
from tqdm import tqdm
from Dataset_process import Generator, TxtImageDataset
from ViT_model import VIT
from CNN import LightCNN, SimpleCNN


def main():
    data_dir = '/Users/liuyihang/Desktop/data/prepared_data'
    # data_dir = '/home/user/ljrFiles/CNN/prepared_data'
    epochs = 1
    generator = Generator(data_dir)
    train_txt, class_names = generator.generate_txt(phase='train')
    val_txt, class_names = generator.generate_txt(phase='val')
    print(class_names)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = TxtImageDataset(train_txt, transform)
    val_dataset = TxtImageDataset(val_txt, transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    # print(f"检测到CUDA设备: {torch.cuda.get_device_name()}")
    # model = VIT(img_size=224, patch_size=16,
    #             num_classes=2, embed_dim=768, img_channels=3,
    #             depth=6, num_heads=4, mlp_ratio=4)
    model = SimpleCNN(num_classes=2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Train"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} loss is {total_loss / len(train_loader)}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1} valid"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"ACC {100 * correct / total: .4}%")


if __name__ == '__main__':
    main()
