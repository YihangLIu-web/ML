import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
from tqdm import tqdm


class Generator:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def generate_txt(self, phase):
        txt_path = f"{phase}.txt"
        data_dir = self.data_dir
        with open(txt_path, 'w') as f:
            class_names = sorted(os.listdir(os.path.join(data_dir, phase)))
            for label, class_name in enumerate(class_names):
                class_dir = os.path.join(data_dir, phase, class_name)
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    f.write(f"{img_path} {label}\n")
        print(f"{txt_path} has been done")
        return txt_path, class_names


# generate_txt(data_dir='/Users/liuyihang/Desktop/prepared_data', phase='train')
# generate_txt(data_dir='/Users/liuyihang/Desktop/prepared_data', phase='val')


class TxtImageDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        super().__init__()
        self.samples = []
        with open(txt_file, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                self.samples.append((path, int(label)))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        path, label = self.samples[item]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
