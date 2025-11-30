# config.py
import os
import json
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime


@dataclass
class PathConfig:
    """
    储存或生成数据、能量Excel的存储绝对位置
    """
    # xsd文件的存储位置，分为训练集以及测试集（不同比例划分需要提前做好，写在了split.py文件里面进行）
    TRAIN_DIR: str = "/Users/liuyihang/Desktop/dataset/train"
    VAL_DIR: str = "/Users/liuyihang/Desktop/dataset/val"

    # Excel 文件 格式：molecule name + energy
    # 可以直接调用位置 excel的命名需要 train_energies.xlsx 即可，调用时使用 cfg.Paths.TRAIN_XLSX 进行私有变量获取
    @property
    def TRAIN_XLSX(self):
        return os.path.join(self.TRAIN_DIR, "train_energies.xlsx")

    @property
    def VAL_XLSX(self):
        return os.path.join(self.VAL_DIR, "val_energies.xlsx")


@dataclass
class DataConfig:
    """
    Dataset相关变量，后续维护可以打印训练集、测试集比例
    Batch_size 相关超参数
    目标能量单位换算
    """
    batch_size: int = 1
    energy_scale: float = 1.0  # Hartree -> eV: 27.2114; if already eV, use 1.0
    num_workers: int = 1  # CUP超参数 由于图的构建在dataset外面已经构建完成，所以训练在线batch的处理不会遇到CPU瓶颈


@dataclass
class TrainConfig:
    """
    训练相关超参数包括 设备、epochs、梯度裁剪、打印循环次数、随机数种子
    """
    device: str = "cpu"  # CPU 仅用于调试， 训练请使用GPU or MPS
    num_epochs: int = 10
    gradient_clip_norm: Optional[float] = 1.0
    logging_steps: int = 10  # 每训练更新10次参数进行记录
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    seed: int = 42

    exp_root: str = 'experiments'
    exp_name: str = 'SchNet_experiments'

    # 运行时候自动填充路径，在__init__里面生成
    exp_dir: str = ""  # 时间戳：experiments/exp_name_时间戳 区分训练的轮次与时间
    ckpt_dir: str = ""
    log_dir: str = ""
    fig_dir: str = ""
    results_dir: str = ""

    best_model_path: str = ""  # 只存 state_dict 的版本（推理用）
    best_ckpt_path: str = ""  # 包含优化器等完整 checkpoint
    last_ckpt_path: str = ""  # 最新训练状态（恢复训练用）
    results_path: str = ""  # 存 test 结果的 json


@dataclass
class ModelConfig:
    """SchNet model hyperparameters"""
    hidden_channels: int = 128
    num_filters: int = 128
    num_interactions: int = 6
    num_gaussians: int = 50
    cutoff: float = 5.0
    max_num_neighbors: int = 64


# 进行总的引入
@dataclass
class ExperimentConfig:
    """
    Top-level config that groups all sub-configs.
    This is the single source of truth for an experiment.
    """
    paths: PathConfig = PathConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()

    def init_experiment_dirs(self) -> "ExperimentConfig":
        """
        # os.makedirs(path, exist_ok=True) 用来自动创建实验目录（包含所有父目录），并且如果目录已存在不会报错。
        # 是深度学习实验中最常用的目录初始化语法。
        :return:
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_dir = os.path.join(self.train.exp_root, f"{self.train.exp_name}_{timestamp}")
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        log_dir = os.path.join(exp_dir, "logs")
        fig_dir = os.path.join(exp_dir, "figures")
        results_dir = os.path.join(exp_dir, "results")

        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        self.train.exp_dir = exp_dir
        self.train.ckpt_dir = ckpt_dir
        self.train.log_dir = log_dir
        self.train.fig_dir = fig_dir
        self.train.results_dir = results_dir

        self.train.best_model_path = os.path.join(ckpt_dir, "best_model.pth")
        self.train.best_ckpt_path = os.path.join(ckpt_dir, "best_ckpt.pt")
        self.train.last_ckpt_path = os.path.join(ckpt_dir, "last_ckpt.pt")
        self.train.results_path = os.path.join(results_dir, "test_results.json")

        config_path = os.path.join(exp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(asdict(self), f, indent=2)

        print(f"[Config] Experiment directory initialized at: {exp_dir}")
        return self


# Instantiate a global configuration object
cfg = ExperimentConfig().init_experiment_dirs()
