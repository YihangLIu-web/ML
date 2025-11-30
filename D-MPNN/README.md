# 分子能量预测：基于 SimpleEGNN / SchNet 的图神经网络实现

本仓库用于从分子结构文件（`.xsd`）与能量标注（Excel）构建图数据，并基于等变图神经网络 SimpleEGNN（接口兼容 SchNet）进行分子能量回归建模。代码结构面向科研场景，便于复现与扩展。

---

## 1. 环境依赖

- Python 3.8+
- PyTorch
- PyTorch Geometric 及其依赖
- ASE
- pandas、tqdm 等常用科学计算库

依赖安装方式可根据本机 CUDA 版本参考 PyTorch / PyG 官方说明进行配置。

---

## 2. 代码结构

```text
.
├── config.py            # 全局配置：路径、模型与训练超参数、实验目录管理
├── Dataset_process.py   # 从 .xsd + Excel 构建分子图（Graph_list / MolecularDataset）
├── model.py             # EGNN 层与 SimpleEGNN 模型定义
├── Trainer.py           # 训练器封装：训练循环、评估与模型保存
├── main.py              # 主入口：数据加载、模型构建与训练启动
└── SchNet_推理.ipynb   # 推理与结果分析示例（可选）