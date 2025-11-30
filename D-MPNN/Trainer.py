
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from config import cfg


# 代码逻辑结构：
# Trainer.__init__()
#     ├── 保存 model
#     ├── 创建 optimizer
#     ├── 创建 loss_fn
#     ├── 创建 dataloader
#     └── 保存 config

class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_dataloder,
            val_dataloder=None,
            loss_fn: Optional[nn.Module] = None,
            config=cfg
    ):
        self.config = config
        self.model = model.to(self.config.train.device)
        self.train_dataloder = train_dataloder
        self.val_dataloder = val_dataloder
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.train.learning_rate,
            weight_decay=self.config.train.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        self.global_step = 0  # 用作记录更新次数
        self.best_val_mse = float("inf")  # 记录当前最优验证集 MSE.所以把“最好”设成正无穷

    def training_step(self, batch):
        """
        Train the model of each batch using one epoch
        :param batch:
        :return: float:loss.items()
        """
        self.model.train()
        batch = batch.to(self.config.train.device)
        outputs = self.model(batch.z, batch.pos, batch.batch)
        outputs = outputs.squeeze(-1)
        loss = self.loss_fn(outputs, batch.y)
        self.optimizer.zero_grad()
        loss.backward()
        #   梯度裁剪，防止梯度爆炸
        if self.config.train.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.train.gradient_clip_norm
            )
        self.optimizer.step()
        return loss.item()

    # 预测 / 推理接口 在一个完整 DataLoader 上做评估
    # 跟training_step 不一样 接受的是一个loder
    @torch.no_grad()
    def evaluate(self, dataloader, split_name="val"):
        """
        在给定 dataloader 上评估模型性能，返回 MSE / MAE。
        典型用法：
            - 训练过程中：evaluate(val_loader, "val")
            - 训练结束后：evaluate(test_loader, "test")
        :param split_name: val
        :param dataloader: 一个完整的batch 需要在一个完整的loder里面进行评估
        :return: dict: MSE / MAE 评估字典
        """
        self.model.eval()
        mse_sum = 0.0
        mae_sum = 0.0
        n_samples = 0
        for batch in dataloader:
            batch = batch.to(self.config.train.device)
            pred = self.model(batch.z, batch.pos, batch.batch)
            pred = pred.squeeze(-1)
            mse = F.mse_loss(pred, batch.y, reduction='sum')
            mae = F.l1_loss(pred, batch.y, reduction='sum')
            mse_sum += mse.item()
            mae_sum += mae.item()
            # 对于 PyG，这是当前 batch 中图的个数
            n_samples += batch.num_graphs
        mse_avg = mse_sum / n_samples
        mae_avg = mae_sum / n_samples
        print(f"{split_name} :  MSE = {mse_avg:.6f}, MAE = {mae_avg:.6f}, N = {n_samples}")
        return {"mse": mse_avg, "mae": mae_avg}

    def _save_checkpoint(self, path: str, epoch: int):
        """保存完整训练状态（模型 + 优化器），用于恢复训练。"""
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(state, path)

    def _save_best(self, epoch: int, val_mse: float):
        """保存当前最优模型（完整 ckpt + 轻量级 model）"""
        # 保存完整的ckpt
        self._save_checkpoint(self.config.train.best_ckpt_path, epoch)
        # 仅模型参数，用于推理或部署
        torch.save(self.model.state_dict(), self.config.train.best_model_path)
        print(
            f"New best model at epoch {epoch}",
            f"val_mse={val_mse:.6f}, saved to {self.config.train.best_model_path}"
        )

    def _save_last(self, epoch: int):
        """保存最近一次训练状态（每个 epoch 覆盖一次）。"""
        self._save_checkpoint(self.config.train.last_ckpt_path, epoch)

    def train(self):
        for epoch in range(self.config.train.num_epochs):
            epoch_loss = 0.0
            steps_in_epoch = 0

            progress_bar = tqdm(self.train_dataloder, desc=f"Epoch {epoch + 1}")

            for batch in progress_bar:
                self.global_step += 1
                steps_in_epoch += 1
                loss = self.training_step(batch)
                epoch_loss += loss

                if self.global_step % self.config.train.logging_steps == 0:
                    progress_bar.set_postfix({"loss": f"{loss:.4f}"})
            avg_epoch_loss = epoch_loss / steps_in_epoch  # avg_epoch_loss 记录的是这一个epoch中的loss的平均值。

            print(f"[Epoch {epoch + 1}] avg_train_loss = {avg_epoch_loss:.4f}")
            # 在每个epoch结束后，用验证集评估一次
        val_metrics = self.evaluate(self.val_dataloder, 'val')
        val_mse = val_metrics['mse']
        # 每个epoch都要最近一次的状态
        self._save_last(epoch + 1)
        # 如果当前的验证集更好，则更新 best
        if val_mse < self.best_val_mse:
            self.best_val_mse = val_mse
            self._save_best(epoch + 1, val_mse)

