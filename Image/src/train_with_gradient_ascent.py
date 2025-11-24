import os
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils import test_model


def train_gradient_ascent(model, Du_dataset, Du_loader, Dr_loader, criterion, optimizer,
                             num_epochs=10, device=None, num_classes=10, save_epoch=1, config=None):
    """
    专门针对Du数据集进行梯度上升训练

    参数:
    - model: 待训练的模型
    - Du_dataset: 需要遗忘的数据集
    - Du_loader: 需要遗忘的数据加载器
    - Dr_loader: 保留的数据加载器(用于评估)
    - criterion: 损失函数
    - optimizer: 优化器
    - num_epochs: 训练轮数
    - device: 训练设备
    - num_classes: 分类数量
    - save_epoch: 多少轮保存一次模型
    - config: 配置参数
    """
    # 配置默认参数
    if config is None:
        config = {}

    # 设置超参数默认值，允许通过config覆盖
    save_path = config.get('save_model_path', None)
    log_path = config.get('save_log_path', None)

    model.to(device)
    writer = SummaryWriter(log_dir=log_path)

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0

        # 进度条显示每个epoch的训练进度
        progress_bar = tqdm(total=len(Du_loader), desc=f'Epoch {epoch}/{num_epochs}')

        # 遍历Du数据集
        for batch_idx, (inputs, labels) in enumerate(Du_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)

            # 计算梯度上升损失(负损失)
            loss = -criterion(outputs, labels)  # 注意这里是负号，实现梯度上升

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1
            progress_bar.update(1)
            progress_bar.set_postfix(loss=running_loss / batch_count)

        progress_bar.close()

        avg_train_loss = running_loss / batch_count

        # 评估模型
        Dr_accuracy = test_model(model, Dr_loader, device)
        Du_accuracy = test_model(model, Du_loader, device)

        if log_path is not None:
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Accuracy/Dr', Dr_accuracy, epoch)
            writer.add_scalar('Accuracy/Du', Du_accuracy, epoch)

        # 保存模型
        if epoch % save_epoch == 0:
            checkpoint_file = os.path.join(save_path, f'epoch_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_file)
            print(f'Model checkpoint saved at epoch {epoch}')

        print(f'Epoch [{epoch}/{num_epochs}], Loss: {avg_train_loss:.4f}, '
              f'Dr Acc: {Dr_accuracy:.2f}, Du Acc: {Du_accuracy:.2f}')

    if log_path is not None:
        writer.close()