from sklearn.metrics import f1_score
import random
import numpy as np
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import torch
import sys
import os
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import SimpleCNN, test_model


# 执行正常的模型训练过程
# 同时记录每个训练步骤的关键信息:保存每个batch的数据索引,保存每个训练步骤后的模型状态,保存优化器状态
# 这些信息构成了"训练证明"(Proof of Training, PoT)

def train_with_proof(train_set, test_loader, model, model_name, num_epochs, bs, seed, device, config, criterion,
                     optimizer):
    """
    执行带证明的训练过程，同时记录训练信息到TensorBoard

    Args:
        train_set: 训练数据集
        test_loader: 测试数据加载器
        model: 模型
        model_name: 模型名称
        num_epochs: 训练轮数
        bs: 批处理大小
        seed: 随机种子
        device: 设备
        config: 配置字典，包含各种路径
        criterion: 损失函数
        optimizer: 优化器
    """

    # 初始化TensorBoard writer
    log_path = config.get('save_log_path', './logs')
    writer = SummaryWriter(log_dir=log_path)

    num_batches = int(np.ceil(len(train_set) / bs))  # ceil() 是向上取整函数.

    print(f"开始训练证明模型，共 {num_epochs} 个epoch，每个epoch有 {num_batches} 个batch")

    for epoch in range(num_epochs):
        # Training phase
        model.train()  # Set the model in training mode

        train_loss = 0
        train_correct_nums = 0
        train_total = 0
        shuffled_indices = torch.randperm(len(train_set))

        # 使用tqdm显示epoch进度
        epoch_progress = tqdm(range(num_batches), desc=f'Epoch {epoch + 1}/{num_epochs}')

        for batch_idx in epoch_progress:
            # Get the indices for the current mini-batch
            start_idx = batch_idx * bs
            end_idx = min(start_idx + bs, len(train_set))
            batch_indices = shuffled_indices[start_idx:end_idx]

            # Extract data and labels using the batch_indices
            batch_data = [train_set[i][0] for i in batch_indices]
            # 创建了一个包含多个单独图像张量的列表
            batch_labels = [train_set[i][1] for i in batch_indices]

            # Convert to tensors
            batch_data = torch.stack(batch_data).to(device)
            # 为了能够一次性对整个batch的数据进行处理，需要将这些单独的图像张量组合成一个四维张量
            batch_labels = torch.tensor(batch_labels).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = model(batch_data)

            # Compute loss
            loss = criterion(outputs, batch_labels)

            # Perform backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct_nums += (predicted == batch_labels).sum().item()
            train_total += batch_labels.size(0)

            # 更新进度条显示当前损失
            current_avg_loss = train_loss / (batch_idx + 1)
            current_accuracy = 100.0 * train_correct_nums / train_total
            epoch_progress.set_postfix({
                'Loss': f'{current_avg_loss:.4f}',
                'Acc': f'{current_accuracy:.2f}%'
            })

            should_save = True

            # 这里注意，我们保存了全部的proof结果，但是运行forge的时候我们只需要保存forge的最后几个结果即可，
            # 因为它包含了整个伪造训练过程的最终状态
            if should_save:
                # 提取公共文件名模板
                file_template = f"{model_name}_epoch_{epoch}_batch_{batch_idx}_seed_{seed}"
                # 保存文件
                torch.save(model.state_dict(), os.path.join(config.get('save_model_path'), f"{file_template}.pth"))
                torch.save(optimizer.state_dict(),
                           os.path.join(config.get('save_optimizer_path'), f"{file_template}.pth"))
                np.save(os.path.join(config.get('save_batch_path'), f"{file_template}.npy"), batch_indices.numpy())

        train_loss /= num_batches
        train_accuracy = 100.0 * train_correct_nums / train_total

        # 评估阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                val_loss += criterion(outputs, targets).item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)

        val_loss /= len(test_loader)
        val_accuracy = 100.0 * val_correct / val_total

        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Test', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/Test', val_accuracy, epoch)

        # 记录学习率（如果优化器有param_groups）
        if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning_Rate', current_lr, epoch)

        # Print the validation loss and accuracy for each epoch
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, "
            f"Train Accuracy: {train_accuracy:.2f}%, Test Loss: {val_loss:.4f},"
            f" Test Accuracy: {val_accuracy:.2f}%")

    # 关闭TensorBoard writer
    writer.close()
    print('Finished Training with Proof')
    print(f'TensorBoard logs saved to: {log_path}')
    print('To view logs, run: tensorboard --logdir={} --port=6006'.format(log_path))