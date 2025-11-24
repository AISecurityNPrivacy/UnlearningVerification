import os
from tqdm import tqdm
import torch
from utils import test_model
from torch.utils.tensorboard import SummaryWriter


def train_subset_model(model, train_loader, test_loader, criterion, optimizer, scheduler=None,
                num_epochs=100, device=None,config=None):
    # 配置默认参数
    if config is None:
        raise ValueError("配置参数config不能为None")
    try:
        log_path = config['save_log_path']
    except KeyError:
        raise KeyError("配置中缺少必要参数'save_log_path'")

    try:
        save_path = config['save_model_path']
    except KeyError:
        raise KeyError("配置中缺少必要参数'save_model_path'")

    try:
        save_epoch = config['save_epoch']
    except KeyError:
        raise KeyError("配置中缺少必要参数'save_epoch'")

    neg = config.get('neg', False)
    resume_epoch = config.get('resume_epoch', 0)  # 支持断点续训

    model.to(device)

    # 初始化TensorBoard writer
    writer = SummaryWriter(log_dir=log_path)

    # 如果需要恢复训练，加载模型
    if resume_epoch > 0:
        checkpoint_path = os.path.join(save_path, f'epoch_{resume_epoch}.pth')
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            print(f"Resuming training from epoch {resume_epoch}")

    for epoch in range(resume_epoch + 1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        # 进度条显示每个 epoch 的训练进度
        # total=len(train_loader)：指定迭代的总次数（即总批次数）
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}/{num_epochs}')

        for i, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # 控制是否反转损失
            if neg:
                loss = 0 - criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels)

            loss.backward() # 反向传播
            optimizer.step()  # 更新参数

            # 更新进度条显示
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1))

        # 更新学习率调度器
        if scheduler is not None:
            scheduler.step()

        # 记录训练损失
        avg_train_loss = running_loss / len(train_loader)

        # 计算并记录训练和测试准确率
        train_accuracy = test_model(model, train_loader, device)
        test_accuracy = test_model(model, test_loader, device)

        # 保存模型,默认每个epoch都保存
        if epoch % save_epoch == 0:
            checkpoint_file = os.path.join(save_path, f'epoch_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_file)
            print(f'Model checkpoint saved at epoch {epoch}')

        # 将训练信息记录到TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)
        if scheduler is not None:
            writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)

        print(f'Epoch [{epoch}/{num_epochs}], Loss: {avg_train_loss:.4f}, '
              f'Train Acc: {train_accuracy:.2f}, Test Acc: {test_accuracy:.2f}')

    # 关闭TensorBoard writer
    writer.close()


    # TensorBoard 记录器
    # pkill -f tensorboard
    # tensorboard --logdir=./logs --port=6006

