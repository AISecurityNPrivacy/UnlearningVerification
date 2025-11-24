import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import sys
import os
from torch.utils.data import Subset

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import test_model


def get_adv_Dv(Dr, dist_dict, retrain_indices, unlearn_indices):
    seen_indices = set()
    Dv_index = []
    for du_idx in unlearn_indices:
        dis = dist_dict[du_idx]
        selected_indices = dis[int(-(len(dis) * 0.025)):]  # 取后2000个索引
        for idx in selected_indices:
            if idx in retrain_indices:
                if idx not in seen_indices:
                    seen_indices.add(idx)
                    Dv_index.append(idx)  # 从 Dr 中获取对应数据并添加
    import random
    # 假设 Dv_index 的长度大于 8000
    if len(Dv_index) > len(Dr) * 0.6:
        Dv_index = random.sample(Dv_index, int(len(Dr) * 0.6))
    print(len(Dv_index) / len(Dr))
    Dr_indices = Dr.indices  # 这是原 train_dataset 中被划分到 Dr 的索引
    # 创建一个从原始索引到 Dr 局部索引的映射表
    global_to_local = {idx: i for i, idx in enumerate(Dr_indices)}
    # 用 Dv_index 中的全局索引去找 Dr 中的局部索引
    Dv_local_index = [global_to_local[idx] for idx in Dv_index if idx in global_to_local]
    # 构建子集
    Dv = Subset(Dr, Dv_local_index)
    return Dv


def get_adv_Du(Dr, dist_dict, retrain_indices, unlearn_indices):
    seen_indices = set()
    Dv_index = []
    for du_idx in unlearn_indices:
        dis = dist_dict[du_idx]
        selected_indices = dis[:3]  # 取后2000个索引
        for idx in selected_indices:
            if idx in retrain_indices:
                if idx not in seen_indices:
                    seen_indices.add(idx)
                    Dv_index.append(idx)  # 从 Dr 中获取对应数据并添加
    import random
    print(len(Dv_index))
    # 假设 Dv_index 的长度大于 8000
    if len(Dv_index) > len(unlearn_indices):
        Dv_index = random.sample(Dv_index, len(unlearn_indices))
    elif len(Dv_index) < len(unlearn_indices):
        needed = len(unlearn_indices) - len(Dv_index)
        remaining_indices = list(set(retrain_indices) - seen_indices)
        if needed > len(remaining_indices):
            raise ValueError("可用于补充的 retrain 样本不足")
        extra_indices = random.sample(remaining_indices, needed)
        Dv_index.extend(extra_indices)
    Dr_indices = Dr.indices  # 这是原 train_dataset 中被划分到 Dr 的索引
    # 创建一个从原始索引到 Dr 局部索引的映射表
    global_to_local = {idx: i for i, idx in enumerate(Dr_indices)}
    # 用 Dv_index 中的全局索引去找 Dr 中的局部索引
    Dv_local_index = [global_to_local[idx] for idx in Dv_index if idx in global_to_local]
    # 构建子集
    Dv = Subset(Dr, Dv_local_index)
    return Dv


class AdvSnDataset(Dataset):
    """优化的Adv_Sn数据集，预处理替换映射并支持高效批量加载"""

    def __init__(self, train_dataset, unlearn_indices, retain_indices, dist_dict):
        self.train_dataset = train_dataset
        self.unlearn_indices = set(unlearn_indices)
        self.retain_indices = set(retain_indices)
        self.dist_dict = dist_dict

        # 预处理替换映射表
        self.replacement_map = self._build_replacement_map()

        # 使用所有样本的索引
        self.all_indices = list(range(len(train_dataset)))

    def _build_replacement_map(self):
        """预先构建替换映射表"""
        replacement_map = {}

        for idx in self.unlearn_indices:
            if idx in self.dist_dict:
                # 寻找替代样本
                substitute_idx = None
                for candidate_idx in self.dist_dict[idx]:
                    if candidate_idx not in self.unlearn_indices and candidate_idx in self.retain_indices:
                        substitute_idx = candidate_idx
                        break

                if substitute_idx is not None:
                    replacement_map[idx] = substitute_idx
                else:
                    # 如果没找到合适的替代样本，保持原样本
                    replacement_map[idx] = idx
            else:
                # 如果没有相似度信息，保持原样本
                replacement_map[idx] = idx

        return replacement_map

    def __len__(self):
        return len(self.all_indices)

    def __getitem__(self, idx):
        original_idx = self.all_indices[idx]

        # 如果是需要遗忘的样本，使用替换后的索引
        if original_idx in self.unlearn_indices:
            actual_idx = self.replacement_map[original_idx]
        else:
            actual_idx = original_idx

        data, label = self.train_dataset[actual_idx]
        return data, label, original_idx  # 返回原始索引用于记录替换情况

    def get_replacement_map(self):
        """获取当前的替换映射表"""
        active_replacements = {}
        for original_idx, substitute_idx in self.replacement_map.items():
            if original_idx != substitute_idx:  # 只记录实际发生替换的情况
                active_replacements[original_idx] = substitute_idx
        return active_replacements


def train_Adv_Sn_model(model, train_dataset, Dr_loader, Du_loader, criterion, optimizer,
                       scheduler=None, batch_size=256, num_epochs=150, device=None,
                       config=None, unlearn_indices=None, retain_indices=None, dist_dict=None):
    # 配置默认参数
    if config is None:
        config = {}

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
    resume_epoch = config.get('resume_epoch', 0)

    # 创建目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if log_path is not None and not os.path.exists(log_path):
        os.makedirs(log_path)

    if log_path is not None:
        writer = SummaryWriter(log_path)

    model.to(device)

    # 从断点恢复训练
    if resume_epoch > 0:
        checkpoint_path = os.path.join(save_path, f'{resume_epoch}.pth')
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            print(f"Resuming training from epoch {resume_epoch}")

    # 创建优化的数据集
    print("构建优化的Adv_Sn数据集...")
    adv_dataset = AdvSnDataset(train_dataset, unlearn_indices, retain_indices, dist_dict)

    # 创建数据加载器，使用多进程加载
    adv_loader = DataLoader(
        adv_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,  # 使用多进程加载
        pin_memory=True,  # 如果使用GPU，启用pin_memory可以加速数据传输
        drop_last=True  # 丢弃最后一个不完整的batch
    )

    print(f"数据集构建完成，共 {len(adv_dataset)} 个样本")
    print(f"预处理替换映射：{len(adv_dataset.get_replacement_map())} 个样本将被替换")

    for epoch in range(resume_epoch + 1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0

        progress_bar = tqdm(adv_loader, desc=f'Epoch {epoch}/{num_epochs}')

        for batch_data, batch_labels, original_indices in progress_bar:
            # 数据已经通过DataLoader高效加载，直接使用
            inputs = batch_data.to(device)
            labels = batch_labels.to(device)

            # 常规的训练步骤
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix(loss=running_loss / batch_count)

        if scheduler is not None:
            scheduler.step()

        avg_train_loss = running_loss / batch_count

        # 评估模型
        Dr_accuracy = test_model(model, Dr_loader, device)
        Du_accuracy = test_model(model, Du_loader, device)

        if log_path is not None:
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Accuracy/train', Dr_accuracy, epoch)
            writer.add_scalar('Accuracy/test', Du_accuracy, epoch)

        if epoch % save_epoch == 0:
            checkpoint_file = os.path.join(save_path, f'epoch_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_file)

            # 保存替换规则映射表
            replacement_map = adv_dataset.get_replacement_map()
            with open(os.path.join(save_path, f'replacement_map_epoch_{epoch}.pkl'), 'wb') as f:
                pickle.dump(replacement_map, f)
            print(f'Model checkpoint and replacement map saved at epoch {epoch}')
            print(f'替换了 {len(replacement_map)} 个样本')

        print(f'Epoch [{epoch}/{num_epochs}], Loss: {avg_train_loss:.4f}, '
              f'Train Acc: {Dr_accuracy:.2f}, Test Acc: {Du_accuracy:.2f}')

    if log_path is not None:
        writer.close()