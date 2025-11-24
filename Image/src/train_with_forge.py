import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from utils import SimpleCNN, seed_setting, create_dataset_directories,dataset_format_convert, load_dataset, generate_verify_subsets, generate_verify_subsets_dataloaders


def train_with_forge(epoch, unlearn_indices, Dr, train_dataset, feat_dist, config, process_id=0, num_process=1,
                       batch_size=2048, dataset_name=None, lr=0.001, seed=None, device=None, model =None, model_name='SimpleCNN',weight_decay=None):
    """
    伪造重训练的证明，通过替换需要遗忘的样本为相似样本来执行

    Args:
        epoch (int): 总训练轮次
        unlearn_indices (list): 需要遗忘的样本索引
        Dr (Subset): 保留的训练集子集
        train_dataset (Dataset): 完整训练集
        feat_dist (dict): 样本相似度信息
        config (dict): 配置信息，包含保存路径
        process_id (int): 当前进程ID
        num_process (int): 总进程数
        batch_size (int): 批处理大小
        dataset_name (str): 数据集名称
        lr (float): 学习率
        seed (int): 随机种子
        device (torch.device): 计算设备
        optimizer (Optimizer): 优化器对象
    """
    num_batches = int(np.ceil(len(train_dataset) / batch_size))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 获取当前进程需要处理的batch范围
    for current_epoch in range(epoch-5, epoch):
        start_idx = process_id * int(np.ceil(num_batches / num_process))
        end_idx = min(num_batches, (process_id + 1) * int(np.ceil(num_batches / num_process)))

        print(f"Processing epoch {current_epoch + 1}/{epoch}, batches {start_idx} to {end_idx - 1}")

        for idx in tqdm(range(start_idx, end_idx)):
            # 加载原始模型和batch索引
            try:
                model_path = f"{config['save_model_path']}/{model_name}_epoch_{current_epoch}_batch_{idx}_seed_{seed}.pth"
                model.load_state_dict(torch.load(model_path, map_location=device))

                batch_path = f"{config['save_batch_path']}/{model_name}_epoch_{current_epoch}_batch_{idx}_seed_{seed}.npy"
                batch_idx = np.load(batch_path, allow_pickle=True)
            except Exception as e:
                print(f"Error loading model or batch data: {e}")
                continue

            # 检查当前batch是否包含需要遗忘的样本
            if len(set(batch_idx) & set(unlearn_indices)) == 0:
                # 如果不包含需要遗忘的样本，进行扰动更新
                # 随机选择一个保留集合的样本
                index = np.random.randint(len(Dr))
                data, target = Dr[index]
                target = torch.tensor([target]).to(device)
                data = data.unsqueeze(0).to(device)

                # 第一次前向传播和计算损失 - 计算并保留梯度
                output = model(data)
                loss = criterion(output, target)
                loss.backward()  # 计算梯度但不更新参数

                # 创建新的优化器，使用较小的扰动学习率
                optimizer_perturb = optimizer

                # 第二次前向传播和损失计算 - 实际更新模型
                output_new = model(data)
                loss_new = criterion(output_new, target)
                optimizer_perturb.zero_grad()  # 清除之前累积的梯度
                loss_new.backward()  # 计算新梯度
                optimizer_perturb.step()  # 应用梯度更新

                # 保存更新后的模型和batch索引
                if (current_epoch == epoch - 1 and idx >= end_idx - 5) or (
                        current_epoch == epoch - 1 and idx == end_idx - 1):
                    torch.save(model.state_dict(),f"{config['save_model_path']}/{model_name}_forged_epoch_{current_epoch}_batch_{idx}_seed_{seed}.pth")
                    np.save(f"{config['save_batch_path']}/{model_name}_forged_epoch_{current_epoch}_batch_{idx}_seed_{seed}.npy", batch_idx)


            else:
                # 如果包含需要遗忘的样本，构造新的batch
                forged_batch = batch_idx.copy()

                # 替换需要遗忘的样本
                for i in range(len(forged_batch)):
                    if forged_batch[i] in unlearn_indices:

                        # 从feat_dist中获取相似样本排序
                        similar_samples = feat_dist[forged_batch[i]]

                        # 查找第一个不在unlearn_indices中的相似样本
                        for similar_idx in similar_samples:
                            if similar_idx not in unlearn_indices:
                                forged_batch[i] = similar_idx
                                break

                # 加载适当的checkpoint
                if current_epoch == 0 and idx == 0:
                    # 第一个batch使用初始模型,初始化的模型保存的是字典类型，proof给出的是整个模型对象
                    model.load_state_dict(torch.load(f"{os.path.dirname(config['save_model_path'])}/{model_name}_init_seed_{seed}.pth", map_location=device))

                elif idx == 0:
                    # 每个epoch的第一个batch使用上一个epoch的最后一个batch的模型
                    last_batch = int(np.ceil(len(train_dataset) / batch_size)) - 1
                    model.load_state_dict(torch.load(f"{config['save_model_path']}/{model_name}_epoch_{current_epoch - 1}_batch_{last_batch}_seed_{seed}.pth",map_location=device))
                    optimizer.load_state_dict(torch.load(f"{config['save_optimizer_path']}/{model_name}_epoch_{current_epoch - 1}_batch_{last_batch}_seed_{seed}.pth"))
                else:
                    # 其他batch使用前一个batch的模型
                    model.load_state_dict(torch.load(f"{config['save_model_path']}/{model_name}_epoch_{current_epoch}_batch_{idx - 1}_seed_{seed}.pth",map_location=device))
                    optimizer.load_state_dict(torch.load(f"{config['save_optimizer_path']}/{model_name}_epoch_{current_epoch}_batch_{idx - 1}_seed_{seed}.pth"))

                # 准备替换后的batch数据
                batch_data = [train_dataset[i][0] for i in forged_batch]
                batch_labels = [train_dataset[i][1] for i in forged_batch]
                batch_data = torch.stack(batch_data).to(device)
                batch_labels = torch.tensor(batch_labels).to(device)

                # 执行一次训练步骤
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                if (current_epoch == epoch - 1 and idx >= end_idx - 5) or (
                        current_epoch == epoch - 1 and idx == end_idx - 1):
                    # 仅保存最后一个epoch的最后几个batch以节省内存空间

                    # 保存伪造的模型和batch数据
                    torch.save(model.state_dict(),f"{config['save_model_path']}/{model_name}_forged_epoch_{current_epoch}_batch_{idx}_seed_{seed}.pth")
                    np.save(f"{config['save_batch_path']}/{model_name}_forged_epoch_{current_epoch}_batch_{idx}_seed_{seed}.npy", forged_batch)

                    # 也保存优化器状态
                    torch.save(optimizer.state_dict(),f"{config['save_optimizer_path']}/{model_name}_forged_epoch_{current_epoch}_batch_{idx}_seed_{seed}.pth")
        print(f"Finished processing epoch {current_epoch + 1}/{epoch}")