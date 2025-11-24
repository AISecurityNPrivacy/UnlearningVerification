import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split


def calculate_distance(dataset_name, train_set, device='cuda'):
    """
    计算训练集样本之间的余弦相似度，并返回每个样本与其类别内样本的排序。

    Args:
        dataset_name (str): 数据集名称，用于适配不同的数据集格式。
        train_set (Dataset): 训练数据集。
        device (str): 指定计算设备('cuda' 或 'cpu')。

    Returns:
        dict: 每个样本与同类样本的相似度排序，键为样本索引，值为排序索引。
    """

    # 根据数据集类型选择不同的处理方式
    if dataset_name in ['CIFAR10', 'SVHN']:
        return calculate_tensor_dataset_distance(dataset_name, train_set, device)
    elif dataset_name in ['Medical_32', 'facescrub']:
        return calculate_imagefolder_distance(dataset_name, train_set, device)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def calculate_tensor_dataset_distance(dataset_name, train_set, device='cuda'):
    """
    处理具有 .data 和 .targets/.labels 属性的数据集（如CIFAR10, SVHN）
    """
    # 提取训练数据和标签
    train_data = torch.tensor(train_set.data).float().reshape(len(train_set), -1).to(device)

    if dataset_name == 'SVHN':
        train_target = torch.tensor(train_set.labels).to(device)
    else:
        train_target = torch.tensor(train_set.targets).to(device)

    print(f"Train data shape: {train_data.shape}, Train target shape: {train_target.shape}")

    return _calculate_similarity_matrix(train_data, train_target, device)


def calculate_imagefolder_distance(dataset_name, train_set, device='cuda'):
    """
    处理 ImageFolder 类型的数据集（如Medical_32, facescrub）
    """
    # 使用DataLoader收集所有数据和标签
    batch_size = 64 if dataset_name == 'Medical_32' else 32  # Medical_32数据较大，使用较大batch_size
    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # 收集所有数据和标签
    all_data = []
    all_targets = []

    print("Loading and processing images...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            # 将图像展平为特征向量
            features = images.reshape(images.size(0), -1).to(device)
            all_data.append(features)
            all_targets.append(labels.to(device))

            # 打印进度
            if batch_idx % 50 == 0:
                print(f"Processed {batch_idx * batch_size} images...")

    # 将所有批次的数据连接起来
    train_data = torch.cat(all_data, dim=0)
    train_target = torch.cat(all_targets, dim=0)

    print(f"Train data shape: {train_data.shape}, Train target shape: {train_target.shape}")

    return _calculate_similarity_matrix(train_data, train_target, device)


def _calculate_similarity_matrix(train_data, train_target, device):
    """
    计算相似度矩阵的核心函数
    """
    # 初始化存储结果的字典
    feat_dist = {}

    # 提前获取所有唯一的标签，并确保是整数类型
    unique_labels = [int(label.item()) for label in torch.unique(train_target)]
    print(f"Found {len(unique_labels)} unique labels: {unique_labels}")

    # 按类别分组样本索引
    class_indices = {label: (train_target == label).nonzero(as_tuple=True)[0] for label in unique_labels}

    # 打印每个类别的样本数量
    for label in unique_labels:
        print(f"Class {label}: {len(class_indices[label])} samples")

    # 计算余弦相似度
    total_samples = len(train_data)
    print(f"Starting similarity calculation for {total_samples} samples...")

    for idx in range(total_samples):
        data_sample = train_data[idx].unsqueeze(0)  # 当前样本
        label = int(train_target[idx].item())
        same_class_indices = class_indices[label]  # 同类别样本索引
        same_class_data = train_data[same_class_indices]  # 同类别样本数据

        # 批量计算余弦相似度
        similarity_vector = F.cosine_similarity(same_class_data, data_sample, dim=1)

        # 按相似度排序，降序
        sorted_indices = same_class_indices[torch.argsort(similarity_vector, descending=True)]

        # 存储排序结果
        feat_dist[idx] = sorted_indices.cpu().numpy()

        # 打印进度
        if idx % 500 == 0:  # 减少打印频率以提高性能
            print(f"Processed {idx}/{total_samples} samples ({idx / total_samples * 100:.1f}%)...")

    print("Similarity calculation completed!")
    return feat_dist


# 保留原有的facescrub函数以保持兼容性
def calculate_facescrub_distance(dataset_name, train_set, device='cuda'):
    """
    专门为facescrub数据集计算相似度（保持向后兼容）
    """
    return calculate_imagefolder_distance(dataset_name, train_set, device)
