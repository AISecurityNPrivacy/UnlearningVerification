import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import MNIST, CIFAR10, SVHN
from torch import optim
from PIL import Image
from torchvision.datasets import ImageFolder

class SimpleCNN(nn.Module):
    def __init__(self, dataset_name, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 根据数据集选择输入通道数
        if dataset_name == 'CIFAR10' or dataset_name == 'SVHN':
            in_channels = 3  # RGB图像
        elif dataset_name == 'facescrub':
            in_channels = 3  # RGB图像
            num_classes = 530  # FaceScrub有530个类别
        elif dataset_name == 'Medical_32':
            in_channels = 3  # RGB图像
            num_classes = 2  # 二分类
        else:
            in_channels = 1  # 灰度图像

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 对于facescrub数据集，输入是64x64，经过两次池化后是16x16
        if dataset_name == 'facescrub':
            self.fc1 = nn.Linear(64 * 16 * 16, 256)
        elif dataset_name == 'Medical_32':
            self.fc1 = nn.Linear(64 * 8 * 8, 128)  # 32x32经过两次池化后是8x8
        else:
            self.fc1 = nn.Linear(64 * 8 * 8, 128)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        if dataset_name == 'facescrub':
            self.fc2 = nn.Linear(256, num_classes)
        elif dataset_name == 'Medical_32':
            self.fc2 = nn.Linear(128, num_classes)  # 明确指定Medical_32的fc2
        else:
            self.fc2 = nn.Linear(128, num_classes)


    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 确保 view 的第一个参数是批量大小！
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def seed_setting(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)  # Python的random模块
    torch.cuda.manual_seed(random_seed)  # CUDA的随机种子
    torch.cuda.manual_seed_all(random_seed)  # 所有GPU的随机种子


def create_dataset_directories(dataset_name):
    """
    为指定的数据集创建必要的目录结构

    Args:
        dataset_name (str): 数据集名称，如'SVHN'、'CIFAR10'等
    """
    # 定义需要创建的子目录
    subdirectories = ['data', 'batches', 'logs', 'models', 'optimizers', 'pol', 'unlearn']

    # 基础路径
    base_path = f'../{dataset_name}'

    print(f"Creating directory structure for {dataset_name}...")

    # 创建基础目录（如果不存在）
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print(f"Created base directory: {base_path}")

    # 创建子目录（如果不存在）
    for subdir in subdirectories:
        subdir_path = os.path.join(base_path, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
            print(f"Created subdirectory: {subdir_path}")
        else:
            print(f"Subdirectory already exists: {subdir_path}")

    print(f"Directory structure for {dataset_name} is ready.")

def dataset_format_convert(dataset_name):
    if dataset_name == 'CIFAR10':
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),  # 将图像从96x96调整到32x32
            transforms.ToTensor(),  # 将图像转换为张量
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),

            # transforms.ToTensor()只作用于图像数据，不会自动转换标签。
            # PyTorch的标准数据集（如CIFAR10、SVHN）返回的标签默认就是Python的int类型。
            # 这里写了也是白写。

        ])
    elif dataset_name == 'SVHN':
        train_transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像转换为张量,SVHN已经是32x32，不需要resize
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif dataset_name == 'facescrub':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # 随机水平翻转增强数据
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
        ])
    elif dataset_name == 'Medical_32':
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),  # 将图像从300x300调整到32x32
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError(f"不支持的数据转换: {dataset_name}")
    return train_transform, test_transform

def load_dataset(dataset_name, train_transform, test_transform, batch_size=256, num_workers=-1):
    if dataset_name == 'CIFAR10' :
        train_dataset = CIFAR10(root=f'../{dataset_name}/data', train=True, download=True, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = CIFAR10(root=f'../{dataset_name}/data', train=False, download=True, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    elif dataset_name == 'SVHN':
        train_dataset = SVHN(root=f'../{dataset_name}/data', split='train', download=True, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = SVHN(root=f'../{dataset_name}/data', split='test', download=True, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    elif dataset_name == 'facescrub':
        # 检查结构化后的数据集是否存在，不存在则创建
        structured_path = f'../{dataset_name}/data/structured'

        force_reorganize = False  # 默认不强制重新组织数据集

        if not os.path.exists(structured_path) or force_reorganize : # 这里每次都处理，加上not，即可判断是否已有，已有则不处理
            print("开始组织FaceScrub数据集...")
            actor_path = f'../{dataset_name}/data/actor_faces'
            actress_path = f'../{dataset_name}/data/actress_faces'
            organize_facescrub_dataset(actor_path, actress_path, structured_path)

        # 将数据分为训练集和测试集 (80%训练, 20%测试)
        full_dataset = ImageFolder(root=structured_path, transform=train_transform)

        dataset_size = len(full_dataset)
        train_size = int(dataset_size * 0.8)
        test_size = dataset_size - train_size

        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size]
        )

        # 为测试集应用测试变换
        test_dataset.dataset = ImageFolder(root=structured_path, transform=test_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    elif dataset_name == 'Medical_32':
        # 加载训练集
        train_dataset = ImageFolder(
            root='../Medical_32/data/melanoma_cancer_dataset/train',
            transform=train_transform
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # 加载测试集
        test_dataset = ImageFolder(
            root='../Medical_32/data/melanoma_cancer_dataset/test',
            transform=test_transform
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    return train_dataset, train_loader, test_dataset, test_loader

def load_dataset_without_transtotensor(dataset_name, batch_size=256, num_workers=-1):
    if dataset_name == 'CIFAR10' :
        train_dataset = CIFAR10(root=f'../{dataset_name}/data', train=True, download=True, transform=None)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = CIFAR10(root=f'../{dataset_name}/data', train=False, download=True, transform=None)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    elif dataset_name == 'SVHN':
        train_dataset = SVHN(root=f'../{dataset_name}/data', split='train', download=True, transform=None)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = SVHN(root=f'../{dataset_name}/data', split='test', download=True, transform=None)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    elif dataset_name == 'Medical_32':
        # 加载训练集
        train_dataset = ImageFolder(
            root='../Medical_32/data/melanoma_cancer_dataset/train',
            transform=None
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # 加载测试集
        test_dataset = ImageFolder(
            root='../Medical_32/data/melanoma_cancer_dataset/test',
            transform=None
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    return train_dataset, train_loader, test_dataset, test_loader


def generate_verify_subsets(dataset, indices_range, subset_size, replace=False, num_subsets=5):
    """
    从给定范围内随机生成多个不同的子集

    :param dataset: 使用到的数据集
    :param indices_range: 索引范围，例如range(dr_len)
    :param subset_size: 每个子集的大小
    :param replace: 是否允许重复采样，默认False
    :param num_subsets: 要生成的子集数量，默认5
    :return: 包含所有子集的列表
    """
    subsets = []
    for _ in range(num_subsets):
        subset = Subset(dataset, np.random.choice(indices_range, size=subset_size, replace=replace))
        subsets.append(subset)

    return subsets


def generate_verify_subsets_dataloaders(subsets, batch_size=256, shuffle=True, num_workers=-1):
    """
    为多个子集创建DataLoader

    :param subsets: 子集列表，如[Dr1, Dr2, ...]
    :param batch_size: 批次大小，默认256
    :param shuffle: 是否打乱数据，默认True
    :param num_workers: 加载数据的工作线程数，默认12
    :return: 包含所有DataLoader的列表
    """
    dataloaders = []
    for subset in subsets:
        loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        dataloaders.append(loader)

    return dataloaders

def test_model(model, data_loader, device):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total * 100



def organize_facescrub_dataset(source_actor_path, source_actress_path, target_path):
    """
    重新组织FaceScrub数据集文件结构，创建编号文件夹

    Args:
        source_actor_path (str): 男演员图片文件夹路径
        source_actress_path (str): 女演员图片文件夹路径
        target_path (str): 新的目标文件夹路径
    """
    # 确保目标文件夹存在
    os.makedirs(target_path, exist_ok=True)

    # 获取男演员和女演员的文件夹列表
    actor_folders = sorted(os.listdir(source_actor_path))
    actress_folders = sorted(os.listdir(source_actress_path))

    # 处理男演员文件夹 (0000-0264)
    print("处理男演员文件夹...")
    for i, actor_name in enumerate(actor_folders):
        src_folder = os.path.join(source_actor_path, actor_name)
        if not os.path.isdir(src_folder):
            continue

        # 创建目标文件夹 (0开头，四位数字)
        dst_folder = os.path.join(target_path, f"0{i:03d}")
        os.makedirs(dst_folder, exist_ok=True)

        # 复制并调整图片尺寸
        for img_file in os.listdir(src_folder):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_path = os.path.join(src_folder, img_file)
                    with Image.open(img_path) as img:
                        # 调整图片为64x64
                        img = img.resize((64, 64))
                        # 保存到目标文件夹
                        save_path = os.path.join(dst_folder, img_file)
                        img.save(save_path)
                except Exception as e:
                    print(f"处理图片出错 {img_file}: {e}")

    # 处理女演员文件夹 (1000-1264)
    print("处理女演员文件夹...")
    for i, actress_name in enumerate(actress_folders):
        src_folder = os.path.join(source_actress_path, actress_name)
        if not os.path.isdir(src_folder):
            continue

        # 创建目标文件夹 (1开头，四位数字)
        dst_folder = os.path.join(target_path, f"1{i:03d}")
        os.makedirs(dst_folder, exist_ok=True)

        # 复制并调整图片尺寸
        for img_file in os.listdir(src_folder):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_path = os.path.join(src_folder, img_file)
                    with Image.open(img_path) as img:
                        # 调整图片为64x64
                        img = img.resize((64, 64))
                        # 保存到目标文件夹
                        save_path = os.path.join(dst_folder, img_file)
                        img.save(save_path)
                except Exception as e:
                    print(f"处理图片出错 {img_file}: {e}")

    print(f"FaceScrub数据集已重新组织到 {target_path}")
