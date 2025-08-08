import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from load import *
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, dataset_name, num_classes=10):
        super(CNN, self).__init__()

        # Select input channels based on dataset
        if dataset_name == 'CIFAR10' or dataset_name == 'SVHN':
            in_channels = 3  # RGB images
        elif dataset_name == 'SkinCancer':
            in_channels = 3  # RGB images
            num_classes = 2  # Binary classification
        else:
            in_channels = 1  # Grayscale images

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Adjust fully connected layer based on dataset input size

        if dataset_name == 'SkinCancer':
            self.fc1 = nn.Linear(64 * 8 * 8, 128)  # 32x32 -> 8x8 after pooling
        else:
            self.fc1 = nn.Linear(64 * 8 * 8, 128)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        if dataset_name == 'SkinCancer':
            self.fc2 = nn.Linear(128, num_classes)
        else:
            self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten while preserving batch size
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, dataset_name, num_classes=None):
        super(ResNet18, self).__init__()

        # 根据数据集选择输入通道数和类别数
        if dataset_name == 'CIFAR10' or dataset_name == 'SVHN':
            in_channels = 3  # RGB图像
            num_classes = 10
        elif dataset_name == 'facescrub':
            in_channels = 3  # RGB图像
            num_classes = 530  # FaceScrub有530个类别
        elif dataset_name == 'SkinCancer':
            in_channels = 3  # RGB图像
            num_classes = 2  # 二分类
        else:
            in_channels = 1  # 灰度图像

        self.in_planes = 64

        # 针对32x32输入优化的第一层
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet18的层结构
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # 全连接层
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)  # 对于32x32输入，最终特征图是4x4
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class SimpleCNN(nn.Module):
    def __init__(self, dataset_name, num_classes=None):
        super(SimpleCNN, self).__init__()
        if dataset_name == 'CIFAR10' or dataset_name == 'SVHN':
            in_channels = 3
            num_classes = 10
        elif dataset_name == 'facescrub':
            in_channels = 3
            num_classes = 530
        elif dataset_name == 'SkinCancer':
            in_channels = 3
            num_classes = 2
        else:
            in_channels = 1

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)

        if dataset_name == 'facescrub':
            fc_input_size = 16 * 16 * 16
        elif dataset_name == 'SkinCancer':
            fc_input_size = 16 * 8 * 8
        else:
            fc_input_size = 16 * 8 * 8

        self.relu = nn.ReLU()
        self.fc = nn.Linear(fc_input_size, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



def seed_setting(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def load_backdoor_dataset(dataset_name, batch_size=256, num_workers=-1):

    train_dataset, train_loader, test_dataset, test_loader, tigger_id = get_backdoor_dataset(batch_size=batch_size,
                                                                                             num_workers=num_workers,
                                                                                             dataset=dataset_name)
    return train_dataset, train_loader, test_dataset, test_loader, tigger_id



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


