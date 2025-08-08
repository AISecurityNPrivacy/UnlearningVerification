import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torch import optim

from load import get_dataset, get_backdoor_dataset, get_vocabsize

import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, dataset_name, num_classes, backdoor=False):
        super().__init__()
        self.embedding = nn.Embedding(get_vocabsize(dataset=dataset_name, backdoor=backdoor), 100, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 100, (k, 100)) for k in [3, 4, 5]
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(100 * len([3, 4, 5]), num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [B, L, E]
        x = x.unsqueeze(1)     # [B, 1, L, E]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(B, C, Lk), ...]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        return self.fc(x)

class TextSCNN(nn.Module):
    def __init__(self, dataset_name, num_classes):
        super().__init__()
        embed_dim = 100
        self.embedding = nn.Embedding(get_vocabsize(dataset=dataset_name), 100, padding_idx=0)

        self.conv = nn.Conv2d(1, 100, (4, embed_dim))

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [B, L, E]
        x = x.unsqueeze(1)  # [B, 1, L, E]
        x = F.relu(self.conv(x))  # [B, 100, L-3, 1]
        x = x.squeeze(3)  # [B, 100, L-3]
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # [B, 100]
        x = self.dropout(x)
        return self.fc(x)


class TextRCNN(nn.Module):
    def __init__(self, dataset_name, num_classes, hidden_size=128):
        super().__init__()
        embed_dim = 100
        self.embedding = nn.Embedding(get_vocabsize(dataset=dataset_name), 100, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, 100, (k, embed_dim)) for k in [3, 4, 5]
        ])

        self.rnn = nn.GRU(
            input_size=300,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)         # [B, L, E]
        x = x.unsqueeze(1)            # [B, 1, L, E]

        conv_outs = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(B, C, L'), ...]
        max_len = max([out.size(2) for out in conv_outs])
        padded_outs = [
            F.pad(out, pad=(0, max_len - out.size(2)), mode='constant', value=0)
            for out in conv_outs
        ]
        conv_cat = torch.cat(padded_outs, dim=1)                           # [B, C_total, L']
        conv_cat = conv_cat.permute(0, 2, 1)
        _, hn = self.rnn(conv_cat)
        hn_cat = torch.cat((hn[0], hn[1]), dim=1)  # [B, hidden_size * 2]
        out = self.dropout(hn_cat)
        return self.fc(out)


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


def load_dataset(dataset_name, batch_size=256, num_workers=-1):

    train_dataset, train_loader, test_dataset, test_loader = get_dataset(batch_size=batch_size,
                                                                         num_workers=num_workers,
                                                                         dataset=dataset_name)
    return train_dataset, train_loader, test_dataset, test_loader


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


