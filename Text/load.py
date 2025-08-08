
import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import os

BBCDATASETFILE = 'bbc_combined.csv'

# 自定义 Dataset
class TextDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

    def __setitem__(self, idx, value):
        input_ids, label = value
        if isinstance(self.inputs, torch.Tensor):
            self.inputs[idx] = torch.tensor(input_ids, dtype=torch.long)
        else:
            self.inputs[idx] = input_ids
        if isinstance(self.labels, torch.Tensor):
            self.labels[idx] = torch.tensor(label, dtype=torch.long)
        else:
            self.labels[idx] = label


def get_backdoor_dataset(batch_size=256, num_workers=-1, MAX_LEN=100, dataset='BBCNews'):
    root = os.path.dirname(__file__)
    if dataset == 'BBCNews':
        df = pd.read_csv(os.path.join(root, BBCDATASETFILE))
    elif dataset == 'AGNews':
        splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
        df = pd.read_parquet("hf://datasets/wangrongsheng/ag_news/" + splits["train"])
    elif dataset == 'IMDb':
        splits = {'train': 'plain_text/train-00000-of-00001.parquet', 'test': 'plain_text/test-00000-of-00001.parquet',
                  'unsupervised': 'plain_text/unsupervised-00000-of-00001.parquet'}
        df = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["train"])

    def tokenize(text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.split()

    df["tokens"] = df["text"].apply(tokenize)

    counter = Counter(token for tokens in df["tokens"] for token in tokens)
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.items():
        if freq >= 2:
            vocab[word] = len(vocab)

    trigger_token = "cftrigger"
    vocab[trigger_token] = len(vocab)

    def encode(tokens):
        return [vocab.get(token, vocab["<unk>"]) for token in tokens]

    def pad(seq, max_len=100):
        return seq[:max_len] + [vocab["<pad>"]] * max(0, max_len - len(seq))

    df["input_ids"] = df["tokens"].apply(lambda x: pad(encode(x)))

    label_encoder = LabelEncoder()
    df["label_id"] = label_encoder.fit_transform(df["label"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["input_ids"].tolist(),
        df["label_id"].tolist(),
        test_size=0.2,
        random_state=42
    )

    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_dataset, train_loader, test_dataset, test_loader, vocab[trigger_token]


def get_dataset(batch_size=256, num_workers=-1, MAX_LEN=100, dataset='BBCNews'):
    root = os.path.dirname(__file__)
    if dataset == 'BBCNews':
        df = pd.read_csv(os.path.join(root, BBCDATASETFILE))
    elif dataset == 'AGNews':
        splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
        df = pd.read_parquet("hf://datasets/wangrongsheng/ag_news/" + splits["train"])
    elif dataset == 'IMDb':
        splits = {'train': 'plain_text/train-00000-of-00001.parquet', 'test': 'plain_text/test-00000-of-00001.parquet',
                  'unsupervised': 'plain_text/unsupervised-00000-of-00001.parquet'}
        df = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["train"])

    def tokenize(text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.split()

    df["tokens"] = df["text"].apply(tokenize)

    counter = Counter(token for tokens in df["tokens"] for token in tokens)
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.items():
        if freq >= 2:
            vocab[word] = len(vocab)

    trigger_token = "cftrigger"
    vocab[trigger_token] = len(vocab)

    # 编码 & 填充
    def encode(tokens):
        return [vocab.get(token, vocab["<unk>"]) for token in tokens]

    def pad(seq, max_len=100):
        return seq[:max_len] + [vocab["<pad>"]] * max(0, max_len - len(seq))

    df["input_ids"] = df["tokens"].apply(lambda x: pad(encode(x)))

    label_encoder = LabelEncoder()
    df["label_id"] = label_encoder.fit_transform(df["label"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["input_ids"].tolist(),
        df["label_id"].tolist(),
        test_size=0.2,
        random_state=42
    )

    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_dataset, train_loader, test_dataset, test_loader


def get_vocabsize(backdoor=False, dataset='BBCNews'):
    root = os.path.dirname(__file__)
    if dataset == 'BBCNews':
        df = pd.read_csv(os.path.join(root, BBCDATASETFILE))
    elif dataset == 'AGNews':
        splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
        df = pd.read_parquet("hf://datasets/wangrongsheng/ag_news/" + splits["train"])
    elif dataset == 'IMDb':
        splits = {'train': 'plain_text/train-00000-of-00001.parquet', 'test': 'plain_text/test-00000-of-00001.parquet',
                  'unsupervised': 'plain_text/unsupervised-00000-of-00001.parquet'}
        df = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["train"])

    def tokenize(text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.split()

    df["tokens"] = df["text"].apply(tokenize)

    counter = Counter(token for tokens in df["tokens"] for token in tokens)
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.items():
        if freq >= 2:
            vocab[word] = len(vocab)
    if backdoor:
        trigger_token = "cftrigger"
        vocab[trigger_token] = len(vocab)
    return len(vocab)
