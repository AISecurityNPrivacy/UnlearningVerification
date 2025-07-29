
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, SVHN, ImageFolder


BBCDATASETFILE = 'bbc_combined.csv'



def get_backdoor_dataset(batch_size=256, num_workers=-1, MAX_LEN=100, dataset='CIFAR10'):
    def dataset_format_convert(dataset_name):
        """
        Configure data transforms based on dataset specifications.
        """
        if dataset_name == 'CIFAR10':
            train_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
            test_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
        elif dataset_name == 'SVHN':
            train_transform = transforms.Compose([
                transforms.ToTensor(),  # SVHN is already 32x32
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        elif dataset_name == 'SkinCancer':
            train_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            test_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            raise ValueError(f"Unsupported dataset transformation: {dataset_name}")
        return train_transform, test_transform

def dataset_format_convert(dataset_name):
    """
    Configure data transforms based on dataset specifications.
    """
    if dataset_name == 'CIFAR10':
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
    elif dataset_name == 'SVHN':
        train_transform = transforms.Compose([
            transforms.ToTensor(),  # SVHN is already 32x32
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif dataset_name == 'SkinCancer':
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError(f"Unsupported dataset transformation: {dataset_name}")
    return train_transform, test_transform

def get_dataset(dataset_name, train_transform, test_transform, batch_size=256, num_workers=-1):
    """
    Load dataset with specified transforms and create DataLoaders.
    """
    if dataset_name == 'CIFAR10':
        train_dataset = CIFAR10(root=f'./{dataset_name}', train=True, download=True, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = CIFAR10(root=f'./{dataset_name}', train=False, download=True, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    elif dataset_name == 'SVHN':
        train_dataset = SVHN(root=f'./{dataset_name}', split='train', download=True, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = SVHN(root=f'./{dataset_name}', split='test', download=True, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    elif dataset_name == 'SkinCancer':
        train_dataset = ImageFolder(
            root='./SkinCancer/train',
            transform=train_transform
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = ImageFolder(
            root='./SkinCancer/test',
            transform=test_transform
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_dataset, train_loader, test_dataset, test_loader


