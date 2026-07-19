import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

_MEAN = (0.5071, 0.4867, 0.4408)
_STD  = (0.2675, 0.2565, 0.2761)


def get_cifar100_loaders(batch_size: int, data_dir: str):
    """Return (train_loader, test_loader) for CIFAR-100 with standard augmentation."""
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

    train_set = torchvision.datasets.CIFAR100(
        root=data_dir, train=True,  download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(
        test_set,  batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available())

    print(f'Train: {len(train_set):,} images | Test: {len(test_set):,} images')
    print(f'Classes: {train_set.classes[:5]} ... ({len(train_set.classes)} total)')
    return train_loader, test_loader


def get_test_dataset(data_dir: str):
    """Return the raw (un-normalized) test set for image complexity scoring."""
    test_set = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True,
        transform=transforms.ToTensor())
    return test_set
