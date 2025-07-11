# Imports
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import random

# Function that gets transformation for the CIFAR10 set
def _get_cifar_transforms(augment=True):
    transform_augment = transforms.Compose([
        transforms.Pad(padding=4, fill=(125,123,113)),
        transforms.RandomCrop(32, padding=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_train = transform_augment if augment else transform_test

    return transform_train, transform_test

# Function that defines the CIFAR10 set used in experiments
def cifar10(root, augment=False):
    transform_train, transform_test = _get_cifar_transforms(augment=augment)
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    return train_set, test_set

# Function that defines the CIFAR100 set used in experiments
def cifar100(root, augment=False):
    transform_train, transform_test = _get_cifar_transforms(augment=augment)
    train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    return train_set, test_set

# Dataset class needed for compatibility with the 3rd party SCRUB repo
class CustomDataset(Dataset):
    def __init__(self, data, targets, indices=None, transform=None, target_transform=None):
        if indices is not None:
            self.data = data[indices]
            self.targets = np.array(targets)[indices]
        else:
            self.data = data
            self.targets = np.array(targets)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def forget_retain_split(dataset, forget_classes=[]):
    # Convert targets to numpy array for easy manipulation
    targets = np.array(dataset.targets)
    
    # Find indices of the classes to forget
    forget_indices = np.where(np.isin(targets, forget_classes))[0]
    retain_indices = np.where(~np.isin(targets, forget_classes))[0]
    
    # Create subsets for forget and retain
    forget_subset = CustomDataset(dataset.data, dataset.targets, forget_indices, transform=dataset.transform)
    retain_subset = CustomDataset(dataset.data, dataset.targets, retain_indices, transform=dataset.transform)

    return forget_subset, retain_subset

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_dataset(root, dataset_name="cifar10"):
    train_set, test_set = None, None

    if dataset_name == "cifar10":
        train_set, test_set = cifar10(root, augment=True)
    elif dataset_name == "cifar100":
        train_set, test_set = cifar100(root, augment=True)

    return train_set, test_set

def get_loaders(root, dataset, forget_classes, validation_split=0.2, batch_size=256, num_workers=2, seed=1):
    # Set the seed for repeatable dataloaders
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    # Load CIFAR-10 dataset
    train_set, test_set = get_dataset(root, dataset_name=dataset)
    if train_set is None or test_set is None:
        raise RuntimeError(f"ERROR: {dataset} is an undefined dataset name!")

    # Creating full train, valid and test dataloaders
    num_train = len(train_set)
    split_idx = int(num_train * (1 - validation_split))
    train_data, train_targets = train_set.data[:split_idx], train_set.targets[:split_idx]
    val_data, val_targets = train_set.data[split_idx:], train_set.targets[split_idx:]
    train_full_subset = CustomDataset(train_data, train_targets, transform=train_set.transform)
    val_full_subset = CustomDataset(val_data, val_targets, transform=train_set.transform)
    test_full_subset = CustomDataset(test_set.data, test_set.targets, transform=test_set.transform)
    train_loader = DataLoader(train_full_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator)
    valid_loader = DataLoader(val_full_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_full_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Creating train forget/retain loaders
    train_forget, train_retain = forget_retain_split(train_set, forget_classes)
    train_forget_loader = DataLoader(train_forget, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator)
    train_retain_loader = DataLoader(train_retain, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator)

    # Creating valid forget/retain loaders 
    valid_forget, valid_retain = forget_retain_split(val_full_subset, forget_classes)
    valid_forget_loader = DataLoader(valid_forget, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    valid_retain_loader = DataLoader(valid_retain, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Creating test forget/retain loaders
    test_forget, test_retain = forget_retain_split(test_full_subset, forget_classes)
    test_forget_loader = DataLoader(test_forget, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_retain_loader = DataLoader(test_retain, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Create dictionary for easy access of loaders
    loaders = {
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'test_loader': test_loader,
        'train_forget_loader': train_forget_loader,
        'train_retain_loader': train_retain_loader,
        'valid_forget_loader': valid_forget_loader,
        'valid_retain_loader': valid_retain_loader,
        'test_forget_loader': test_forget_loader,
        'test_retain_loader': test_retain_loader
    }
    
    # Returning all loaders
    return loaders
