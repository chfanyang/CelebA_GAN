"""Dataset loading for CelebA, Fashion-MNIST, CIFAR-10, and Places2 subset."""

import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.datasets as tv_datasets
import torchvision.transforms as transforms


class Places2Dataset(Dataset):
    """Local Places2 subset dataset.

    Recursively reads jpg/png/jpeg images from a directory.
    Supports --max_samples to limit the number of images.
    """

    def __init__(self, root: str, image_size: int = 128,
                 max_samples: int = None):
        self.root = Path(root)
        self.image_size = image_size

        # Recursively find all image files
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(str(self.root / "**" / ext),
                                              recursive=True))

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {root}. "
                               f"Supported: jpg, jpeg, png")

        # Limit samples if specified
        if max_samples is not None and max_samples > 0:
            self.image_paths = self.image_paths[:max_samples]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Return a random image on error
            new_idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(new_idx)
        return self.transform(image), 0


def get_fashion_mnist(root: str, image_size: int = 32,
                      train: bool = True):
    """Get Fashion-MNIST dataset."""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    dataset = tv_datasets.FashionMNIST(
        root=root, train=train, download=True, transform=transform
    )
    return dataset


def get_cifar10(root: str, image_size: int = 32,
                train: bool = True):
    """Get CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.Resize(image_size) if image_size != 32 else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    dataset = tv_datasets.CIFAR10(
        root=root, train=train, download=True, transform=transform
    )
    return dataset


def get_celeba(root: str, image_size: int = 128,
                train: bool = True, max_samples: int = None):
    """Get CelebA dataset for face image inpainting.

    Uses torchvision.datasets.CelebA with automatic download.
    Falls back with a clear message if download fails.

    Preprocessing: CenterCrop → Resize → ToTensor → Normalize to [-1, 1].

    Args:
        root: data root directory
        image_size: target image size (default 128)
        train: True for training set, False for validation/test
        max_samples: if set, limit to this many samples

    Returns:
        torch.utils.data.Dataset
    """
    split = "train" if train else "valid"

    transform = transforms.Compose([
        transforms.CenterCrop(128),  # CelebA images are 178×218; crop to square first
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    try:
        dataset = tv_datasets.CelebA(
            root=root, split=split, target_type=[],
            transform=transform, download=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load CelebA dataset from {root}.\n"
            f"Error: {e}\n\n"
            f"CelebA automatic download may fail due to Google Drive quota limits.\n"
            f"Please manually download CelebA:\n"
            f"  1. Download img_align_celeba.zip from the official source\n"
            f"  2. Extract to {root}/celeba/img_align_celeba/\n"
            f"  3. Download list_eval_partition.txt and identity_CelebA.txt\n"
            f"  4. Place them in {root}/celeba/\n"
            f"Or visit: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html\n"
        )

    if max_samples is not None and max_samples > 0 and len(dataset) > max_samples:
        indices = list(range(max_samples))
        dataset = Subset(dataset, indices)

    return dataset


def get_places2(root: str, image_size: int = 128,
                max_samples: int = None, train: bool = True):
    """Get Places2 local subset dataset.

    Note: train/test split is handled externally via random split
    if the dataset doesn't have pre-defined splits.
    """
    dataset = Places2Dataset(
        root=root, image_size=image_size, max_samples=max_samples
    )
    return dataset


def get_dataloader(name: str, root: str, image_size: int,
                   batch_size: int, train: bool = True,
                   max_samples: int = None,
                   num_workers: int = 4) -> DataLoader:
    """Factory function to get a DataLoader for a dataset.

    Args:
        name: "celeba", "fashion_mnist", "cifar10", or "places2"
        root: data root directory
        image_size: target image size
        batch_size: batch size
        train: True for training set, False for test set
        max_samples: max samples (for Places2/CelebA)
        num_workers: number of data loading workers

    Returns:
        torch.utils.data.DataLoader
    """
    if name == "celeba":
        dataset = get_celeba(root, image_size, train, max_samples)
    elif name == "fashion_mnist":
        dataset = get_fashion_mnist(root, image_size, train)
    elif name == "cifar10":
        dataset = get_cifar10(root, image_size, train)
    elif name == "places2":
        dataset = get_places2(root, image_size, max_samples, train)
        # For Places2, split train/val if needed
        if not train:
            # Use a portion as test set
            total = len(dataset)
            val_size = int(total * 0.1)
            indices = list(range(total - val_size, total))
            dataset = torch.utils.data.Subset(dataset, indices)
        else:
            total = len(dataset)
            val_size = int(total * 0.1)
            indices = list(range(0, total - val_size))
            dataset = torch.utils.data.Subset(dataset, indices)
    else:
        raise ValueError(f"Unknown dataset: {name}. "
                         f"Expected 'celeba', 'fashion_mnist', 'cifar10', or 'places2'.")

    shuffle = train
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        drop_last=True,
    )
    return dataloader
