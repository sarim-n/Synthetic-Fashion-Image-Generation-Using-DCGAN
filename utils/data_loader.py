"""
Fashion-MNIST Data Pipeline for Synthetic Fashion DCGAN.
Handles downloading, transformation (ToTensor + Normalize), and DataLoader creation.
"""

from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import BATCH_SIZE, DATA_DIR


def get_data_loader(
    data_dir: str = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    download: bool = True,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Tuple[datasets.FashionMNIST, DataLoader]:
    """
    Constructs and returns the Fashion-MNIST training dataset and PyTorch DataLoader.

    Args:
        data_dir (str): Path to directory where raw dataset will be stored.
        batch_size (int): Batch size for DataLoader. Defaults to BATCH_SIZE from config.
        download (bool): If True, downloads dataset if not present.
        shuffle (bool): If True, shuffles dataset every epoch.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        Tuple[datasets.FashionMNIST, DataLoader]: (dataset instance, dataloader instance)
    """
    # Define transformations:
    # 1. ToTensor(): Converts PIL image to FloatTensor in range [0.0, 1.0], shape [1, 28, 28]
    # 2. Normalize((0.5,), (0.5,)): Normalizes grayscale values to range [-1.0, 1.0]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Load or download Fashion-MNIST dataset
    dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform,
    )

    # Create PyTorch DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )

    return dataset, dataloader
