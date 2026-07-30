# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 22:36:47 2026

@author: salim
"""

from torchvision import (
    transforms,
    datasets,
)
from torch.utils.data import (
    DataLoader, 
    DistributedSampler, # distributes data across differents processes
)


def get_transforms():
    """
    Create the transformations applied to MNIST images.

    Returns
    -------
    torchvision.transforms.Compose
        Transformations applied to every image.
    """
    
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.1307,),
                std=(0.3081,)
            ),
        ]
    )
    
    return transform
    
    
def get_datasets(
        root: str,
        transform,
):
    """
    Create MNIST training and test datasets.

    Parameters
    ----------
    root : str
        Directory where MNIST data is stored.

    transform :
        Transformations applied to images.

    Returns
    -------
    tuple
        Training dataset and test dataset.
    """

    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        transform=transform,
        download=True,
    )
    
    return train_dataset, test_dataset


def get_dataloaders(
    batch_size: int,
    num_workers: int = 0,
    root: str = "./data",
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    """
    Create MNIST dataloaders.

    Parameters
    ----------
    batch_size : int
        Number of samples per batch.

    num_workers : int
        Number of subprocesses used for data loading.

    root : str
        Directory where MNIST is stored.

    Returns
    -------
    tuple
        Training and test dataloaders.
    """

    transform = get_transforms()

    train_dataset, test_dataset = get_datasets(
        root=root,
        transform=transform,
    )
    
    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader, train_sampler
