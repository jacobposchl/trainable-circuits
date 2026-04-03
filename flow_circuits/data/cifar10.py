from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10


CIFAR10_STATS = {
    "mean": (0.4914, 0.4822, 0.4465),
    "std": (0.2470, 0.2435, 0.2616),
}


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int] | None = None) -> None:
        self.dataset = dataset
        self.indices = list(range(len(dataset))) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        image, label = self.dataset[base_idx]
        return image, label, base_idx


def cifar10_transforms(augment: bool) -> transforms.Compose:
    ops: list[object] = []
    if augment:
        ops.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    ops.extend([transforms.ToTensor(), transforms.Normalize(CIFAR10_STATS["mean"], CIFAR10_STATS["std"])])
    return transforms.Compose(ops)


def _split_indices(seed: int) -> dict[str, list[int]]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(50000, generator=generator).tolist()
    return {
        "fit": permutation[:40000],
        "val": permutation[40000:45000],
        "discovery": permutation[45000:50000],
    }


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_cifar10_splits(
    *,
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    seed: int = 0,
    augment_fit: bool = True,
    download: bool = True,
) -> dict[str, DataLoader]:
    train_aug = CIFAR10(data_dir, train=True, download=download, transform=cifar10_transforms(augment_fit))
    train_eval = CIFAR10(data_dir, train=True, download=download, transform=cifar10_transforms(False))
    test_ds = CIFAR10(data_dir, train=False, download=download, transform=cifar10_transforms(False))
    split_indices = _split_indices(seed)
    fit_generator = torch.Generator().manual_seed(seed)

    fit_loader = DataLoader(
        IndexedDataset(train_aug, split_indices["fit"]),
        batch_size=batch_size,
        shuffle=True,
        generator=fit_generator,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
    )
    val_loader = DataLoader(
        IndexedDataset(train_eval, split_indices["val"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
    )
    discovery_loader = DataLoader(
        IndexedDataset(train_eval, split_indices["discovery"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
    )
    test_loader = DataLoader(
        IndexedDataset(test_ds),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
    )
    return {
        "fit": fit_loader,
        "val": val_loader,
        "discovery": discovery_loader,
        "test": test_loader,
    }
