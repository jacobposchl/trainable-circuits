from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.datasets import CIFAR10


CIFAR10_STATS = {
    "mean": (0.4914, 0.4822, 0.4465),
    "std": (0.2470, 0.2435, 0.2616),
}

CIFAR10_CORRUPTION_NAMES = (
    "gaussian_noise",
    "gaussian_blur",
    "contrast",
    "brightness",
    "pixelate",
    "occlusion",
)


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


class CorruptedIndexedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        *,
        corruption_name: str,
        severity: int,
        indices: list[int] | None = None,
    ) -> None:
        self.dataset = dataset
        self.corruption_name = str(corruption_name)
        self.severity = int(severity)
        self.indices = list(range(len(dataset))) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        image, label = self.dataset[base_idx]
        tensor = TF.to_tensor(image)
        corrupted = apply_cifar10_corruption(
            tensor,
            corruption_name=self.corruption_name,
            severity=self.severity,
            seed=base_idx,
        )
        normalized = TF.normalize(corrupted, CIFAR10_STATS["mean"], CIFAR10_STATS["std"])
        return normalized, label, base_idx


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


def apply_cifar10_corruption(
    image: torch.Tensor,
    *,
    corruption_name: str,
    severity: int,
    seed: int | None = None,
) -> torch.Tensor:
    severity = int(severity)
    if severity < 1 or severity > 5:
        raise ValueError(f"severity must be between 1 and 5, got {severity}")
    corruption_name = str(corruption_name)
    if corruption_name not in CIFAR10_CORRUPTION_NAMES:
        raise ValueError(f"Unknown corruption_name: {corruption_name}")

    corrupted = image.detach().clone().float().clamp(0.0, 1.0)
    rng = np.random.default_rng(None if seed is None else int(seed) + (97 * severity))

    if corruption_name == "gaussian_noise":
        sigma = [0.04, 0.06, 0.08, 0.10, 0.12][severity - 1]
        noise = torch.from_numpy(rng.normal(0.0, sigma, size=tuple(corrupted.shape))).to(dtype=corrupted.dtype)
        corrupted = corrupted + noise
    elif corruption_name == "gaussian_blur":
        kernel_size = [3, 3, 5, 5, 7][severity - 1]
        sigma = [0.45, 0.65, 0.85, 1.05, 1.25][severity - 1]
        corrupted = TF.gaussian_blur(corrupted, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    elif corruption_name == "contrast":
        factor = [0.75, 0.60, 0.45, 0.30, 0.15][severity - 1]
        mean = corrupted.mean(dim=(1, 2), keepdim=True)
        corrupted = (corrupted - mean) * factor + mean
    elif corruption_name == "brightness":
        factor = [0.90, 0.80, 0.70, 0.60, 0.50][severity - 1]
        corrupted = corrupted * factor
    elif corruption_name == "pixelate":
        side = [28, 24, 20, 16, 12][severity - 1]
        reduced = F.interpolate(corrupted.unsqueeze(0), size=(side, side), mode="bilinear", align_corners=False)
        corrupted = F.interpolate(reduced, size=(32, 32), mode="nearest").squeeze(0)
    else:  # occlusion
        patch = [4, 6, 8, 10, 12][severity - 1]
        max_offset = max(1, 32 - patch + 1)
        y0 = int(rng.integers(0, max_offset))
        x0 = int(rng.integers(0, max_offset))
        corrupted[:, y0 : y0 + patch, x0 : x0 + patch] = 0.0

    return corrupted.clamp(0.0, 1.0)


def _split_indices(seed: int) -> dict[str, list[int]]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(50000, generator=generator).tolist()
    return {
        "fit": permutation[:40000],
        "val": permutation[40000:45000],
        "discovery": permutation[45000:50000],
    }


def _backbone_split_indices(seed: int, val_size: int = 5000) -> dict[str, list[int]]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(50000, generator=generator).tolist()
    train_size = max(0, 50000 - val_size)
    return {
        "train": permutation[:train_size],
        "val": permutation[train_size:],
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


def build_supervised_cifar10_loaders(
    *,
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    seed: int = 0,
    augment_train: bool = True,
    download: bool = True,
    val_size: int = 5000,
) -> dict[str, DataLoader]:
    train_aug = CIFAR10(data_dir, train=True, download=download, transform=cifar10_transforms(augment_train))
    train_eval = CIFAR10(data_dir, train=True, download=download, transform=cifar10_transforms(False))
    test_ds = CIFAR10(data_dir, train=False, download=download, transform=cifar10_transforms(False))
    split_indices = _backbone_split_indices(seed, val_size=val_size)
    train_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        IndexedDataset(train_aug, split_indices["train"]),
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
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
    test_loader = DataLoader(
        IndexedDataset(test_ds),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
    )
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


def build_cifar10_corruption_splits(
    *,
    data_dir: str,
    batch_size: int,
    corruption_name: str,
    severity: int,
    num_workers: int = 4,
    seed: int = 0,
    augment_fit: bool = True,
    download: bool = True,
) -> dict[str, DataLoader]:
    train_aug = CIFAR10(data_dir, train=True, download=download, transform=cifar10_transforms(augment_fit))
    train_raw = CIFAR10(data_dir, train=True, download=download, transform=None)
    test_raw = CIFAR10(data_dir, train=False, download=download, transform=None)
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
        CorruptedIndexedDataset(
            train_raw,
            corruption_name=corruption_name,
            severity=severity,
            indices=split_indices["val"],
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
    )
    test_loader = DataLoader(
        CorruptedIndexedDataset(
            test_raw,
            corruption_name=corruption_name,
            severity=severity,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
    )
    return {
        "fit": fit_loader,
        "val": val_loader,
        "test": test_loader,
    }
