"""
Multi-dataset loaders for generalization experiments.

All outputs are resized to 32×32 so they are compatible with backbones whose
compression modules were built with a 32×32 dummy input.  This allows a model
trained on CIFAR-10 to be evaluated on other datasets without modification.

Supported datasets: 'cifar10', 'cifar100', 'stl10'
"""
from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, STL10


# Per-dataset normalization statistics
_STATS: dict[str, dict] = {
    "cifar10":  {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616)},
    "cifar100": {"mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761)},
    "stl10":    {"mean": (0.4467, 0.4398, 0.4066), "std": (0.2242, 0.2215, 0.2239)},
}

_N_CLASSES: dict[str, int] = {
    "cifar10": 10, "cifar100": 100, "stl10": 10,
}


def get_loaders(
    dataset: str,
    data_dir: str,
    batch_size: int = 256,
    num_workers: int = 4,
    augment: bool = True,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders for the specified dataset.

    All images are resized to 32×32 to match backbones built with a 32×32
    dummy input (e.g. ResNet trained on CIFAR-10).

    Args:
        dataset:     One of 'cifar10', 'cifar100', 'stl10' (case-insensitive).
        data_dir:    Root directory for dataset storage.
        batch_size:  Batch size for both loaders.
        num_workers: DataLoader worker count.
        augment:     Apply random crop + flip augmentations to training split.
        download:    Download dataset if not already present.

    Returns:
        (train_loader, val_loader)
    """
    key = dataset.lower().replace("-", "")
    if key not in _STATS:
        raise ValueError(
            f"Unsupported dataset: {dataset!r}. "
            f"Choose from: {list(_STATS)}"
        )

    mean, std = _STATS[key]["mean"], _STATS[key]["std"]
    # STL-10 images are 96×96 — resize to 32×32 for backbone compatibility
    native_size = 96 if key == "stl10" else 32
    train_tf = _build_transform(mean, std, native_size=native_size, augment=augment)
    val_tf   = _build_transform(mean, std, native_size=native_size, augment=False)

    if key == "cifar10":
        train_ds = CIFAR10(data_dir, train=True,  transform=train_tf, download=download)
        val_ds   = CIFAR10(data_dir, train=False, transform=val_tf,   download=download)
    elif key == "cifar100":
        train_ds = CIFAR100(data_dir, train=True,  transform=train_tf, download=download)
        val_ds   = CIFAR100(data_dir, train=False, transform=val_tf,   download=download)
    elif key == "stl10":
        train_ds = STL10(data_dir, split="train", transform=train_tf, download=download)
        val_ds   = STL10(data_dir, split="test",  transform=val_tf,   download=download)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def n_classes(dataset: str) -> int:
    """Return the number of classes for the given dataset name."""
    key = dataset.lower().replace("-", "")
    if key not in _N_CLASSES:
        raise ValueError(f"Unsupported dataset: {dataset!r}")
    return _N_CLASSES[key]


def _build_transform(
    mean: tuple, std: tuple, native_size: int = 32, augment: bool = True
) -> transforms.Compose:
    ops: list = []
    if native_size != 32:
        ops.append(transforms.Resize((32, 32)))
    if augment:
        ops += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)
