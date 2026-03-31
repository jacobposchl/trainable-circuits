import torch
import pytest
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchvision import transforms

from data import datasets


class DummyDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __len__(self):
        return 6

    def __getitem__(self, index):
        return torch.zeros(3, 32, 32), index


def _patch_dataset(monkeypatch, attr_name, bucket):
    def factory(*args, **kwargs):
        dataset = DummyDataset(*args, **kwargs)
        bucket.append(dataset)
        return dataset

    monkeypatch.setattr(datasets, attr_name, factory)


def test_n_classes_supports_case_and_hyphen_normalization():
    assert datasets.n_classes("CIFAR-10") == 10
    assert datasets.n_classes("cifar100") == 100
    assert datasets.n_classes("stl10") == 10


def test_n_classes_rejects_unknown_dataset():
    with pytest.raises(ValueError):
        datasets.n_classes("imagenet")


def test_build_transform_adds_resize_for_non_native_inputs():
    transform = datasets._build_transform((0.1, 0.2, 0.3), (1.0, 1.0, 1.0), native_size=96, augment=False)
    assert isinstance(transform.transforms[0], transforms.Resize)
    assert isinstance(transform.transforms[-2], transforms.ToTensor)
    assert isinstance(transform.transforms[-1], transforms.Normalize)


def test_build_transform_adds_augmentations_for_training():
    transform = datasets._build_transform((0.1, 0.2, 0.3), (1.0, 1.0, 1.0), native_size=32, augment=True)
    names = [type(op).__name__ for op in transform.transforms]
    assert names[:3] == ["RandomCrop", "RandomHorizontalFlip", "ColorJitter"]


def test_get_loaders_builds_cifar10_loaders(monkeypatch):
    created = []
    _patch_dataset(monkeypatch, "CIFAR10", created)

    train_loader, val_loader = datasets.get_loaders(
        "cifar-10",
        data_dir="unused",
        batch_size=2,
        num_workers=0,
        augment=True,
        download=False,
    )

    assert len(created) == 2
    assert created[0].kwargs["train"] is True
    assert created[1].kwargs["train"] is False
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)
    assert train_loader.batch_size == 2
    assert val_loader.batch_size == 2


def test_get_loaders_builds_cifar100_loaders(monkeypatch):
    created = []
    _patch_dataset(monkeypatch, "CIFAR100", created)

    train_loader, val_loader = datasets.get_loaders(
        "cifar100",
        data_dir="unused",
        batch_size=3,
        num_workers=0,
        augment=False,
        download=False,
    )

    assert len(created) == 2
    assert created[0].kwargs["train"] is True
    assert created[1].kwargs["train"] is False
    assert train_loader.batch_size == 3
    assert val_loader.batch_size == 3


def test_get_loaders_builds_stl10_loaders_with_expected_splits(monkeypatch):
    created = []
    _patch_dataset(monkeypatch, "STL10", created)

    train_loader, val_loader = datasets.get_loaders(
        "stl10",
        data_dir="unused",
        batch_size=4,
        num_workers=0,
        augment=False,
        download=False,
    )

    assert len(created) == 2
    assert created[0].kwargs["split"] == "train"
    assert created[1].kwargs["split"] == "test"
    assert isinstance(created[0].kwargs["transform"].transforms[0], transforms.Resize)
    assert train_loader.batch_size == 4
    assert val_loader.batch_size == 4


def test_get_loaders_rejects_unknown_dataset():
    with pytest.raises(ValueError):
        datasets.get_loaders("imagenet", data_dir="unused", download=False)
