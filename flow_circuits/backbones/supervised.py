from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from flow_circuits.backbones.resnet import build_cifar_resnet_classifier
from flow_circuits.data import build_supervised_cifar10_loaders
from flow_circuits.utils import seed_everything


@dataclass
class SupervisedBackboneSummary:
    arch: str
    best_epoch: int
    best_val_accuracy: float
    test_accuracy: float
    output_path: str

    def to_dict(self) -> dict:
        return asdict(self)


class SupervisedBackboneTrainer:
    def __init__(
        self,
        *,
        arch: str,
        data_dir: str,
        output_path: str | Path,
        batch_size: int = 128,
        num_workers: int = 4,
        seed: int = 0,
        pretrained: bool = True,
        download: bool = True,
        val_size: int = 5000,
        epochs: int = 15,
        lr: float = 0.05,
        momentum: float = 0.9,
        weight_decay: float = 5.0e-4,
    ) -> None:
        self.arch = arch
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        seed_everything(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_cifar_resnet_classifier(
            arch=arch,
            pretrained=pretrained,
            num_classes=10,
            weights_path=None,
        ).to(self.device)
        self.loaders = build_supervised_cifar10_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            augment_train=True,
            download=download,
            val_size=val_size,
        )
        self.epochs = epochs
        self.optimizer = SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(epochs, 1))
        self.criterion = nn.CrossEntropyLoss()

    def train(self) -> SupervisedBackboneSummary:
        best_val_accuracy = float("-inf")
        best_epoch = 0
        best_state = None
        for epoch in range(1, self.epochs + 1):
            self._run_epoch(self.loaders["train"], train=True)
            val_accuracy = self._run_epoch(self.loaders["val"], train=False)
            self.scheduler.step()
            print(f"[backbone epoch {epoch}/{self.epochs}]  val_acc={val_accuracy:.4f}", flush=True)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}

        if best_state is None:
            raise RuntimeError("Backbone training did not produce a checkpoint state.")

        self.model.load_state_dict(best_state)
        test_accuracy = self._run_epoch(self.loaders["test"], train=False)
        checkpoint = {
            "arch": self.arch,
            "num_classes": 10,
            "state_dict": self.model.state_dict(),
            "best_epoch": best_epoch,
            "best_val_accuracy": float(best_val_accuracy),
            "test_accuracy": float(test_accuracy),
            "seed": int(self.seed),
        }
        torch.save(checkpoint, self.output_path)
        return SupervisedBackboneSummary(
            arch=self.arch,
            best_epoch=best_epoch,
            best_val_accuracy=float(best_val_accuracy),
            test_accuracy=float(test_accuracy),
            output_path=str(self.output_path),
        )

    def _run_epoch(self, loader, *, train: bool) -> float:
        self.model.train(train)
        correct = 0
        total = 0
        for images, labels, _ in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            if train:
                self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            if train:
                loss.backward()
                self.optimizer.step()
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
        return float(correct / max(total, 1))
