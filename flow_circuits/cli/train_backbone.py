from __future__ import annotations

import argparse
import json

from flow_circuits.backbones import SupervisedBackboneTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a supervised CIFAR-10 ResNet backbone for flow-circuits")
    parser.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--data-dir", required=True, help="Root directory containing CIFAR-10 data")
    parser.add_argument("--output", required=True, help="Checkpoint output path")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5.0e-4)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--download", action="store_true", help="Allow torchvision to download/extract CIFAR-10 if needed")
    parser.add_argument("--no-pretrained", action="store_true", help="Start from random initialization instead of torchvision pretrained weights")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = SupervisedBackboneTrainer(
        arch=args.arch,
        data_dir=args.data_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pretrained=not args.no_pretrained,
        download=args.download,
        val_size=args.val_size,
        epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    summary = trainer.train()
    print(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()
