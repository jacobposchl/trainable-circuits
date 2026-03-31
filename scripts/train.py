"""
CLI entry point for Phase 1 meta-encoder training.

Examples:
    # Train with default config
    python scripts/train.py --config configs/phase1.yaml

    # Resume from checkpoint
    python scripts/train.py --config configs/phase1.yaml --resume experiments/phase1/epoch_50.pt

    # Ablation: info loss only
    python scripts/train.py --config configs/ablations/info_only.yaml
"""

import argparse

import yaml
from training.unified_trainer import Phase1Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Phase 1 meta-encoder")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    checkpoint_dir = config["logging"]["checkpoint_dir"]
    print(f"Config:     {args.config}")
    print(f"Output dir: {checkpoint_dir}")
    if args.resume:
        print(f"Resuming:   {args.resume}")

    trainer = Phase1Trainer(config)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
