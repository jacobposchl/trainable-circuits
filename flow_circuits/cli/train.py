from __future__ import annotations

import argparse

import yaml

from flow_circuits.training import FlowCircuitTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the flow-based candidate-circuit model")
    parser.add_argument("--config", required=True, help="Path to a flow config YAML file")
    parser.add_argument("--resume", help="Optional checkpoint path to resume training from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    trainer = FlowCircuitTrainer(config, resume_from=args.resume)
    summary = trainer.train()
    checkpoint_dir = config.get("logging", {}).get("checkpoint_dir", ".")
    print(f"Checkpoint saved to: {checkpoint_dir}/final.pt")


if __name__ == "__main__":
    main()
