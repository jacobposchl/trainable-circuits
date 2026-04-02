from __future__ import annotations

import argparse
import json

import yaml

from flow_circuits.training import FlowCircuitTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the flow-based candidate-circuit model")
    parser.add_argument("--config", required=True, help="Path to a flow config YAML file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    trainer = FlowCircuitTrainer(config)
    summary = trainer.train()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
