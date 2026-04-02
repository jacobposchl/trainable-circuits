from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from flow_circuits.data import build_cifar10_splits
from flow_circuits.interventions import run_circuit_interventions
from flow_circuits.training import collect_model_outputs, load_components_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run held-out residual-patch interventions")
    parser.add_argument("--checkpoint", required=True, help="Path to a flow-circuits checkpoint")
    parser.add_argument("--circuits", required=True, help="Candidate circuit artifact JSON")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    components = load_components_from_checkpoint(args.checkpoint, device=device)
    config = components.config
    loaders = build_cifar10_splits(
        data_dir=config["data"]["data_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"].get("num_workers", 4),
        seed=config["data"].get("seed", 0),
        augment_fit=config["data"].get("augment_fit", True),
        download=config["data"].get("download", True),
    )
    circuits_artifact = json.loads(Path(args.circuits).read_text(encoding="utf-8"))
    outputs = collect_model_outputs(
        components,
        loaders["test"],
        device=device,
        max_images=config["interventions"].get("max_images"),
    )
    results = run_circuit_interventions(
        components,
        circuits_artifact,
        outputs,
        alpha=config["interventions"].get("alpha", 0.05),
        output_path=args.output or Path(args.circuits).with_name("intervention_summary.json"),
    )
    csv_path = Path(args.output or Path(args.circuits).with_name("intervention_summary.json")).with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].to_dict().keys()) if results else ["circuit_id"])
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())
    print(json.dumps([result.to_dict() for result in results], indent=2))


if __name__ == "__main__":
    main()
