from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from flow_circuits.data import build_cifar10_splits
from flow_circuits.interventions import run_circuit_interventions
from flow_circuits.training import collect_model_outputs, load_components_from_checkpoint
from flow_circuits.utils import seed_everything


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
    seed_everything(config["data"].get("seed", 0))
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
    bar = "=" * 64
    n_validated = sum(1 for r in results if r.validated)
    print(f"\n{bar}", flush=True)
    print(f"Intervention Results  ({len(results)} circuits tested)", flush=True)
    print(bar, flush=True)
    print(f"  Validated circuits (all 3 tests pass) : {n_validated}/{len(results)}", flush=True)
    if results:
        print(f"\n  {'Circuit':>7}  {'Members':>7}  {'Controls':>8}  {'Member delta':>12}  {'Status'}", flush=True)
        print(f"  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*12}  {'-'*8}", flush=True)
        for r in results:
            status = "VALIDATED" if r.validated else "rejected"
            print(
                f"  {r.circuit_id:>7}  {r.n_members:>7}  {r.n_controls:>8}"
                f"  {r.mean_member_delta_margin:>+12.4f}  {status}",
                flush=True,
            )
    print(f"\n  A circuit is validated when ablating its active nodes causes a")
    print(f"  larger confidence drop for member images than for matched controls,")
    print(f"  random nodes, and random cells (all Holm-corrected p < alpha).")
    print(f"\n  Full results saved to file.\n{bar}\n", flush=True)


if __name__ == "__main__":
    main()
