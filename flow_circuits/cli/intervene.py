from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

from flow_circuits.data import build_cifar10_splits
from flow_circuits.interventions import run_circuit_interventions
from flow_circuits.training import collect_intervention_outputs, load_components_from_checkpoint
from flow_circuits.utils import seed_everything


def _format_seconds(seconds: float) -> str:
    seconds = int(max(0, round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


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
    intervention_batch_size = config["interventions"].get("batch_size", config["data"]["batch_size"])
    loaders = build_cifar10_splits(
        data_dir=config["data"]["data_dir"],
        batch_size=intervention_batch_size,
        num_workers=config["data"].get("num_workers", 4),
        seed=config["data"].get("seed", 0),
        augment_fit=config["data"].get("augment_fit", True),
        download=config["data"].get("download", True),
    )
    circuits_artifact = json.loads(Path(args.circuits).read_text(encoding="utf-8"))

    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    def collect_progress(*, batch_idx: int, total_batches: int | None, seen_images: int, target_images: int | None) -> None:
        target = target_images if target_images is not None else "all"
        total = total_batches if total_batches is not None else "?"
        print(
            f"[{time.strftime('%H:%M:%S')}] Intervention data pass: batch {batch_idx}/{total}  images {seen_images}/{target}",
            flush=True,
        )

    log("Collecting test-set outputs for interventions...")
    outputs = collect_intervention_outputs(
        components,
        loaders["test"],
        device=device,
        max_images=config["interventions"].get("max_images"),
        progress_callback=collect_progress,
    )
    log("Intervention feature collection complete.")
    n_circuits = len(circuits_artifact.get("circuits", []))
    log(
        f"Running interventions on {n_circuits} candidate circuit(s)"
        f" - ablating active nodes in member vs control images for each circuit..."
    )

    intervention_t0: list[float | None] = [None]

    def intervention_progress(**event) -> None:
        if intervention_t0[0] is None:
            intervention_t0[0] = time.time()
        completed = event["completed"]
        total = event["total"]
        eta_str = ""
        if completed > 0:
            elapsed = time.time() - intervention_t0[0]
            rate = elapsed / completed
            remaining = rate * (total - completed)
            eta_str = f"  ETA ~{_format_seconds(remaining)}"
        print(
            f"[{time.strftime('%H:%M:%S')}] Interventions:"
            f" circuit {completed}/{total}"
            f" (circuit_id={event['circuit_id']}, {event['status']}){eta_str}",
            flush=True,
        )

    results = run_circuit_interventions(
        components,
        circuits_artifact,
        outputs,
        alpha=config["interventions"].get("alpha", 0.05),
        output_path=args.output or Path(args.circuits).with_name("intervention_summary.json"),
        progress_callback=intervention_progress,
        n_jobs=max(1, int(config["interventions"].get("n_jobs", 1))),
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
    print("\n  A circuit is validated when ablating its active nodes causes a")
    print("  larger confidence drop for member images than for matched controls,")
    print("  random nodes, and random cells (all Holm-corrected p < alpha).")
    print(f"\n  Full results saved to file.\n{bar}\n", flush=True)


if __name__ == "__main__":
    main()
