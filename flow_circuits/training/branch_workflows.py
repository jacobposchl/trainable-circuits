from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

from flow_circuits.backbones import SupervisedBackboneTrainer
from flow_circuits.data import build_cifar10_splits
from flow_circuits.evaluation.metrics import evaluate_representation_metrics
from flow_circuits.training.trainer import (
    FlowCircuitTrainer,
    LoadedFlowComponents,
    _forward_pass,
    collect_model_outputs,
    load_components_from_checkpoint,
    save_flow_checkpoint,
)


def load_yaml_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_backbone_and_z_training_workflow(
    base_config: dict,
    *,
    backbone_epochs: int,
    phase_a_epochs: int,
    phase_b_epochs: int,
    phase_c_max_epochs: int,
    phase_c_milestones: list[int] | tuple[int, ...],
    lambda_traj_candidates: list[float] | tuple[float, ...],
    output_dir: str | Path,
    joint_branch_enabled: bool = True,
    joint_backbone_lr_multiplier: float = 0.1,
    joint_ce_weight: float = 1.0,
    force_rerun: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _print_section("Notebook 1 Workflow")
    print(f"Output directory: {output_dir}")
    print(f"Force rerun     : {force_rerun}")
    backbone_checkpoint = output_dir / "backbone_supervised.pt"
    backbone_summary_path = output_dir / "backbone_summary.json"

    if force_rerun or not backbone_checkpoint.exists():
        print("\n[Backbone] Training supervised backbone because checkpoint is missing or FORCE_RERUN=True.")
        backbone_summary = _train_supervised_backbone(
            base_config,
            backbone_epochs=backbone_epochs,
            output_path=backbone_checkpoint,
        )
        backbone_summary_path.write_text(json.dumps(backbone_summary, indent=2), encoding="utf-8")
    else:
        print(f"\n[Backbone] Reusing existing checkpoint: {backbone_checkpoint.name}")
        backbone_summary = json.loads(backbone_summary_path.read_text(encoding="utf-8")) if backbone_summary_path.exists() else {
            "output_path": str(backbone_checkpoint)
        }

    frozen_dir = output_dir / "frozen_branch"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    phase_b_frozen_path = frozen_dir / "phase_b_frozen.pt"
    if force_rerun or not phase_b_frozen_path.exists():
        print("\n[Frozen Branch] Training Phase A/B because frozen Phase B checkpoint is missing or FORCE_RERUN=True.")
        frozen_summary = _train_frozen_phase_ab(
            base_config,
            backbone_checkpoint=backbone_checkpoint,
            checkpoint_dir=frozen_dir,
            phase_a_epochs=phase_a_epochs,
            phase_b_epochs=phase_b_epochs,
        )
        shutil.copy2(frozen_dir / "phase_b.pt", phase_b_frozen_path)
        print(f"[Frozen Branch] Saved canonical frozen Phase B checkpoint: {phase_b_frozen_path.name}")
    else:
        print(f"\n[Frozen Branch] Reusing existing Phase B checkpoint: {phase_b_frozen_path.name}")
        frozen_summary = json.loads((frozen_dir / "phase_ab_summary.json").read_text(encoding="utf-8")) if (frozen_dir / "phase_ab_summary.json").exists() else {}

    frozen_candidates = _run_phase_c_milestone_sweep(
        base_config=base_config,
        phase_b_checkpoint=phase_b_frozen_path,
        branch_tag="frozen",
        output_dir=frozen_dir,
        max_epochs=phase_c_max_epochs,
        milestones=phase_c_milestones,
        lambda_traj_candidates=lambda_traj_candidates,
        train_backbone=False,
        ce_weight=0.0,
        backbone_lr_multiplier=joint_backbone_lr_multiplier,
        force_rerun=force_rerun,
    )

    joint_candidates = []
    if joint_branch_enabled:
        joint_dir = output_dir / "joint_branch"
        joint_dir.mkdir(parents=True, exist_ok=True)
        print("\n[Joint Branch] Warm-starting joint Phase C sweeps from frozen Phase B checkpoint.")
        joint_candidates = _run_phase_c_milestone_sweep(
            base_config=base_config,
            phase_b_checkpoint=phase_b_frozen_path,
            branch_tag="joint",
            output_dir=joint_dir,
            max_epochs=phase_c_max_epochs,
            milestones=phase_c_milestones,
            lambda_traj_candidates=lambda_traj_candidates,
            train_backbone=True,
            ce_weight=joint_ce_weight,
            backbone_lr_multiplier=joint_backbone_lr_multiplier,
            force_rerun=force_rerun,
        )
    else:
        print("\n[Joint Branch] Disabled. Skipping joint Phase C sweeps.")

    summary = {
        "backbone_checkpoint": str(backbone_checkpoint),
        "backbone_summary": backbone_summary,
        "frozen_phase_b_checkpoint": str(phase_b_frozen_path),
        "frozen_phase_ab_summary": frozen_summary,
        "frozen_phase_c_candidates": frozen_candidates,
        "joint_phase_c_candidates": joint_candidates,
    }
    (output_dir / "training_candidates.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved training candidate manifest: {(output_dir / 'training_candidates.json').name}")
    return summary


def _train_supervised_backbone(
    base_config: dict,
    *,
    backbone_epochs: int,
    output_path: str | Path,
) -> dict:
    cfg = deepcopy(base_config)
    trainer = SupervisedBackboneTrainer(
        arch=cfg["backbone"]["arch"],
        data_dir=cfg["data"]["data_dir"],
        output_path=output_path,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 4),
        seed=cfg["data"].get("seed", 0),
        pretrained=cfg["backbone"].get("pretrained", True),
        download=cfg["data"].get("download", True),
        epochs=int(backbone_epochs),
    )
    return trainer.train().to_dict()


def _train_frozen_phase_ab(
    base_config: dict,
    *,
    backbone_checkpoint: str | Path,
    checkpoint_dir: str | Path,
    phase_a_epochs: int,
    phase_b_epochs: int,
) -> dict:
    cfg = deepcopy(base_config)
    cfg["experiment"]["mode"] = "base"
    cfg.setdefault("backbone", {})
    cfg["backbone"]["weights_path"] = str(backbone_checkpoint)
    cfg["backbone"]["require_trained_checkpoint"] = True
    cfg["backbone"]["freeze_backbone"] = True
    cfg.setdefault("training", {})
    cfg["training"]["train_backbone"] = False
    cfg["training"]["ce_weight"] = 0.0
    cfg["training"]["phase_epochs"] = {
        "phase_a": int(phase_a_epochs),
        "phase_b": int(phase_b_epochs),
        "phase_c": 0,
    }
    cfg["logging"]["checkpoint_dir"] = str(checkpoint_dir)
    trainer = FlowCircuitTrainer(cfg)
    summary = trainer.train()
    (Path(checkpoint_dir) / "phase_ab_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _run_phase_c_milestone_sweep(
    *,
    base_config: dict,
    phase_b_checkpoint: str | Path,
    branch_tag: str,
    output_dir: str | Path,
    max_epochs: int,
    milestones: list[int] | tuple[int, ...],
    lambda_traj_candidates: list[float] | tuple[float, ...],
    train_backbone: bool,
    ce_weight: float,
    backbone_lr_multiplier: float,
    force_rerun: bool,
) -> list[dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{branch_tag}_phase_c_manifest.json"
    milestones = sorted({int(epoch) for epoch in milestones if 1 <= int(epoch) <= int(max_epochs)})
    if max_epochs not in milestones:
        milestones.append(int(max_epochs))
    expected_paths = _expected_phase_c_paths(
        output_dir=output_dir,
        branch_tag=branch_tag,
        lambda_traj_candidates=lambda_traj_candidates,
        milestones=milestones,
    )

    _print_section(f"{branch_tag.title()} Phase C Sweep")
    print(f"Starting checkpoint : {Path(phase_b_checkpoint).name}")
    print(f"Train backbone      : {train_backbone}")
    print(f"CE weight           : {ce_weight}")
    print(f"Milestone epochs    : {milestones}")
    print("Checkpoint status:")
    for line in _checkpoint_status_lines(expected_paths):
        print(f"  {line}")

    if manifest_path.exists() and not force_rerun:
        print(f"Manifest exists, reusing sweep outputs: {manifest_path.name}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    if force_rerun:
        print("FORCE_RERUN=True, recomputing sweep even if checkpoint files already exist.")
    elif any(path.exists() for path in expected_paths.values()):
        print("Some sweep checkpoints already exist, but manifest is missing. Reconstructing manifest from saved checkpoints and training only missing milestones.")

    cfg = deepcopy(base_config)
    cfg.setdefault("backbone", {})
    cfg["backbone"]["freeze_backbone"] = not train_backbone
    cfg.setdefault("training", {})
    cfg["training"]["train_backbone"] = bool(train_backbone)
    cfg["training"]["ce_weight"] = float(ce_weight)
    cfg["training"]["backbone_lr_multiplier"] = float(backbone_lr_multiplier)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_lr = float(cfg["training"].get("lr", 1.0e-3))
    weight_decay = float(cfg["training"].get("weight_decay", 1.0e-4))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))
    validation_images = int(cfg["training"].get("validation_images", 1024))
    loaders = build_cifar10_splits(
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 4),
        seed=cfg["data"].get("seed", 0),
        augment_fit=cfg["data"].get("augment_fit", True),
        download=cfg["data"].get("download", True),
    )

    existing_candidates = _reconstruct_phase_c_candidates(
        output_dir=output_dir,
        branch_tag=branch_tag,
        lambda_traj_candidates=lambda_traj_candidates,
        milestones=milestones,
    )
    candidates: list[dict] = list(existing_candidates)
    existing_candidate_keys = {
        (float(row["lambda_traj"]), int(row["epoch"]))
        for row in existing_candidates
    }

    for lambda_traj in [float(value) for value in lambda_traj_candidates]:
        print(f"\n[{branch_tag}] lambda_traj={lambda_traj:.4f}")
        existing_epochs_for_lambda = sorted(
            epoch
            for candidate_lambda, epoch in existing_candidate_keys
            if candidate_lambda == float(lambda_traj)
        )
        for epoch in milestones:
            checkpoint_path = expected_paths[(float(lambda_traj), int(epoch))]
            status = "exists" if checkpoint_path.exists() else "missing"
            print(f"  milestone epoch {epoch:>3}: {status:>7} -> {checkpoint_path.name}")
        if existing_epochs_for_lambda and max(existing_epochs_for_lambda) >= int(max_epochs):
            print(
                f"  action: SKIP lambda={lambda_traj:.4f} because all requested milestones already exist "
                f"(latest saved epoch={max(existing_epochs_for_lambda)})."
            )
            continue

        latest_existing_epoch = max(existing_epochs_for_lambda) if existing_epochs_for_lambda else 0
        if latest_existing_epoch > 0:
            resume_checkpoint = expected_paths[(float(lambda_traj), int(latest_existing_epoch))]
            print(
                f"  action: RESUME lambda={lambda_traj:.4f} from {resume_checkpoint.name} "
                f"(latest saved epoch={latest_existing_epoch}; next epoch={latest_existing_epoch + 1})."
            )
            components = load_components_from_checkpoint(
                resume_checkpoint,
                device,
                config_overrides={
                    "backbone": {"freeze_backbone": not train_backbone},
                    "training": {
                        "train_backbone": bool(train_backbone),
                        "ce_weight": float(ce_weight),
                        "backbone_lr_multiplier": float(backbone_lr_multiplier),
                    },
                },
            )
            optimizer = _build_optimizer(
                components,
                base_lr=base_lr,
                weight_decay=weight_decay,
                train_backbone=train_backbone,
                backbone_lr_multiplier=backbone_lr_multiplier,
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=max(int(max_epochs), 1))
            _restore_optimizer_scheduler_from_checkpoint(
                checkpoint=components.checkpoint,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            start_epoch = int(latest_existing_epoch) + 1
        else:
            print(
                f"  action: START lambda={lambda_traj:.4f} from Phase B checkpoint "
                f"because no saved milestones exist yet."
            )
            components = load_components_from_checkpoint(
                phase_b_checkpoint,
                device,
                config_overrides={
                    "backbone": {"freeze_backbone": not train_backbone},
                    "training": {
                        "train_backbone": bool(train_backbone),
                        "ce_weight": float(ce_weight),
                        "backbone_lr_multiplier": float(backbone_lr_multiplier),
                    },
                },
            )
            optimizer = _build_optimizer(
                components,
                base_lr=base_lr,
                weight_decay=weight_decay,
                train_backbone=train_backbone,
                backbone_lr_multiplier=backbone_lr_multiplier,
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=max(int(max_epochs), 1))
            start_epoch = 1

        criterion = nn.CrossEntropyLoss()
        history: list[dict] = []
        for epoch_idx in range(int(start_epoch), int(max_epochs) + 1):
            print(
                f"  [{branch_tag}] lambda={lambda_traj:.4f} | epoch {epoch_idx:>2}/{int(max_epochs)}"
                f" | running train+val"
            )
            train_metrics = _run_branch_epoch(
                components,
                loader=loaders["fit"],
                optimizer=optimizer,
                scheduler=None,
                criterion=criterion,
                device=device,
                lambda_rec=float(cfg["objectives"].get("lambda_rec", 0.2)),
                lambda_traj=float(lambda_traj),
                ce_weight=float(ce_weight),
                grad_clip=grad_clip,
                train=True,
            )
            val_metrics = _run_branch_epoch(
                components,
                loader=loaders["val"],
                optimizer=None,
                scheduler=None,
                criterion=criterion,
                device=device,
                lambda_rec=float(cfg["objectives"].get("lambda_rec", 0.2)),
                lambda_traj=float(lambda_traj),
                ce_weight=float(ce_weight),
                grad_clip=grad_clip,
                train=False,
            )
            scheduler.step()
            history.append(
                {
                    "epoch": int(epoch_idx),
                    "train": train_metrics,
                    "val": val_metrics,
                }
            )
            print(
                "    "
                f"train_loss={train_metrics['loss']:.4f} "
                f"pred_cos={train_metrics['prediction_cosine']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_pred_cos={val_metrics['prediction_cosine']:.4f}"
                + (f" ce={val_metrics['ce_loss']:.4f}" if ce_weight > 0.0 else "")
            )
            if epoch_idx not in milestones:
                continue
            print(f"    milestone reached at epoch {epoch_idx}; collecting held-out validation metrics...")
            rep_metrics = _validation_metrics(
                components,
                loader=loaders["val"],
                device=device,
                max_images=validation_images,
                alignment_max_pairs=int(cfg["training"].get("alignment_max_pairs", 2048)),
                seed=int(cfg["data"].get("seed", 0)),
            )
            checkpoint_path = expected_paths[(float(lambda_traj), int(epoch_idx))]
            checkpoint_config = deepcopy(cfg)
            checkpoint_config["objectives"]["lambda_traj"] = float(lambda_traj)
            checkpoint_config["logging"]["checkpoint_dir"] = str(output_dir)
            save_flow_checkpoint(
                path=checkpoint_path,
                components=components,
                optimizer=optimizer,
                scheduler=scheduler,
                config=checkpoint_config,
                phase="phase_c",
                validation=rep_metrics,
                extra_summary={
                    "branch_tag": branch_tag,
                    "lambda_traj": float(lambda_traj),
                    "epoch": int(epoch_idx),
                    "train_backbone": bool(train_backbone),
                    "ce_weight": float(ce_weight),
                    "history": history,
                },
            )
            print(
                "    saved "
                f"{checkpoint_path.name} | pred_cos={rep_metrics['prediction_cosine_mean']:.4f} "
                f"recon_cos={rep_metrics['reconstruction_cosine_mean']:.4f} "
                f"traj_align={rep_metrics['trajectory_alignment_mean']:.4f}"
            )
            candidates.append(
                {
                    "branch_tag": branch_tag,
                    "lambda_traj": float(lambda_traj),
                    "epoch": int(epoch_idx),
                    "checkpoint_path": str(checkpoint_path),
                    "prediction_cosine_mean": float(rep_metrics["prediction_cosine_mean"]),
                    "reconstruction_cosine_mean": float(rep_metrics["reconstruction_cosine_mean"]),
                    "trajectory_alignment_mean": float(rep_metrics["trajectory_alignment_mean"]),
                    "train_backbone": bool(train_backbone),
                    "ce_weight": float(ce_weight),
                }
            )
            existing_candidate_keys.add((float(lambda_traj), int(epoch_idx)))
    candidates = sorted(
        candidates,
        key=lambda row: (str(row["branch_tag"]), float(row["lambda_traj"]), int(row["epoch"])),
    )
    manifest_path.write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    print(f"\n[{branch_tag}] Sweep complete. Wrote manifest: {manifest_path.name}")
    return candidates


def _build_optimizer(
    components: LoadedFlowComponents,
    *,
    base_lr: float,
    weight_decay: float,
    train_backbone: bool,
    backbone_lr_multiplier: float,
) -> AdamW:
    groups = [
        {
            "params": list(components.tokenizer.parameters())
            + list(components.encoder.parameters())
            + list(components.objective.parameters()),
            "lr": float(base_lr),
            "initial_lr": float(base_lr),
            "weight_decay": float(weight_decay),
        }
    ]
    if train_backbone:
        backbone_lr = float(base_lr) * float(backbone_lr_multiplier)
        groups.append(
            {
                "params": list(components.observer.model.parameters()),
                "lr": backbone_lr,
                "initial_lr": backbone_lr,
                "weight_decay": float(weight_decay),
            }
        )
    return AdamW(groups)


def _run_branch_epoch(
    components: LoadedFlowComponents,
    *,
    loader,
    optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    lambda_rec: float,
    lambda_traj: float,
    ce_weight: float,
    grad_clip: float,
    train: bool,
) -> dict:
    train_backbone = not bool(getattr(components.observer, "freeze_backbone", True))
    components.observer.train(train if train_backbone else False)
    components.tokenizer.train(train)
    components.encoder.train(train)
    components.objective.train(train)
    aggregate = {
        "loss": 0.0,
        "pred_loss": 0.0,
        "rec_loss": 0.0,
        "traj_loss": 0.0,
        "ce_loss": 0.0,
        "prediction_cosine": 0.0,
        "reconstruction_cosine": 0.0,
    }
    n_batches = 0
    grad_params = list(components.tokenizer.parameters()) + list(components.encoder.parameters()) + list(components.objective.parameters())
    if train_backbone:
        grad_params += list(components.observer.model.parameters())
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        if train and optimizer is not None:
            optimizer.zero_grad()
        _, _, objective_output, logits = _forward_pass(
            components,
            images,
            lambda_rec=lambda_rec,
            lambda_traj=lambda_traj,
            traj_topk=components.config["objectives"].get("traj_topk", 8),
            traj_gamma=components.config["objectives"].get("traj_gamma", 0.2),
            traj_tau=components.config["objectives"].get("traj_tau", 0.1),
        )
        ce_loss = criterion(logits, labels) if ce_weight > 0.0 else torch.zeros((), device=device)
        total_loss = objective_output.total_loss + (float(ce_weight) * ce_loss)
        if train and optimizer is not None:
            total_loss.backward()
            nn.utils.clip_grad_norm_(grad_params, max_norm=float(grad_clip))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        aggregate["loss"] += float(total_loss.item())
        aggregate["pred_loss"] += float(objective_output.pred_loss.item())
        aggregate["rec_loss"] += float(objective_output.rec_loss.item())
        aggregate["traj_loss"] += float(objective_output.traj_loss.item())
        aggregate["ce_loss"] += float(ce_loss.item())
        aggregate["prediction_cosine"] += float(objective_output.prediction_cosine.item())
        aggregate["reconstruction_cosine"] += float(objective_output.reconstruction_cosine.item())
        n_batches += 1
    if n_batches == 0:
        return aggregate
    return {key: value / n_batches for key, value in aggregate.items()}


def _validation_metrics(
    components: LoadedFlowComponents,
    *,
    loader,
    device: torch.device,
    max_images: int,
    alignment_max_pairs: int,
    seed: int,
) -> dict:
    outputs = collect_model_outputs(
        components,
        loader,
        device=device,
        max_images=max_images,
    )
    metrics = evaluate_representation_metrics(
        outputs["z"],
        outputs["local_features"],
        outputs["flow_targets"],
        outputs["future_descriptors"],
        outputs["predicted_next"],
        outputs["reconstructed_current"],
        max_alignment_pairs=alignment_max_pairs,
        alignment_seed=seed,
    )
    return metrics.to_dict()


def _lambda_key(value: float) -> str:
    return str(value).replace(".", "p")


def _expected_phase_c_paths(
    *,
    output_dir: Path,
    branch_tag: str,
    lambda_traj_candidates: list[float] | tuple[float, ...],
    milestones: list[int],
) -> dict[tuple[float, int], Path]:
    paths: dict[tuple[float, int], Path] = {}
    for lambda_traj in [float(value) for value in lambda_traj_candidates]:
        for epoch in [int(value) for value in milestones]:
            checkpoint_name = f"phase_c_{branch_tag}_lambda_{_lambda_key(lambda_traj)}_epoch_{epoch}.pt"
            paths[(float(lambda_traj), int(epoch))] = output_dir / checkpoint_name
    return paths


def _checkpoint_status_lines(expected_paths: dict[tuple[float, int], Path]) -> list[str]:
    lines: list[str] = []
    for (lambda_traj, epoch), path in sorted(expected_paths.items(), key=lambda item: (item[0][0], item[0][1])):
        status = "exists" if path.exists() else "missing"
        lines.append(f"lambda={lambda_traj:<4} epoch={epoch:<3} {status:<7} {path.name}")
    return lines


def _print_section(title: str) -> None:
    line = "=" * max(64, len(title) + 8)
    print(f"\n{line}")
    print(title)
    print(line)


def _reconstruct_phase_c_candidates(
    *,
    output_dir: Path,
    branch_tag: str,
    lambda_traj_candidates: list[float] | tuple[float, ...],
    milestones: list[int],
) -> list[dict]:
    expected_paths = _expected_phase_c_paths(
        output_dir=output_dir,
        branch_tag=branch_tag,
        lambda_traj_candidates=lambda_traj_candidates,
        milestones=milestones,
    )
    candidates: list[dict] = []
    for (lambda_traj, epoch), checkpoint_path in sorted(expected_paths.items(), key=lambda item: (item[0][0], item[0][1])):
        if not checkpoint_path.exists():
            continue
        candidates.append(
            _candidate_row_from_checkpoint(
                checkpoint_path=checkpoint_path,
                branch_tag=branch_tag,
                lambda_traj=lambda_traj,
                epoch=epoch,
            )
        )
    return candidates


def _candidate_row_from_checkpoint(
    *,
    checkpoint_path: Path,
    branch_tag: str,
    lambda_traj: float,
    epoch: int,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validation = checkpoint.get("validation", {})
    summary = checkpoint.get("summary", {})
    return {
        "branch_tag": str(summary.get("branch_tag", branch_tag)),
        "lambda_traj": float(summary.get("lambda_traj", lambda_traj)),
        "epoch": int(summary.get("epoch", epoch)),
        "checkpoint_path": str(checkpoint_path),
        "prediction_cosine_mean": float(validation.get("prediction_cosine_mean", 0.0)),
        "reconstruction_cosine_mean": float(validation.get("reconstruction_cosine_mean", 0.0)),
        "trajectory_alignment_mean": float(validation.get("trajectory_alignment_mean", 0.0)),
        "train_backbone": bool(summary.get("train_backbone", False)),
        "ce_weight": float(summary.get("ce_weight", 0.0)),
    }


def _restore_optimizer_scheduler_from_checkpoint(
    *,
    checkpoint: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> None:
    optimizer_state = checkpoint.get("optimizer_state")
    scheduler_state = checkpoint.get("scheduler_state")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
