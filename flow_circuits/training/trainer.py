from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import itertools
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

from flow_circuits.backbones import FrozenResNetObserver
from flow_circuits.data import build_cifar10_splits
from flow_circuits.encoders import SpatiotemporalEncoder
from flow_circuits.evaluation import RepresentationMetrics, evaluate_representation_metrics
from flow_circuits.objectives import FlowObjective
from flow_circuits.tokenization import FlowTokenizer
from flow_circuits.training.baselines import BaselineRegressors
from flow_circuits.utils import seed_everything


@dataclass
class LoadedFlowComponents:
    observer: FrozenResNetObserver
    tokenizer: FlowTokenizer
    encoder: SpatiotemporalEncoder
    objective: FlowObjective
    config: dict
    checkpoint: dict


def build_components(config: dict, device: torch.device) -> LoadedFlowComponents:
    bcfg = config["backbone"]
    tcfg = config["tokenization"]
    ecfg = config["encoder"]
    ocfg = config["objectives"]

    observer = FrozenResNetObserver(
        arch=bcfg["arch"],
        pretrained=bcfg.get("pretrained", True),
        num_classes=bcfg.get("num_classes", 10),
        grid_size=tcfg.get("grid_size", 4),
        weights_path=bcfg.get("weights_path"),
        require_trained_checkpoint=bcfg.get("require_trained_checkpoint", False),
    ).to(device)
    tokenizer = FlowTokenizer(
        layer_channels=observer.layer_channels,
        token_dim=tcfg.get("token_dim", 128),
        flow_dim=tcfg.get("flow_dim", 256),
        traj_dim=tcfg.get("traj_dim", 256),
        grid_size=tcfg.get("grid_size", 4),
        eps=tcfg.get("eps", 1.0e-6),
    ).to(device)
    encoder = SpatiotemporalEncoder(
        n_layers=len(observer.layer_channels),
        grid_size=tcfg.get("grid_size", 4),
        token_dim=tcfg.get("token_dim", 128),
        n_heads=ecfg.get("n_heads", 4),
        n_transformer_layers=ecfg.get("n_transformer_layers", 2),
        mlp_dim=ecfg.get("mlp_dim"),
        dropout=ecfg.get("dropout", 0.0),
    ).to(device)
    objective = FlowObjective(
        n_layers=len(observer.layer_channels),
        token_dim=tcfg.get("token_dim", 128),
        flow_dim=tcfg.get("flow_dim", 256),
        pred_hidden_dim=ocfg.get("pred_hidden_dim"),
        rec_hidden_dim=ocfg.get("rec_hidden_dim"),
    ).to(device)
    return LoadedFlowComponents(
        observer=observer,
        tokenizer=tokenizer,
        encoder=encoder,
        objective=objective,
        config=config,
        checkpoint={},
    )


def load_components_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> LoadedFlowComponents:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    build_config = deepcopy(config)
    build_config.setdefault("backbone", {})
    build_config["backbone"]["weights_path"] = None
    build_config["backbone"]["require_trained_checkpoint"] = False
    build_config["backbone"]["pretrained"] = False
    components = build_components(build_config, device)
    components.observer.load_state_dict(checkpoint["observer_state"])
    components.tokenizer.load_state_dict(checkpoint["tokenizer_state"])
    components.encoder.load_state_dict(checkpoint["encoder_state"])
    components.objective.load_state_dict(checkpoint["objective_state"])
    observer_metadata = checkpoint.get("observer_metadata", {})
    components.observer.classifier_is_trained = bool(observer_metadata.get("classifier_is_trained", False))
    components.observer.weights_path = config["backbone"].get("weights_path")
    components.observer.require_trained_checkpoint = config["backbone"].get("require_trained_checkpoint", False)
    components.config = config
    components.checkpoint = checkpoint
    components.observer.eval()
    components.tokenizer.eval()
    components.encoder.eval()
    components.objective.eval()
    return components


def collect_model_outputs(
    components: LoadedFlowComponents,
    loader,
    *,
    device: torch.device,
    max_images: int | None = None,
    progress_callback=None,
) -> dict[str, torch.Tensor]:
    outputs = {
        "z": [],
        "local_features": None,
        "flow_targets": [],
        "future_descriptors": [],
        "predicted_next": [],
        "reconstructed_current": [],
        "images": [],
        "logits": [],
        "labels": [],
        "indices": [],
    }
    seen = 0
    batch_size = getattr(loader, "batch_size", None) or components.config["data"].get("batch_size")
    total_batches = None
    if hasattr(loader, "__len__"):
        total_batches = len(loader)
        if max_images is not None and batch_size:
            total_batches = min(total_batches, int(np.ceil(max_images / batch_size)))
    with torch.no_grad():
        for batch_idx, (images, labels, indices) in enumerate(loader, start=1):
            device_images = images.to(device)
            tokenized, z, objective_output = _forward_pass(components, device_images, lambda_rec=1.0, lambda_traj=0.0)
            outputs["z"].append(z.cpu())
            if outputs["local_features"] is None:
                outputs["local_features"] = [[] for _ in tokenized.local_features]
            for layer_idx, layer_features in enumerate(tokenized.local_features):
                outputs["local_features"][layer_idx].append(layer_features.cpu())
            outputs["flow_targets"].append(tokenized.flow_targets.cpu())
            outputs["future_descriptors"].append(tokenized.future_descriptors.cpu())
            outputs["predicted_next"].append(objective_output.predicted_next.cpu())
            outputs["reconstructed_current"].append(objective_output.reconstructed_current.cpu())
            outputs["images"].append(images.cpu())
            with torch.no_grad():
                outputs["logits"].append(components.observer.model(device_images).cpu())
            outputs["labels"].append(labels.cpu())
            outputs["indices"].append(indices.cpu())
            seen += device_images.shape[0]
            if progress_callback is not None:
                progress_callback(
                    batch_idx=batch_idx,
                    total_batches=total_batches,
                    seen_images=seen,
                    target_images=max_images,
                )
            if max_images is not None and seen >= max_images:
                break

    for key, value in outputs.items():
        if not value:
            continue
        if key == "local_features":
            continue
        outputs[key] = torch.cat(value, dim=0)
        if max_images is not None:
            outputs[key] = outputs[key][:max_images]
    if outputs["local_features"]:
        concatenated_features = []
        for values in outputs["local_features"]:
            layer_tensor = torch.cat(values, dim=0)
            if max_images is not None:
                layer_tensor = layer_tensor[:max_images]
            concatenated_features.append(layer_tensor)
        outputs["local_features"] = concatenated_features
    return outputs


def collect_discovery_outputs(
    components: LoadedFlowComponents,
    loader,
    *,
    device: torch.device,
    max_images: int | None = None,
    progress_callback=None,
) -> dict[str, torch.Tensor]:
    outputs = {
        "flow_targets": [],
        "future_descriptors": [],
        "predicted_next": [],
        "labels": [],
        "indices": [],
    }
    seen = 0
    total_batches = _infer_total_batches(loader, components, max_images=max_images)
    with torch.no_grad():
        for batch_idx, (images, labels, indices) in enumerate(loader, start=1):
            device_images = images.to(device)
            observations = components.observer(device_images)
            tokenized = components.tokenizer(observations)
            z, _ = components.encoder(tokenized.token_inputs)
            predicted_next = _predict_next_from_latents(
                components.objective,
                z,
                tokenized.flow_targets,
            )
            outputs["flow_targets"].append(tokenized.flow_targets.cpu())
            outputs["future_descriptors"].append(tokenized.future_descriptors.cpu())
            outputs["predicted_next"].append(predicted_next.cpu())
            outputs["labels"].append(labels.cpu())
            outputs["indices"].append(indices.cpu())
            seen += device_images.shape[0]
            _maybe_report_progress(
                progress_callback,
                batch_idx=batch_idx,
                total_batches=total_batches,
                seen_images=seen,
                target_images=max_images,
            )
            if max_images is not None and seen >= max_images:
                break
    return _concatenate_output_tensors(outputs, max_images=max_images)


def collect_intervention_outputs(
    components: LoadedFlowComponents,
    loader,
    *,
    device: torch.device,
    max_images: int | None = None,
    progress_callback=None,
) -> dict[str, torch.Tensor]:
    outputs = {
        "future_descriptors": [],
        "images": [],
        "logits": [],
        "labels": [],
        "indices": [],
    }
    seen = 0
    total_batches = _infer_total_batches(loader, components, max_images=max_images)
    with torch.no_grad():
        for batch_idx, (images, labels, indices) in enumerate(loader, start=1):
            device_images = images.to(device)
            observations = components.observer(device_images)
            tokenized = components.tokenizer(observations)
            outputs["future_descriptors"].append(tokenized.future_descriptors.cpu())
            outputs["images"].append(images.cpu())
            outputs["logits"].append(components.observer.model(device_images).cpu())
            outputs["labels"].append(labels.cpu())
            outputs["indices"].append(indices.cpu())
            seen += device_images.shape[0]
            _maybe_report_progress(
                progress_callback,
                batch_idx=batch_idx,
                total_batches=total_batches,
                seen_images=seen,
                target_images=max_images,
            )
            if max_images is not None and seen >= max_images:
                break
    return _concatenate_output_tensors(outputs, max_images=max_images)


def collect_baseline_features(
    components: LoadedFlowComponents,
    loader,
    *,
    device: torch.device,
    max_images: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    locals_by_layer: list[list[np.ndarray]] | None = None
    flows_by_layer: list[list[np.ndarray]] | None = None
    next_by_layer: list[list[np.ndarray]] | None = None
    seen = 0
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device)
            observations = components.observer(images)
            tokenized = components.tokenizer(observations)
            if locals_by_layer is None:
                locals_by_layer = [[] for _ in range(tokenized.flow_targets.shape[1] - 1)]
                flows_by_layer = [[] for _ in range(tokenized.flow_targets.shape[1] - 1)]
                next_by_layer = [[] for _ in range(tokenized.flow_targets.shape[1] - 1)]
            for layer_idx in range(tokenized.flow_targets.shape[1] - 1):
                locals_by_layer[layer_idx].append(tokenized.local_features[layer_idx].cpu().numpy())
                flows_by_layer[layer_idx].append(tokenized.flow_targets[:, layer_idx].cpu().numpy())
                next_by_layer[layer_idx].append(tokenized.flow_targets[:, layer_idx + 1].cpu().numpy())
            seen += images.shape[0]
            if max_images is not None and seen >= max_images:
                break
    return (
        [np.concatenate(values, axis=0) for values in locals_by_layer or []],
        [np.concatenate(values, axis=0) for values in flows_by_layer or []],
        [np.concatenate(values, axis=0) for values in next_by_layer or []],
    )


def _infer_total_batches(loader, components: LoadedFlowComponents, *, max_images: int | None) -> int | None:
    batch_size = getattr(loader, "batch_size", None) or components.config["data"].get("batch_size")
    if not hasattr(loader, "__len__"):
        return None
    total_batches = len(loader)
    if max_images is not None and batch_size:
        total_batches = min(total_batches, int(math.ceil(max_images / batch_size)))
    return total_batches


def _maybe_report_progress(progress_callback, **kwargs) -> None:
    if progress_callback is not None:
        progress_callback(**kwargs)


def _concatenate_output_tensors(outputs: dict, *, max_images: int | None) -> dict:
    for key, value in outputs.items():
        if not value:
            continue
        outputs[key] = torch.cat(value, dim=0)
        if max_images is not None:
            outputs[key] = outputs[key][:max_images]
    return outputs


class FlowCircuitTrainer:
    def __init__(self, config: dict, *, resume_from: str | Path | None = None) -> None:
        self.config = config
        seed_everything(config["data"].get("seed", 0))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resume_from = Path(resume_from) if resume_from is not None else None
        self.resume_checkpoint: dict | None = None
        self.components = build_components(config, self.device)
        self.loaders = build_cifar10_splits(
            data_dir=config["data"]["data_dir"],
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"].get("num_workers", 4),
            seed=config["data"].get("seed", 0),
            augment_fit=config["data"].get("augment_fit", True),
            download=config["data"].get("download", True),
        )
        self.checkpoint_dir = Path(config["logging"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        train_params = list(self.components.tokenizer.parameters())
        train_params += list(self.components.encoder.parameters())
        train_params += list(self.components.objective.parameters())
        self.optimizer = AdamW(
            train_params,
            lr=config["training"].get("lr", 1.0e-3),
            weight_decay=config["training"].get("weight_decay", 1.0e-4),
        )
        _pe = config["training"]["phase_epochs"]
        ab_epochs = _pe.get("phase_a", 0) + _pe.get("phase_b", 0)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(ab_epochs, 1))
        if self.resume_from is not None:
            self._load_training_checkpoint(self.resume_from)

    def train(self) -> dict:
        history = []
        phase_epochs = self.config["training"]["phase_epochs"]
        objectives = self.config["objectives"]
        resumed_phase = self.resume_checkpoint.get("phase") if self.resume_checkpoint is not None else None

        if resumed_phase == "phase_c":
            _log_section("Resume")
            _log(f"Loaded completed Phase C checkpoint from: {self.resume_from}")
            _log("Training is already complete; reusing the saved final summary.")
            return self.resume_checkpoint.get("summary") or {
                "history": [],
                "final_phase": "phase_c",
                "final_metrics": self.resume_checkpoint.get("validation", {}),
                "phase_c": {"accepted": True},
            }

        if resumed_phase == "phase_b":
            _log_section("Resume")
            _log(f"Loaded Phase B checkpoint from: {self.resume_from}")
            _log("Skipping Phases A and B and continuing from the saved checkpoint state.")
            history = list(self.resume_checkpoint.get("summary", {}).get("history", []))
            phase_b_metrics = RepresentationMetrics(**self.resume_checkpoint.get("validation", {}))
        else:
            n_a = phase_epochs.get("phase_a", 0)
            n_b = phase_epochs.get("phase_b", 0)
            _log_section(f"Phase A - Prediction  ({n_a} epoch{'s' if n_a != 1 else ''})")
            _log("The encoder learns to predict how each ResNet layer's residual flow")
            _log("will change into the next layer (next-step prediction objective).")
            history.extend(self._run_phase("phase_a", n_a, lambda_rec=0.0, lambda_traj=0.0))
            _log_section(f"Phase B - Prediction + Reconstruction  ({n_b} epoch{'s' if n_b != 1 else ''})")
            _log("Adds a reconstruction objective: the encoder must also recover the")
            _log("current layer's flow from its own output.")
            history.extend(
                self._run_phase(
                    "phase_b",
                    n_b,
                    lambda_rec=objectives.get("lambda_rec", 0.2),
                    lambda_traj=0.0,
                )
            )
            _log("\nRunning held-out validation...")
            phase_b_metrics = self._full_validation_metrics(self.loaders["val"])
            _log(f"  pred_cos={phase_b_metrics.prediction_cosine_mean:.4f}  recon_cos={phase_b_metrics.reconstruction_cosine_mean:.4f}")
            self._save_checkpoint(
                name="phase_b.pt",
                phase="phase_b",
                validation=phase_b_metrics.to_dict(),
            )

        mode = self.config["experiment"].get("mode", "base")
        final_phase = "phase_b"
        final_metrics = phase_b_metrics
        phase_c_summary = None

        if mode == "aligned" and phase_epochs.get("phase_c", 0) > 0:
            _log_section("Baseline Check")
            _log("Fitting three simple predictors to compare against the encoder:")
            _log("  (1) per-node mean predictor  (2) local CNN MLP  (3) flow target MLP")
            _log("The encoder must beat the best of these to proceed to Phase C.\n")
            baseline_regressors = self._fit_baselines()
            baseline_metrics = self._evaluate_baselines(baseline_regressors)
            bm = baseline_metrics
            enc = phase_b_metrics.prediction_cosine_mean
            _log(f"  Mean predictor  : pred_cos = {bm.mean_baseline:.4f}{'  <- best' if bm.best_baseline_name == 'mean_baseline' else ''}")
            _log(f"  Local CNN MLP   : pred_cos = {bm.local_baseline:.4f}{'  <- best' if bm.best_baseline_name == 'local_baseline' else ''}")
            _log(f"  Flow target MLP : pred_cos = {bm.flow_baseline:.4f}{'  <- best' if bm.best_baseline_name == 'flow_baseline' else ''}")
            _log(f"  Our encoder     : pred_cos = {enc:.4f}")
            if phase_b_metrics.prediction_cosine_mean > baseline_metrics.best_baseline:
                _log(f"\n  Encoder beats best baseline by +{enc - bm.best_baseline:.4f}")
                _log("  Phase C (trajectory alignment sweep) will now run.")
                phase_b_snapshot = self._snapshot_state()
                phase_c_result = self._run_phase_c_sweep(
                    phase_b_snapshot=phase_b_snapshot,
                    phase_b_metrics=phase_b_metrics,
                    epochs=phase_epochs["phase_c"],
                )
                phase_c_summary = phase_c_result
                if phase_c_result["accepted"]:
                    final_phase = "phase_c"
                    final_metrics = phase_c_result["metrics"]
            else:
                _log(f"\n  Encoder ({enc:.4f}) does not beat best baseline ({bm.best_baseline:.4f}).")
                _log("  Phase C will be skipped. Consider training for more epochs.")
                phase_c_summary = {
                    "accepted": False,
                    "reason": "phase_b_prediction_not_above_best_baseline",
                    "baseline_metrics": baseline_metrics.to_dict(),
                }

        summary = {
            "history": history,
            "final_phase": final_phase,
            "final_metrics": final_metrics.to_dict(),
            "phase_c": phase_c_summary,
        }
        self._save_checkpoint(
            name="final.pt",
            phase=final_phase,
            validation=summary["final_metrics"],
            extra_summary=summary,
        )
        _log_section("Training Complete")
        _log(f"  Final phase        : {final_phase}")
        _log(f"  Prediction cosine  : {final_metrics.prediction_cosine_mean:.4f}  (1.0 = perfect; higher is better)")
        _log(f"  Reconstruction cos : {final_metrics.reconstruction_cosine_mean:.4f}")
        _log(f"  Traj. alignment    : {final_metrics.trajectory_alignment_mean:.4f}  (spatial consistency across images)")
        _log()
        return summary

    def _run_phase(self, phase: str, epochs: int, *, lambda_rec: float, lambda_traj: float) -> list[dict]:
        results = []
        for epoch_idx in range(epochs):
            train_metrics = self._run_epoch(
                loader=self.loaders["fit"],
                train=True,
                lambda_rec=lambda_rec,
                lambda_traj=lambda_traj,
            )
            val_metrics = self._run_epoch(
                loader=self.loaders["val"],
                train=False,
                lambda_rec=lambda_rec,
                lambda_traj=lambda_traj,
            )
            self.scheduler.step()
            result = {
                "phase": phase,
                "epoch_in_phase": epoch_idx + 1,
                "train": train_metrics,
                "val": val_metrics,
            }
            results.append(result)
            prev_cos = results[-2]["val"]["prediction_cosine"] if len(results) >= 2 else None
            trend = ""
            if prev_cos is not None:
                d = val_metrics["prediction_cosine"] - prev_cos
                trend = " (improving)" if d > 0.001 else (" (dropping)" if d < -0.001 else "")
            w = len(str(epochs))
            _log(
                f"  Epoch {epoch_idx + 1:{w}}/{epochs}"
                f"  |  train  loss={train_metrics['loss']:.4f}  pred_cos={train_metrics['prediction_cosine']:.4f}"
                f"  |  val  pred_cos={val_metrics['prediction_cosine']:.4f}{trend}"
            )
        return results

    def _run_epoch(
        self,
        *,
        loader,
        train: bool,
        lambda_rec: float,
        lambda_traj: float,
    ) -> dict:
        self.components.tokenizer.train(train)
        self.components.encoder.train(train)
        self.components.objective.train(train)
        aggregate = {
            "loss": 0.0,
            "pred_loss": 0.0,
            "rec_loss": 0.0,
            "traj_loss": 0.0,
            "prediction_cosine": 0.0,
            "reconstruction_cosine": 0.0,
        }
        n_batches = 0
        for images, _, _ in loader:
            images = images.to(self.device)
            if train:
                self.optimizer.zero_grad()
            tokenized, _, objective_output = _forward_pass(
                self.components,
                images,
                lambda_rec=lambda_rec,
                lambda_traj=lambda_traj,
                traj_topk=self.config["objectives"].get("traj_topk", 8),
                traj_gamma=self.config["objectives"].get("traj_gamma", 0.2),
                traj_tau=self.config["objectives"].get("traj_tau", 0.1),
            )
            if train:
                objective_output.total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.components.tokenizer.parameters())
                    + list(self.components.encoder.parameters())
                    + list(self.components.objective.parameters()),
                    max_norm=self.config["training"].get("grad_clip", 1.0),
                )
                self.optimizer.step()

            aggregate["loss"] += float(objective_output.total_loss.item())
            aggregate["pred_loss"] += float(objective_output.pred_loss.item())
            aggregate["rec_loss"] += float(objective_output.rec_loss.item())
            aggregate["traj_loss"] += float(objective_output.traj_loss.item())
            aggregate["prediction_cosine"] += float(objective_output.prediction_cosine.item())
            aggregate["reconstruction_cosine"] += float(objective_output.reconstruction_cosine.item())
            n_batches += 1
        if n_batches == 0:
            return aggregate
        return {key: value / n_batches for key, value in aggregate.items()}

    def _fit_baselines(self) -> BaselineRegressors:
        max_images = self.config["training"].get("baseline_fit_images", 1024)
        local_features, flow_features, next_targets = self._collect_baseline_features(self.loaders["fit"], max_images=max_images)
        return BaselineRegressors.fit(
            local_features=local_features,
            flow_features=flow_features,
            next_targets=next_targets,
            hidden_dim=self.config["training"].get("baseline_hidden_dim"),
            epochs=self.config["training"].get("baseline_epochs", 10),
            batch_size=self.config["training"].get("baseline_batch_size", 1024),
            lr=self.config["training"].get("baseline_lr", 1.0e-3),
            weight_decay=self.config["training"].get("baseline_weight_decay", 1.0e-4),
            seed=self.config["data"].get("seed", 0),
            device=self.device,
        )

    def _evaluate_baselines(self, regressors: BaselineRegressors, *, loader=None, max_images: int | None = None):
        loader = loader or self.loaders["val"]
        if max_images is None:
            max_images = self.config["training"].get("baseline_eval_images", 1024)
        local_features, flow_features, next_targets = self._collect_baseline_features(loader, max_images=max_images)
        return regressors.evaluate(
            local_features=local_features,
            flow_features=flow_features,
            next_targets=next_targets,
        )

    def _score_baselines(self, regressors: BaselineRegressors, *, loader, max_images: int) -> dict[str, np.ndarray]:
        local_features, flow_features, next_targets = self._collect_baseline_features(loader, max_images=max_images)
        return regressors.score_predictions(
            local_features=local_features,
            flow_features=flow_features,
            next_targets=next_targets,
        )

    def _collect_baseline_features(self, loader, *, max_images: int | None) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        return collect_baseline_features(
            self.components,
            loader,
            device=self.device,
            max_images=max_images,
        )

    def _full_validation_metrics(self, loader) -> RepresentationMetrics:
        outputs = collect_model_outputs(
            self.components,
            loader,
            device=self.device,
            max_images=self.config["training"].get("validation_images", 512),
        )
        return evaluate_representation_metrics(
            outputs["z"],
            outputs["local_features"],
            outputs["flow_targets"],
            outputs["future_descriptors"],
            outputs["predicted_next"],
            outputs["reconstructed_current"],
            max_alignment_pairs=self.config["training"].get("alignment_max_pairs", 2048),
            alignment_seed=self.config["data"].get("seed", 0),
        )

    def _run_phase_c_sweep(self, *, phase_b_snapshot: dict, phase_b_metrics: RepresentationMetrics, epochs: int) -> dict:
        best_candidate = None
        exploratory_candidate = None
        ocfg = self.config["objectives"]
        base_lr = self.config["training"].get("lr", 1.0e-3)
        missing = object()
        original_traj_config = {
            key: ocfg.get(key, missing)
            for key in ("lambda_traj", "traj_topk", "traj_gamma", "traj_tau")
        }
        lambda_candidates = ocfg.get("lambda_traj_candidates", [0.1, 0.2, 0.5])
        topk_candidates = ocfg.get("traj_topk_candidates", [ocfg.get("traj_topk", 8)])
        gamma_candidates = ocfg.get("traj_gamma_candidates", [ocfg.get("traj_gamma", 0.2)])
        tau_candidates = ocfg.get("traj_tau_candidates", [ocfg.get("traj_tau", 0.1)])

        n_candidates = len(lambda_candidates) * len(topk_candidates) * len(gamma_candidates) * len(tau_candidates)
        _log_section(
            f"Phase C — Trajectory Alignment Sweep"
            f"  ({n_candidates} candidate{'s' if n_candidates != 1 else ''} x {epochs} epoch{'s' if epochs != 1 else ''})"
        )
        _log("Goal: find hyperparameters where a contrastive alignment loss improves")
        _log("      spatial consistency without hurting prediction quality.")
        _log(f"      Testing {len(lambda_candidates)} lambda x {len(topk_candidates)} topk x {len(gamma_candidates)} gamma x {len(tau_candidates)} tau combinations.\n")
        candidate_idx = 0
        for lambda_traj, traj_topk, traj_gamma, traj_tau in itertools.product(
            lambda_candidates, topk_candidates, gamma_candidates, tau_candidates
        ):
            candidate_idx += 1
            _log(f"  Candidate {candidate_idx}/{n_candidates}  lambda_traj={lambda_traj}  topk={traj_topk}  gamma={traj_gamma}  tau={traj_tau}")
            self._restore_snapshot(phase_b_snapshot)
            self._reset_optimizer_lr(base_lr)
            # Fresh cosine schedule for each Phase C candidate so it decays
            # over phase_c epochs rather than inheriting Phase A+B T_max.
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(epochs, 1))
            # Temporarily override per-candidate trajectory hyperparameters so
            # _run_epoch picks them up from self.config.
            ocfg["lambda_traj"] = lambda_traj
            ocfg["traj_topk"] = traj_topk
            ocfg["traj_gamma"] = traj_gamma
            ocfg["traj_tau"] = traj_tau
            history = self._run_phase(
                "phase_c",
                epochs,
                lambda_rec=ocfg.get("lambda_rec", 0.2),
                lambda_traj=lambda_traj,
            )
            metrics = self._full_validation_metrics(self.loaders["val"])
            accepted = (
                metrics.trajectory_alignment_mean > phase_b_metrics.trajectory_alignment_mean
                and metrics.prediction_cosine_mean >= (phase_b_metrics.prediction_cosine_mean - phase_b_metrics.prediction_cosine_sem)
            )
            traj_diff = metrics.trajectory_alignment_mean - phase_b_metrics.trajectory_alignment_mean
            _log(
                f"    -> traj_align={metrics.trajectory_alignment_mean:.4f} ({'+' if traj_diff >= 0 else ''}{traj_diff:.4f})"
                f"  pred_cos={metrics.prediction_cosine_mean:.4f}"
                f"  {'ACCEPTED' if accepted else 'rejected'}"
            )
            candidate = {
                "lambda_traj": lambda_traj,
                "traj_topk": traj_topk,
                "traj_gamma": traj_gamma,
                "traj_tau": traj_tau,
                "accepted": accepted,
                "history": history,
                "metrics": metrics,
                "snapshot": self._snapshot_state(),
            }
            if exploratory_candidate is None or (
                metrics.trajectory_alignment_mean > exploratory_candidate["metrics"].trajectory_alignment_mean
                or (
                    metrics.trajectory_alignment_mean == exploratory_candidate["metrics"].trajectory_alignment_mean
                    and metrics.prediction_cosine_mean > exploratory_candidate["metrics"].prediction_cosine_mean
                )
            ):
                exploratory_candidate = candidate
            if accepted and (
                best_candidate is None
                or metrics.trajectory_alignment_mean > best_candidate["metrics"].trajectory_alignment_mean
            ):
                best_candidate = candidate

        selected_phase_c_candidate = best_candidate or exploratory_candidate
        if selected_phase_c_candidate is None:
            self._restore_snapshot(phase_b_snapshot)
            _pe = self.config["training"]["phase_epochs"]
            ab_epochs = _pe.get("phase_a", 0) + _pe.get("phase_b", 0)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(ab_epochs, 1))
            for key, value in original_traj_config.items():
                if value is missing:
                    ocfg.pop(key, None)
                else:
                    ocfg[key] = value
            return {
                "accepted": False,
                "reason": "phase_c_candidate_generation_failed",
                "metrics": phase_b_metrics,
            }

        ocfg["lambda_traj"] = selected_phase_c_candidate["lambda_traj"]
        ocfg["traj_topk"] = selected_phase_c_candidate["traj_topk"]
        ocfg["traj_gamma"] = selected_phase_c_candidate["traj_gamma"]
        ocfg["traj_tau"] = selected_phase_c_candidate["traj_tau"]
        self._restore_snapshot(selected_phase_c_candidate["snapshot"])
        self._save_checkpoint(
            name="phase_c.pt",
            phase="phase_c",
            validation=selected_phase_c_candidate["metrics"].to_dict(),
            extra_summary={
                "accepted_by_rule": selected_phase_c_candidate["accepted"],
                "phase_b_metrics": phase_b_metrics.to_dict(),
                "phase_c_metrics": selected_phase_c_candidate["metrics"].to_dict(),
                "lambda_traj": selected_phase_c_candidate["lambda_traj"],
                "traj_topk": selected_phase_c_candidate["traj_topk"],
                "traj_gamma": selected_phase_c_candidate["traj_gamma"],
                "traj_tau": selected_phase_c_candidate["traj_tau"],
            },
        )
        phase_c_checkpoint = str(self.checkpoint_dir / "phase_c.pt")

        if best_candidate is None:
            self._restore_snapshot(phase_b_snapshot)
            # Restore Phase A+B scheduler so the final checkpoint is consistent.
            _pe = self.config["training"]["phase_epochs"]
            ab_epochs = _pe.get("phase_a", 0) + _pe.get("phase_b", 0)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(ab_epochs, 1))
            for key, value in original_traj_config.items():
                if value is missing:
                    ocfg.pop(key, None)
                else:
                    ocfg[key] = value
            _log("\nNo candidate improved trajectory alignment without hurting prediction.")
            _log("Reverting to Phase B checkpoint.\n")
            return {
                "accepted": False,
                "reason": "no_phase_c_candidate_met_acceptance_rule",
                "metrics": selected_phase_c_candidate["metrics"],
                "saved_checkpoint": phase_c_checkpoint,
                "lambda_traj": selected_phase_c_candidate["lambda_traj"],
                "traj_topk": selected_phase_c_candidate["traj_topk"],
                "traj_gamma": selected_phase_c_candidate["traj_gamma"],
                "traj_tau": selected_phase_c_candidate["traj_tau"],
            }

        # Persist the winning hyperparameters in config so the saved checkpoint
        # is self-consistent.
        ocfg["lambda_traj"] = best_candidate["lambda_traj"]
        ocfg["traj_topk"] = best_candidate["traj_topk"]
        ocfg["traj_gamma"] = best_candidate["traj_gamma"]
        ocfg["traj_tau"] = best_candidate["traj_tau"]
        self._restore_snapshot(best_candidate["snapshot"])
        _log(f"\nBest candidate: lambda_traj={best_candidate['lambda_traj']}  topk={best_candidate['traj_topk']}  gamma={best_candidate['traj_gamma']}  tau={best_candidate['traj_tau']}")
        _log(f"  traj_align={best_candidate['metrics'].trajectory_alignment_mean:.4f}  pred_cos={best_candidate['metrics'].prediction_cosine_mean:.4f}\n")
        return {
            "accepted": True,
            "lambda_traj": best_candidate["lambda_traj"],
            "traj_topk": best_candidate["traj_topk"],
            "traj_gamma": best_candidate["traj_gamma"],
            "traj_tau": best_candidate["traj_tau"],
            "metrics": best_candidate["metrics"],
            "saved_checkpoint": phase_c_checkpoint,
        }

    def _snapshot_state(self) -> dict:
        return {
            "observer": deepcopy(self.components.observer.state_dict()),
            "tokenizer": deepcopy(self.components.tokenizer.state_dict()),
            "encoder": deepcopy(self.components.encoder.state_dict()),
            "objective": deepcopy(self.components.objective.state_dict()),
            "optimizer": deepcopy(self.optimizer.state_dict()),
            "scheduler": deepcopy(self.scheduler.state_dict()),
        }

    def _restore_snapshot(self, snapshot: dict) -> None:
        self.components.observer.load_state_dict(snapshot["observer"])
        self.components.tokenizer.load_state_dict(snapshot["tokenizer"])
        self.components.encoder.load_state_dict(snapshot["encoder"])
        self.components.objective.load_state_dict(snapshot["objective"])
        self.optimizer.load_state_dict(snapshot["optimizer"])
        self.scheduler.load_state_dict(snapshot["scheduler"])

    def _reset_optimizer_lr(self, lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
            param_group["initial_lr"] = lr

    def _load_training_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.components.observer.load_state_dict(checkpoint["observer_state"])
        self.components.tokenizer.load_state_dict(checkpoint["tokenizer_state"])
        self.components.encoder.load_state_dict(checkpoint["encoder_state"])
        self.components.objective.load_state_dict(checkpoint["objective_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        observer_metadata = checkpoint.get("observer_metadata", {})
        self.components.observer.classifier_is_trained = bool(observer_metadata.get("classifier_is_trained", False))
        self.resume_checkpoint = checkpoint

    def _save_checkpoint(
        self,
        *,
        name: str,
        phase: str,
        validation: dict,
        extra_summary: dict | None = None,
    ) -> None:
        checkpoint = {
            "version": 1,
            "phase": phase,
            "config": self.config,
            "observer_state": self.components.observer.state_dict(),
            "observer_metadata": {
                "classifier_is_trained": bool(self.components.observer.classifier_is_trained),
            },
            "tokenizer_state": self.components.tokenizer.state_dict(),
            "encoder_state": self.components.encoder.state_dict(),
            "objective_state": self.components.objective.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "validation": validation,
            "summary": extra_summary or {},
        }
        torch.save(checkpoint, self.checkpoint_dir / name)


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _log_section(title: str) -> None:
    bar = "=" * 64
    print(f"\n{bar}", flush=True)
    print(title, flush=True)
    print(bar, flush=True)


def _forward_pass(
    components: LoadedFlowComponents,
    images: torch.Tensor,
    *,
    lambda_rec: float,
    lambda_traj: float,
    traj_topk: int = 8,
    traj_gamma: float = 0.2,
    traj_tau: float = 0.1,
) -> tuple:
    observations = components.observer(images)
    tokenized = components.tokenizer(observations)
    z, _ = components.encoder(tokenized.token_inputs)
    output = components.objective(
        z,
        tokenized.flow_targets,
        tokenized.future_descriptors,
        lambda_pred=components.config["objectives"].get("lambda_pred", 1.0),
        lambda_rec=lambda_rec,
        lambda_traj=lambda_traj,
        traj_topk=traj_topk,
        traj_gamma=traj_gamma,
        traj_tau=traj_tau,
    )
    return tokenized, z, output


def _predict_next_from_latents(
    objective: FlowObjective,
    z: torch.Tensor,
    flow_targets: torch.Tensor,
) -> torch.Tensor:
    batch_size, n_layers, n_cells, token_dim = z.shape
    flow_dim = flow_targets.shape[-1]
    predicted_next = torch.zeros(
        batch_size,
        n_layers - 1,
        n_cells,
        flow_dim,
        device=z.device,
        dtype=z.dtype,
    )
    for layer_idx in range(n_layers - 1):
        z_layer = z[:, layer_idx].reshape(batch_size * n_cells, token_dim)
        pred = objective.prediction_decoders[layer_idx](z_layer).view(batch_size, n_cells, flow_dim)
        predicted_next[:, layer_idx] = pred
    return predicted_next
