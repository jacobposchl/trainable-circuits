"""
Evaluation utilities for causal circuit interventions.

These helpers are notebook-facing and intentionally keep the intervention
pipeline modular:
  - frozen-feature linear probing for a task-level readout
  - grad-enabled CTLS forward passes (without changing training behavior)
  - circuit prototype construction and selection
  - norm-bounded input-space interventions with matched controls
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from evaluation.circuit_analysis import CIFAR10_CLASSES, _MEAN, _STD
from evaluation.discovery import SpanCentricDiscovery


@dataclass
class CircuitPrototype:
    """Per-layer prototype for a discovered circuit span."""

    name: str
    span: tuple[int, int]
    vectors: list[torch.Tensor]
    circuit_type: str
    size: int
    purity: float
    elevation_sigma: float
    associated_class: int | None = None
    associated_label: str | None = None
    source_cluster_id: int | None = None


def _register_penultimate_hook(model: nn.Module, storage: list[torch.Tensor]):
    if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        return model.fc.register_forward_pre_hook(
            lambda module, inputs: storage.append(inputs[0])
        )
    if hasattr(model, "head") and isinstance(model.head, nn.Module):
        return model.head.register_forward_pre_hook(
            lambda module, inputs: storage.append(inputs[0])
        )
    raise ValueError("Unsupported backbone model: no classifier head found.")


def extract_penultimate_features(
    backbone,
    images: torch.Tensor,
) -> torch.Tensor:
    """
    Extract frozen penultimate features used by the evaluation-only linear probe.
    """
    feats: list[torch.Tensor] = []
    handle = _register_penultimate_hook(backbone.model, feats)
    try:
        backbone.model(images)
    finally:
        handle.remove()
    if not feats:
        raise RuntimeError("Failed to capture penultimate backbone features.")
    features = feats[0]
    if features.dim() > 2:
        features = features.flatten(1)
    return features


def forward_ctls_with_grad(
    backbone,
    meta_encoder,
    images: torch.Tensor,
) -> dict:
    """
    Grad-enabled CTLS forward pass for evaluation-time interventions.

    This mirrors the frozen backbone compression used during training while
    avoiding the no_grad / detach behavior in ``FrozenBackbone.forward``.
    """
    trajectory: list[torch.Tensor] = []
    flow_targets: list[torch.Tensor] = []
    penultimate: list[torch.Tensor] = []
    handles: list = []

    def _traj_hook(idx: int):
        def hook(module, inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if tensor.dim() == 3:
                tensor = tensor[:, 0]
            compressed = backbone.traj_compressors[idx](tensor)
            trajectory.append(F.normalize(compressed, dim=-1))

        return hook

    def _flow_hook(idx: int):
        def hook(module, inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            compressed = backbone.flow_compressors[idx](tensor)
            flow_targets.append(F.normalize(compressed, dim=-1))

        return hook

    for idx, module in enumerate(backbone._block_modules):
        handles.append(module.register_forward_hook(_traj_hook(idx)))

    if not backbone._is_vit and backbone._bn2_modules:
        for idx, module in enumerate(backbone._bn2_modules):
            handles.append(module.register_forward_hook(_flow_hook(idx)))

    handles.append(_register_penultimate_hook(backbone.model, penultimate))

    try:
        logits = backbone.model(images)
    finally:
        for handle in handles:
            handle.remove()

    if backbone._is_vit:
        flow_targets = list(trajectory)

    z_list = meta_encoder(trajectory)
    features = penultimate[0]
    if features.dim() > 2:
        features = features.flatten(1)

    return {
        "trajectory": trajectory,
        "flow_targets": flow_targets,
        "z_list": z_list,
        "penultimate": features,
        "logits": logits,
    }


@torch.no_grad()
def collect_probe_features(
    backbone,
    loader: DataLoader,
    device: torch.device,
    max_samples: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect penultimate frozen-backbone features for probe fitting."""
    all_features = []
    all_labels = []
    n = 0

    for images, labels in loader:
        images = images.to(device)
        features = extract_penultimate_features(backbone, images).cpu()
        all_features.append(features)
        all_labels.append(labels.cpu())
        n += images.shape[0]
        if max_samples is not None and n >= max_samples:
            break

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    if max_samples is not None:
        features = features[:max_samples]
        labels = labels[:max_samples]
    return features, labels


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def fit_linear_probe_from_features(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    num_classes: int | None = None,
    epochs: int = 100,
    lr: float = 1.0e-2,
    weight_decay: float = 1.0e-4,
    batch_size: int = 256,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict]:
    """
    Fit an evaluation-only linear probe on frozen backbone features.
    """
    if num_classes is None:
        num_classes = int(max(train_labels.max(), val_labels.max()).item()) + 1

    device = torch.device(device)
    probe = nn.Linear(train_features.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    train_ds = TensorDataset(train_features.float(), train_labels.long())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_state = None
    best_val_acc = -float("inf")
    history = []

    val_features = val_features.to(device).float()
    val_labels = val_labels.to(device).long()

    for epoch in range(epochs):
        probe.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device).float()
            batch_labels = batch_labels.to(device).long()

            optimizer.zero_grad()
            logits = probe(batch_features)
            loss = F.cross_entropy(logits, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_labels.shape[0]
            total_correct += int((logits.argmax(dim=1) == batch_labels).sum().item())
            total_examples += batch_labels.shape[0]

        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_features)
            val_acc = _accuracy(val_logits, val_labels)

        train_loss = total_loss / max(total_examples, 1)
        train_acc = total_correct / max(total_examples, 1)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in probe.state_dict().items()
            }

    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()

    return probe, {"best_val_acc": best_val_acc, "history": history}


def fit_linear_probe(
    backbone,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    epochs: int = 100,
    lr: float = 1.0e-2,
    weight_decay: float = 1.0e-4,
    batch_size: int = 256,
) -> tuple[nn.Module, dict]:
    """Collect frozen features and fit a linear probe."""
    train_features, train_labels = collect_probe_features(
        backbone, train_loader, device, max_samples=max_train_samples
    )
    val_features, val_labels = collect_probe_features(
        backbone, val_loader, device, max_samples=max_val_samples
    )
    probe, metrics = fit_linear_probe_from_features(
        train_features,
        train_labels,
        val_features,
        val_labels,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        device=device,
    )
    metrics.update(
        {
            "train_features": train_features,
            "train_labels": train_labels,
            "val_features": val_features,
            "val_labels": val_labels,
        }
    )
    return probe, metrics


def _majority_label(labels: np.ndarray) -> tuple[int, float]:
    counts = np.bincount(labels)
    cls = int(counts.argmax())
    purity = float(counts.max()) / len(labels)
    return cls, purity


def build_circuit_prototype(
    circuit: dict,
    z_list: list[np.ndarray] | list[torch.Tensor],
    *,
    name: str | None = None,
    circuit_type: str = "unknown",
    associated_class: int | None = None,
    associated_label: str | None = None,
    purity: float = 0.0,
    elevation_sigma: float = 0.0,
) -> CircuitPrototype:
    """Build a per-layer circuit prototype across the discovered span."""
    l_start, l_end = circuit["span"]
    mask = np.asarray(circuit["image_mask"], dtype=bool)
    vectors = []
    for layer_idx in range(l_start, l_end + 1):
        layer = z_list[layer_idx]
        if isinstance(layer, np.ndarray):
            proto = torch.from_numpy(layer[mask]).float().mean(dim=0)
        else:
            proto = layer[mask].float().mean(dim=0)
        vectors.append(F.normalize(proto, dim=0))

    return CircuitPrototype(
        name=name or f"span_L{l_start+1}_L{l_end+1}_cluster_{circuit.get('cluster_id', 0)}",
        span=circuit["span"],
        vectors=vectors,
        circuit_type=circuit_type,
        size=int(circuit.get("size", int(mask.sum()))),
        purity=float(purity),
        elevation_sigma=float(elevation_sigma),
        associated_class=associated_class,
        associated_label=associated_label,
        source_cluster_id=circuit.get("cluster_id"),
    )


def build_circuit_library(
    circuits: list[dict],
    z_list: list[np.ndarray] | list[torch.Tensor],
    labels: np.ndarray | torch.Tensor,
    *,
    purity_specific: float = 0.7,
    purity_agnostic: float = 0.3,
) -> list[dict]:
    """Annotate discovered circuits with labels, purity, and prototypes."""
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    library = []

    for idx, circuit in enumerate(circuits):
        mask = np.asarray(circuit["image_mask"], dtype=bool)
        if mask.sum() == 0:
            continue
        member_labels = labels_np[mask]
        majority_class, purity = _majority_label(member_labels)
        if purity >= purity_specific:
            circuit_type = "class_specific"
        elif purity <= purity_agnostic:
            circuit_type = "class_agnostic"
        else:
            circuit_type = "mixed"

        name = f"{circuit_type}_{idx+1}_L{circuit['span'][0]+1}_L{circuit['span'][1]+1}"
        prototype = build_circuit_prototype(
            circuit,
            z_list,
            name=name,
            circuit_type=circuit_type,
            associated_class=majority_class if circuit_type != "class_agnostic" else None,
            associated_label=CIFAR10_CLASSES[majority_class] if majority_class < len(CIFAR10_CLASSES) else str(majority_class),
            purity=purity,
            elevation_sigma=float(circuit.get("elevation_sigma", 0.0)),
        )

        enriched = dict(circuit)
        enriched.update(
            {
                "name": name,
                "member_indices": np.flatnonzero(mask),
                "majority_class": majority_class,
                "associated_class": prototype.associated_class,
                "associated_label": prototype.associated_label,
                "purity": purity,
                "circuit_type": circuit_type,
                "prototype": prototype,
            }
        )
        library.append(enriched)

    return library


def select_circuit_set(
    library: list[dict],
    *,
    n_specific: int = 2,
    n_agnostic: int = 2,
    min_size: int = 20,
) -> list[dict]:
    """
    Select a small mixed set of high-quality circuits for interventions.
    """
    eligible = [c for c in library if c["size"] >= min_size]

    specific = sorted(
        [c for c in eligible if c["circuit_type"] == "class_specific"],
        key=lambda c: (c.get("elevation_sigma", 0.0), c["purity"], c["size"]),
        reverse=True,
    )
    agnostic = sorted(
        [c for c in eligible if c["circuit_type"] == "class_agnostic"],
        key=lambda c: (c.get("elevation_sigma", 0.0), c["size"]),
        reverse=True,
    )

    chosen = []
    seen_classes = set()
    for circuit in specific:
        cls = circuit.get("associated_class")
        if cls in seen_classes:
            continue
        chosen.append(circuit)
        seen_classes.add(cls)
        if len([c for c in chosen if c["circuit_type"] == "class_specific"]) >= n_specific:
            break

    chosen.extend(agnostic[:n_agnostic])
    return chosen


def compute_circuit_score(
    z_list: list[torch.Tensor],
    prototype: CircuitPrototype,
) -> torch.Tensor:
    """Mean cosine alignment over the prototype's span."""
    scores = []
    for offset, layer_idx in enumerate(range(prototype.span[0], prototype.span[1] + 1)):
        proto = prototype.vectors[offset].to(z_list[layer_idx].device)
        proto = F.normalize(proto, dim=0)
        scores.append((z_list[layer_idx] * proto.unsqueeze(0)).sum(dim=1))
    return torch.stack(scores, dim=1).mean(dim=1)


def random_direction_prototype(
    span: tuple[int, int],
    dim: int,
    *,
    generator: torch.Generator | None = None,
    name: str = "random_direction",
) -> CircuitPrototype:
    vectors = []
    for _ in range(span[0], span[1] + 1):
        vec = torch.randn(dim, generator=generator)
        vectors.append(F.normalize(vec, dim=0))
    return CircuitPrototype(
        name=name,
        span=span,
        vectors=vectors,
        circuit_type="control",
        size=0,
        purity=0.0,
        elevation_sigma=0.0,
    )


def build_control_prototypes(
    target: dict,
    library: list[dict],
    z_list: list[np.ndarray] | list[torch.Tensor],
    *,
    seed: int = 0,
) -> dict[str, CircuitPrototype]:
    """
    Build matched control prototypes for a selected circuit.
    """
    rng = np.random.default_rng(seed)
    span = target["span"]
    span_len = span[1] - span[0] + 1
    dim = target["prototype"].vectors[0].numel()

    candidate_indices = np.arange(z_list[0].shape[0])
    available = np.setdiff1d(candidate_indices, target["member_indices"])
    sample_size = min(target["size"], max(len(available), 1))
    if len(available) == 0:
        available = candidate_indices
    sampled = rng.choice(available, size=sample_size, replace=False)
    random_mask = np.zeros(z_list[0].shape[0], dtype=bool)
    random_mask[sampled] = True

    matched_proto = build_circuit_prototype(
        {"span": span, "image_mask": random_mask, "size": int(random_mask.sum())},
        z_list,
        name=f"{target['name']}_matched_random",
        circuit_type="control",
    )

    wrong_candidates = [
        c for c in library
        if c["name"] != target["name"] and (c["span"][1] - c["span"][0] + 1) == span_len
    ]
    if not wrong_candidates:
        wrong_candidates = [c for c in library if c["name"] != target["name"]]
    wrong = max(
        wrong_candidates,
        key=lambda c: c.get("elevation_sigma", 0.0),
        default=target,
    )

    random_proto = random_direction_prototype(
        span,
        dim,
        generator=torch.Generator().manual_seed(seed),
        name=f"{target['name']}_random_direction",
    )

    return {
        "matched_random": matched_proto,
        "wrong_circuit": wrong["prototype"],
        "random_direction": random_proto,
    }


def normalized_linf_budget(eps_pixels: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert pixel-space Linf epsilon / step sizes into normalized space."""
    std = _STD.float()
    eps = torch.full((3,), eps_pixels) / std
    lower = (torch.zeros(3) - _MEAN.float()) / std
    upper = (torch.ones(3) - _MEAN.float()) / std
    return eps, torch.stack([lower, upper], dim=0)


def optimize_images_for_score(
    images: torch.Tensor,
    score_fn,
    *,
    mode: str,
    eps: torch.Tensor,
    step_size: torch.Tensor,
    n_steps: int,
    clamp_bounds: torch.Tensor,
) -> torch.Tensor:
    """
    Generic Linf-bounded PGD optimizer for a differentiable score.
    """
    eps = eps.to(images.device).view(1, -1, 1, 1)
    step_size = step_size.to(images.device).view(1, -1, 1, 1)
    lower = clamp_bounds[0].to(images.device).view(1, -1, 1, 1)
    upper = clamp_bounds[1].to(images.device).view(1, -1, 1, 1)

    x0 = images.detach()
    x = x0.clone()

    for _ in range(n_steps):
        x.requires_grad_(True)
        score = score_fn(x).mean()
        loss = -score if mode == "activate" else score
        grad = torch.autograd.grad(loss, x)[0]
        x = x.detach() - step_size * grad.sign()
        x = torch.max(torch.min(x, x0 + eps), x0 - eps)
        x = torch.max(torch.min(x, upper), lower)

    return x.detach()


@torch.no_grad()
def summarize_probe_outputs(
    logits: torch.Tensor,
    *,
    associated_class: int | None = None,
) -> dict:
    probs = logits.softmax(dim=1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1)
    confidence, preds = probs.max(dim=1)

    summary = {
        "mean_entropy": float(entropy.mean().item()),
        "mean_confidence": float(confidence.mean().item()),
        "top1_classes": preds.cpu(),
        "probs": probs.cpu(),
        "logits": logits.cpu(),
    }

    if associated_class is not None:
        summary.update(
            {
                "mean_associated_logit": float(logits[:, associated_class].mean().item()),
                "mean_associated_prob": float(probs[:, associated_class].mean().item()),
                "associated_pred_rate": float((preds == associated_class).float().mean().item()),
            }
        )
    return summary


def run_intervention_batch(
    backbone,
    meta_encoder,
    probe: nn.Module,
    images: torch.Tensor,
    prototype: CircuitPrototype,
    *,
    mode: str = "activate",
    eps_pixels: float = 8.0 / 255.0,
    step_pixels: float = 2.0 / 255.0,
    n_steps: int = 10,
) -> dict:
    """
    Run a norm-bounded input-space intervention against a circuit score.
    """
    backbone.eval()
    meta_encoder.eval()
    probe.eval()

    eps, clamp_bounds = normalized_linf_budget(eps_pixels)
    step_size = torch.full((3,), step_pixels) / _STD.float()

    def _score_fn(batch_images: torch.Tensor) -> torch.Tensor:
        outputs = forward_ctls_with_grad(backbone, meta_encoder, batch_images)
        return compute_circuit_score(outputs["z_list"], prototype)

    before = forward_ctls_with_grad(backbone, meta_encoder, images)
    before_scores = compute_circuit_score(before["z_list"], prototype)
    before_probe = probe(before["penultimate"])
    before_summary = summarize_probe_outputs(
        before_probe,
        associated_class=prototype.associated_class,
    )

    intervened_images = optimize_images_for_score(
        images,
        _score_fn,
        mode=mode,
        eps=eps,
        step_size=step_size,
        n_steps=n_steps,
        clamp_bounds=clamp_bounds,
    )

    after = forward_ctls_with_grad(backbone, meta_encoder, intervened_images)
    after_scores = compute_circuit_score(after["z_list"], prototype)
    after_probe = probe(after["penultimate"])
    after_summary = summarize_probe_outputs(
        after_probe,
        associated_class=prototype.associated_class,
    )

    summary = {
        "circuit_name": prototype.name,
        "circuit_type": prototype.circuit_type,
        "mode": mode,
        "mean_score_before": float(before_scores.mean().item()),
        "mean_score_after": float(after_scores.mean().item()),
        "delta_score": float((after_scores - before_scores).mean().item()),
        "mean_confidence_before": before_summary["mean_confidence"],
        "mean_confidence_after": after_summary["mean_confidence"],
        "delta_confidence": after_summary["mean_confidence"] - before_summary["mean_confidence"],
        "mean_entropy_before": before_summary["mean_entropy"],
        "mean_entropy_after": after_summary["mean_entropy"],
        "delta_entropy": after_summary["mean_entropy"] - before_summary["mean_entropy"],
        "associated_class": prototype.associated_class,
        "associated_label": prototype.associated_label,
    }

    if prototype.associated_class is not None:
        summary.update(
            {
                "mean_associated_logit_before": before_summary["mean_associated_logit"],
                "mean_associated_logit_after": after_summary["mean_associated_logit"],
                "delta_associated_logit": after_summary["mean_associated_logit"] - before_summary["mean_associated_logit"],
                "mean_associated_prob_before": before_summary["mean_associated_prob"],
                "mean_associated_prob_after": after_summary["mean_associated_prob"],
                "delta_associated_prob": after_summary["mean_associated_prob"] - before_summary["mean_associated_prob"],
                "associated_pred_rate_before": before_summary["associated_pred_rate"],
                "associated_pred_rate_after": after_summary["associated_pred_rate"],
                "delta_associated_pred_rate": after_summary["associated_pred_rate"] - before_summary["associated_pred_rate"],
            }
        )

    return {
        "images_before": images.detach().cpu(),
        "images_after": intervened_images.detach().cpu(),
        "scores_before": before_scores.detach().cpu(),
        "scores_after": after_scores.detach().cpu(),
        "probe_logits_before": before_probe.detach().cpu(),
        "probe_logits_after": after_probe.detach().cpu(),
        "summary": summary,
    }


def summarize_intervention_results(results: list[dict]) -> list[dict]:
    """Flatten intervention outputs into a notebook-friendly table."""
    return [dict(result["summary"]) for result in results]
