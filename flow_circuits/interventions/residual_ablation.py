from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch

from flow_circuits.evaluation import bootstrap_mean_ci
from flow_circuits.training import LoadedFlowComponents


@dataclass
class InterventionResult:
    circuit_id: int
    n_members: int
    n_controls: int
    mean_member_delta_margin: float
    mean_member_delta_true: float
    mean_nonmember_delta_margin: float
    mean_nonmember_delta_true: float
    mean_random_node_delta_margin: float
    mean_random_cell_delta_margin: float
    p_member_vs_nonmember: float
    p_member_vs_random_node: float
    p_member_vs_random_cell: float
    corrected_p_member_vs_nonmember: float
    corrected_p_member_vs_random_node: float
    corrected_p_member_vs_random_cell: float
    ci_member_vs_nonmember: list[float]
    ci_member_vs_random_node: list[float]
    ci_member_vs_random_cell: list[float]
    validated: bool

    def to_dict(self) -> dict:
        return asdict(self)


class ResidualPatchAblator:
    def __init__(self, components: LoadedFlowComponents, grid_size: int = 4) -> None:
        self.components = components
        self.grid_size = grid_size

    def ablate(self, images: torch.Tensor, nodes: list[tuple[int, int]]) -> torch.Tensor:
        handles = []
        nodes_by_layer: dict[int, list[int]] = {}
        for layer_idx, cell_idx in nodes:
            nodes_by_layer.setdefault(int(layer_idx), []).append(int(cell_idx))

        def make_hook(cell_indices: list[int]):
            def hook(module, inputs, output):
                tensor = output.clone()
                _, _, height, width = tensor.shape
                for cell_idx in cell_indices:
                    row = cell_idx // self.grid_size
                    col = cell_idx % self.grid_size
                    h0, h1 = _bounds(height, row, self.grid_size)
                    w0, w1 = _bounds(width, col, self.grid_size)
                    tensor[:, :, h0:h1, w0:w1] = 0.0
                return tensor

            return hook

        for layer_idx, cell_indices in nodes_by_layer.items():
            handles.append(
                self.components.observer.flow_modules[layer_idx].register_forward_hook(make_hook(cell_indices))
            )
        try:
            with torch.no_grad():
                return self.components.observer.model(images)
        finally:
            for handle in handles:
                handle.remove()


def assign_circuit_members(
    circuit: dict,
    future_descriptors: torch.Tensor,
    dataset_indices: torch.Tensor,
) -> torch.Tensor:
    active_nodes = [tuple(node) for node in circuit["active_nodes"]]
    if not active_nodes:
        return torch.zeros(future_descriptors.shape[0], dtype=torch.bool)
    representative_node = tuple(circuit["representative_node"])
    device = future_descriptors.device
    dtype = future_descriptors.dtype
    layer_indices = torch.tensor([node[0] for node in active_nodes], dtype=torch.long, device=device)
    cell_indices = torch.tensor([node[1] for node in active_nodes], dtype=torch.long, device=device)
    centroid_matrix = torch.stack(
        [
            torch.tensor(circuit["centroids"][f"{layer_idx}:{cell_idx}"], dtype=dtype, device=device)
            for layer_idx, cell_idx in active_nodes
        ],
        dim=0,
    )
    threshold_vector = torch.tensor(
        [float(circuit["thresholds"][f"{layer_idx}:{cell_idx}"]) for layer_idx, cell_idx in active_nodes],
        dtype=dtype,
        device=device,
    )
    representative_position = active_nodes.index(representative_node)
    node_descriptors = future_descriptors[:, layer_indices, cell_indices, :]
    scores = (node_descriptors * centroid_matrix.unsqueeze(0)).sum(dim=-1)
    representative_ok = scores[:, representative_position] >= threshold_vector[representative_position]
    satisfied = (scores >= threshold_vector.unsqueeze(0)).sum(dim=1)
    return representative_ok & (satisfied >= max(1, int(np.ceil(0.5 * len(active_nodes)))))


def run_circuit_interventions(
    components: LoadedFlowComponents,
    circuits_artifact: dict,
    test_outputs: dict[str, torch.Tensor],
    *,
    alpha: float = 0.05,
    output_path: str | Path | None = None,
    progress_callback=None,
    n_jobs: int = 1,
) -> list[InterventionResult]:
    components.observer.require_semantic_logits()
    future_descriptors = test_outputs["future_descriptors"]
    dataset_indices = test_outputs["indices"]
    labels = test_outputs["labels"]
    logits = test_outputs["logits"]
    margins = _margin(logits)
    predicted_classes = logits.argmax(dim=1)
    ablator = ResidualPatchAblator(components, grid_size=circuits_artifact["metadata"]["grid_size"])

    raw_results = []
    total_circuits = len(circuits_artifact["circuits"])
    for circuit_idx, circuit in enumerate(circuits_artifact["circuits"], start=1):
        member_mask = assign_circuit_members(circuit, future_descriptors, dataset_indices)
        member_rows = torch.nonzero(member_mask, as_tuple=False).flatten()
        if member_rows.numel() == 0:
            if progress_callback is not None:
                progress_callback(
                    completed=circuit_idx,
                    total=total_circuits,
                    circuit_id=int(circuit["id"]),
                    status="skipped_no_members",
                )
            continue

        member_images = test_outputs["images"][member_rows].to(components.encoder.final_norm.weight.device)
        member_labels = labels[member_rows]
        member_before = logits[member_rows]
        member_after = ablator.ablate(member_images, [tuple(node) for node in circuit["active_nodes"]]).cpu()
        member_delta_margin = _margin(member_before) - _margin(member_after)
        member_delta_true = _true_logit_delta(member_before, member_after, member_labels)

        control_rows = _matched_nonmembers(
            member_rows=member_rows,
            member_classes=predicted_classes[member_rows],
            member_margins=margins[member_rows],
            predicted_classes=predicted_classes,
            margins=margins,
        )
        if control_rows.numel() == 0:
            if progress_callback is not None:
                progress_callback(
                    completed=circuit_idx,
                    total=total_circuits,
                    circuit_id=int(circuit["id"]),
                    status="skipped_no_controls",
                )
            continue
        control_images = test_outputs["images"][control_rows].to(member_images.device)
        control_labels = labels[control_rows]
        control_before = logits[control_rows]
        control_after = ablator.ablate(control_images, [tuple(node) for node in circuit["active_nodes"]]).cpu()
        control_delta_margin = _margin(control_before) - _margin(control_after)
        control_delta_true = _true_logit_delta(control_before, control_after, control_labels)

        random_nodes = _random_nodes_like(
            [tuple(node) for node in circuit["active_nodes"]],
            circuits_artifact["metadata"]["n_cells"],
            seed=int(circuit["id"]) + 17,
        )
        random_after = ablator.ablate(member_images, random_nodes).cpu()
        random_delta_margin = _margin(member_before) - _margin(random_after)

        random_cells = _same_layer_random_cells(
            [tuple(node) for node in circuit["active_nodes"]],
            circuits_artifact["metadata"]["n_cells"],
            seed=int(circuit["id"]) + 29,
        )
        random_cell_after = ablator.ablate(member_images, random_cells).cpu()
        random_cell_delta_margin = _margin(member_before) - _margin(random_cell_after)

        member_vs_nonmember = (member_delta_margin[:control_delta_margin.shape[0]] - control_delta_margin).numpy()
        member_vs_random = (member_delta_margin - random_delta_margin).numpy()
        member_vs_random_cell = (member_delta_margin - random_cell_delta_margin).numpy()
        raw_results.append(
            {
                "circuit_id": int(circuit["id"]),
                "n_members": int(member_rows.numel()),
                "n_controls": int(control_rows.numel()),
                "mean_member_delta_margin": float(member_delta_margin.mean().item()),
                "mean_member_delta_true": float(member_delta_true.mean().item()),
                "mean_nonmember_delta_margin": float(control_delta_margin.mean().item()),
                "mean_nonmember_delta_true": float(control_delta_true.mean().item()),
                "mean_random_node_delta_margin": float(random_delta_margin.mean().item()),
                "mean_random_cell_delta_margin": float(random_cell_delta_margin.mean().item()),
                "member_vs_nonmember": member_vs_nonmember,
                "member_vs_random_node": member_vs_random,
                "member_vs_random_cell": member_vs_random_cell,
            }
        )
        if progress_callback is not None:
            progress_callback(
                completed=circuit_idx,
                total=total_circuits,
                circuit_id=int(circuit["id"]),
                status="completed",
            )

    summarized = _summarize_intervention_statistics(raw_results, n_jobs=max(1, int(n_jobs)))
    corrected_nonmember = _holm([item["p_member_vs_nonmember"] for item in summarized])
    corrected_random = _holm([item["p_member_vs_random_node"] for item in summarized])
    corrected_random_cell = _holm([item["p_member_vs_random_cell"] for item in summarized])
    results = []
    for idx, item in enumerate(summarized):
        validated = (
            corrected_nonmember[idx] < alpha
            and corrected_random[idx] < alpha
            and corrected_random_cell[idx] < alpha
            and item["ci_member_vs_nonmember"][0] > 0.0
            and item["ci_member_vs_random_node"][0] > 0.0
            and item["ci_member_vs_random_cell"][0] > 0.0
        )
        result = InterventionResult(
            circuit_id=item["circuit_id"],
            n_members=item["n_members"],
            n_controls=item["n_controls"],
            mean_member_delta_margin=item["mean_member_delta_margin"],
            mean_member_delta_true=item["mean_member_delta_true"],
            mean_nonmember_delta_margin=item["mean_nonmember_delta_margin"],
            mean_nonmember_delta_true=item["mean_nonmember_delta_true"],
            mean_random_node_delta_margin=item["mean_random_node_delta_margin"],
            mean_random_cell_delta_margin=item["mean_random_cell_delta_margin"],
            p_member_vs_nonmember=item["p_member_vs_nonmember"],
            p_member_vs_random_node=item["p_member_vs_random_node"],
            p_member_vs_random_cell=item["p_member_vs_random_cell"],
            corrected_p_member_vs_nonmember=float(corrected_nonmember[idx]),
            corrected_p_member_vs_random_node=float(corrected_random[idx]),
            corrected_p_member_vs_random_cell=float(corrected_random_cell[idx]),
            ci_member_vs_nonmember=item["ci_member_vs_nonmember"],
            ci_member_vs_random_node=item["ci_member_vs_random_node"],
            ci_member_vs_random_cell=item["ci_member_vs_random_cell"],
            validated=validated,
        )
        results.append(result)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump([result.to_dict() for result in results], handle, indent=2)
    return results


def _summarize_intervention_statistics(raw_results: list[dict], *, n_jobs: int) -> list[dict]:
    if n_jobs <= 1 or len(raw_results) <= 1:
        return [_compute_statistical_summary(item) for item in raw_results]
    with ThreadPoolExecutor(max_workers=min(n_jobs, len(raw_results))) as executor:
        return list(executor.map(_compute_statistical_summary, raw_results))


def _compute_statistical_summary(item: dict) -> dict:
    circuit_id = int(item["circuit_id"])
    member_vs_nonmember = item["member_vs_nonmember"]
    member_vs_random = item["member_vs_random_node"]
    member_vs_random_cell = item["member_vs_random_cell"]
    summary = {
        key: value
        for key, value in item.items()
        if key not in {"member_vs_nonmember", "member_vs_random_node", "member_vs_random_cell"}
    }
    p_nonmember = _paired_permutation_pvalue(member_vs_nonmember, seed=circuit_id + 101)
    p_random = _paired_permutation_pvalue(member_vs_random, seed=circuit_id + 151)
    p_random_cell = _paired_permutation_pvalue(member_vs_random_cell, seed=circuit_id + 181)
    ci_nonmember = bootstrap_mean_ci(member_vs_nonmember, n_bootstrap=500, seed=circuit_id + 201)
    ci_random = bootstrap_mean_ci(member_vs_random, n_bootstrap=500, seed=circuit_id + 251)
    ci_random_cell = bootstrap_mean_ci(member_vs_random_cell, n_bootstrap=500, seed=circuit_id + 281)
    summary["p_member_vs_nonmember"] = float(p_nonmember)
    summary["p_member_vs_random_node"] = float(p_random)
    summary["p_member_vs_random_cell"] = float(p_random_cell)
    summary["ci_member_vs_nonmember"] = [float(ci_nonmember[0]), float(ci_nonmember[1])]
    summary["ci_member_vs_random_node"] = [float(ci_random[0]), float(ci_random[1])]
    summary["ci_member_vs_random_cell"] = [float(ci_random_cell[0]), float(ci_random_cell[1])]
    return summary


def _margin(logits: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(logits, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def _true_logit_delta(before: torch.Tensor, after: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    label_idx = labels.view(-1, 1)
    before_true = before.gather(1, label_idx).squeeze(1)
    after_true = after.gather(1, label_idx).squeeze(1)
    return before_true - after_true


def _matched_nonmembers(
    *,
    member_rows: torch.Tensor,
    member_classes: torch.Tensor,
    member_margins: torch.Tensor,
    predicted_classes: torch.Tensor,
    margins: torch.Tensor,
) -> torch.Tensor:
    available = torch.ones_like(predicted_classes, dtype=torch.bool)
    available[member_rows] = False
    chosen = []
    for member_class, member_margin in zip(member_classes.tolist(), member_margins.tolist()):
        class_mask = available & (predicted_classes == member_class)
        candidates = torch.nonzero(class_mask, as_tuple=False).flatten()
        if candidates.numel() == 0:
            candidates = torch.nonzero(available, as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        distances = (margins[candidates] - member_margin).abs()
        best = candidates[int(torch.argmin(distances).item())]
        chosen.append(int(best.item()))
        available[best] = False
    return torch.tensor(chosen, dtype=torch.long)


def _random_nodes_like(nodes: list[tuple[int, int]], n_cells: int, seed: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    nodes_by_layer: dict[int, list[int]] = {}
    for layer_idx, cell_idx in nodes:
        nodes_by_layer.setdefault(layer_idx, []).append(cell_idx)

    random_nodes = []
    for layer_idx, cell_indices in nodes_by_layer.items():
        available = [cell for cell in range(n_cells) if cell not in cell_indices]
        if len(available) >= len(cell_indices):
            sampled = rng.choice(available, size=len(cell_indices), replace=False).tolist()
        else:
            fallback = available if available else list(range(n_cells))
            sampled = rng.choice(fallback, size=len(cell_indices), replace=True).tolist()
        random_nodes.extend((layer_idx, int(cell_idx)) for cell_idx in sampled)
    return random_nodes


def _same_layer_random_cells(nodes: list[tuple[int, int]], n_cells: int, seed: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    out = []
    for layer_idx, cell_idx in nodes:
        choices = [cell for cell in range(n_cells) if cell != cell_idx]
        out.append((layer_idx, int(rng.choice(choices))))
    return out


def _bounds(size: int, idx: int, n_bins: int) -> tuple[int, int]:
    start = int(np.floor((idx * size) / n_bins))
    end = int(np.floor(((idx + 1) * size) / n_bins))
    return start, max(end, start + 1)


def _paired_permutation_pvalue(differences: np.ndarray, *, seed: int, n_resamples: int = 1000) -> float:
    if differences.size < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    observed = float(np.mean(differences))
    null_means = np.empty(n_resamples, dtype=np.float64)
    for idx in range(n_resamples):
        signs = rng.choice(np.array([-1.0, 1.0]), size=differences.size, replace=True)
        null_means[idx] = float(np.mean(differences * signs))
    return float((np.sum(null_means >= observed) + 1.0) / (n_resamples + 1.0))


def _holm(pvalues: list[float]) -> list[float]:
    if not pvalues:
        return []
    order = np.argsort(pvalues)
    sorted_adjusted = np.empty(len(pvalues), dtype=float)
    m = len(pvalues)
    running_max = 0.0
    for rank, idx in enumerate(order):
        corrected = min((m - rank) * pvalues[idx], 1.0)
        running_max = max(running_max, corrected)
        sorted_adjusted[rank] = running_max
    adjusted = np.empty(len(pvalues), dtype=float)
    for rank, idx in enumerate(order):
        adjusted[idx] = sorted_adjusted[rank]
    return adjusted.tolist()
