from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.stats import permutation_test
import torch

from flow_circuits.training import LoadedFlowComponents


@dataclass
class InterventionResult:
    circuit_id: int
    n_members: int
    n_controls: int
    mean_member_delta_margin: float
    mean_nonmember_delta_margin: float
    mean_random_node_delta_margin: float
    mean_random_cell_delta_margin: float
    p_member_vs_nonmember: float
    p_member_vs_random_node: float
    corrected_p_member_vs_nonmember: float
    corrected_p_member_vs_random_node: float
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
    representative_node = tuple(circuit["representative_node"])
    centroids = {
        tuple(int(value) for value in key.split(":")): torch.tensor(vec, dtype=future_descriptors.dtype)
        for key, vec in circuit["centroids"].items()
    }
    thresholds = {
        tuple(int(value) for value in key.split(":")): float(value)
        for key, value in circuit["thresholds"].items()
    }
    row_mask = torch.zeros(future_descriptors.shape[0], dtype=torch.bool)

    for row_idx in range(future_descriptors.shape[0]):
        rep_centroid = centroids[representative_node].to(future_descriptors.device)
        rep_score = torch.dot(
            future_descriptors[row_idx, representative_node[0], representative_node[1]],
            rep_centroid,
        ).item()
        if rep_score < thresholds[representative_node]:
            continue
        satisfied = 0
        for node in active_nodes:
            centroid = centroids[node].to(future_descriptors.device)
            score = torch.dot(future_descriptors[row_idx, node[0], node[1]], centroid).item()
            if score >= thresholds[node]:
                satisfied += 1
        if satisfied >= max(1, int(np.ceil(0.5 * len(active_nodes)))):
            row_mask[row_idx] = True
    return row_mask


def run_circuit_interventions(
    components: LoadedFlowComponents,
    circuits_artifact: dict,
    test_outputs: dict[str, torch.Tensor],
    *,
    alpha: float = 0.05,
    output_path: str | Path | None = None,
) -> list[InterventionResult]:
    future_descriptors = test_outputs["future_descriptors"]
    dataset_indices = test_outputs["indices"]
    labels = test_outputs["labels"]
    logits = test_outputs["logits"]
    margins = _margin(logits)
    predicted_classes = logits.argmax(dim=1)
    ablator = ResidualPatchAblator(components, grid_size=circuits_artifact["metadata"]["grid_size"])

    raw_results = []
    for circuit in circuits_artifact["circuits"]:
        member_mask = assign_circuit_members(circuit, future_descriptors, dataset_indices)
        member_rows = torch.nonzero(member_mask, as_tuple=False).flatten()
        if member_rows.numel() == 0:
            continue

        member_images = test_outputs["images"][member_rows].to(components.encoder.final_norm.weight.device)
        member_labels = labels[member_rows]
        member_before = logits[member_rows]
        member_after = ablator.ablate(member_images, [tuple(node) for node in circuit["active_nodes"]]).cpu()
        member_delta_margin = _margin(member_before) - _margin(member_after)

        control_rows = _matched_nonmembers(
            member_rows=member_rows,
            member_classes=predicted_classes[member_rows],
            member_margins=margins[member_rows],
            predicted_classes=predicted_classes,
            margins=margins,
        )
        if control_rows.numel() == 0:
            control_rows = member_rows[: min(2, member_rows.numel())]
        control_images = test_outputs["images"][control_rows].to(member_images.device)
        control_before = logits[control_rows]
        control_after = ablator.ablate(control_images, [tuple(node) for node in circuit["active_nodes"]]).cpu()
        control_delta_margin = _margin(control_before) - _margin(control_after)

        random_nodes = _random_nodes_like(
            [tuple(node) for node in circuit["active_nodes"]],
            circuits_artifact["metadata"]["n_layers"],
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

        p_nonmember = _permutation_pvalue(member_delta_margin.numpy(), control_delta_margin.numpy())
        p_random = _permutation_pvalue(member_delta_margin.numpy(), random_delta_margin.numpy())
        raw_results.append(
            {
                "circuit_id": int(circuit["id"]),
                "n_members": int(member_rows.numel()),
                "n_controls": int(control_rows.numel()),
                "mean_member_delta_margin": float(member_delta_margin.mean().item()),
                "mean_nonmember_delta_margin": float(control_delta_margin.mean().item()),
                "mean_random_node_delta_margin": float(random_delta_margin.mean().item()),
                "mean_random_cell_delta_margin": float(random_cell_delta_margin.mean().item()),
                "p_member_vs_nonmember": float(p_nonmember),
                "p_member_vs_random_node": float(p_random),
            }
        )

    corrected_nonmember = _holm([item["p_member_vs_nonmember"] for item in raw_results])
    corrected_random = _holm([item["p_member_vs_random_node"] for item in raw_results])
    results = []
    for idx, item in enumerate(raw_results):
        validated = corrected_nonmember[idx] < alpha and corrected_random[idx] < alpha
        result = InterventionResult(
            circuit_id=item["circuit_id"],
            n_members=item["n_members"],
            n_controls=item["n_controls"],
            mean_member_delta_margin=item["mean_member_delta_margin"],
            mean_nonmember_delta_margin=item["mean_nonmember_delta_margin"],
            mean_random_node_delta_margin=item["mean_random_node_delta_margin"],
            mean_random_cell_delta_margin=item["mean_random_cell_delta_margin"],
            p_member_vs_nonmember=item["p_member_vs_nonmember"],
            p_member_vs_random_node=item["p_member_vs_random_node"],
            corrected_p_member_vs_nonmember=float(corrected_nonmember[idx]),
            corrected_p_member_vs_random_node=float(corrected_random[idx]),
            validated=validated,
        )
        results.append(result)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump([result.to_dict() for result in results], handle, indent=2)
    return results


def _margin(logits: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(logits, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


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


def _random_nodes_like(nodes: list[tuple[int, int]], n_layers: int, n_cells: int, seed: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    return [(rng.integers(0, n_layers), rng.integers(0, n_cells)) for _ in nodes]


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


def _permutation_pvalue(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2:
        return 1.0
    statistic = lambda a, b, axis: np.mean(a, axis=axis) - np.mean(b, axis=axis)
    result = permutation_test((left, right), statistic, n_resamples=500, alternative="greater")
    return float(result.pvalue)


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
