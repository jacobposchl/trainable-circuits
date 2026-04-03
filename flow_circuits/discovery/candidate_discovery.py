from __future__ import annotations

from collections import defaultdict, deque
import json
from pathlib import Path

import hdbscan
import numpy as np

from flow_circuits.evaluation import bootstrap_mean_ci


class CandidateCircuitDiscoverer:
    def __init__(
        self,
        *,
        grid_size: int = 4,
        min_cluster_fraction: float = 0.005,
        max_cluster_fraction: float = 0.40,
        min_cluster_size: int = 20,
        bootstrap_iterations: int = 20,
        stability_threshold: float = 0.60,
        merge_threshold: float = 0.70,
        node_threshold: float = 0.70,
        random_seed: int = 0,
    ) -> None:
        self.grid_size = grid_size
        self.min_cluster_fraction = min_cluster_fraction
        self.max_cluster_fraction = max_cluster_fraction
        self.min_cluster_size = min_cluster_size
        self.bootstrap_iterations = bootstrap_iterations
        self.stability_threshold = stability_threshold
        self.merge_threshold = merge_threshold
        self.node_threshold = node_threshold
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def discover(
        self,
        *,
        future_descriptors: np.ndarray,
        predicted_next: np.ndarray,
        flow_targets: np.ndarray,
        dataset_indices: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> dict:
        n_images, n_layers, n_cells, _ = future_descriptors.shape
        node_clusters = self._discover_node_clusters(future_descriptors, dataset_indices)
        circuits = self._merge_node_clusters(
            node_clusters=node_clusters,
            future_descriptors=future_descriptors,
            predicted_next=predicted_next,
            flow_targets=flow_targets,
            dataset_indices=dataset_indices,
            labels=labels,
        )
        artifact = {
            "metadata": {
                "n_images": int(n_images),
                "n_layers": int(n_layers),
                "n_cells": int(n_cells),
                "grid_size": int(self.grid_size),
                "random_seed": int(self.random_seed),
            },
            "node_clusters": node_clusters,
            "circuits": circuits,
        }
        return artifact

    def save(self, artifact: dict, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2)

    def _discover_node_clusters(self, future_descriptors: np.ndarray, dataset_indices: np.ndarray) -> list[dict]:
        n_images, n_layers, n_cells, _ = future_descriptors.shape
        node_clusters = []
        n_min = max(self.min_cluster_size, int(np.ceil(self.min_cluster_fraction * n_images)))
        n_max = int(np.floor(self.max_cluster_fraction * n_images))
        for layer_idx in range(n_layers):
            for cell_idx in range(n_cells):
                descriptors = future_descriptors[:, layer_idx, cell_idx, :]
                labels = self._cluster_descriptors(descriptors)
                for cluster_id in sorted(set(labels.tolist())):
                    if cluster_id == -1:
                        continue
                    members = np.flatnonzero(labels == cluster_id)
                    if not (n_min <= members.size <= n_max):
                        continue
                    stability = self._bootstrap_stability(descriptors, members)
                    if stability < self.stability_threshold:
                        continue
                    node_clusters.append(
                        {
                            "node": [int(layer_idx), int(cell_idx)],
                            "image_set": dataset_indices[members].tolist(),
                            "row_indices": members.tolist(),
                            "size": int(members.size),
                            "stability": float(stability),
                        }
                    )
        return node_clusters

    def _cluster_descriptors(self, descriptors: np.ndarray) -> np.ndarray:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(self.min_cluster_size, 2),
            min_samples=None,
            metric="euclidean",
        )
        return clusterer.fit_predict(descriptors)

    def _bootstrap_stability(self, descriptors: np.ndarray, members: np.ndarray) -> float:
        base_members = set(members.tolist())
        scores = []
        for _ in range(self.bootstrap_iterations):
            bootstrap_rows = self.rng.integers(0, descriptors.shape[0], size=descriptors.shape[0])
            bootstrap_labels = self._cluster_descriptors(descriptors[bootstrap_rows])
            best = 0.0
            for cluster_id in set(bootstrap_labels.tolist()):
                if cluster_id == -1:
                    continue
                cluster_rows = set(np.unique(bootstrap_rows[bootstrap_labels == cluster_id]).tolist())
                score = _jaccard(base_members, cluster_rows)
                if score > best:
                    best = score
            scores.append(best)
        return float(np.mean(scores)) if scores else 0.0

    def _merge_node_clusters(
        self,
        *,
        node_clusters: list[dict],
        future_descriptors: np.ndarray,
        predicted_next: np.ndarray,
        flow_targets: np.ndarray,
        dataset_indices: np.ndarray,
        labels: np.ndarray | None,
    ) -> list[dict]:
        if not node_clusters:
            return []

        cluster_sets = [set(cluster["image_set"]) for cluster in node_clusters]
        adjacency = defaultdict(set)
        for left in range(len(node_clusters)):
            for right in range(left + 1, len(node_clusters)):
                if _jaccard(cluster_sets[left], cluster_sets[right]) >= self.merge_threshold:
                    adjacency[left].add(right)
                    adjacency[right].add(left)

        components = []
        seen = set()
        for idx in range(len(node_clusters)):
            if idx in seen:
                continue
            queue = deque([idx])
            component = []
            seen.add(idx)
            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in adjacency[current]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
            components.append(component)

        engagement = (predicted_next * flow_targets[:, 1:]).sum(axis=-1)
        mu = engagement.mean(axis=0)
        sigma = engagement.std(axis=0)

        circuits = []
        for circuit_id, component in enumerate(components):
            family_clusters = [node_clusters[idx] for idx in component]
            medoid_idx = max(
                component,
                key=lambda idx: np.mean([
                    _jaccard(cluster_sets[idx], cluster_sets[other]) for other in component
                ]),
            )
            canonical_set = set(node_clusters[medoid_idx]["image_set"])
            active_nodes = []
            for cluster in family_clusters:
                if _jaccard(set(cluster["image_set"]), canonical_set) >= self.node_threshold:
                    active_nodes.append(tuple(cluster["node"]))
            active_nodes = self._largest_connected_component(active_nodes)
            if len(active_nodes) < 3 or len({layer for layer, _ in active_nodes}) < 2:
                continue

            row_lookup = {int(dataset_idx): row_idx for row_idx, dataset_idx in enumerate(dataset_indices.tolist())}
            member_rows = np.array([row_lookup[idx] for idx in canonical_set if idx in row_lookup], dtype=int)
            if member_rows.size == 0:
                continue

            engagement_profile = {}
            engaged_nodes = []
            checkable_count = 0
            for layer_idx, cell_idx in active_nodes:
                if layer_idx >= engagement.shape[1]:
                    continue
                checkable_count += 1
                score = float(engagement[member_rows, layer_idx, cell_idx].mean())
                threshold = float(mu[layer_idx, cell_idx] + sigma[layer_idx, cell_idx])
                engagement_profile[f"{layer_idx}:{cell_idx}"] = score
                if score >= threshold:
                    engaged_nodes.append((layer_idx, cell_idx))
            if len(engaged_nodes) < max(1, checkable_count // 2):
                continue
            if not self._has_depth_path(engaged_nodes):
                continue

            centroids = {}
            thresholds = {}
            for layer_idx, cell_idx in active_nodes:
                vectors = future_descriptors[member_rows, layer_idx, cell_idx, :]
                centroid = vectors.mean(axis=0)
                centroid = centroid / np.clip(np.linalg.norm(centroid), 1.0e-8, None)
                similarities = vectors @ centroid
                centroids[f"{layer_idx}:{cell_idx}"] = centroid.tolist()
                thresholds[f"{layer_idx}:{cell_idx}"] = float(np.percentile(similarities, 10.0))

            purity = None
            if labels is not None:
                member_labels = labels[member_rows]
                counts = np.bincount(member_labels.astype(int))
                purity = float(counts.max() / max(member_labels.size, 1))

            circuits.append(
                {
                    "id": int(circuit_id),
                    "image_set": sorted(int(idx) for idx in canonical_set),
                    "representative_node": list(node_clusters[medoid_idx]["node"]),
                    "active_nodes": [list(node) for node in active_nodes],
                    "engagement_profile": engagement_profile,
                    "centroids": centroids,
                    "thresholds": thresholds,
                    "stability": {
                        "mean_cluster_stability": float(np.mean([cluster["stability"] for cluster in family_clusters])),
                        "n_node_clusters": int(len(family_clusters)),
                    },
                    "purity": purity,
                }
            )
        return circuits

    def _largest_connected_component(self, active_nodes: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not active_nodes:
            return []
        node_set = set(active_nodes)
        components = []
        seen = set()
        for node in active_nodes:
            if node in seen:
                continue
            queue = deque([node])
            current_component = []
            seen.add(node)
            while queue:
                current = queue.popleft()
                current_component.append(current)
                for neighbor in self._neighbors(current):
                    if neighbor in node_set and neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
            components.append(current_component)
        return max(components, key=len)

    def _neighbors(self, node: tuple[int, int]) -> list[tuple[int, int]]:
        layer_idx, cell_idx = node
        row = cell_idx // self.grid_size
        col = cell_idx % self.grid_size
        neighbors = []
        if layer_idx > 0:
            neighbors.append((layer_idx - 1, cell_idx))
        neighbors.append((layer_idx + 1, cell_idx))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr = row + dr
            nc = col + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                neighbors.append((layer_idx, (nr * self.grid_size) + nc))
        return neighbors

    @staticmethod
    def _has_depth_path(nodes: list[tuple[int, int]]) -> bool:
        layers_by_cell: dict[int, list[int]] = defaultdict(list)
        for layer_idx, cell_idx in nodes:
            layers_by_cell[cell_idx].append(layer_idx)
        for layers in layers_by_cell.values():
            ordered = sorted(set(layers))
            required_run_length = 3 if len(ordered) >= 3 else 2
            run_length = 1
            for left, right in zip(ordered, ordered[1:]):
                if right == left + 1:
                    run_length += 1
                    if run_length >= required_run_length:
                        return True
                else:
                    run_length = 1
        return False


def _jaccard(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def summarize_seed_stability(
    seed_runs: list[dict],
    *,
    bootstrap_iterations: int = 500,
    seed: int = 0,
) -> dict:
    if not seed_runs:
        return {"n_seed_runs": 0, "per_circuit": []}

    reference = seed_runs[0]
    comparison_runs = seed_runs[1:]
    rng = np.random.default_rng(seed)
    per_circuit = []
    for circuit in reference["circuits"]:
        observed_image_jaccards = []
        observed_active_f1 = []
        null_image_jaccards = []
        null_active_f1 = []
        for run in comparison_runs:
            match = _match_circuit(circuit, run["circuits"])
            if match is None:
                continue
            observed_image_jaccards.append(_jaccard(set(circuit["image_set"]), set(match["image_set"])))
            observed_active_f1.append(_node_f1(circuit["active_nodes"], match["active_nodes"]))

            null_candidate = _matched_null_circuit(circuit, run["circuits"], rng=rng)
            if null_candidate is not None:
                null_image_jaccards.append(_jaccard(set(circuit["image_set"]), set(null_candidate["image_set"])))
                null_active_f1.append(_node_f1(circuit["active_nodes"], null_candidate["active_nodes"]))

        n_observed = min(len(observed_image_jaccards), len(null_image_jaccards))
        image_diff = (
            np.asarray(observed_image_jaccards[:n_observed], dtype=np.float64)
            - np.asarray(null_image_jaccards[:n_observed], dtype=np.float64)
        )
        active_diff = (
            np.asarray(observed_active_f1[:n_observed], dtype=np.float64)
            - np.asarray(null_active_f1[:n_observed], dtype=np.float64)
        )
        image_ci = bootstrap_mean_ci(image_diff, n_bootstrap=bootstrap_iterations, seed=seed) if image_diff.size else (0.0, 0.0)
        active_ci = bootstrap_mean_ci(active_diff, n_bootstrap=bootstrap_iterations, seed=seed + 1) if active_diff.size else (0.0, 0.0)
        per_circuit.append(
            {
                "circuit_id": int(circuit["id"]),
                "n_matches": int(len(observed_image_jaccards)),
                "mean_image_jaccard": float(np.mean(observed_image_jaccards)) if observed_image_jaccards else 0.0,
                "mean_active_node_f1": float(np.mean(observed_active_f1)) if observed_active_f1 else 0.0,
                "mean_null_image_jaccard": float(np.mean(null_image_jaccards)) if null_image_jaccards else 0.0,
                "mean_null_active_node_f1": float(np.mean(null_active_f1)) if null_active_f1 else 0.0,
                "image_jaccard_improvement_ci": [float(image_ci[0]), float(image_ci[1])],
                "active_node_f1_improvement_ci": [float(active_ci[0]), float(active_ci[1])],
                "stable": bool(image_ci[0] > 0.0 and active_ci[0] > 0.0),
            }
        )
    return {
        "n_seed_runs": int(len(seed_runs)),
        "reference_seed": int(reference["seed"]),
        "per_circuit": per_circuit,
    }


def run_node_shuffle_null(
    *,
    future_descriptors: np.ndarray,
    predicted_next: np.ndarray,
    flow_targets: np.ndarray,
    dataset_indices: np.ndarray,
    labels: np.ndarray | None,
    discoverer_kwargs: dict,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    n_layers = future_descriptors.shape[1]
    n_cells = future_descriptors.shape[2]
    permutation = rng.permutation(n_layers * n_cells)
    shuffled = future_descriptors.reshape(future_descriptors.shape[0], n_layers * n_cells, future_descriptors.shape[-1])
    shuffled = shuffled[:, permutation, :].reshape(future_descriptors.shape)
    artifact = CandidateCircuitDiscoverer(**discoverer_kwargs, random_seed=seed).discover(
        future_descriptors=shuffled,
        predicted_next=predicted_next,
        flow_targets=flow_targets,
        dataset_indices=dataset_indices,
        labels=labels,
    )
    return {
        "seed": int(seed),
        "n_node_clusters": int(len(artifact["node_clusters"])),
        "n_circuits": int(len(artifact["circuits"])),
    }


def _match_circuit(reference_circuit: dict, candidate_circuits: list[dict]) -> dict | None:
    representative_node = reference_circuit["representative_node"]
    matching_candidates = [
        candidate
        for candidate in candidate_circuits
        if candidate["representative_node"] == representative_node
    ]
    if not matching_candidates:
        return None
    rep_key = f"{representative_node[0]}:{representative_node[1]}"
    reference_centroid = np.asarray(reference_circuit["centroids"][rep_key], dtype=np.float64)
    return max(
        matching_candidates,
        key=lambda candidate: float(
            np.dot(reference_centroid, np.asarray(candidate["centroids"].get(rep_key, reference_centroid), dtype=np.float64))
        ),
    )


def _matched_null_circuit(reference_circuit: dict, candidate_circuits: list[dict], *, rng: np.random.Generator) -> dict | None:
    if not candidate_circuits:
        return None
    reference_node_count = len(reference_circuit["active_nodes"])
    reference_image_count = len(reference_circuit["image_set"])
    ordered = sorted(
        candidate_circuits,
        key=lambda candidate: (
            abs(len(candidate["active_nodes"]) - reference_node_count),
            abs(len(candidate["image_set"]) - reference_image_count),
            candidate["id"],
        ),
    )
    shortlist = ordered[: max(1, min(3, len(ordered)))]
    return shortlist[int(rng.integers(0, len(shortlist)))]


def _node_f1(left_nodes: list[list[int]], right_nodes: list[list[int]]) -> float:
    left_set = {tuple(node) for node in left_nodes}
    right_set = {tuple(node) for node in right_nodes}
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    true_positive = len(left_set & right_set)
    precision = true_positive / len(right_set)
    recall = true_positive / len(left_set)
    if precision + recall == 0.0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)
