from __future__ import annotations

from collections import defaultdict, deque
import json
from pathlib import Path

import hdbscan
import numpy as np


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
            for layer_idx, cell_idx in active_nodes:
                if layer_idx >= engagement.shape[1]:
                    continue
                score = float(engagement[member_rows, layer_idx, cell_idx].mean())
                threshold = float(mu[layer_idx, cell_idx] + sigma[layer_idx, cell_idx])
                engagement_profile[f"{layer_idx}:{cell_idx}"] = score
                if score >= threshold:
                    engaged_nodes.append((layer_idx, cell_idx))
            if len(engaged_nodes) < max(1, len(active_nodes) // 2):
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
        node_set = set(nodes)
        for layer_idx, cell_idx in nodes:
            if (layer_idx + 1, cell_idx) in node_set:
                return True
        return False


def _jaccard(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)
