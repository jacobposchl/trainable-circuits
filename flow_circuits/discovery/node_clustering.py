from __future__ import annotations

import hdbscan
import numpy as np
from sklearn.decomposition import PCA


def prepare_descriptors(descriptors: np.ndarray, *, random_seed: int) -> np.ndarray:
    """Reduce descriptor dimensionality once before repeated clustering."""
    effective = descriptors
    n_samples, n_dims = descriptors.shape
    if n_dims > 32 and n_samples > 100:
        n_components = min(32, n_samples - 1, n_dims)
        reduced = PCA(n_components=n_components, random_state=random_seed).fit_transform(descriptors)
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        effective = reduced / np.clip(norms, 1.0e-8, None)
    return effective


def cluster_prepared_descriptors(descriptors: np.ndarray, *, min_cluster_size: int) -> np.ndarray:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(int(min_cluster_size), 2),
        min_samples=None,
        metric="euclidean",
    )
    return clusterer.fit_predict(descriptors)


def cluster_descriptors(
    descriptors: np.ndarray,
    *,
    min_cluster_size: int,
    random_seed: int,
) -> np.ndarray:
    prepared = prepare_descriptors(descriptors, random_seed=random_seed)
    return cluster_prepared_descriptors(prepared, min_cluster_size=min_cluster_size)


def bootstrap_cluster_stability(
    descriptors: np.ndarray,
    members: np.ndarray,
    *,
    bootstrap_iterations: int,
    rng: np.random.Generator,
    min_cluster_size: int,
) -> float:
    base_members = set(members.tolist())
    scores = []
    for _ in range(int(bootstrap_iterations)):
        bootstrap_rows = rng.integers(0, descriptors.shape[0], size=descriptors.shape[0])
        bootstrap_labels = cluster_prepared_descriptors(
            descriptors[bootstrap_rows],
            min_cluster_size=min_cluster_size,
        )
        best = 0.0
        for cluster_id in set(bootstrap_labels.tolist()):
            if cluster_id == -1:
                continue
            cluster_rows = set(np.unique(bootstrap_rows[bootstrap_labels == cluster_id]).tolist())
            score = jaccard(base_members, cluster_rows)
            if score > best:
                best = score
        scores.append(best)
    return float(np.mean(scores)) if scores else 0.0


def discover_node_clusters(
    descriptor_grid: np.ndarray,
    dataset_indices: np.ndarray,
    *,
    min_cluster_fraction: float,
    max_cluster_fraction: float,
    min_cluster_size: int,
    bootstrap_iterations: int,
    stability_threshold: float,
    random_seed: int,
    progress_callback=None,
    node_subset: list[list[int]] | list[tuple[int, int]] | None = None,
) -> list[dict]:
    n_images, n_layers, n_cells, _ = descriptor_grid.shape
    n_min = max(int(min_cluster_size), int(np.ceil(min_cluster_fraction * n_images)))
    n_max = int(np.floor(max_cluster_fraction * n_images))
    if node_subset is None:
        nodes_to_scan = [(layer_idx, cell_idx) for layer_idx in range(n_layers) for cell_idx in range(n_cells)]
    else:
        nodes_to_scan = [(int(layer_idx), int(cell_idx)) for layer_idx, cell_idx in node_subset]

    node_clusters = []
    rng = np.random.default_rng(random_seed)
    total_nodes = len(nodes_to_scan)
    for completed_nodes, (layer_idx, cell_idx) in enumerate(nodes_to_scan, start=1):
        descriptors = descriptor_grid[:, layer_idx, cell_idx, :]
        prepared_descriptors = prepare_descriptors(descriptors, random_seed=random_seed)
        labels = cluster_prepared_descriptors(prepared_descriptors, min_cluster_size=min_cluster_size)
        for cluster_id in sorted(set(labels.tolist())):
            if cluster_id == -1:
                continue
            members = np.flatnonzero(labels == cluster_id)
            if not (n_min <= members.size <= n_max):
                continue
            stability = bootstrap_cluster_stability(
                prepared_descriptors,
                members,
                bootstrap_iterations=bootstrap_iterations,
                rng=rng,
                min_cluster_size=min_cluster_size,
            )
            if stability < stability_threshold:
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
        if progress_callback is not None:
            progress_callback(
                stage="node_clustering",
                completed=completed_nodes,
                total=total_nodes,
                layer_idx=int(layer_idx),
                cell_idx=int(cell_idx),
                n_node_clusters=len(node_clusters),
            )
    return node_clusters


def jaccard(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)
