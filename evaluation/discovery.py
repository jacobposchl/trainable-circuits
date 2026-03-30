"""
Image-centric span-centric circuit discovery pipeline.

For each candidate contiguous span [l_start, l_end], concatenates per-image
z-vectors across the span, runs UMAP for dimensionality reduction, then
HDBSCAN to cluster images that activate the same circuit.

Why image-centric instead of pairwise:
  Pairwise dot products between L2-normalized high-dimensional vectors suffer
  from concentration of measure — all similarities collapse to ~0 regardless
  of whether images share a circuit.  Working directly on per-image z-vectors
  avoids this: UMAP+HDBSCAN finds genuine density clusters in the
  representation space rather than in a derived scalar similarity space.

A circuit is a cluster of IMAGES that activate a contiguous span of layers
in the same direction in z-space.  One image can belong to multiple circuits
(at different spans), which is the correct behavior for multi-circuit inputs.
"""
from __future__ import annotations

import numpy as np
import hdbscan
import umap


class SpanCentricDiscovery:
    """
    Discover canonical circuits via image-centric UMAP + HDBSCAN clustering.
    """

    def __init__(
        self,
        n_layers: int,
        umap_n_components: int = 15,
        umap_n_neighbors: int = 15,
        min_cluster_fraction: float = 0.01,
        max_cluster_fraction: float = 0.40,
        min_cluster_size: int = 5,
    ):
        """
        Args:
            n_layers:             Number of backbone layers (L).
            umap_n_components:    Target dimensionality for UMAP reduction.
            umap_n_neighbors:     n_neighbors parameter for UMAP.
            min_cluster_fraction: Minimum fraction of total images for canonicality.
            max_cluster_fraction: Maximum fraction — clusters larger than this are
                                  degenerate and are excluded.
            min_cluster_size:     HDBSCAN min_cluster_size parameter.
        """
        self.n_layers = n_layers
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.min_cluster_fraction = min_cluster_fraction
        self.max_cluster_fraction = max_cluster_fraction
        self.min_cluster_size = min_cluster_size

    def enumerate_spans(self) -> list[tuple[int, int]]:
        """All L(L+1)/2 candidate contiguous spans [l_start, l_end]."""
        spans = []
        for l_start in range(self.n_layers):
            for l_end in range(l_start, self.n_layers):
                spans.append((l_start, l_end))
        return spans

    def _embed_span(
        self, z_list: list[np.ndarray], span: tuple[int, int]
    ) -> np.ndarray:
        """
        Concatenate z-vectors for each image across span layers.

        Args:
            z_list: list of L arrays, each [N, d]
            span:   (l_start, l_end) inclusive

        Returns:
            [N, span_len * d] feature matrix, one row per image
        """
        l_start, l_end = span
        return np.concatenate(
            [z_list[l] for l in range(l_start, l_end + 1)], axis=1
        )

    def _reduce(self, X: np.ndarray) -> np.ndarray:
        """
        UMAP dimensionality reduction with cosine metric.

        Cosine metric is appropriate because z-vectors are L2-normalized, so
        cosine distance captures directional structure rather than magnitude.
        """
        n_components = min(self.umap_n_components, X.shape[0] - 2)
        n_neighbors  = min(self.umap_n_neighbors,  X.shape[0] - 1)
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            metric="cosine",
            random_state=42,
            low_memory=False,
        )
        return reducer.fit_transform(X)

    def _cluster(self, X: np.ndarray) -> np.ndarray:
        """HDBSCAN clustering on UMAP-reduced features."""
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=None,
            metric="euclidean",
        )
        return clusterer.fit_predict(X)

    def filter_canonical(
        self, labels: np.ndarray, n_total: int
    ) -> list[int]:
        """
        Keep only clusters within the [min_cluster_fraction, max_cluster_fraction]
        size window.

        Args:
            labels:  [N] cluster assignments (-1 = noise)
            n_total: total number of images

        Returns:
            List of canonical cluster IDs
        """
        min_thresh = self.min_cluster_fraction * n_total
        max_thresh = self.max_cluster_fraction * n_total
        canonical = []
        for cid in set(labels):
            if cid == -1:
                continue
            size = int((labels == cid).sum())
            if min_thresh <= size <= max_thresh:
                canonical.append(cid)
        return canonical

    def discover_all(
        self, z_list: list[np.ndarray]
    ) -> list[dict]:
        """
        Full discovery pipeline across all spans.

        Args:
            z_list: list of L np.ndarray, each [N, d] — per-image z-vectors
                    (L2-normalized) from the meta-encoder, one per layer.

        Returns:
            List of canonical circuit dicts, each containing:
              - span:       (l_start, l_end) inclusive
              - cluster_id: int
              - image_mask: [N] boolean mask — True for images in this circuit
              - size:       number of images in cluster
        """
        N = z_list[0].shape[0]
        circuits = []

        for span in self.enumerate_spans():
            # Build per-image feature matrix for this span
            X = self._embed_span(z_list, span)  # [N, span_len * d]

            # UMAP reduction
            try:
                X_reduced = self._reduce(X)
            except Exception:
                continue

            # HDBSCAN clustering
            labels = self._cluster(X_reduced)

            # Filter canonical
            canonical_ids = self.filter_canonical(labels, N)

            for cid in canonical_ids:
                image_mask = labels == cid
                circuits.append({
                    "span":       span,
                    "cluster_id": cid,
                    "image_mask": image_mask,
                    "size":       int(image_mask.sum()),
                })

        return circuits

    def compute_span_similarities(
        self,
        z_list: list[np.ndarray],
        span: tuple[int, int],
        image_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute mean pairwise z-cosine similarity over span layers.

        For each pair of images (optionally restricted to those in image_mask),
        returns the mean dot product across all layers in the span.  Since
        z-vectors are L2-normalized, dot product = cosine similarity.

        Args:
            z_list:     list of L arrays, each [N, d]
            span:       (l_start, l_end) inclusive
            image_mask: optional [N] boolean — restrict to these images

        Returns:
            1D array of per-pair mean cosine similarities
        """
        l_start, l_end = span
        span_len = l_end - l_start + 1

        if image_mask is not None:
            zs = [z_list[l][image_mask] for l in range(l_start, l_end + 1)]
        else:
            zs = [z_list[l] for l in range(l_start, l_end + 1)]

        n = zs[0].shape[0]
        sim_matrix = np.zeros((n, n))
        for z in zs:
            sim_matrix += z @ z.T
        sim_matrix /= span_len

        idx_a, idx_b = np.triu_indices(n, k=1)
        return sim_matrix[idx_a, idx_b]

    @staticmethod
    def compute_class_purity(
        circuit: dict,
        labels: np.ndarray,
    ) -> float:
        """
        Compute class purity for a circuit.

        Purity = fraction of images in the circuit belonging to the most
        common class label.

        Args:
            circuit: circuit dict from discover_all (must have 'image_mask')
            labels:  [N] integer class labels

        Returns:
            Purity score in [0, 1]
        """
        circuit_labels = labels[circuit["image_mask"]]
        if len(circuit_labels) == 0:
            return 0.0
        counts = np.bincount(circuit_labels)
        return float(counts.max()) / len(circuit_labels)

    @staticmethod
    def multi_circuit_membership(
        circuits: list[dict], n_images: int
    ) -> np.ndarray:
        """
        Count how many canonical circuits each image participates in.

        Args:
            circuits: list of circuit dicts from discover_all
            n_images: total number of images

        Returns:
            [N] array of membership counts
        """
        counts = np.zeros(n_images, dtype=int)
        for circuit in circuits:
            counts += circuit["image_mask"].astype(int)
        return counts

    @staticmethod
    def compute_prototypes(
        circuits: list[dict],
        z_list: list[np.ndarray],
    ) -> list[np.ndarray]:
        """
        Compute circuit prototype (centroid of z-vectors) for each circuit.

        For each circuit at span [l_start, l_end], averages z-vectors across
        all images in the cluster and all layers in the span.

        Args:
            circuits: list of circuit dicts
            z_list:   list of L arrays, each [N, d]

        Returns:
            List of prototype arrays, one per circuit, each [d]
        """
        prototypes = []
        for circuit in circuits:
            l_start, l_end = circuit["span"]
            mask = circuit["image_mask"]
            span_z = [z_list[l][mask] for l in range(l_start, l_end + 1)]
            stacked = np.stack(span_z, axis=0)   # [span_len, N_cluster, d]
            prototype = stacked.mean(axis=(0, 1))  # [d]
            prototypes.append(prototype)
        return prototypes
