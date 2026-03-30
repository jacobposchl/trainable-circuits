"""
Span-centric circuit discovery pipeline.

Enumerates all candidate contiguous spans [l_start, l_end], and for each span
independently clusters input pairs by their within-span similarity structure.
This allows a single pair to participate in multiple circuits at different
depth ranges, which is the correct behavior for multi-circuit inputs.

Key steps per span:
  1. Extract the span sub-vector from each pair's raw profile
  2. Cluster with HDBSCAN on raw (un-normalized) sub-vectors
  3. Filter by canonicality criterion: size in (min_cluster_fraction, max_cluster_fraction)

Raw sub-vectors (no softmax sharpening) are used so that the discovered clusters
have elevated absolute similarity — the property the within-span elevation test
(Criterion 3) checks.  Softmax sharpening would cluster by profile shape while
discarding magnitude, making the elevation test impossible to pass.
"""
from __future__ import annotations

from collections import defaultdict

import hdbscan
import numpy as np
import torch


class SpanCentricDiscovery:
    """
    Discover canonical circuits via span-centric clustering of alignment
    profiles.
    """

    def __init__(
        self,
        n_layers: int,
        min_cluster_fraction: float = 0.01,
        max_cluster_fraction: float = 0.40,
        min_cluster_size: int = 5,
        max_pairs: int = 100_000,
    ):
        """
        Args:
            n_layers:             Number of backbone layers (L).
            min_cluster_fraction: Minimum fraction of total pairs for canonicality.
            max_cluster_fraction: Maximum fraction of total pairs — clusters larger
                                  than this are degenerate mega-clusters (nearly the
                                  whole dataset) and are excluded.
            min_cluster_size:     HDBSCAN min_cluster_size parameter.
            max_pairs:            Maximum pairs to cluster per span (subsample
                                  if more, for memory/compute tractability).
        """
        self.n_layers = n_layers
        self.min_cluster_fraction = min_cluster_fraction
        self.max_cluster_fraction = max_cluster_fraction
        self.min_cluster_size = min_cluster_size
        self.max_pairs = max_pairs

    def enumerate_spans(self) -> list[tuple[int, int]]:
        """All L(L+1)/2 candidate contiguous spans [l_start, l_end]."""
        spans = []
        for l_start in range(self.n_layers):
            for l_end in range(l_start, self.n_layers):
                spans.append((l_start, l_end))
        return spans

    @staticmethod
    def extract_span_subvector(
        profiles: np.ndarray, span: tuple[int, int]
    ) -> np.ndarray:
        """
        Extract the contiguous slice of raw profiles for a given span.

        Args:
            profiles: [N_pairs, L] raw alignment profile vectors
            span:     (l_start, l_end) inclusive

        Returns:
            [N_pairs, span_len] sub-vectors
        """
        l_start, l_end = span
        return profiles[:, l_start:l_end + 1]

    def cluster_span(
        self, subvectors: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        """
        Run HDBSCAN on raw sub-vectors for one span.

        Args:
            subvectors: [N_pairs, span_len] raw (un-normalized) similarity values

        Returns:
            labels:       [N_pairs] cluster assignments (-1 = noise)
            cluster_info: dict mapping cluster_id -> {size, ...}
        """
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=None,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(subvectors)

        cluster_info = {}
        for cid in set(labels):
            if cid == -1:
                continue
            mask = labels == cid
            cluster_info[cid] = {"size": int(mask.sum())}

        return labels, cluster_info

    def filter_canonical(
        self, labels: np.ndarray, n_total_pairs: int
    ) -> list[int]:
        """
        Keep only clusters within the [min_cluster_fraction, max_cluster_fraction]
        size window.  Clusters below the minimum are too sparse to be canonical.
        Clusters above the maximum are degenerate mega-clusters that cover most of
        the dataset and cannot produce meaningful within-span elevation.

        Args:
            labels:        [N_pairs] cluster assignments
            n_total_pairs: total number of pairs across all spans

        Returns:
            List of canonical cluster IDs
        """
        min_threshold = self.min_cluster_fraction * n_total_pairs
        max_threshold = self.max_cluster_fraction * n_total_pairs
        canonical = []
        for cid in set(labels):
            if cid == -1:
                continue
            size = (labels == cid).sum()
            if min_threshold <= size <= max_threshold:
                canonical.append(cid)
        return canonical

    def discover_all(
        self,
        profiles: np.ndarray,
        pair_indices: np.ndarray | None = None,
    ) -> list[dict]:
        """
        Full discovery pipeline across all spans.

        Clusters on raw scalar similarity profiles so that discovered clusters
        have genuinely elevated within-span similarity (both high shape-consistency
        AND high absolute level), which is what Criterion 3 tests.

        Args:
            profiles:     [N_pairs, L] raw alignment profile vectors
            pair_indices: [N_pairs, 2] optional array of (idx_a, idx_b) for
                          each pair, used for downstream analysis

        Returns:
            List of canonical circuit dicts, each containing:
              - span: (l_start, l_end)
              - cluster_id: int
              - pair_mask: boolean mask into profiles array
              - size: number of pairs in cluster
              - mean_similarity: mean within-span similarity
              - std_similarity: std within-span similarity
        """
        N = profiles.shape[0]
        spans = self.enumerate_spans()
        circuits = []

        for span in spans:
            # Extract sub-vector for this span
            subvec = self.extract_span_subvector(profiles, span)

            # Skip single-layer spans
            if subvec.shape[1] == 1:
                continue

            # Subsample if too many pairs
            if N > self.max_pairs:
                sample_idx = np.random.choice(N, self.max_pairs, replace=False)
                subvec_sample = subvec[sample_idx]
            else:
                sample_idx = np.arange(N)
                subvec_sample = subvec

            # Cluster on raw sub-vectors (no normalization — preserves magnitude)
            labels, cluster_info = self.cluster_span(subvec_sample)

            # Filter canonical: must be in (min, max) size window
            canonical_ids = self.filter_canonical(labels, N)

            for cid in canonical_ids:
                mask_in_sample = labels == cid
                global_indices = sample_idx[mask_in_sample]

                # Compute mean within-span similarity from raw sub-vector
                cluster_subvec = subvec[global_indices]
                mean_sim = float(cluster_subvec.mean())
                std_sim = float(cluster_subvec.std())

                # Build full mask into profiles array
                full_mask = np.zeros(N, dtype=bool)
                full_mask[global_indices] = True

                circuits.append({
                    "span": span,
                    "cluster_id": cid,
                    "pair_mask": full_mask,
                    "size": int(mask_in_sample.sum()),
                    "mean_similarity": mean_sim,
                    "std_similarity": std_sim,
                })

        return circuits

    @staticmethod
    def compute_class_purity(
        circuit: dict,
        pair_indices: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Compute class purity for a circuit cluster.

        Class purity = fraction of unique inputs in the cluster that share
        the most common class label.

        Args:
            circuit:      circuit dict from discover_all
            pair_indices: [N_pairs, 2] array of input indices per pair
            labels:       [N_inputs] class labels

        Returns:
            Purity score in [0, 1]
        """
        mask = circuit["pair_mask"]
        pair_idx = pair_indices[mask]
        unique_inputs = np.unique(pair_idx.ravel())
        input_labels = labels[unique_inputs]

        if len(input_labels) == 0:
            return 0.0

        counts = np.bincount(input_labels)
        return float(counts.max()) / len(input_labels)

    @staticmethod
    def multi_circuit_membership(
        circuits: list[dict], n_pairs: int
    ) -> np.ndarray:
        """
        Count how many canonical circuits each pair participates in.

        Args:
            circuits: list of circuit dicts from discover_all
            n_pairs:  total number of pairs

        Returns:
            [N_pairs] array of membership counts
        """
        counts = np.zeros(n_pairs, dtype=int)
        for circuit in circuits:
            counts += circuit["pair_mask"].astype(int)
        return counts

    @staticmethod
    def compute_prototypes(
        circuits: list[dict],
        z_list: list[np.ndarray],
        pair_indices: np.ndarray,
    ) -> list[np.ndarray]:
        """
        Compute circuit prototype (centroid of z-vectors) for each circuit.

        For each circuit at span [l_start, l_end], collect z-vectors at
        those layers for all unique inputs in the cluster, and average.

        Args:
            circuits:     list of circuit dicts
            z_list:       list of L arrays, each [N_inputs, d]
            pair_indices: [N_pairs, 2]

        Returns:
            List of prototype arrays, one per circuit
        """
        prototypes = []
        for circuit in circuits:
            l_start, l_end = circuit["span"]
            mask = circuit["pair_mask"]
            unique_inputs = np.unique(pair_indices[mask].ravel())

            # Collect z-vectors for span layers
            span_z = []
            for l in range(l_start, l_end + 1):
                span_z.append(z_list[l][unique_inputs])  # [N_unique, d]

            # Stack and average over inputs and layers
            stacked = np.stack(span_z, axis=0)  # [span_len, N_unique, d]
            prototype = stacked.mean(axis=(0, 1))  # [d]
            prototypes.append(prototype)

        return prototypes
