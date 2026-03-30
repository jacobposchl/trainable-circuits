"""
Unit tests for image-centric span-centric circuit discovery.
Run with: pytest tests/
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from evaluation.discovery import SpanCentricDiscovery


D = 16   # small projection_dim for testing
L = 8    # layers
N = 50   # images


def make_z_list(N=N, L=L, d=D):
    """Random L2-normalized per-layer z-vectors."""
    z_list = []
    for _ in range(L):
        z = np.random.randn(N, d).astype(np.float32)
        z = z / np.linalg.norm(z, axis=1, keepdims=True)
        z_list.append(z)
    return z_list


def make_discovery(**kwargs):
    defaults = dict(n_layers=L)
    defaults.update(kwargs)
    return SpanCentricDiscovery(**defaults)


# --------------------------------------------------------------------------- #
# Span Enumeration
# --------------------------------------------------------------------------- #

class TestSpanEnumeration:
    def test_count_for_8_layers(self):
        disc = make_discovery()
        spans = disc.enumerate_spans()
        assert len(spans) == 36  # L(L+1)/2

    def test_count_for_4_layers(self):
        disc = SpanCentricDiscovery(n_layers=4)
        spans = disc.enumerate_spans()
        assert len(spans) == 10  # 4*5/2

    def test_spans_are_valid(self):
        disc = make_discovery()
        for l_start, l_end in disc.enumerate_spans():
            assert 0 <= l_start <= l_end < L

    def test_single_layer_spans_included(self):
        disc = make_discovery()
        spans = disc.enumerate_spans()
        for l in range(L):
            assert (l, l) in spans

    def test_full_range_span_included(self):
        disc = make_discovery()
        assert (0, L - 1) in disc.enumerate_spans()


# --------------------------------------------------------------------------- #
# Span Embedding
# --------------------------------------------------------------------------- #

class TestEmbedSpan:
    def test_single_layer_shape(self):
        disc = make_discovery()
        z_list = make_z_list()
        X = disc._embed_span(z_list, (2, 2))
        assert X.shape == (N, D)

    def test_multi_layer_shape(self):
        disc = make_discovery()
        z_list = make_z_list()
        X = disc._embed_span(z_list, (1, 4))   # 4 layers
        assert X.shape == (N, 4 * D)

    def test_full_span_shape(self):
        disc = make_discovery()
        z_list = make_z_list()
        X = disc._embed_span(z_list, (0, L - 1))
        assert X.shape == (N, L * D)

    def test_values_match_concatenation(self):
        disc = make_discovery()
        z_list = make_z_list()
        X = disc._embed_span(z_list, (0, 1))
        expected = np.concatenate([z_list[0], z_list[1]], axis=1)
        np.testing.assert_array_equal(X, expected)


# --------------------------------------------------------------------------- #
# Canonicality Filter
# --------------------------------------------------------------------------- #

class TestCanonicalityFilter:
    def test_filters_small_clusters(self):
        disc = make_discovery(min_cluster_fraction=0.1)
        labels = np.array([0]*50 + [1]*5 + [-1]*45)
        canonical = disc.filter_canonical(labels, n_total=100)
        assert 0 in canonical
        assert 1 not in canonical  # 5/100 = 0.05 < 0.1

    def test_filters_large_clusters(self):
        disc = make_discovery(min_cluster_fraction=0.01, max_cluster_fraction=0.30)
        labels = np.array([0]*40 + [1]*20 + [-1]*40)
        canonical = disc.filter_canonical(labels, n_total=100)
        assert 0 not in canonical  # 40% > 30%
        assert 1 in canonical      # 20%, within window

    def test_noise_excluded(self):
        disc = make_discovery()
        labels = np.array([-1]*100)
        assert len(disc.filter_canonical(labels, n_total=100)) == 0

    def test_boundary_inclusive(self):
        disc = make_discovery(min_cluster_fraction=0.10, max_cluster_fraction=0.40)
        labels = np.array([0]*10 + [1]*40 + [-1]*50)
        canonical = disc.filter_canonical(labels, n_total=100)
        assert 0 in canonical   # exactly at min
        assert 1 in canonical   # exactly at max


# --------------------------------------------------------------------------- #
# Span Similarities
# --------------------------------------------------------------------------- #

class TestSpanSimilarities:
    def test_output_shape_full(self):
        disc = make_discovery()
        z_list = make_z_list()
        sims = disc.compute_span_similarities(z_list, (0, 2))
        n_pairs = N * (N - 1) // 2
        assert sims.shape == (n_pairs,)

    def test_output_shape_masked(self):
        disc = make_discovery()
        z_list = make_z_list()
        mask = np.zeros(N, dtype=bool)
        mask[:10] = True
        sims = disc.compute_span_similarities(z_list, (0, 2), image_mask=mask)
        n_pairs = 10 * 9 // 2
        assert sims.shape == (n_pairs,)

    def test_values_in_range(self):
        disc = make_discovery()
        z_list = make_z_list()
        sims = disc.compute_span_similarities(z_list, (0, 1))
        assert sims.min() >= -1.01
        assert sims.max() <= 1.01


# --------------------------------------------------------------------------- #
# Class Purity
# --------------------------------------------------------------------------- #

class TestClassPurity:
    def test_pure_circuit(self):
        circuit = {"image_mask": np.array([True]*10 + [False]*10)}
        labels  = np.array([3]*10 + [5]*10)
        assert SpanCentricDiscovery.compute_class_purity(circuit, labels) == 1.0

    def test_uniform_circuit(self):
        circuit = {"image_mask": np.array([True]*10)}
        labels  = np.arange(10)   # 10 different classes, 1 each
        purity  = SpanCentricDiscovery.compute_class_purity(circuit, labels)
        assert abs(purity - 0.1) < 1e-6

    def test_empty_circuit(self):
        circuit = {"image_mask": np.zeros(10, dtype=bool)}
        labels  = np.zeros(10, dtype=int)
        assert SpanCentricDiscovery.compute_class_purity(circuit, labels) == 0.0


# --------------------------------------------------------------------------- #
# Multi-Circuit Membership
# --------------------------------------------------------------------------- #

class TestMultiCircuitMembership:
    def test_no_circuits_gives_zeros(self):
        counts = SpanCentricDiscovery.multi_circuit_membership([], n_images=10)
        assert counts.shape == (10,)
        assert counts.sum() == 0

    def test_counts_accumulate(self):
        n = 20
        c1 = {"image_mask": np.array([True]*10 + [False]*10)}
        c2 = {"image_mask": np.array([False]*5 + [True]*10 + [False]*5)}
        counts = SpanCentricDiscovery.multi_circuit_membership([c1, c2], n_images=n)
        assert counts[7] == 2    # in both
        assert counts[0] == 1    # only c1
        assert counts[15] == 0   # neither
