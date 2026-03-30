"""
Unit tests for span-centric circuit discovery.
Run with: pytest tests/
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from evaluation.discovery import SpanCentricDiscovery


# --------------------------------------------------------------------------- #
# Span Enumeration
# --------------------------------------------------------------------------- #

class TestSpanEnumeration:
    def test_count_for_8_layers(self):
        disc = SpanCentricDiscovery(n_layers=8)
        spans = disc.enumerate_spans()
        # L(L+1)/2 = 8*9/2 = 36
        assert len(spans) == 36

    def test_count_for_4_layers(self):
        disc = SpanCentricDiscovery(n_layers=4)
        spans = disc.enumerate_spans()
        assert len(spans) == 10  # 4*5/2

    def test_spans_are_valid(self):
        disc = SpanCentricDiscovery(n_layers=8)
        spans = disc.enumerate_spans()
        for l_start, l_end in spans:
            assert 0 <= l_start <= l_end < 8

    def test_single_layer_spans_included(self):
        disc = SpanCentricDiscovery(n_layers=4)
        spans = disc.enumerate_spans()
        for l in range(4):
            assert (l, l) in spans

    def test_full_range_span_included(self):
        disc = SpanCentricDiscovery(n_layers=4)
        spans = disc.enumerate_spans()
        assert (0, 3) in spans


# --------------------------------------------------------------------------- #
# Extract Span Sub-Vector
# --------------------------------------------------------------------------- #

class TestExtractSpanSubvector:
    def test_correct_slice(self):
        profiles = np.random.rand(10, 8)  # 10 pairs, 8 layers
        subvec = SpanCentricDiscovery.extract_span_subvector(profiles, (2, 5))
        assert subvec.shape == (10, 4)  # layers 2, 3, 4, 5
        np.testing.assert_array_equal(subvec, profiles[:, 2:6])

    def test_single_layer_span(self):
        profiles = np.random.rand(5, 8)
        subvec = SpanCentricDiscovery.extract_span_subvector(profiles, (3, 3))
        assert subvec.shape == (5, 1)

    def test_full_span(self):
        profiles = np.random.rand(5, 8)
        subvec = SpanCentricDiscovery.extract_span_subvector(profiles, (0, 7))
        assert subvec.shape == (5, 8)
        np.testing.assert_array_equal(subvec, profiles)


# --------------------------------------------------------------------------- #
# Canonicality Filter
# --------------------------------------------------------------------------- #

class TestCanonicalityFilter:
    def test_filters_small_clusters(self):
        disc = SpanCentricDiscovery(n_layers=8, min_cluster_fraction=0.1)
        # 100 pairs: cluster 0 has 50, cluster 1 has 5, noise has 45
        labels = np.array([0]*50 + [1]*5 + [-1]*45)
        canonical = disc.filter_canonical(labels, n_total_pairs=100)
        assert 0 in canonical
        assert 1 not in canonical  # 5/100 = 0.05 < 0.1

    def test_filters_large_clusters(self):
        disc = SpanCentricDiscovery(
            n_layers=8, min_cluster_fraction=0.01, max_cluster_fraction=0.30
        )
        # 100 pairs: cluster 0 has 40 (too large), cluster 1 has 20 (ok)
        labels = np.array([0]*40 + [1]*20 + [-1]*40)
        canonical = disc.filter_canonical(labels, n_total_pairs=100)
        assert 0 not in canonical  # 40/100 = 0.40 > 0.30
        assert 1 in canonical      # 20/100 = 0.20, within window

    def test_noise_excluded(self):
        disc = SpanCentricDiscovery(n_layers=8)
        labels = np.array([-1]*100)
        canonical = disc.filter_canonical(labels, n_total_pairs=100)
        assert len(canonical) == 0

    def test_boundary_inclusive(self):
        disc = SpanCentricDiscovery(
            n_layers=8, min_cluster_fraction=0.10, max_cluster_fraction=0.40
        )
        # cluster 0 exactly at min boundary, cluster 1 exactly at max boundary
        labels = np.array([0]*10 + [1]*40 + [-1]*50)
        canonical = disc.filter_canonical(labels, n_total_pairs=100)
        assert 0 in canonical   # 10/100 = 0.10 == min_threshold
        assert 1 in canonical   # 40/100 = 0.40 == max_threshold


# --------------------------------------------------------------------------- #
# Multi-Circuit Membership
# --------------------------------------------------------------------------- #

class TestMultiCircuitMembership:
    def test_no_circuits_gives_zeros(self):
        counts = SpanCentricDiscovery.multi_circuit_membership([], n_pairs=10)
        assert counts.shape == (10,)
        assert counts.sum() == 0

    def test_counts_accumulate(self):
        n = 20
        c1 = {"pair_mask": np.array([True]*10 + [False]*10)}
        c2 = {"pair_mask": np.array([False]*5 + [True]*10 + [False]*5)}
        counts = SpanCentricDiscovery.multi_circuit_membership([c1, c2], n_pairs=n)
        # Indices 5-9 are in both circuits
        assert counts[7] == 2
        # Index 0 is in only circuit 1
        assert counts[0] == 1
        # Index 15 is in neither
        assert counts[15] == 0
