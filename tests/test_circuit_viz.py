import sys
import types

import matplotlib.pyplot as plt
import numpy as np

from evaluation.circuit_viz import (
    plot_circuit_members,
    plot_multi_circuit_histogram,
    plot_per_layer_umap,
    plot_span_coverage,
    plot_span_heatmap,
)


def test_plot_span_coverage_draws_one_bar_per_circuit():
    circuits = [{"span": (0, 1), "size": 4}, {"span": (2, 3), "size": 2}]
    fig = plot_span_coverage(circuits, n_layers=4)
    try:
        assert len(fig.axes) == 1
        assert len(fig.axes[0].patches) == 2
    finally:
        plt.close(fig)


def test_plot_multi_circuit_histogram_returns_figure():
    fig = plot_multi_circuit_histogram(np.array([0, 1, 1, 2, 3]))
    try:
        assert len(fig.axes) == 1
        assert len(fig.axes[0].patches) > 0
    finally:
        plt.close(fig)


def test_plot_span_heatmap_aggregates_metric_by_mean():
    circuits = [
        {"span": (0, 1), "elevation_sigma": 1.0},
        {"span": (0, 1), "elevation_sigma": 3.0},
        {"span": (1, 2), "elevation_sigma": 2.0},
    ]
    fig = plot_span_heatmap(circuits, n_layers=3, agg="mean")
    try:
        heatmap = np.asarray(fig.axes[0].images[0].get_array())
        assert heatmap[0, 1] == 2.0
        assert heatmap[1, 2] == 2.0
    finally:
        plt.close(fig)


def test_plot_span_heatmap_aggregates_metric_by_max():
    circuits = [
        {"span": (0, 0), "purity": 0.2},
        {"span": (0, 0), "purity": 0.9},
    ]
    fig = plot_span_heatmap(circuits, n_layers=2, metric="purity", agg="max")
    try:
        heatmap = np.asarray(fig.axes[0].images[0].get_array())
        assert heatmap[0, 0] == 0.9
    finally:
        plt.close(fig)


def test_plot_circuit_members_handles_small_batches():
    images = np.random.rand(3, 3, 32, 32).astype(np.float32)
    labels = np.array([0, 1, 2])
    profiles = np.random.rand(3, 4).astype(np.float32)
    fig = plot_circuit_members(images, labels, profiles, span=(1, 2), n_show=6)
    try:
        assert len(fig.axes) >= 4
    finally:
        plt.close(fig)


def test_plot_per_layer_umap_uses_requested_layers(monkeypatch):
    class FakeUMAP:
        def __init__(self, n_components, random_state, n_neighbors):
            self.n_components = n_components
            self.random_state = random_state
            self.n_neighbors = n_neighbors

        def fit_transform(self, x):
            return x[:, :2]

    monkeypatch.setitem(sys.modules, "umap", types.SimpleNamespace(UMAP=FakeUMAP))

    z_list = [np.random.randn(20, 4).astype(np.float32) for _ in range(3)]
    labels = np.arange(20) % 10
    fig = plot_per_layer_umap(z_list, labels, layer_indices=[0, 2], max_samples=20)
    try:
        assert len(fig.axes) == 2
        assert fig.axes[0].get_title() == "Layer 1"
        assert fig.axes[1].get_title() == "Layer 3"
    finally:
        plt.close(fig)
