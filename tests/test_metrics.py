import numpy as np
import pytest
from scipy.stats import ConstantInputWarning

from evaluation.metrics import (
    circuit_diversity,
    class_purity_distribution,
    geometric_consistency,
    profile_reconstruction_r2,
    rich_profile_reconstruction_r2,
    within_span_elevation,
)


def test_profile_reconstruction_r2_is_one_for_perfect_prediction():
    true = np.array([[0.2, 0.4], [0.6, 0.8]])
    result = profile_reconstruction_r2(true, true)
    assert result["r2"] == 1.0
    assert bool(result["passes"]) is True


def test_profile_reconstruction_r2_handles_zero_variance_targets():
    predicted = np.zeros((3, 2))
    true = np.ones((3, 2))
    result = profile_reconstruction_r2(predicted, true)
    assert np.isfinite(result["r2"])
    assert bool(result["passes"]) is False


def test_rich_profile_reconstruction_reports_per_layer_scores():
    true = [np.ones((4, 2)), np.full((4, 2), 2.0)]
    pred = [true[0].copy(), true[1] * 0.0]
    result = rich_profile_reconstruction_r2(pred, true)
    assert len(result["per_layer_r2"]) == 2
    assert result["per_layer_r2"][0] == 1.0
    assert bool(result["passes"]) is False


def test_geometric_consistency_passes_for_monotonic_layerwise_rankings():
    base = np.arange(10, dtype=float)
    z_sims = np.stack([base, base + 1.0, base + 2.0], axis=1)
    true_sims = np.stack([base * 2.0, base * 3.0, base * 4.0], axis=1)
    result = geometric_consistency(z_sims, true_sims, n_layers=3)
    assert all(abs(rho - 1.0) < 1e-6 for rho in result["per_layer_rho"])
    assert result["passes"] is True


def test_geometric_consistency_replaces_nan_correlations_with_zero():
    z_sims = np.ones((6, 2))
    true_sims = np.ones((6, 2))
    with pytest.warns(ConstantInputWarning):
        result = geometric_consistency(z_sims, true_sims, n_layers=2)
    assert result["per_layer_rho"] == [0.0, 0.0]
    assert result["passes"] is False


def test_within_span_elevation_passes_when_cluster_exceeds_population_by_one_std():
    cluster = np.array([0.8, 0.9, 1.0])
    population = np.array([0.2, 0.3, 0.4, 0.5])
    result = within_span_elevation(cluster, population)
    assert result["elevation_sigma"] > 1.0
    assert result["passes"] is True


def test_within_span_elevation_handles_zero_std_population():
    cluster = np.array([0.5, 0.5])
    population = np.array([0.5, 0.5, 0.5])
    result = within_span_elevation(cluster, population)
    assert result["elevation_sigma"] == 0.0
    assert result["passes"] is False


def test_circuit_diversity_reports_coverage_and_count():
    result = circuit_diversity([(0, 1), (3, 4)], total_layers=5)
    assert result["coverage"] == 0.8
    assert result["covered_layers"] == {0, 1, 3, 4}
    assert result["n_circuits"] == 2
    assert result["passes"] is True


def test_circuit_diversity_handles_empty_input():
    result = circuit_diversity([], total_layers=4)
    assert result["coverage"] == 0.0
    assert result["covered_layers"] == set()
    assert result["passes"] is False


def test_class_purity_distribution_detects_bimodality():
    result = class_purity_distribution([0.1, 0.2, 0.8, 0.95, 0.5])
    assert result["n_agnostic"] == 2
    assert result["n_specific"] == 2
    assert result["n_middle"] == 1
    assert result["passes"] is True


def test_class_purity_distribution_fails_without_both_modes():
    result = class_purity_distribution([0.4, 0.5, 0.6])
    assert result["n_agnostic"] == 0
    assert result["n_specific"] == 0
    assert result["passes"] is False
