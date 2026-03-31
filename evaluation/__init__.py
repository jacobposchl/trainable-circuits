from evaluation.circuit_analysis import CircuitAnalyzer, load_checkpoint
from evaluation.circuit_viz import (
    plot_per_layer_umap,
    plot_span_coverage,
    plot_span_heatmap,
)
from evaluation.discovery import SpanCentricDiscovery
from evaluation.interventions import (
    build_control_prototypes,
    build_circuit_library,
    build_circuit_prototype,
    collect_probe_features,
    compute_circuit_score,
    fit_linear_probe,
    fit_linear_probe_from_features,
    forward_ctls_with_grad,
    run_intervention_batch,
    select_circuit_set,
    select_intervention_images,
    summarize_intervention_results,
)
from evaluation.metrics import (
    profile_reconstruction_r2,
    geometric_consistency,
    within_span_elevation,
    circuit_diversity,
    class_purity_distribution,
)
from evaluation.trajectory_viz import (
    animate_trajectory,
    collect_raw_activations,
    get_softmax_probs,
    precompute_circuit_flow,
)

__all__ = [
    "CircuitAnalyzer",
    "load_checkpoint",
    "SpanCentricDiscovery",
    "plot_per_layer_umap",
    "plot_span_coverage",
    "plot_span_heatmap",
    "collect_probe_features",
    "fit_linear_probe",
    "fit_linear_probe_from_features",
    "forward_ctls_with_grad",
    "build_circuit_prototype",
    "build_circuit_library",
    "build_control_prototypes",
    "compute_circuit_score",
    "select_circuit_set",
    "select_intervention_images",
    "run_intervention_batch",
    "summarize_intervention_results",
    "profile_reconstruction_r2",
    "geometric_consistency",
    "within_span_elevation",
    "circuit_diversity",
    "class_purity_distribution",
    "animate_trajectory",
    "collect_raw_activations",
    "get_softmax_probs",
    "precompute_circuit_flow",
]
