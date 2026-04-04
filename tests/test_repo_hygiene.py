from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
EXPECTED_NOTEBOOKS = {
    "nb01_training_and_representation_metrics.ipynb",
    "nb02_efficient_representation_and_circuit_validation.ipynb",
    "nb03_recurring_motif_core_validation.ipynb",
    "nb04_motif_extended_characterization.ipynb",
    "nb05_motif_visual_interpretability_and_probe_analysis.ipynb",
    "nb06_hard_pair_correction_from_z.ipynb",
}
LEGACY_NOTEBOOKS = {
    "nb01_training_and_validation.ipynb",
    "nb02_analysis.ipynb",
    "nb02_candidate_circuit_discovery_and_stability.ipynb",
    "nb03_causal_interventions.ipynb",
    "nb03_interventions_and_qualitative_analysis.ipynb",
    "nb03_trajectory_animation.ipynb",
}
LEGACY_PATTERNS = (
    re.compile(r"^from models\b", re.MULTILINE),
    re.compile(r"^from losses\b", re.MULTILINE),
    re.compile(r"^from training\b", re.MULTILINE),
    re.compile(r"^from evaluation\b", re.MULTILINE),
    re.compile(r"^from data\b", re.MULTILINE),
    re.compile(r"\bctls-"),
    re.compile(r"\bInfoLoss\b"),
    re.compile(r"\bSpanCentricDiscovery\b"),
    re.compile(r"\bPhase1Trainer\b"),
)
TEXT_EXTENSIONS = {".md", ".py", ".toml", ".yaml", ".yml", ".txt", ".ipynb"}
SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".mypy_cache",
}


def _iter_repo_text_files():
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(REPO_ROOT)
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in path.parts):
            continue
        if relative.parts[:1] == ("tests",):
            continue
        if path.suffix in TEXT_EXTENSIONS:
            yield path


def test_repo_contains_only_new_notebook_suite():
    notebook_names = {path.name for path in NOTEBOOK_DIR.glob("*.ipynb")}
    assert notebook_names == EXPECTED_NOTEBOOKS
    assert not any((NOTEBOOK_DIR / name).exists() for name in LEGACY_NOTEBOOKS)


def test_notebooks_have_expected_structure_and_compilable_code_cells():
    required_config_names = {
        "nb01_training_and_representation_metrics.ipynb": {
            "TRAINING_MODE",
            "CONFIG_NAME",
            "CHECKPOINT_PATH",
            "OUTPUT_DIR",
        },
        "nb02_efficient_representation_and_circuit_validation.ipynb": {
            "RUN_MODE",
            "CONFIG_NAME",
            "EXPERIMENTS",
            "PHASE_B_CHECKPOINT",
            "PHASE_C_CHECKPOINT",
            "OUTPUT_DIR",
            "FORCE_RERUN",
        },
        "nb03_recurring_motif_core_validation.ipynb": {
            "RUN_MODE",
            "CONFIG_NAME",
            "EXPERIMENTS",
            "PHASE_B_CHECKPOINT",
            "PHASE_C_CHECKPOINT",
            "OUTPUT_DIR",
            "FORCE_RERUN",
        },
        "nb04_motif_extended_characterization.ipynb": {
            "RUN_MODE",
            "CONFIG_NAME",
            "EXPERIMENTS",
            "PHASE_B_CHECKPOINT",
            "PHASE_C_CHECKPOINT",
            "NB03_OUTPUT_DIR",
            "OUTPUT_DIR",
            "FORCE_RERUN",
        },
        "nb05_motif_visual_interpretability_and_probe_analysis.ipynb": {
            "RUN_MODE",
            "CONFIG_NAME",
            "EXPERIMENTS",
            "PHASE_B_CHECKPOINT",
            "PHASE_C_CHECKPOINT",
            "NB03_OUTPUT_DIR",
            "NB04_OUTPUT_DIR",
            "OUTPUT_DIR",
            "FORCE_RERUN",
        },
        "nb06_hard_pair_correction_from_z.ipynb": {
            "RUN_MODE",
            "CONFIG_NAME",
            "EXPERIMENTS",
            "PHASE_C_CHECKPOINT",
            "OUTPUT_DIR",
            "FORCE_RERUN",
        },
    }
    for notebook_name in EXPECTED_NOTEBOOKS:
        path = NOTEBOOK_DIR / notebook_name
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["cells"]) >= 5
        assert data["cells"][0]["cell_type"] == "markdown"
        assert data["cells"][1]["cell_type"] == "code"
        assert data["cells"][2]["cell_type"] == "code"
        assert data["cells"][3]["cell_type"] == "markdown"
        assert data["cells"][4]["cell_type"] == "code"

        setup_source = "".join(data["cells"][1]["source"])
        assert "REPO_URL =" in setup_source, f"{notebook_name} missing GitHub bootstrap setup"
        assert "drive.mount(" in setup_source, f"{notebook_name} missing Drive mount setup"

        imports_source = "".join(data["cells"][2]["source"]).strip().splitlines()
        assert imports_source, f"{notebook_name} import cell should import flow_circuits symbols"
        for line in imports_source:
            stripped = line.strip()
            if not stripped or stripped in {"(", ")"}:
                continue
            if stripped.startswith("from ") or stripped.startswith("import "):
                assert stripped.startswith("from flow_circuits.") or stripped.startswith("import flow_circuits"), (
                    f"{notebook_name} import cell must import only flow_circuits symbols: {stripped}"
                )

        config_source = "".join(data["cells"][4]["source"])
        for name in required_config_names[notebook_name]:
            assert re.search(rf"^{name}\s*=", config_source, re.MULTILINE), f"{notebook_name} missing {name} in config cell"

        for index, cell in enumerate(data["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            compile(source, f"{notebook_name}:cell{index}", "exec")


def test_nb01_phase_c_schedule_matches_documented_modes():
    data = json.loads((NOTEBOOK_DIR / "nb01_training_and_representation_metrics.ipynb").read_text(encoding="utf-8"))
    mode_cell_source = "".join(data["cells"][5]["source"])

    assert "import sys" in mode_cell_source
    assert "'phase_c': 1" in mode_cell_source
    assert "'phase_c': 5" in mode_cell_source
    assert "effective_phase_epochs['phase_c'] = 0" in mode_cell_source


def test_nb02_requires_both_checkpoints_and_never_retrains():
    data = json.loads((NOTEBOOK_DIR / "nb02_efficient_representation_and_circuit_validation.ipynb").read_text(encoding="utf-8"))
    all_code = "\n".join("".join(cell["source"]) for cell in data["cells"] if cell["cell_type"] == "code")

    assert "phase_b.pt" in all_code
    assert "phase_c.pt" in all_code
    assert "Missing required checkpoint" in all_code
    assert "flow-train" not in all_code
    assert "flow-discover" not in all_code
    assert "flow-intervene" not in all_code


def test_nb03_requires_both_checkpoints_and_uses_motif_package_apis():
    data = json.loads((NOTEBOOK_DIR / "nb03_recurring_motif_core_validation.ipynb").read_text(encoding="utf-8"))
    all_code = "\n".join("".join(cell["source"]) for cell in data["cells"] if cell["cell_type"] == "code")

    assert "phase_b.pt" in all_code
    assert "phase_c.pt" in all_code
    assert "Missing required checkpoint" in all_code
    assert "discover_motif_families" in all_code
    assert "run_motif_gallery_experiment" in all_code
    assert "run_motif_persistence_experiment" in all_code
    assert "run_motif_predictiveness_experiment" in all_code
    assert "run_motif_intervention_experiment" in all_code
    assert "flow-train" not in all_code


def test_nb04_requires_nb03_artifacts_and_never_retrains():
    data = json.loads((NOTEBOOK_DIR / "nb04_motif_extended_characterization.ipynb").read_text(encoding="utf-8"))
    all_code = "\n".join("".join(cell["source"]) for cell in data["cells"] if cell["cell_type"] == "code")

    assert "phase_b.pt" in all_code
    assert "phase_c.pt" in all_code
    assert "Missing required motif artifact from nb03" in all_code
    assert "run_motif_cooccurrence_experiment" in all_code
    assert "run_motif_phase_match_experiment" in all_code
    assert "run_motif_topology_experiment" in all_code
    assert "run_motif_transfer_stability_experiment" in all_code
    assert "discover_motif_families" not in all_code
    assert "flow-train" not in all_code


def test_nb05_requires_nb03_artifacts_and_uses_interpretability_package_apis():
    data = json.loads((NOTEBOOK_DIR / "nb05_motif_visual_interpretability_and_probe_analysis.ipynb").read_text(encoding="utf-8"))
    all_code = "\n".join("".join(cell["source"]) for cell in data["cells"] if cell["cell_type"] == "code")

    assert "phase_b.pt" in all_code
    assert "phase_c.pt" in all_code
    assert "Missing required motif artifact from nb03" in all_code
    assert "run_motif_visual_report_experiment" in all_code
    assert "run_phase_motif_comparison_experiment" in all_code
    assert "run_motif_case_study_experiment" in all_code
    assert "run_linear_probe_suite_experiment" in all_code
    assert "run_probe_confusion_analysis_experiment" in all_code
    assert "run_probe_error_analysis_experiment" in all_code
    assert "collect_interpretability_outputs" in all_code
    assert "discover_motif_families" not in all_code
    assert "flow-train" not in all_code


def test_nb06_requires_phase_c_checkpoint_and_uses_selective_hard_pair_package_apis():
    data = json.loads((NOTEBOOK_DIR / "nb06_hard_pair_correction_from_z.ipynb").read_text(encoding="utf-8"))
    all_code = "\n".join("".join(cell["source"]) for cell in data["cells"] if cell["cell_type"] == "code")

    assert "phase_c.pt" in all_code
    assert "Missing required checkpoint" in all_code
    assert "run_hard_example_audit_experiment" in all_code
    assert "run_selective_hybrid_correction_experiment" in all_code
    assert "run_confidence_and_calibration_experiment" in all_code
    assert "run_hard_pair_case_study_experiment" in all_code
    assert "collect_interpretability_outputs" in all_code
    assert "flow-train" not in all_code


def test_nb02_exposes_exact_experiment_selector_and_cache_controls():
    data = json.loads((NOTEBOOK_DIR / "nb02_efficient_representation_and_circuit_validation.ipynb").read_text(encoding="utf-8"))
    config_source = "".join(data["cells"][4]["source"])
    helper_source = "".join(data["cells"][5]["source"])
    all_code = "\n".join("".join(cell["source"]) for cell in data["cells"] if cell["cell_type"] == "code")

    assert 'EXPERIMENTS = "all"' in config_source
    for experiment_id in (
        "neighbor_agreement",
        "activation_probe",
        "discovery_pilot",
        "topk_interventions",
    ):
        assert experiment_id in all_code
    assert "FORCE_RERUN = False" in config_source
    assert "nb02_efficient_validation" in all_code
    assert "_cache_path" in helper_source


def test_nb03_and_nb04_expose_exact_experiment_selectors_and_cache_controls():
    expectations = {
        "nb03_recurring_motif_core_validation.ipynb": (
            "CORE_MOTIF_EXPERIMENT_IDS",
            "nb03_recurring_motif_core_validation",
            (
                "motif_families",
                "motif_galleries",
                "motif_persistence",
                "motif_predictiveness",
                "motif_interventions",
            ),
        ),
        "nb04_motif_extended_characterization.ipynb": (
            "EXTENDED_MOTIF_EXPERIMENT_IDS",
            "nb04_motif_extended_characterization",
            (
                "motif_cooccurrence_graph",
                "motif_phase_match",
                "motif_topology",
                "motif_transfer_stability",
            ),
        ),
    }
    for notebook_name, (experiment_constant, output_dir_name, experiment_ids) in expectations.items():
        data = json.loads((NOTEBOOK_DIR / notebook_name).read_text(encoding="utf-8"))
        config_source = "".join(data["cells"][4]["source"])
        helper_source = "".join(data["cells"][5]["source"])
        all_code = "\n".join("".join(cell["source"]) for cell in data["cells"] if cell["cell_type"] == "code")

        assert 'EXPERIMENTS = "all"' in config_source
        assert "FORCE_RERUN = False" in config_source
        assert experiment_constant in all_code
        assert output_dir_name in all_code
        assert "_cache_path" in helper_source
        for experiment_id in experiment_ids:
            assert experiment_id in all_code


def test_nb05_exposes_exact_experiment_selector_and_cache_controls():
    data = json.loads((NOTEBOOK_DIR / "nb05_motif_visual_interpretability_and_probe_analysis.ipynb").read_text(encoding="utf-8"))
    config_source = "".join(data["cells"][4]["source"])
    helper_source = "".join(data["cells"][5]["source"])
    all_code = "\n".join("".join(cell["source"]) for cell in data["cells"] if cell["cell_type"] == "code")

    assert 'EXPERIMENTS = "all"' in config_source
    assert "FORCE_RERUN = False" in config_source
    assert "NB05_EXPERIMENT_IDS" in all_code
    assert "nb05_motif_visual_interpretability_and_probe_analysis" in all_code
    assert "_cache_path" in helper_source
    for experiment_id in (
        "motif_reports",
        "phase_comparison",
        "intervention_cases",
        "class_probe_suite",
        "confusion_analysis",
        "error_analysis",
    ):
        assert experiment_id in all_code


def test_nb06_exposes_exact_experiment_selector_and_cache_controls():
    data = json.loads((NOTEBOOK_DIR / "nb06_hard_pair_correction_from_z.ipynb").read_text(encoding="utf-8"))
    config_source = "".join(data["cells"][4]["source"])
    helper_source = "".join(data["cells"][5]["source"])
    all_code = "\n".join("".join(cell["source"]) for cell in data["cells"] if cell["cell_type"] == "code")

    assert 'EXPERIMENTS = "all"' in config_source
    assert "FORCE_RERUN = False" in config_source
    assert "NB06_EXPERIMENT_IDS" in all_code
    assert "nb06_hard_pair_correction_from_z" in all_code
    assert "_run_or_load" in helper_source
    for experiment_id in (
        "hard_example_audit",
        "selective_hybrid_correction",
        "confidence_and_calibration",
        "correction_case_studies",
    ):
        assert experiment_id in all_code


def test_nb02_uses_package_apis_and_progress_callbacks_not_heartbeats():
    data = json.loads((NOTEBOOK_DIR / "nb02_efficient_representation_and_circuit_validation.ipynb").read_text(encoding="utf-8"))
    all_code = "\n".join("".join(cell["source"]) for cell in data["cells"] if cell["cell_type"] == "code")

    assert "run_neighbor_agreement_experiment" in all_code
    assert "run_activation_probe_experiment" in all_code
    assert "run_discovery_pilot_experiment" in all_code
    assert "run_topk_intervention_experiment" in all_code
    assert "_progress_logger" in all_code
    assert "still running... elapsed" not in all_code
    assert "import queue" not in all_code
    assert "import threading" not in all_code


def test_repo_text_has_no_legacy_runtime_references():
    offenders: list[str] = []
    for path in _iter_repo_text_files():
        content = path.read_text(encoding="utf-8")
        for pattern in LEGACY_PATTERNS:
            if pattern.search(content):
                offenders.append(f"{path.relative_to(REPO_ROOT)} -> {pattern.pattern}")
    assert offenders == []
