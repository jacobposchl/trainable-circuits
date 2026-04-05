from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
EXPECTED_NOTEBOOKS = {
    "nb01_backbone_and_z_training.ipynb",
    "nb02_q_validation.ipynb",
    "nb03_z_motif_discovery_and_analysis.ipynb",
    "nb04_motif_utility_and_robustness.ipynb",
    "nb05_motif_semantic_interpretation.ipynb",
}
LEGACY_NOTEBOOKS = {
    "nb01_training_and_representation_metrics.ipynb",
    "nb02_efficient_representation_and_circuit_validation.ipynb",
    "nb03_recurring_motif_core_validation.ipynb",
    "nb04_motif_extended_characterization.ipynb",
    "nb05_motif_visual_interpretability_and_probe_analysis.ipynb",
    "nb06_hard_pair_correction_from_z.ipynb",
}
TEXT_EXTENSIONS = {".md", ".py", ".toml", ".yaml", ".yml", ".txt", ".ipynb"}
SKIP_DIRS = {".git", ".pytest_cache", "__pycache__", ".mypy_cache"}


def _iter_repo_text_files():
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in path.parts):
            continue
        if path.suffix in TEXT_EXTENSIONS:
            yield path


def test_repo_contains_only_five_notebook_suite():
    notebook_names = {path.name for path in NOTEBOOK_DIR.glob("*.ipynb")}
    assert notebook_names == EXPECTED_NOTEBOOKS
    assert not any((NOTEBOOK_DIR / name).exists() for name in LEGACY_NOTEBOOKS)


def test_notebooks_have_bootstrap_setup_config_and_compilable_code():
    required_config_names = {
        "nb01_backbone_and_z_training.ipynb": {
            "BACKBONE_EPOCHS",
            "PHASE_A_EPOCHS",
            "PHASE_B_EPOCHS",
            "PHASE_C_MAX_EPOCHS",
            "PHASE_C_MILESTONES",
            "LAMBDA_TRAJ_CANDIDATES",
            "JOINT_BRANCH_ENABLED",
            "JOINT_BACKBONE_LR_MULTIPLIER",
            "JOINT_CE_WEIGHT",
            "OUTPUT_DIR",
            "FORCE_RERUN",
        },
        "nb02_q_validation.ipynb": {
            "FROZEN_CHECKPOINT_DIR",
            "JOINT_CHECKPOINT_DIR",
            "Q_VALIDATION_SPLIT",
            "TOP_K_CANDIDATES_TO_SUMMARIZE",
            "OUTPUT_DIR",
            "FORCE_RERUN",
        },
        "nb03_z_motif_discovery_and_analysis.ipynb": {
            "DISCOVERY_MAX_IMAGES",
            "BOOTSTRAP_ITERATIONS",
            "MIN_MOTIF_LAYERS",
            "MIN_CLUSTER_SIZE",
            "MOTIF_OVERLAP_THRESHOLD",
            "TOP_MOTIFS_TO_RENDER",
            "OUTPUT_DIR",
            "FORCE_RERUN",
        },
        "nb04_motif_utility_and_robustness.ipynb": {
            "HARD_PAIRS_TOP_K",
            "TRIGGER_MODE",
            "MARGIN_QUANTILE",
            "TOP_MOTIF_FRACTION",
            "MIN_TOP_MOTIFS",
            "MAX_TOP_MOTIFS",
            "CORRUPTION_NAMES",
            "CORRUPTION_SEVERITIES",
            "FROZEN_MOTIF_ARTIFACT",
            "JOINT_MOTIF_ARTIFACT",
            "OUTPUT_DIR",
            "FORCE_RERUN",
        },
        "nb05_motif_semantic_interpretation.ipynb": {
            "PRIMARY_BRANCH",
            "REFERENCE_BRANCH",
            "TOP_MOTIFS_TO_RENDER",
            "MOTIF_IDS",
            "MAX_IMAGES",
            "SHOW_REFERENCE_COMPARISON",
            "OUTPUT_DIR",
            "FORCE_RERUN",
        },
    }
    for notebook_name in EXPECTED_NOTEBOOKS:
        data = json.loads((NOTEBOOK_DIR / notebook_name).read_text(encoding="utf-8"))
        assert len(data["cells"]) >= 6
        assert data["cells"][0]["cell_type"] == "markdown"
        setup_source = "".join(data["cells"][1]["source"])
        assert "REPO_URL =" in setup_source
        assert "drive.mount(" in setup_source

        imports_source = "".join(data["cells"][2]["source"])
        assert "flow_circuits" in imports_source

        config_source = "".join(data["cells"][4]["source"])
        for name in required_config_names[notebook_name]:
            assert re.search(rf"^{name}\s*=", config_source, re.MULTILINE), f"{notebook_name} missing {name}"

        for index, cell in enumerate(data["cells"]):
            if cell["cell_type"] != "code":
                continue
            compile("".join(cell["source"]), f"{notebook_name}:cell{index}", "exec")


def test_notebook_api_contract_matches_new_workflow():
    nb01 = "\n".join("".join(cell["source"]) for cell in json.loads((NOTEBOOK_DIR / "nb01_backbone_and_z_training.ipynb").read_text(encoding="utf-8"))["cells"] if cell["cell_type"] == "code")
    nb02 = "\n".join("".join(cell["source"]) for cell in json.loads((NOTEBOOK_DIR / "nb02_q_validation.ipynb").read_text(encoding="utf-8"))["cells"] if cell["cell_type"] == "code")
    nb03 = "\n".join("".join(cell["source"]) for cell in json.loads((NOTEBOOK_DIR / "nb03_z_motif_discovery_and_analysis.ipynb").read_text(encoding="utf-8"))["cells"] if cell["cell_type"] == "code")
    nb04 = "\n".join("".join(cell["source"]) for cell in json.loads((NOTEBOOK_DIR / "nb04_motif_utility_and_robustness.ipynb").read_text(encoding="utf-8"))["cells"] if cell["cell_type"] == "code")
    nb05 = "\n".join("".join(cell["source"]) for cell in json.loads((NOTEBOOK_DIR / "nb05_motif_semantic_interpretation.ipynb").read_text(encoding="utf-8"))["cells"] if cell["cell_type"] == "code")

    assert "run_backbone_and_z_training_workflow" in nb01
    assert "run_q_checkpoint_validation_experiment" in nb02
    assert "discover_motif_families" in nb03
    assert "use_all_nodes=True" in nb03
    assert "run_motif_clean_utility_experiment" in nb04
    assert "run_motif_corruption_utility_experiment" in nb04
    assert "run_motif_semantic_report_experiment" in nb05
    assert "run_motif_spatial_footprint_experiment" in nb05
    assert "run_motif_borderline_member_experiment" in nb05
    assert "discover_motif_families" not in nb05
    assert "run_q_checkpoint_validation_experiment" not in nb05


def test_only_nb02_mentions_q_validation_api():
    notebook_sources = {
        notebook_name: "\n".join(
            "".join(cell["source"])
            for cell in json.loads((NOTEBOOK_DIR / notebook_name).read_text(encoding="utf-8"))["cells"]
            if cell["cell_type"] == "code"
        )
        for notebook_name in EXPECTED_NOTEBOOKS
    }
    assert "run_q_checkpoint_validation_experiment" in notebook_sources["nb02_q_validation.ipynb"]
    for notebook_name in EXPECTED_NOTEBOOKS - {"nb02_q_validation.ipynb"}:
        assert "run_q_checkpoint_validation_experiment" not in notebook_sources[notebook_name]


def test_repo_docs_reference_new_notebook_suite():
    docs = [
        REPO_ROOT / "notebooks" / "README.md",
        REPO_ROOT / "documents" / "experiment_guide.md",
        REPO_ROOT / "documents" / "repo_structure.md",
    ]
    for path in docs:
        text = path.read_text(encoding="utf-8")
        for notebook_name in EXPECTED_NOTEBOOKS:
            assert notebook_name in text, f"{path.name} missing {notebook_name}"
        for notebook_name in LEGACY_NOTEBOOKS:
            assert notebook_name not in text, f"{path.name} still references {notebook_name}"


def test_no_legacy_notebook_names_remain_in_repo_text():
    legacy_names = tuple(sorted(LEGACY_NOTEBOOKS))
    for path in _iter_repo_text_files():
        if path.name == "test_repo_hygiene.py":
            continue
        text = path.read_text(encoding="utf-8")
        for legacy in legacy_names:
            assert legacy not in text, f"Legacy notebook name {legacy} still found in {path}"
