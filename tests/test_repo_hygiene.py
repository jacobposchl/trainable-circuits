from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
EXPECTED_NOTEBOOKS = {
    "nb01_training_and_representation_metrics.ipynb",
    "nb02_candidate_circuit_discovery_and_stability.ipynb",
    "nb03_interventions_and_qualitative_analysis.ipynb",
}
LEGACY_NOTEBOOKS = {
    "nb01_training_and_validation.ipynb",
    "nb02_analysis.ipynb",
    "nb03_causal_interventions.ipynb",
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
        "TRAINING_MODE",
        "CONFIG_NAME",
        "CHECKPOINT_PATH",
        "OUTPUT_DIR",
    }
    for notebook_name in EXPECTED_NOTEBOOKS:
        path = NOTEBOOK_DIR / notebook_name
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["cells"]) >= 5
        assert data["cells"][0]["cell_type"] == "markdown"
        assert data["cells"][1]["cell_type"] == "code"
        assert data["cells"][2]["cell_type"] == "code"
        # cell 3 is now a markdown config-explanation cell
        assert data["cells"][3]["cell_type"] == "markdown"
        # cell 4 is the config code cell
        assert data["cells"][4]["cell_type"] == "code"

        setup_source = "".join(data["cells"][1]["source"])
        assert "REPO_URL =" in setup_source, f"{notebook_name} missing GitHub bootstrap setup"
        assert "drive.mount(" in setup_source, f"{notebook_name} missing Drive mount setup"

        imports_source = "".join(data["cells"][2]["source"]).strip().splitlines()
        assert imports_source, f"{notebook_name} second cell should import flow_circuits symbols"
        for line in imports_source:
            stripped = line.strip()
            assert stripped.startswith("from flow_circuits.") or stripped.startswith("import flow_circuits"), (
                f"{notebook_name} import cell must import only flow_circuits symbols: {stripped}"
            )

        config_source = "".join(data["cells"][4]["source"])
        for name in required_config_names:
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


def test_nb02_and_nb03_phase_c_schedule_matches_nb01_behavior():
    for notebook_name in (
        "nb02_candidate_circuit_discovery_and_stability.ipynb",
        "nb03_interventions_and_qualitative_analysis.ipynb",
    ):
        data = json.loads((NOTEBOOK_DIR / notebook_name).read_text(encoding="utf-8"))
        mode_cell_source = "".join(data["cells"][5]["source"])

        assert "import sys" in mode_cell_source
        assert "phase_b.pt" in mode_cell_source
        assert "phase_c.pt" in mode_cell_source
        assert "flow-train" not in mode_cell_source


def test_notebooks_auto_resume_from_phase_b_checkpoint():
    data = json.loads((NOTEBOOK_DIR / "nb01_training_and_representation_metrics.ipynb").read_text(encoding="utf-8"))
    run_cell_source = "".join(data["cells"][7]["source"])

    assert "RESUME_CHECKPOINT = PHASE_B_CHECKPOINT" in run_cell_source
    assert "'--resume', str(RESUME_CHECKPOINT)" in run_cell_source


def test_nb02_and_nb03_never_retrain_flow_model():
    for notebook_name in (
        "nb02_candidate_circuit_discovery_and_stability.ipynb",
        "nb03_interventions_and_qualitative_analysis.ipynb",
    ):
        data = json.loads((NOTEBOOK_DIR / notebook_name).read_text(encoding="utf-8"))
        mode_cell_source = "".join(data["cells"][5]["source"])
        run_cell_source = "".join(data["cells"][7]["source"])

        assert "flow-train" not in mode_cell_source
        assert "flow-train" not in run_cell_source
        assert "requires pre-trained phase_b.pt and phase_c.pt checkpoints" in run_cell_source


def test_notebooks_expose_checkpoint_paths_for_phase_b_and_phase_c():
    data = json.loads((NOTEBOOK_DIR / "nb01_training_and_representation_metrics.ipynb").read_text(encoding="utf-8"))
    config_cell_source = "".join(data["cells"][5]["source"])

    assert "PHASE_C_CHECKPOINT" in config_cell_source
    assert "print(f\"Phase C ckpt:" in config_cell_source

    for notebook_name in (
        "nb02_candidate_circuit_discovery_and_stability.ipynb",
        "nb03_interventions_and_qualitative_analysis.ipynb",
    ):
        data = json.loads((NOTEBOOK_DIR / notebook_name).read_text(encoding="utf-8"))
        config_cell_source = "".join(data["cells"][5]["source"])

        assert "phase_b.pt" in config_cell_source
        assert "phase_c.pt" in config_cell_source
        assert "MODEL_ORDER = [('phase_b', 'Phase B'), ('phase_c', 'Phase C')]" in config_cell_source


def test_nb02_and_nb03_compare_phase_b_and_phase_c_outputs():
    for notebook_name in (
        "nb02_candidate_circuit_discovery_and_stability.ipynb",
        "nb03_interventions_and_qualitative_analysis.ipynb",
    ):
        data = json.loads((NOTEBOOK_DIR / notebook_name).read_text(encoding="utf-8"))
        config_cell_source = "".join(data["cells"][5]["source"])
        run_cell_source = "".join(data["cells"][7]["source"])

        assert "MODEL_ORDER = [('phase_b', 'Phase B'), ('phase_c', 'Phase C')]" in config_cell_source
        assert "for tag, label in MODEL_ORDER:" in run_cell_source


def test_nb02_and_nb03_use_cli_progress_not_notebook_heartbeats():
    for notebook_name in (
        "nb02_candidate_circuit_discovery_and_stability.ipynb",
        "nb03_interventions_and_qualitative_analysis.ipynb",
    ):
        data = json.loads((NOTEBOOK_DIR / notebook_name).read_text(encoding="utf-8"))
        config_cell_source = "".join(data["cells"][5]["source"])

        assert "still running... elapsed" not in config_cell_source
        assert "import queue" not in config_cell_source
        assert "import threading" not in config_cell_source


def test_repo_text_has_no_legacy_runtime_references():
    offenders: list[str] = []
    for path in _iter_repo_text_files():
        content = path.read_text(encoding="utf-8")
        for pattern in LEGACY_PATTERNS:
            if pattern.search(content):
                offenders.append(f"{path.relative_to(REPO_ROOT)} -> {pattern.pattern}")
    assert offenders == []
