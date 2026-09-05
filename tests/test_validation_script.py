"""
Exercises validate_splits.py as a subprocess (the way it's actually run in
the pipeline) to confirm:
  - it exits 0 on the current, valid project state;
  - it exits non-zero when a core primary invariant is violated.

The simulated-failure case temporarily corrupts a DERIVED manifest (never a
frozen data/source_audit/ file) and restores it in a finally block, so the
project is left exactly as it was regardless of test outcome.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import project_config as cfg

PROJECT_ROOT = cfg.get_project_root()
VALIDATE_SCRIPT = PROJECT_ROOT / "src" / "validate_splits.py"
TARGET_MANIFEST = cfg.get_manifests_dir() / "patient_grouped_split_deduplicated.csv"


def _run_validate():
    env = os.environ.copy()
    return subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT)],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def test_validate_splits_passes_on_clean_project():
    result = _run_validate()
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def test_validate_splits_fails_on_simulated_core_violation():
    backup = TARGET_MANIFEST.read_text(encoding="utf-8")
    try:
        lines = backup.splitlines()
        header, first_row = lines[0], lines[1]
        corrupted = "\n".join([header, first_row, first_row] + lines[2:])
        TARGET_MANIFEST.write_text(corrupted, encoding="utf-8")

        result = _run_validate()
        assert result.returncode != 0, "validate_splits.py should fail on a duplicated relative_path row"
    finally:
        TARGET_MANIFEST.write_text(backup, encoding="utf-8")
        # sanity: file restored exactly
        assert TARGET_MANIFEST.read_text(encoding="utf-8") == backup
