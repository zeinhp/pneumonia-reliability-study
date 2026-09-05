"""
project_config.py

Single shared source of truth for filesystem paths used across this project.
No absolute session/agent paths are hardcoded anywhere - everything is
derived from this file's own location (pathlib) plus, for the dataset root,
an environment variable with a per-OS fallback read from
config/dataset_config.yaml.

Import this module from any script in src/ or tests/ as follows:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import project_config as cfg

    cfg.PROJECT_ROOT
    cfg.get_dataset_root()
    cfg.get_source_audit_dir()
    ...
"""
from __future__ import annotations

import os
import platform
from pathlib import Path

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyYAML is required (see requirements.txt): pip install PyYAML"
    ) from e

# ---------------------------------------------------------------------------
# Project root: derived dynamically from this file's location.
# src/project_config.py -> parents[0] = src/, parents[1] = project root.
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATASET_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "dataset_config.yaml"

ENV_VAR_DATASET_ROOT = "CHEST_XRAY_DATA_ROOT"


def get_project_root() -> Path:
    """Root of the pneumonia_reliability_study project."""
    return PROJECT_ROOT


def get_source_audit_dir() -> Path:
    return PROJECT_ROOT / "data" / "source_audit"


def get_manifests_dir() -> Path:
    return PROJECT_ROOT / "data" / "manifests"


def get_checksums_dir() -> Path:
    return PROJECT_ROOT / "data" / "checksums"


def get_reports_dir() -> Path:
    return PROJECT_ROOT / "data" / "reports"


def get_logs_dir() -> Path:
    return PROJECT_ROOT / "logs"


def get_configs_dir() -> Path:
    return PROJECT_ROOT / "config"


def get_experiments_dir() -> Path:
    """Root of all experiment outputs (checkpoints, histories, predictions, ...)."""
    return PROJECT_ROOT / "experiments"


def get_experiment_a_dir() -> Path:
    """Output tree for Experiment A (protocol section 19.2)."""
    return get_experiments_dir() / "experiment_a"


def _load_dataset_config() -> dict:
    if not DATASET_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"dataset_config.yaml not found at {DATASET_CONFIG_PATH}. "
            "This file is required to resolve the dataset root when "
            f"{ENV_VAR_DATASET_ROOT} is not set."
        )
    with open(DATASET_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dataset_root() -> Path:
    """
    Resolve the read-only chest X-ray dataset root directory.

    Priority order:
      1. Environment variable CHEST_XRAY_DATA_ROOT (works on any OS).
      2. config/dataset_config.yaml -> dataset.data_root_windows (on Windows).
      3. config/dataset_config.yaml -> dataset.data_root (on POSIX).

    Raises FileNotFoundError with an actionable message if the resolved
    path does not exist or is not a directory. Never falls back to a
    hardcoded temporary/agent-session path.
    """
    env_value = os.environ.get(ENV_VAR_DATASET_ROOT)
    source_used = None

    if env_value:
        candidate = Path(env_value)
        source_used = f"environment variable {ENV_VAR_DATASET_ROOT}"
    else:
        cfg = _load_dataset_config()
        ds = cfg.get("dataset", {})
        if platform.system() == "Windows":
            raw = ds.get("data_root_windows")
            source_used = "config/dataset_config.yaml -> dataset.data_root_windows"
        else:
            raw = ds.get("data_root")
            source_used = "config/dataset_config.yaml -> dataset.data_root"
        if not raw:
            raise FileNotFoundError(
                "No dataset root configured for this OS and "
                f"{ENV_VAR_DATASET_ROOT} is not set.\n"
                "Set it before running any script, e.g.:\n"
                '  PowerShell:  $env:CHEST_XRAY_DATA_ROOT = "D:\\Claude\\Pneumonia\\chest_xray\\chest_xray"\n'
                '  POSIX:       export CHEST_XRAY_DATA_ROOT="/path/to/chest_xray"\n'
            )
        candidate = Path(raw)

    candidate = candidate.expanduser()

    if not candidate.is_dir():
        raise FileNotFoundError(
            f"Dataset root not found (resolved via {source_used}): {candidate}\n"
            "The original chest X-ray dataset is required and must not be moved, "
            "renamed, or deleted. Set the CHEST_XRAY_DATA_ROOT environment "
            "variable to point at the correct location, e.g.:\n"
            '  PowerShell:  $env:CHEST_XRAY_DATA_ROOT = "D:\\Claude\\Pneumonia\\chest_xray\\chest_xray"\n'
            '  POSIX:       export CHEST_XRAY_DATA_ROOT="/path/to/chest_xray"\n'
        )
    return candidate


def get_audit_import_source_dir() -> Path:
    """
    Location of the ORIGINAL (pre-freeze) audit output folder, used only by
    import_audit_results.py to populate data/source_audit/. Derived
    relative to the project root (audit_hasil is a sibling directory of the
    project), never a hardcoded session path. Can be overridden with the
    CHEST_XRAY_AUDIT_SOURCE_DIR environment variable if the audit output
    lives elsewhere.
    """
    env_value = os.environ.get("CHEST_XRAY_AUDIT_SOURCE_DIR")
    if env_value:
        return Path(env_value).expanduser()
    return PROJECT_ROOT.parent / "audit_hasil"
