"""
Guards against regressions of the path-reproducibility refactor:
  1. No hardcoded agent/session path anywhere in src/, tests/, notebooks/.
  2. PROJECT_ROOT is derived dynamically (not a literal string constant).
  3. The dataset root can be overridden via the CHEST_XRAY_DATA_ROOT
     environment variable.
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import project_config as cfg

PROJECT_ROOT = cfg.get_project_root()

# Any of these substrings appearing in source/test/notebook files indicates a
# regression back to a hardcoded ephemeral agent-session path.
FORBIDDEN_SUBSTRINGS = [
    "/sessions/beautiful-tender-feynman",
]

SCAN_DIRS = ["src", "tests", "notebooks"]
SCAN_EXTENSIONS = {".py", ".ipynb", ".yaml", ".yml", ".md"}


_THIS_FILE = Path(__file__).resolve()


def _files_to_scan():
    for d in SCAN_DIRS:
        base = PROJECT_ROOT / d
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if (
                p.is_file()
                and p.suffix in SCAN_EXTENSIONS
                and "__pycache__" not in p.parts
                and p.resolve() != _THIS_FILE  # this file legitimately references the
                                                # forbidden string as check data, not as
                                                # a path in use
            ):
                yield p


def test_no_hardcoded_session_paths():
    offenders = []
    for fpath in _files_to_scan():
        text = fpath.read_text(encoding="utf-8", errors="ignore")
        for bad in FORBIDDEN_SUBSTRINGS:
            if bad in text:
                offenders.append((str(fpath.relative_to(PROJECT_ROOT)), bad))
    assert offenders == [], f"hardcoded session path(s) found: {offenders}"


def test_project_root_is_dynamic():
    # PROJECT_ROOT must equal the actual parent of src/, not a hardcoded string,
    # and must exist on disk (proves it was resolved, not guessed).
    assert cfg.PROJECT_ROOT == Path(__file__).resolve().parents[1]
    assert cfg.PROJECT_ROOT.is_dir()
    assert (cfg.PROJECT_ROOT / "src" / "project_config.py").is_file()


def test_dataset_root_overridable_via_env_var(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "train" / "NORMAL").mkdir(parents=True)
        monkeypatch.setenv(cfg.ENV_VAR_DATASET_ROOT, str(tmp_path))
        resolved = cfg.get_dataset_root()
        assert resolved == tmp_path.resolve() or resolved == tmp_path


def test_dataset_root_raises_clear_error_when_unset_and_unresolvable(monkeypatch):
    monkeypatch.delenv(cfg.ENV_VAR_DATASET_ROOT, raising=False)
    monkeypatch.setattr(cfg, "platform", cfg.platform)  # no-op, keeps import explicit
    import platform as _platform
    if _platform.system() != "Windows":
        # on POSIX with no env var and an empty data_root in the yaml, this must
        # raise a clear, actionable FileNotFoundError - never fall back to a
        # hardcoded temp/agent path.
        with pytest.raises(FileNotFoundError):
            cfg.get_dataset_root()
