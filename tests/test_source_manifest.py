import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import project_config as cfg

AUDIT = cfg.get_source_audit_dir()


@pytest.fixture(scope="module")
def manifest():
    return pd.read_csv(AUDIT / "manifest.csv")


@pytest.fixture(scope="module")
def proposed():
    return pd.read_csv(AUDIT / "proposed_split_mapping.csv")


def test_manifest_row_count(manifest):
    assert len(manifest) == 5856


def test_unique_patient_count(manifest):
    assert manifest["patient_id"].nunique(dropna=True) == 2790


def test_no_empty_patient_id(manifest):
    assert manifest["patient_id"].isna().sum() == 0


def test_no_duplicate_filepath(manifest):
    assert manifest["filepath"].duplicated().sum() == 0


def test_no_patient_with_mixed_labels(manifest):
    per_patient = manifest.groupby("patient_id")["label"].nunique()
    assert (per_patient > 1).sum() == 0


def test_label_domain(manifest):
    assert set(manifest["label"].unique()) <= {"NORMAL", "PNEUMONIA"}


def test_split_domain(manifest):
    assert set(manifest["split"].unique()) <= {"train", "val", "test"}


def test_original_split_patient_overlap_is_264(manifest):
    per_patient = manifest.groupby("patient_id")["split"].nunique()
    assert (per_patient > 1).sum() == 264


def test_proposed_mapping_row_count(proposed):
    assert len(proposed) == 5856


def test_proposed_split_counts(proposed):
    counts = proposed["new_split"].value_counts()
    assert counts.get("train_new", 0) == 4634
    assert counts.get("val_new", 0) == 607
    assert counts.get("test_new", 0) == 615


def test_proposed_split_zero_patient_overlap(proposed):
    per_patient = proposed.groupby("patient_id")["new_split"].nunique()
    assert (per_patient > 1).sum() == 0


def test_source_audit_checksums_unchanged():
    """Frozen source_audit files must never be modified in place."""
    import hashlib
    sha_df = pd.read_csv(cfg.get_checksums_dir() / "source_audit_sha256.csv")
    mismatches = []
    for _, r in sha_df.iterrows():
        fpath = AUDIT / r["filename"]
        h = hashlib.sha256()
        with open(fpath, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != r["sha256"]:
            mismatches.append(r["filename"])
    assert mismatches == [], f"source_audit files changed: {mismatches}"
