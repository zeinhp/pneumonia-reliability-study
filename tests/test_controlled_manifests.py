"""
Unit tests mirroring protokol.docx bagian 6.7 - manifest terkontrol untuk
Eksperimen A (Clean vs Controlled Leaky, common anchor test).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import project_config as cfg

MANIFESTS = cfg.get_manifests_dir()
AUDIT = cfg.get_source_audit_dir()
CHECKSUMS = cfg.get_checksums_dir()

manifest_src = pd.read_csv(AUDIT / "manifest.csv")
manifest_src["relative_path"] = manifest_src["split"] + "/" + manifest_src["label"] + "/" + manifest_src["filename"]
MD5_MAP = manifest_src.set_index("relative_path")["md5"]
EXISTING_PATHS = set(pd.read_csv(CHECKSUMS / "dataset_files_sha256.csv")["relative_path"])


@pytest.fixture(scope="module")
def anchor():
    return pd.read_csv(MANIFESTS / "controlled_anchor_test.csv")


@pytest.fixture(scope="module")
def clean_train():
    return pd.read_csv(MANIFESTS / "controlled_clean_train.csv")


@pytest.fixture(scope="module")
def clean_val():
    return pd.read_csv(MANIFESTS / "controlled_clean_validation.csv")


@pytest.fixture(scope="module")
def leaky_train():
    return pd.read_csv(MANIFESTS / "controlled_leaky_train.csv")


@pytest.fixture(scope="module")
def repl_map():
    return pd.read_csv(MANIFESTS / "controlled_leakage_replacement_map.csv")


def test_anchor_exactly_one_per_patient(anchor):
    assert len(anchor) == 279
    assert anchor["patient_id"].is_unique


def test_anchor_class_distribution_matches_frozen_numbers(anchor):
    counts = anchor["label"].value_counts()
    assert counts.get("PNEUMONIA", 0) == 169
    assert counts.get("NORMAL", 0) == 110


def test_K_sibling_eligible_patients_is_146(repl_map):
    assert len(repl_map) == 146


def test_sibling_class_breakdown_matches_frozen_numbers(repl_map):
    counts = repl_map["label"].value_counts()
    assert counts.get("PNEUMONIA", 0) == 112
    assert counts.get("NORMAL", 0) == 34


def test_train_size_clean_equals_leaky(clean_train, leaky_train):
    assert len(clean_train) == len(leaky_train) == 4607


def test_class_distribution_train_comparable(clean_train, leaky_train):
    c = clean_train["label"].value_counts().sort_index()
    l = leaky_train["label"].value_counts().sort_index()
    assert c.equals(l)


def test_anchor_not_leaked_into_train_or_validation(anchor, clean_train, clean_val, leaky_train):
    paths = set(anchor["relative_path"])
    assert not (paths & set(clean_train["relative_path"]))
    assert not (paths & set(leaky_train["relative_path"]))
    assert not (paths & set(clean_val["relative_path"]))


def test_no_repeated_md5_in_any_controlled_manifest(anchor, clean_train, clean_val, leaky_train):
    for df in [anchor, clean_train, clean_val, leaky_train]:
        md5s = df["relative_path"].map(MD5_MAP)
        assert md5s.duplicated().sum() == 0


def test_clean_condition_zero_patient_overlap(clean_train, clean_val, anchor):
    assert not (set(clean_train["patient_id"]) & set(clean_val["patient_id"]))
    assert not (set(clean_train["patient_id"]) & set(anchor["patient_id"]))
    assert not (set(clean_val["patient_id"]) & set(anchor["patient_id"]))


def test_leaky_condition_overlap_matches_contamination_map_exactly(leaky_train, anchor, clean_val, repl_map):
    overlap = set(leaky_train["patient_id"]) & set(anchor["patient_id"])
    assert overlap == set(repl_map["sibling_patient_id"])
    assert not (set(leaky_train["patient_id"]) & set(clean_val["patient_id"]))


def test_all_siblings_come_from_anchor_test_patients(repl_map, anchor):
    assert set(repl_map["sibling_patient_id"]) <= set(anchor["patient_id"])


def test_replacement_images_not_from_validation_or_test_patients(repl_map, clean_val, anchor):
    forbidden = set(clean_val["patient_id"]) | set(anchor["patient_id"])
    assert not (set(repl_map["replaced_patient_id"]) & forbidden)


def test_replacement_max_one_image_per_training_patient(repl_map):
    counts = repl_map["replaced_patient_id"].value_counts()
    assert (counts <= 1).all()


def test_all_controlled_paths_exist(anchor, clean_train, clean_val, leaky_train):
    for df in [anchor, clean_train, clean_val, leaky_train]:
        missing = [p for p in df["relative_path"] if p not in EXISTING_PATHS]
        assert missing == []


def test_source_audit_checksums_unchanged():
    import hashlib
    sha_df = pd.read_csv(CHECKSUMS / "source_audit_sha256.csv")
    mismatches = []
    for _, r in sha_df.iterrows():
        fpath = AUDIT / r["filename"]
        h = hashlib.sha256()
        with open(fpath, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != r["sha256"]:
            mismatches.append(r["filename"])
    assert mismatches == []


def test_controlled_manifests_use_relative_paths_only(anchor, clean_train, clean_val, leaky_train):
    for df in [anchor, clean_train, clean_val, leaky_train]:
        assert not df["relative_path"].astype(str).str.startswith("/").any()
        assert not df["relative_path"].astype(str).str.contains(r"^[A-Za-z]:\\").any()
