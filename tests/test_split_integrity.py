import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import project_config as cfg

AUDIT = cfg.get_source_audit_dir()
MANIFESTS = cfg.get_manifests_dir()
CHECKSUMS = cfg.get_checksums_dir()

manifest_src = pd.read_csv(AUDIT / "manifest.csv")
manifest_src["relative_path"] = manifest_src["split"] + "/" + manifest_src["label"] + "/" + manifest_src["filename"]
MD5_MAP = manifest_src.set_index("relative_path")["md5"]

# Existence is checked against the pre-computed dataset checksum manifest
# rather than live filesystem stat() calls per row - avoids ~35,000
# individual disk hits across a slow network-mounted drive during the test
# suite. dataset_files_sha256.csv is itself produced by walking the real
# dataset root (checksum_dataset_files.py) and is re-validated independently
# in validate_source_audit.py.
EXISTING_PATHS = set(pd.read_csv(CHECKSUMS / "dataset_files_sha256.csv")["relative_path"])

exact_dup = pd.read_csv(AUDIT / "exact_duplicates.csv")
exact_dup["relative_path"] = exact_dup["split"] + "/" + exact_dup["label"] + "/" + exact_dup["filename"]

ALL_METHODS = {
    "original_all_images": ("original_split_manifest.csv", "experimental_split"),
    "image_level_all_images": ("image_stratified_split_manifest.csv", "experimental_split"),
    "patient_grouped_all_images": ("patient_grouped_split_all_images.csv", "experimental_split"),
    "original_deduplicated": ("original_split_deduplicated.csv", "experimental_split"),
    "image_level_stratified_deduplicated": ("image_stratified_split_deduplicated.csv", "experimental_split"),
    "patient_grouped_deduplicated": ("patient_grouped_split_deduplicated.csv", "experimental_split"),
}

PRIMARY_METHODS = [
    "original_deduplicated",
    "image_level_stratified_deduplicated",
    "patient_grouped_deduplicated",
]


@pytest.fixture(scope="module", params=list(ALL_METHODS.keys()))
def split_df(request):
    fname, split_col = ALL_METHODS[request.param]
    df = pd.read_csv(MANIFESTS / fname)
    return request.param, df, split_col


def test_all_paths_exist(split_df):
    name, df, _ = split_df
    missing = [p for p in df["relative_path"] if p not in EXISTING_PATHS]
    assert missing == [], f"{name}: {len(missing)} missing files"


def test_no_duplicate_paths(split_df):
    name, df, _ = split_df
    assert df["relative_path"].duplicated().sum() == 0


def test_image_in_single_split(split_df):
    name, df, split_col = split_df
    multi = df.groupby("relative_path")[split_col].nunique()
    assert (multi > 1).sum() == 0


def test_labels_valid(split_df):
    name, df, _ = split_df
    assert set(df["label"].unique()) <= {"NORMAL", "PNEUMONIA"}


def test_both_classes_present_per_split(split_df):
    name, df, split_col = split_df
    class_counts = df.groupby(split_col)["label"].nunique()
    assert (class_counts >= 2).all()


def test_test_split_not_empty(split_df):
    name, df, split_col = split_df
    assert (df[split_col] == "test").sum() > 0


def test_paths_are_relative(split_df):
    name, df, _ = split_df
    assert not df["relative_path"].astype(str).str.startswith("/").any()
    assert not df["relative_path"].astype(str).str.contains(r"^[A-Za-z]:\\").any()


# ---------------------------------------------------------------------------
# PRIMARY-manifest-only invariants (must hold with zero tolerance)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module", params=PRIMARY_METHODS)
def primary_df(request):
    fname, split_col = ALL_METHODS[request.param]
    df = pd.read_csv(MANIFESTS / fname)
    return request.param, df, split_col


def test_primary_no_repeated_md5(primary_df):
    name, df, _ = primary_df
    md5s = df["relative_path"].map(MD5_MAP)
    assert md5s.duplicated().sum() == 0, f"{name}: repeated md5 found"


def test_primary_no_cross_split_exact_duplicate_groups(primary_df):
    name, df, split_col = primary_df
    present = exact_dup[exact_dup["relative_path"].isin(set(df["relative_path"]))].copy()
    present["_split_here"] = present["relative_path"].map(df.set_index("relative_path")[split_col])
    n_cross = int((present.groupby("md5")["_split_here"].nunique() > 1).sum())
    assert n_cross == 0, f"{name}: {n_cross} exact-duplicate groups span >1 split"


def test_patient_grouped_deduplicated_zero_overlap():
    df = pd.read_csv(MANIFESTS / "patient_grouped_split_deduplicated.csv")
    overlap = (df.groupby("patient_id")["experimental_split"].nunique() > 1).sum()
    assert overlap == 0


def test_image_level_stratified_deduplicated_has_measured_overlap():
    """The image-level control split is NOT patient-grouped by design, so
    patient overlap is expected to be nonzero - this test guards that the
    control condition still behaves as intended (i.e. nobody 'fixed' it by
    accidentally grouping)."""
    df = pd.read_csv(MANIFESTS / "image_stratified_split_deduplicated.csv")
    overlap = (df.groupby("patient_id")["experimental_split"].nunique() > 1).sum()
    assert overlap > 0


def test_original_deduplicated_preserves_original_split_assignment():
    orig_all = pd.read_csv(MANIFESTS / "original_split_manifest.csv").set_index("relative_path")["experimental_split"]
    dedup = pd.read_csv(MANIFESTS / "original_split_deduplicated.csv")
    mismatches = [
        rp for rp, s in zip(dedup["relative_path"], dedup["experimental_split"])
        if orig_all.get(rp) != s
    ]
    assert mismatches == [], f"{len(mismatches)} rows changed split assignment after dedup filtering"


def test_three_primary_manifests_share_identical_cohort():
    sets = {}
    for name in PRIMARY_METHODS:
        fname, _ = ALL_METHODS[name]
        df = pd.read_csv(MANIFESTS / fname)
        sets[name] = set(df["relative_path"])
        assert len(df) == 5824, f"{name}: expected 5824 rows, got {len(df)}"
    vals = list(sets.values())
    assert vals[0] == vals[1] == vals[2]


def test_patient_grouped_all_images_counts_unchanged():
    df = pd.read_csv(MANIFESTS / "patient_grouped_split_all_images.csv")
    counts = df["experimental_split"].value_counts()
    assert counts.get("train", 0) == 4634
    assert counts.get("validation", 0) == 607
    assert counts.get("test", 0) == 615


def test_image_level_all_images_full_coverage():
    df = pd.read_csv(MANIFESTS / "image_stratified_split_manifest.csv")
    assert len(df) == 5856
    assert df["split_seed"].unique().tolist() == [42]
