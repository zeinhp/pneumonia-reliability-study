import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import project_config as cfg

MANIFESTS = cfg.get_manifests_dir()
REPORTS = cfg.get_reports_dir()
AUDIT = cfg.get_source_audit_dir()


def test_global_keep_set_has_5856_decisions():
    df = pd.read_csv(MANIFESTS / "global_deduplication_keep_set.csv")
    assert len(df) == 5856


def test_global_keep_set_counts():
    df = pd.read_csv(MANIFESTS / "global_deduplication_keep_set.csv")
    counts = df["dedup_action"].value_counts()
    assert counts.get("keep", 0) == 5824
    assert counts.get("exclude", 0) == 32


def test_global_keep_set_rule_documented():
    df = pd.read_csv(MANIFESTS / "global_deduplication_keep_set.csv")
    excluded = df[df["dedup_action"] == "exclude"]
    assert (excluded["dedup_reason"].str.startswith("non_representative_")).all()
    kept_dup = df[(df["dedup_action"] == "keep") & df["exact_duplicate_group_id"].notna()]
    assert (kept_dup["dedup_reason"].str.startswith("representative_")).all()


def test_global_keep_set_no_near_duplicate_exclusions():
    """Near-duplicates must never be used to exclude images - only exact MD5
    duplicate groups (as recorded in exact_duplicates.csv) may appear as a
    reason for exclusion."""
    df = pd.read_csv(MANIFESTS / "global_deduplication_keep_set.csv")
    excluded = df[df["dedup_action"] == "exclude"]
    assert excluded["exact_duplicate_group_id"].notna().all()


def test_dedup_row_count():
    df = pd.read_csv(MANIFESTS / "patient_grouped_split_deduplicated.csv")
    assert len(df) == 5824


def test_dedup_no_repeated_md5():
    manifest = pd.read_csv(AUDIT / "manifest.csv")
    manifest["relative_path"] = manifest["split"] + "/" + manifest["label"] + "/" + manifest["filename"]
    md5_map = manifest.set_index("relative_path")["md5"]

    dedup = pd.read_csv(MANIFESTS / "patient_grouped_split_deduplicated.csv")
    dedup["_md5"] = dedup["relative_path"].map(md5_map)
    assert dedup["_md5"].duplicated().sum() == 0


def test_deduplication_report_row_count():
    report = pd.read_csv(REPORTS / "deduplication_report.csv")
    assert len(report) == 32


def test_deduplication_report_rule_documented():
    report = pd.read_csv(REPORTS / "deduplication_report.csv")
    assert (report["rule"] == "lexicographically_smallest_filename_within_md5_group").all()


def test_dedup_preserves_zero_patient_overlap():
    df = pd.read_csv(MANIFESTS / "patient_grouped_split_deduplicated.csv")
    overlap = (df.groupby("patient_id")["experimental_split"].nunique() > 1).sum()
    assert overlap == 0


def test_known_leak_pair_resolved_in_patient_grouped_deduplicated():
    """The IM-0095 (test) / IM-0096 (train) exact-duplicate pair that spanned
    the frozen patient-grouped split must be resolved by the global keep-set:
    only the lexicographically-smaller filename (IM-0095) may remain."""
    df = pd.read_csv(MANIFESTS / "patient_grouped_split_deduplicated.csv")
    paths = set(df["relative_path"])
    assert "test/NORMAL/NORMAL2-IM-0095-0001.jpeg" in paths
    assert "test/NORMAL/NORMAL2-IM-0096-0001.jpeg" not in paths
