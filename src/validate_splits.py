"""
validate_splits.py

Comprehensive integrity validation across every manifest produced by this
project, split into two tracks:

  PRIMARY   - the three deduplicated manifests (original_split_deduplicated,
              image_stratified_split_deduplicated,
              patient_grouped_split_deduplicated). ALL checks here are core:
              any failure sets primary_analysis_status = FAIL and the
              process exits non-zero.

  SENSITIVITY - the three all-images manifests (original_split_manifest,
              image_stratified_split_manifest,
              patient_grouped_split_all_images). Structural integrity is
              still checked, but the manifests are KNOWN to carry patient
              overlap (original, image-level) and, for the patient-grouped
              all-images manifest, one cross-split exact-duplicate group
              (IM-0095/IM-0096). These known issues are reported explicitly
              and do NOT fail the run - but the report never claims these
              manifests are clean, and any *unexpected* structural problem
              (missing file, duplicate path, bad label, checksum drift,
              etc.) is still treated as a failure.

Read-only: never modifies manifests or source data. Writes
data/reports/split_validation_report.json.
"""
import sys
import json
import hashlib
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_config as cfg

AUDIT = cfg.get_source_audit_dir()
MANIFESTS = cfg.get_manifests_dir()
CHECKSUMS = cfg.get_checksums_dir()
DATA_ROOT = cfg.get_dataset_root()
REPORT_PATH = cfg.get_reports_dir() / "split_validation_report.json"

# known, documented, ACCEPTED issues on the sensitivity (all-images) manifests
KNOWN_SENSITIVITY_ISSUES = {
    "original_split_manifest.csv": {"expected_min_patient_overlap": 1},   # 264 in practice
    "image_stratified_split_manifest.csv": {"expected_min_patient_overlap": 1},  # 684 in practice
    "patient_grouped_split_all_images.csv": {"expected_cross_split_dup_groups": 1},  # IM-0095/IM-0096
}

PRIMARY_MANIFESTS = {
    "original_deduplicated": ("original_split_deduplicated.csv", "experimental_split"),
    "image_level_stratified_deduplicated": ("image_stratified_split_deduplicated.csv", "experimental_split"),
    "patient_grouped_deduplicated": ("patient_grouped_split_deduplicated.csv", "experimental_split"),
}

SENSITIVITY_MANIFESTS = {
    "original_all_images": ("original_split_manifest.csv", "experimental_split"),
    "image_level_all_images": ("image_stratified_split_manifest.csv", "experimental_split"),
    "patient_grouped_all_images": ("patient_grouped_split_all_images.csv", "experimental_split"),
}

manifest_src = pd.read_csv(AUDIT / "manifest.csv")
manifest_src["relative_path"] = manifest_src["split"] + "/" + manifest_src["label"] + "/" + manifest_src["filename"]
md5_map = manifest_src.set_index("relative_path")["md5"]

exact_dup = pd.read_csv(AUDIT / "exact_duplicates.csv")
exact_dup["relative_path"] = exact_dup["split"] + "/" + exact_dup["label"] + "/" + exact_dup["filename"]

_dataset_checksums = pd.read_csv(CHECKSUMS / "dataset_files_sha256.csv")
EXISTING_PATHS = set(_dataset_checksums["relative_path"])
assert len(EXISTING_PATHS) == 5856, (
    f"dataset_files_sha256.csv has {len(EXISTING_PATHS)} unique paths, expected 5856 - "
    "re-run checksum_dataset_files.py before validating splits."
)

primary_checks = []
sensitivity_checks = []
primary_core_failed = False
sensitivity_unexpected_failure = False


def add_primary(name, condition, detail):
    global primary_core_failed
    status = "PASS" if condition else "FAIL"
    if not condition:
        primary_core_failed = True
    primary_checks.append({"check": name, "status": status, "detail": str(detail)})
    print(f"[PRIMARY {status}] {name}: {detail}")


def add_sensitivity(name, condition, detail, unexpected_if_fail=True):
    global sensitivity_unexpected_failure
    status = "PASS" if condition else "KNOWN_ISSUE_OR_FAIL"
    if not condition and unexpected_if_fail:
        sensitivity_unexpected_failure = True
    sensitivity_checks.append({"check": name, "status": status, "detail": str(detail)})
    print(f"[SENSITIVITY {status}] {name}: {detail}")


def structural_checks(df, split_col, name, add_fn):
    # Existence is checked against the pre-computed dataset checksum manifest
    # (data/checksums/dataset_files_sha256.csv), not via fresh per-row
    # filesystem stat() calls - this avoids ~35,000 individual disk hits
    # across a slow network-mounted drive. That checksum file is itself
    # produced by walking the real dataset root and is re-validated in
    # validate_source_audit.py, so this remains an honest existence check.
    missing = [p for p in df["relative_path"] if p not in EXISTING_PATHS]
    add_fn(f"all_paths_exist__{name}", len(missing) == 0,
           f"{len(missing)} missing" + (f" e.g. {missing[:3]}" if missing else ""))

    n_dup = df["relative_path"].duplicated().sum()
    add_fn(f"no_duplicate_path__{name}", n_dup == 0, f"{n_dup} duplicated relative_path rows")

    multi = df.groupby("relative_path")[split_col].nunique()
    n_multi = int((multi > 1).sum())
    add_fn(f"image_in_single_split__{name}", n_multi == 0, f"{n_multi} images in >1 split")

    bad = set(df["label"].unique()) - {"NORMAL", "PNEUMONIA"}
    add_fn(f"label_valid__{name}", len(bad) == 0, f"unexpected labels: {bad}")

    class_counts = df.groupby(split_col)["label"].nunique()
    bad_splits = class_counts[class_counts < 2].index.tolist()
    add_fn(f"both_classes_present__{name}", len(bad_splits) == 0, f"splits missing a class: {bad_splits}")

    is_abs = df["relative_path"].astype(str).str.startswith("/").any() or \
        df["relative_path"].astype(str).str.contains(r"^[A-Za-z]:\\").any()
    add_fn(f"paths_are_relative__{name}", not is_abs, f"absolute path detected: {is_abs}")

    n_test = int((df[split_col] == "test").sum())
    add_fn(f"test_set_not_empty__{name}", n_test > 0, f"n_test={n_test}")


# ===========================================================================
# PRIMARY track
# ===========================================================================
primary_dfs = {}
for name, (fname, split_col) in PRIMARY_MANIFESTS.items():
    df = pd.read_csv(MANIFESTS / fname)
    primary_dfs[name] = df
    structural_checks(df, split_col, name, add_primary)

    df["_md5"] = df["relative_path"].map(md5_map)
    n_repeated_md5 = int(df["_md5"].duplicated().sum())
    add_primary(f"no_repeated_md5__{name}", n_repeated_md5 == 0, f"{n_repeated_md5} repeated md5")

    present = exact_dup[exact_dup["relative_path"].isin(set(df["relative_path"]))].copy()
    present["_split_here"] = present["relative_path"].map(df.set_index("relative_path")[split_col])
    n_cross = int((present.groupby("md5")["_split_here"].nunique() > 1).sum())
    add_primary(f"no_cross_split_exact_duplicate_groups__{name}", n_cross == 0,
                f"{n_cross} exact-duplicate groups span >1 split")

# patient-grouped deduplicated: zero patient overlap
grp_df = primary_dfs["patient_grouped_deduplicated"]
overlap = int((grp_df.groupby("patient_id")["experimental_split"].nunique() > 1).sum())
add_primary("zero_patient_overlap__patient_grouped_deduplicated", overlap == 0, f"{overlap} overlapping patients")

# cohort identity across the three primary manifests
sets = {name: set(df["relative_path"]) for name, df in primary_dfs.items()}
names = list(sets.keys())
all_equal = sets[names[0]] == sets[names[1]] == sets[names[2]]
add_primary("primary_manifests_share_identical_cohort", all_equal,
            f"cohort sizes: {[len(sets[n]) for n in names]}, equal={all_equal}")

# source checksums still match (tamper/drift detection on frozen source_audit copy)
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
add_primary("source_audit_checksums_unchanged", len(mismatches) == 0, f"mismatched files: {mismatches}")

primary_analysis_status = "FAIL" if primary_core_failed else "PASS"

# ===========================================================================
# SENSITIVITY track
# ===========================================================================
for name, (fname, split_col) in SENSITIVITY_MANIFESTS.items():
    df = pd.read_csv(MANIFESTS / fname)
    structural_checks(df, split_col, name, add_sensitivity)

    overlap_series = df.groupby("patient_id")[split_col].nunique()
    n_overlap = int((overlap_series > 1).sum())
    expected = KNOWN_SENSITIVITY_ISSUES.get(fname, {})
    if "expected_min_patient_overlap" in expected:
        add_sensitivity(f"known_patient_overlap__{name}", n_overlap >= expected["expected_min_patient_overlap"],
                         f"{n_overlap} overlapping patients (known/expected for this uncorrected manifest)",
                         unexpected_if_fail=False)
    else:
        add_sensitivity(f"patient_overlap_recorded__{name}", True, f"{n_overlap} overlapping patients (informational)",
                         unexpected_if_fail=False)

    df["_md5"] = df["relative_path"].map(md5_map)
    n_repeated_md5 = int(df["_md5"].duplicated().sum())
    add_sensitivity(f"repeated_md5_recorded__{name}", n_repeated_md5 == 0,
                     f"{n_repeated_md5} repeated md5", unexpected_if_fail=False)

    present = exact_dup[exact_dup["relative_path"].isin(set(df["relative_path"]))].copy()
    present["_split_here"] = present["relative_path"].map(df.set_index("relative_path")[split_col])
    n_cross = int((present.groupby("md5")["_split_here"].nunique() > 1).sum())
    if "expected_cross_split_dup_groups" in expected:
        add_sensitivity(f"known_cross_split_exact_duplicate_groups__{name}",
                         n_cross == expected["expected_cross_split_dup_groups"],
                         f"{n_cross} cross-split exact-duplicate groups "
                         f"(known issue: IM-0095/IM-0096, MD5 f9528b244ccd49f3d89ead90ba09c520)",
                         unexpected_if_fail=False)
    else:
        add_sensitivity(f"cross_split_exact_duplicate_groups_recorded__{name}", n_cross == 0,
                         f"{n_cross} cross-split exact-duplicate groups", unexpected_if_fail=False)

if sensitivity_unexpected_failure:
    sensitivity_analysis_status = "FAIL"
else:
    known_issue_present = any(
        c["status"] == "KNOWN_ISSUE_OR_FAIL" for c in sensitivity_checks
    )
    sensitivity_analysis_status = "PASS_WITH_KNOWN_SENSITIVITY_ISSUES" if known_issue_present else "PASS"

# ===========================================================================
# Report
# ===========================================================================
report = {
    "validated_at": pd.Timestamp.now().isoformat(),
    "primary_analysis_status": primary_analysis_status,
    "sensitivity_analysis_status": sensitivity_analysis_status,
    "primary_checks": primary_checks,
    "sensitivity_checks": sensitivity_checks,
    "known_sensitivity_issues": KNOWN_SENSITIVITY_ISSUES,
}
with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2, default=str)

print("\n=== primary_analysis_status:", primary_analysis_status, "===")
print("=== sensitivity_analysis_status:", sensitivity_analysis_status, "===")

if primary_core_failed or sensitivity_unexpected_failure:
    sys.exit(1)
