"""
validate_source_audit.py

Validates the frozen source audit (manifest.csv, proposed_split_mapping.csv,
exact_duplicates.csv) against the invariants documented for the frozen
audit, WITHOUT recomputing anything. Also confirms that every image
referenced by the manifest physically exists at the (read-only) dataset
root configured via project_config.get_dataset_root().

If any CORE check fails, the script exits non-zero and the pipeline must
stop before any derived manifest is built (per project rules).
"""
import sys
import json
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_config as cfg

AUDIT = cfg.get_source_audit_dir()
DATA_ROOT = cfg.get_dataset_root()
REPORT_PATH = cfg.get_reports_dir() / "source_validation_report.json"

EXPECTED = {
    "n_manifest_rows": 5856,
    "n_unique_patients": 2790,
    "n_overlap_patients_original": 264,
    "n_exact_dup_files": 62,
    "n_exact_dup_groups": 30,
    "n_proposed_rows": 5856,
    "train_new": 4634,
    "val_new": 607,
    "test_new": 615,
}

checks = []
core_failed = False


def check(name, condition, detail, core=True):
    global core_failed
    status = "PASS" if condition else "FAIL"
    if not condition and core:
        core_failed = True
    checks.append({"check": name, "status": status, "core": core, "detail": detail})
    print(f"[{status}] {name}: {detail}")


manifest = pd.read_csv(AUDIT / "manifest.csv")
proposed = pd.read_csv(AUDIT / "proposed_split_mapping.csv")
exact_dup = pd.read_csv(AUDIT / "exact_duplicates.csv")

# 1. manifest.csv exact row count
check("manifest_row_count", len(manifest) == EXPECTED["n_manifest_rows"],
      f"got {len(manifest)}, expected {EXPECTED['n_manifest_rows']}")

# 2. unique patient IDs
n_unique_patients = manifest["patient_id"].nunique(dropna=True)
check("unique_patient_count", n_unique_patients == EXPECTED["n_unique_patients"],
      f"got {n_unique_patients}, expected {EXPECTED['n_unique_patients']}")

# 3. no empty patient_id
n_empty_pid = manifest["patient_id"].isna().sum()
check("no_empty_patient_id", n_empty_pid == 0, f"{n_empty_pid} empty patient_id rows")

# 4. no duplicate filepath
n_dup_filepath = manifest["filepath"].duplicated().sum()
check("no_duplicate_filepath", n_dup_filepath == 0, f"{n_dup_filepath} duplicate filepath rows")

# 5. no patient with both NORMAL and PNEUMONIA labels
label_per_patient = manifest.groupby("patient_id")["label"].nunique()
n_mixed_label_patients = int((label_per_patient > 1).sum())
check("no_patient_with_mixed_labels", n_mixed_label_patients == 0,
      f"{n_mixed_label_patients} patients with both NORMAL and PNEUMONIA")

# 6. label only NORMAL or PNEUMONIA
bad_labels = set(manifest["label"].unique()) - {"NORMAL", "PNEUMONIA"}
check("label_domain_valid", len(bad_labels) == 0, f"unexpected labels: {bad_labels}")

# 7. split only train/val/test
bad_splits = set(manifest["split"].unique()) - {"train", "val", "test"}
check("split_domain_valid", len(bad_splits) == 0, f"unexpected splits: {bad_splits}")

# 8. exactly 264 patients appear in more than one original split
patient_split_counts = manifest.groupby("patient_id")["split"].nunique()
n_overlap = int((patient_split_counts > 1).sum())
check("original_split_patient_overlap_count", n_overlap == EXPECTED["n_overlap_patients_original"],
      f"got {n_overlap}, expected {EXPECTED['n_overlap_patients_original']}")

# 9. exact_duplicates.csv: 62 files in 30 groups
n_dup_files = len(exact_dup)
n_dup_groups = exact_dup["md5"].nunique()
check("exact_dup_file_count", n_dup_files == EXPECTED["n_exact_dup_files"],
      f"got {n_dup_files}, expected {EXPECTED['n_exact_dup_files']}")
check("exact_dup_group_count", n_dup_groups == EXPECTED["n_exact_dup_groups"],
      f"got {n_dup_groups}, expected {EXPECTED['n_exact_dup_groups']}")

# 10. proposed_split_mapping.csv has 5856 rows
check("proposed_mapping_row_count", len(proposed) == EXPECTED["n_proposed_rows"],
      f"got {len(proposed)}, expected {EXPECTED['n_proposed_rows']}")

# 11. all files in proposed mapping exist in source manifest (match on split+label+filename)
manifest_keys = set(zip(manifest["split"], manifest["label"], manifest["filename"]))
proposed_keys = list(zip(proposed["split"], proposed["label"], proposed["filename"]))
n_missing_in_manifest = sum(1 for k in proposed_keys if k not in manifest_keys)
check("proposed_files_exist_in_manifest", n_missing_in_manifest == 0,
      f"{n_missing_in_manifest} proposed rows not found in source manifest")

# 12. each file appears exactly once in proposed mapping
n_dup_proposed = pd.Series(proposed_keys).duplicated().sum()
check("proposed_mapping_no_duplicate_files", n_dup_proposed == 0,
      f"{n_dup_proposed} duplicated file entries in proposed mapping")

# 13. patient overlap on proposed split is zero
prop_patient_split = proposed.groupby("patient_id")["new_split"].nunique()
n_prop_overlap = int((prop_patient_split > 1).sum())
check("proposed_split_zero_patient_overlap", n_prop_overlap == 0,
      f"{n_prop_overlap} patients span multiple proposed splits")

# 14. proposed split sizes
counts = proposed["new_split"].value_counts()
check("proposed_train_new_count", counts.get("train_new", 0) == EXPECTED["train_new"],
      f"got {counts.get('train_new', 0)}, expected {EXPECTED['train_new']}")
check("proposed_val_new_count", counts.get("val_new", 0) == EXPECTED["val_new"],
      f"got {counts.get('val_new', 0)}, expected {EXPECTED['val_new']}")
check("proposed_test_new_count", counts.get("test_new", 0) == EXPECTED["test_new"],
      f"got {counts.get('test_new', 0)}, expected {EXPECTED['test_new']}")

# 15. every manifest image physically exists under the configured dataset root.
missing_files = []
for _, r in manifest.iterrows():
    rel = Path(r["split"]) / r["label"] / r["filename"]
    if not (DATA_ROOT / rel).is_file():
        missing_files.append(str(rel))
check("all_manifest_images_exist_at_project_data_root", len(missing_files) == 0,
      f"{len(missing_files)} missing files at {DATA_ROOT}" + (f" (e.g. {missing_files[:3]})" if missing_files else ""))

# also count actual files at DATA_ROOT for symmetry
n_root_files = 0
for split in ["train", "test", "val"]:
    for label in ["NORMAL", "PNEUMONIA"]:
        d = DATA_ROOT / split / label
        if d.is_dir():
            n_root_files += len([p for p in d.iterdir() if p.is_file() and not p.name.startswith(".")])
check("data_root_file_count_matches_manifest", n_root_files == len(manifest),
      f"got {n_root_files} files at data root, manifest has {len(manifest)} rows")

overall_status = "FAIL" if core_failed else "PASS"

report = {
    "audit_date_frozen": "2026-07-23",
    "validated_at": pd.Timestamp.now().isoformat(),
    "data_root": str(DATA_ROOT),
    "overall_status": overall_status,
    "checks": checks,
}

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2, default=str)

print("\n=== OVERALL:", overall_status, "===")
if core_failed:
    sys.exit(1)
