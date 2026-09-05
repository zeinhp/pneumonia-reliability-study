"""
build_split_summary.py

Aggregates per-split statistics across all six split manifests (three
all-images / sensitivity-only, three deduplicated / primary) into
data/reports/split_summary.csv. Reads existing manifests only - no
recomputation of splits, hashes, or patient IDs.

n_overlapping_patients is a METHOD-level metric (number of unique patients
that appear in more than one subset for that split method) and is
deliberately repeated on every row belonging to that method - this is
documented here and in the CSV's companion note in manifest_lineage.md, not
a bug.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_config as cfg

AUDIT = cfg.get_source_audit_dir()
MANIFESTS = cfg.get_manifests_dir()
REPORTS = cfg.get_reports_dir()

master = pd.read_csv(MANIFESTS / "master_manifest_derived.csv")
quality = master.set_index("relative_path")[["is_low_resolution", "is_blur_candidate", "is_exact_duplicate"]]

manifest_src = pd.read_csv(AUDIT / "manifest.csv")
manifest_src["relative_path"] = manifest_src["split"] + "/" + manifest_src["label"] + "/" + manifest_src["filename"]
md5_map = manifest_src.set_index("relative_path")["md5"]

exact_dup = pd.read_csv(AUDIT / "exact_duplicates.csv")
exact_dup["relative_path"] = exact_dup["split"] + "/" + exact_dup["label"] + "/" + exact_dup["filename"]

# (filename, split_col, cohort, analysis_role, primary_eligible)
STRATEGIES = [
    ("original_split_manifest.csv", "experimental_split", "all_images_5856", "sensitivity_only", False),
    ("image_stratified_split_manifest.csv", "experimental_split", "all_images_5856", "sensitivity_only", False),
    ("patient_grouped_split_all_images.csv", "experimental_split", "all_images_5856", "sensitivity_only", False),
    ("original_split_deduplicated.csv", "experimental_split", "global_exact_deduplicated_5824", "primary_control", True),
    ("image_stratified_split_deduplicated.csv", "experimental_split", "global_exact_deduplicated_5824", "primary_control", True),
    ("patient_grouped_split_deduplicated.csv", "experimental_split", "global_exact_deduplicated_5824", "primary", True),
]

SPLIT_METHOD_NAMES = {
    "original_split_manifest.csv": "original",
    "image_stratified_split_manifest.csv": "image_level_stratified",
    "patient_grouped_split_all_images.csv": "patient_level_grouped_stratified_frozen",
    "original_split_deduplicated.csv": "original_deduplicated",
    "image_stratified_split_deduplicated.csv": "image_level_stratified_deduplicated",
    "patient_grouped_split_deduplicated.csv": "patient_level_grouped_stratified_frozen_deduplicated",
}

rows = []
for fname, split_col, cohort, analysis_role, primary_eligible in STRATEGIES:
    df = pd.read_csv(MANIFESTS / fname)
    df = df.join(quality, on="relative_path")
    df["_md5"] = df["relative_path"].map(md5_map)

    method_name = SPLIT_METHOD_NAMES[fname]

    # method-level: patient overlap across subsets of THIS method
    overlap_series = df.groupby("patient_id")[split_col].nunique()
    n_overlap = int((overlap_series > 1).sum())

    # method-level: repeated md5 within this manifest
    n_repeated_md5 = int(df["_md5"].duplicated().sum())

    # method-level: exact-duplicate groups whose members (present in this
    # manifest) span more than one split value
    present = exact_dup[exact_dup["relative_path"].isin(set(df["relative_path"]))].copy()
    present["_split_here"] = present["relative_path"].map(df.set_index("relative_path")[split_col])
    n_cross_split_groups = int((present.groupby("md5")["_split_here"].nunique() > 1).sum())

    for split_val, g in df.groupby(split_col):
        n_normal = int((g["label"] == "NORMAL").sum())
        n_pneumonia = int((g["label"] == "PNEUMONIA").sum())
        n_total = len(g)
        rows.append({
            "cohort": cohort,
            "analysis_role": analysis_role,
            "primary_eligible": primary_eligible,
            "split_method": method_name,
            "experimental_split": split_val,
            "n_images": n_total,
            "n_patients": int(g["patient_id"].nunique()),
            "n_normal": n_normal,
            "n_pneumonia": n_pneumonia,
            "normal_percentage": round(n_normal / n_total * 100, 2) if n_total else 0,
            "pneumonia_percentage": round(n_pneumonia / n_total * 100, 2) if n_total else 0,
            "n_overlapping_patients": n_overlap,
            "n_repeated_md5": n_repeated_md5,
            "n_cross_split_exact_duplicate_groups": n_cross_split_groups,
        })

summary_df = pd.DataFrame(rows)
OUT = REPORTS / "split_summary.csv"
summary_df.to_csv(OUT, index=False)
print(f"Wrote {len(summary_df)} rows to {OUT}")
print(summary_df.to_string())
print(
    "\nNOTE: n_overlapping_patients, n_repeated_md5, and "
    "n_cross_split_exact_duplicate_groups are METHOD-level metrics "
    "(computed once per split_method across all its subsets) and are "
    "intentionally repeated on every row of that method."
)
