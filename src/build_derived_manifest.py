"""
build_derived_manifest.py

Builds data/manifests/master_manifest_derived.csv, original_split_manifest.csv,
and patient_grouped_split_all_images.csv by JOINING the frozen source-audit
files. No source values are recomputed or altered; this script only merges
existing columns and adds boolean flags / stable IDs / cohort metadata.

Inputs (read-only, from data/source_audit/):
    manifest.csv
    proposed_split_mapping.csv
    low_resolution_images.csv
    potential_blurry_images.csv
    exact_duplicates.csv

Outputs (data/manifests/):
    master_manifest_derived.csv
    original_split_manifest.csv
    patient_grouped_split_all_images.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_config as cfg

AUDIT = cfg.get_source_audit_dir()
MANIFESTS = cfg.get_manifests_dir()

manifest = pd.read_csv(AUDIT / "manifest.csv")
proposed = pd.read_csv(AUDIT / "proposed_split_mapping.csv")
low_res = pd.read_csv(AUDIT / "low_resolution_images.csv")
blurry = pd.read_csv(AUDIT / "potential_blurry_images.csv")
exact_dup = pd.read_csv(AUDIT / "exact_duplicates.csv")

key = ["split", "label", "filename"]

prop_map = proposed.set_index(key)["new_split"]
low_res_keys = set(zip(low_res["split"], low_res["label"], low_res["filename"]))
blur_keys = set(zip(blurry["split"], blurry["label"], blurry["filename"]))
dup_group_map = exact_dup.set_index(key)["md5"]

rows = []
for _, r in manifest.iterrows():
    k = (r["split"], r["label"], r["filename"])
    relative_path = f"{r['split']}/{r['label']}/{r['filename']}"
    image_id = relative_path.replace("/", "__")

    dup_group_id = dup_group_map.get(k, None)
    is_exact_duplicate = pd.notna(dup_group_id)

    rows.append({
        "image_id": image_id,
        "relative_path": relative_path,
        "filename": r["filename"],
        "label": r["label"],
        "patient_id": r["patient_id"],
        "original_split": r["split"],
        "proposed_grouped_split": prop_map.get(k, None),
        "width": r["width"],
        "height": r["height"],
        "mode": r["mode"],
        "filesize_bytes": r["filesize_bytes"],
        "md5": r["md5"],
        "phash": r["phash"],
        "blur_variance": r["blur_variance"],
        "corrupt": r["corrupt"],
        "is_low_resolution": k in low_res_keys,
        "is_blur_candidate": k in blur_keys,
        "exact_duplicate_group_id": dup_group_id,
        "is_exact_duplicate": is_exact_duplicate,
        "include_default": (r["corrupt"] == False),
        "source_manifest": "manifest.csv",
    })

df = pd.DataFrame(rows)

assert df["image_id"].is_unique, "image_id is not unique!"
assert len(df) == len(manifest) == 5856, f"row count mismatch: {len(df)}"

OUT = MANIFESTS / "master_manifest_derived.csv"
df.to_csv(OUT, index=False)
print(f"Wrote {len(df)} rows to {OUT}")
print("is_exact_duplicate count:", df["is_exact_duplicate"].sum())
print("is_low_resolution count:", df["is_low_resolution"].sum())
print("is_blur_candidate count:", df["is_blur_candidate"].sum())
print("proposed_grouped_split nulls:", df["proposed_grouped_split"].isna().sum())

# ---------------------------------------------------------------------------
# original_split_manifest.csv
# Reformats the ORIGINAL (unmodified) train/val/test split from manifest.csv.
# Patient overlap is intentionally PRESERVED here (this manifest exists as
# the uncorrected baseline for comparison experiments). Tagged as an
# all-images / sensitivity-only cohort - the deduplicated primary variant is
# original_split_deduplicated.csv (see create_deduplicated_manifest.py).
# ---------------------------------------------------------------------------
orig_rows = []
for _, r in manifest.iterrows():
    relative_path = f"{r['split']}/{r['label']}/{r['filename']}"
    orig_rows.append({
        "image_id": relative_path.replace("/", "__"),
        "relative_path": relative_path,
        "filename": r["filename"],
        "label": r["label"],
        "patient_id": r["patient_id"],
        "experimental_split": r["split"],
        "split_method": "original",
        "split_seed": "NA",
        "cohort": "all_images_5856",
        "analysis_role": "sensitivity_only",
    })
orig_df = pd.DataFrame(orig_rows)
assert len(orig_df) == 5856
ORIG_OUT = MANIFESTS / "original_split_manifest.csv"
orig_df.to_csv(ORIG_OUT, index=False)
orig_overlap = int((orig_df.groupby("patient_id")["experimental_split"].nunique() > 1).sum())
print(f"\nWrote {len(orig_df)} rows to {ORIG_OUT}")
print("original_split patient overlap (expected 264, preserved on purpose):", orig_overlap)

# ---------------------------------------------------------------------------
# patient_grouped_split_all_images.csv
# Built DIRECTLY from the frozen proposed_split_mapping.csv. The split itself
# is NOT recomputed - only relabeled (train_new->train, val_new->validation,
# test_new->test) and given the standard split-manifest schema. Tagged
# primary_eligible=false because a known exact-duplicate pair (MD5
# f9528b244ccd49f3d89ead90ba09c520, patients IM-0095/IM-0096) spans
# train/test in this manifest - see manifest_lineage.md.
# ---------------------------------------------------------------------------
SPLIT_RENAME = {"train_new": "train", "val_new": "validation", "test_new": "test"}
grouped_rows = []
for _, r in proposed.iterrows():
    relative_path = f"{r['split']}/{r['label']}/{r['filename']}"
    grouped_rows.append({
        "image_id": relative_path.replace("/", "__"),
        "relative_path": relative_path,
        "filename": r["filename"],
        "label": r["label"],
        "patient_id": r["patient_id"],
        "experimental_split": SPLIT_RENAME[r["new_split"]],
        "split_method": "patient_level_grouped_stratified_frozen",
        "split_seed": 42,
        "split_source": "proposed_split_mapping.csv",
        "cohort": "all_images_5856",
        "analysis_role": "sensitivity_only",
        "primary_eligible": False,
        "ineligibility_reason": "exact_duplicate_md5_spans_train_and_test",
    })
grouped_df = pd.DataFrame(grouped_rows)
assert len(grouped_df) == 5856
GROUPED_OUT = MANIFESTS / "patient_grouped_split_all_images.csv"
grouped_df.to_csv(GROUPED_OUT, index=False)
grouped_overlap = int((grouped_df.groupby("patient_id")["experimental_split"].nunique() > 1).sum())
counts = grouped_df["experimental_split"].value_counts()
print(f"\nWrote {len(grouped_df)} rows to {GROUPED_OUT}")
print("patient_grouped_split_all_images patient overlap (must be 0):", grouped_overlap)
print(counts)
assert grouped_overlap == 0, "FATAL: frozen grouped split shows nonzero patient overlap after relabel!"
assert counts.get("train", 0) == 4634 and counts.get("validation", 0) == 607 and counts.get("test", 0) == 615, \
    "FATAL: grouped split counts changed after relabel - must match audit exactly!"
