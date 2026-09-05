"""
create_image_control_split.py

Creates the image-level stratified 80/10/10 control split (train/validation/
test) on the FULL all-images cohort (5,856 images), stratified by label only
(NOT grouped by patient_id). This is the all-images / sensitivity-only
variant; the primary (deduplicated-cohort) counterpart is built separately
by create_deduplicated_manifest.py as image_stratified_split_deduplicated.csv.

Patient overlap is measured and reported here, never removed - it is the
expected behavior of an image-level (non-grouped) split.

Seed: 42 (fixed, per project rules).
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_config as cfg

AUDIT = cfg.get_source_audit_dir()
MANIFESTS = cfg.get_manifests_dir()
OUT = MANIFESTS / "image_stratified_split_manifest.csv"

SEED = 42

manifest = pd.read_csv(AUDIT / "manifest.csv")
manifest["relative_path"] = manifest["split"] + "/" + manifest["label"] + "/" + manifest["filename"]
manifest["image_id"] = manifest["relative_path"].str.replace("/", "__", regex=False)

trainval_df, test_df = train_test_split(
    manifest, test_size=0.10, stratify=manifest["label"], random_state=SEED
)
val_frac_of_remainder = 0.10 / 0.90
train_df, val_df = train_test_split(
    trainval_df, test_size=val_frac_of_remainder, stratify=trainval_df["label"], random_state=SEED
)

train_df = train_df.copy(); train_df["experimental_split"] = "train"
val_df = val_df.copy(); val_df["experimental_split"] = "validation"
test_df = test_df.copy(); test_df["experimental_split"] = "test"

full = pd.concat([train_df, val_df, test_df], ignore_index=True)
full["split_method"] = "image_level_stratified"
full["split_seed"] = SEED
full["cohort"] = "all_images_5856"
full["analysis_role"] = "sensitivity_only"

out_cols = ["image_id", "relative_path", "filename", "label", "patient_id",
            "experimental_split", "split_method", "split_seed", "cohort", "analysis_role"]
out_df = full[out_cols].copy()

assert len(out_df) == 5856, f"expected 5856 rows, got {len(out_df)}"
assert out_df["image_id"].is_unique
assert out_df.groupby("image_id")["experimental_split"].apply(lambda s: s.nunique() == 1).all()

out_df.to_csv(OUT, index=False)

counts = out_df["experimental_split"].value_counts()
class_counts = out_df.groupby(["experimental_split", "label"]).size().unstack(fill_value=0)
overlap = out_df.groupby("patient_id")["experimental_split"].nunique()
n_overlap_patients = int((overlap > 1).sum())

print(f"Wrote {len(out_df)} rows to {OUT}")
print("\nSplit sizes:\n", counts)
print("\nClass balance per split:\n", class_counts)
pct_pneumonia = (class_counts["PNEUMONIA"] / class_counts.sum(axis=1) * 100).round(2)
print("\n% PNEUMONIA per split:\n", pct_pneumonia)
print(f"\nPatient overlap across image-level splits (EXPECTED, not removed): {n_overlap_patients} patients")
print(f"Total unique patients touched: {out_df['patient_id'].nunique()}")
