"""
create_deduplicated_manifest.py

Builds the GLOBAL exact-duplicate deduplication keep-set and the three
PRIMARY deduplicated manifests derived from it. No images are deleted or
moved on disk - everything here is manifest-level filtering.

Sections:
  B.  data/manifests/global_deduplication_keep_set.csv
      One row per image in the full 5,856-image manifest, recording a
      deterministic keep/exclude decision for every exact-duplicate (MD5)
      group. Rule: within each MD5 group, keep the file whose filename is
      lexicographically SMALLEST; exclude the rest. Near-duplicates
      (perceptual hash) are never used for exclusion.

  C1. data/manifests/original_split_deduplicated.csv
      Original train/val/test assignment, filtered to the keep-set.
      Patient overlap is preserved on purpose (uncorrected control).

  C2. data/manifests/image_stratified_split_deduplicated.csv
      A FRESH image-level stratified 80/10/10 split (seed 42), built only
      on the keep-set cohort - not a filter of the old all-images split
      (filtering would have skewed the 80/10/10 ratio).

  C3. data/manifests/patient_grouped_split_deduplicated.csv
      The frozen patient-grouped split (proposed_split_mapping.csv),
      filtered to the keep-set. StratifiedGroupKFold is NOT re-run.

Finally, a cross-check asserts that all three primary manifests share an
identical set of relative_path values (5,824 images) - only the split
assignment may differ between them.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_config as cfg

AUDIT = cfg.get_source_audit_dir()
MANIFESTS = cfg.get_manifests_dir()
REPORTS = cfg.get_reports_dir()

SEED = 42
COHORT_LABEL = "global_exact_deduplicated_5824"
DEDUP_RULE = "lexicographically_smallest_filename_within_md5_group"

# ===========================================================================
# B. Global deduplication keep-set
# ===========================================================================
master = pd.read_csv(MANIFESTS / "master_manifest_derived.csv")
exact_dup = pd.read_csv(AUDIT / "exact_duplicates.csv")
exact_dup["relative_path"] = exact_dup["split"] + "/" + exact_dup["label"] + "/" + exact_dup["filename"]

# representative (keep) per md5 group: lexicographically smallest filename
representative_by_md5 = {}
for md5, group in exact_dup.groupby("md5"):
    ordered = group.sort_values("filename")
    representative_by_md5[md5] = ordered.iloc[0]["relative_path"]

dup_group_relpaths = set(exact_dup["relative_path"])

keep_rows = []
for _, r in master.iterrows():
    rel = r["relative_path"]
    group_id = r["exact_duplicate_group_id"]
    is_dup = bool(r["is_exact_duplicate"])
    if not is_dup:
        keep_rows.append({
            "image_id": r["image_id"],
            "relative_path": rel,
            "filename": r["filename"],
            "patient_id": r["patient_id"],
            "label": r["label"],
            "md5": r["md5"],
            "exact_duplicate_group_id": None,
            "dedup_action": "keep",
            "dedup_reason": "unique_md5",
            "representative_relative_path": rel,
        })
    else:
        rep = representative_by_md5[group_id]
        if rel == rep:
            keep_rows.append({
                "image_id": r["image_id"],
                "relative_path": rel,
                "filename": r["filename"],
                "patient_id": r["patient_id"],
                "label": r["label"],
                "md5": r["md5"],
                "exact_duplicate_group_id": group_id,
                "dedup_action": "keep",
                "dedup_reason": f"representative_{DEDUP_RULE}",
                "representative_relative_path": rep,
            })
        else:
            keep_rows.append({
                "image_id": r["image_id"],
                "relative_path": rel,
                "filename": r["filename"],
                "patient_id": r["patient_id"],
                "label": r["label"],
                "md5": r["md5"],
                "exact_duplicate_group_id": group_id,
                "dedup_action": "exclude",
                "dedup_reason": f"non_representative_{DEDUP_RULE}",
                "representative_relative_path": rep,
            })

keep_df = pd.DataFrame(keep_rows)
assert len(keep_df) == 5856, f"expected 5856 decisions, got {len(keep_df)}"

n_keep = int((keep_df["dedup_action"] == "keep").sum())
n_exclude = int((keep_df["dedup_action"] == "exclude").sum())
assert n_keep == 5824, f"expected 5824 keep, got {n_keep}"
assert n_exclude == 32, f"expected 32 exclude, got {n_exclude}"

KEEPSET_OUT = MANIFESTS / "global_deduplication_keep_set.csv"
keep_df.to_csv(KEEPSET_OUT, index=False)
print(f"[B] Wrote {len(keep_df)} rows to {KEEPSET_OUT} (keep={n_keep}, exclude={n_exclude})")

# --- consistency check against the PRIOR dedup rule/result, if present ---
prior_report_path = REPORTS / "deduplication_report.csv"
if prior_report_path.is_file():
    prior_report = pd.read_csv(prior_report_path)
    prior_excluded = set(prior_report["excluded_relative_path"])
    new_excluded = set(keep_df.loc[keep_df["dedup_action"] == "exclude", "relative_path"])
    if prior_excluded != new_excluded:
        print("FATAL: new global keep-set exclusions differ from the prior "
              "patient_grouped_split_deduplicated.csv deduplication_report.csv!")
        print("Only in prior:", prior_excluded - new_excluded)
        print("Only in new:", new_excluded - prior_excluded)
        sys.exit(1)
    print("[B] Consistency check OK: exclusion set matches prior deduplication_report.csv exactly.")

keep_paths = set(keep_df.loc[keep_df["dedup_action"] == "keep", "relative_path"])
assert len(keep_paths) == 5824

# ===========================================================================
# C1. original_split_deduplicated.csv
# ===========================================================================
orig_all = pd.read_csv(MANIFESTS / "original_split_manifest.csv")
orig_dedup = orig_all[orig_all["relative_path"].isin(keep_paths)].copy()
orig_dedup["split_method"] = "original_deduplicated"
orig_dedup["split_seed"] = "NA"
orig_dedup["cohort"] = COHORT_LABEL
orig_dedup["analysis_role"] = "primary_control"
orig_dedup = orig_dedup[["image_id", "relative_path", "filename", "label", "patient_id",
                          "experimental_split", "split_method", "split_seed", "cohort", "analysis_role"]]

assert len(orig_dedup) == 5824, f"expected 5824 rows, got {len(orig_dedup)}"
C1_OUT = MANIFESTS / "original_split_deduplicated.csv"
orig_dedup.to_csv(C1_OUT, index=False)
c1_counts = orig_dedup["experimental_split"].value_counts()
c1_overlap = int((orig_dedup.groupby("patient_id")["experimental_split"].nunique() > 1).sum())
print(f"\n[C1] Wrote {len(orig_dedup)} rows to {C1_OUT}")
print("  split counts:", c1_counts.to_dict())
print("  patient overlap (preserved characteristic, not removed):", c1_overlap)

# ===========================================================================
# C2. image_stratified_split_deduplicated.csv  (FRESH split on 5,824 cohort)
# ===========================================================================
manifest_src = pd.read_csv(AUDIT / "manifest.csv")
manifest_src["relative_path"] = manifest_src["split"] + "/" + manifest_src["label"] + "/" + manifest_src["filename"]
manifest_src["image_id"] = manifest_src["relative_path"].str.replace("/", "__", regex=False)

cohort_df = manifest_src[manifest_src["relative_path"].isin(keep_paths)].copy()
assert len(cohort_df) == 5824, f"expected 5824 cohort rows, got {len(cohort_df)}"

trainval_df, test_df = train_test_split(
    cohort_df, test_size=0.10, stratify=cohort_df["label"], random_state=SEED
)
val_frac_of_remainder = 0.10 / 0.90
train_df, val_df = train_test_split(
    trainval_df, test_size=val_frac_of_remainder, stratify=trainval_df["label"], random_state=SEED
)
train_df = train_df.copy(); train_df["experimental_split"] = "train"
val_df = val_df.copy(); val_df["experimental_split"] = "validation"
test_df = test_df.copy(); test_df["experimental_split"] = "test"

img_dedup = pd.concat([train_df, val_df, test_df], ignore_index=True)
img_dedup["split_method"] = "image_level_stratified_deduplicated"
img_dedup["split_seed"] = SEED
img_dedup["cohort"] = COHORT_LABEL
img_dedup["analysis_role"] = "primary_control"
img_dedup = img_dedup[["image_id", "relative_path", "filename", "label", "patient_id",
                        "experimental_split", "split_method", "split_seed", "cohort", "analysis_role"]]

assert len(img_dedup) == 5824
assert img_dedup["image_id"].is_unique
C2_OUT = MANIFESTS / "image_stratified_split_deduplicated.csv"
img_dedup.to_csv(C2_OUT, index=False)
c2_counts = img_dedup["experimental_split"].value_counts()
c2_class = img_dedup.groupby(["experimental_split", "label"]).size().unstack(fill_value=0)
c2_overlap = int((img_dedup.groupby("patient_id")["experimental_split"].nunique() > 1).sum())
print(f"\n[C2] Wrote {len(img_dedup)} rows to {C2_OUT}")
print("  split counts:", c2_counts.to_dict())
print("  class balance:\n", c2_class)
print("  patient overlap (expected nonzero, measured not removed):", c2_overlap)

# ===========================================================================
# C3. patient_grouped_split_deduplicated.csv  (filter frozen proposed mapping)
# ===========================================================================
grouped_all = pd.read_csv(MANIFESTS / "patient_grouped_split_all_images.csv")
grp_dedup = grouped_all[grouped_all["relative_path"].isin(keep_paths)].copy()
grp_dedup["split_method"] = "patient_level_grouped_stratified_frozen_deduplicated"
grp_dedup["split_seed"] = 42
grp_dedup["cohort"] = COHORT_LABEL
grp_dedup["analysis_role"] = "primary"
grp_dedup = grp_dedup[["image_id", "relative_path", "filename", "label", "patient_id",
                        "experimental_split", "split_method", "split_seed", "cohort", "analysis_role"]]

assert len(grp_dedup) == 5824, f"expected 5824 rows, got {len(grp_dedup)}"
C3_OUT = MANIFESTS / "patient_grouped_split_deduplicated.csv"
grp_dedup.to_csv(C3_OUT, index=False)

c3_overlap = int((grp_dedup.groupby("patient_id")["experimental_split"].nunique() > 1).sum())
md5_map = manifest_src.set_index("relative_path")["md5"]
grp_dedup["_md5"] = grp_dedup["relative_path"].map(md5_map)
n_repeated_md5 = int(grp_dedup["_md5"].duplicated().sum())

exact_dup["_grp_split"] = exact_dup["relative_path"].map(
    grp_dedup.set_index("relative_path")["experimental_split"]
)
present_dup = exact_dup[exact_dup["relative_path"].isin(keep_paths)]
n_cross_split_groups = int((present_dup.groupby("md5")["_grp_split"].nunique() > 1).sum())

print(f"\n[C3] Wrote {len(grp_dedup)} rows to {C3_OUT}")
print("  split counts:", grp_dedup["experimental_split"].value_counts().to_dict())
print("  patient overlap (must be 0):", c3_overlap)
print("  repeated md5 (must be 0):", n_repeated_md5)
print("  cross-split exact-duplicate groups (must be 0):", n_cross_split_groups)

assert c3_overlap == 0, "FATAL: patient overlap nonzero in patient_grouped_split_deduplicated.csv!"
assert n_repeated_md5 == 0, "FATAL: repeated MD5 in patient_grouped_split_deduplicated.csv!"
assert n_cross_split_groups == 0, "FATAL: exact-duplicate group still spans splits after dedup!"

# ===========================================================================
# Cross-check: all three primary manifests share an identical relative_path set
# ===========================================================================
set1 = set(orig_dedup["relative_path"])
set2 = set(img_dedup["relative_path"])
set3 = set(grp_dedup["relative_path"])

assert set1 == set2 == set3, (
    "FATAL: the three primary deduplicated manifests do NOT share an identical "
    f"cohort! |set1-set2|={len(set1 - set2)} |set1-set3|={len(set1 - set3)} "
    f"|set2-set3|={len(set2 - set3)}"
)
print(f"\n[cross-check] OK: all three primary manifests share the identical "
      f"{len(set1)}-image cohort ({COHORT_LABEL}).")

# ===========================================================================
# deduplication_report.csv (kept for backward compatibility / human review)
# ===========================================================================
excl_df = keep_df[keep_df["dedup_action"] == "exclude"].rename(columns={
    "relative_path": "excluded_relative_path",
})[["excluded_relative_path", "md5", "exact_duplicate_group_id", "representative_relative_path", "dedup_reason"]]
excl_df["rule"] = DEDUP_RULE
excl_df.to_csv(REPORTS / "deduplication_report.csv", index=False)
print(f"\nWrote {len(excl_df)} rows to {REPORTS / 'deduplication_report.csv'}")
