"""
build_controlled_manifests_sensitivity_dedup.py

Sensitivity-dedup (protocol section 28, the "Experiment B" of the original
protocol version - NOT this project's "Experiment B" used for robustness/E).
Purpose: test whether the Experiment A leakage conclusion (patient-grouped
clean vs leaky) is robust to the Stage 1 deduplication decision (32
exact-duplicate images removed from the main cohort, 5,824 -> full cohort
5,856).

Follows the structure & logic of build_controlled_manifests.py (Phase 1,
protocol 6.2-6.6) IDENTICALLY, but sources from
`patient_grouped_split_all_images.csv` (5,856 images, the same FROZEN
patient-grouped split - proposed_split_mapping.csv, seed 42 - already
prepared in Stage 1 specifically for this sensitivity-dedup purpose). Does
NOT re-run StratifiedGroupKFold, does NOT recompute patient_id/MD5.

Important finding (documented, DELIBERATELY left in place - user decision
2026-08-02): one exact-duplicate pair (NORMAL2-IM-0095-0001.jpeg /
NORMAL2-IM-0096-0001.jpeg, identical MD5) was assigned DIFFERENT patient_ids
in the source data (IM-0095 in test, IM-0096 in train) - this is the
original finding that motivated the Stage 1 dedup decision. The non-dedup
cohort here DELIBERATELY includes it as-is (not just the patient-level
leakage intentionally varied as in the Leaky condition, but a real
image-level MD5 collision between train and test) so this sensitivity check
answers the real question: whether this one real leak out of 5,856 images is
enough to change the Experiment A conclusion. See
validate_controlled_manifests_sensitivity_dedup.py for the explicit allowlist
of this one expected MD5 collision (and ONLY this one) - the QC gate still
FAILs on any other unexpected MD5 collision.

Output (data/manifests/):
    controlled_anchor_test_nondedup.csv
    controlled_clean_train_nondedup.csv
    controlled_clean_validation_nondedup.csv
    controlled_leaky_train_nondedup.csv
    controlled_leakage_replacement_map_nondedup.csv
    sensitivity_dedup_known_md5_exceptions.csv   (dokumentasi expected MD5 collision)
"""
import sys
import random
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_config as cfg

MANIFESTS = cfg.get_manifests_dir()
AUDIT = cfg.get_source_audit_dir()
REPORTS = cfg.get_reports_dir()

SEED = 42
COHORT_LABEL = "global_all_images_5856_sensitivity_dedup"

grouped = pd.read_csv(MANIFESTS / "patient_grouped_split_all_images.csv")
assert len(grouped) == 5856, f"expected 5856 rows in patient_grouped_split_all_images.csv, got {len(grouped)}"

test_df = grouped[grouped["experimental_split"] == "test"].copy()
train_df = grouped[grouped["experimental_split"] == "train"].copy()
val_df = grouped[grouped["experimental_split"] == "validation"].copy()

assert test_df["patient_id"].nunique() == 279, (
    f"FATAL: expected 279 test patients (identical to the dedup cohort - the patient-grouped "
    f"split does not change, only additional images for existing patients), got {test_df['patient_id'].nunique()}"
)
print(f"[cohort] rows total={len(grouped)} train={len(train_df)} validation={len(val_df)} "
      f"test={len(test_df)} test_patients={test_df['patient_id'].nunique()}")

# ===========================================================================
# 6.2 Common anchor test set: 1 image/patient, lexicographically first
#     (logic IDENTICAL to Phase 1 - not independently regenerated, so the
#     anchor test stays the SAME 279 patients; the per-patient image
#     composition can grow since the cohort is larger, so the anchor image
#     for some patients CAN differ from the dedup cohort if a new image sorts
#     earlier lexicographically - explicitly documented below, not hidden)
# ===========================================================================
anchor_rows = []
sibling_rows = []

for pid, g in test_df.groupby("patient_id"):
    g_sorted = g.sort_values("relative_path").reset_index(drop=True)
    anchor_rows.append(g_sorted.iloc[0])
    if len(g_sorted) > 1:
        sibling_rows.append(g_sorted.iloc[1])

anchor_df = pd.DataFrame(anchor_rows).reset_index(drop=True)
sibling_df = pd.DataFrame(sibling_rows).reset_index(drop=True)

assert len(anchor_df) == 279 and anchor_df["patient_id"].is_unique

K = len(sibling_df)
assert sibling_df["patient_id"].is_unique
assert set(sibling_df["patient_id"]) <= set(anchor_df["patient_id"])

# cross-check vs the dedup-cohort anchor set: report how many patients have a
# DIFFERENT anchor image (because an additional image sorts earlier lexicographically)
dedup_anchor = pd.read_csv(MANIFESTS / "controlled_anchor_test.csv")
dedup_anchor_map = dedup_anchor.set_index("patient_id")["relative_path"].to_dict()
changed_anchor = [
    pid for pid, path in zip(anchor_df["patient_id"], anchor_df["relative_path"])
    if dedup_anchor_map.get(pid) != path
]
print(f"[6.2] Anchor test (non-dedup): {len(anchor_df)} images, {anchor_df['patient_id'].nunique()} patients")
print(f"      class distribution: {anchor_df['label'].value_counts().to_dict()}")
print(f"      anchor image DIFFERENT from dedup-cohort anchor for {len(changed_anchor)} patient(s): "
      f"{changed_anchor}")
print(f"[6.3] Sibling set K={K} (dedup cohort K=146), class distribution: "
      f"{sibling_df['label'].value_counts().to_dict()}")

# ===========================================================================
# 6.4 Condition A - Clean patient-grouped
# ===========================================================================
test_patients = set(test_df["patient_id"])
leaking_in_train = train_df[train_df["patient_id"].isin(test_patients)]
leaking_in_val = val_df[val_df["patient_id"].isin(test_patients)]
assert len(leaking_in_train) == 0, "FATAL: test patient images found in train (clean)"
assert len(leaking_in_val) == 0, "FATAL: test patient images found in validation (clean)"

clean_train_df = train_df.copy()
clean_val_df = val_df.copy()

overlap_train_val = set(clean_train_df["patient_id"]) & set(clean_val_df["patient_id"])
overlap_train_test = set(clean_train_df["patient_id"]) & set(anchor_df["patient_id"])
overlap_val_test = set(clean_val_df["patient_id"]) & set(anchor_df["patient_id"])
assert len(overlap_train_val) == 0
assert len(overlap_train_test) == 0
assert len(overlap_val_test) == 0

manifest_src = pd.read_csv(AUDIT / "manifest.csv")
manifest_src["relative_path"] = manifest_src["split"] + "/" + manifest_src["label"] + "/" + manifest_src["filename"]
md5_map = manifest_src.set_index("relative_path")["md5"]

clean_train_md5 = clean_train_df["relative_path"].map(md5_map)
n_dup_clean = int(clean_train_md5.duplicated().sum())
print(f"[6.4] Clean train: {len(clean_train_df)} images, {clean_train_df['patient_id'].nunique()} patients "
      f"(repeated MD5 within clean train={n_dup_clean})")
print(f"      Clean validation: {len(clean_val_df)} images, {clean_val_df['patient_id'].nunique()} patients")
print(f"      patient_overlap(train,val)=0 patient_overlap(train,test)=0 patient_overlap(val,test)=0 "
      f"-- all patient-level guarantees still OK (the difference vs the dedup cohort is ONLY at the MD5/image level)")

# ===========================================================================
# 6.5 Condition B - Controlled patient leakage (logic identical to Phase 1)
# ===========================================================================
rng = random.Random(SEED)

sib_counts_by_class = sibling_df["label"].value_counts().to_dict()
excluded_patient_pool = set(clean_val_df["patient_id"]) | set(anchor_df["patient_id"])

replacement_rows = []
excluded_relpaths = set()

for label, n_needed in sib_counts_by_class.items():
    class_train = clean_train_df[
        (clean_train_df["label"] == label) & (~clean_train_df["patient_id"].isin(excluded_patient_pool))
    ]
    per_patient_first = (
        class_train.sort_values("relative_path")
        .groupby("patient_id", as_index=False)
        .first()
    )
    candidate_patients = per_patient_first["patient_id"].tolist()
    rng.shuffle(candidate_patients)

    if len(candidate_patients) < n_needed:
        raise RuntimeError(
            f"FATAL: not enough distinct training patients of class {label} "
            f"({len(candidate_patients)}) to remove {n_needed} images."
        )

    chosen_patients = candidate_patients[:n_needed]
    chosen_rows = per_patient_first[per_patient_first["patient_id"].isin(chosen_patients)]
    excluded_relpaths.update(chosen_rows["relative_path"].tolist())

    print(f"[6.5] class={label}: need {n_needed} replacements, "
          f"{len(candidate_patients)} eligible train patients available, "
          f"{len(chosen_rows)} chosen (1/patient)")

assert len(excluded_relpaths) == K, f"FATAL: replacement pool size {len(excluded_relpaths)} != K={K}"

sibling_sorted = sibling_df.sort_values(["label", "relative_path"]).reset_index(drop=True)
excluded_df = clean_train_df[clean_train_df["relative_path"].isin(excluded_relpaths)].copy()
excluded_sorted = excluded_df.sort_values(["label", "relative_path"]).reset_index(drop=True)

replacement_map_rows = []
for label in sib_counts_by_class:
    sib_l = sibling_sorted[sibling_sorted["label"] == label].reset_index(drop=True)
    exc_l = excluded_sorted[excluded_sorted["label"] == label].reset_index(drop=True)
    assert len(sib_l) == len(exc_l)
    for i in range(len(sib_l)):
        replacement_map_rows.append({
            "label": label,
            "sibling_relative_path": sib_l.loc[i, "relative_path"],
            "sibling_patient_id": sib_l.loc[i, "patient_id"],
            "sibling_image_id": sib_l.loc[i, "image_id"],
            "replaced_relative_path": exc_l.loc[i, "relative_path"],
            "replaced_patient_id": exc_l.loc[i, "patient_id"],
            "replaced_image_id": exc_l.loc[i, "image_id"],
            "selection_rule": "seed42_shuffle_of_train_patients_max1image_per_patient",
            "selection_seed": SEED,
        })

replacement_map_df = pd.DataFrame(replacement_map_rows)
assert len(replacement_map_df) == K

leaky_train_df = pd.concat(
    [clean_train_df[~clean_train_df["relative_path"].isin(excluded_relpaths)], sibling_df],
    ignore_index=True,
)

assert len(leaky_train_df) == len(clean_train_df)
leaky_class = leaky_train_df["label"].value_counts()
clean_class = clean_train_df["label"].value_counts()
print(f"[6.5] Leaky train: {len(leaky_train_df)} images (== clean train)")
print(f"      class clean : {clean_class.to_dict()}")
print(f"      class leaky : {leaky_class.to_dict()}")
assert (clean_class.sort_index() == leaky_class.sort_index()).all()

leaky_overlap_patients = set(leaky_train_df["patient_id"]) & set(anchor_df["patient_id"])
assert leaky_overlap_patients == set(sibling_df["patient_id"])
print(f"[6.5] Leaky train patient-overlap with anchor test = {len(leaky_overlap_patients)} "
      f"(must equal K={K}) -- OK")

# ===========================================================================
# Document the known MD5 exception (IM-0095/IM-0096) - check where it sits
# relative to anchor/sibling/clean_train/leaky_train so the QC gate knows
# exactly which exception is allowlisted.
# ===========================================================================
# NB: relative_path uses the original Kaggle SOURCE folder prefix (e.g. "test/..."),
# NOT the experimental_split resulting from patient-grouping - both images
# originate from the same source folder "test/NORMAL/" but are assigned
# DIFFERENT experimental_split values (IM-0095 -> test, IM-0096 -> train)
# because their patient_id differs.
known_pair = {"a": "test/NORMAL/NORMAL2-IM-0095-0001.jpeg", "b": "test/NORMAL/NORMAL2-IM-0096-0001.jpeg"}

def locate(path, dfs):
    hits = []
    for name, df in dfs.items():
        if path in set(df["relative_path"]):
            hits.append(name)
    return hits

dfs_map = {
    "anchor_test": anchor_df, "sibling(not-in-train)": sibling_df,
    "clean_train": clean_train_df, "clean_validation": clean_val_df,
    "leaky_train": leaky_train_df,
}
loc_a = locate(known_pair["a"], dfs_map)
loc_b = locate(known_pair["b"], dfs_map)
print(f"[known-exception] {known_pair['a']} found in: {loc_a}")
print(f"[known-exception] {known_pair['b']} found in: {loc_b}")

known_exceptions_df = pd.DataFrame([{
    "relative_path_1": known_pair["a"], "found_in_1": ";".join(loc_a),
    "relative_path_2": known_pair["b"], "found_in_2": ";".join(loc_b),
    "md5": md5_map.get(known_pair["a"]),
    "reason": "exact_duplicate_md5_identical_but_source_patient_id_differs_(IM-0095_test_vs_IM-0096_train); "
              "deliberately included in the non-dedup cohort per user decision 2026-08-02 - THIS IS EXACTLY WHAT IS BEING TESTED",
}])

# ===========================================================================
# 6.6 Write the five manifests (+ known-exception documentation)
# ===========================================================================
COLS = ["image_id", "relative_path", "filename", "label", "patient_id",
        "experimental_split", "condition", "cohort", "selection_rule", "selection_seed"]


def finalize(df, experimental_split, condition, selection_rule, selection_seed):
    out = df[["image_id", "relative_path", "filename", "label", "patient_id"]].copy()
    out["experimental_split"] = experimental_split
    out["condition"] = condition
    out["cohort"] = COHORT_LABEL
    out["selection_rule"] = selection_rule
    out["selection_seed"] = selection_seed
    return out[COLS]


anchor_out = finalize(anchor_df, "test", "shared_anchor_test",
                       "lexicographic_first_relative_path_per_test_patient_nondedup_cohort", "NA")
clean_train_out = finalize(clean_train_df, "train", "clean",
                            "verbatim_train_subset_of_patient_grouped_split_all_images", "NA")
clean_val_out = finalize(clean_val_df, "validation", "clean",
                          "verbatim_validation_subset_of_patient_grouped_split_all_images", "NA")
leaky_train_out = finalize(leaky_train_df, "train", "leaky",
                            "clean_train_plus_one_sibling_per_eligible_test_patient_minus_seed42_replacement_nondedup",
                            SEED)

anchor_out.to_csv(MANIFESTS / "controlled_anchor_test_nondedup.csv", index=False)
clean_train_out.to_csv(MANIFESTS / "controlled_clean_train_nondedup.csv", index=False)
clean_val_out.to_csv(MANIFESTS / "controlled_clean_validation_nondedup.csv", index=False)
leaky_train_out.to_csv(MANIFESTS / "controlled_leaky_train_nondedup.csv", index=False)
replacement_map_df.to_csv(MANIFESTS / "controlled_leakage_replacement_map_nondedup.csv", index=False)
known_exceptions_df.to_csv(REPORTS / "sensitivity_dedup_known_md5_exceptions.csv", index=False)

print("\nWrote:")
for f in ["controlled_anchor_test_nondedup.csv", "controlled_clean_train_nondedup.csv",
          "controlled_clean_validation_nondedup.csv", "controlled_leaky_train_nondedup.csv",
          "controlled_leakage_replacement_map_nondedup.csv"]:
    print(" -", MANIFESTS / f)
print(" -", REPORTS / "sensitivity_dedup_known_md5_exceptions.csv")
