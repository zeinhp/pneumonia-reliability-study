"""
build_controlled_manifests.py

Phase 1 (Experiment A protocol, section 6.2-6.6): builds the five controlled
manifests for the main analysis (Clean vs Controlled Leaky) on top of the
deduplicated cohort (5,824 images), using the official patient-grouped split
(patient_grouped_split_deduplicated.csv) as the basis. Does not re-run
StratifiedGroupKFold, does not recompute patient_id/MD5/phash, and does not
touch any physical image files.

Output (data/manifests/):
    controlled_anchor_test.csv
    controlled_clean_train.csv
    controlled_clean_validation.csv
    controlled_leaky_train.csv
    controlled_leakage_replacement_map.csv
"""
import sys
import random
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_config as cfg

MANIFESTS = cfg.get_manifests_dir()
AUDIT = cfg.get_source_audit_dir()

SEED = 42
COHORT_LABEL = "global_exact_deduplicated_5824"

grouped = pd.read_csv(MANIFESTS / "patient_grouped_split_deduplicated.csv")
assert len(grouped) == 5824, f"expected 5824 rows in patient_grouped_split_deduplicated.csv, got {len(grouped)}"

test_df = grouped[grouped["experimental_split"] == "test"].copy()
train_df = grouped[grouped["experimental_split"] == "train"].copy()
val_df = grouped[grouped["experimental_split"] == "validation"].copy()

assert len(test_df) == 614 and test_df["patient_id"].nunique() == 279
assert len(train_df) == 4607
assert len(val_df) == 603

# ===========================================================================
# 6.2 Common anchor test set: 1 image/patient, lexicographically first
# ===========================================================================
anchor_rows = []
sibling_rows = []  # chosen sibling per eligible patient (K patients)

for pid, g in test_df.groupby("patient_id"):
    g_sorted = g.sort_values("relative_path").reset_index(drop=True)
    anchor_rows.append(g_sorted.iloc[0])
    if len(g_sorted) > 1:
        sibling_rows.append(g_sorted.iloc[1])  # first sibling after anchor

anchor_df = pd.DataFrame(anchor_rows).reset_index(drop=True)
sibling_df = pd.DataFrame(sibling_rows).reset_index(drop=True)

assert len(anchor_df) == 279, f"expected 279 anchor images, got {len(anchor_df)}"
assert anchor_df["patient_id"].nunique() == 279
assert anchor_df["patient_id"].is_unique

K = len(sibling_df)
assert K == 146, f"expected K=146 sibling-eligible patients, got {K}"
assert sibling_df["patient_id"].is_unique  # max 1 sibling per patient
assert set(sibling_df["patient_id"]) <= set(anchor_df["patient_id"]), \
    "FATAL: sibling patient(s) not found among anchor test patients"

anchor_class_counts = anchor_df["label"].value_counts()
sibling_class_counts = sibling_df["label"].value_counts()
print(f"[6.2] Anchor test: {len(anchor_df)} images, {anchor_df['patient_id'].nunique()} patients")
print(f"      class distribution: {anchor_class_counts.to_dict()}")
print(f"[6.3] Sibling set K={K}, class distribution: {sibling_class_counts.to_dict()}")

# ===========================================================================
# 6.4 Condition A - Clean patient-grouped
# ===========================================================================
# sanity: no test-patient images already present in train/validation (guaranteed
# by the patient-grouped split, verified explicitly here per protocol wording)
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
assert len(overlap_train_val) == 0, f"FATAL: {len(overlap_train_val)} patients overlap train/validation"
assert len(overlap_train_test) == 0, f"FATAL: {len(overlap_train_test)} patients overlap train/anchor_test"
assert len(overlap_val_test) == 0, f"FATAL: {len(overlap_val_test)} patients overlap validation/anchor_test"

manifest_src = pd.read_csv(AUDIT / "manifest.csv")
manifest_src["relative_path"] = manifest_src["split"] + "/" + manifest_src["label"] + "/" + manifest_src["filename"]
md5_map = manifest_src.set_index("relative_path")["md5"]
clean_train_md5 = clean_train_df["relative_path"].map(md5_map)
assert clean_train_md5.duplicated().sum() == 0, "FATAL: repeated MD5 in clean train"

print(f"[6.4] Clean train: {len(clean_train_df)} images, {clean_train_df['patient_id'].nunique()} patients")
print(f"      Clean validation: {len(clean_val_df)} images, {clean_val_df['patient_id'].nunique()} patients")
print(f"      patient_overlap(train,val)={len(overlap_train_val)} "
      f"patient_overlap(train,test)={len(overlap_train_test)} "
      f"patient_overlap(val,test)={len(overlap_val_test)} repeated_md5=0 -- all guarantees OK")

# ===========================================================================
# 6.5 Condition B - Controlled patient leakage
# ===========================================================================
rng = random.Random(SEED)

# how many sibling images per class must be inserted
sib_counts_by_class = sibling_df["label"].value_counts().to_dict()

# candidate pool for replacement: clean_train images, grouped by patient, per class,
# NEVER touching validation/test patients (already disjoint, but filter defensively).
excluded_patient_pool = set(clean_val_df["patient_id"]) | set(anchor_df["patient_id"])

replacement_rows = []
excluded_relpaths = set()

for label, n_needed in sib_counts_by_class.items():
    class_train = clean_train_df[
        (clean_train_df["label"] == label) & (~clean_train_df["patient_id"].isin(excluded_patient_pool))
    ]
    # group by patient, keep at most 1 candidate image per patient (the
    # lexicographically-first image of that patient) so that a single pass
    # removes at most one image per training patient ("as far as possible,
    # at most one image per training patient")
    per_patient_first = (
        class_train.sort_values("relative_path")
        .groupby("patient_id", as_index=False)
        .first()
    )
    candidate_patients = per_patient_first["patient_id"].tolist()
    rng.shuffle(candidate_patients)  # seed 42, deterministic

    if len(candidate_patients) < n_needed:
        raise RuntimeError(
            f"FATAL: not enough distinct training patients of class {label} "
            f"({len(candidate_patients)}) to remove {n_needed} images while keeping "
            f"max 1 removal/patient."
        )

    chosen_patients = candidate_patients[:n_needed]
    chosen_rows = per_patient_first[per_patient_first["patient_id"].isin(chosen_patients)]
    excluded_relpaths.update(chosen_rows["relative_path"].tolist())

    print(f"[6.5] class={label}: need {n_needed} replacements, "
          f"{len(candidate_patients)} eligible train patients available, "
          f"{len(chosen_rows)} chosen (1/patient)")

assert len(excluded_relpaths) == K, \
    f"FATAL: replacement pool size {len(excluded_relpaths)} != K={K}"

# pair siblings with replaced images (both sorted deterministically for a stable,
# reproducible pairing within each class)
sibling_sorted = sibling_df.sort_values(["label", "relative_path"]).reset_index(drop=True)
excluded_df = clean_train_df[clean_train_df["relative_path"].isin(excluded_relpaths)].copy()
excluded_sorted = excluded_df.sort_values(["label", "relative_path"]).reset_index(drop=True)

replacement_map_rows = []
for label in sib_counts_by_class:
    sib_l = sibling_sorted[sibling_sorted["label"] == label].reset_index(drop=True)
    exc_l = excluded_sorted[excluded_sorted["label"] == label].reset_index(drop=True)
    assert len(sib_l) == len(exc_l), f"FATAL: mismatched pairing count for label={label}"
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

assert len(leaky_train_df) == len(clean_train_df), (
    f"FATAL: n_train_leaky ({len(leaky_train_df)}) != n_train_clean ({len(clean_train_df)})"
)
leaky_class = leaky_train_df["label"].value_counts()
clean_class = clean_train_df["label"].value_counts()
print(f"[6.5] Leaky train: {len(leaky_train_df)} images (== clean train)")
print(f"      class clean : {clean_class.to_dict()}")
print(f"      class leaky : {leaky_class.to_dict()}")
assert (clean_class.sort_index() == leaky_class.sort_index()).all(), \
    "FATAL: class distribution changed between clean and leaky train"

leaky_train_md5 = leaky_train_df["relative_path"].map(md5_map)
assert leaky_train_md5.duplicated().sum() == 0, "FATAL: repeated MD5 in leaky train"

# planned overlap: leaky train must overlap anchor-test patients EXACTLY on the
# K sibling-contributing patients, nothing more/less
leaky_overlap_patients = set(leaky_train_df["patient_id"]) & set(anchor_df["patient_id"])
assert leaky_overlap_patients == set(sibling_df["patient_id"]), (
    "FATAL: leaky train patient-overlap with anchor test does not exactly match "
    "the planned K sibling patients"
)
print(f"[6.5] Leaky train patient-overlap with anchor test = {len(leaky_overlap_patients)} "
      f"(must equal K={K}) -- OK, matches planned contamination map exactly")

# ===========================================================================
# 6.6 Write the five manifests
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
                       "lexicographic_first_relative_path_per_test_patient", "NA")
clean_train_out = finalize(clean_train_df, "train", "clean",
                            "verbatim_train_subset_of_patient_grouped_split_deduplicated", "NA")
clean_val_out = finalize(clean_val_df, "validation", "clean",
                          "verbatim_validation_subset_of_patient_grouped_split_deduplicated", "NA")
leaky_train_out = finalize(leaky_train_df, "train", "leaky",
                            "clean_train_plus_one_sibling_per_eligible_test_patient_minus_seed42_replacement",
                            SEED)

anchor_out.to_csv(MANIFESTS / "controlled_anchor_test.csv", index=False)
clean_train_out.to_csv(MANIFESTS / "controlled_clean_train.csv", index=False)
clean_val_out.to_csv(MANIFESTS / "controlled_clean_validation.csv", index=False)
leaky_train_out.to_csv(MANIFESTS / "controlled_leaky_train.csv", index=False)
replacement_map_df.to_csv(MANIFESTS / "controlled_leakage_replacement_map.csv", index=False)

print("\nWrote:")
for f in ["controlled_anchor_test.csv", "controlled_clean_train.csv",
          "controlled_clean_validation.csv", "controlled_leaky_train.csv",
          "controlled_leakage_replacement_map.csv"]:
    print(" -", MANIFESTS / f)
