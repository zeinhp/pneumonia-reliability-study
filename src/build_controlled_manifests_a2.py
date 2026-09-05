"""
build_controlled_manifests_a2.py

Experiment A2 (varying-draw leakage replication, CLAUDE.md #6b point 2):
builds 5 variants of the `leaky_train` manifest that differ in exactly ONE
variable - the seed used to choose which clean-train images are EXCLUDED to
offset size/class balance when the sibling is inserted (protocol 6.5). In the
original Experiment A, this exclusion seed was ALWAYS 42 across all 30 runs
(2 conditions x 3 architectures x 5 model-seeds) - a single fixed "draw".
This limitation was noted in the Experiment A report (the leakage-effect CI
does not capture uncertainty in the contamination pattern, only training
noise).

A2 varies the exclusion seed IN LOCKSTEP with the existing 5 model-seeds
(draw_seed = model_seed): 42, 123, 456, 789, 2026. The sibling inserted into
leaky (K=146 patients, 1 sibling/patient, chosen as the lexicographically
first image after the anchor - protocol 6.3) does NOT change - it stays
deterministic per patient, per the user's decision (only the
exclusion/replacement seed is varied, not the sibling selection).

The Clean, validation, and anchor test conditions do NOT change at all -
identical to Experiment A (re-read from the existing
controlled_clean_train.csv, controlled_clean_validation.csv,
controlled_anchor_test.csv). Only leaky_train has 5 variants.

Important sanity check: the draw_seed=42 variant MUST be identical to
controlled_leaky_train.csv (the original Experiment A manifest) - otherwise
there is a bug.

Status: EXPLORATORY (protocol §24) - the anchor test has been open since
Experiment A Phase 5. This analysis does NOT change the Experiment A
confirmatory conclusions; it tests the single-draw limitation noted in the
Experiment A report.

Output (data/manifests/):
    controlled_leaky_train_a2_seed{S}.csv        for S in [42,123,456,789,2026]
    controlled_leakage_replacement_map_a2_seed{S}.csv
"""
import sys
import random
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_config as cfg

MANIFESTS = cfg.get_manifests_dir()
AUDIT = cfg.get_source_audit_dir()

DRAW_SEEDS = [42, 123, 456, 789, 2026]
COHORT_LABEL = "global_exact_deduplicated_5824"

# ---------------------------------------------------------------------------
# Reload the components that do NOT change across draws (identical to Experiment A)
# ---------------------------------------------------------------------------
clean_train_df = pd.read_csv(MANIFESTS / "controlled_clean_train.csv")
clean_val_df = pd.read_csv(MANIFESTS / "controlled_clean_validation.csv")
anchor_df = pd.read_csv(MANIFESTS / "controlled_anchor_test.csv")

assert len(clean_train_df) == 4607, f"clean_train must be 4607, got {len(clean_train_df)}"
assert len(clean_val_df) == 603, f"clean_val must be 603, got {len(clean_val_df)}"
assert len(anchor_df) == 279 and anchor_df["patient_id"].nunique() == 279

# reconstruct sibling_df using exactly the protocol 6.3 logic from the original test_df (dedup cohort)
grouped = pd.read_csv(MANIFESTS / "patient_grouped_split_deduplicated.csv")
assert len(grouped) == 5824
test_df = grouped[grouped["experimental_split"] == "test"].copy()

sibling_rows = []
for pid, g in test_df.groupby("patient_id"):
    g_sorted = g.sort_values("relative_path").reset_index(drop=True)
    if len(g_sorted) > 1:
        sibling_rows.append(g_sorted.iloc[1])
sibling_df = pd.DataFrame(sibling_rows).reset_index(drop=True)

K = len(sibling_df)
assert K == 146, f"expected K=146, got {K}"
assert sibling_df["patient_id"].is_unique
assert set(sibling_df["patient_id"]) <= set(anchor_df["patient_id"])

manifest_src = pd.read_csv(AUDIT / "manifest.csv")
manifest_src["relative_path"] = manifest_src["split"] + "/" + manifest_src["label"] + "/" + manifest_src["filename"]
md5_map = manifest_src.set_index("relative_path")["md5"]

sib_counts_by_class = sibling_df["label"].value_counts().to_dict()
excluded_patient_pool = set(clean_val_df["patient_id"]) | set(anchor_df["patient_id"])

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


def build_draw(draw_seed: int):
    rng = random.Random(draw_seed)
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
                f"FATAL (draw_seed={draw_seed}): not enough distinct training patients of "
                f"class {label} ({len(candidate_patients)}) to remove {n_needed} images."
            )

        chosen_patients = candidate_patients[:n_needed]
        chosen_rows = per_patient_first[per_patient_first["patient_id"].isin(chosen_patients)]
        excluded_relpaths.update(chosen_rows["relative_path"].tolist())

    assert len(excluded_relpaths) == K, \
        f"FATAL (draw_seed={draw_seed}): replacement pool size {len(excluded_relpaths)} != K={K}"

    sibling_sorted = sibling_df.sort_values(["label", "relative_path"]).reset_index(drop=True)
    excluded_df = clean_train_df[clean_train_df["relative_path"].isin(excluded_relpaths)].copy()
    excluded_sorted = excluded_df.sort_values(["label", "relative_path"]).reset_index(drop=True)

    replacement_map_rows = []
    for label in sib_counts_by_class:
        sib_l = sibling_sorted[sibling_sorted["label"] == label].reset_index(drop=True)
        exc_l = excluded_sorted[excluded_sorted["label"] == label].reset_index(drop=True)
        assert len(sib_l) == len(exc_l), f"FATAL (draw_seed={draw_seed}): mismatched pairing for label={label}"
        for i in range(len(sib_l)):
            replacement_map_rows.append({
                "label": label,
                "sibling_relative_path": sib_l.loc[i, "relative_path"],
                "sibling_patient_id": sib_l.loc[i, "patient_id"],
                "sibling_image_id": sib_l.loc[i, "image_id"],
                "replaced_relative_path": exc_l.loc[i, "relative_path"],
                "replaced_patient_id": exc_l.loc[i, "patient_id"],
                "replaced_image_id": exc_l.loc[i, "image_id"],
                "selection_rule": "seedDRAW_shuffle_of_train_patients_max1image_per_patient",
                "selection_seed": draw_seed,
            })
    replacement_map_df = pd.DataFrame(replacement_map_rows)
    assert len(replacement_map_df) == K

    leaky_train_df = pd.concat(
        [clean_train_df[~clean_train_df["relative_path"].isin(excluded_relpaths)], sibling_df],
        ignore_index=True,
    )
    assert len(leaky_train_df) == len(clean_train_df), \
        f"FATAL (draw_seed={draw_seed}): leaky size mismatch"

    leaky_class = leaky_train_df["label"].value_counts()
    clean_class = clean_train_df["label"].value_counts()
    assert (clean_class.sort_index() == leaky_class.sort_index()).all(), \
        f"FATAL (draw_seed={draw_seed}): class distribution changed"

    leaky_md5 = leaky_train_df["relative_path"].map(md5_map)
    assert leaky_md5.duplicated().sum() == 0, f"FATAL (draw_seed={draw_seed}): repeated MD5 in leaky train"

    leaky_overlap_patients = set(leaky_train_df["patient_id"]) & set(anchor_df["patient_id"])
    assert leaky_overlap_patients == set(sibling_df["patient_id"]), \
        f"FATAL (draw_seed={draw_seed}): leaky/anchor patient overlap != planned K sibling set"

    return leaky_train_df, replacement_map_df, excluded_relpaths


print(f"[A2] Building {len(DRAW_SEEDS)} leaky_train variants (draw_seed = model_seed)")
print(f"     K={K} sibling-eligible patients, class dist sibling={sib_counts_by_class}")

all_excluded = {}
for seed in DRAW_SEEDS:
    leaky_df, repl_df, excluded = build_draw(seed)
    leaky_out = finalize(
        leaky_df, "train", "leaky",
        "clean_train_plus_one_sibling_per_eligible_test_patient_minus_seedDRAW_replacement",
        seed,
    )
    leaky_path = MANIFESTS / f"controlled_leaky_train_a2_seed{seed}.csv"
    repl_path = MANIFESTS / f"controlled_leakage_replacement_map_a2_seed{seed}.csv"
    leaky_out.to_csv(leaky_path, index=False)
    repl_df.to_csv(repl_path, index=False)
    all_excluded[seed] = excluded
    print(f"  draw_seed={seed}: leaky_train={len(leaky_out)} rows -> {leaky_path.name}, "
          f"replacement_map={len(repl_df)} rows -> {repl_path.name}")

# ---------------------------------------------------------------------------
# Sanity: draw_seed=42 must be identical to the original controlled_leaky_train.csv (Exp A)
# ---------------------------------------------------------------------------
orig_leaky = pd.read_csv(MANIFESTS / "controlled_leaky_train.csv")
a2_seed42 = pd.read_csv(MANIFESTS / "controlled_leaky_train_a2_seed42.csv")
orig_relpaths = set(orig_leaky["relative_path"])
a2_relpaths = set(a2_seed42["relative_path"])
identical = orig_relpaths == a2_relpaths
print(f"\n[sanity] draw_seed=42 vs controlled_leaky_train.csv (original Experiment A): "
      f"{'IDENTICAL' if identical else 'DIFFERENT -- RECHECK!'} "
      f"({len(orig_relpaths)} vs {len(a2_relpaths)} unique relative_path, "
      f"symmetric_diff={len(orig_relpaths ^ a2_relpaths)})")
assert identical, "FATAL: draw_seed=42 variant MUST be identical to the original Experiment A manifest"

# ---------------------------------------------------------------------------
# Sanity: all five draws are genuinely different from each other (not degenerate)
# ---------------------------------------------------------------------------
print("\n[sanity] Pairwise differences between draws (number of differing exclusion relative_paths):")
seeds = list(all_excluded.keys())
any_identical_pair = False
for i in range(len(seeds)):
    for j in range(i + 1, len(seeds)):
        s1, s2 = seeds[i], seeds[j]
        diff = all_excluded[s1] ^ all_excluded[s2]
        print(f"  seed {s1} vs seed {s2}: {len(diff)} of {K} exclusions differ")
        if len(diff) == 0:
            any_identical_pair = True
if any_identical_pair:
    print("  [WARNING] some pair of draws has exactly identical exclusions - check RNG/seed.")
else:
    print("  OK - all 5 draws produce mutually different exclusion patterns.")

print("\n[A2] Done. 5 leaky_train variants + replacement maps written to data/manifests/.")
