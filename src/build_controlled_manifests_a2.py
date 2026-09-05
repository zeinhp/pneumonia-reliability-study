"""
build_controlled_manifests_a2.py

Eksperimen A2 (varying-draw leakage replication, CLAUDE.md #6b poin 2):
membangun 5 varian manifest `leaky_train` yang berbeda dalam HAL SATU variabel
saja - seed yang dipakai untuk memilih citra clean-train mana yang DIKELUARKAN
untuk mengimbangi ukuran/kelas saat sibling disisipkan (protokol 6.5). Di
Eksperimen A asli, seed exclusion ini SELALU 42 untuk seluruh 30 run (2 kondisi
x 3 arsitektur x 5 model-seed) - satu "draw" tetap. Limitasi ini dicatat di
Laporan Eksperimen A (CI efek-leakage tidak menangkap ketidakpastian pola
kontaminasi, cuma noise training).

A2 memvariasikan seed exclusion SELARAS dengan 5 model-seed yang sudah ada
(draw_seed = model_seed): 42, 123, 456, 789, 2026. Sibling yang disisipkan ke
leaky (K=146 pasien, 1 sibling/pasien, dipilih citra pertama setelah anchor
secara leksikografis - protokol 6.3) TIDAK berubah - itu tetap deterministik
per pasien, sesuai keputusan user (hanya seed exclusion/replacement yang
divariasikan, bukan pemilihan sibling).

Kondisi Clean, validation, dan anchor test TIDAK berubah sama sekali - persis
sama dengan Eksperimen A (dibaca ulang dari controlled_clean_train.csv,
controlled_clean_validation.csv, controlled_anchor_test.csv yang sudah ada).
Hanya leaky_train yang punya 5 varian.

Sanity check penting: varian draw_seed=42 HARUS identik dengan
controlled_leaky_train.csv (manifest asli Eksperimen A) - kalau tidak, ada bug.

Status: EXPLORATORY (protokol §24) - anchor test sudah terbuka sejak Fase 5
Eksperimen A. Analisis ini TIDAK mengubah kesimpulan confirmatory A, menguji
limitasi single-draw yang tercatat di Laporan A.

Output (data/manifests/):
    controlled_leaky_train_a2_seed{S}.csv        untuk S in [42,123,456,789,2026]
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
# Muat ulang komponen yang TIDAK berubah antar-draw (persis Eksperimen A)
# ---------------------------------------------------------------------------
clean_train_df = pd.read_csv(MANIFESTS / "controlled_clean_train.csv")
clean_val_df = pd.read_csv(MANIFESTS / "controlled_clean_validation.csv")
anchor_df = pd.read_csv(MANIFESTS / "controlled_anchor_test.csv")

assert len(clean_train_df) == 4607, f"clean_train harus 4607, dapat {len(clean_train_df)}"
assert len(clean_val_df) == 603, f"clean_val harus 603, dapat {len(clean_val_df)}"
assert len(anchor_df) == 279 and anchor_df["patient_id"].nunique() == 279

# rekonstruksi sibling_df persis logic protokol 6.3 dari test_df asli (dedup cohort)
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


print(f"[A2] Membangun {len(DRAW_SEEDS)} varian leaky_train (draw_seed = model_seed)")
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
# Sanity: draw_seed=42 harus identik dgn controlled_leaky_train.csv asli (Eksp A)
# ---------------------------------------------------------------------------
orig_leaky = pd.read_csv(MANIFESTS / "controlled_leaky_train.csv")
a2_seed42 = pd.read_csv(MANIFESTS / "controlled_leaky_train_a2_seed42.csv")
orig_relpaths = set(orig_leaky["relative_path"])
a2_relpaths = set(a2_seed42["relative_path"])
identical = orig_relpaths == a2_relpaths
print(f"\n[sanity] draw_seed=42 vs controlled_leaky_train.csv (Eksperimen A asli): "
      f"{'IDENTIK' if identical else 'BEDA -- CEK ULANG!'} "
      f"({len(orig_relpaths)} vs {len(a2_relpaths)} relative_path unik, "
      f"symmetric_diff={len(orig_relpaths ^ a2_relpaths)})")
assert identical, "FATAL: draw_seed=42 variant HARUS identik dengan manifest Eksperimen A asli"

# ---------------------------------------------------------------------------
# Sanity: kelima draw benar-benar berbeda satu sama lain (bukan degenerate)
# ---------------------------------------------------------------------------
print("\n[sanity] Perbedaan pairwise antar draw (jumlah relative_path exclusion yang beda):")
seeds = list(all_excluded.keys())
any_identical_pair = False
for i in range(len(seeds)):
    for j in range(i + 1, len(seeds)):
        s1, s2 = seeds[i], seeds[j]
        diff = all_excluded[s1] ^ all_excluded[s2]
        print(f"  seed {s1} vs seed {s2}: {len(diff)} dari {K} exclusion berbeda")
        if len(diff) == 0:
            any_identical_pair = True
if any_identical_pair:
    print("  [WARNING] ada pasangan draw yang exclusion-nya identik persis - cek RNG/seed.")
else:
    print("  OK - semua 5 draw menghasilkan pola exclusion yang berbeda satu sama lain.")

print("\n[A2] Selesai. 5 varian leaky_train + replacement map ditulis ke data/manifests/.")
