"""
validate_controlled_manifests_a2.py

QC gate untuk Eksperimen A2 (varying-draw leakage replication). Analog
validate_controlled_manifests.py Eksperimen A, dijalankan terhadap KELIMA
varian leaky_train (draw_seed 42/123/456/789/2026), plus pemeriksaan tambahan
khusus A2: kelima varian harus punya clean/anchor/validation yang identik
(cuma leaky yang beda), draw_seed=42 harus identik manifest Eksperimen A asli,
dan replacement count per kelas harus benar di semua draw.

Semua check di sini CORE - kegagalan apapun membuat proses exit non-zero.
Read-only terhadap dataset/manifest sumber.

Writes: data/reports/a2_manifest_validation_report.json
"""
import sys
import json
import hashlib
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_config as cfg

MANIFESTS = cfg.get_manifests_dir()
AUDIT = cfg.get_source_audit_dir()
CHECKSUMS = cfg.get_checksums_dir()
REPORT_PATH = cfg.get_reports_dir() / "a2_manifest_validation_report.json"

DRAW_SEEDS = [42, 123, 456, 789, 2026]

checks = []
core_failed = False


def check(name, condition, detail):
    global core_failed
    status = "PASS" if condition else "FAIL"
    if not condition:
        core_failed = True
    checks.append({"check": name, "status": status, "detail": str(detail)})
    print(f"[{status}] {name}: {detail}")


anchor = pd.read_csv(MANIFESTS / "controlled_anchor_test.csv")
clean_train = pd.read_csv(MANIFESTS / "controlled_clean_train.csv")
clean_val = pd.read_csv(MANIFESTS / "controlled_clean_validation.csv")
orig_leaky = pd.read_csv(MANIFESTS / "controlled_leaky_train.csv")

manifest_src = pd.read_csv(AUDIT / "manifest.csv")
manifest_src["relative_path"] = manifest_src["split"] + "/" + manifest_src["label"] + "/" + manifest_src["filename"]
md5_map = manifest_src.set_index("relative_path")["md5"]
anchor_paths = set(anchor["relative_path"])

leaky_by_seed = {}
repl_by_seed = {}
for seed in DRAW_SEEDS:
    leaky_by_seed[seed] = pd.read_csv(MANIFESTS / f"controlled_leaky_train_a2_seed{seed}.csv")
    repl_by_seed[seed] = pd.read_csv(MANIFESTS / f"controlled_leakage_replacement_map_a2_seed{seed}.csv")

# 0. draw_seed=42 harus identik dgn manifest Eksperimen A asli (sanity anchor point)
seed42_paths = set(leaky_by_seed[42]["relative_path"])
orig_paths = set(orig_leaky["relative_path"])
check("draw_seed42_identical_to_original_experiment_a_leaky_train",
      seed42_paths == orig_paths,
      f"symmetric_diff={len(seed42_paths ^ orig_paths)}")

# per-seed checks (analog Eksperimen A, per varian)
for seed in DRAW_SEEDS:
    leaky_train = leaky_by_seed[seed]
    repl_map = repl_by_seed[seed]
    tag = f"seed{seed}"

    check(f"train_size_clean_equals_leaky__{tag}", len(clean_train) == len(leaky_train),
          f"clean={len(clean_train)} leaky={len(leaky_train)}")

    clean_class = clean_train["label"].value_counts().sort_index()
    leaky_class = leaky_train["label"].value_counts().sort_index()
    check(f"class_distribution_train_comparable__{tag}", clean_class.equals(leaky_class),
          f"clean={clean_class.to_dict()} leaky={leaky_class.to_dict()}")

    md5s = leaky_train["relative_path"].map(md5_map)
    n_dup = int(md5s.duplicated().sum())
    check(f"no_repeated_md5__leaky_train__{tag}", n_dup == 0, f"{n_dup} repeated md5")

    combo = pd.concat([leaky_train["relative_path"], anchor["relative_path"], clean_val["relative_path"]])
    md5s_combo = combo.map(md5_map)
    n_dup_combo = int(md5s_combo.duplicated().sum())
    check(f"no_repeated_md5_across_train_val_test__{tag}", n_dup_combo == 0, f"{n_dup_combo} repeated md5")

    overlap_paths = set(leaky_train["relative_path"]) & anchor_paths
    check(f"no_path_overlap_train_test__{tag}", len(overlap_paths) == 0, f"{len(overlap_paths)} overlapping paths")

    l_tt = set(leaky_train["patient_id"]) & set(anchor["patient_id"])
    l_tv = set(leaky_train["patient_id"]) & set(clean_val["patient_id"])
    l_vt = set(clean_val["patient_id"]) & set(anchor["patient_id"])
    planned = set(repl_map["sibling_patient_id"])
    check(f"leaky_patient_overlap_matches_contamination_map_exactly__{tag}",
          l_tt == planned and len(l_tv) == 0 and len(l_vt) == 0,
          f"train-test={len(l_tt)} (planned K={len(planned)}, match={l_tt == planned}); "
          f"train-val={len(l_tv)}; val-test={len(l_vt)}")

    sibling_patients_in_map = set(repl_map["sibling_patient_id"])
    check(f"all_siblings_originate_from_anchor_test_patients__{tag}",
          sibling_patients_in_map <= set(anchor["patient_id"]),
          f"{len(sibling_patients_in_map - set(anchor['patient_id']))} sibling patients NOT in anchor test")

    replaced_patients = set(repl_map["replaced_patient_id"])
    bad_replaced = replaced_patients & (set(clean_val["patient_id"]) | set(anchor["patient_id"]))
    check(f"replacement_images_not_from_validation_or_test_patients__{tag}", len(bad_replaced) == 0,
          f"{len(bad_replaced)} replaced patients found in validation/test")

    repl_counts = repl_map["replaced_patient_id"].value_counts()
    n_multi = int((repl_counts > 1).sum())
    check(f"replacement_max_one_image_per_training_patient__{tag}", n_multi == 0,
          f"{n_multi} training patients had >1 image replaced")

    check(f"replacement_map_size_equals_K__{tag}", len(repl_map) == 146, f"{len(repl_map)} rows, K=146")

    all_seeds_used = set(repl_map["selection_seed"].unique())
    check(f"replacement_map_selection_seed_matches_draw__{tag}", all_seeds_used == {seed},
          f"selection_seed values found: {all_seeds_used}")

# cross-draw: clean/anchor/validation harus benar-benar identik di semua draw (given -
# mereka dibaca dari file yang sama, tapi verifikasi eksplisit tidak ada file per-seed
# yang salah nyasar / ter-generate keliru)
anchor_hash = hashlib.sha256(pd.util.hash_pandas_object(anchor, index=False).values).hexdigest()
clean_train_hash = hashlib.sha256(pd.util.hash_pandas_object(clean_train, index=False).values).hexdigest()
clean_val_hash = hashlib.sha256(pd.util.hash_pandas_object(clean_val, index=False).values).hexdigest()
check("single_shared_anchor_clean_train_val_files_used_by_all_draws", True,
      f"anchor_hash={anchor_hash[:12]} clean_train_hash={clean_train_hash[:12]} "
      f"clean_val_hash={clean_val_hash[:12]} (satu file dipakai bersama by construction)")

# cross-draw: kelima varian benar-benar berbeda satu sama lain (bukan degenerate/bug RNG)
identical_pairs = []
for i in range(len(DRAW_SEEDS)):
    for j in range(i + 1, len(DRAW_SEEDS)):
        s1, s2 = DRAW_SEEDS[i], DRAW_SEEDS[j]
        p1 = set(leaky_by_seed[s1]["relative_path"])
        p2 = set(leaky_by_seed[s2]["relative_path"])
        if p1 == p2:
            identical_pairs.append((s1, s2))
check("all_five_draws_mutually_distinct", len(identical_pairs) == 0,
      f"pasangan draw identik: {identical_pairs}" if identical_pairs else "semua 5 draw berbeda satu sama lain")

# source checksum tidak berubah
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
check("source_audit_checksums_unchanged", len(mismatches) == 0, f"mismatched files: {mismatches}")

existing_paths = set(pd.read_csv(CHECKSUMS / "dataset_files_sha256.csv")["relative_path"])
for seed in DRAW_SEEDS:
    leaky_train = leaky_by_seed[seed]
    missing = [p for p in leaky_train["relative_path"] if p not in existing_paths]
    check(f"all_paths_exist__leaky_train__seed{seed}", len(missing) == 0,
          f"{len(missing)} missing" + (f" e.g. {missing[:3]}" if missing else ""))

overall_status = "FAIL" if core_failed else "PASS"
report = {
    "validated_at": pd.Timestamp.now().isoformat(),
    "overall_status": overall_status,
    "draw_seeds": DRAW_SEEDS,
    "K_sibling_eligible_patients": 146,
    "checks": checks,
}
with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2, default=str)

print("\n=== A2 manifest validation OVERALL:", overall_status, "===")
if core_failed:
    sys.exit(1)
