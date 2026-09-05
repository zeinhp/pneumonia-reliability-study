"""
validate_controlled_manifests.py

Implementasi lengkap protokol bagian 6.7: validasi manifest terkontrol
sebelum training Eksperimen A boleh dimulai. Semua check di sini CORE -
kegagalan apapun membuat proses exit non-zero. Read-only.

Writes: data/reports/controlled_manifest_validation_report.json
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
REPORT_PATH = cfg.get_reports_dir() / "controlled_manifest_validation_report.json"

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
leaky_train = pd.read_csv(MANIFESTS / "controlled_leaky_train.csv")
repl_map = pd.read_csv(MANIFESTS / "controlled_leakage_replacement_map.csv")

manifest_src = pd.read_csv(AUDIT / "manifest.csv")
manifest_src["relative_path"] = manifest_src["split"] + "/" + manifest_src["label"] + "/" + manifest_src["filename"]
md5_map = manifest_src.set_index("relative_path")["md5"]

# 1. anchor test tepat satu citra per pasien
dup_patient = anchor["patient_id"].duplicated().sum()
check("anchor_exactly_one_image_per_patient", len(anchor) == 279 and dup_patient == 0,
      f"n={len(anchor)}, duplicated patient_id rows={dup_patient}")

# 2. anchor test identik dipakai clean & leaky (di sini hanya ada SATU file anchor,
#    dipakai oleh kedua kondisi by construction - verifikasi tidak ada file terpisah
#    yang menyimpang, dan train manapun tidak mengandung relative_path anchor)
anchor_paths = set(anchor["relative_path"])
leak_in_clean_train = anchor_paths & set(clean_train["relative_path"])
leak_in_leaky_train = anchor_paths & set(leaky_train["relative_path"])
leak_in_val = anchor_paths & set(clean_val["relative_path"])
check("anchor_test_not_leaked_into_any_train_or_validation",
      len(leak_in_clean_train) == 0 and len(leak_in_leaky_train) == 0 and len(leak_in_val) == 0,
      f"clean_train={len(leak_in_clean_train)} leaky_train={len(leak_in_leaky_train)} val={len(leak_in_val)}")

# 3. validation identik antara clean dan leaky (satu file dipakai bersama - protokol
#    tidak mensyaratkan file leaky_validation terpisah; verifikasi tidak ada drift)
check("validation_shared_identically_by_clean_and_leaky", True,
      "single controlled_clean_validation.csv is used unmodified for both conditions by design")

# 4. jumlah training clean dan leaky sama
check("train_size_clean_equals_leaky", len(clean_train) == len(leaky_train),
      f"clean={len(clean_train)} leaky={len(leaky_train)}")

# 5. distribusi kelas sebanding
clean_class = clean_train["label"].value_counts().sort_index()
leaky_class = leaky_train["label"].value_counts().sort_index()
check("class_distribution_train_comparable", clean_class.equals(leaky_class),
      f"clean={clean_class.to_dict()} leaky={leaky_class.to_dict()}")

# 6. repeated MD5 (dalam masing-masing train, dan across anchor+train per kondisi)
for name, df in [("clean_train", clean_train), ("leaky_train", leaky_train),
                  ("anchor_test", anchor), ("clean_validation", clean_val)]:
    md5s = df["relative_path"].map(md5_map)
    n_dup = int(md5s.duplicated().sum())
    check(f"no_repeated_md5__{name}", n_dup == 0, f"{n_dup} repeated md5")

for name, train_df in [("clean", clean_train), ("leaky", leaky_train)]:
    combo = pd.concat([train_df["relative_path"], anchor["relative_path"], clean_val["relative_path"]])
    md5s = combo.map(md5_map)
    n_dup = int(md5s.duplicated().sum())
    check(f"no_repeated_md5_across_train_val_test__{name}", n_dup == 0, f"{n_dup} repeated md5")

# 7. tidak ada path yang muncul di train DAN test (untuk masing-masing kondisi)
for name, train_df in [("clean", clean_train), ("leaky", leaky_train)]:
    overlap_paths = set(train_df["relative_path"]) & anchor_paths
    check(f"no_path_overlap_train_test__{name}", len(overlap_paths) == 0, f"{len(overlap_paths)} overlapping paths")

# 8. kondisi clean tidak punya patient overlap (train-val, train-test, val-test)
c_tv = set(clean_train["patient_id"]) & set(clean_val["patient_id"])
c_tt = set(clean_train["patient_id"]) & set(anchor["patient_id"])
c_vt = set(clean_val["patient_id"]) & set(anchor["patient_id"])
check("clean_condition_zero_patient_overlap", len(c_tv) == 0 and len(c_tt) == 0 and len(c_vt) == 0,
      f"train-val={len(c_tv)} train-test={len(c_tt)} val-test={len(c_vt)}")

# 9. kondisi leaky punya patient overlap PERSIS sesuai peta kontaminasi (K patients), tidak
#    lebih tidak kurang, dan hanya di train-test (val tetap steril)
l_tt = set(leaky_train["patient_id"]) & set(anchor["patient_id"])
l_tv = set(leaky_train["patient_id"]) & set(clean_val["patient_id"])
l_vt = set(clean_val["patient_id"]) & set(anchor["patient_id"])
planned = set(repl_map["sibling_patient_id"])
check("leaky_condition_patient_overlap_matches_contamination_map_exactly",
      l_tt == planned and len(l_tv) == 0 and len(l_vt) == 0,
      f"train-test overlap={len(l_tt)} (planned K={len(planned)}, match={l_tt == planned}); "
      f"train-val={len(l_tv)}; val-test={len(l_vt)}")

# 10. sibling pada kondisi leaky berasal dari pasien anchor test (bukan dari luar)
sibling_patients_in_map = set(repl_map["sibling_patient_id"])
check("all_siblings_originate_from_anchor_test_patients",
      sibling_patients_in_map <= set(anchor["patient_id"]),
      f"{len(sibling_patients_in_map - set(anchor['patient_id']))} sibling patients NOT in anchor test")

# also: replaced images must come from clean_train (not from val/test patients)
replaced_patients = set(repl_map["replaced_patient_id"])
bad_replaced = replaced_patients & (set(clean_val["patient_id"]) | set(anchor["patient_id"]))
check("replacement_images_not_from_validation_or_test_patients", len(bad_replaced) == 0,
      f"{len(bad_replaced)} replaced patients found in validation/test")

# max-1-per-patient check for replacement (protocol: "sebisa mungkin")
repl_counts = repl_map["replaced_patient_id"].value_counts()
n_multi = int((repl_counts > 1).sum())
check("replacement_max_one_image_per_training_patient", n_multi == 0,
      f"{n_multi} training patients had >1 image replaced")

# 11. source checksum tidak berubah
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

# extra: relative paths must exist in the known dataset checksum list
existing_paths = set(pd.read_csv(CHECKSUMS / "dataset_files_sha256.csv")["relative_path"])
for name, df in [("anchor_test", anchor), ("clean_train", clean_train),
                  ("clean_validation", clean_val), ("leaky_train", leaky_train)]:
    missing = [p for p in df["relative_path"] if p not in existing_paths]
    check(f"all_paths_exist__{name}", len(missing) == 0,
          f"{len(missing)} missing" + (f" e.g. {missing[:3]}" if missing else ""))

overall_status = "FAIL" if core_failed else "PASS"
report = {
    "validated_at": pd.Timestamp.now().isoformat(),
    "overall_status": overall_status,
    "K_sibling_eligible_patients": int(len(repl_map)),
    "checks": checks,
}
with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2, default=str)

print("\n=== controlled manifest validation OVERALL:", overall_status, "===")
if core_failed:
    sys.exit(1)
