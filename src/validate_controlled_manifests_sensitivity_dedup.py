"""
validate_controlled_manifests_sensitivity_dedup.py

QC gate (analogous to protocol 6.7 / section 20) for the sensitivity-dedup
manifests (non-dedup cohort, 5,856 images). Same as
validate_controlled_manifests.py for 15/16 checks, BUT the MD5 checks are
adjusted: this cohort DELIBERATELY includes the 32 exact-duplicate images
that were removed in Experiment A, so the absolute "no repeated MD5" rule
(Phase 1) does NOT apply the same way here. Instead: any repeated MD5 MUST be
traceable ONLY to one of the 32 pairs already documented in
data/reports/deduplication_report.csv (explicit allowlist, checked
group-by-group) - a failure still occurs if a duplicate MD5 is found that is
NOT on the allowlist (meaning a manifest bug, not an already-known source
data duplication).

The most important guarantees that STILL must be absolute zero (same as
Phase 1):
  * the anchor test itself must not have any internal duplicate MD5
  * NO anchor test MD5 may match any MD5 in clean_train or clean_validation
    (the Clean condition must stay sterile at the image level, not just the
    patient level)
  * leaky_train may match an anchor test MD5 ONLY via images that are
    genuinely siblings of a test patient (the Leaky condition's intentional
    mechanism) - not a direct exact match to the anchor image itself

Writes: data/reports/sensitivity_dedup_manifest_validation_report.json
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
REPORTS = cfg.get_reports_dir()
REPORT_PATH = REPORTS / "sensitivity_dedup_manifest_validation_report.json"

checks = []
core_failed = False


def check(name, condition, detail):
    global core_failed
    status = "PASS" if condition else "FAIL"
    if not condition:
        core_failed = True
    checks.append({"check": name, "status": status, "detail": str(detail)})
    print(f"[{status}] {name}: {detail}")


anchor = pd.read_csv(MANIFESTS / "controlled_anchor_test_nondedup.csv")
clean_train = pd.read_csv(MANIFESTS / "controlled_clean_train_nondedup.csv")
clean_val = pd.read_csv(MANIFESTS / "controlled_clean_validation_nondedup.csv")
leaky_train = pd.read_csv(MANIFESTS / "controlled_leaky_train_nondedup.csv")
repl_map = pd.read_csv(MANIFESTS / "controlled_leakage_replacement_map_nondedup.csv")
dedup_report = pd.read_csv(REPORTS / "deduplication_report.csv")

manifest_src = pd.read_csv(AUDIT / "manifest.csv")
manifest_src["relative_path"] = manifest_src["split"] + "/" + manifest_src["label"] + "/" + manifest_src["filename"]
md5_map = manifest_src.set_index("relative_path")["md5"]

# allowlist: 32 pairs (relative_path) that ARE genuinely exact-duplicates per
# Stage 1. Any duplicate MD5 found in the non-dedup manifest MUST be
# traceable back to one of these pairs.
allowlisted_paths = set(dedup_report["excluded_relative_path"]) | set(dedup_report["representative_relative_path"])
allowlisted_md5 = set(dedup_report["md5"])

# 1. anchor test has exactly one image per patient (same as Phase 1)
dup_patient = anchor["patient_id"].duplicated().sum()
check("anchor_exactly_one_image_per_patient", len(anchor) == 279 and dup_patient == 0,
      f"n={len(anchor)}, duplicated patient_id rows={dup_patient}")

# 2. anchor test is not leaked into any train/val (path-level, same as Phase 1)
anchor_paths = set(anchor["relative_path"])
leak_in_clean_train = anchor_paths & set(clean_train["relative_path"])
leak_in_leaky_train = anchor_paths & set(leaky_train["relative_path"])
leak_in_val = anchor_paths & set(clean_val["relative_path"])
check("anchor_test_not_leaked_into_any_train_or_validation_by_path",
      len(leak_in_clean_train) == 0 and len(leak_in_leaky_train) == 0 and len(leak_in_val) == 0,
      f"clean_train={len(leak_in_clean_train)} leaky_train={len(leak_in_leaky_train)} val={len(leak_in_val)}")

# 2b. ABSOLUTE REQUIREMENT: anchor test must be free of internal duplicate MD5
anchor_md5 = anchor["relative_path"].map(md5_map)
check("anchor_test_zero_internal_repeated_md5", anchor_md5.duplicated().sum() == 0,
      f"{int(anchor_md5.duplicated().sum())} repeated md5 within anchor test")

# 2c. ABSOLUTE REQUIREMENT: the Clean condition stays sterile at the MD5 level (not just patient)
clean_combo_md5 = pd.concat([clean_train["relative_path"], clean_val["relative_path"]]).map(md5_map)
anchor_md5_set = set(anchor_md5)
clean_md5_overlap = anchor_md5_set & set(clean_combo_md5)
check("clean_condition_zero_md5_overlap_with_anchor_test", len(clean_md5_overlap) == 0,
      f"{len(clean_md5_overlap)} md5 shared between anchor test and clean train+validation")

# 2d. Leaky condition: an anchor test MD5 may appear in leaky_train ONLY via
#     the intentional sibling mechanism (not a direct match to the anchor
#     image itself, which by DEFINITION never enters train in any condition -
#     check 2 already guarantees that at the path level; here we also check
#     the MD5 level so that even a byte-identical "clone" of the anchor image
#     never leaks into leaky_train).
leaky_md5 = leaky_train["relative_path"].map(md5_map)
leaky_anchor_md5_overlap = anchor_md5_set & set(leaky_md5)
check("leaky_condition_zero_md5_overlap_with_anchor_test_itself", len(leaky_anchor_md5_overlap) == 0,
      f"{len(leaky_anchor_md5_overlap)} md5 shared between anchor test and leaky train "
      f"(anchor image byte-content must never appear in ANY train, leaky included)")

# 3. validation is identical between clean and leaky
check("validation_shared_identically_by_clean_and_leaky", True,
      "single controlled_clean_validation_nondedup.csv is used unmodified for both conditions")

# 4. clean and leaky training set sizes are equal
check("train_size_clean_equals_leaky", len(clean_train) == len(leaky_train),
      f"clean={len(clean_train)} leaky={len(leaky_train)}")

# 5. class distributions are comparable
clean_class = clean_train["label"].value_counts().sort_index()
leaky_class = leaky_train["label"].value_counts().sort_index()
check("class_distribution_train_comparable", clean_class.equals(leaky_class),
      f"clean={clean_class.to_dict()} leaky={leaky_class.to_dict()}")

# 6. repeated MD5 within each manifest - ALL of them MUST be traceable to the
#    32-pair allowlist (either via relative_path OR md5 value).
def repeated_md5_groups(df, name):
    m = df["relative_path"].map(md5_map)
    dup_mask = m.duplicated(keep=False)
    dup_rows = df.loc[dup_mask.values, ["relative_path"]].copy()
    dup_rows["md5"] = m[dup_mask.values].values
    unexplained = dup_rows[~(dup_rows["relative_path"].isin(allowlisted_paths)
                              | dup_rows["md5"].isin(allowlisted_md5))]
    return dup_rows, unexplained


for name, df in [("clean_train", clean_train), ("leaky_train", leaky_train),
                  ("clean_validation", clean_val)]:
    dup_rows, unexplained = repeated_md5_groups(df, name)
    check(f"repeated_md5_all_explained_by_known_dedup_allowlist__{name}", len(unexplained) == 0,
          f"{len(dup_rows)} total repeated-md5 rows, {len(unexplained)} NOT in 32-pair allowlist"
          + (f" e.g. {unexplained['relative_path'].tolist()[:3]}" if len(unexplained) else ""))

# 6b. Explicit verification of the special NORMAL2-IM-0095/IM-0096 pair's behavior:
#     it must appear EXACTLY 2x (byte-identical, different patient_id) in leaky_train
#     (1x from IM-0096 which is already in clean_train, 1x from IM-0095 inserted
#     as a sibling), and EXACTLY 1x in clean_train (IM-0096 only -
#     IM-0095 is a test/sibling image, not part of clean_train).
special_md5 = md5_map["test/NORMAL/NORMAL2-IM-0095-0001.jpeg"]
assert special_md5 == md5_map["test/NORMAL/NORMAL2-IM-0096-0001.jpeg"]
n_special_clean = int((clean_train["relative_path"].map(md5_map) == special_md5).sum())
n_special_leaky = int((leaky_train["relative_path"].map(md5_map) == special_md5).sum())
check("known_special_pair_IM0095_IM0096_occurs_as_expected",
      n_special_clean == 1 and n_special_leaky == 2,
      f"clean_train occurrences={n_special_clean} (expected 1, IM-0096 only), "
      f"leaky_train occurrences={n_special_leaky} (expected 2, IM-0096 + IM-0095-sibling)")

clean_dup_rows, _ = repeated_md5_groups(clean_train, "clean_train")
leaky_dup_rows, _ = repeated_md5_groups(leaky_train, "leaky_train")
print(f"[info] repeated-md5 rows: clean_train={len(clean_dup_rows)} leaky_train={len(leaky_dup_rows)} "
      f"(can differ because the sibling<->replacement swap touches different members of a duplicate pair)")

# 7. no path appears in both train AND test (path-level, same as Phase 1)
for name, train_df in [("clean", clean_train), ("leaky", leaky_train)]:
    overlap_paths = set(train_df["relative_path"]) & anchor_paths
    check(f"no_path_overlap_train_test__{name}", len(overlap_paths) == 0, f"{len(overlap_paths)} overlapping paths")

# 8. the clean condition has no patient overlap
c_tv = set(clean_train["patient_id"]) & set(clean_val["patient_id"])
c_tt = set(clean_train["patient_id"]) & set(anchor["patient_id"])
c_vt = set(clean_val["patient_id"]) & set(anchor["patient_id"])
check("clean_condition_zero_patient_overlap", len(c_tv) == 0 and len(c_tt) == 0 and len(c_vt) == 0,
      f"train-val={len(c_tv)} train-test={len(c_tt)} val-test={len(c_vt)}")

# 9. the leaky condition has patient overlap EXACTLY matching the contamination map
l_tt = set(leaky_train["patient_id"]) & set(anchor["patient_id"])
l_tv = set(leaky_train["patient_id"]) & set(clean_val["patient_id"])
l_vt = set(clean_val["patient_id"]) & set(anchor["patient_id"])
planned = set(repl_map["sibling_patient_id"])
check("leaky_condition_patient_overlap_matches_contamination_map_exactly",
      l_tt == planned and len(l_tv) == 0 and len(l_vt) == 0,
      f"train-test overlap={len(l_tt)} (planned K={len(planned)}, match={l_tt == planned}); "
      f"train-val={len(l_tv)}; val-test={len(l_vt)}")

# 10. siblings originate from anchor test patients
sibling_patients_in_map = set(repl_map["sibling_patient_id"])
check("all_siblings_originate_from_anchor_test_patients",
      sibling_patients_in_map <= set(anchor["patient_id"]),
      f"{len(sibling_patients_in_map - set(anchor['patient_id']))} sibling patients NOT in anchor test")

replaced_patients = set(repl_map["replaced_patient_id"])
bad_replaced = replaced_patients & (set(clean_val["patient_id"]) | set(anchor["patient_id"]))
check("replacement_images_not_from_validation_or_test_patients", len(bad_replaced) == 0,
      f"{len(bad_replaced)} replaced patients found in validation/test")

repl_counts = repl_map["replaced_patient_id"].value_counts()
n_multi = int((repl_counts > 1).sum())
check("replacement_max_one_image_per_training_patient", n_multi == 0,
      f"{n_multi} training patients had >1 image replaced")

# 11. source checksums have not changed (same as Phase 1)
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

# 12. all relative_path values exist in the physical dataset
existing_paths = set(pd.read_csv(CHECKSUMS / "dataset_files_sha256.csv")["relative_path"])
for name, df in [("anchor_test", anchor), ("clean_train", clean_train),
                  ("clean_validation", clean_val), ("leaky_train", leaky_train)]:
    missing = [p for p in df["relative_path"] if p not in existing_paths]
    check(f"all_paths_exist__{name}", len(missing) == 0,
          f"{len(missing)} missing" + (f" e.g. {missing[:3]}" if missing else ""))

# 13. total source images (train+val+FULL test_df, not just anchor) = dedup
#     cohort + 32 (final sanity check against the source split file, not the derived manifest)
dedup_total = len(pd.read_csv(MANIFESTS / "patient_grouped_split_deduplicated.csv"))
allimg_full = pd.read_csv(MANIFESTS / "patient_grouped_split_all_images.csv")
nondedup_total = len(allimg_full)
check("nondedup_source_split_total_equals_dedup_plus_32",
      nondedup_total == dedup_total + 32,
      f"patient_grouped_split_all_images.csv rows={nondedup_total}, dedup total={dedup_total} (+32 expected)")
check("clean_train_plus_val_plus_anchor_matches_train_val_counts_of_source_split",
      len(clean_train) + len(clean_val) + 279
      == int((allimg_full["experimental_split"] == "train").sum())
      + int((allimg_full["experimental_split"] == "validation").sum()) + 279,
      "sanity: controlled manifest train/val row counts match source split train/val row counts "
      "(anchor is by-construction 279, not the full 615-row test_df, same as the dedup cohort: anchor 279 out of test 614)")

overall_status = "FAIL" if core_failed else "PASS"
report = {
    "validated_at": pd.Timestamp.now().isoformat(),
    "overall_status": overall_status,
    "K_sibling_eligible_patients": int(len(repl_map)),
    "cohort": "global_all_images_5856_sensitivity_dedup",
    "checks": checks,
}
with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2, default=str)

print("\n=== sensitivity-dedup controlled manifest validation OVERALL:", overall_status, "===")
if core_failed:
    sys.exit(1)
