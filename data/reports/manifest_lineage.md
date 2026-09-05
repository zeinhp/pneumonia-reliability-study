# Manifest Lineage

This document explains the derivation lineage of every manifest in the
`pneumonia_reliability_study` project, what was **not** regenerated, and how the
cross-split exact-duplicate leakage (a review finding) was fixed.

```
manifest.csv (frozen, data/source_audit/)
  |
  ├── original_split_manifest.csv            [all_images_5856, sensitivity_only]
  ├── master_manifest_derived.csv            [combined audit flags]
  ├── image_stratified_split_manifest.csv    [all_images_5856, sensitivity_only, NEW split seed=42]
  |
  └── proposed_split_mapping.csv (frozen, data/source_audit/)
        └── patient_grouped_split_all_images.csv   [all_images_5856, sensitivity_only,
                                                      primary_eligible=false: exact-duplicate
                                                      MD5 f9528b244ccd49f3d89ead90ba09c520
                                                      (IM-0095/IM-0096) span train & test]

master_manifest_derived.csv + exact_duplicates.csv
  |
  └── global_deduplication_keep_set.csv      [5,856 decisions: 5,824 keep, 32 exclude]
        |
        ├── original_split_deduplicated.csv              [global_exact_deduplicated_5824, primary_control]
        ├── image_stratified_split_deduplicated.csv       [global_exact_deduplicated_5824, primary_control,
        |                                                   NEW split on the 5,824 cohort, seed=42]
        └── patient_grouped_split_deduplicated.csv        [global_exact_deduplicated_5824, primary,
                                                             filtered from proposed_split_mapping.csv]
```

## Why the deduplicated cohort became the primary analysis

Review found that `patient_grouped_split_all_images.csv` (the official patient-grouped
split from the legacy audit) has zero *patient overlap*, but still leaks at the image
level: two MD5-identical files (`f9528b244ccd49f3d89ead90ba09c520`) sit in different
splits because they are recorded as two different patients:

- `test/NORMAL/NORMAL2-IM-0095-0001.jpeg` (patient_id `IM-0095`) -> split `test`
- `test/NORMAL/NORMAL2-IM-0096-0001.jpeg` (patient_id `IM-0096`) -> split `train`

This is **image-level leakage**: the model can see a byte-for-byte identical image
during training and then be evaluated again on the exact same image at test time. Zero
patient overlap is not sufficient to guarantee no leakage when the source dataset
contains exact duplicates mislabeled under different patients.

Because of this, `patient_grouped_split_all_images.csv` (and the two other all-images
manifests) were downgraded to **sensitivity-only** status - useful for
sensitivity/historical comparison analysis, but no longer used as the study's primary
condition.

## The fix: global deduplication keep-set

`global_deduplication_keep_set.csv` records the keep/exclude decision for all 5,856
images, using **one deterministic rule** identical to the one used in the earlier
version of `patient_grouped_split_deduplicated.csv` (verified automatically at build
time - see `src/create_deduplicated_manifest.py`, the "consistency check" section):
within each MD5 group, keep the image with the lexicographically smallest filename and
exclude the rest. Near-duplicates (perceptual hash) are NEVER used as a reason to
exclude.

For the IM-0095/IM-0096 pair, `NORMAL2-IM-0095-0001.jpeg` < `NORMAL2-IM-0096-0001.jpeg`
lexicographically, so IM-0095 (split `test`) becomes the representative and IM-0096
(split `train`) is excluded from the deduplicated patient-grouped manifest. After this,
no exact-duplicate group is spread across splits in any primary manifest (verified as a
core check in `validate_splits.py` and the unit test
`test_primary_no_cross_split_exact_duplicate_groups`).

## Three primary manifests - identical 5,824-image cohort

`original_split_deduplicated.csv`, `image_stratified_split_deduplicated.csv`, and
`patient_grouped_split_deduplicated.csv` are all built from the EXACT SAME
`global_exact_deduplicated_5824` cohort (5,824 identical `relative_path` entries, only
the split assignment differs). This is what makes the three fairly comparable in
Experiment A:

- **original_deduplicated** - control: the original Kaggle split (train/val/test)
  uncorrected, patient overlap retained as an observed characteristic (not a flaw fixed
  here).
- **image_stratified_split_deduplicated** - control: a NEW image-level split
  (stratified by label, seed 42, 80/10/10) rebuilt from scratch on the 5,824 cohort
  (not a filtered version of the old 5,856 split, so the 80/10/10 ratio stays precise).
  Patient overlap still occurs (measured, not removed) since it is deliberately not
  patient-grouped.
- **patient_grouped_deduplicated** - the primary condition: `StratifiedGroupKFold` is
  NOT re-run; `proposed_split_mapping.csv` is filtered with the same keep-set. Zero
  patient overlap, zero repeated MD5, zero exact-duplicate group spread across splits.

## What was NOT regenerated

- The dataset audit (frozen July 23, 2026): every file under `data/source_audit/`,
  frozen verbatim, checksummed in `data/checksums/source_audit_sha256.csv`, re-validated
  on every run (`validate_source_audit.py`, `validate_splits.py`, and the test
  `test_source_audit_checksums_unchanged`).
- Patient ID, MD5, perceptual hash, and blur variance: taken directly from the legacy
  `manifest.csv`.
- The grouped split (patient-grouped stratified): `StratifiedGroupKFold` is NOT re-run
  at any stage, including when building `patient_grouped_split_deduplicated.csv` - that
  manifest is only a filtered version of `proposed_split_mapping.csv`.

## What was NEWLY created

- `global_deduplication_keep_set.csv` - the global keep/exclude decision, one
  consistent rule for the entire dataset.
- `image_stratified_split_deduplicated.csv` - the only genuinely new image split on the
  deduplicated cohort (aside from the earlier stage's all-images
  `image_stratified_split_manifest.csv`).
- `original_split_deduplicated.csv` and `patient_grouped_split_deduplicated.csv` -
  filtered manifests, not new splits.

## No original image was deleted, moved, or modified

Every operation in this project is manifest-level only. The original dataset is only
read, never rewritten - verified via `data/checksums/dataset_files_sha256.csv` (5,856
files), re-checked on every validation.

## Path configuration (CHEST_XRAY_DATA_ROOT)

Every script determines `PROJECT_ROOT` dynamically (`Path(__file__).resolve().parents[1]`)
and resolves the dataset root via `src/project_config.py`, with priority order:
environment variable `CHEST_XRAY_DATA_ROOT` > `config/dataset_config.yaml`
(`data_root_windows` on Windows, `data_root` on POSIX). No agent session path is
hardcoded anywhere in `src/`, `tests/`, or `notebooks/` - guaranteed by the test
`test_no_hardcoded_session_paths`.

## Note for human review (as of Stage 1) - ALREADY FIXED

The earlier finding (exact-duplicate IM-0095/IM-0096 spread across splits in
`patient_grouped_split_all_images.csv`) has been addressed via the mechanism above: that
all-images manifest remains available as a sensitivity analysis (flagged
`primary_eligible=false`, reason documented), while the primary manifests
(`*_deduplicated.csv`) are free of this issue. There is no open issue requiring further
human decision at this stage.
