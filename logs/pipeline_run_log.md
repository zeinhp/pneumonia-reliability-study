# Pipeline Run Log

## Experiment A - protocol Phase 1 (controlled manifests, 2026-07-24)

After protocol.docx (v1.0) was read and the a priori MDES was frozen
(`data/reports/mdes_power_analysis.md`), Experiment A protocol Phase 1 was run:

| Step | Script | Status |
|---|---|---|
| 6.2-6.6 | src/build_controlled_manifests.py | OK - 5 controlled manifests created (anchor 279, clean train 4,607, clean val 603, leaky train 4,607, replacement map K=146) |
| 6.7 | src/validate_controlled_manifests.py | PASS - 22/22 core checks |
| checksum | src/generate_checksums.py manifests | OK - 13 final manifests checksummed |
| test | python -m pytest tests/ -v (+ new tests/test_controlled_manifests.py) | PASS - 98/98 |

Phase 2 (environment: dependency, hardware/software recording) not yet started. No
model training was run.

## Stage 1 (initial project setup, before the Experiment A protocol)

| Step | Script | Status |
|---|---|---|
| 1 | src/import_audit_results.py (freeze audit) | OK - 11 files frozen |
| 2 | src/generate_checksums.py source_audit | OK - 11 SHA-256 checksums |
| 3 | src/validate_source_audit.py | PASS - 19/19 core checks |
| 4 | src/build_derived_manifest.py | OK - master_manifest_derived.csv, original_split_manifest.csv, patient_grouped_split_all_images.csv |
| 5 | src/create_image_control_split.py | OK - image_stratified_split_manifest.csv (seed 42) |
| 6 | src/create_deduplicated_manifest.py (initial version) | OK - patient_grouped_split_deduplicated.csv, 32 files excluded |
| 7 | src/checksum_dataset_files.py (chunked) + src/_consolidate_dataset_checksums.py | OK - 5,856 SHA-256 checksums |
| 8 | src/generate_checksums.py manifests | OK - 5 final manifest checksums |
| 9 | src/build_split_summary.py | OK - 12 rows (4 methods x 3 splits) |
| 10 | src/validate_splits.py (initial version) | PASS (1 non-core note: exact-duplicate IM-0095/IM-0096 spans splits) |
| 11 | python -m pytest tests/ -v | PASS - 47/47 tests |

## Phase 2 (reproducibility fixes & dedup leakage)

Goal: remove hardcoded session paths, close the exact-duplicate image-level leakage
(IM-0095/IM-0096), and provide three primary manifests sharing an identical deduplicated
cohort.

| Step | Script | Status |
|---|---|---|
| A1-A2 | src/project_config.py (new) + config/dataset_config.yaml (updated) | OK - dynamic PROJECT_ROOT, dataset root via CHEST_XRAY_DATA_ROOT |
| A3 | Refactored all src/*.py, tests/*.py, notebooks/*.ipynb | OK - 0 hardcoded session paths (verified by test_no_hardcoded_session_paths) |
| B | src/create_deduplicated_manifest.py (rewrite, part B) | OK - global_deduplication_keep_set.csv, 5,824 keep / 32 exclude, consistent with the old rule |
| C1 | src/create_deduplicated_manifest.py (part C1) | OK - original_split_deduplicated.csv, 5,824 rows, overlap 264 (retained) |
| C2 | src/create_deduplicated_manifest.py (part C2) | OK - image_stratified_split_deduplicated.csv, a NEW 80/10/10 split on the 5,824 cohort, seed 42 |
| C3 | src/create_deduplicated_manifest.py (part C3) | OK - patient_grouped_split_deduplicated.csv rebuilt from the global keep-set, overlap 0, repeated-MD5 0, cross-split-dup-group 0 |
| cross-check | src/create_deduplicated_manifest.py (final step) | OK - the 3 primary manifests share an identical 5,824-image cohort |
| D | src/build_derived_manifest.py, src/create_image_control_split.py (column updates) | OK - cohort/analysis_role/primary_eligible columns added to the 3 all-images manifests |
| E | src/validate_splits.py (rewrite) | OK - primary_analysis_status=PASS, sensitivity_analysis_status=PASS_WITH_KNOWN_SENSITIVITY_ISSUES; optimized (35 seconds -> 1.2 seconds) by checking file existence against dataset_files_sha256.csv instead of a direct per-row stat() |
| F | src/build_split_summary.py (rewrite) | OK - 18 rows (6 methods x 3 splits), columns cohort/analysis_role/primary_eligible/n_repeated_md5/n_cross_split_exact_duplicate_groups |
| G | README.md, data/reports/manifest_lineage.md, config/split_config.yaml, logs/pipeline_run_log.md | OK - documented |
| H | .gitignore (new) | OK |
| I | src/generate_checksums.py manifests | OK - 8 final manifest checksums; source_audit & dataset checksums verified unchanged |
| J | tests/test_no_hardcoded_paths.py, tests/test_deduplication.py, tests/test_split_integrity.py, tests/test_validation_script.py (new/rewrite) | OK - 82 tests |
| K | Full pytest run | PASS - 82/82 |

Optimization note: `test_split_integrity.py::test_all_paths_exist` originally also
performed a direct `stat()` against the mounted drive for every row across 6 manifests
(~35,000 calls), causing a timeout. Fixed by matching `relative_path` against
`dataset_files_sha256.csv`, which was already computed once upfront - same result, much
faster, with no loss of validity (that checksum file is itself built from an actual scan
of the dataset root and re-validated in `validate_source_audit.py`).

No model training was run at any stage.
