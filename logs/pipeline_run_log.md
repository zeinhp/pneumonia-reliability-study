# Pipeline Run Log

## Eksperimen A - Fase 1 protokol (manifest terkontrol, 2026-07-24)

Setelah protokol.docx (v1.0) dibaca dan MDES a priori dibekukan
(`data/reports/mdes_power_analysis.md`), Fase 1 protokol Eksperimen A dijalankan:

| Langkah | Script | Status |
|---|---|---|
| 6.2-6.6 | src/build_controlled_manifests.py | OK - 5 manifest terkontrol dibuat (anchor 279, clean train 4.607, clean val 603, leaky train 4.607, replacement map K=146) |
| 6.7 | src/validate_controlled_manifests.py | PASS - 22/22 check core |
| checksum | src/generate_checksums.py manifests | OK - 13 manifest final ter-checksum |
| test | python -m pytest tests/ -v (+ tests/test_controlled_manifests.py baru) | PASS - 98/98 |

Fase 2 (environment: dependency, hardware/software recording) belum dimulai. Tidak ada
training model yang dijalankan.

## Tahap 1 (setup awal proyek, sebelum protokol Eksperimen A)

| Langkah | Script | Status |
|---|---|---|
| 1 | src/import_audit_results.py (freeze audit) | OK - 11 file dibekukan |
| 2 | src/generate_checksums.py source_audit | OK - 11 checksum SHA-256 |
| 3 | src/validate_source_audit.py | PASS - 19/19 check core |
| 4 | src/build_derived_manifest.py | OK - master_manifest_derived.csv, original_split_manifest.csv, patient_grouped_split_all_images.csv |
| 5 | src/create_image_control_split.py | OK - image_stratified_split_manifest.csv (seed 42) |
| 6 | src/create_deduplicated_manifest.py (versi awal) | OK - patient_grouped_split_deduplicated.csv, 32 file dikeluarkan |
| 7 | src/checksum_dataset_files.py (chunked) + src/_consolidate_dataset_checksums.py | OK - 5.856 checksum SHA-256 |
| 8 | src/generate_checksums.py manifests | OK - 5 checksum manifest final |
| 9 | src/build_split_summary.py | OK - 12 baris (4 metode x 3 split) |
| 10 | src/validate_splits.py (versi awal) | PASS (1 catatan non-core: exact-dup IM-0095/IM-0096 lintas split) |
| 11 | python -m pytest tests/ -v | PASS - 47/47 test |

## Fase 2 (perbaikan reproducibility & dedup leakage)

Tujuan: hilangkan hardcoded session path, tutup image-level leakage exact-duplicate
IM-0095/IM-0096, dan sediakan tiga manifest utama dengan cohort deduplicated identik.

| Langkah | Script | Status |
|---|---|---|
| A1-A2 | src/project_config.py (baru) + config/dataset_config.yaml (update) | OK - PROJECT_ROOT dinamis, dataset root via CHEST_XRAY_DATA_ROOT |
| A3 | Refactor seluruh src/*.py, tests/*.py, notebooks/*.ipynb | OK - 0 hardcoded session path (diverifikasi test_no_hardcoded_session_paths) |
| B | src/create_deduplicated_manifest.py (rewrite, bagian B) | OK - global_deduplication_keep_set.csv, 5.824 keep / 32 exclude, konsisten dengan aturan lama |
| C1 | src/create_deduplicated_manifest.py (bagian C1) | OK - original_split_deduplicated.csv, 5.824 baris, overlap 264 (dipertahankan) |
| C2 | src/create_deduplicated_manifest.py (bagian C2) | OK - image_stratified_split_deduplicated.csv, split BARU 80/10/10 pada cohort 5.824, seed 42 |
| C3 | src/create_deduplicated_manifest.py (bagian C3) | OK - patient_grouped_split_deduplicated.csv dibangun ulang dari keep-set global, overlap 0, repeated-md5 0, cross-split-dup-group 0 |
| cross-check | src/create_deduplicated_manifest.py (akhir) | OK - 3 manifest utama berbagi cohort 5.824 identik |
| D | src/build_derived_manifest.py, src/create_image_control_split.py (update kolom) | OK - cohort/analysis_role/primary_eligible ditambahkan ke 3 manifest all-images |
| E | src/validate_splits.py (rewrite) | OK - primary_analysis_status=PASS, sensitivity_analysis_status=PASS_WITH_KNOWN_SENSITIVITY_ISSUES; dioptimasi (35 detik -> 1.2 detik) dengan mengecek eksistensi file lewat dataset_files_sha256.csv, bukan stat() langsung per baris |
| F | src/build_split_summary.py (rewrite) | OK - 18 baris (6 metode x 3 split), kolom cohort/analysis_role/primary_eligible/n_repeated_md5/n_cross_split_exact_duplicate_groups |
| G | README.md, data/reports/manifest_lineage.md, config/split_config.yaml, logs/pipeline_run_log.md | OK - didokumentasikan |
| H | .gitignore (baru) | OK |
| I | src/generate_checksums.py manifests | OK - 8 checksum manifest final; source_audit & dataset checksum diverifikasi tidak berubah |
| J | tests/test_no_hardcoded_paths.py, tests/test_deduplication.py, tests/test_split_integrity.py, tests/test_validation_script.py (baru/rewrite) | OK - 82 test |
| K | Full pytest run | PASS - 82/82 |

Catatan optimasi: `test_split_integrity.py::test_all_paths_exist` awalnya juga melakukan
`stat()` langsung ke drive mounted untuk setiap baris x 6 manifest (~35.000 panggilan),
menyebabkan timeout. Diperbaiki dengan mencocokkan `relative_path` terhadap
`dataset_files_sha256.csv` yang sudah dihitung sekali di awal - hasil sama, jauh lebih cepat,
tanpa mengorbankan validitas (checksum itu sendiri dibangun dari pemindaian nyata dataset root
dan divalidasi ulang di `validate_source_audit.py`).

Tidak ada training model yang dijalankan pada tahap manapun.
