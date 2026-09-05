# Pneumonia Reliability Study - Stage 1

Companion code and manifests for "Beyond AUROC: Patient Leakage, Calibration, Robustness,
and Shortcut Learning in Pediatric Pneumonia Classification" (Pradana et al., submitted to
IEEE J-BHI). This repository contains the manifest-generation scripts, training/evaluation
code, exact-duplicate/patient-grouped-split manifests, and the figure-regeneration script
(`figures/regen_figures.py`) used to produce the paper's results. The underlying chest X-ray
images are not redistributed here; they are available from the original
[Kermany/Kaggle dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
(also mirrored on [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/2)).
Licensed under the MIT License (see `LICENSE`).

This project builds the manifest and validation foundation for a reliability study of
pediatric pneumonia detection models on chest X-ray images, on top of an existing dataset
audit. **This stage performs no model training whatsoever.**

## Core principles

- The original dataset (`chest_xray/chest_xray/`) is treated as **read-only**. No file is
  renamed, moved, deleted, or modified.
- The legacy audit (`manifest.csv`, `proposed_split_mapping.csv`, etc.) in
  `data/source_audit/` is **frozen**, not regenerated.
- Patient ID, MD5, perceptual hash, blur variance, and the patient-grouped split are
  **never recomputed**. `StratifiedGroupKFold` is never re-run in this project.
- `proposed_split_mapping.csv` is the source of truth for the patient-grouped split.
- All manifests use a `relative_path` relative to the dataset root, never an absolute path.
- No session/agent path is hardcoded anywhere - see "Path configuration" below.

## Two cohorts, two analytical roles

| Cohort | n_images | Role |
|---|---|---|
| `all_images_5856` | 5,856 | **sensitivity_only** - includes 62 exact-duplicate files |
| `global_exact_deduplicated_5824` | 5,824 | **primary** - used for the main analysis |

Review found that the official patient-grouped split (`patient_grouped_split_all_images.csv`)
has zero *patient overlap* but still leaks at the image level: two MD5-identical files
(`NORMAL2-IM-0095-0001.jpeg` in the `test` split, `NORMAL2-IM-0096-0001.jpeg` in the `train`
split) are recorded as two different patients. Because of this, every all-images manifest was
downgraded to **sensitivity-only** status, and three new manifests were built on top of the
deduplicated cohort (5,824 images) as the **primary analysis**. Full details:
`data/reports/manifest_lineage.md`.

## Three primary manifests - identical cohort

| Manifest | Role | train | val | test | Patient overlap |
|---|---|---|---|---|---|
| `original_split_deduplicated.csv` | primary_control | 5,190 | 16 | 618 | 264 (intentional, retained) |
| `image_stratified_split_deduplicated.csv` | primary_control | 4,658 | 583 | 583 | 649 (intentional, measured) |
| `patient_grouped_split_deduplicated.csv` | **primary** | 4,607 | 603 | 614 | **0** |

All three use the **identical** set of 5,824 `relative_path` entries - only the split
assignment differs (verified automatically, see
`test_three_primary_manifests_share_identical_cohort`).

**The test set in any manifest must never be used for model selection or hyperparameter
tuning** - final evaluation only.

## Sensitivity-only manifests (all-images, 5,856 images)

| Manifest | Note |
|---|---|
| `original_split_manifest.csv` | patient overlap 264 (characteristic of the original Kaggle split) |
| `image_stratified_split_manifest.csv` | patient overlap 684; 7 exact-duplicate groups spread across splits (a random effect, not a bug) |
| `patient_grouped_split_all_images.csv` | `primary_eligible=false`: 1 exact-duplicate group (IM-0095/IM-0096) spread across train/test |

## Path configuration

The project root is determined automatically from the file's own location
(`Path(__file__).resolve().parents[1]`). The dataset root is resolved via
`src/project_config.py` with the following priority:

1. Environment variable `CHEST_XRAY_DATA_ROOT` (works on any OS).
2. Windows: `config/dataset_config.yaml` -> `dataset.data_root_windows`.
3. POSIX: `config/dataset_config.yaml` -> `dataset.data_root` (empty by default - **must**
   be set via the environment variable, since there is no portable POSIX mount location).

**Windows (PowerShell):**
```powershell
$env:CHEST_XRAY_DATA_ROOT = "D:\Claude\Pneumonia\chest_xray\chest_xray"
cd D:\Claude\Pneumonia\pneumonia_reliability_study
python src\validate_source_audit.py
```

**POSIX:**
```bash
export CHEST_XRAY_DATA_ROOT="/path/to/chest_xray"
cd /path/to/pneumonia_reliability_study
python3 src/validate_source_audit.py
```

If the environment variable is not set and there is no valid default for the current OS,
`project_config.get_dataset_root()` raises a `FileNotFoundError` with a clear message -
it never silently falls back to a temporary session path.

## Project structure

```
config/             Dataset & split configuration (YAML)
data/source_audit/  Read-only snapshot of the legacy audit
data/manifests/     Derived manifests (see data/reports/manifest_lineage.md)
data/checksums/     SHA-256 checksums of the dataset, audit, and manifests
data/reports/       Validation reports, split summaries, lineage
src/                Pipeline scripts (see "Execution order" below)
tests/              Unit tests (pytest, 82 tests)
notebooks/          Manifest review notebook (read-only, no training)
logs/               Pipeline execution log
```

## Execution order

All commands are run from the project root, with `CHEST_XRAY_DATA_ROOT` already set
(see "Path configuration").

1. `python src/import_audit_results.py` - freezes the legacy audit into
   `data/source_audit/` + checksums.
2. `python src/generate_checksums.py source_audit` - checksums the frozen audit files.
3. `python src/validate_source_audit.py` - **mandatory gate**. Stops (`exit 1`) if a core
   check fails.
4. `python src/build_derived_manifest.py` - `master_manifest_derived.csv`,
   `original_split_manifest.csv`, `patient_grouped_split_all_images.csv` (all-images,
   sensitivity-only).
5. `python src/create_image_control_split.py` - `image_stratified_split_manifest.csv`
   (all-images, sensitivity-only, seed 42).
6. `python src/create_deduplicated_manifest.py` - builds
   `global_deduplication_keep_set.csv`, then the three primary manifests
   (`original_split_deduplicated.csv`, `image_stratified_split_deduplicated.csv`,
   `patient_grouped_split_deduplicated.csv`) + cross-checks the identical cohort.
7. `python src/checksum_dataset_files.py <batch> <seconds>` (repeated until complete,
   resumable) + `python src/_consolidate_dataset_checksums.py` - SHA-256 checksums of all
   5,856 original image files.
8. `python src/generate_checksums.py manifests` - checksums of all final manifests.
9. `python src/build_split_summary.py` - `data/reports/split_summary.csv`, 6 strategies.
10. `python src/validate_splits.py` - validates the primary manifests (core, must PASS) +
    sensitivity manifests (known issues, does not fail the run unless something unexpected
    appears).
11. `python -m pytest tests/ -v` - 82 unit tests.

## What was intentionally NOT done

- No model training.
- No deletion/moving of physical image files.
- No recomputation of patient ID / MD5 / phash / blur / `StratifiedGroupKFold`.
- The test set was never used for model selection.
