# Pneumonia Reliability Study - Tahap 1

Companion code and manifests for "Beyond AUROC: Patient Leakage, Calibration, Robustness,
and Shortcut Learning in Pediatric Pneumonia Classification" (Pradana et al., submitted to
IEEE J-BHI). This repository contains the manifest-generation scripts, training/evaluation
code, exact-duplicate/patient-grouped-split manifests, and the figure-regeneration script
(`figures/regen_figures.py`) used to produce the paper's results. The underlying chest X-ray
images are not redistributed here; they are available from the original
[Kermany/Kaggle dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
(also mirrored on [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/2)).
Licensed under the MIT License (see `LICENSE`).

Proyek ini membangun fondasi manifest & validasi untuk penelitian reliabilitas model
deteksi pneumonia pediatrik dari citra chest X-ray, di atas hasil audit dataset yang
sudah ada. **Tahap ini tidak melakukan training model apapun.**

## Prinsip utama

- Dataset asli (`chest_xray/chest_xray/`) diperlakukan **read-only**. Tidak ada file yang
  di-rename, dipindah, dihapus, atau diubah isinya.
- Audit lama (`manifest.csv`, `proposed_split_mapping.csv`, dll.) di `data/source_audit/`
  **dibekukan**, bukan diregenerasi.
- Patient ID, MD5, perceptual hash, blur variance, dan split patient-grouped **tidak
  dihitung ulang**. `StratifiedGroupKFold` tidak pernah dijalankan ulang di proyek ini.
- `proposed_split_mapping.csv` adalah sumber kebenaran (source of truth) split
  patient-grouped.
- Semua manifest memakai `relative_path` terhadap root dataset, bukan path absolut.
- Tidak ada path sesi/agent yang hardcoded di manapun - lihat "Konfigurasi path" di bawah.

## Dua cohort, dua peran analisis

| Cohort | n_images | Peran |
|---|---|---|
| `all_images_5856` | 5.856 | **sensitivity_only** - termasuk 62 file exact-duplicate |
| `global_exact_deduplicated_5824` | 5.824 | **primary** - dipakai untuk analisis utama |

Review menemukan bahwa split patient-grouped resmi (`patient_grouped_split_all_images.csv`)
punya nol *patient overlap* tapi tetap bocor di level citra: dua file identik MD5
(`NORMAL2-IM-0095-0001.jpeg` di split `test`, `NORMAL2-IM-0096-0001.jpeg` di split `train`)
tercatat sebagai dua pasien berbeda. Karena itu, seluruh manifest all-images diturunkan
statusnya jadi **sensitivity-only**, dan tiga manifest baru dibangun di atas cohort
deduplicated (5.824 citra) sebagai **analisis utama**. Detail lengkap:
`data/reports/manifest_lineage.md`.

## Tiga manifest utama (primary) - cohort identik

| Manifest | Peran | train | val | test | Overlap pasien |
|---|---|---|---|---|---|
| `original_split_deduplicated.csv` | primary_control | 5.190 | 16 | 618 | 264 (disengaja, dipertahankan) |
| `image_stratified_split_deduplicated.csv` | primary_control | 4.658 | 583 | 583 | 649 (disengaja, diukur) |
| `patient_grouped_split_deduplicated.csv` | **primary** | 4.607 | 603 | 614 | **0** |

Ketiganya memakai 5.824 `relative_path` yang **identik** - hanya assignment split yang
berbeda (diverifikasi otomatis, lihat `test_three_primary_manifests_share_identical_cohort`).

**Test set pada manifest manapun tidak boleh dipakai untuk pemilihan model atau
hyperparameter tuning** - hanya untuk evaluasi akhir.

## Manifest sensitivity-only (all-images, 5.856 citra)

| Manifest | Catatan |
|---|---|
| `original_split_manifest.csv` | overlap pasien 264 (karakteristik split asli Kaggle) |
| `image_stratified_split_manifest.csv` | overlap pasien 684; 7 grup exact-duplicate tersebar lintas split (efek acak, bukan bug) |
| `patient_grouped_split_all_images.csv` | `primary_eligible=false`: 1 grup exact-duplicate (IM-0095/IM-0096) tersebar lintas train/test |

## Konfigurasi path

Root proyek ditentukan otomatis dari lokasi file (`Path(__file__).resolve().parents[1]`).
Dataset root di-resolve lewat `src/project_config.py` dengan prioritas:

1. Environment variable `CHEST_XRAY_DATA_ROOT` (berlaku di OS manapun).
2. Windows: `config/dataset_config.yaml` -> `dataset.data_root_windows`.
3. POSIX: `config/dataset_config.yaml` -> `dataset.data_root` (default kosong - **wajib** di-set
   lewat environment variable karena tidak ada lokasi mount POSIX yang portable).

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

Jika env var tidak di-set dan tidak ada default yang valid untuk OS saat ini,
`project_config.get_dataset_root()` akan melempar `FileNotFoundError` dengan pesan yang
jelas - tidak pernah diam-diam jatuh ke path sesi sementara.

## Struktur proyek

```
config/             Konfigurasi dataset & split (YAML)
data/source_audit/  Snapshot read-only dari audit lama
data/manifests/     Manifest turunan (lihat data/reports/manifest_lineage.md)
data/checksums/     Checksum SHA-256 dataset, audit, dan manifest
data/reports/       Laporan validasi, ringkasan split, lineage
src/                Script pipeline (lihat "Urutan eksekusi" di bawah)
tests/              Unit test (pytest, 82 test)
notebooks/           Notebook review manifest (read-only, tidak ada training)
logs/               Log eksekusi pipeline
```

## Urutan eksekusi

Semua perintah dijalankan dari root proyek, dengan `CHEST_XRAY_DATA_ROOT` sudah di-set
(lihat "Konfigurasi path").

1. `python src/import_audit_results.py` - freeze audit lama ke `data/source_audit/` + checksum.
2. `python src/generate_checksums.py source_audit` - checksum file audit yang dibekukan.
3. `python src/validate_source_audit.py` - **gate wajib**. Berhenti (`exit 1`) jika core check gagal.
4. `python src/build_derived_manifest.py` - `master_manifest_derived.csv`,
   `original_split_manifest.csv`, `patient_grouped_split_all_images.csv` (semua all-images,
   sensitivity-only).
5. `python src/create_image_control_split.py` - `image_stratified_split_manifest.csv`
   (all-images, sensitivity-only, seed 42).
6. `python src/create_deduplicated_manifest.py` - membangun `global_deduplication_keep_set.csv`
   lalu tiga manifest utama (`original_split_deduplicated.csv`,
   `image_stratified_split_deduplicated.csv`, `patient_grouped_split_deduplicated.csv`) +
   cross-check cohort identik.
7. `python src/checksum_dataset_files.py <batch> <detik>` (diulang sampai selesai, resumable)
   + `python src/_consolidate_dataset_checksums.py` - checksum SHA-256 seluruh 5.856 file gambar asli.
8. `python src/generate_checksums.py manifests` - checksum seluruh manifest final.
9. `python src/build_split_summary.py` - `data/reports/split_summary.csv`, 6 strategi.
10. `python src/validate_splits.py` - validasi primary (core, wajib PASS) + sensitivity
    (known-issues, tidak menggagalkan run kecuali ada yang tidak terduga).
11. `python -m pytest tests/ -v` - 82 unit test.

## Yang sengaja TIDAK dilakukan

- Tidak ada training model.
- Tidak ada penghapusan/pemindahan file gambar fisik.
- Tidak ada perhitungan ulang patient ID / MD5 / phash / blur / `StratifiedGroupKFold`.
- Test set tidak digunakan untuk pemilihan model.
