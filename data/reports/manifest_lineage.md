# Manifest Lineage

Dokumen ini menjelaskan alur turunan (lineage) seluruh manifest dalam proyek `pneumonia_reliability_study`,
apa yang **tidak** diregenerasi, dan bagaimana leakage exact-duplicate lintas split (temuan review)
diperbaiki.

```
manifest.csv (frozen, data/source_audit/)
  |
  ├── original_split_manifest.csv            [all_images_5856, sensitivity_only]
  ├── master_manifest_derived.csv            [gabungan seluruh flag audit]
  ├── image_stratified_split_manifest.csv    [all_images_5856, sensitivity_only, split BARU seed=42]
  |
  └── proposed_split_mapping.csv (frozen, data/source_audit/)
        └── patient_grouped_split_all_images.csv   [all_images_5856, sensitivity_only,
                                                      primary_eligible=false: exact-duplicate
                                                      MD5 f9528b244ccd49f3d89ead90ba09c520
                                                      (IM-0095/IM-0096) span train & test]

master_manifest_derived.csv + exact_duplicates.csv
  |
  └── global_deduplication_keep_set.csv      [5.856 keputusan: 5.824 keep, 32 exclude]
        |
        ├── original_split_deduplicated.csv              [global_exact_deduplicated_5824, primary_control]
        ├── image_stratified_split_deduplicated.csv       [global_exact_deduplicated_5824, primary_control,
        |                                                   split BARU pada cohort 5.824, seed=42]
        └── patient_grouped_split_deduplicated.csv        [global_exact_deduplicated_5824, primary,
                                                             filter dari proposed_split_mapping.csv]
```

## Mengapa cohort deduplicated menjadi analisis utama

Review menemukan bahwa `patient_grouped_split_all_images.csv` (split patient-grouped resmi dari audit
lama) memiliki nol *patient overlap*, tapi tetap bocor pada level citra: dua file identik MD5
(`f9528b244ccd49f3d89ead90ba09c520`) berada di split berbeda karena tercatat sebagai dua pasien
berbeda:

- `test/NORMAL/NORMAL2-IM-0095-0001.jpeg` (patient_id `IM-0095`) -> split `test`
- `test/NORMAL/NORMAL2-IM-0096-0001.jpeg` (patient_id `IM-0096`) -> split `train`

Ini adalah **image-level leakage**: model bisa melihat gambar yang identik byte-per-byte di train dan
dievaluasi ulang pada gambar yang sama (persis) di test. Nol patient-overlap tidak cukup untuk
menjamin tidak ada leakage bila dataset sumber mengandung duplikat exact yang salah label pasien.

Karena itu, `patient_grouped_split_all_images.csv` (dan dua manifest all-images lain) diturunkan
statusnya menjadi **sensitivity-only** - berguna untuk analisis sensitivitas/pembanding historis,
tapi tidak lagi dipakai sebagai kondisi utama penelitian.

## Perbaikan: global deduplication keep-set

`global_deduplication_keep_set.csv` mencatat keputusan keep/exclude untuk seluruh 5.856 citra,
menggunakan **satu aturan deterministik** yang sama persis dengan yang dipakai di
`patient_grouped_split_deduplicated.csv` versi sebelumnya (diverifikasi otomatis saat build - lihat
`src/create_deduplicated_manifest.py`, bagian "consistency check"): dalam setiap grup MD5, simpan
citra dengan nama file yang urut leksikografis terkecil, keluarkan sisanya. Near-duplicate (perceptual
hash) TIDAK PERNAH dipakai sebagai alasan exclude.

Untuk pasangan IM-0095/IM-0096, `NORMAL2-IM-0095-0001.jpeg` < `NORMAL2-IM-0096-0001.jpeg` secara
leksikografis, sehingga IM-0095 (split `test`) menjadi representative dan IM-0096 (split `train`)
dikeluarkan dari manifest patient-grouped deduplicated. Setelah ini, tidak ada lagi exact-duplicate
group yang tersebar lintas split pada manifest utama manapun (diverifikasi sebagai core check di
`validate_splits.py` dan unit test `test_primary_no_cross_split_exact_duplicate_groups`).

## Tiga manifest utama (primary) - cohort identik 5.824 citra

`original_split_deduplicated.csv`, `image_stratified_split_deduplicated.csv`, dan
`patient_grouped_split_deduplicated.csv` semuanya dibangun dari cohort
`global_exact_deduplicated_5824` yang SAMA PERSIS (5.824 `relative_path` identik, hanya assignment
split yang berbeda). Ini yang membuat ketiganya bisa dibandingkan secara adil di Eksperimen A:

- **original_deduplicated** - kontrol: split asli Kaggle (train/val/test) tanpa dikoreksi, overlap
  pasien dipertahankan sebagai karakteristik yang diamati (bukan cacat yang diperbaiki di sini).
- **image_stratified_split_deduplicated** - kontrol: split BARU level-citra (stratified by label,
  seed 42, 80/10/10) yang dibangun ULANG dari cohort 5.824 (bukan hasil filter split 5.856 lama, agar
  rasio 80/10/10 tetap presisi). Overlap pasien tetap terjadi (diukur, tidak dihilangkan) karena
  memang bukan patient-grouped.
- **patient_grouped_deduplicated** - kondisi utama: `StratifiedGroupKFold` TIDAK dijalankan ulang;
  `proposed_split_mapping.csv` difilter dengan keep-set yang sama. Nol patient overlap, nol MD5
  berulang, nol exact-duplicate group lintas split.

## Yang TIDAK diregenerasi

- Audit dataset (frozen 23 Juli 2026): seluruh file di `data/source_audit/`, dibekukan verbatim,
  checksum di `data/checksums/source_audit_sha256.csv`, divalidasi ulang setiap run
  (`validate_source_audit.py`, `validate_splits.py`, dan test `test_source_audit_checksums_unchanged`).
- Patient ID, MD5, perceptual hash, dan blur variance: diambil langsung dari `manifest.csv` lama.
- Grouped split (patient-grouped stratified): `StratifiedGroupKFold` TIDAK dijalankan ulang di fase
  manapun, termasuk saat membangun `patient_grouped_split_deduplicated.csv` - manifest itu hanyalah
  filter dari `proposed_split_mapping.csv`.

## Yang BARU dibuat

- `global_deduplication_keep_set.csv` - keputusan keep/exclude global, satu aturan konsisten untuk
  seluruh dataset.
- `image_stratified_split_deduplicated.csv` - satu-satunya split citra yang benar-benar baru pada
  cohort deduplicated (selain `image_stratified_split_manifest.csv` all-images dari fase sebelumnya).
- `original_split_deduplicated.csv` dan `patient_grouped_split_deduplicated.csv` - manifest filter,
  bukan split baru.

## Tidak ada gambar asli yang dihapus, dipindah, atau dimodifikasi

Seluruh operasi di proyek ini bersifat manifest-level. Dataset asli hanya dibaca, tidak pernah ditulis
ulang - diverifikasi lewat `data/checksums/dataset_files_sha256.csv` (5.856 file) yang dicek ulang
setiap validasi.

## Konfigurasi path (CHEST_XRAY_DATA_ROOT)

Seluruh script menentukan `PROJECT_ROOT` secara dinamis (`Path(__file__).resolve().parents[1]`) dan
me-resolve dataset root lewat `src/project_config.py`, dengan urutan prioritas: environment variable
`CHEST_XRAY_DATA_ROOT` > `config/dataset_config.yaml` (`data_root_windows` di Windows, `data_root` di
POSIX). Tidak ada path sesi agent yang hardcoded di manapun dalam `src/`, `tests/`, atau `notebooks/`
- dijamin oleh test `test_no_hardcoded_session_paths`.

## Catatan untuk peninjauan manusia (per Tahap 1) - SUDAH DIPERBAIKI

Temuan sebelumnya (exact-duplicate IM-0095/IM-0096 tersebar lintas split pada
`patient_grouped_split_all_images.csv`) sudah ditangani lewat mekanisme di atas: manifest all-images
tersebut tetap ada sebagai sensitivity analysis (ditandai `primary_eligible=false`, alasan
didokumentasikan), sementara manifest utama (`*_deduplicated.csv`) sudah bebas dari masalah ini.
Tidak ada isu terbuka yang memerlukan keputusan manusia lebih lanjut pada tahap ini.
