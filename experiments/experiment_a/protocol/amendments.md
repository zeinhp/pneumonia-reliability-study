# Amendments & Retry Log — Eksperimen A

Dokumen ini mencatat (a) **retry kegagalan teknis** sesuai protokol §23, dan (b) amendment
desain sesuai §24. Semua entri di bawah adalah **retry teknis (§23)** — TIDAK ada perubahan
desain eksperimen, hipotesis, split, arsitektur, atau kriteria evaluasi. Test set tidak
dibuka untuk mengambil keputusan apa pun di sini.

---

## §23-A — Fase 6 (benchmark sekunder), split `original`: val_loader kosong

**Tanggal dicatat:** 2026-07-26
**Fase:** 6 (benchmark sekunder, DESKRIPTIF; bukan bagian confirmatory Fase 4–5).
**Kelas kegagalan:** kegagalan teknis pipeline (bukan amendment desain).

### Gejala
9 run untuk split `original` (3 arsitektur × 3 seed) gagal dengan `rc=1`. Traceback:
`RuntimeError: torch.cat(): expected a non-empty list of Tensors` di `engine.evaluate`.

### Sebab (root cause)
Manifest `original_split_deduplicated.csv` melabeli baris validasi dengan nilai split
**`val`** (hanya 16 citra — split validasi asli Kermany yang memang kecil), sedangkan
`train_benchmark.py` menyaring loader validasi dengan string **`"validation"`**. Akibatnya
`val_loader` kosong dan `engine.evaluate` memanggil `torch.cat([])`.

### Perbaikan (changed_parameters) — sesuai protokol §6.10
Untuk split `original` SAJA, `train_benchmark.py` sekarang **melewati evaluasi validasi
sepenuhnya**: model dilatih `--original-epochs 9` epoch fixed (9 = median `best_epoch` pada
controlled clean patient-grouped, keputusan beku), lalu model epoch terakhir disimpan.
Validasi 16-citra memang tidak boleh dipakai untuk seleksi apa pun (protokol §6.10). Split
`patient_grouped` dan `image_level` TIDAK berubah (tetap pakai early-stop berbasis val).

Perubahan hanya menyentuh alur kontrol training benchmark; tidak ada perubahan pada data,
label, arsitektur, seed, atau metrik evaluasi test.

### Verifikasi sebelum rerun
- Smoke test split `original` dijalankan ulang → lolos (tidak ada `torch.cat` kosong).
- Master log lama dibackup (`fase6_master.log.bak`); entri gagal dibersihkan dari log aktif.

### Retry — 9 run original, semua rc=0
Sumber timestamp: `fase6_original_nohup.out`. Semua UTC, 2026-07-25.

| original_run_id                         | seed | retry_start (UTC)   | retry_end (UTC)     | rc |
|-----------------------------------------|------|---------------------|---------------------|----|
| benchmark__original__densenet121__s42   | 42   | 2026-07-25 14:49:47 | 2026-07-25 14:56:35 | 0  |
| benchmark__original__densenet121__s456  | 456  | 2026-07-25 14:56:35 | 2026-07-25 15:05:07 | 0  |
| benchmark__original__densenet121__s2026 | 2026 | 2026-07-25 15:05:07 | 2026-07-25 15:14:09 | 0  |
| benchmark__original__efficientnet_b0__s42   | 42   | 2026-07-25 15:14:09 | 2026-07-25 15:19:25 | 0  |
| benchmark__original__efficientnet_b0__s456  | 456  | 2026-07-25 15:19:25 | 2026-07-25 15:24:44 | 0  |
| benchmark__original__efficientnet_b0__s2026 | 2026 | 2026-07-25 15:24:44 | 2026-07-25 15:30:01 | 0  |
| benchmark__original__swin_tiny__s42     | 42   | 2026-07-25 15:30:01 | 2026-07-25 15:47:42 | 0  |
| benchmark__original__swin_tiny__s456    | 456  | 2026-07-25 15:47:42 | 2026-07-25 16:05:06 | 0  |
| benchmark__original__swin_tiny__s2026   | 2026 | 2026-07-25 16:05:06 | 2026-07-25 16:22:18 | 0  |

`n_train` original = 5190; `best_epoch` tercatat = 9 (fixed) untuk semua 9 run
(lihat `benchmark/analysis/benchmark_per_run.csv`).

### Dampak terhadap kesimpulan
Tidak ada. Benchmark Fase 6 bersifat DESKRIPTIF (protokol §6.9/H6/§16); tidak ada uji
hipotesis lintas-split. Perbaikan hanya memungkinkan run original selesai; angka AUROC test
original tetap dilaporkan apa adanya (justru lebih rendah dari patient-grouped — efek
distribusi test, bukan leakage; lihat laporan §benchmark).

---

## §24 — Amendment desain (confirmatory)

**NIHIL.** Tidak ada amendment desain setelah test set dibuka (Fase 5). Seluruh analisis
confirmatory (Fase 7 §16.1–16.3) dijalankan sesuai protokol pra-registrasi v1.0. Analisis
apa pun yang diusulkan setelah titik ini WAJIB dilabeli *exploratory* (protokol §24, CLAUDE.md §8).
