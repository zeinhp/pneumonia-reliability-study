"""analyze_a2_ci_sensitivity.py

Sensitivity check untuk klaim "A2 vs Eksperimen A: lebar ketidakpastian efek-leakage"
(dihasilkan analyze_statistics_a2.py -> a2_vs_expA_ci_width_comparison.csv).

Latar belakang: rasio SD antar-seed A2/Eksperimen A paling dramatis di
EfficientNet-B0 (0.24). Kecurigaan: itu didorong SATU outlier di Eksperimen A
(seed 456, dAUROC=-0.0203, jauh dari 4 seed lain yang -0.003..-0.008), bukan
pola sistematis bahwa variasi draw kontaminasi A2 mengurangi ketidakpastian.
n=5 per kondisi juga terlalu kecil untuk membandingkan varians secara andal
tanpa uji eksplisit.

Skrip ini (murni CPU, TIDAK butuh GPU/checkpoint - aman dijalankan kapan pun,
termasuk saat GPU dipakai job lain):
  1. Leave-one-seed-out SD untuk EfficientNet-B0 Eksperimen A (5 versi, masing2
     buang 1 seed) -> tunjukkan seberapa besar seed 456 mendominasi SD.
  2. SD Eksperimen A tanpa seed 456 vs SD A2 (recompute rasio).
  3. F-test (rasio varians, asumsi longgar - GARIS BESAR sensitivity, bukan
     inferensi formal krn n kecil & bukan sampel acak sebenarnya) untuk
     ketiga arsitektur: apakah rasio SD A2/A yang teramati konsisten dengan
     H0 varians sama, dengan p-value.
  4. Bootstrap kasar (resample-with-replacement 5 titik seed, n=2000) buat
     interval rasio SD A2/A per arsitektur - supaya ada gambaran incertitude
     dari estimasi rasio itu sendiri (bukan cuma titik rasio tunggal).

Input: dua file per_run (Eksperimen A - lokal read-only source_csv freeze,
A2 - hasil experiments/experiment_a2/statistics/clean_vs_leaky_per_run.csv).
Path Eksperimen A di-override lewat argumen --expa-per-run kalau dijalankan
bukan di lokal (mis. di Hub Linux, path Windows default tidak akan ada).

Output: cetak ke stdout + simpan
experiments/experiment_a2/statistics/a2_vs_expA_ci_sensitivity_check.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_config as cfg  # noqa: E402

ARCHS = ["densenet121", "efficientnet_b0", "swin_tiny"]
N_BOOT = 2000
BOOT_SEED = 42

DEFAULT_EXP_A_PER_RUN = Path(
    r"D:\Claude\Pneumonia\experiment_a_deliverables\source_csv\statistics\clean_vs_leaky_per_run.csv"
)


def parse_args():
    p = argparse.ArgumentParser(description="Sensitivity check: A2 vs Eksperimen A CI width.")
    p.add_argument("--expa-per-run", default=str(DEFAULT_EXP_A_PER_RUN),
                   help="Path clean_vs_leaky_per_run.csv Eksperimen A (source_csv freeze). "
                        "WAJIB di-set eksplisit kalau dijalankan di Hub/Linux - default "
                        "path Windows lokal tidak akan ada di sana.")
    p.add_argument("--a2-per-run", default=None,
                   help="Path clean_vs_leaky_per_run.csv A2. Default: "
                        "experiments/experiment_a2/statistics/clean_vs_leaky_per_run.csv")
    return p.parse_args()


def leave_one_out_sd(vals):
    """SD (ddof=1) tiap kali buang 1 titik, dari n titik."""
    vals = np.asarray(vals, float)
    n = len(vals)
    out = []
    for i in range(n):
        sub = np.delete(vals, i)
        out.append(float(np.std(sub, ddof=1)))
    return out


def bootstrap_sd_ratio(vals_a, vals_a2, n_boot=N_BOOT, seed=BOOT_SEED):
    """Bootstrap kasar (resample seed-level titik dgn penggantian) utk interval
    rasio SD(A2)/SD(A). n kecil (5) jadi interval ini indikatif saja, bukan
    jaminan cakupan 95% yang ketat - didokumentasikan sbg keterbatasan."""
    rng = np.random.default_rng(seed)
    vals_a = np.asarray(vals_a, float); vals_a2 = np.asarray(vals_a2, float)
    na, na2 = len(vals_a), len(vals_a2)
    ratios = np.empty(n_boot)
    for b in range(n_boot):
        sa = vals_a[rng.integers(0, na, na)]
        sa2 = vals_a2[rng.integers(0, na2, na2)]
        sd_a = np.std(sa, ddof=1) if np.std(sa, ddof=1) > 0 else np.nan
        sd_a2 = np.std(sa2, ddof=1)
        ratios[b] = sd_a2 / sd_a if sd_a and not np.isnan(sd_a) else np.nan
    ratios = ratios[~np.isnan(ratios) & ~np.isinf(ratios)]
    return {
        "boot_ratio_median": float(np.median(ratios)) if ratios.size else np.nan,
        "boot_ratio_ci_low": float(np.percentile(ratios, 2.5)) if ratios.size else np.nan,
        "boot_ratio_ci_high": float(np.percentile(ratios, 97.5)) if ratios.size else np.nan,
        "boot_n_valid": int(ratios.size),
    }


def f_test_variance_ratio(vals_a, vals_a2):
    """F-test klasik utk H0: varians sama. CAVEAT KERAS: ini bukan dua sampel
    acak independen dari populasi yang sama dalam arti klasik (seed A dan
    seed A2 draw beda, n=5 sangat kecil, normalitas tak diverifikasi) - hanya
    dipakai sbg sensitivity indikatif, BUKAN klaim inferensial formal."""
    vals_a = np.asarray(vals_a, float); vals_a2 = np.asarray(vals_a2, float)
    var_a = np.var(vals_a, ddof=1); var_a2 = np.var(vals_a2, ddof=1)
    if var_a == 0 or var_a2 == 0:
        return np.nan, np.nan
    F = var_a2 / var_a
    df1, df2 = len(vals_a2) - 1, len(vals_a) - 1
    p_one_sided = stats.f.cdf(F, df1, df2) if F < 1 else stats.f.sf(F, df1, df2)
    p_two_sided = min(1.0, 2 * p_one_sided)
    return float(F), float(p_two_sided)


def main():
    args = parse_args()
    expa_path = Path(args.expa_per_run)
    a2_path = Path(args.a2_per_run) if args.a2_per_run else (
        cfg.get_experiments_dir() / "experiment_a2" / "statistics" / "clean_vs_leaky_per_run.csv")

    if not expa_path.exists():
        raise SystemExit(f"File Eksperimen A tidak ditemukan: {expa_path}. "
                          f"Set --expa-per-run eksplisit (mis. setelah upload ke Hub).")
    if not a2_path.exists():
        raise SystemExit(f"File A2 tidak ditemukan: {a2_path}. Jalankan Fase 7 A2 dulu.")

    dfa = pd.read_csv(expa_path)
    dfa2 = pd.read_csv(a2_path)

    print("=== Sensitivity check: outlier & robustness rasio SD A2/Eksperimen A ===\n")
    rows = []
    for arch in ARCHS:
        va = dfa[dfa.architecture == arch].sort_values("seed")["dAUROC"].to_numpy()
        va2 = dfa2[dfa2.architecture == arch].sort_values("seed")["dAUROC"].to_numpy()
        seeds_a = dfa[dfa.architecture == arch].sort_values("seed")["seed"].to_numpy()

        sd_a_full = float(np.std(va, ddof=1))
        sd_a2_full = float(np.std(va2, ddof=1))
        loo = leave_one_out_sd(va)
        loo_pairs = list(zip(seeds_a.tolist(), loo))

        print(f"--- {arch} ---")
        print(f"  Eksperimen A dAUROC per seed: {dict(zip(seeds_a.tolist(), va.round(5).tolist()))}")
        print(f"  SD penuh (5 seed): expA={sd_a_full:.5f}  a2={sd_a2_full:.5f}  "
              f"rasio={sd_a2_full / sd_a_full:.3f}")
        print("  Leave-one-out SD Eksperimen A (buang 1 seed):")
        max_drop_seed, max_drop_sd = None, -1
        for seed_dropped, sd_loo in loo_pairs:
            print(f"    buang seed {seed_dropped}: SD sisanya = {sd_loo:.5f}")
            if sd_a_full - sd_loo > max_drop_sd:
                max_drop_sd = sd_a_full - sd_loo
                max_drop_seed = seed_dropped

        F, p_f = f_test_variance_ratio(va, va2)
        boot = bootstrap_sd_ratio(va, va2)

        print(f"  Seed paling dominan menaikkan SD (penurunan SD terbesar saat dibuang): "
              f"seed {max_drop_seed} (SD turun {max_drop_sd:.5f} kalau dibuang)")
        print(f"  F-test rasio varians (indikatif, n=5 vs n=5, CAVEAT n kecil): "
              f"F={F:.3f} p={p_f:.4f}")
        print(f"  Bootstrap kasar rasio SD A2/A: median={boot['boot_ratio_median']:.3f} "
              f"95% CI [{boot['boot_ratio_ci_low']:.3f}, {boot['boot_ratio_ci_high']:.3f}] "
              f"(n_valid={boot['boot_n_valid']}/{N_BOOT})\n")

        rows.append({
            "architecture": arch,
            "expA_sd_full5": sd_a_full, "a2_sd_full5": sd_a2_full,
            "ratio_full5": sd_a2_full / sd_a_full,
            "expA_most_influential_seed": max_drop_seed,
            "expA_sd_drop_if_removed": max_drop_sd,
            "f_test_F": F, "f_test_p_two_sided": p_f,
            "boot_ratio_median": boot["boot_ratio_median"],
            "boot_ratio_ci_low": boot["boot_ratio_ci_low"],
            "boot_ratio_ci_high": boot["boot_ratio_ci_high"],
        })

    out = pd.DataFrame(rows)
    out_dir = cfg.get_experiments_dir() / "experiment_a2" / "statistics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "a2_vs_expA_ci_sensitivity_check.csv"
    out.to_csv(out_path, index=False)
    print(f"Output: {out_path}")
    print("\nCatatan interpretasi WAJIB dibaca sebelum dipakai di laporan:")
    print("  - n=5 per kondisi SANGAT kecil; F-test & bootstrap di sini indikatif,")
    print("    bukan inferensi formal yang ketat (asumsi normalitas/independensi longgar).")
    print("  - Kalau 'expA_sd_drop_if_removed' besar utk satu seed tertentu (terutama")
    print("    EfficientNet-B0 seed 456 yg diduga outlier), klaim 'A2 lebih sempit' harus")
    print("    dihedge sbg 'didominasi 1 run', BUKAN pola sistematis variasi-draw.")


if __name__ == "__main__":
    main()
