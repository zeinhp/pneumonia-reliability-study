"""analyze_statistics_a2.py

Fase 7 untuk Eksperimen A2 (varying-draw leakage replication, CLAUDE.md #6b poin 2).
IDENTIK logic-nya dgn analyze_statistics.py (Fase 7 Eksperimen A) dan
analyze_statistics_sensitivity_dedup.py - satu-satunya beda adalah sumber prediksi
(`experiment_a2/predictions`, run_id prefix `EXP_A__a2__...`) dan folder output
(`experiment_a2/statistics`). Dibuat sebagai skrip terpisah supaya skrip
confirmatory Eksperimen A yang asli (analyze_statistics.py) TIDAK disentuh sama
sekali (protokol §24: tidak boleh mengubah confirmatory analysis).

EXPLORATORY (§24) - bukan bagian confirmatory Eksperimen A. Tujuan utama A2:
di Eksperimen A, sibling yang disisipkan ke leaky_train (K=146) SATU draw
tetap (deterministik protokol 6.3) dipakai untuk kelima seed model. Di A2,
exclusion/replacement (protokol 6.5, yaitu CITRA CLEAN yang dikeluarkan utk
menjaga ukuran train) divariasikan per model-seed (draw_seed=model_seed:
42/123/456/789/2026) - sibling yang disisipkan sendiri TETAP deterministik
sama persis Eksperimen A. Pertanyaan A2: apakah variasi pola exclusion/
replacement ini menambah lebar interval ketidakpastian efek-leakage
dibanding Eksperimen A (yang hanya py 1 draw tetap, jadi variasi
antar-seed HANYA berasal dari noise training/inisialisasi, bukan dari
variasi pola kontaminasi).

Output ke experiments/experiment_a2/statistics/:
  - clean_vs_leaky_per_run.csv        (satu baris per arch x seed, sama skema A)
  - clean_vs_leaky_per_architecture.csv (agregat per arsitektur, sama skema A)
  - a2_vs_expA_ci_width_comparison.csv (analisis tambahan khusus A2)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_config as cfg  # noqa: E402

N_BOOT = 2000
BOOT_SEED = 42
ARCHS = ["densenet121", "efficientnet_b0", "swin_tiny"]

# Sumber Eksperimen A (satu draw tetap) untuk perbandingan lebar CI. Path ini
# adalah source_csv freeze deliverable A - dibaca read-only, tidak pernah ditulis.
EXP_A_PER_RUN_CSV = Path(
    r"D:\Claude\Pneumonia\experiment_a_deliverables\source_csv\statistics\clean_vs_leaky_per_run.csv"
)
EXP_A_PER_ARCH_CSV = Path(
    r"D:\Claude\Pneumonia\experiment_a_deliverables\source_csv\statistics\clean_vs_leaky_per_architecture.csv"
)


def _auroc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def _sens_spec(y, pred):
    tp = np.sum((pred == 1) & (y == 1)); fn = np.sum((pred == 0) & (y == 1))
    tn = np.sum((pred == 0) & (y == 0)); fp = np.sum((pred == 1) & (y == 0))
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return sens, spec


def _delta_metrics(y, pc, pl, thr=0.5):
    """All leaky-minus-clean deltas on one (bootstrap) sample."""
    predc, predl = (pc >= thr).astype(int), (pl >= thr).astype(int)
    sensc, specc = _sens_spec(y, predc)
    sensl, specl = _sens_spec(y, predl)
    out = {
        "AUROC": _auroc(y, pl) - _auroc(y, pc),
        "AUPRC": (average_precision_score(y, pl) - average_precision_score(y, pc))
                 if len(np.unique(y)) == 2 else np.nan,
        "sensitivity": sensl - sensc,
        "specificity": specl - specc,
        "balanced_accuracy": ((sensl + specl) / 2) - ((sensc + specc) / 2),
        "Brier": (brier_score_loss(y, pl) - brier_score_loss(y, pc))
                 if len(np.unique(y)) == 2 else np.nan,
    }
    return out


def paired_patient_bootstrap(y, pc, pl):
    """n=2000 resample of patients (anchor = 1 img/patient). Returns observed deltas,
    95% CI, and two-sided bootstrap p-value for ΔAUROC."""
    rng = np.random.default_rng(BOOT_SEED)
    n = len(y)
    obs = _delta_metrics(y, pc, pl)
    boot = {k: np.empty(N_BOOT) for k in obs}
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        d = _delta_metrics(y[idx], pc[idx], pl[idx])
        for k in obs:
            boot[k][b] = d[k]
    res = {}
    for k in obs:
        arr = boot[k][~np.isnan(boot[k])]
        res[k] = {
            "observed": obs[k],
            "mean": float(np.mean(arr)) if arr.size else np.nan,
            "median": float(np.median(arr)) if arr.size else np.nan,
            "ci_low": float(np.percentile(arr, 2.5)) if arr.size else np.nan,
            "ci_high": float(np.percentile(arr, 97.5)) if arr.size else np.nan,
        }
    # two-sided bootstrap p for ΔAUROC
    a = boot["AUROC"][~np.isnan(boot["AUROC"])]
    if a.size:
        p = 2 * min(np.mean(a <= 0), np.mean(a >= 0))
        p = min(max(p, 1.0 / (a.size + 1)), 1.0)
    else:
        p = np.nan
    res["AUROC"]["boot_p_two_sided"] = float(p)
    return res


def mcnemar_p(y, predc, predl):
    """Discordant table of correctness clean vs leaky (16.2)."""
    corr_c = (predc == y); corr_l = (predl == y)
    b = int(np.sum(corr_c & ~corr_l))   # clean right, leaky wrong
    c = int(np.sum(~corr_c & corr_l))   # clean wrong, leaky right
    table = [[int(np.sum(corr_c & corr_l)), b], [c, int(np.sum(~corr_c & ~corr_l))]]
    res = mcnemar(table, exact=(b + c) < 25)
    return float(res.pvalue), b, c


def stouffer(pvals, signs):
    """Signed Stouffer combination of two-sided p-values (equal weights)."""
    pvals = np.asarray(pvals, float); signs = np.asarray(signs, float)
    m = ~np.isnan(pvals)
    if not m.any():
        return np.nan
    z = signs[m] * stats.norm.isf(pvals[m] / 2.0)   # signed z from two-sided p
    Z = np.sum(z) / np.sqrt(m.sum())
    return float(2 * stats.norm.sf(abs(Z)))


def compute_core(pred_dir: Path, out_dir: Path, run_glob: str):
    files = sorted(pred_dir.glob(run_glob))
    if not files:
        raise SystemExit(f"Tidak ada file prediksi di {pred_dir} (pola {run_glob}). Jalankan Fase 5 dulu.")
    if len(files) != 30:
        print(f"  [warn] ditemukan {len(files)} file prediksi, ekspektasi 30.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    per_run_rows, per_arch_rows = [], []

    for arch in ARCHS:
        seed_pvals, seed_signs, seed_daurocs = [], [], []
        for seed in sorted(df[df.architecture == arch].seed.unique()):
            sub = df[(df.architecture == arch) & (df.seed == seed)]
            clean = sub[sub.condition == "clean"][["image_id", "label", "probability", "predicted_label_0_5"]]
            leaky = sub[sub.condition == "leaky"][["image_id", "probability", "predicted_label_0_5"]]
            m = clean.merge(leaky, on="image_id", suffixes=("_clean", "_leaky"))
            y = m["label"].to_numpy(int)
            pc = m["probability_clean"].to_numpy(float); pl = m["probability_leaky"].to_numpy(float)
            predc = m["predicted_label_0_5_clean"].to_numpy(int)
            predl = m["predicted_label_0_5_leaky"].to_numpy(int)

            res = paired_patient_bootstrap(y, pc, pl)
            mp, bcell, ccell = mcnemar_p(y, predc, predl)

            sensc, specc = _sens_spec(y, predc); sensl, specl = _sens_spec(y, predl)

            daur = res["AUROC"]["observed"]
            seed_daurocs.append(daur)
            seed_pvals.append(res["AUROC"]["boot_p_two_sided"])
            seed_signs.append(np.sign(daur) if daur != 0 else 1.0)

            per_run_rows.append({
                "architecture": arch, "seed": int(seed),
                "dAUROC": daur, "dAUROC_ci_low": res["AUROC"]["ci_low"],
                "dAUROC_ci_high": res["AUROC"]["ci_high"], "dAUROC_boot_p": res["AUROC"]["boot_p_two_sided"],
                "dAUPRC": res["AUPRC"]["observed"], "dBrier": res["Brier"]["observed"],
                "dSensitivity_PNEUMONIA": sensl - sensc, "dSpecificity_NORMAL": specl - specc,
                "dBalancedAcc": res["balanced_accuracy"]["observed"],
                "mcnemar_p": mp, "mcnemar_b_cleanRight_leakyWrong": bcell,
                "mcnemar_c_cleanWrong_leakyRight": ccell,
            })

        combined_p = stouffer(seed_pvals, seed_signs)
        arr = np.array(seed_daurocs, float)
        per_arch_rows.append({
            "architecture": arch, "n_seeds": len(seed_daurocs),
            "dAUROC_mean": float(np.nanmean(arr)), "dAUROC_sd": float(np.nanstd(arr, ddof=1)) if arr.size > 1 else np.nan,
            "dAUROC_median": float(np.nanmedian(arr)),
            "dAUROC_min": float(np.nanmin(arr)), "dAUROC_max": float(np.nanmax(arr)),
            "stouffer_p": combined_p,
        })

    pa = pd.DataFrame(per_arch_rows)
    valid = pa["stouffer_p"].notna()
    pa["holm_p"] = np.nan
    pa["significant_holm_0_05"] = False
    if valid.any():
        rej, padj, _, _ = multipletests(pa.loc[valid, "stouffer_p"].to_numpy(), alpha=0.05, method="holm")
        pa.loc[valid, "holm_p"] = padj
        pa.loc[valid, "significant_holm_0_05"] = rej

    pr = pd.DataFrame(per_run_rows)
    pr.to_csv(out_dir / "clean_vs_leaky_per_run.csv", index=False)
    pa.to_csv(out_dir / "clean_vs_leaky_per_architecture.csv", index=False)
    return pr, pa


def compare_ci_width_to_exp_a(pr_a2: pd.DataFrame, pa_a2: pd.DataFrame, out_dir: Path):
    """Bandingkan lebar ketidakpastian efek-leakage A2 (5 seed, draw bervariasi
    per model_seed) vs Eksperimen A (5 seed, draw TETAP satu pola). Dua metrik
    dibandingkan per arsitektur:
      1. between-seed SD dari dAUROC (dispersi titik-estimasi antar seed) -
         proxy langsung ketidakpastian "yang mana pola kontaminasi yang benar".
      2. rata-rata lebar bootstrap-CI per-run (dAUROC_ci_high - dAUROC_ci_low),
         yaitu ketidakpastian sampling di DALAM tiap run (n=279 anchor) - untuk
         menunjukkan bahwa within-run CI relatif tidak berubah (karena n dan
         metode bootstrap sama), sehingga selisih apa pun terutama datang dari
         between-seed dispersion (poin 1), bukan dari perubahan metode.
    Kalau file source Eksperimen A tidak ditemukan (mis. dijalankan bukan di
    lokal), fungsi ini di-skip dengan warning - TIDAK menghentikan Fase 7 A2.
    """
    if not (EXP_A_PER_RUN_CSV.exists() and EXP_A_PER_ARCH_CSV.exists()):
        print(f"  [warn] source Eksperimen A tidak ditemukan di lokasi lokal "
              f"({EXP_A_PER_RUN_CSV} / {EXP_A_PER_ARCH_CSV}) - lewati perbandingan "
              f"lebar CI A2 vs Eksperimen A. Jalankan bagian ini di sesi/mesin yang "
              f"punya akses ke experiment_a_deliverables.")
        return None

    pr_a = pd.read_csv(EXP_A_PER_RUN_CSV)
    pa_a = pd.read_csv(EXP_A_PER_ARCH_CSV)

    rows = []
    for arch in ARCHS:
        a_arch = pa_a[pa_a.architecture == arch].iloc[0]
        a2_arch = pa_a2[pa_a2.architecture == arch].iloc[0]
        a_run = pr_a[pr_a.architecture == arch]
        a2_run = pr_a2[pr_a2.architecture == arch]
        rows.append({
            "architecture": arch,
            "expA_dAUROC_sd_betweenSeed": a_arch["dAUROC_sd"],
            "a2_dAUROC_sd_betweenSeed": a2_arch["dAUROC_sd"],
            "sd_ratio_a2_over_expA": (a2_arch["dAUROC_sd"] / a_arch["dAUROC_sd"])
                                      if a_arch["dAUROC_sd"] not in (0, np.nan) else np.nan,
            "expA_dAUROC_range": a_arch["dAUROC_max"] - a_arch["dAUROC_min"],
            "a2_dAUROC_range": a2_arch["dAUROC_max"] - a2_arch["dAUROC_min"],
            "expA_mean_withinRun_CIwidth": float((a_run["dAUROC_ci_high"] - a_run["dAUROC_ci_low"]).mean()),
            "a2_mean_withinRun_CIwidth": float((a2_run["dAUROC_ci_high"] - a2_run["dAUROC_ci_low"]).mean()),
            "expA_stouffer_p": a_arch["stouffer_p"], "a2_stouffer_p": a2_arch["stouffer_p"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "a2_vs_expA_ci_width_comparison.csv", index=False)

    print("\n=== A2 vs Eksperimen A: lebar ketidakpastian efek-leakage (EXPLORATORY) ===")
    for _, r in out.iterrows():
        wider = "A2 LEBIH LEBAR" if r["sd_ratio_a2_over_expA"] > 1.05 else (
                "A2 LEBIH SEMPIT" if r["sd_ratio_a2_over_expA"] < 0.95 else "MIRIP")
        print(f"  {r['architecture']:16s} between-seed SD: expA={r['expA_dAUROC_sd_betweenSeed']:.4f} "
              f"a2={r['a2_dAUROC_sd_betweenSeed']:.4f} (rasio={r['sd_ratio_a2_over_expA']:.2f}, {wider}); "
              f"within-run CI width: expA={r['expA_mean_withinRun_CIwidth']:.4f} "
              f"a2={r['a2_mean_withinRun_CIwidth']:.4f}")
    return out


def main():
    pred_dir = cfg.get_experiments_dir() / "experiment_a2" / "predictions"
    out_dir = cfg.get_experiments_dir() / "experiment_a2" / "statistics"
    out_dir.mkdir(parents=True, exist_ok=True)

    pr, pa = compute_core(pred_dir, out_dir, "EXP_A__a2__*.anchor_predictions.csv")

    print("=== Fase 7 (Eksperimen A2, EXPLORATORY §24): clean vs leaky per arsitektur ===")
    for _, r in pa.iterrows():
        print(f"  {r['architecture']:16s} ΔAUROC={r['dAUROC_mean']:+.4f}±{r['dAUROC_sd']:.4f} "
              f"(median {r['dAUROC_median']:+.4f}, range [{r['dAUROC_min']:+.4f},{r['dAUROC_max']:+.4f}]) "
              f"Stouffer p={r['stouffer_p']:.4g} Holm p={r['holm_p']:.4g} "
              f"{'SIG' if r['significant_holm_0_05'] else 'ns'}")

    compare_ci_width_to_exp_a(pr, pa, out_dir)
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
