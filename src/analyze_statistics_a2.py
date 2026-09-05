"""analyze_statistics_a2.py

Phase 7 for Experiment A2 (varying-draw leakage replication, CLAUDE.md #6b
point 2). IDENTICAL logic to analyze_statistics.py (Phase 7 Experiment A) and
analyze_statistics_sensitivity_dedup.py - the only differences are the
prediction source (`experiment_a2/predictions`, run_id prefix
`EXP_A__a2__...`) and the output folder (`experiment_a2/statistics`). Written
as a separate script so the original Experiment A confirmatory script
(analyze_statistics.py) is NOT touched at all (protocol §24: confirmatory
analysis must not be modified).

EXPLORATORY (§24) - not part of the Experiment A confirmatory analysis. Main
goal of A2: in Experiment A, the siblings inserted into leaky_train (K=146)
used a SINGLE fixed draw (deterministic per protocol 6.3) shared across all
five model seeds. In A2, the exclusion/replacement (protocol 6.5, i.e. the
CLEAN IMAGES removed to keep the train set size constant) is varied per
model-seed (draw_seed=model_seed: 42/123/456/789/2026) - the inserted
siblings themselves remain deterministic, identical to Experiment A. A2's
question: does varying this exclusion/replacement pattern widen the
uncertainty interval of the leakage effect compared to Experiment A (which
used only 1 fixed draw, so between-seed variation there comes ONLY from
training/initialization noise, not from variation in the contamination
pattern).

Output to experiments/experiment_a2/statistics/:
  - clean_vs_leaky_per_run.csv        (one row per arch x seed, same schema as A)
  - clean_vs_leaky_per_architecture.csv (per-architecture aggregate, same schema as A)
  - a2_vs_expA_ci_width_comparison.csv (additional A2-specific analysis)
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

# Experiment A source (single fixed draw) for the CI-width comparison. This
# path is the frozen deliverable A source_csv - read-only, never written to.
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
        raise SystemExit(f"No prediction files found in {pred_dir} (pattern {run_glob}). Run Phase 5 first.")
    if len(files) != 30:
        print(f"  [warn] found {len(files)} prediction files, expected 30.")
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
    """Compare the width of the leakage-effect uncertainty for A2 (5 seeds,
    draw varying per model_seed) vs Experiment A (5 seeds, one FIXED draw
    pattern). Two metrics are compared per architecture:
      1. between-seed SD of dAUROC (dispersion of point estimates across
         seeds) - a direct proxy for the uncertainty about "which
         contamination pattern is correct".
      2. mean per-run bootstrap-CI width (dAUROC_ci_high - dAUROC_ci_low),
         i.e. the sampling uncertainty WITHIN each run (n=279 anchor) - to
         show that the within-run CI stays relatively unchanged (since n and
         the bootstrap method are the same), so any difference mainly comes
         from between-seed dispersion (point 1), not from a change in method.
    If the Experiment A source files are not found (e.g. running somewhere
    other than locally), this function is skipped with a warning - it does
    NOT stop Phase 7 A2.
    """
    if not (EXP_A_PER_RUN_CSV.exists() and EXP_A_PER_ARCH_CSV.exists()):
        print(f"  [warn] Experiment A source not found at the local location "
              f"({EXP_A_PER_RUN_CSV} / {EXP_A_PER_ARCH_CSV}) - skipping the "
              f"A2 vs Experiment A CI-width comparison. Run this part on a "
              f"session/machine that has access to experiment_a_deliverables.")
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

    print("\n=== A2 vs Experiment A: leakage-effect uncertainty width (EXPLORATORY) ===")
    for _, r in out.iterrows():
        wider = "A2 WIDER" if r["sd_ratio_a2_over_expA"] > 1.05 else (
                "A2 NARROWER" if r["sd_ratio_a2_over_expA"] < 0.95 else "SIMILAR")
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

    print("=== Phase 7 (Experiment A2, EXPLORATORY §24): clean vs leaky per architecture ===")
    for _, r in pa.iterrows():
        print(f"  {r['architecture']:16s} ΔAUROC={r['dAUROC_mean']:+.4f}±{r['dAUROC_sd']:.4f} "
              f"(median {r['dAUROC_median']:+.4f}, range [{r['dAUROC_min']:+.4f},{r['dAUROC_max']:+.4f}]) "
              f"Stouffer p={r['stouffer_p']:.4g} Holm p={r['holm_p']:.4g} "
              f"{'SIG' if r['significant_holm_0_05'] else 'ns'}")

    compare_ci_width_to_exp_a(pr, pa, out_dir)
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
