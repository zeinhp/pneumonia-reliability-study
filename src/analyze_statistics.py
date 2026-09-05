"""analyze_statistics.py

Phase 7 (protocol section 15-16): confirmatory clean-vs-leaky statistical
analysis on the common anchor test, using the Phase 5 prediction files.

Seed-aggregation decision (FROZEN, decided by the user before the test set was
opened -> stays confirmatory): Option C - compute per (architecture, seed),
then combine the 5 per-seed p-values per architecture with the Stouffer
method (signed-Z), then Holm-Bonferroni across the 3 architectures.

Per (architecture, seed):
  - paired patient-cluster bootstrap (n=2000, seed 42) for the metric deltas
    (AUROC, AUPRC, balanced_accuracy, sensitivity, specificity, Brier) -> mean,
    median, 95% CI, two-sided bootstrap p-value (16.1).
  - McNemar at threshold 0.5 (16.2).
  - sensitivity/specificity deltas computed SEPARATELY per class PNEUMONIA vs
    NORMAL (required by frozen decision #1 in CLAUDE.md).
Per architecture:
  - Stouffer combines the 5 ΔAUROC p-values (signed) -> 1 p-value.
  - seed aggregation 16.6: mean±sd, median, min-max.
Holm family: 3 combined p-values (one per architecture).

Output to experiments/experiment_a/statistics/.
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


def main():
    pred_dir = cfg.get_experiment_a_dir() / "predictions"
    out_dir = cfg.get_experiment_a_dir() / "statistics"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(pred_dir.glob("EXP_A__controlled__*.anchor_predictions.csv"))
    if not files:
        raise SystemExit(f"No prediction files found in {pred_dir}. Run Phase 5 first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    per_run_rows, per_arch_rows = [], []
    arch_pvals = {}   # arch -> list of (seed, p, sign)

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

            # per-class sens/spec deltas (decision #1): PNEUMONIA=1 -> sensitivity; NORMAL=0 -> specificity
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
        arch_pvals[arch] = combined_p
        arr = np.array(seed_daurocs, float)
        per_arch_rows.append({
            "architecture": arch, "n_seeds": len(seed_daurocs),
            "dAUROC_mean": float(np.nanmean(arr)), "dAUROC_sd": float(np.nanstd(arr, ddof=1)) if arr.size > 1 else np.nan,
            "dAUROC_median": float(np.nanmedian(arr)),
            "dAUROC_min": float(np.nanmin(arr)), "dAUROC_max": float(np.nanmax(arr)),
            "stouffer_p": combined_p,
        })

    # Holm across the 3 architectures (family: clean-vs-leaky per arch)
    pa = pd.DataFrame(per_arch_rows)
    valid = pa["stouffer_p"].notna()
    pa["holm_p"] = np.nan
    pa["significant_holm_0_05"] = False
    if valid.any():
        rej, padj, _, _ = multipletests(pa.loc[valid, "stouffer_p"].to_numpy(), alpha=0.05, method="holm")
        pa.loc[valid, "holm_p"] = padj
        pa.loc[valid, "significant_holm_0_05"] = rej

    pd.DataFrame(per_run_rows).to_csv(out_dir / "clean_vs_leaky_per_run.csv", index=False)
    pa.to_csv(out_dir / "clean_vs_leaky_per_architecture.csv", index=False)

    print("=== Phase 7: clean vs leaky (per architecture, Holm across 3) ===")
    for _, r in pa.iterrows():
        print(f"  {r['architecture']:16s} ΔAUROC={r['dAUROC_mean']:+.4f}±{r['dAUROC_sd']:.4f} "
              f"(median {r['dAUROC_median']:+.4f}, range [{r['dAUROC_min']:+.4f},{r['dAUROC_max']:+.4f}]) "
              f"Stouffer p={r['stouffer_p']:.4g} Holm p={r['holm_p']:.4g} "
              f"{'SIG' if r['significant_holm_0_05'] else 'ns'}")
    print(f"\nOutput: {out_dir}")
    print("Remember the frozen interpretation rule (CLAUDE.md #3): the headline is the size of "
          "the inflation + CI, not the p-value alone; a null result != 'no effect' (see MDES).")


if __name__ == "__main__":
    main()
