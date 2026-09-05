"""analyze_confirmatory_extra.py

Fase 7 - analisis confirmatory tambahan pada COMMON ANCHOR TEST (memakai file
prediksi Fase 5), melengkapi analyze_statistics.py (yang hanya meng-cover 16.1-16.2
clean-vs-leaky). Seluruh analisis di sini SUDAH dipra-registrasi di protokol.docx
sebelum test dibuka, sehingga tetap CONFIRMATORY (bagian 24 hanya melarang analisis
BARU diklaim confirmatory).

Cakupan:
  A. 16.3  Perbandingan tiga arsitektur (paired patient-level bootstrap ΔAUROC untuk
           tiap pasang: densenet-vs-efficientnet, densenet-vs-swin, efficientnet-vs-swin),
           DILAKUKAN TERPISAH pada kondisi clean dan leaky. Agregasi seed = Opsi C
           (dibekukan): Stouffer signed-Z gabung p 5 seed per pasang, lalu Holm-Bonferroni
           lintas 3 pasang di DALAM tiap kondisi (16.4). alpha=0.05 two-sided (16.5).
  B. 17    Analisis peringkat arsitektur: rank per (kondisi, seed) berdasar AUROC,
           frekuensi perubahan peringkat clean-vs-leaky, Spearman rank correlation.
  C. 18    Analisis kesalahan pada anchor test: kasus selalu salah di semua seed,
           kasus benar-di-leaky/salah-di-clean dan sebaliknya, confidence pada kesalahan,
           distribusi label kasus tidak stabil.
  D. 14.3/14.4  Calibration baseline (Brier, NLL, ECE) + reliability-diagram data (dan
           PNG bila matplotlib ada) + uncertainty baseline (confidence pada prediksi
           benar vs salah). Eksperimen A hanya baseline; kalibrasi khusus di Eksperimen D.

Bootstrap: patient-cluster, n=2000, seed 42 (bagian 15). Pada anchor test 1 citra/pasien
sehingga resample baris = resample pasien.

Output ke experiments/experiment_a/statistics/ dan figures/.

Contoh:
  python src/analyze_confirmatory_extra.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_config as cfg  # noqa: E402

N_BOOT = 2000
BOOT_SEED = 42
ARCHS = ["densenet121", "efficientnet_b0", "swin_tiny"]
CONDITIONS = ["clean", "leaky"]
ALPHA = 0.05


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _auroc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def _ece(y, p, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(np.mean(y[mask]) - np.mean(p[mask]))
    return float(ece)


def paired_auroc_bootstrap(y, pa, pb):
    """Paired patient bootstrap for ΔAUROC = AUROC(b) - AUROC(a) on the same test.
    Returns observed delta, 95% CI, two-sided bootstrap p-value."""
    rng = np.random.default_rng(BOOT_SEED)
    n = len(y)
    obs = _auroc(y, pb) - _auroc(y, pa)
    boot = np.empty(N_BOOT)
    for k in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boot[k] = _auroc(y[idx], pb[idx]) - _auroc(y[idx], pa[idx])
    boot = boot[~np.isnan(boot)]
    if not boot.size:
        return obs, np.nan, np.nan, np.nan
    ci_low, ci_high = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    p = 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))
    p = float(min(max(p, 1.0 / (boot.size + 1)), 1.0))
    return float(obs), ci_low, ci_high, p


def stouffer(pvals, signs):
    pvals = np.asarray(pvals, float); signs = np.asarray(signs, float)
    m = ~np.isnan(pvals)
    if not m.any():
        return np.nan
    z = signs[m] * stats.norm.isf(pvals[m] / 2.0)
    Z = np.sum(z) / np.sqrt(m.sum())
    return float(2 * stats.norm.sf(abs(Z)))


def _pair_frame(df, cond, seed, arch_a, arch_b):
    """Merge two architectures' anchor predictions on image_id (same cond, seed)."""
    sa = df[(df.condition == cond) & (df.seed == seed) & (df.architecture == arch_a)]
    sb = df[(df.condition == cond) & (df.seed == seed) & (df.architecture == arch_b)]
    a = sa[["image_id", "label", "probability"]]
    b = sb[["image_id", "probability"]]
    m = a.merge(b, on="image_id", suffixes=("_a", "_b"))
    y = m["label"].to_numpy(int)
    return y, m["probability_a"].to_numpy(float), m["probability_b"].to_numpy(float)


# --------------------------------------------------------------------------- #
# A. 16.3 architecture comparison (per condition)
# --------------------------------------------------------------------------- #
def architecture_comparison(df, out_dir):
    per_seed_rows, per_pair_rows = [], []
    for cond in CONDITIONS:
        seeds = sorted(df[df.condition == cond].seed.unique())
        pair_pvals = {}
        for arch_a, arch_b in combinations(ARCHS, 2):
            sp, ss, sd = [], [], []
            for seed in seeds:
                y, pa, pb = _pair_frame(df, cond, seed, arch_a, arch_b)
                if y.size == 0:
                    continue
                d, lo, hi, p = paired_auroc_bootstrap(y, pa, pb)
                sp.append(p); ss.append(np.sign(d) if d != 0 else 1.0); sd.append(d)
                per_seed_rows.append({"condition": cond, "pair": f"{arch_b}_minus_{arch_a}",
                                      "seed": int(seed), "dAUROC": d, "ci_low": lo,
                                      "ci_high": hi, "boot_p": p})
            comb = stouffer(sp, ss)
            arr = np.array(sd, float)
            pair_pvals[(cond, f"{arch_b}_minus_{arch_a}")] = comb
            per_pair_rows.append({
                "condition": cond, "pair": f"{arch_b}_minus_{arch_a}", "n_seeds": len(sd),
                "dAUROC_mean": float(np.nanmean(arr)) if arr.size else np.nan,
                "dAUROC_sd": float(np.nanstd(arr, ddof=1)) if arr.size > 1 else np.nan,
                "dAUROC_median": float(np.nanmedian(arr)) if arr.size else np.nan,
                "dAUROC_min": float(np.nanmin(arr)) if arr.size else np.nan,
                "dAUROC_max": float(np.nanmax(arr)) if arr.size else np.nan,
                "stouffer_p": comb,
            })
    pair = pd.DataFrame(per_pair_rows)
    # Holm within each condition family (3 pairs)
    pair["holm_p"] = np.nan
    pair["significant_holm_0_05"] = False
    for cond in CONDITIONS:
        mask = (pair.condition == cond) & pair.stouffer_p.notna()
        if mask.any():
            rej, padj, _, _ = multipletests(pair.loc[mask, "stouffer_p"].to_numpy(),
                                            alpha=ALPHA, method="holm")
            pair.loc[mask, "holm_p"] = padj
            pair.loc[mask, "significant_holm_0_05"] = rej
    pd.DataFrame(per_seed_rows).to_csv(out_dir / "arch_comparison_per_seed.csv", index=False)
    pair.to_csv(out_dir / "arch_comparison_per_pair.csv", index=False)

    print("=== 16.3 Perbandingan arsitektur (paired bootstrap ΔAUROC, Holm per kondisi) ===")
    for cond in CONDITIONS:
        print(f"  [{cond}]")
        for _, r in pair[pair.condition == cond].iterrows():
            print(f"    {r['pair']:34s} ΔAUROC={r['dAUROC_mean']:+.4f}±{r['dAUROC_sd']:.4f} "
                  f"Stouffer p={r['stouffer_p']:.4g} Holm p={r['holm_p']:.4g} "
                  f"{'SIG' if r['significant_holm_0_05'] else 'ns'}")
    return pair


# --------------------------------------------------------------------------- #
# B. 17 architecture ranking
# --------------------------------------------------------------------------- #
def rank_analysis(df, out_dir):
    rank_rows = []
    auroc_by = {}
    for cond in CONDITIONS:
        for seed in sorted(df[df.condition == cond].seed.unique()):
            aurocs = {}
            for arch in ARCHS:
                s = df[(df.condition == cond) & (df.seed == seed) & (df.architecture == arch)]
                aurocs[arch] = _auroc(s["label"].to_numpy(int), s["probability"].to_numpy(float))
            auroc_by[(cond, seed)] = aurocs
            order = sorted(ARCHS, key=lambda a: (-aurocs[a] if not np.isnan(aurocs[a]) else np.inf))
            ranks = {a: order.index(a) + 1 for a in ARCHS}  # 1 = best
            row = {"condition": cond, "seed": int(seed)}
            row.update({f"auroc_{a}": aurocs[a] for a in ARCHS})
            row.update({f"rank_{a}": ranks[a] for a in ARCHS})
            rank_rows.append(row)
    ranks_df = pd.DataFrame(rank_rows)
    ranks_df.to_csv(out_dir / "architecture_ranks_per_seed.csv", index=False)

    # rank change frequency + Spearman (clean vs leaky), matched by seed
    common_seeds = sorted(set(df[df.condition == "clean"].seed.unique())
                          & set(df[df.condition == "leaky"].seed.unique()))
    changes, spearmans = 0, []
    per_seed = []
    for seed in common_seeds:
        rc = [auroc_by[("clean", seed)][a] for a in ARCHS]
        rl = [auroc_by[("leaky", seed)][a] for a in ARCHS]
        order_c = [ARCHS[i] for i in np.argsort([-x for x in rc])]
        order_l = [ARCHS[i] for i in np.argsort([-x for x in rl])]
        changed = order_c != order_l
        changes += int(changed)
        rho = stats.spearmanr(
            [order_c.index(a) for a in ARCHS], [order_l.index(a) for a in ARCHS]
        ).statistic if len(ARCHS) > 1 else np.nan
        spearmans.append(rho)
        per_seed.append({"seed": int(seed), "order_clean": ">".join(order_c),
                         "order_leaky": ">".join(order_l), "ordering_changed": changed,
                         "spearman_rho": rho})
    summary = pd.DataFrame(per_seed)
    summary.to_csv(out_dir / "rank_stability_per_seed.csv", index=False)

    print("\n=== 17 Analisis peringkat arsitektur (clean vs leaky) ===")
    print(f"  Ordering berubah pada {changes}/{len(common_seeds)} seed.")
    if spearmans:
        print(f"  Spearman rho (rata-rata): {np.nanmean(spearmans):+.3f}")
    for _, r in summary.iterrows():
        flag = "  <-- BERUBAH" if r["ordering_changed"] else ""
        print(f"    seed {r['seed']}: clean {r['order_clean']} | leaky {r['order_leaky']}{flag}")
    return ranks_df


# --------------------------------------------------------------------------- #
# C. 18 error analysis
# --------------------------------------------------------------------------- #
def error_analysis(df, out_dir):
    df = df.copy()
    df["correct"] = (df["predicted_label_0_5"].to_numpy(int) == df["label"].to_numpy(int))
    if "entropy" not in df.columns:
        p = df["probability"].to_numpy(float).clip(1e-12, 1 - 1e-12)
        df["entropy"] = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    df["confidence"] = np.maximum(df["probability"], 1 - df["probability"])

    summary_rows, case_rows = [], []
    for arch in ARCHS:
        sub = df[df.architecture == arch]
        n_seeds_c = sub[sub.condition == "clean"].seed.nunique()
        n_seeds_l = sub[sub.condition == "leaky"].seed.nunique()

        # correctness matrix per image across seeds
        def correctness(cond):
            s = sub[sub.condition == cond]
            return s.pivot_table(index="image_id", columns="seed", values="correct",
                                 aggfunc="first")
        cc, cl = correctness("clean"), correctness("leaky")
        always_wrong_clean = int((cc.sum(axis=1) == 0).sum()) if cc.size else 0
        always_wrong_leaky = int((cl.sum(axis=1) == 0).sum()) if cl.size else 0
        unstable_clean = int(((cc.sum(axis=1) > 0) & (cc.sum(axis=1) < n_seeds_c)).sum()) if cc.size else 0
        unstable_leaky = int(((cl.sum(axis=1) > 0) & (cl.sum(axis=1) < n_seeds_l)).sum()) if cl.size else 0

        # per (seed,image) flip clean<->leaky
        merged = sub[sub.condition == "clean"][["image_id", "seed", "label", "correct"]].merge(
            sub[sub.condition == "leaky"][["image_id", "seed", "correct"]],
            on=["image_id", "seed"], suffixes=("_clean", "_leaky"))
        right_leaky_wrong_clean = int(((~merged.correct_clean) & (merged.correct_leaky)).sum())
        wrong_leaky_right_clean = int(((merged.correct_clean) & (~merged.correct_leaky)).sum())

        conf_wrong = float(sub.loc[~sub.correct, "confidence"].mean()) if (~sub.correct).any() else np.nan
        conf_right = float(sub.loc[sub.correct, "confidence"].mean()) if (sub.correct).any() else np.nan

        summary_rows.append({
            "architecture": arch,
            "always_wrong_all_seeds_clean": always_wrong_clean,
            "always_wrong_all_seeds_leaky": always_wrong_leaky,
            "unstable_cases_clean": unstable_clean, "unstable_cases_leaky": unstable_leaky,
            "right_leaky_wrong_clean": right_leaky_wrong_clean,
            "wrong_leaky_right_clean": wrong_leaky_right_clean,
            "mean_confidence_on_errors": conf_wrong,
            "mean_confidence_on_correct": conf_right,
        })

        # label distribution of always-wrong (clean) cases
        if cc.size:
            aw_ids = cc.index[cc.sum(axis=1) == 0]
            lab = sub[sub.image_id.isin(aw_ids)][["image_id", "label"]].drop_duplicates()
            for _, rr in lab.iterrows():
                case_rows.append({"architecture": arch, "image_id": rr["image_id"],
                                  "label": int(rr["label"]), "category": "always_wrong_clean"})

    pd.DataFrame(summary_rows).to_csv(out_dir / "error_analysis_summary.csv", index=False)
    pd.DataFrame(case_rows).to_csv(out_dir / "error_analysis_always_wrong_clean.csv", index=False)

    print("\n=== 18 Analisis kesalahan (anchor test) ===")
    for r in summary_rows:
        print(f"  {r['architecture']:16s} always-wrong(clean/leaky)={r['always_wrong_all_seeds_clean']}"
              f"/{r['always_wrong_all_seeds_leaky']}  right@leaky/wrong@clean={r['right_leaky_wrong_clean']}"
              f"  wrong@leaky/right@clean={r['wrong_leaky_right_clean']}  "
              f"conf(err/ok)={r['mean_confidence_on_errors']:.3f}/{r['mean_confidence_on_correct']:.3f}")


# --------------------------------------------------------------------------- #
# D. 14.3 / 14.4 calibration & uncertainty baseline
# --------------------------------------------------------------------------- #
def calibration_uncertainty(df, out_dir, fig_dir):
    df = df.copy()
    df["correct"] = (df["predicted_label_0_5"].to_numpy(int) == df["label"].to_numpy(int))
    df["confidence"] = np.maximum(df["probability"], 1 - df["probability"])
    if "entropy" not in df.columns:
        p = df["probability"].to_numpy(float).clip(1e-12, 1 - 1e-12)
        df["entropy"] = -(p * np.log(p) + (1 - p) * np.log(1 - p))

    cal_rows, unc_rows, rel_rows = [], [], []
    for arch in ARCHS:
        for cond in CONDITIONS:
            block = df[(df.architecture == arch) & (df.condition == cond)]
            if block.empty:
                continue
            # per-seed calibration
            for seed in sorted(block.seed.unique()):
                s = block[block.seed == seed]
                y = s["label"].to_numpy(int); p = s["probability"].to_numpy(float)
                both = len(np.unique(y)) == 2
                try:
                    nll = float(log_loss(y, p, labels=[0, 1]))
                except Exception:
                    nll = np.nan
                cal_rows.append({"architecture": arch, "condition": cond, "seed": int(seed),
                                 "brier": float(brier_score_loss(y, p)) if both else float(np.mean((p - y) ** 2)),
                                 "nll": nll, "ece": _ece(y, p)})
            # uncertainty: confidence on correct vs wrong (pooled seeds)
            unc_rows.append({
                "architecture": arch, "condition": cond,
                "mean_conf_correct": float(block.loc[block.correct, "confidence"].mean()),
                "mean_conf_wrong": float(block.loc[~block.correct, "confidence"].mean()) if (~block.correct).any() else np.nan,
                "mean_entropy_correct": float(block.loc[block.correct, "entropy"].mean()),
                "mean_entropy_wrong": float(block.loc[~block.correct, "entropy"].mean()) if (~block.correct).any() else np.nan,
            })
            # reliability-diagram data (pooled seeds), 10 bins
            y = block["label"].to_numpy(int); p = block["probability"].to_numpy(float)
            bins = np.linspace(0, 1, 11)
            for i in range(10):
                lo, hi = bins[i], bins[i + 1]
                mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
                if mask.sum() == 0:
                    continue
                rel_rows.append({"architecture": arch, "condition": cond,
                                 "bin_low": lo, "bin_high": hi, "count": int(mask.sum()),
                                 "mean_confidence": float(np.mean(p[mask])),
                                 "empirical_accuracy": float(np.mean(y[mask]))})

    pd.DataFrame(cal_rows).to_csv(out_dir / "calibration_baseline.csv", index=False)
    pd.DataFrame(unc_rows).to_csv(out_dir / "uncertainty_baseline.csv", index=False)
    rel = pd.DataFrame(rel_rows)
    rel.to_csv(out_dir / "reliability_diagram_data.csv", index=False)

    # optional reliability PNGs
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_dir.mkdir(parents=True, exist_ok=True)
        for arch in ARCHS:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
            for cond in CONDITIONS:
                d = rel[(rel.architecture == arch) & (rel.condition == cond)]
                if not d.empty:
                    ax.plot(d["mean_confidence"], d["empirical_accuracy"], marker="o", label=cond)
            ax.set_xlabel("mean predicted probability"); ax.set_ylabel("empirical accuracy")
            ax.set_title(f"Reliability - {arch}"); ax.legend(); fig.tight_layout()
            fig.savefig(fig_dir / f"reliability_{arch}.png", dpi=150); plt.close(fig)
        made = "PNG reliability dibuat."
    except Exception as e:  # matplotlib absen -> data CSV tetap cukup
        made = f"PNG dilewati ({type(e).__name__}); pakai reliability_diagram_data.csv."

    print("\n=== 14.3/14.4 Calibration & uncertainty baseline ===")
    cb = pd.DataFrame(cal_rows)
    for arch in ARCHS:
        for cond in CONDITIONS:
            d = cb[(cb.architecture == arch) & (cb.condition == cond)]
            if d.empty:
                continue
            print(f"  {arch:16s} [{cond:5s}] Brier={d.brier.mean():.4f} NLL={d.nll.mean():.4f} "
                  f"ECE={d.ece.mean():.4f}")
    print(f"  {made}")


# --------------------------------------------------------------------------- #
def main():
    exp = cfg.get_experiment_a_dir()
    pred_dir = exp / "predictions"
    out_dir = exp / "statistics"
    fig_dir = exp / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(pred_dir.glob("EXP_A__controlled__*.anchor_predictions.csv"))
    if not files:
        raise SystemExit(f"Tidak ada prediksi anchor di {pred_dir}. Jalankan Fase 5 dulu.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    architecture_comparison(df, out_dir)
    rank_analysis(df, out_dir)
    error_analysis(df, out_dir)
    calibration_uncertainty(df, out_dir, fig_dir)

    print(f"\nOutput: {out_dir}  (+ figures/ untuk reliability PNG)")
    print("Semua confirmatory (pra-registrasi 16.3/17/18/14). Ingat CLAUDE.md: null result "
          "!= 'tidak ada efek'; jangan simpulkan superioritas arsitektur bila CI tumpang tindih "
          "dan peringkat tidak stabil (17).")


if __name__ == "__main__":
    main()
