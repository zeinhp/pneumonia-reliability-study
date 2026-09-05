"""analyze_benchmark.py

Fase 6 agregasi (protokol bagian 12.2-13): merangkum benchmark sekunder yang
menunjukkan bagaimana STRATEGI SPLIT memengaruhi performa yang TAMPAK. Membaca
file prediksi & metrik yang ditulis train_benchmark.py untuk 3 split x 3 arsitektur
x 3 seed (default seeds 42, 456, 2026 -> 27 run).

CATATAN METODOLOGIS (penting untuk interpretasi):
- Tiap split dievaluasi pada TEST SET-nya masing-masing:
    patient_grouped -> test bersih (0 patient overlap)   = angka JUJUR
    image_level     -> test dgn patient overlap           = angka menggelembung
    original        -> test dgn patient overlap           = angka menggelembung
- Karena test set BERBEDA antar split, perbandingan lintas-split bersifat UNPAIRED
  (tidak seperti Fase 7 clean-vs-leaky yang paired pada anchor yang sama).
- STATUS = DESKRIPTIF, dikunci oleh pra-registrasi (protokol.docx):
    * 6.9  : hasil hanya boleh dibandingkan DESKRIPTIF; selisih bukan estimasi kausal;
             McNemar & paired bootstrap DILARANG antar-test-set berbeda.
    * H6   : hasil optimistis original/image-level "dianggap deskriptif karena test
             set-nya berbeda".
    * 16   : TIDAK ada satu pun uji hipotesis lintas-split yang dipra-registrasi.
  Maka: TIDAK ADA p-value / uji hipotesis di sini. Output confirmatory-grade =
  AUROC per split + patient-cluster bootstrap CI (pra-registrasi 15). "Inflasi"
  dilaporkan sebagai selisih deskriptif (contaminated - patient_grouped) + CI
  bootstrap UNPAIRED, sebagai deskriptif saja. Klaim yang diizinkan: 26 (image-level
  bisa terlalu optimistis; patient-grouped lebih konservatif) - tanpa kata "signifikan".

Bootstrap: patient-cluster (resample pasien, bukan citra), n=2000, seed 42 - konsisten
dengan Fase 7. Untuk tiap cell (split x arch), ketiga seed berbagi test set yang sama;
AUROC di-rata-rata lintas seed pada tiap resample (menangkap ketidakpastian sampling
test), sementara variasi antar-seed dilaporkan terpisah (mean+-sd, median, min-max).

Output ke experiments/experiment_a/benchmark/analysis/.

Contoh:
  python src/analyze_benchmark.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_config as cfg  # noqa: E402

N_BOOT = 2000
BOOT_SEED = 42
ARCHS = ["densenet121", "efficientnet_b0", "swin_tiny"]
SPLITS = ["patient_grouped", "image_level", "original"]   # honest first
CONTAMINATED = ["image_level", "original"]
REFERENCE = "patient_grouped"
EXPECTED_RUNS = 27


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


def point_metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    sens, spec = _sens_spec(y, pred)
    both = len(np.unique(y)) == 2
    return {
        "auroc": _auroc(y, p),
        "auprc": average_precision_score(y, p) if both else np.nan,
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": (sens + spec) / 2 if not (np.isnan(sens) or np.isnan(spec)) else np.nan,
        "brier": brier_score_loss(y, p) if both else float(np.mean((p - y) ** 2)),
        "accuracy": np.mean(pred == y),
    }


class Cell:
    """One (split, architecture) cell: identical test images across its seeds."""
    def __init__(self, sub: pd.DataFrame):
        seeds = sorted(sub["seed"].unique())
        base = (sub[sub["seed"] == seeds[0]][["image_id", "patient_id", "label"]]
                .reset_index(drop=True))
        self.seeds = seeds
        self.y = base["label"].to_numpy(int)
        self.patient_id = base["patient_id"].to_numpy()
        self.probs = {}   # seed -> prob array aligned to base image order
        for s in seeds:
            ss = sub[sub["seed"] == s][["image_id", "probability"]]
            merged = base[["image_id"]].merge(ss, on="image_id", how="left")
            self.probs[s] = merged["probability"].to_numpy(float)
        # patient clusters (list of row-index arrays) for cluster bootstrap
        order = pd.DataFrame({"pid": self.patient_id})
        self.clusters = [idx.to_numpy() for _, idx in order.groupby("pid").groups.items()]

    def seedmean_auroc(self, idx=None):
        vals = []
        for s in self.seeds:
            y = self.y if idx is None else self.y[idx]
            p = self.probs[s] if idx is None else self.probs[s][idx]
            a = _auroc(y, p)
            if not np.isnan(a):
                vals.append(a)
        return float(np.mean(vals)) if vals else np.nan

    def bootstrap_indices(self, rng):
        pick = rng.integers(0, len(self.clusters), len(self.clusters))
        return np.concatenate([self.clusters[i] for i in pick])

    def bootstrap_auroc_ci(self):
        rng = np.random.default_rng(BOOT_SEED)
        boot = np.empty(N_BOOT)
        for b in range(N_BOOT):
            boot[b] = self.seedmean_auroc(self.bootstrap_indices(rng))
        boot = boot[~np.isnan(boot)]
        return (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot.size else (np.nan, np.nan)


def load_predictions(pred_dir: Path) -> pd.DataFrame:
    files = sorted(pred_dir.glob("EXP_A__benchmark__*.test_predictions.csv"))
    if not files:
        raise SystemExit(f"Tidak ada file prediksi benchmark di {pred_dir}. Jalankan Fase 6 dulu.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df


def load_metric_records(metrics_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(metrics_dir.glob("EXP_A__benchmark__*.test_metrics.json")):
        rows.append(json.loads(f.read_text()))
    return pd.DataFrame(rows)


def main():
    bench_root = cfg.get_experiment_a_dir() / "benchmark"
    pred_dir = bench_root / "predictions"
    metrics_dir = bench_root / "metrics"
    out_dir = bench_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_predictions(pred_dir)
    n_runs = df.groupby(["split", "architecture", "seed"]).ngroups
    if n_runs < EXPECTED_RUNS:
        print(f"[PERINGATAN] baru {n_runs}/{EXPECTED_RUNS} run tersedia - hasil parsial, "
              f"jalankan lagi setelah Fase 6 lengkap.")

    try:
        recs = load_metric_records(metrics_dir)
    except Exception:
        recs = pd.DataFrame()

    # ---- per-run point metrics ----
    per_run_rows = []
    for (split, arch, seed), sub in df.groupby(["split", "architecture", "seed"]):
        y = sub["label"].to_numpy(int)
        p = sub["probability"].to_numpy(float)
        m = point_metrics(y, p)
        row = {"split": split, "architecture": arch, "seed": int(seed),
               "n_test": len(sub), "n_patients_test": sub["patient_id"].nunique()}
        row.update({k: m[k] for k in ["auroc", "auprc", "sensitivity", "specificity",
                                      "balanced_accuracy", "brier", "accuracy"]})
        if not recs.empty:
            rr = recs[(recs.split == split) & (recs.architecture == arch) & (recs.seed == seed)]
            if len(rr):
                row["best_epoch"] = int(rr.iloc[0].get("best_epoch", -1))
                row["n_train"] = int(rr.iloc[0].get("n_train", -1))
        per_run_rows.append(row)
    per_run = pd.DataFrame(per_run_rows).sort_values(["split", "architecture", "seed"])
    per_run.to_csv(out_dir / "benchmark_per_run.csv", index=False)

    # ---- per-cell (split x arch): across-seed aggregation + bootstrap CI ----
    cells = {}
    per_cell_rows = []
    for split in SPLITS:
        for arch in ARCHS:
            sub = df[(df.split == split) & (df.architecture == arch)]
            if sub.empty:
                continue
            cell = Cell(sub)
            cells[(split, arch)] = cell
            aurocs = per_run[(per_run.split == split) & (per_run.architecture == arch)]["auroc"].to_numpy()
            ci_low, ci_high = cell.bootstrap_auroc_ci()
            per_cell_rows.append({
                "split": split, "architecture": arch, "n_seeds": len(cell.seeds),
                "auroc_seedmean": cell.seedmean_auroc(),
                "auroc_across_seed_sd": float(np.std(aurocs, ddof=1)) if aurocs.size > 1 else np.nan,
                "auroc_median": float(np.median(aurocs)),
                "auroc_min": float(np.min(aurocs)), "auroc_max": float(np.max(aurocs)),
                "auroc_boot_ci_low": ci_low, "auroc_boot_ci_high": ci_high,
                "n_test": cell.y.size, "n_patients_test": len(cell.clusters),
            })
    per_cell = pd.DataFrame(per_cell_rows)
    per_cell.to_csv(out_dir / "benchmark_per_cell.csv", index=False)

    # ---- inflation: contaminated split minus patient_grouped (DESKRIPTIF, unpaired) ----
    # Tanpa uji hipotesis / p-value (protokol 6.9, H6, 16). Hanya selisih deskriptif
    # + CI bootstrap unpaired untuk menunjukkan besar & arah gap.
    infl_rows = []
    for arch in ARCHS:
        ref = cells.get((REFERENCE, arch))
        if ref is None:
            continue
        for split in CONTAMINATED:
            cell = cells.get((split, arch))
            if cell is None:
                continue
            obs = cell.seedmean_auroc() - ref.seedmean_auroc()
            rng = np.random.default_rng(BOOT_SEED)
            boot = np.empty(N_BOOT)
            for b in range(N_BOOT):
                a = cell.seedmean_auroc(cell.bootstrap_indices(rng))
                r = ref.seedmean_auroc(ref.bootstrap_indices(rng))
                boot[b] = a - r
            boot = boot[~np.isnan(boot)]
            ci_low = float(np.percentile(boot, 2.5)) if boot.size else np.nan
            ci_high = float(np.percentile(boot, 97.5)) if boot.size else np.nan
            infl_rows.append({
                "architecture": arch, "comparison": f"{split}_minus_{REFERENCE}",
                "dAUROC_inflation_descriptive": obs,
                "boot_ci_low_descriptive": ci_low, "boot_ci_high_descriptive": ci_high,
                "auroc_contaminated": cell.seedmean_auroc(),
                "auroc_reference_honest": ref.seedmean_auroc(),
                "note": "DESKRIPTIF unpaired (test set beda); bukan uji hipotesis - protokol 6.9/H6/16",
            })
    infl = pd.DataFrame(infl_rows)
    infl.to_csv(out_dir / "benchmark_inflation.csv", index=False)

    # ---- console summary ----
    print("=== Fase 6: AUROC yang TAMPAK per split (seed-mean, bootstrap 95% CI) ===")
    for arch in ARCHS:
        print(f"  {arch}")
        for split in SPLITS:
            r = per_cell[(per_cell.split == split) & (per_cell.architecture == arch)]
            if r.empty:
                continue
            r = r.iloc[0]
            tag = "  (JUJUR)" if split == REFERENCE else ""
            print(f"    {split:16s} AUROC={r['auroc_seedmean']:.4f} "
                  f"CI[{r['auroc_boot_ci_low']:.4f},{r['auroc_boot_ci_high']:.4f}] "
                  f"seed-sd={r['auroc_across_seed_sd']:.4f}{tag}")
    print("\n=== Inflasi DESKRIPTIF (split kontaminasi - patient_grouped, unpaired) ===")
    for _, r in infl.iterrows():
        print(f"  {r['architecture']:16s} {r['comparison']:28s} "
              f"+{r['dAUROC_inflation_descriptive']:.4f} "
              f"CI[{r['boot_ci_low_descriptive']:+.4f},{r['boot_ci_high_descriptive']:+.4f}]")
    print(f"\nOutput: {out_dir}")
    print("CATATAN: Fase 6 = DESKRIPTIF (pra-registrasi 6.9/H6/16). Test set beda antar split -> "
          "tidak ada uji hipotesis / p-value; McNemar & paired bootstrap DILARANG di sini. "
          "Headline = AUROC per split + CI, dan gap deskriptif. Klaim diizinkan (26): image-level "
          "bisa terlalu optimistis, patient-grouped lebih konservatif - TANPA kata 'signifikan'.")


if __name__ == "__main__":
    main()
