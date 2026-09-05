"""predict_anchor_test.py

Phase 5 (protocol section 22 Phase 5, 13.2, 19.3): ONE-TIME inference on the
common anchor test set, run ONLY after all Phase 4 checkpoints are locked.

This is the point where the test set is opened for the FIRST time. After this
no training changes are allowed (protocol section 24). This script is pure
inference: it loads each best checkpoint, runs the model on the anchor test,
and writes one prediction CSV per run (columns per protocol 19.3) plus a
metrics summary.

Guard: refuses to run if the number of non-smoke checkpoints is < the
required number of runs (default 30) unless --allow-partial is given. This
prevents opening the test set before all checkpoints are locked.

Example:
    python src/predict_anchor_test.py            # requires 30 checkpoints
    python src/predict_anchor_test.py --allow-partial   # for trial runs
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_config as cfg  # noqa: E402
from training import reproducibility as repro  # noqa: E402

CKPT_RE = re.compile(
    r"^EXP_A__(?P<analysis>[a-z0-9]+)__(?P<condition>[a-z0-9]+)__"
    r"(?P<arch>[a-z0-9_]+)__seed_(?P<seed>\d+)\.best\.pt$"
)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 5: anchor-test inference (one-shot).")
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoints-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--metrics-dir", default=None,
                   help="Folder for anchor_test_metrics_summary.csv. Default: same as "
                        "--out-dir if given, otherwise experiments/experiment_a/metrics "
                        "(for backward compatibility with old runs). ALWAYS set explicitly "
                        "when using a non-default --output-root/--checkpoints-dir so it "
                        "doesn't overwrite the Experiment A summary.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--require", type=int, default=30,
                   help="Number of non-smoke checkpoints required before opening the test set.")
    p.add_argument("--allow-partial", action="store_true",
                   help="Allow running even if checkpoints < --require (for trial runs, NOT confirmatory).")
    return p.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from training.data import PneumoniaDataset, load_manifest, collate_meta
    from training import models as M
    from training import engine

    config_path = Path(args.config) if args.config else cfg.get_configs_dir() / "train_config.yaml"
    conf = load_config(config_path)

    exp_root = cfg.get_experiment_a_dir()
    ckpt_dir = Path(args.checkpoints_dir) if args.checkpoints_dir else exp_root / "checkpoints"
    out_dir = Path(args.out_dir) if args.out_dir else exp_root / "predictions"
    if args.metrics_dir:
        metrics_dir = Path(args.metrics_dir)
    elif args.out_dir:
        # --out-dir is set (non-default analysis) but --metrics-dir is not -> don't
        # silently overwrite Experiment A's experiments/experiment_a/metrics.
        metrics_dir = out_dir.parent / "metrics"
        print(f"  [info] --metrics-dir not given, using {metrics_dir} "
              f"(derived from --out-dir) so Experiment A's metrics aren't overwritten.")
    else:
        metrics_dir = exp_root / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ---- collect locked checkpoints (exclude smoke) ----
    ckpts = sorted(
        c for c in ckpt_dir.glob("EXP_A__*.best.pt") if not c.name.startswith("SMOKE__")
    )
    if len(ckpts) < args.require and not args.allow_partial:
        raise SystemExit(
            f"SAFETY ABORT: only {len(ckpts)} checkpoints found, {args.require} required "
            "before opening the anchor test. Finish all Phase 4 runs first, or use "
            "--allow-partial ONLY for trial runs (results are not confirmatory)."
        )

    print("=" * 70)
    print(f"PHASE 5 - OPENING THE ANCHOR TEST SET ({len(ckpts)} checkpoints)")
    print("After this point no training changes are allowed (protocol section 24).")
    print("=" * 70)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # ---- anchor test set (eval transform, no augmentation) ----
    manifests_dir = cfg.get_manifests_dir()
    anchor_name = conf["manifests"]["anchor_test"]
    anchor_df = load_manifest(manifests_dir, anchor_name)
    anchor_sha = repro.sha256_file(manifests_dir / anchor_name)
    dataset_root = cfg.get_dataset_root()
    ds = PneumoniaDataset(anchor_df, dataset_root, conf, train=False)
    batch = int(conf["training"]["batch_size"])
    dl = conf["dataloader"]
    loader = DataLoader(ds, batch_size=batch, shuffle=False, drop_last=False,
                        num_workers=int(dl["num_workers"]), pin_memory=bool(dl["pin_memory"]),
                        collate_fn=collate_meta)
    criterion = nn.BCEWithLogitsLoss()
    threshold = float(conf["threshold"]["primary"])

    pred_cols = ["run_id", "architecture", "condition", "seed", "image_id", "relative_path",
                 "patient_id", "label", "logit", "probability", "predicted_label_0_5", "entropy"]
    summary_rows = []

    for ckpt_path in ckpts:
        m = CKPT_RE.match(ckpt_path.name)
        if not m:
            print(f"  [skip] unrecognized checkpoint name: {ckpt_path.name}")
            continue
        arch, cond, seed = m["arch"], m["condition"], int(m["seed"])
        run_id = ckpt_path.name[:-len(".best.pt")]

        repro.set_global_determinism(seed, conf["determinism"])
        model = M.build_model(arch, conf, seed).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])

        metrics, loss, preds = engine.evaluate(
            model, loader, criterion, device, threshold, collect_predictions=True)

        # per-run prediction CSV (protocol 19.3)
        df = pd.DataFrame(preds)
        df.insert(0, "seed", seed)
        df.insert(0, "condition", cond)
        df.insert(0, "architecture", arch)
        df.insert(0, "run_id", run_id)
        df = df[pred_cols]
        df.to_csv(out_dir / f"{run_id}.anchor_predictions.csv", index=False)

        row = {"run_id": run_id, "architecture": arch, "condition": cond, "seed": seed,
               "anchor_manifest_sha256": anchor_sha, "test_loss": loss}
        row.update({f"test_{k}": v for k, v in metrics.items()})
        summary_rows.append(row)
        print(f"  {run_id}: test_auroc={metrics['auroc']:.4f} "
              f"sens={metrics['sensitivity']:.3f} spec={metrics['specificity']:.3f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = pd.DataFrame(summary_rows).sort_values(["architecture", "condition", "seed"])
    summary_path = metrics_dir / "anchor_test_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("-" * 70)
    print(f"Per-run predictions: {out_dir}")
    print(f"Metrics summary: {summary_path}  ({len(summary)} runs)")
    print("Phase 5 done. Proceed to Phase 7 (statistics) using these prediction files.")


if __name__ == "__main__":
    main()
