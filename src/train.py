"""train.py

CLI orchestrator for a single Experiment A training run (protocol section 11,
19, 22 Fase 3-4). One run = one (analysis, condition, architecture, seed).

Two-stage transfer learning:
  Stage 1 - frozen backbone, head only, fixed 3 epochs, LR 3e-4
  Stage 2 - full fine-tuning, LR 1e-4, max 40 epochs, ReduceLROnPlateau,
            early stopping on validation AUROC (patience 7)

Guardrails baked in:
  * The anchor TEST manifest is never loaded here. Only train + validation are
    read. A run aborts if the validation manifest resolves to the anchor test.
  * FP32 only, full determinism, no AMP (CLAUDE.md #4).

Usage (single run):
  python src/train.py --condition clean --architecture densenet121 --seed 42

Smoke test (Fase 3, no test set opened):
  python src/train.py --condition clean --architecture swin_tiny --seed 42 --smoke

Task-parallel across 2 GPUs is done at the shell level, e.g.:
  $env:CUDA_VISIBLE_DEVICES=1 ; python src/train.py ... (GPU #2, primary)
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_config as cfg  # noqa: E402
from training import reproducibility as repro  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Experiment A single training run.")
    p.add_argument("--condition", required=True, choices=["clean", "leaky"])
    p.add_argument("--architecture", required=True,
                   choices=["densenet121", "efficientnet_b0", "swin_tiny"])
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--analysis", default="controlled")
    p.add_argument("--config", default=None,
                   help="Path to train_config.yaml (default: config/train_config.yaml)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Physical batch size override; effective stays 32 via accumulation.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-root", default=None,
                   help="Override experiment output root (default: experiments/experiment_a).")
    # smoke test
    p.add_argument("--smoke", action="store_true",
                   help="Fase 3 smoke test: tiny subset, few epochs, measure fit/timing.")
    p.add_argument("--smoke-train", type=int, default=32)
    p.add_argument("--smoke-val", type=int, default=16)
    p.add_argument("--smoke-epochs", type=int, default=1)
    return p.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stratified_head(df: pd.DataFrame, n: int, pos: str) -> pd.DataFrame:
    """Take ~n rows keeping both classes present (for smoke only)."""
    if n >= len(df):
        return df
    n_pos = max(1, n // 2)
    n_neg = n - n_pos
    pos_df = df[df["label"] == pos].head(n_pos)
    neg_df = df[df["label"] != pos].head(n_neg)
    return pd.concat([pos_df, neg_df]).reset_index(drop=True)


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
    config_sha = repro.sha256_file(config_path)

    run_id = repro.build_run_id(args.analysis, args.condition, args.architecture, args.seed)
    if args.smoke:
        run_id = "SMOKE__" + run_id

    # ---- determinism ----
    repro.set_global_determinism(args.seed, conf["determinism"])
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # ---- manifests (TRAIN + VALIDATION only; TEST is never opened here) ----
    manifests_dir = cfg.get_manifests_dir()
    cond_manifests = conf["manifests"][args.condition]
    anchor_name = conf["manifests"]["anchor_test"]
    if cond_manifests["validation"] == anchor_name or cond_manifests["train"] == anchor_name:
        raise RuntimeError("SAFETY ABORT: a training/validation manifest points at the anchor "
                           "TEST set. The test set must not be opened during training.")

    train_df = load_manifest(manifests_dir, cond_manifests["train"])
    val_df = load_manifest(manifests_dir, cond_manifests["validation"])
    pos = conf["label"]["positive_class"]

    if args.smoke:
        train_df = stratified_head(train_df, args.smoke_train, pos)
        val_df = stratified_head(val_df, args.smoke_val, pos)

    dataset_root = cfg.get_dataset_root()
    train_ds = PneumoniaDataset(train_df, dataset_root, conf, train=True)
    val_ds = PneumoniaDataset(val_df, dataset_root, conf, train=False)

    phys_batch = args.batch_size or int(conf["training"]["batch_size"])
    eff_batch = int(conf["training"]["effective_batch_size"])
    accum_steps = max(1, math.ceil(eff_batch / phys_batch))
    dl = conf["dataloader"]
    g = repro.make_dataloader_generator(args.seed)
    common = dict(num_workers=int(dl["num_workers"]), pin_memory=bool(dl["pin_memory"]),
                  collate_fn=collate_meta, worker_init_fn=repro.seed_worker, generator=g)
    train_loader = DataLoader(train_ds, batch_size=phys_batch, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, batch_size=phys_batch, shuffle=False, drop_last=False, **common)

    # ---- model ----
    model = M.build_model(args.architecture, conf, args.seed).to(device)
    criterion = nn.BCEWithLogitsLoss()
    tr = conf["training"]
    threshold = float(conf["threshold"]["primary"])
    grad_clip = float(tr["grad_clip_max_norm"])

    history = []
    env = repro.capture_environment()
    print(f"[{run_id}] device={device} phys_batch={phys_batch} accum={accum_steps} "
          f"train={len(train_ds)} val={len(val_ds)}")
    print(f"[env] torch={env['torch']} cuda={env['cuda_version']} "
          f"gpus={[gpu['name'] for gpu in env['gpus']] or 'CPU'}")

    t0 = time.time()
    train_seconds = 0.0        # pure training-loop time (excludes build/eval/save)
    train_images_total = 0

    # ================= Stage 1: frozen backbone, head only =================
    M.set_backbone_frozen(model, True)
    opt1 = torch.optim.AdamW(M.trainable_parameters(model), lr=float(tr["stage1_lr"]),
                             weight_decay=float(tr["weight_decay"]),
                             betas=tuple(tr["betas"]))
    s1_epochs = args.smoke_epochs if args.smoke else int(tr["stage1_head_epochs"])
    for ep in range(1, s1_epochs + 1):
        _t = time.time()
        tl = engine.train_one_epoch(model, train_loader, opt1, criterion, device, grad_clip, accum_steps)
        train_seconds += time.time() - _t
        train_images_total += len(train_ds)
        vm, vloss, _ = engine.evaluate(model, val_loader, criterion, device, threshold)
        history.append({"stage": 1, "epoch": ep, "train_loss": tl, "val_loss": vloss,
                        "val_auroc": vm["auroc"], "lr": opt1.param_groups[0]["lr"]})
        print(f"  [S1 {ep}/{s1_epochs}] train_loss={tl:.4f} val_loss={vloss:.4f} val_auroc={vm['auroc']:.4f}")

    # ================= Stage 2: full fine-tuning + early stopping =================
    M.set_backbone_frozen(model, False)
    opt2 = torch.optim.AdamW(model.parameters(), lr=float(tr["stage2_lr"]),
                             weight_decay=float(tr["weight_decay"]),
                             betas=tuple(tr["betas"]))
    sch = conf["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt2, mode=sch["mode"], factor=float(sch["factor"]),
        patience=int(sch["patience"]), min_lr=float(sch["min_lr"]))
    es = conf["early_stopping"]
    max_ep = args.smoke_epochs if args.smoke else int(tr["stage2_max_epochs"])

    best = {"val_auroc": -1.0, "val_loss": math.inf, "epoch": -1, "state": None}
    epochs_no_improve = 0
    for ep in range(1, max_ep + 1):
        _t = time.time()
        tl = engine.train_one_epoch(model, train_loader, opt2, criterion, device, grad_clip, accum_steps)
        train_seconds += time.time() - _t
        train_images_total += len(train_ds)
        vm, vloss, _ = engine.evaluate(model, val_loader, criterion, device, threshold)
        auroc = vm["auroc"] if not math.isnan(vm["auroc"]) else -1.0
        scheduler.step(auroc)
        history.append({"stage": 2, "epoch": ep, "train_loss": tl, "val_loss": vloss,
                        "val_auroc": vm["auroc"], "lr": opt2.param_groups[0]["lr"]})
        print(f"  [S2 {ep}/{max_ep}] train_loss={tl:.4f} val_loss={vloss:.4f} "
              f"val_auroc={vm['auroc']:.4f} lr={opt2.param_groups[0]['lr']:.2e}")

        improved = auroc > best["val_auroc"] + float(es["min_delta"])
        tie_better = (abs(auroc - best["val_auroc"]) <= float(es["min_delta"])) and (vloss < best["val_loss"])
        if improved or tie_better:
            best = {"val_auroc": auroc, "val_loss": vloss, "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            epochs_no_improve = 0 if improved else epochs_no_improve + 1
        else:
            epochs_no_improve += 1

        if not args.smoke and epochs_no_improve >= int(es["patience"]):
            print(f"  early stopping at epoch {ep} (no val_auroc improvement in {es['patience']})")
            break

    wall = time.time() - t0

    # ---- outputs (protocol 19.2) ----
    exp_root = Path(args.output_root) if args.output_root else cfg.get_experiment_a_dir()
    if args.smoke:
        exp_root = exp_root / "_smoke"
    for sub in ["checkpoints", "histories", "configs", "metrics", "logs"]:
        (exp_root / sub).mkdir(parents=True, exist_ok=True)

    ckpt_path = exp_root / "checkpoints" / f"{run_id}.best.pt"
    if best["state"] is not None:
        torch.save({"model_state": best["state"], "run_id": run_id,
                    "best_epoch": best["epoch"], "best_val_auroc": best["val_auroc"],
                    "architecture": args.architecture, "condition": args.condition,
                    "seed": args.seed}, ckpt_path)

    repro.save_json({"run_id": run_id, "history": history}, exp_root / "histories" / f"{run_id}.history.json")

    # Best-epoch validation metrics (recompute on best weights).
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    best_val_metrics, best_val_loss, _ = engine.evaluate(model, val_loader, criterion, device, threshold)
    repro.save_json({"run_id": run_id, "best_epoch": best["epoch"],
                     "val_loss": best_val_loss, "val_metrics": best_val_metrics},
                    exp_root / "metrics" / f"{run_id}.val_metrics.json")

    # Full run record (protocol 19).
    run_record = {
        "run_id": run_id,
        "timestamp": repro.utc_timestamp(),
        "git_commit": repro.git_commit(),
        "analysis": args.analysis,
        "condition": args.condition,
        "manifest_train": cond_manifests["train"],
        "manifest_validation": cond_manifests["validation"],
        "manifest_train_sha256": repro.sha256_file(manifests_dir / cond_manifests["train"]),
        "manifest_validation_sha256": repro.sha256_file(manifests_dir / cond_manifests["validation"]),
        "protocol_version": conf.get("protocol_version"),
        "config_sha256": config_sha,
        "architecture": args.architecture,
        "pretrained_weights": "imagenet",
        "seed": args.seed,
        "hardware": env["gpus"],
        "software_versions": {k: env[k] for k in ["python", "os", "torch", "torchvision", "timm", "cuda_version", "cudnn_version"]},
        "optimizer": "AdamW",
        "learning_rate": {"stage1": float(tr["stage1_lr"]), "stage2": float(tr["stage2_lr"])},
        "batch_size": phys_batch,
        "effective_batch_size": eff_batch,
        "accum_steps": accum_steps,
        "augmentation_config": conf["augmentation"],
        "best_epoch": best["epoch"],
        "best_val_auroc": best["val_auroc"],
        "stopping_epoch": history[-1]["epoch"] if history else None,
        "checkpoint_path": str(ckpt_path),
        "wall_clock_seconds": round(wall, 1),
        "smoke": args.smoke,
    }
    repro.save_json(run_record, exp_root / "configs" / f"{run_id}.run_record.json")

    print(f"[{run_id}] DONE best_epoch={best['epoch']} best_val_auroc={best['val_auroc']:.4f} "
          f"wall={wall:.1f}s ckpt={ckpt_path.name}")
    if args.smoke:
        ips = train_images_total / train_seconds if train_seconds > 0 else 0.0
        est = 4607 / ips if ips > 0 else float("nan")
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            print(f"[SMOKE] peak_VRAM={peak:.0f} MB at phys_batch={phys_batch}.")
        print(f"[SMOKE] train throughput={ips:.1f} img/s -> ~{est:.0f}s per full epoch "
              f"(4607 img). NOTE: small-batch smoke includes warmup; treat as rough lower bound.")
        print("[SMOKE] OK - no test set opened.")


if __name__ == "__main__":
    main()
