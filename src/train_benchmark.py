"""train_benchmark.py

Fase 6 (protokol bagian 22 Fase 6, 12.2, 13): benchmark sekunder yang menunjukkan
bagaimana strategi split memengaruhi performa yang TAMPAK. Melatih tiap arsitektur
pada tiga split dan mengevaluasi pada test set MASING-MASING split:

    original        -> original_split_deduplicated.csv        (ada patient overlap)
    image_level     -> image_stratified_split_deduplicated.csv (ada patient overlap)
    patient_grouped -> patient_grouped_split_deduplicated.csv  (0 patient overlap, jujur)

Aturan epoch (protokol 13, "epoch rule"): test tidak boleh dipakai memilih epoch.
- patient_grouped & image_level: early stopping pada validation-nya sendiri (val AUROC).
- original: JUMLAH epoch stage-2 DIKUNCI = median best-epoch dari controlled clean
  (patient-grouped) runs Fase 4, diberikan lewat --original-epochs. Tanpa early
  stopping, tanpa memakai original test untuk seleksi apa pun.

Hyperparameter identik dengan Eksperimen A (config train_config.yaml).
Seed sekunder minimal: 42, 456, 2026 (protokol 12.2).

Contoh:
  python src/train_benchmark.py --split patient_grouped --architecture densenet121 --seed 42
  python src/train_benchmark.py --split original --architecture densenet121 --seed 42 --original-epochs 12
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

SPLIT_MANIFEST = {
    "original": "original_split_deduplicated.csv",
    "image_level": "image_stratified_split_deduplicated.csv",
    "patient_grouped": "patient_grouped_split_deduplicated.csv",
}


def parse_args():
    p = argparse.ArgumentParser(description="Fase 6: secondary split benchmark.")
    p.add_argument("--split", required=True, choices=list(SPLIT_MANIFEST))
    p.add_argument("--architecture", required=True,
                   choices=["densenet121", "efficientnet_b0", "swin_tiny"])
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--config", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--original-epochs", type=int, default=None,
                   help="Wajib untuk --split original: jumlah stage-2 epoch (median best-epoch "
                        "dari controlled clean runs). Diabaikan untuk split lain.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-train", type=int, default=32)
    p.add_argument("--smoke-val", type=int, default=16)
    return p.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stratified_head(df, n, pos):
    if n >= len(df):
        return df
    n_pos = max(1, n // 2)
    return pd.concat([df[df["label"] == pos].head(n_pos),
                      df[df["label"] != pos].head(n - n_pos)]).reset_index(drop=True)


def main():
    args = parse_args()
    if args.split == "original" and args.original_epochs is None and not args.smoke:
        raise SystemExit("--original-epochs wajib untuk --split original (median best-epoch "
                         "dari controlled clean runs). Hitung dulu dari run records Fase 4.")

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from training.data import PneumoniaDataset, load_manifest, collate_meta
    from training import models as M
    from training import engine

    config_path = Path(args.config) if args.config else cfg.get_configs_dir() / "train_config.yaml"
    conf = load_config(config_path)
    config_sha = repro.sha256_file(config_path)

    run_id = f"EXP_A__benchmark__{args.split}__{args.architecture}__seed_{args.seed}"
    if args.smoke:
        run_id = "SMOKE__" + run_id

    repro.set_global_determinism(args.seed, conf["determinism"])
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    manifests_dir = cfg.get_manifests_dir()
    manifest_name = SPLIT_MANIFEST[args.split]
    full = load_manifest(manifests_dir, manifest_name)
    split_col = "experimental_split"
    train_df = full[full[split_col] == "train"].copy()
    val_df = full[full[split_col] == "validation"].copy()
    test_df = full[full[split_col] == "test"].copy()
    pos = conf["label"]["positive_class"]

    if args.smoke:
        train_df = stratified_head(train_df, args.smoke_train, pos)
        val_df = stratified_head(val_df, args.smoke_val, pos)
        test_df = stratified_head(test_df, args.smoke_val, pos)

    dataset_root = cfg.get_dataset_root()
    train_ds = PneumoniaDataset(train_df, dataset_root, conf, train=True)
    val_ds = PneumoniaDataset(val_df, dataset_root, conf, train=False)
    test_ds = PneumoniaDataset(test_df, dataset_root, conf, train=False)

    phys_batch = args.batch_size or int(conf["training"]["batch_size"])
    accum_steps = max(1, math.ceil(int(conf["training"]["effective_batch_size"]) / phys_batch))
    dl = conf["dataloader"]
    g = repro.make_dataloader_generator(args.seed)
    common = dict(num_workers=int(dl["num_workers"]), pin_memory=bool(dl["pin_memory"]),
                  collate_fn=collate_meta, worker_init_fn=repro.seed_worker, generator=g)
    train_loader = DataLoader(train_ds, batch_size=phys_batch, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=phys_batch, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=phys_batch, shuffle=False,
                             num_workers=int(dl["num_workers"]), pin_memory=bool(dl["pin_memory"]),
                             collate_fn=collate_meta)

    model = M.build_model(args.architecture, conf, args.seed).to(device)
    criterion = nn.BCEWithLogitsLoss()
    tr = conf["training"]
    threshold = float(conf["threshold"]["primary"])
    grad_clip = float(tr["grad_clip_max_norm"])
    fixed_epochs = args.split == "original"   # original: fixed epochs, no early stopping

    # ---- Stage 1: frozen backbone head ----
    M.set_backbone_frozen(model, True)
    opt1 = torch.optim.AdamW(M.trainable_parameters(model), lr=float(tr["stage1_lr"]),
                             weight_decay=float(tr["weight_decay"]), betas=tuple(tr["betas"]))
    s1 = 1 if args.smoke else int(tr["stage1_head_epochs"])
    for ep in range(1, s1 + 1):
        engine.train_one_epoch(model, train_loader, opt1, criterion, device, grad_clip, accum_steps)

    # ---- Stage 2: full fine-tuning ----
    M.set_backbone_frozen(model, False)
    opt2 = torch.optim.AdamW(model.parameters(), lr=float(tr["stage2_lr"]),
                             weight_decay=float(tr["weight_decay"]), betas=tuple(tr["betas"]))
    sch = conf["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt2, mode=sch["mode"], factor=float(sch["factor"]),
        patience=int(sch["patience"]), min_lr=float(sch["min_lr"]))
    es = conf["early_stopping"]

    if fixed_epochs:
        max_ep = 1 if args.smoke else int(args.original_epochs)
    else:
        max_ep = 1 if args.smoke else int(tr["stage2_max_epochs"])

    best = {"val_auroc": -1.0, "val_loss": math.inf, "epoch": -1, "state": None}
    no_improve = 0
    for ep in range(1, max_ep + 1):
        engine.train_one_epoch(model, train_loader, opt2, criterion, device, grad_clip, accum_steps)
        if fixed_epochs:
            # original (protokol 6.10): validation 16-citra TIDAK dipakai untuk seleksi,
            # scheduling, atau early stopping. Latih jumlah epoch tetap (--original-epochs),
            # simpan model epoch terakhir. Tidak ada evaluasi val di sini (val_loader tidak
            # disentuh; manifest original melabeli val sebagai 'val', bukan 'validation').
            best = {"val_auroc": float("nan"), "val_loss": float("nan"), "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            continue
        # patient_grouped & image_level: early stopping berbasis validation AUROC (11.4)
        vm, vloss, _ = engine.evaluate(model, val_loader, criterion, device, threshold)
        auroc = vm["auroc"] if not math.isnan(vm["auroc"]) else -1.0
        scheduler.step(auroc)
        if not fixed_epochs:
            improved = auroc > best["val_auroc"] + float(es["min_delta"])
            tie = (abs(auroc - best["val_auroc"]) <= float(es["min_delta"])) and (vloss < best["val_loss"])
            if improved or tie:
                best = {"val_auroc": auroc, "val_loss": vloss, "epoch": ep,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
                no_improve = 0 if improved else no_improve + 1
            else:
                no_improve += 1
            if not args.smoke and no_improve >= int(es["patience"]):
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    # ---- Evaluate on THIS split's own test set (Fase 6) ----
    test_metrics, test_loss, test_preds = engine.evaluate(
        model, test_loader, criterion, device, threshold, collect_predictions=True)

    bench_root = cfg.get_experiment_a_dir() / "benchmark"
    for sub in ["checkpoints", "predictions", "metrics"]:
        (bench_root / sub).mkdir(parents=True, exist_ok=True)

    torch.save({"model_state": best["state"], "run_id": run_id, "best_epoch": best["epoch"],
                "split": args.split, "architecture": args.architecture, "seed": args.seed},
               bench_root / "checkpoints" / f"{run_id}.best.pt")

    df = pd.DataFrame(test_preds)
    df.insert(0, "seed", args.seed); df.insert(0, "split", args.split)
    df.insert(0, "architecture", args.architecture); df.insert(0, "run_id", run_id)
    df.to_csv(bench_root / "predictions" / f"{run_id}.test_predictions.csv", index=False)

    rec = {"run_id": run_id, "split": args.split, "architecture": args.architecture,
           "seed": args.seed, "best_epoch": best["epoch"], "fixed_epochs": fixed_epochs,
           "original_epochs": args.original_epochs, "config_sha256": config_sha,
           "manifest": manifest_name, "test_loss": test_loss,
           "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)}
    rec.update({f"test_{k}": v for k, v in test_metrics.items()})
    repro.save_json(rec, bench_root / "metrics" / f"{run_id}.test_metrics.json")

    print(f"[{run_id}] split={args.split} test_auroc={test_metrics['auroc']:.4f} "
          f"n_test={len(test_df)} best_epoch={best['epoch']}")


if __name__ == "__main__":
    main()
