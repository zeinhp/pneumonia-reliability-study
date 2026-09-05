"""predict_anchor_test_e_contrast.py

Experiment E (protocol §28) - the last unexecuted dimension: robustness to
brightness/contrast. The other four dimensions (noise, blur, compression,
resolution) were already run as "Experiment B" (see
experiment_b_deliverables/). This script completes E to 5/5 dimensions.

Design (follows the pattern of exp_b_degrade_infer.py, LOCKED in CLAUDE.md §9
Experiment B):
  - INFERENCE ONLY, no training. The anchor test has been open since
    Experiment A Phase 5 -> this script's results are EXPLORATORY (protocol
    §24), they do not change the Experiment A confirmatory conclusions.
  - Uses the 15 CLEAN Experiment A checkpoints (3 architectures x 5 seeds) -
    the 'leaky' checkpoints are NOT used (matching Experiment B, which also
    only uses clean).
  - Degradation insertion point = preprocess: grayscale -> square-pad ->
    resize 224 (uint8) -> APPLY brightness/contrast -> replicate to 3
    channels -> ToTensor -> ImageNet normalization. Same insertion point as
    Experiment B's degradations (not native-image).
  - Severity locked in CLAUDE.md §9 (row "Brightness/Contrast +- (optional)"):
    level 1/2/3 = 20% / 35% / 50%.
  - DESIGN DECISION (new, documented here because the CLAUDE.md table only
    states the "+-" magnitude without fixing a direction): brightness &
    contrast are DECREASED (factor = 1 - pct), not increased or randomized in
    direction. Rationale: the other 4 degradation dimensions in Experiment B
    are all one-directional degrees of damage (noise up, blur up, JPEG
    quality down, resolution down) - the image gets harder to read, this is
    not a bidirectional stress test. Decreasing brightness+contrast is
    consistent with that (an under-exposed/low-contrast radiograph is a
    realistic acquisition-quality failure). Fully deterministic (not a
    random process) - no need for per-image seeding like Gaussian noise.
  - The 'clean' condition (severity 0) is NOT repeated here - it already
    exists in experiment_b_deliverables/source_csv/exp_b_metrics.csv
    (transform='clean'), used together when combining the full E robustness
    curves.
  - The output metrics schema is IDENTICAL to exp_b_metrics.csv (arch, seed,
    transform, severity, auroc, auprc, sensitivity, specificity,
    balanced_accuracy, f1, brier, nll, ece) so it can be directly concatenated
    (pd.concat) with the B results when writing the E report.

Run in an environment where the 15 clean EXP_A checkpoints are available
(Hub, same as previously used for predict_anchor_test.py /
exp_b_degrade_infer.py):

    export CHEST_XRAY_DATA_ROOT=$HOME/chest_xray/chest_xray
    python src/predict_anchor_test_e_contrast.py

Output:
    experiments/experiment_e/metrics/exp_e_contrast_metrics.csv   (aggregate metrics, 45 rows = 15 ckpt x 3 severity)
    experiments/experiment_e/predictions/*.anchor_predictions.csv (per-run per-severity predictions, same format as predict_anchor_test.py)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image, ImageEnhance
import torchvision.transforms as T
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_config as cfg  # noqa: E402
from training import reproducibility as repro  # noqa: E402
from training.data import SquarePad, _interp  # noqa: E402

CKPT_RE = re.compile(
    r"^EXP_A__(?P<analysis>[a-z0-9]+)__(?P<condition>[a-z0-9]+)__"
    r"(?P<arch>[a-z0-9_]+)__seed_(?P<seed>\d+)\.best\.pt$"
)

# Severity table locked in CLAUDE.md section 9 ("Brightness/Contrast +- opsional").
# factor = 1 - pct: darkens & flattens the image (matches the "degradation", not
# "stress in either direction", framing used by the other 4 Experiment E dimensions).
SEVERITY_PCT = {1: 0.20, 2: 0.35, 3: 0.50}


class BrightnessContrastDegrade:
    """Deterministic brightness+contrast reduction, applied to the padded/resized
    uint8 grayscale image (post step 3, pre channel-replication) - same insertion
    point used for noise/blur/jpeg/resolution in Experiment B."""

    def __init__(self, pct: float):
        self.factor = 1.0 - pct

    def __call__(self, img: Image.Image) -> Image.Image:
        img = ImageEnhance.Brightness(img).enhance(self.factor)
        img = ImageEnhance.Contrast(img).enhance(self.factor)
        return img


def build_contrast_eval_transform(train_cfg: dict, severity: int):
    """Eval-time (no augmentation) preprocessing pipeline + brightness/contrast
    degradation inserted at the locked point (post-resize, pre channel-replicate)."""
    img_cfg = train_cfg["image"]
    interp = _interp(img_cfg["interpolation"])
    size = int(img_cfg["size"])
    fill = int(img_cfg["pad_value"])
    pct = SEVERITY_PCT[severity]

    return T.Compose([
        T.Grayscale(num_output_channels=1),
        SquarePad(fill=fill),
        T.Resize((size, size), interpolation=interp),
        BrightnessContrastDegrade(pct),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=img_cfg["normalize_mean"], std=img_cfg["normalize_std"]),
    ])


class DegradedAnchorDataset:
    """Anchor test set read through a fixed (severity-specific) degradation transform."""

    def __init__(self, manifest_df, dataset_root: Path, transform, positive_class: str):
        self.df = manifest_df.reset_index(drop=True)
        self.root = Path(dataset_root)
        self.transform = transform
        self.pos = positive_class

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import torch
        row = self.df.iloc[idx]
        path = self.root / row["relative_path"]
        with Image.open(path) as im:
            im = im.convert("L")
            tensor = self.transform(im)
        label = 1.0 if row["label"] == self.pos else 0.0
        meta = {
            "image_id": row["image_id"],
            "relative_path": row["relative_path"],
            "patient_id": row["patient_id"],
            "label": row["label"],
        }
        return tensor, torch.tensor(label, dtype=torch.float32), meta


def collate_meta(batch):
    import torch
    tensors = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return tensors, labels, metas


def main():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from training import models as M
    from training import engine
    from training.data import load_manifest

    config_path = cfg.get_configs_dir() / "train_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    exp_a_dir = cfg.get_experiment_a_dir()
    ckpt_dir = exp_a_dir / "checkpoints"
    exp_e_dir = cfg.get_experiments_dir() / "experiment_e"
    out_dir = exp_e_dir / "predictions"
    metrics_dir = exp_e_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # only clean checkpoints - identical scope to Experiment B (leaky excluded)
    ckpts = sorted(
        c for c in ckpt_dir.glob("EXP_A__*__clean__*.best.pt")
        if not c.name.startswith("SMOKE__")
    )
    if len(ckpts) < 15:
        raise SystemExit(
            f"Found {len(ckpts)} clean checkpoints (need 15 = 3 architectures x 5 seeds). "
            "Run this script in an environment with the full Experiment A checkpoints "
            "(e.g. Hub, the same directory used by predict_anchor_test.py / Experiment B)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifests_dir = cfg.get_manifests_dir()
    anchor_name = conf["manifests"]["anchor_test"]
    anchor_df = load_manifest(manifests_dir, anchor_name)
    anchor_sha = repro.sha256_file(manifests_dir / anchor_name)
    dataset_root = cfg.get_dataset_root()
    batch = int(conf["training"]["batch_size"])
    dl_cfg = conf["dataloader"]
    criterion = nn.BCEWithLogitsLoss()
    threshold = float(conf["threshold"]["primary"])
    pos_class = conf["label"]["positive_class"]

    pred_cols = ["run_id", "architecture", "condition", "seed", "severity", "image_id",
                 "relative_path", "patient_id", "label", "logit", "probability",
                 "predicted_label_0_5", "entropy"]
    metric_rows = []

    print("=" * 70)
    print(f"EXPERIMENT E - brightness/contrast ({len(ckpts)} clean checkpoints x 3 severity)")
    print("EXPLORATORY (the anchor test has been open since Experiment A Phase 5).")
    print("=" * 70)

    for ckpt_path in ckpts:
        m = CKPT_RE.match(ckpt_path.name)
        if not m:
            print(f"  [skip] unrecognized checkpoint name: {ckpt_path.name}")
            continue
        arch, seed = m["arch"], int(m["seed"])
        run_id = ckpt_path.name[: -len(".best.pt")]

        repro.set_global_determinism(seed, conf["determinism"])
        model = M.build_model(arch, conf, seed).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])

        for severity in (1, 2, 3):
            transform = build_contrast_eval_transform(conf, severity)
            ds = DegradedAnchorDataset(anchor_df, dataset_root, transform, pos_class)
            loader = DataLoader(ds, batch_size=batch, shuffle=False, drop_last=False,
                                 num_workers=int(dl_cfg["num_workers"]),
                                 pin_memory=bool(dl_cfg["pin_memory"]),
                                 collate_fn=collate_meta)

            metrics, loss, preds = engine.evaluate(
                model, loader, criterion, device, threshold, collect_predictions=True)

            df = pd.DataFrame(preds)
            df.insert(0, "severity", severity)
            df.insert(0, "seed", seed)
            df.insert(0, "condition", "clean")  # underlying checkpoint condition
            df.insert(0, "architecture", arch)
            df.insert(0, "run_id", f"{run_id}__contrast_sev{severity}")
            df = df[pred_cols]
            df.to_csv(out_dir / f"{run_id}__contrast_sev{severity}.anchor_predictions.csv",
                       index=False)

            metric_rows.append({
                "arch": arch, "seed": seed, "transform": "contrast", "severity": severity,
                "auroc": metrics["auroc"], "auprc": metrics["auprc"],
                "sensitivity": metrics["sensitivity"], "specificity": metrics["specificity"],
                "balanced_accuracy": metrics["balanced_accuracy"], "f1": metrics["f1"],
                "brier": metrics["brier"], "nll": metrics["nll"], "ece": metrics["ece"],
            })
            print(f"  {run_id} sev{severity} (factor={1 - SEVERITY_PCT[severity]:.2f}): "
                  f"auroc={metrics['auroc']:.4f} bal_acc={metrics['balanced_accuracy']:.4f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = pd.DataFrame(metric_rows).sort_values(["arch", "seed", "severity"])
    out_path = metrics_dir / "exp_e_contrast_metrics.csv"
    out.to_csv(out_path, index=False)
    print("-" * 70)
    print(f"anchor_manifest_sha256: {anchor_sha}")
    print(f"Metrics: {out_path} ({len(out)} rows = 15 checkpoints x 3 severity)")
    print("Combine with experiment_b_deliverables/source_csv/exp_b_metrics.csv "
          "(same columns) for the full Experiment E robustness curves (5/5 dimensions).")


if __name__ == "__main__":
    main()
