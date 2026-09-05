"""predict_anchor_test_f_shortcut.py

Experiment F (protocol §28): "shortcut-learning audit with lung masking".
The protocol only writes one line for F (no method specification like B-E) -
the method below was DECIDED together with the user before execution (see
CLAUDE.md §9 Experiment F subsection for the full decision notes):

  1. Region-masking ABLATION (main quantitative evidence): blackout lung vs
     blackout non-lung (compared with clean), measure AUROC/sensitivity/
     specificity. If the model is still far above chance when the LUNGS are
     blacked out (only background/bones/markers visible), that is strong
     evidence the model is using a shortcut outside the lungs.
  2. Grad-CAM overlap (supporting visual evidence): IoU & percentage of
     saliency energy inside the lung mask, computed on the ORIGINAL
     (unmasked) images so it reflects the model's actual behavior as used in
     Experiments A-E.

INFERENCE ONLY, no new training. The anchor test has been open since
Experiment A Phase 5 -> this script's results are EXPLORATORY (protocol
§24), they do not change the Experiment A confirmatory conclusions. Uses the
15 CLEAN Experiment A checkpoints (3 architectures x 5 seeds) - the 'leaky'
checkpoints are NOT used (consistent with the B/D/E pattern).

LUNG MASK SOURCE: the Kermany dataset has NO ground-truth segmentation. An
external pretrained model, `torchxrayvision` - `PSPNet` trained on
ChestX-Det (14 anatomical structures, channels 4 = Left Lung & 5 = Right
Lung are used), is used instead. It is not part of the Experiment A training
pipeline; it is only used to build the evaluation masks for the F audit.

    pip install torchxrayvision   # once; downloads the PSPNet weights automatically
                                    # (~90MB, from the torchxrayvision github release)

ALIGNMENT: the lung mask is computed on the EXACT SAME 224x224 CANVAS seen by
the classifier (grayscale -> square-pad -> resize 224, a preprocessing point
identical to training.data.build_transforms), so mask pixels and model input
pixels correspond 1:1. PSPNet is called on this canvas (it resizes internally
to 512 on its own), and the 512x512x14 output is downsized with NEAREST
interpolation back to 224x224.

Run in stages, in an environment with the 15 clean EXP_A checkpoints (Hub).
The working location on Hub is `$HOME/Pneumonia_research/`:

    cd $HOME/Pneumonia_research/pneumonia_reliability_study
    source $HOME/Pneumonia_research/venv-train/bin/activate
    export CHEST_XRAY_DATA_ROOT=$HOME/Pneumonia_research/chest_xray/chest_xray
    pip install torchxrayvision
    python src/predict_anchor_test_f_shortcut.py --stage masks       # once, ~279 images
    python src/predict_anchor_test_f_shortcut.py --stage ablation    # 15 ckpt x 3 conditions
    python src/predict_anchor_test_f_shortcut.py --stage gradcam     # 15 ckpt x 279 images
    # or all at once:
    python src/predict_anchor_test_f_shortcut.py --stage all

Output (under experiments/experiment_f/):
    masks/lung_masks.npz              - lung masks (279, 224, 224) bool, keyed by image_id order
    masks/lung_coverage_qc.csv        - QC: %lung area per image (detects failed/odd masks)
    metrics/exp_f_ablation_metrics.csv       - 45 rows = 15 ckpt x {clean,lung_blackout,nonlung_blackout}
    predictions/*.anchor_predictions.csv     - per-run per-condition predictions (same format as Exp A/E)
    metrics/exp_f_gradcam_overlap.csv        - 15 rows = 15 ckpt, Grad-CAM IoU & %energy within the lung mask
    gradcam/{run_id}_mean_cam.npy            - mean CAM (224,224) per checkpoint, for figures
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_config as cfg  # noqa: E402
from training import reproducibility as repro  # noqa: E402

CKPT_RE = re.compile(
    r"^EXP_A__(?P<analysis>[a-z0-9]+)__(?P<condition>[a-z0-9]+)__"
    r"(?P<arch>[a-z0-9_]+)__seed_(?P<seed>\d+)\.best\.pt$"
)

LEFT_LUNG_CH = 4
RIGHT_LUNG_CH = 5
CAM_TOP_FRACTION = 0.25  # top-25% saliency mass -> "attended region" for IoU


def parse_args():
    p = argparse.ArgumentParser(description="Experiment F: shortcut-learning audit (lung masking).")
    p.add_argument("--stage", choices=["masks", "ablation", "gradcam", "all"], default="all")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_config() -> dict:
    with open(cfg.get_configs_dir() / "train_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_exp_f_dirs():
    root = cfg.get_experiments_dir() / "experiment_f"
    dirs = {
        "root": root,
        "masks": root / "masks",
        "metrics": root / "metrics",
        "predictions": root / "predictions",
        "gradcam": root / "gradcam",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def clean_checkpoints(ckpt_dir: Path) -> list[Path]:
    ckpts = sorted(
        c for c in ckpt_dir.glob("EXP_A__*__clean__*.best.pt")
        if not c.name.startswith("SMOKE__")
    )
    if len(ckpts) < 15:
        raise SystemExit(
            f"Found {len(ckpts)} clean checkpoints (need 15 = 3 architectures x 5 seeds). "
            "Run in an environment with the full Experiment A checkpoints (Hub)."
        )
    return ckpts


def anchor_canvas(image_path: Path, train_cfg: dict) -> Image.Image:
    """Grayscale -> square-pad -> resize 224. Identical to the first 3 steps of
    training.data.build_transforms (the point at which the classifier's input
    and the lung mask must align pixel-for-pixel)."""
    from training.data import SquarePad, _interp
    import torchvision.transforms as T

    img_cfg = train_cfg["image"]
    interp = _interp(img_cfg["interpolation"])
    size = int(img_cfg["size"])
    fill = int(img_cfg["pad_value"])
    tfm = T.Compose([
        T.Grayscale(num_output_channels=1),
        SquarePad(fill=fill),
        T.Resize((size, size), interpolation=interp),
    ])
    with Image.open(image_path) as im:
        im = im.convert("L")
        return tfm(im)


# ---------------------------------------------------------------------------
# Stage 1: lung mask generation (torchxrayvision PSPNet, run once)
# ---------------------------------------------------------------------------

def stage_masks(conf: dict, dirs: dict):
    import torch
    import torchxrayvision as xrv
    from training.data import load_manifest

    print("=" * 70)
    print("EXPERIMENT F - stage 'masks': lung segmentation (torchxrayvision PSPNet)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_model = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()

    manifests_dir = cfg.get_manifests_dir()
    anchor_name = conf["manifests"]["anchor_test"]
    anchor_df = load_manifest(manifests_dir, anchor_name)
    dataset_root = cfg.get_dataset_root()
    size = int(conf["image"]["size"])

    masks = np.zeros((len(anchor_df), size, size), dtype=bool)
    qc_rows = []

    for i, row in anchor_df.reset_index(drop=True).iterrows():
        canvas = anchor_canvas(dataset_root / row["relative_path"], conf)  # PIL, mode L, size x size
        arr = np.asarray(canvas, dtype=np.float32)  # [0,255]

        # torchxrayvision convention: map 8-bit -> roughly [-1024, 1024]
        # (matches xrv.datasets.normalize(img, maxval=255)).
        xrv_in = ((arr / 255.0) * 2.0 - 1.0) * 1024.0
        x = torch.from_numpy(xrv_in).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]

        with torch.no_grad():
            out = seg_model(x)  # [1, 14, 512, 512] raw logits
            probs = torch.sigmoid(out)[0]  # [14, 512, 512]
            lung = (probs[LEFT_LUNG_CH] > 0.5) | (probs[RIGHT_LUNG_CH] > 0.5)  # [512,512] bool

        lung_np = lung.cpu().numpy().astype(np.uint8) * 255
        lung_img = Image.fromarray(lung_np).resize((size, size), Image.NEAREST)
        lung_mask = np.asarray(lung_img) > 127
        masks[i] = lung_mask

        coverage_pct = 100.0 * lung_mask.sum() / lung_mask.size
        qc_rows.append({
            "image_id": row["image_id"], "relative_path": row["relative_path"],
            "label": row["label"], "lung_coverage_pct": coverage_pct,
        })
        if (i + 1) % 50 == 0 or (i + 1) == len(anchor_df):
            print(f"  {i + 1}/{len(anchor_df)} masks done")

    np.savez_compressed(
        dirs["masks"] / "lung_masks.npz",
        masks=masks,
        image_id=anchor_df["image_id"].to_numpy(),
    )
    qc = pd.DataFrame(qc_rows)
    qc_path = dirs["masks"] / "lung_coverage_qc.csv"
    qc.to_csv(qc_path, index=False)

    print("-" * 70)
    print(f"Masks saved: {dirs['masks'] / 'lung_masks.npz'} ({len(anchor_df)} images)")
    print(f"QC coverage: mean={qc['lung_coverage_pct'].mean():.1f}%  "
          f"min={qc['lung_coverage_pct'].min():.1f}%  max={qc['lung_coverage_pct'].max():.1f}%")
    n_bad = ((qc["lung_coverage_pct"] < 5) | (qc["lung_coverage_pct"] > 70)).sum()
    if n_bad:
        print(f"  WARNING: {n_bad} images with coverage outside the plausible range (<5% or >70%) - "
              f"check {qc_path}, segmentation likely failed for these images (e.g. extreme quality/positioning).")
    else:
        print("  QC OK: all images within the plausible coverage range (5-70%).")


def load_lung_masks(dirs: dict):
    npz = np.load(dirs["masks"] / "lung_masks.npz", allow_pickle=True)
    return npz["masks"], list(npz["image_id"])


# ---------------------------------------------------------------------------
# Stage 2: ablation (blackout paru vs blackout non-paru vs clean)
# ---------------------------------------------------------------------------

class MaskedAnchorDataset:
    """Anchor test read via the standard eval transform, with an optional
    binary region (paru) blacked out (condition='lung_blackout') or KEPT while
    everything else is blacked out (condition='nonlung_blackout'), applied at
    the same canvas stage as the lung mask (post square-pad+resize, pre
    channel-replicate/normalize)."""

    def __init__(self, manifest_df, dataset_root: Path, train_cfg: dict, masks: np.ndarray,
                 image_id_order: list, condition: str, positive_class: str):
        import torchvision.transforms as T

        self.df = manifest_df.reset_index(drop=True)
        self.root = Path(dataset_root)
        self.cfg = train_cfg
        self.masks = masks
        self.id_to_idx = {iid: i for i, iid in enumerate(image_id_order)}
        self.condition = condition
        self.pos = positive_class
        self.fill = int(train_cfg["image"]["pad_value"])
        img_cfg = train_cfg["image"]
        self.final = T.Compose([
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=img_cfg["normalize_mean"], std=img_cfg["normalize_std"]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import torch
        row = self.df.iloc[idx]
        canvas = anchor_canvas(self.root / row["relative_path"], self.cfg)  # PIL L, size x size
        arr = np.asarray(canvas).copy()

        if self.condition != "clean":
            mask = self.masks[self.id_to_idx[row["image_id"]]]  # True = paru
            if self.condition == "lung_blackout":
                arr[mask] = self.fill
            elif self.condition == "nonlung_blackout":
                arr[~mask] = self.fill
            else:
                raise ValueError(self.condition)

        canvas_masked = Image.fromarray(arr, mode="L")
        tensor = self.final(canvas_masked)
        label = 1.0 if row["label"] == self.pos else 0.0
        meta = {"image_id": row["image_id"], "relative_path": row["relative_path"],
                "patient_id": row["patient_id"], "label": row["label"]}
        return tensor, torch.tensor(label, dtype=torch.float32), meta


def collate_meta(batch):
    import torch
    tensors = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return tensors, labels, metas


def stage_ablation(conf: dict, dirs: dict):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from training import models as M
    from training import engine
    from training.data import load_manifest

    print("=" * 70)
    print("EXPERIMENT F - stage 'ablation': lung_blackout vs nonlung_blackout vs clean")
    print("EXPLORATORY (the anchor test has been open since Experiment A Phase 5).")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_a_dir = cfg.get_experiment_a_dir()
    ckpts = clean_checkpoints(exp_a_dir / "checkpoints")

    manifests_dir = cfg.get_manifests_dir()
    anchor_name = conf["manifests"]["anchor_test"]
    anchor_df = load_manifest(manifests_dir, anchor_name)
    anchor_sha = repro.sha256_file(manifests_dir / anchor_name)
    dataset_root = cfg.get_dataset_root()
    masks, image_id_order = load_lung_masks(dirs)

    batch = int(conf["training"]["batch_size"])
    dl_cfg = conf["dataloader"]
    criterion = nn.BCEWithLogitsLoss()
    threshold = float(conf["threshold"]["primary"])
    pos_class = conf["label"]["positive_class"]

    pred_cols = ["run_id", "architecture", "condition_ckpt", "seed", "mask_condition", "image_id",
                 "relative_path", "patient_id", "label", "logit", "probability",
                 "predicted_label_0_5", "entropy"]
    metric_rows = []

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

        for mask_condition in ("clean", "lung_blackout", "nonlung_blackout"):
            ds = MaskedAnchorDataset(anchor_df, dataset_root, conf, masks, image_id_order,
                                      mask_condition, pos_class)
            loader = DataLoader(ds, batch_size=batch, shuffle=False, drop_last=False,
                                 num_workers=int(dl_cfg["num_workers"]),
                                 pin_memory=bool(dl_cfg["pin_memory"]), collate_fn=collate_meta)

            metrics, loss, preds = engine.evaluate(
                model, loader, criterion, device, threshold, collect_predictions=True)

            df = pd.DataFrame(preds)
            df.insert(0, "mask_condition", mask_condition)
            df.insert(0, "seed", seed)
            df.insert(0, "condition_ckpt", "clean")
            df.insert(0, "architecture", arch)
            df.insert(0, "run_id", f"{run_id}__{mask_condition}")
            df = df[pred_cols]
            df.to_csv(dirs["predictions"] / f"{run_id}__{mask_condition}.anchor_predictions.csv",
                       index=False)

            metric_rows.append({
                "arch": arch, "seed": seed, "mask_condition": mask_condition,
                "auroc": metrics["auroc"], "auprc": metrics["auprc"],
                "sensitivity": metrics["sensitivity"], "specificity": metrics["specificity"],
                "balanced_accuracy": metrics["balanced_accuracy"], "f1": metrics["f1"],
                "brier": metrics["brier"], "nll": metrics["nll"], "ece": metrics["ece"],
            })
            print(f"  {run_id} [{mask_condition}]: auroc={metrics['auroc']:.4f} "
                  f"bal_acc={metrics['balanced_accuracy']:.4f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = pd.DataFrame(metric_rows).sort_values(["arch", "seed", "mask_condition"])
    out_path = dirs["metrics"] / "exp_f_ablation_metrics.csv"
    out.to_csv(out_path, index=False)
    print("-" * 70)
    print(f"anchor_manifest_sha256: {anchor_sha}")
    print(f"Ablation metrics: {out_path} ({len(out)} rows = 15 checkpoints x 3 conditions)")


# ---------------------------------------------------------------------------
# Stage 3: Grad-CAM overlap with lung mask (on clean/unmasked anchor images)
# ---------------------------------------------------------------------------

def _to_nchw(t, num_features: int):
    """timm forward_features output can be NCHW (CNNs) or NHWC (Swin, newer
    timm). Detect by comparing dims against model.num_features and permute
    to NCHW if needed."""
    if t.dim() != 4:
        raise ValueError(f"Expected a 4D feature map, got shape {tuple(t.shape)}")
    if t.shape[1] == num_features:
        return t  # already NCHW
    if t.shape[-1] == num_features:
        return t.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
    raise ValueError(f"Cannot infer layout for shape {tuple(t.shape)} (num_features={num_features})")


def compute_gradcam_batch(model, images):
    """Grad-CAM w.r.t. the single logit, using forward_features/forward_head
    (standard timm API, works uniformly across DenseNet121/EfficientNet-B0/
    Swin-Tiny without architecture-specific hooks). Returns cam [B,H,W] in
    [0,1], resized to input resolution."""
    import torch
    import torch.nn.functional as F

    model.zero_grad(set_to_none=True)
    feat = model.forward_features(images)
    feat.retain_grad()
    logits = model.forward_head(feat).squeeze(-1)
    logits.sum().backward()

    grad = feat.grad
    feat_d, grad_d = feat.detach(), grad.detach()
    feat_nchw = _to_nchw(feat_d, model.num_features)
    grad_nchw = _to_nchw(grad_d, model.num_features)

    weights = grad_nchw.mean(dim=(2, 3), keepdim=True)          # [B,C,1,1]
    cam = F.relu((weights * feat_nchw).sum(dim=1))               # [B,h,w]
    cam_max = cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    cam = cam / cam_max
    cam = F.interpolate(cam.unsqueeze(1), size=images.shape[-2:], mode="bilinear",
                         align_corners=False).squeeze(1)
    return cam.cpu().numpy()


def stage_gradcam(conf: dict, dirs: dict):
    import torch
    from torch.utils.data import DataLoader
    from training import models as M
    from training.data import load_manifest

    print("=" * 70)
    print("EXPERIMENT F - stage 'gradcam': saliency overlap with the lung mask")
    print("EXPLORATORY (the anchor test has been open since Experiment A Phase 5).")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_a_dir = cfg.get_experiment_a_dir()
    ckpts = clean_checkpoints(exp_a_dir / "checkpoints")

    manifests_dir = cfg.get_manifests_dir()
    anchor_name = conf["manifests"]["anchor_test"]
    anchor_df = load_manifest(manifests_dir, anchor_name)
    dataset_root = cfg.get_dataset_root()
    masks, image_id_order = load_lung_masks(dirs)
    id_to_idx = {iid: i for i, iid in enumerate(image_id_order)}

    ds = MaskedAnchorDataset(anchor_df, dataset_root, conf, masks, image_id_order,
                              "clean", conf["label"]["positive_class"])
    batch = int(conf["training"]["batch_size"])
    dl_cfg = conf["dataloader"]
    loader = DataLoader(ds, batch_size=batch, shuffle=False, drop_last=False,
                         num_workers=int(dl_cfg["num_workers"]),
                         pin_memory=bool(dl_cfg["pin_memory"]), collate_fn=collate_meta)

    summary_rows = []

    for ckpt_path in ckpts:
        m = CKPT_RE.match(ckpt_path.name)
        if not m:
            continue
        arch, seed = m["arch"], int(m["seed"])
        run_id = ckpt_path.name[: -len(".best.pt")]

        repro.set_global_determinism(seed, conf["determinism"])
        model = M.build_model(arch, conf, seed).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])
        model.eval()

        ious, energy_pcts = [], []
        cam_sum = np.zeros((int(conf["image"]["size"]),) * 2, dtype=np.float64)
        n_seen = 0

        for images, labels, meta in loader:
            images = images.to(device)
            with torch.enable_grad():
                cams = compute_gradcam_batch(model, images)  # [B,H,W] numpy, [0,1]

            for i, mm in enumerate(meta):
                mask = masks[id_to_idx[mm["image_id"]]]
                cam = cams[i]
                cam_sum += cam
                n_seen += 1

                # IoU: top-CAM_TOP_FRACTION-by-mass region vs lung mask
                flat = cam.flatten()
                order = np.argsort(flat)[::-1]
                cum = np.cumsum(flat[order])
                cutoff = cum[-1] * CAM_TOP_FRACTION if cum[-1] > 0 else 0
                k = np.searchsorted(cum, cutoff) + 1 if cum[-1] > 0 else 0
                attended = np.zeros_like(flat, dtype=bool)
                attended[order[:k]] = True
                attended = attended.reshape(cam.shape)

                inter = np.logical_and(attended, mask).sum()
                union = np.logical_or(attended, mask).sum()
                iou = inter / union if union > 0 else float("nan")
                ious.append(iou)

                total_energy = cam.sum()
                energy_in_lung = cam[mask].sum() if total_energy > 0 else 0.0
                energy_pct = 100.0 * energy_in_lung / total_energy if total_energy > 0 else float("nan")
                energy_pcts.append(energy_pct)

        mean_cam = cam_sum / max(n_seen, 1)
        np.save(dirs["gradcam"] / f"{run_id}_mean_cam.npy", mean_cam)

        summary_rows.append({
            "run_id": run_id, "arch": arch, "seed": seed,
            "mean_iou_lung": float(np.nanmean(ious)),
            "mean_energy_in_lung_pct": float(np.nanmean(energy_pcts)),
            "n_images": n_seen,
        })
        print(f"  {run_id}: mean_IoU_lung={np.nanmean(ious):.3f}  "
              f"mean_energy_in_lung%={np.nanmean(energy_pcts):.1f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = pd.DataFrame(summary_rows).sort_values(["arch", "seed"])
    out_path = dirs["metrics"] / "exp_f_gradcam_overlap.csv"
    out.to_csv(out_path, index=False)
    print("-" * 70)
    print(f"Grad-CAM overlap: {out_path} ({len(out)} rows = 15 checkpoints)")
    print(f"Mean-CAM per checkpoint: {dirs['gradcam']}/*_mean_cam.npy (for figures)")


def main():
    args = parse_args()
    conf = load_config()
    dirs = get_exp_f_dirs()

    if args.stage in ("masks", "all"):
        stage_masks(conf, dirs)
    if args.stage in ("ablation", "all"):
        stage_ablation(conf, dirs)
    if args.stage in ("gradcam", "all"):
        stage_gradcam(conf, dirs)

    print("=" * 70)
    print("Experiment F stage(s) done.")


if __name__ == "__main__":
    main()
