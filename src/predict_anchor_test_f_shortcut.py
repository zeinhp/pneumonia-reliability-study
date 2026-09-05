"""predict_anchor_test_f_shortcut.py

Eksperimen F (protokol §28): "shortcut-learning audit dengan lung masking".
Protokol hanya menulis satu baris untuk F (tidak ada spesifikasi metode seperti
B-E) - metode di bawah ini DIPUTUSKAN bersama user sebelum eksekusi (lihat
CLAUDE.md §9 subbagian Eksperimen F untuk catatan keputusan lengkap):

  1. Region-masking ABLATION (bukti kuantitatif utama): blackout paru vs
     blackout non-paru (bandingkan dengan clean), ukur AUROC/sensitivitas/
     spesifisitas. Kalau model masih jauh di atas-chance saat PARU di-blackout
     (hanya latar/tulang/marker terlihat), itu bukti kuat model memakai
     shortcut di luar paru.
  2. Grad-CAM overlap (bukti pendukung visual): IoU & persentase energi
     saliency di dalam mask paru, dihitung pada citra ORIGINAL (tidak
     dimasking) supaya mencerminkan perilaku model yang sesungguhnya dipakai
     di Eksperimen A-E.

INFERENSI SAJA, tanpa training baru. Anchor test sudah terbuka sejak Fase 5
Eksperimen A -> hasil skrip ini EXPLORATORY (protokol §24), tidak mengubah
kesimpulan confirmatory A. Memakai 15 checkpoint CLEAN Eksperimen A (3
arsitektur x 5 seed) - checkpoint 'leaky' TIDAK dipakai (konsisten dgn pola
B/D/E).

SUMBER MASK PARU: dataset Kermany TIDAK punya ground-truth segmentasi. Dipakai
model eksternal pretrained `torchxrayvision` - `PSPNet` terlatih di
ChestX-Det (14 struktur anatomis, dipakai channel 4 = Left Lung & 5 = Right
Lung). Bukan bagian dari pipeline training Eksperimen A; hanya dipakai untuk
membangun mask evaluasi audit F.

    pip install torchxrayvision   # sekali; download weight PSPNet otomatis
                                    # (~90MB, dari github release torchxrayvision)

ALIGNMENT: mask paru dihitung pada CANVAS 224x224 YANG SAMA PERSIS dengan yang
dilihat classifier (grayscale -> square-pad -> resize 224, titik preprocessing
identik dgn training.data.build_transforms), supaya piksel mask & piksel input
model berkorespondensi 1:1. PSPNet dipanggil pada canvas ini (ia me-resize
sendiri ke 512 secara internal), output 512x512x14 di-downsize NEAREST balik
ke 224x224.

Jalankan bertahap di lingkungan yang punya 15 checkpoint clean EXP_A (Hub). Lokasi kerja di Hub
ada di `$HOME/Pneumonia_research/`:

    cd $HOME/Pneumonia_research/pneumonia_reliability_study
    source $HOME/Pneumonia_research/venv-train/bin/activate
    export CHEST_XRAY_DATA_ROOT=$HOME/Pneumonia_research/chest_xray/chest_xray
    pip install torchxrayvision
    python src/predict_anchor_test_f_shortcut.py --stage masks       # sekali, ~279 citra
    python src/predict_anchor_test_f_shortcut.py --stage ablation    # 15 ckpt x 3 kondisi
    python src/predict_anchor_test_f_shortcut.py --stage gradcam     # 15 ckpt x 279 citra
    # atau semua sekaligus:
    python src/predict_anchor_test_f_shortcut.py --stage all

Output (di experiments/experiment_f/):
    masks/lung_masks.npz              - mask paru (279, 224, 224) bool, keyed by image_id order
    masks/lung_coverage_qc.csv        - QC: %area paru per citra (deteksi mask gagal/aneh)
    metrics/exp_f_ablation_metrics.csv       - 45 baris = 15 ckpt x {clean,lung_blackout,nonlung_blackout}
    predictions/*.anchor_predictions.csv     - prediksi per-run per-kondisi (format sama Exp A/E)
    metrics/exp_f_gradcam_overlap.csv        - 15 baris = 15 ckpt, IoU & %energi Grad-CAM dlm mask paru
    gradcam/{run_id}_mean_cam.npy            - CAM rata-rata (224,224) per checkpoint, utk figure
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
    p = argparse.ArgumentParser(description="Eksperimen F: shortcut-learning audit (lung masking).")
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
            f"Ditemukan {len(ckpts)} checkpoint clean (butuh 15 = 3 arsitektur x 5 seed). "
            "Jalankan di lingkungan yang punya checkpoint Eksperimen A lengkap (Hub)."
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
    print("EKSPERIMEN F - stage 'masks': segmentasi paru (torchxrayvision PSPNet)")
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
            print(f"  {i + 1}/{len(anchor_df)} mask selesai")

    np.savez_compressed(
        dirs["masks"] / "lung_masks.npz",
        masks=masks,
        image_id=anchor_df["image_id"].to_numpy(),
    )
    qc = pd.DataFrame(qc_rows)
    qc_path = dirs["masks"] / "lung_coverage_qc.csv"
    qc.to_csv(qc_path, index=False)

    print("-" * 70)
    print(f"Mask tersimpan: {dirs['masks'] / 'lung_masks.npz'} ({len(anchor_df)} citra)")
    print(f"QC coverage: mean={qc['lung_coverage_pct'].mean():.1f}%  "
          f"min={qc['lung_coverage_pct'].min():.1f}%  max={qc['lung_coverage_pct'].max():.1f}%")
    n_bad = ((qc["lung_coverage_pct"] < 5) | (qc["lung_coverage_pct"] > 70)).sum()
    if n_bad:
        print(f"  PERINGATAN: {n_bad} citra dengan coverage di luar rentang wajar (<5% atau >70%) - "
              f"cek {qc_path}, kemungkinan segmentasi gagal untuk citra tsb (mis. kualitas/posisi ekstrem).")
    else:
        print("  QC OK: semua citra dalam rentang coverage wajar (5-70%).")


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
    print("EKSPERIMEN F - stage 'ablation': lung_blackout vs nonlung_blackout vs clean")
    print("EXPLORATORY (anchor test sudah terbuka sejak Fase 5 Eksperimen A).")
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
            print(f"  [skip] nama checkpoint tak dikenal: {ckpt_path.name}")
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
    print(f"Metrik ablation: {out_path} ({len(out)} baris = 15 checkpoint x 3 kondisi)")


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
    print("EKSPERIMEN F - stage 'gradcam': saliency overlap dengan mask paru")
    print("EXPLORATORY (anchor test sudah terbuka sejak Fase 5 Eksperimen A).")
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
    print(f"Grad-CAM overlap: {out_path} ({len(out)} baris = 15 checkpoint)")
    print(f"Mean-CAM per checkpoint: {dirs['gradcam']}/*_mean_cam.npy (utk figure)")


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
    print("Eksperimen F stage(s) selesai.")


if __name__ == "__main__":
    main()
