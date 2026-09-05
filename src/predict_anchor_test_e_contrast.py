"""predict_anchor_test_e_contrast.py

Eksperimen E (protokol §28) - dimensi terakhir yang belum dijalankan:
robustness terhadap brightness/contrast. Empat dimensi lain (noise, blur,
compression, resolution) sudah dieksekusi sebagai "Eksperimen B" (lihat
experiment_b_deliverables/). Skrip ini melengkapi E jadi 5/5 dimensi.

Desain (mengikuti pola exp_b_degrade_infer.py, DIKUNCI CLAUDE.md §9 Eksperimen B):
  - INFERENSI SAJA, tanpa training. Anchor test sudah terbuka sejak Fase 5 Eksperimen A
    -> hasil skrip ini EXPLORATORY (protokol §24), tidak mengubah kesimpulan confirmatory A.
  - Memakai 15 checkpoint CLEAN Eksperimen A (3 arsitektur x 5 seed) - checkpoint
    'leaky' TIDAK dipakai (samakan dengan Eksperimen B, yang juga hanya pakai clean).
  - Titik degradasi = preprocess: grayscale -> square-pad -> resize 224 (uint8) ->
    APPLY brightness/contrast -> replikasi 3-channel -> ToTensor -> normalisasi ImageNet.
    Sama seperti titik degradasi Eksperimen B (bukan native-image).
  - Severity dikunci di CLAUDE.md §9 (baris "Brightness/Contrast +- (opsional)"):
    level 1/2/3 = 20% / 35% / 50%.
  - KEPUTUSAN DESAIN (baru, didokumentasikan di sini karena tabel CLAUDE.md hanya
    menulis magnitudo "+-" tanpa menetapkan arah): brightness & contrast diturunkan
    (factor = 1 - pct), bukan dinaikkan atau diacak arah. Alasan: 4 dimensi degradasi
    lain di Eksperimen B semuanya derajat kerusakan searah (noise naik, blur naik,
    kualitas JPEG turun, resolusi turun) - citra makin sulit dibaca, bukan bidirectional
    stress test. Menurunkan brightness+contrast konsisten dengan itu (radiograf under-
    exposed/low-contrast = kegagalan realistis kualitas akuisisi). Deterministik penuh
    (bukan proses acak) - tidak perlu seeding per-citra seperti noise Gaussian.
  - Kondisi 'clean' (severity 0) TIDAK diulang di sini - sudah ada di
    experiment_b_deliverables/source_csv/exp_b_metrics.csv (transform='clean'), dipakai
    bersama saat menggabungkan kurva robustness E lengkap.
  - Skema metrik output SAMA PERSIS dengan exp_b_metrics.csv (arch, seed, transform,
    severity, auroc, auprc, sensitivity, specificity, balanced_accuracy, f1, brier, nll, ece)
    supaya bisa langsung digabung (pd.concat) dengan hasil B saat menulis laporan E.

Jalankan di lingkungan tempat 15 checkpoint clean EXP_A tersedia (Hub, sama seperti
predict_anchor_test.py / exp_b_degrade_infer.py sebelumnya):

    export CHEST_XRAY_DATA_ROOT=$HOME/chest_xray/chest_xray
    python src/predict_anchor_test_e_contrast.py

Output:
    experiments/experiment_e/metrics/exp_e_contrast_metrics.csv   (metrik agregat, 45 baris = 15 ckpt x 3 severity)
    experiments/experiment_e/predictions/*.anchor_predictions.csv (prediksi per-run per-severity, format sama predict_anchor_test.py)
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
            f"Ditemukan {len(ckpts)} checkpoint clean (butuh 15 = 3 arsitektur x 5 seed). "
            "Jalankan skrip ini di lingkungan yang punya checkpoint Eksperimen A lengkap "
            "(mis. Hub, direktori yang sama dipakai predict_anchor_test.py / Eksperimen B)."
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
    print(f"EKSPERIMEN E - brightness/contrast ({len(ckpts)} checkpoint clean x 3 severity)")
    print("EXPLORATORY (anchor test sudah terbuka sejak Fase 5 Eksperimen A).")
    print("=" * 70)

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
    print(f"Metrik: {out_path} ({len(out)} baris = 15 checkpoint x 3 severity)")
    print("Gabungkan dengan experiment_b_deliverables/source_csv/exp_b_metrics.csv "
          "(kolom sama) untuk kurva robustness Eksperimen E lengkap (5/5 dimensi).")


if __name__ == "__main__":
    main()
