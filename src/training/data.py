"""data.py

Protocol-compliant dataset + preprocessing/augmentation (protocol section 8-9).

Pipeline (in order):
  1. read image, convert to grayscale (L)                        [8.2]
  2. square pad (symmetric, fill=black), NOT stretch             [8.3]
  3. resize 224x224, bilinear                                    [8.3]
  4. (train only) RandomAffine(rot +-7, translate 5%, scale .95-1.05) + ColorJitter(b/c 10%)  [9]
  5. replicate grayscale to 3 channels (R=G=B)                   [8.2]
  6. ToTensor + ImageNet normalize                               [8.4]

Val/test use steps 1-3, 5-6 only (no augmentation)              [8.5]
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class SquarePad:
    """Pad a PIL image to a square with symmetric padding (fill=black)."""

    def __init__(self, fill: int = 0):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        pad_l = (side - w) // 2
        pad_t = (side - h) // 2
        pad_r = side - w - pad_l
        pad_b = side - h - pad_t
        # torchvision pad order: (left, top, right, bottom)
        return TF.pad(img, [pad_l, pad_t, pad_r, pad_b], fill=self.fill)


def _interp(name: str):
    return {
        "bilinear": T.InterpolationMode.BILINEAR,
        "bicubic": T.InterpolationMode.BICUBIC,
        "nearest": T.InterpolationMode.NEAREST,
    }[name]


def build_transforms(cfg: dict, train: bool):
    """Compose the preprocessing pipeline from train_config.yaml."""
    img = cfg["image"]
    interp = _interp(img["interpolation"])
    size = int(img["size"])
    fill = int(img["pad_value"])

    steps = [
        T.Grayscale(num_output_channels=1),      # ensure single-channel L
        SquarePad(fill=fill),
        T.Resize((size, size), interpolation=interp),
    ]

    if train:
        a = cfg["augmentation"]
        steps.append(
            T.RandomAffine(
                degrees=float(a["rotation_deg"]),
                translate=(float(a["translate_frac"]), float(a["translate_frac"])),
                scale=(float(a["scale_min"]), float(a["scale_max"])),
                interpolation=interp,
                fill=fill,
            )
        )
        steps.append(
            T.ColorJitter(
                brightness=float(a["brightness"]),
                contrast=float(a["contrast"]),
            )
        )

    steps.append(T.Grayscale(num_output_channels=3))   # replicate L -> RGB
    steps.append(T.ToTensor())
    steps.append(T.Normalize(mean=img["normalize_mean"], std=img["normalize_std"]))
    return T.Compose(steps)


class PneumoniaDataset(Dataset):
    """Reads images listed in a controlled manifest CSV.

    Returns (image_tensor, label_float, meta_dict). label: PNEUMONIA=1, NORMAL=0.
    """

    def __init__(self, manifest_df: pd.DataFrame, dataset_root: Path, cfg: dict, train: bool):
        self.df = manifest_df.reset_index(drop=True)
        self.root = Path(dataset_root)
        self.transform = build_transforms(cfg, train=train)
        self.pos = cfg["label"]["positive_class"]
        # Fail fast on unexpected labels.
        valid = {cfg["label"]["positive_class"], cfg["label"]["negative_class"]}
        bad = set(self.df["label"].unique()) - valid
        if bad:
            raise ValueError(f"Unexpected labels in manifest: {bad}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
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


def load_manifest(manifests_dir: Path, filename: str) -> pd.DataFrame:
    df = pd.read_csv(Path(manifests_dir) / filename)
    required = {"image_id", "relative_path", "patient_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{filename} missing required columns: {missing}")
    return df


def collate_meta(batch):
    """Collate that keeps meta as a list of dicts (default collate mangles it)."""
    tensors = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return tensors, labels, metas
