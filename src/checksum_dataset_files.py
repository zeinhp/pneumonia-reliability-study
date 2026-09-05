"""
checksum_dataset_files.py

Computes SHA-256 for every original image file under the read-only dataset
root (does NOT read/modify/rename anything else). Supports resumable
chunked execution (checkpoints to data/checksums/_dataset_sha256_parts/)
because a full run can exceed a single shell call's time budget. These
checkpoint files are intermediate/scratch and are covered by .gitignore -
they are intentionally NOT deleted automatically by this script.

Usage:
    python3 checksum_dataset_files.py <batch_size> <time_budget_seconds>
"""
import sys
import json
import hashlib
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_config as cfg

DATA_ROOT = cfg.get_dataset_root()
CHECKSUMS_DIR = cfg.get_checksums_dir()
PARTS_DIR = CHECKSUMS_DIR / "_dataset_sha256_parts"
FILELIST_PATH = CHECKSUMS_DIR / "_dataset_filelist.json"
PARTS_DIR.mkdir(parents=True, exist_ok=True)

if not FILELIST_PATH.exists():
    filelist = []
    for split in ["train", "test", "val"]:
        for label in ["NORMAL", "PNEUMONIA"]:
            d = DATA_ROOT / split / label
            if not d.is_dir():
                continue
            for fpath in sorted(d.iterdir()):
                if fpath.is_file() and not fpath.name.startswith("."):
                    rel = f"{split}/{label}/{fpath.name}"
                    filelist.append({"relative_path": rel, "filepath": str(fpath)})
    with open(FILELIST_PATH, "w") as f:
        json.dump(filelist, f)
else:
    with open(FILELIST_PATH) as f:
        filelist = json.load(f)

TOTAL = len(filelist)
BATCH_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
TIME_BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 35.0

done = set()
for pf in PARTS_DIR.iterdir():
    if pf.suffix == ".json":
        done.add(int(pf.stem.replace("part_", "")))

start = time.time()
n_done_this_run = 0
for i, item in enumerate(filelist):
    if i in done:
        continue
    if time.time() - start > TIME_BUDGET or n_done_this_run >= BATCH_SIZE:
        break
    h = hashlib.sha256()
    with open(item["filepath"], "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    rec = {
        "relative_path": item["relative_path"],
        "sha256": h.hexdigest(),
        "file_size_bytes": Path(item["filepath"]).stat().st_size,
    }
    with open(PARTS_DIR / f"part_{i}.json", "w") as f:
        json.dump(rec, f)
    n_done_this_run += 1

n_total_done = len([p for p in PARTS_DIR.iterdir() if p.suffix == ".json"])
print(f"this_run={n_done_this_run} done={n_total_done} total={TOTAL} remaining={TOTAL - n_total_done}")

if n_total_done == TOTAL:
    rows = []
    for pf in PARTS_DIR.iterdir():
        if pf.suffix == ".json":
            with open(pf) as f:
                rows.append(json.load(f))
    df = pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)
    out_csv = CHECKSUMS_DIR / "dataset_files_sha256.csv"
    df.to_csv(out_csv, index=False)
    print(f"CONSOLIDATED: wrote {len(df)} rows to {out_csv}")
    assert len(df) == 5856, f"expected 5856, got {len(df)}"
