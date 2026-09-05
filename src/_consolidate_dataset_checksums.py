"""
_consolidate_dataset_checksums.py

Consolidates the per-file JSON checkpoints written by
checksum_dataset_files.py into the final data/checksums/dataset_files_sha256.csv.
Separate script because reading thousands of small JSON files can exceed a
single shell call's time budget when combined with the hashing pass itself.
"""
import sys
import json
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_config as cfg

CHECKSUMS_DIR = cfg.get_checksums_dir()
PARTS_DIR = CHECKSUMS_DIR / "_dataset_sha256_parts"

rows = []
for pf in PARTS_DIR.iterdir():
    if pf.suffix == ".json":
        with open(pf) as f:
            rows.append(json.load(f))

df = pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)
out_csv = CHECKSUMS_DIR / "dataset_files_sha256.csv"
df.to_csv(out_csv, index=False)
print(f"Wrote {len(df)} rows to {out_csv}")
assert len(df) == 5856, f"expected 5856, got {len(df)}"
