"""
import_audit_results.py

Freezes the existing dataset audit into this project as a read-only
snapshot. Copies the audit outputs verbatim into data/source_audit/ (no
edits) and records their SHA-256 checksums in
data/checksums/source_audit_sha256.csv.

This script is idempotent - re-running it re-copies the same source files
and recomputes the same checksums; it never regenerates the audit itself.

The audit source directory is resolved dynamically (sibling of the project
root, or CHEST_XRAY_AUDIT_SOURCE_DIR env var) - no hardcoded session path.
"""
import sys
import shutil
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_config as cfg

REQUIRED_FILES = [
    "manifest.csv",
    "proposed_split_mapping.csv",
    "class_distribution.csv",
    "exact_duplicates.csv",
    "near_duplicates.csv",
    "patient_split_overlap.csv",
    "low_resolution_images.csv",
    "potential_blurry_images.csv",
    "resolution_stats.csv",
    "mode_counts.csv",
    "Audit_Dataset_Chest_Xray_Pneumonia.docx",
]


def main():
    audit_src = cfg.get_audit_import_source_dir()
    dest = cfg.get_source_audit_dir()
    dest.mkdir(parents=True, exist_ok=True)

    missing_src = [f for f in REQUIRED_FILES if not (audit_src / f).is_file()]
    if missing_src:
        print(f"FATAL: missing source audit files in {audit_src}: {missing_src}")
        sys.exit(1)

    for f in REQUIRED_FILES:
        shutil.copy2(audit_src / f, dest / f)
    print(f"Copied {len(REQUIRED_FILES)} audit files from {audit_src} to {dest}")

    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here / "generate_checksums.py"), "source_audit"], check=True)


if __name__ == "__main__":
    main()
