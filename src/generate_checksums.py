"""
generate_checksums.py
Compute SHA-256 checksums for files. Read-only operation; does not modify sources.

Usage:
    python3 generate_checksums.py <mode>
    mode = "source_audit" | "manifests"
"""
import sys
import hashlib
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_config as cfg


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    mode = sys.argv[1]

    if mode == "source_audit":
        src_dir = cfg.get_source_audit_dir()
        out_csv = cfg.get_checksums_dir() / "source_audit_sha256.csv"
        rows = []
        for fpath in sorted(src_dir.iterdir()):
            if not fpath.is_file():
                continue
            rows.append({
                "filename": fpath.name,
                "sha256": sha256_of(fpath),
                "file_size_bytes": fpath.stat().st_size,
            })
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["filename", "sha256", "file_size_bytes"])
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows to {out_csv}")

    elif mode == "manifests":
        man_dir = cfg.get_manifests_dir()
        out_path = cfg.get_checksums_dir() / "manifest_checksums.sha256"
        lines = []
        for fpath in sorted(man_dir.iterdir()):
            if not fpath.is_file():
                continue
            digest = sha256_of(fpath)
            lines.append(f"{digest}  {fpath.name}")
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Wrote {len(lines)} lines to {out_path}")
    else:
        print("unknown mode", mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
