"""record_environment.py

Membekukan environment record untuk QC gate (protokol bagian 19.4 & 20):
  - experiments/experiment_a/protocol/environment_record.json
      (Python, OS, torch, torchvision, timm, CUDA, cuDNN, GPU model, driver)
  - requirements-lock.txt  (pip freeze, versi persis reproducible)

Jalankan DI DALAM venv training, setelah semua dependency terpasang:
    python src\record_environment.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_config as cfg
from training import reproducibility as repro


def nvidia_driver_version() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0:
            return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return "UNKNOWN"


def pip_freeze(lock_path: Path) -> int:
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                         capture_output=True, text=True)
    lock_path.write_text(out.stdout, encoding="utf-8")
    return len(out.stdout.splitlines())


def main():
    env = repro.capture_environment()
    env["nvidia_driver_version"] = nvidia_driver_version()
    env["git_commit"] = repro.git_commit()

    proto_dir = cfg.get_experiment_a_dir() / "protocol"
    proto_dir.mkdir(parents=True, exist_ok=True)
    env_path = proto_dir / "environment_record.json"
    repro.save_json(env, env_path)

    lock_path = cfg.get_project_root() / "requirements-lock.txt"
    n = pip_freeze(lock_path)

    print("Environment record written:")
    print(f"  {env_path}")
    print(f"  {lock_path}  ({n} pinned packages)")
    print()
    print(f"  python={env['python']} torch={env['torch']} cuda={env['cuda_version']} "
          f"cudnn={env['cudnn_version']} driver={env['nvidia_driver_version']}")
    for g in env["gpus"]:
        print(f"  GPU {g['index']}: {g['name']} ({g['total_memory_MB']} MB, cc {g['capability']})")


if __name__ == "__main__":
    main()
