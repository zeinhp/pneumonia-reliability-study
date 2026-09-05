"""reproducibility.py

Seeding, full determinism, environment capture, run-ID construction, and run
record persistence (protocol section 19).

torch is imported lazily *inside* functions so that pure-metadata helpers
(run_id, sha256, config loading) can be unit-tested without a torch install.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Hashing / IDs (torch-free)
# ---------------------------------------------------------------------------
def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_run_id(analysis: str, condition: str, architecture: str, seed: int) -> str:
    """Protocol 19.1 format:
    EXP_A__<analysis>__<condition>__<architecture>__seed_<seed>
    e.g. EXP_A__controlled__clean__densenet121__seed_42
    """
    return f"EXP_A__{analysis}__{condition}__{architecture}__seed_{seed}"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit(default: str = "UNKNOWN") -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return default


# ---------------------------------------------------------------------------
# Determinism + environment (need torch)
# ---------------------------------------------------------------------------
def set_global_determinism(seed: int, det_cfg: dict) -> None:
    """Seed every RNG and enforce deterministic algorithms (CLAUDE.md #4)."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if det_cfg.get("cudnn_deterministic", True):
        torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = bool(det_cfg.get("cudnn_benchmark", False))

    if det_cfg.get("use_deterministic_algorithms", True):
        # CUBLAS workspace config is required for deterministic CUDA matmul.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some ops lack deterministic kernels; warn_only keeps run going
            # while still flipping the deterministic flag where possible.
            torch.use_deterministic_algorithms(True, warn_only=True)


def make_dataloader_generator(seed: int):
    """Seeded generator so DataLoader shuffling is reproducible."""
    import torch
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:  # DataLoader worker_init_fn
    import random
    import numpy as np
    import torch
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def capture_environment() -> dict[str, Any]:
    """Software/hardware fingerprint (protocol 19.4)."""
    import torch
    try:
        import torchvision
        tv = torchvision.__version__
    except Exception:
        tv = "NOT_INSTALLED"
    try:
        import timm
        timm_v = timm.__version__
    except Exception:
        timm_v = "NOT_INSTALLED"

    cuda_available = torch.cuda.is_available()
    gpus = []
    if cuda_available:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i, "name": p.name,
                "total_memory_MB": round(p.total_memory / (1024 ** 2)),
                "capability": f"{p.major}.{p.minor}",
            })

    return {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": tv,
        "timm": timm_v,
        "cuda_available": cuda_available,
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": (torch.backends.cudnn.version() if cuda_available else None),
        "gpus": gpus,
        "captured_at": utc_timestamp(),
    }


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
