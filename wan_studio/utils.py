from __future__ import annotations

import os
import json
import hashlib
import platform
import subprocess
import shutil
from typing import Any, Dict, Optional, List

def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def sha256_file(path: str, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def run_cmd(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

def safe_makedirs(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def env_report() -> Dict[str, Any]:
    rep: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import torch
        rep["torch"] = torch.__version__
        rep["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            rep["cuda"] = torch.version.cuda
            rep["gpu_name"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info()
            rep["vram_total_gb"] = round(total / (1024**3), 2)
            rep["vram_free_gb"] = round(free / (1024**3), 2)
    except Exception as e:
        rep["torch_error"] = str(e)

    try:
        import diffusers
        rep["diffusers"] = getattr(diffusers, "__version__", "unknown")
    except Exception as e:
        rep["diffusers_error"] = str(e)

    return rep

def write_json(path: str, data: Any) -> None:
    safe_makedirs(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
