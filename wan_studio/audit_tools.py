from __future__ import annotations

import os
import platform
import subprocess
import sys
import textwrap
from datetime import datetime
from typing import Any, Optional

from .config import ProjectConfig

def _run(cmd: list[str]) -> str:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        out = (p.stdout or "").strip()
        return out
    except Exception as e:
        return f"(error running {cmd!r}: {e})"

def _try_import(name: str) -> str:
    try:
        mod = __import__(name)
        v = getattr(mod, "__version__", None)
        return f"{name}={v}" if v is not None else f"{name}=ok"
    except Exception as e:
        return f"{name}=missing ({e})"

def run_audit(cfg: ProjectConfig, *, include_env: bool = True) -> str:
    """Best-effort diagnostic report to paste into issues."""
    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"Wan Studio Audit — {now}")
    lines.append("=" * 60)
    lines.append(f"python: {sys.version.split()[0]}  exe: {sys.executable}")
    lines.append(f"platform: {platform.platform()}")
    lines.append(f"cwd: {os.getcwd()}")
    lines.append("")

    # Core config snapshot (high-signal)
    try:
        lines.append("ProjectConfig (high-signal)")
        lines.append("-" * 60)
        lines.append(f"backend={getattr(cfg,'backend',None)}  mode={getattr(cfg,'mode',None)}")
        lines.append(f"model_id={getattr(cfg,'model_id',None)}")
        lines.append(f"fps={getattr(cfg,'fps',None)}  size={getattr(cfg,'width',None)}x{getattr(cfg,'height',None)}")
        lines.append(f"steps={getattr(cfg,'num_inference_steps',None)}  cfg1={getattr(cfg,'guidance_scale',None)}  cfg2={getattr(cfg,'guidance_scale_2',None)}")
        lines.append(f"chunk_seconds={getattr(cfg,'chunk_seconds',None)}  overlap_frames={getattr(cfg,'overlap_frames',None)}  blend_mode={getattr(cfg,'blend_mode',None)}")
        lines.append(f"use_unipc={bool(getattr(cfg,'use_unipc',False))}  flow_shift={getattr(cfg,'flow_shift',None)}")
        lines.append("stability: "
                     f"cut_strict={bool(getattr(cfg,'cut_strict_between_clips',False))}  "
                     f"reset_between={bool(getattr(cfg,'reset_continuity_between_clips',False))}  "
                     f"strict_clip={bool(getattr(cfg,'strict_clip_coherence',False))}  "
                     f"force_first={bool(getattr(cfg,'force_first_frame_to_conditioning',False))}  "
                     f"ultra_isolated={bool(getattr(cfg,'ultra_isolated_clips',False))}")
        lines.append("cache: "
                     f"clip_cache={bool(getattr(cfg,'clip_cache_enabled',False))}  "
                     f"clip_cache_dirname={getattr(cfg,'clip_cache_dirname',None)}  "
                     f"cache_max_gb={getattr(cfg,'cache_max_gb',None)}  "
                     f"prune_on_start={bool(getattr(cfg,'prune_cache_on_start',False))}")
        lines.append("")
    except Exception:
        pass

    # Package versions
    lines.append("Python packages")
    lines.append("-" * 60)
    for name in ("torch", "diffusers", "transformers", "accelerate", "peft", "safetensors", "opencv-python"):
        lines.append(_try_import(name))
    lines.append("")

    # Torch + CUDA
    try:
        import torch  # type: ignore
        lines.append("Torch / CUDA")
        lines.append("-" * 60)
        lines.append(f"torch={getattr(torch,'__version__','?')}")
        lines.append(f"cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            try:
                n = torch.cuda.device_count()
                lines.append(f"device_count={n}")
                for i in range(n):
                    name = torch.cuda.get_device_name(i)
                    cap = torch.cuda.get_device_capability(i)
                    lines.append(f"  [{i}] {name}  capability={cap}")
            except Exception as e:
                lines.append(f"cuda_device_info_error: {e}")
        lines.append("")
    except Exception as e:
        lines.append(f"Torch import failed: {e}")
        lines.append("")

    if include_env:
        # ffmpeg
        lines.append("External tools")
        lines.append("-" * 60)
        lines.append("ffmpeg:")
        lines.append(_run(["ffmpeg", "-version"]).splitlines()[0] if _run(["ffmpeg","-version"]) else "(missing)")
        lines.append("")

        # nvidia-smi (if available)
        lines.append("nvidia-smi:")
        smi = _run(["nvidia-smi"])
        if smi:
            # keep it short
            lines.append("\n".join(smi.splitlines()[:20]))
        else:
            lines.append("(missing)")
        lines.append("")

    return "\n".join(lines).strip() + "\n"
