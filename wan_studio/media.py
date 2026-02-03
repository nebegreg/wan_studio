from __future__ import annotations

import os
import time
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

@dataclass
class ExtractResult:
    ok: bool
    path: str
    message: str = ""

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _run(cmd: list[str]) -> Tuple[bool, str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        out = p.stdout or ""
        return (p.returncode == 0), out
    except Exception as e:
        return False, str(e)

def extract_frame_ffmpeg(video_path: str, t_seconds: float, out_png: str) -> ExtractResult:
    ensure_dir(os.path.dirname(os.path.abspath(out_png)))
    cmd = ["ffmpeg", "-y", "-ss", str(float(t_seconds)), "-i", video_path, "-frames:v", "1", "-q:v", "2", out_png]
    ok, out = _run(cmd)
    return ExtractResult(ok=ok and os.path.exists(out_png), path=out_png, message=out)

def exr_to_png(exr_path: str, out_png: str, exposure_ev: float = 0.0, gamma: float = 2.2, tonemap: str = "reinhard") -> ExtractResult:
    """Convert EXR -> PNG for conditioning.

    Strategy:
      1) Try python OpenEXR (best control).
      2) Fallback to ffmpeg conversion if OpenEXR isn't available.
    """
    ensure_dir(os.path.dirname(os.path.abspath(out_png)))

    # --- 1) OpenEXR path ---
    try:
        import OpenEXR  # type: ignore
        import Imath    # type: ignore

        exr = OpenEXR.InputFile(exr_path)
        header = exr.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        # pick channels
        chs = header.get("channels", {})
        def get(ch: str) -> bytes:
            return exr.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))

        # Prefer RGB, fallback to BGR, fallback to single channel
        if all(c in chs for c in ("R","G","B")):
            R = np.frombuffer(get("R"), dtype=np.float32).reshape((height, width))
            G = np.frombuffer(get("G"), dtype=np.float32).reshape((height, width))
            B = np.frombuffer(get("B"), dtype=np.float32).reshape((height, width))
        elif all(c in chs for c in ("B","G","R")):
            B = np.frombuffer(get("B"), dtype=np.float32).reshape((height, width))
            G = np.frombuffer(get("G"), dtype=np.float32).reshape((height, width))
            R = np.frombuffer(get("R"), dtype=np.float32).reshape((height, width))
        else:
            # take first available channel
            first = next(iter(chs.keys()))
            C = np.frombuffer(get(first), dtype=np.float32).reshape((height, width))
            R = G = B = C

        img = np.stack([R,G,B], axis=-1)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        img = np.clip(img, 0.0, None)

        # exposure
        img = img * (2.0 ** float(exposure_ev))

        # tonemap
        if tonemap.lower() == "reinhard":
            img = img / (1.0 + img)
        elif tonemap.lower() == "aces":
            # simple ACES-ish curve (approx)
            a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
            img = np.clip((img*(a*img+b)) / (img*(c*img+d)+e), 0.0, 1.0)
        else:
            img = np.clip(img, 0.0, 1.0)

        # gamma to sRGB-ish
        g = max(0.1, float(gamma))
        img = np.clip(img, 0.0, 1.0) ** (1.0 / g)

        out = (img * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(out, mode="RGB").save(out_png, "PNG")
        return ExtractResult(ok=True, path=out_png, message="Converted via OpenEXR")
    except Exception as e:
        # continue to ffmpeg fallback
        openexr_err = str(e)

    # --- 2) ffmpeg fallback ---
    cmd = ["ffmpeg", "-y", "-i", exr_path, "-frames:v", "1", out_png]
    ok, out = _run(cmd)
    msg = "Converted via ffmpeg" if ok else f"EXR convert failed. OpenEXR error: {openexr_err}\nFFmpeg output:\n{out}"
    return ExtractResult(ok=ok and os.path.exists(out_png), path=out_png, message=msg)

def make_inputs_dir(base_dir: str = ".") -> str:
    p = os.path.abspath(os.path.join(base_dir, "inputs"))
    ensure_dir(p)
    return p

def unique_path(dir_path: str, prefix: str, ext: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(dir_path, f"{prefix}_{ts}.{ext.lstrip('.')}")
