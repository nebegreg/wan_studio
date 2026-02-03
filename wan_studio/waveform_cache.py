"""Waveform peak extraction with caching.

Used by the Studio Premiere timeline to draw lightweight audio waveforms
for per-shot stems (dialogue / vfx). Designed to be fast and robust.

Dependencies: ffmpeg + numpy (already in requirements).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from typing import Optional

import numpy as np


_MEM_CACHE: dict[tuple[str, float, int, int], np.ndarray] = {}


def _disk_cache_path(key_hex: str) -> str:
    base = os.path.join(tempfile.gettempdir(), "wan_studio_waveforms")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, key_hex + ".npy")


def load_peaks(path: str, *, width: int = 512, sr: int = 8000) -> Optional[np.ndarray]:
    """Return a normalized peak envelope (0..1) of length ~width.

    - path: audio file path
    - width: number of buckets (points) to return
    - sr: decode sample rate (lower = faster)
    """
    if not path:
        return None
    p = os.path.abspath(str(path))
    if not os.path.exists(p):
        return None

    try:
        mtime = float(os.path.getmtime(p))
    except Exception:
        mtime = 0.0

    width = int(max(32, min(2048, width)))
    sr = int(max(2000, min(48000, sr)))

    mem_key = (p, mtime, width, sr)
    if mem_key in _MEM_CACHE:
        return _MEM_CACHE[mem_key]

    h = hashlib.md5(f"{p}|{mtime}|{width}|{sr}".encode("utf-8", "ignore")).hexdigest()
    disk = _disk_cache_path(h)
    if os.path.exists(disk):
        try:
            arr = np.load(disk)
            _MEM_CACHE[mem_key] = arr
            return arr
        except Exception:
            pass

    # Decode with ffmpeg to float32 mono.
    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", p,
        "-ac", "1",
        "-ar", str(sr),
        "-f", "f32le",
        "-"
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw, _ = proc.communicate(timeout=20)
    except Exception:
        return None

    if not raw:
        return None

    audio = np.frombuffer(raw, dtype=np.float32)
    if audio.size == 0:
        return None

    # Peak per bucket
    bucket = max(1, int(audio.size // width))
    n = int(np.ceil(audio.size / bucket))
    peaks = np.empty((n,), dtype=np.float32)
    for i in range(n):
        seg = audio[i*bucket:(i+1)*bucket]
        if seg.size:
            peaks[i] = float(np.max(np.abs(seg)))
        else:
            peaks[i] = 0.0

    # Normalize
    mx = float(np.max(peaks)) if peaks.size else 0.0
    if mx > 1e-9:
        peaks /= mx

    # Save (best-effort)
    try:
        np.save(disk, peaks)
    except Exception:
        pass

    _MEM_CACHE[mem_key] = peaks
    return peaks

def load_peaks_segment(path: str, *, start_sec: float, dur_sec: float, width: int = 512, sr: int = 8000) -> Optional[np.ndarray]:
    """Return peaks for a segment of an audio file (for global music bed slicing).

    Uses ffmpeg -ss/-t (fast seek, sufficient for UI waveforms). Cached on disk.
    """
    if not path:
        return None
    p = os.path.abspath(str(path))
    if not os.path.exists(p):
        return None

    start_sec = max(0.0, float(start_sec or 0.0))
    dur_sec = max(0.0, float(dur_sec or 0.0))
    if dur_sec <= 1e-6:
        return None

    try:
        mtime = float(os.path.getmtime(p))
    except Exception:
        mtime = 0.0

    width = int(max(32, min(2048, width)))
    sr = int(max(2000, min(48000, sr)))

    # segment cache key includes time window
    mem_key = (p, mtime, width, sr, round(start_sec, 3), round(dur_sec, 3))
    if mem_key in _MEM_CACHE:
        return _MEM_CACHE[mem_key]

    h = hashlib.md5(f"{p}|{mtime}|{width}|{sr}|{start_sec:.3f}|{dur_sec:.3f}".encode("utf-8", "ignore")).hexdigest()
    disk = _disk_cache_path(h)
    if os.path.exists(disk):
        try:
            arr = np.load(disk)
            _MEM_CACHE[mem_key] = arr
            return arr
        except Exception:
            pass

    cmd = [
        "ffmpeg",
        "-v", "error",
        "-ss", f"{start_sec:.6f}",
        "-t", f"{dur_sec:.6f}",
        "-i", p,
        "-ac", "1",
        "-ar", str(sr),
        "-f", "f32le",
        "-"
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw, _ = proc.communicate(timeout=20)
    except Exception:
        return None

    if not raw:
        return None
    audio = np.frombuffer(raw, dtype=np.float32)
    if audio.size == 0:
        return None

    bucket = max(1, int(audio.size // width))
    n = int(np.ceil(audio.size / bucket))
    peaks = np.empty((n,), dtype=np.float32)
    for i in range(n):
        seg = audio[i*bucket:(i+1)*bucket]
        peaks[i] = float(np.max(np.abs(seg))) if seg.size else 0.0

    mx = float(np.max(peaks)) if peaks.size else 0.0
    if mx > 1e-9:
        peaks /= mx

    try:
        np.save(disk, peaks)
    except Exception:
        pass

    _MEM_CACHE[mem_key] = peaks
    return peaks

