from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional

def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def encode_png_sequence_to_mp4(frames_dir: str, out_path: str, fps: int, crf: int = 18, preset: str = "slow") -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    ffmpeg = which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg introuvable dans le PATH.")

    cmd = [
        ffmpeg, "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%06d.png"),
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_path,
    ]
    subprocess.run(cmd, check=True)

def encode_prores_master(in_path: str, out_path: str, fps: int, profile: str = "422hq") -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    ffmpeg = which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg introuvable dans le PATH.")
    PROFILE_MAP = {
        "proxy": ("0", "yuv422p10le"),
        "lt": ("1", "yuv422p10le"),
        "422": ("2", "yuv422p10le"),
        "422hq": ("3", "yuv422p10le"),
        "4444": ("4", "yuv444p10le"),
        "4444xq": ("5", "yuv444p10le"),
    }
    prof_key = (profile or "422hq").strip().lower()
    prof_num, pix_fmt = PROFILE_MAP.get(prof_key, ("3", "yuv422p10le"))

    cmd = [
        ffmpeg, "-y",
        "-i", in_path,
        "-r", str(fps),
        "-c:v", "prores_ks",
        "-profile:v", prof_num,
        "-pix_fmt", pix_fmt,
        "-vendor", "apl0",
        "-movflags", "+faststart",
        out_path,
    ]
    subprocess.run(cmd, check=True)

def encode_h265_main10(in_path: str, out_path: str, fps: int, crf: int = 18, preset: str = "slow") -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    ffmpeg = which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg introuvable dans le PATH.")
    cmd = [
        ffmpeg, "-y",
        "-i", in_path,
        "-r", str(fps),
        "-c:v", "libx265",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p10le",
        "-x265-params", "profile=main10",
        "-tag:v", "hvc1",
        "-movflags", "+faststart",
        out_path,
    ]
    subprocess.run(cmd, check=True)
