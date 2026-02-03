from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from typing import Callable, List, Optional, Tuple

import numpy as np

LogCallback = Optional[Callable[[str], None]]


def _log(log: LogCallback, msg: str) -> None:
    try:
        if callable(log):
            log(msg)
    except Exception:
        pass


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def ensure_ffmpeg(log: LogCallback = None) -> Tuple[str, str]:
    ffmpeg = _which("ffmpeg")
    ffprobe = _which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg/ffprobe not found in PATH. Please install ffmpeg.")
    return ffmpeg, ffprobe


def resolve_ckpt_dir(ckpt_dir: str, log: LogCallback = None) -> str:
    ckpt_dir = (ckpt_dir or "").strip()
    if not ckpt_dir:
        raise RuntimeError("LingBot ckpt_dir is empty. Set cfg.lingbot.ckpt_dir.")
    if os.path.isdir(ckpt_dir):
        return os.path.abspath(ckpt_dir)

    # Best-effort HF snapshot_download
    if "/" in ckpt_dir and not os.path.exists(ckpt_dir):
        try:
            from huggingface_hub import snapshot_download  # type: ignore
            cache_root = os.path.abspath("./tools/models/lingbot_world")
            os.makedirs(cache_root, exist_ok=True)
            local_dir = os.path.join(cache_root, ckpt_dir.replace("/", "__"))
            _log(log, f"[LingBot] Downloading checkpoint: {ckpt_dir} -> {local_dir}")
            path = snapshot_download(repo_id=ckpt_dir, local_dir=local_dir, local_dir_use_symlinks=False)
            return os.path.abspath(path)
        except Exception as e:
            raise RuntimeError(f"Checkpoint '{ckpt_dir}' not found and huggingface download failed: {e}")

    return os.path.abspath(ckpt_dir)


def run_lingbot_generate(
    *,
    third_party_dir: str,
    ckpt_dir: str,
    prompt: str,
    image_path: str,
    action_path: str,
    size: str,
    frame_num: int,
    sample_steps: int,
    sample_shift: float,
    guide_scale: float,
    seed: int,
    offload_model: bool,
    t5_cpu: bool,
    ulysses_size: int,
    save_file: str,
    log: LogCallback = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> str:
    ffmpeg, ffprobe = ensure_ffmpeg(log)
    ckpt_local = resolve_ckpt_dir(ckpt_dir, log=log)

    gen_py = os.path.join(third_party_dir, "lingbot_world", "generate.py")
    if not os.path.exists(gen_py):
        raise RuntimeError(f"LingBot generate.py not found: {gen_py}")

    python = sys.executable

    cmd = [
        python, gen_py,
        "--task", "i2v-A14B",
        "--ckpt_dir", ckpt_local,
        "--prompt", prompt,
        "--image", image_path,
        "--action_path", action_path,
        "--size", size,
        "--frame_num", str(int(frame_num)),
        "--sample_steps", str(int(sample_steps)),
        "--sample_shift", str(float(sample_shift)),
        "--sample_guide_scale", str(float(guide_scale)),
        "--base_seed", str(int(seed)),
        "--ulysses_size", str(int(ulysses_size)),
        "--save_file", save_file,
        "--offload_model", str(bool(offload_model)),
    ]
    if t5_cpu:
        cmd.append("--t5_cpu")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(third_party_dir, "lingbot_world"), env.get("PYTHONPATH", "")])

    _log(log, f"[LingBot] cmd: {' '.join(cmd)}")
    if is_cancelled and is_cancelled():
        raise RuntimeError("Cancelled")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if line:
                _log(log, "[LingBot] " + line.rstrip())
            if is_cancelled and is_cancelled():
                proc.kill()
                raise RuntimeError("Cancelled")
    finally:
        rc = proc.wait()

    if rc != 0:
        raise RuntimeError(f"LingBot generation failed (exit {rc}). Check logs above.")
    if not os.path.exists(save_file):
        raise RuntimeError(f"LingBot finished but output file not found: {save_file}")
    return save_file


def ffprobe_video_info(path: str) -> Tuple[int, int, Optional[int]]:
    ffmpeg, ffprobe = ensure_ffmpeg()
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames", "-of", "json", path]
    out = subprocess.check_output(cmd, text=True)
    j = json.loads(out)
    st = (j.get("streams") or [{}])[0]
    w = int(st.get("width") or 0)
    h = int(st.get("height") or 0)
    nb = st.get("nb_frames")
    nb_i = int(nb) if nb and str(nb).isdigit() else None
    return w, h, nb_i


def decode_mp4_to_frames(path: str, log: LogCallback = None) -> List[np.ndarray]:
    ffmpeg, ffprobe = ensure_ffmpeg(log)
    w, h, nb = ffprobe_video_info(path)
    if w <= 0 or h <= 0:
        raise RuntimeError("ffprobe failed to read video dimensions")
    _log(log, f"[LingBot] Decoding mp4 ({w}x{h}) to frames…")
    cmd = [ffmpeg, "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {err.decode('utf-8', errors='ignore')}")
    frame_size = w * h * 3
    if len(raw) < frame_size:
        raise RuntimeError("ffmpeg returned empty output")
    n = len(raw) // frame_size
    raw = raw[: n * frame_size]
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((n, h, w, 3))
    return [arr[i].copy() for i in range(n)]


def resample_frames(frames: List[np.ndarray], target_count: int) -> List[np.ndarray]:
    if not frames:
        return []
    target_count = int(max(1, target_count))
    if len(frames) == target_count:
        return frames
    idxs = np.linspace(0, len(frames) - 1, target_count)
    out: List[np.ndarray] = []
    for t in idxs:
        j = int(round(float(t)))
        j = max(0, min(len(frames) - 1, j))
        out.append(frames[j])
    return out
