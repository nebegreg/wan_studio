from __future__ import annotations

import glob
import os
import shutil
from typing import Callable, Optional, List, Tuple

import numpy as np
from PIL import Image

LogFn = Optional[Callable[[str], None]]


def _thumb_rgb(path: str, size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """Load a tiny RGB thumbnail as float32 in [0, 1]."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize(size, Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def repair_glitch_frames(
    frames_dir: str,
    threshold_factor: float = 10.0,
    neighbor_similarity_factor: float = 0.35,
    max_repairs: int = 200,
    log: LogFn = None,
) -> int:
    """Detect and repair glitch frames in a PNG sequence.

    V24 heuristic fixed "isolated" glitches only. In practice you can get short *bursts*
    of 2–3 corrupted frames. This version repairs:
      - isolated 1-frame glitches (blend neighbors)
      - 2/3-frame glitch bursts (interpolate between stable endpoints)
      - unreadable/corrupted PNGs (replace from nearest valid neighbor)

    The algorithm is conservative: it only edits when endpoints look stable.
    """

    if not frames_dir:
        return 0

    paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if len(paths) < 3:
        return 0

    thumbs: List[np.ndarray] = []
    broken: List[bool] = []
    for p in paths:
        try:
            thumbs.append(_thumb_rgb(p))
            broken.append(False)
        except Exception:
            thumbs.append(np.zeros((64, 64, 3), dtype=np.float32))
            broken.append(True)

    diffs = [_mse(thumbs[i], thumbs[i - 1]) for i in range(1, len(thumbs))]
    med = float(np.median(diffs)) if diffs else 0.0
    if med <= 1e-10:
        med = 1e-10

    thr = float(med) * float(threshold_factor)
    neigh_thr = thr * float(neighbor_similarity_factor)

    backup_dir = os.path.join(frames_dir, "_glitch_backup")
    os.makedirs(backup_dir, exist_ok=True)

    repaired_mask = [False] * len(paths)

    def _backup(i: int):
        try:
            shutil.copy2(paths[i], os.path.join(backup_dir, os.path.basename(paths[i])))
        except Exception:
            pass

    def _save_interp(i: int, a_idx: int, b_idx: int, alpha: float):
        with Image.open(paths[a_idx]) as a_im, Image.open(paths[b_idx]) as b_im:
            a = a_im.convert("RGBA")
            b = b_im.convert("RGBA")
            out = Image.blend(a, b, float(alpha)).convert("RGB")
            out.save(paths[i])
            thumbs[i] = np.asarray(out.resize((64, 64), Image.BILINEAR), dtype=np.float32) / 255.0

    repaired = 0

    # 0) Fix unreadable frames first (copy nearest neighbor)
    for i, is_broken in enumerate(broken):
        if not is_broken:
            continue
        _backup(i)
        src = None
        for j in range(i - 1, -1, -1):
            if not broken[j]:
                src = j
                break
        if src is None:
            for j in range(i + 1, len(paths)):
                if not broken[j]:
                    src = j
                    break
        if src is None:
            continue
        try:
            shutil.copy2(paths[src], paths[i])
            thumbs[i] = thumbs[src].copy()
            repaired_mask[i] = True
            repaired += 1
            if log:
                log(f"[FrameSafety] repaired broken {os.path.basename(paths[i])} by copy from {os.path.basename(paths[src])}")
        except Exception:
            pass
        if repaired >= int(max_repairs):
            return repaired

    def endpoints_similar(a_idx: int, b_idx: int) -> bool:
        try:
            return _mse(thumbs[a_idx], thumbs[b_idx]) < neigh_thr
        except Exception:
            return False

    # 1) Burst repair (2–3 frames)
    for burst_len in (3, 2):
        i = 1
        while i < len(paths) - (burst_len + 1):
            if repaired >= int(max_repairs):
                break
            a = i - 1
            b = i + burst_len
            if any(repaired_mask[j] for j in range(i, i + burst_len)):
                i += 1
                continue
            if broken[a] or broken[b]:
                i += 1
                continue
            if not endpoints_similar(a, b):
                i += 1
                continue

            ok = True
            for j in range(i, i + burst_len):
                if _mse(thumbs[j], thumbs[a]) <= thr or _mse(thumbs[j], thumbs[b]) <= thr:
                    ok = False
                    break
            if not ok:
                i += 1
                continue

            for k, j in enumerate(range(i, i + burst_len), start=1):
                _backup(j)
                alpha = k / float(burst_len + 1)
                try:
                    _save_interp(j, a, b, alpha)
                    repaired_mask[j] = True
                    repaired += 1
                    if log:
                        log(f"[FrameSafety] repaired burst {os.path.basename(paths[j])} alpha={alpha:.2f}")
                except Exception as e:
                    if log:
                        log(f"[FrameSafety] burst repair failed for {os.path.basename(paths[j])}: {e}")
                if repaired >= int(max_repairs):
                    break
            i += burst_len
        if repaired >= int(max_repairs):
            break

    # 2) Isolated repair
    for i in range(1, len(paths) - 1):
        if repaired >= int(max_repairs):
            break
        if repaired_mask[i]:
            continue

        d_prev = _mse(thumbs[i], thumbs[i - 1])
        d_next = _mse(thumbs[i], thumbs[i + 1])
        d_pn = _mse(thumbs[i - 1], thumbs[i + 1])

        if d_prev > thr and d_next > thr and d_pn < neigh_thr:
            _backup(i)
            try:
                with Image.open(paths[i - 1]) as a_im, Image.open(paths[i + 1]) as b_im:
                    a = a_im.convert("RGBA")
                    b = b_im.convert("RGBA")
                    blended = Image.blend(a, b, 0.5).convert("RGB")
                    blended.save(paths[i])
                    thumbs[i] = np.asarray(blended.resize((64, 64), Image.BILINEAR), dtype=np.float32) / 255.0
                repaired_mask[i] = True
                repaired += 1
                if log:
                    log(f"[FrameSafety] repaired isolated {os.path.basename(paths[i])} (d_prev={d_prev:.6f} d_next={d_next:.6f} med={med:.6f})")
            except Exception as e:
                if log:
                    log(f"[FrameSafety] failed to repair {os.path.basename(paths[i])}: {e}")

    if log is not None and repaired > 0:
        try:
            log(f"[FrameSafety] total repaired: {repaired}")
        except Exception:
            pass

    return repaired
