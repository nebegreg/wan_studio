from __future__ import annotations
from typing import List, Literal, Callable
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

BlendMode = Literal["cut", "flow", "crossfade"]
EaseMode = Literal["linear", "smoothstep", "ease_in_out"]

def _to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    frame = np.clip(frame, 0, 255)
    return frame.astype(np.uint8)

def _ease_fn(mode: EaseMode) -> Callable[[float], float]:
    if mode == "linear":
        return lambda t: t
    if mode == "smoothstep":
        return lambda t: t * t * (3.0 - 2.0 * t)
    # ease_in_out (cubic)
    return lambda t: 4*t*t*t if t < 0.5 else 1 - (-2*t + 2)**3 / 2

def crossfade_overlap(prev_tail: List[np.ndarray], next_head: List[np.ndarray], ease: EaseMode = "linear") -> List[np.ndarray]:
    assert len(prev_tail) == len(next_head)
    n = len(prev_tail)
    easef = _ease_fn(ease)
    out = []
    for i in range(n):
        t = (i + 1) / (n + 1)
        a = float(easef(t))
        p = _to_uint8(prev_tail[i]).astype(np.float32)
        q = _to_uint8(next_head[i]).astype(np.float32)
        blended = (1.0 - a) * p + a * q
        out.append(np.clip(blended, 0, 255).astype(np.uint8))
    return out

def flow_blend_overlap(prev_tail: List[np.ndarray], next_head: List[np.ndarray], ease: EaseMode = "linear") -> List[np.ndarray]:
    if cv2 is None:
        return crossfade_overlap(prev_tail, next_head, ease=ease)

    assert len(prev_tail) == len(next_head)
    n = len(prev_tail)
    easef = _ease_fn(ease)
    out = []

    for i in range(n):
        t = (i + 1) / (n + 1)
        a = float(easef(t))
        prev = _to_uint8(prev_tail[i])
        nxt = _to_uint8(next_head[i])

        prev_g = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        nxt_g = cv2.cvtColor(nxt, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            nxt_g, prev_g, None,
            pyr_scale=0.5, levels=3, winsize=21,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        h, w = prev_g.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        warped_nxt = cv2.remap(nxt, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        p = prev.astype(np.float32)
        q = warped_nxt.astype(np.float32)
        blended = (1.0 - a) * p + a * q
        out.append(np.clip(blended, 0, 255).astype(np.uint8))

    return out

def blend_overlap(prev_tail: List[np.ndarray], next_head: List[np.ndarray], mode: BlendMode, ease: EaseMode = "linear") -> List[np.ndarray]:
    """Blend `prev_tail` into `next_head` for overlap stitching.

    - cut: hard cut (returns next_head frames)
    - crossfade: alpha blend
    - flow: optical-flow warp blend (falls back to crossfade if cv2 missing)
    """
    if mode == "cut":
        return list(next_head)
    return flow_blend_overlap(prev_tail, next_head, ease=ease) if mode == "flow" else crossfade_overlap(prev_tail, next_head, ease=ease)
