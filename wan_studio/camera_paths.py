from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n


def _look_at_c2w_opencv(pos: np.ndarray, target: np.ndarray, up_world: np.ndarray) -> np.ndarray:
    """
    Build a camera-to-world matrix in an OpenCV-like coordinate system:
      x right, y down, z forward.
    """
    forward = _normalize(target - pos)            # +Z
    right = _normalize(np.cross(up_world, forward))  # +X
    down = _normalize(np.cross(forward, right))      # +Y (down)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = pos
    return c2w


def ensure_4n1(frame_num: int) -> int:
    """Ensure frame_num is 4n+1 (required by Wan/LingBot)."""
    frame_num = int(max(5, frame_num))
    return ((frame_num - 1) // 4) * 4 + 1


@dataclass
class CameraPathParams:
    preset: str = "static"
    fov_deg: float = 50.0
    amount: float = 1.0
    strength: float = 1.0


def intrinsics_from_fov(width: int, height: int, fov_deg: float) -> np.ndarray:
    fov = float(max(5.0, min(175.0, fov_deg)))
    fov_rad = math.radians(fov)
    fx = 0.5 * width / math.tan(0.5 * fov_rad)
    fy = fx
    cx = width * 0.5
    cy = height * 0.5
    return np.array([fx, fy, cx, cy], dtype=np.float32)


def generate_camera_path_opencv(
    *,
    preset: str,
    frame_num: int,
    width: int,
    height: int,
    fov_deg: float = 50.0,
    amount: float = 1.0,
    strength: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    preset = (preset or "static").strip().lower()
    F = ensure_4n1(frame_num)
    amount = float(amount)
    strength = float(strength)

    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up_world = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    base_dist = 2.5
    dist = base_dist * max(0.25, strength)

    poses = np.zeros((F, 4, 4), dtype=np.float32)

    for i in range(F):
        t = 0.0 if F <= 1 else (i / (F - 1))
        t2 = t * t * (3 - 2 * t)  # smoothstep

        if preset in ("static", "lockoff", "tripod"):
            pos = np.array([0.0, 0.0, -dist], dtype=np.float32)

        elif preset in ("dolly_in", "dolly-in", "dolly"):
            z = -dist + (amount * 0.8) * t2
            pos = np.array([0.0, 0.0, z], dtype=np.float32)

        elif preset in ("dolly_out", "dolly-out"):
            z = -dist - (amount * 0.8) * t2
            pos = np.array([0.0, 0.0, z], dtype=np.float32)

        elif preset in ("truck_left", "truck-left"):
            x = -(amount * 0.9) * (t2 - 0.5)
            pos = np.array([x, 0.0, -dist], dtype=np.float32)

        elif preset in ("truck_right", "truck-right"):
            x = +(amount * 0.9) * (t2 - 0.5)
            pos = np.array([x, 0.0, -dist], dtype=np.float32)

        elif preset in ("pan_left", "pan-left"):
            pos = np.array([0.0, 0.0, -dist], dtype=np.float32)
            yaw = -amount * 0.6 * (t2 - 0.5)
            target = np.array([math.sin(yaw) * 0.2, 0.0, math.cos(yaw) * 0.2], dtype=np.float32)

        elif preset in ("pan_right", "pan-right"):
            pos = np.array([0.0, 0.0, -dist], dtype=np.float32)
            yaw = +amount * 0.6 * (t2 - 0.5)
            target = np.array([math.sin(yaw) * 0.2, 0.0, math.cos(yaw) * 0.2], dtype=np.float32)

        elif preset in ("orbit_left", "orbit-left", "orbit"):
            ang = -amount * 0.9 * (t2 - 0.5)
            radius = dist
            pos = np.array([math.sin(ang) * radius, 0.0, -math.cos(ang) * radius], dtype=np.float32)

        elif preset in ("orbit_right", "orbit-right"):
            ang = +amount * 0.9 * (t2 - 0.5)
            radius = dist
            pos = np.array([math.sin(ang) * radius, 0.0, -math.cos(ang) * radius], dtype=np.float32)

        else:
            pos = np.array([0.0, 0.0, -dist], dtype=np.float32)

        poses[i] = _look_at_c2w_opencv(pos, target, up_world)

    K_row = intrinsics_from_fov(width, height, fov_deg)
    intr = np.repeat(K_row[None, :], poses.shape[0], axis=0)
    return poses, intr


def write_action_path(
    action_dir: str,
    *,
    preset: str,
    frame_num: int,
    width: int,
    height: int,
    fov_deg: float = 50.0,
    amount: float = 1.0,
    strength: float = 1.0,
) -> str:
    os.makedirs(action_dir, exist_ok=True)
    F = ensure_4n1(frame_num)
    poses, intr = generate_camera_path_opencv(
        preset=preset, frame_num=F, width=width, height=height, fov_deg=fov_deg, amount=amount, strength=strength
    )
    np.save(os.path.join(action_dir, "poses.npy"), poses)
    np.save(os.path.join(action_dir, "intrinsics.npy"), intr)
    meta = {
        "preset": preset,
        "frame_num": int(F),
        "width": int(width),
        "height": int(height),
        "fov_deg": float(fov_deg),
        "amount": float(amount),
        "strength": float(strength),
    }
    with open(os.path.join(action_dir, "camera_path.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return action_dir
