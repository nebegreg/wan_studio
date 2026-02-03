from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .config import ProjectConfig, SceneSpec
from .cinema_prompts import build_scene_prompts
from .cache_utils import prune_to_quota


@dataclass
class ClipCacheEntry:
    key: str
    frames_dir: str
    last_frame_path: str
    num_frames: int
    fps: int
    created_ts: int
    meta: Dict[str, Any]


def _safe_abspath(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def clip_cache_root(cfg: ProjectConfig) -> str:
    """Stable cache root independent of a particular run folder."""
    root = _safe_abspath(os.path.join(cfg.output_dir, cfg.project_name))
    cache_dirname = getattr(cfg, 'clip_cache_dirname', '.wan_cache') or '.wan_cache'
    cache_root = os.path.join(root, cache_dirname, 'clip_renders')
    os.makedirs(cache_root, exist_ok=True)
    return cache_root


def _fingerprint_image(img: Image.Image) -> str:
    """Cheap, stable fingerprint for a conditioning image."""
    try:
        im = img.convert('RGB').resize((64, 64), Image.BILINEAR)
        b = im.tobytes()
        return hashlib.sha256(b).hexdigest()
    except Exception:
        return 'unknown'


def _hash_file_hint(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{_safe_abspath(path)}|{st.st_size}|{int(st.st_mtime)}"
    except Exception:
        return _safe_abspath(path)


def compute_clip_cache_key(
    cfg: ProjectConfig,
    scene: SceneSpec,
    *,
    effective_backend: str,
    effective_model_id: str,
    effective_mode: str,
    conditioning_image: Optional[Image.Image],
    conditioning_path_hint: Optional[str] = None,
    init_frame_key_hint: Optional[str] = None,
    effective_seed: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Compute a per-clip cache key.

    The key is intended to be stable for the same clip inputs/settings,
    and should change if:
      - prompt/neg changes (including injected characters/location)
      - model/backend/mode changes
      - clip length/fps/size changes
      - conditioning anchor changes (continuity or init frame)
      - generation settings change

    Notes:
      - If `conditioning_image` is provided, we fingerprint pixels.
      - Else if `conditioning_path_hint` is provided, we hash its file hint.
      - Else if `init_frame_key_hint` is provided, we include it.
    """
    prompt, neg = build_scene_prompts(cfg, scene)

    parts: List[str] = []
    # bump when we change what participates in the hash
    parts.append('v=2')
    parts.append('backend=' + str(effective_backend))
    parts.append('model_id=' + str(effective_model_id))
    parts.append('mode=' + str(effective_mode))
    parts.append(f"fps={int(getattr(cfg,'fps',24))}")
    parts.append(f"wh={int(getattr(cfg,'width',0))}x{int(getattr(cfg,'height',0))}")
    parts.append(f"seconds={float(getattr(scene,'seconds',0.0)):.4f}")

    # chunking/stitching settings
    parts.append(f"chunk_seconds={float(getattr(cfg,'chunk_seconds',5.0)):.4f}")
    parts.append(f"overlap_frames={int(getattr(cfg,'overlap_frames',0))}")
    parts.append('blend=' + str(getattr(scene,'blend_mode',None) or getattr(cfg,'blend_mode','crossfade')))
    parts.append('ease=' + str(getattr(cfg,'scene_transition_ease','smoothstep')))

    # generation settings (scene overrides included via SceneSpec)
    parts.append(f"steps={int(scene.num_inference_steps if scene.num_inference_steps is not None else getattr(cfg,'num_inference_steps',0))}")
    parts.append(f"gs={float(scene.guidance_scale if scene.guidance_scale is not None else getattr(cfg,'guidance_scale',0.0)):.4f}")
    parts.append(f"gs2={float(scene.guidance_scale_2 if scene.guidance_scale_2 is not None else getattr(cfg,'guidance_scale_2',0.0)):.4f}")
    parts.append(f"br={float(scene.boundary_ratio if scene.boundary_ratio is not None else getattr(cfg,'boundary_ratio',0.0)):.4f}")

    # seeds
    if effective_seed is not None:
        try:
            parts.append(f"seed={int(effective_seed)}")
        except Exception:
            parts.append(f"seed={str(effective_seed)}")

    # style pack affects prompt injection and suggested settings
    pack_name = (getattr(scene, "style_pack", None) or getattr(cfg, "global_style_pack", None) or "").strip()
    if pack_name and pack_name.lower() in ("(inherit)", "inherit", "none", "(none)"):
        pack_name = ""
    parts.append("style_pack=" + pack_name)

    # LoRAs (order matters; users often tune stacks)
    try:
        loras = getattr(cfg, "loras", None) or []
        lora_sig = []
        for l in loras:
            try:
                if hasattr(l, "enabled") and not bool(getattr(l, "enabled")):
                    continue
            except Exception:
                pass
            lora_sig.append({
                "name": str(getattr(l, "name", "") or ""),
                "weight": float(getattr(l, "weight", 0.0) or 0.0),
                "t2": bool(getattr(l, "t2", False)),
                "repo_id": str(getattr(l, "repo_id", "") or ""),
                "weight_name": str(getattr(l, "weight_name", "") or ""),
                "local_path": str(getattr(l, "local_path", "") or ""),
            })
        parts.append("loras=" + json.dumps(lora_sig, sort_keys=True, ensure_ascii=False))
    except Exception:
        pass

    # runtime/device options that change outputs
    try:
        parts.append("dtype=" + str(getattr(cfg, "dtype", "")))
        parts.append("device_strategy=" + str(getattr(cfg, "device_strategy", "")))
        parts.append("use_unipc=" + str(bool(getattr(cfg, "use_unipc", False))))
        parts.append("flow_shift=" + str(float(getattr(cfg, "flow_shift", 0.0) or 0.0)))
        parts.append("vae_decode_fp32=" + str(bool(getattr(cfg, "vae_decode_fp32", False))))
    except Exception:
        pass

    # prompt
    parts.append('prompt=' + (prompt or ''))
    parts.append('neg=' + (neg or ''))

    # conditioning
    if conditioning_image is not None:
        parts.append('cond_img=' + _fingerprint_image(conditioning_image))
    elif conditioning_path_hint:
        parts.append('cond_path=' + _hash_file_hint(conditioning_path_hint))
    elif init_frame_key_hint:
        parts.append('init_key=' + str(init_frame_key_hint))
    else:
        parts.append('cond=none')

    if extra:
        try:
            parts.append("extra=" + json.dumps(extra, sort_keys=True, ensure_ascii=False))
        except Exception:
            parts.append("extra=" + str(extra))

    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode('utf-8', errors='ignore'))
        h.update(b'\n')
    return h.hexdigest()


def _entry_dir(cache_root: str, key: str) -> str:
    return os.path.join(cache_root, key[:16])


def _meta_path(cache_root: str, key: str) -> str:
    return os.path.join(_entry_dir(cache_root, key), 'meta.json')


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_entry(cache_root: str, key: str) -> Optional[ClipCacheEntry]:
    mp = _meta_path(cache_root, key)
    meta = _load_json(mp)
    if not meta:
        return None
    frames_dir = str(meta.get('frames_dir', ''))
    last_frame = str(meta.get('last_frame_path', ''))
    num_frames = int(meta.get('num_frames', 0) or 0)
    fps = int(meta.get('fps', 0) or 0)
    created_ts = int(meta.get('created_ts', 0) or 0)
    if not frames_dir:
        frames_dir = os.path.join(_entry_dir(cache_root, key), 'frames')
    if not last_frame:
        last_frame = os.path.join(_entry_dir(cache_root, key), 'last.png')
    if not os.path.isdir(frames_dir) or not os.path.exists(last_frame):
        return None
    if num_frames <= 0:
        # infer from files
        try:
            num_frames = len([p for p in os.listdir(frames_dir) if p.endswith('.png')])
        except Exception:
            num_frames = 0
    return ClipCacheEntry(
        key=key,
        frames_dir=frames_dir,
        last_frame_path=last_frame,
        num_frames=num_frames,
        fps=fps,
        created_ts=created_ts,
        meta=meta,
    )


def delete_entry(cache_root: str, key: str) -> bool:
    'Delete a clip cache entry (frames + meta).'
    try:
        ed = _entry_dir(cache_root, key)
        if os.path.isdir(ed):
            import shutil
            shutil.rmtree(ed, ignore_errors=True)
        return True
    except Exception:
        return False


_PRUNE_ONCE = False


def maybe_prune_cache(cfg: ProjectConfig, cache_root: str, *, log=None) -> None:
    'Prune clip cache to cfg.cache_max_gb (best-effort; runs at most once per process).'
    global _PRUNE_ONCE
    if _PRUNE_ONCE:
        return
    try:
        max_gb = float(getattr(cfg, 'cache_max_gb', 0.0) or 0.0)
    except Exception:
        max_gb = 0.0
    if max_gb <= 0:
        return
    try:
        prune_to_quota(cache_root, int(max_gb * 1024**3), log=log, keep_min=1)
        _PRUNE_ONCE = True
    except Exception:
        pass


def save_clip(cache_root: str, key: str, frames: Sequence[np.ndarray], fps: int, extra_meta: Optional[Dict[str, Any]] = None) -> ClipCacheEntry:
    ed = _entry_dir(cache_root, key)
    frames_dir = os.path.join(ed, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    # Write frames
    for i, fr in enumerate(frames, start=1):
        p = os.path.join(frames_dir, f'frame_{i:06d}.png')
        Image.fromarray(fr).save(p, 'PNG')

    # Save last frame
    last_path = os.path.join(ed, 'last.png')
    if frames:
        Image.fromarray(frames[-1]).save(last_path, 'PNG')

    meta: Dict[str, Any] = {
        'key': key,
        'frames_dir': frames_dir,
        'last_frame_path': last_path,
        'num_frames': int(len(frames)),
        'fps': int(fps),
        'created_ts': int(time.time()),
    }
    if extra_meta:
        meta.update(extra_meta)

    _save_json(_meta_path(cache_root, key), meta)
    return ClipCacheEntry(
        key=key,
        frames_dir=frames_dir,
        last_frame_path=last_path,
        num_frames=int(len(frames)),
        fps=int(fps),
        created_ts=int(meta['created_ts']),
        meta=meta,
    )


def load_frames(entry: ClipCacheEntry, limit: Optional[int] = None) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    if not entry.frames_dir or not os.path.isdir(entry.frames_dir):
        return out

    files = [p for p in os.listdir(entry.frames_dir) if p.startswith('frame_') and p.endswith('.png')]
    files.sort()
    if limit is not None:
        files = files[: max(0, int(limit))]

    for fn in files:
        p = os.path.join(entry.frames_dir, fn)
        try:
            out.append(np.array(Image.open(p).convert('RGB'), dtype=np.uint8))
        except Exception:
            pass
    return out


def load_last_frame(entry: ClipCacheEntry) -> Optional[Image.Image]:
    try:
        if entry.last_frame_path and os.path.exists(entry.last_frame_path):
            return Image.open(entry.last_frame_path).convert('RGB')
    except Exception:
        return None
    return None


def find_latest_entry_for_scene(cache_root: str, scene_i: int, scene_label: Optional[str] = None) -> Optional[ClipCacheEntry]:
    """Best-effort lookup: find the newest cache entry matching a scene index (and optionally label).

    Useful when the UI wants to preview a clip but doesn't know the exact cache key
    (e.g. after loading an older project file).
    """
    try:
        scene_i = int(scene_i)
    except Exception:
        return None
    want_label = (scene_label or '').strip()
    best: Optional[ClipCacheEntry] = None
    best_ts = -1
    try:
        if not os.path.isdir(cache_root):
            return None
        for d in os.listdir(cache_root):
            ed = os.path.join(cache_root, d)
            mp = os.path.join(ed, 'meta.json')
            if not os.path.isdir(ed) or not os.path.exists(mp):
                continue
            meta = _load_json(mp) or {}
            try:
                mi = int(meta.get('scene_i', -1))
            except Exception:
                mi = -1
            if mi != scene_i:
                continue
            if want_label:
                ml = str(meta.get('scene_label', '') or '').strip()
                if ml and ml != want_label:
                    # labels differ -> skip (reduces wrong matches when cache is large)
                    continue
            try:
                ts = int(meta.get('created_ts', 0) or 0)
            except Exception:
                ts = 0
            key = str(meta.get('key', '') or '')
            if not key:
                # infer key from folder name (first 16) is not enough for load_entry
                # so keep going.
                continue
            ent = load_entry(cache_root, key)
            if ent is None:
                continue
            if ts > best_ts:
                best = ent
                best_ts = ts
    except Exception:
        return best
    return best
