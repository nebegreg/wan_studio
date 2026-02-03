from __future__ import annotations

import os
import warnings
warnings.filterwarnings('ignore', message=r'No LoRA keys associated to .*', category=UserWarning)
import math
import time
import shutil
import glob
from dataclasses import replace
from typing import Callable, Optional, List, Dict, Any, Tuple

import numpy as np
from PIL import Image

from .config import ProjectConfig, SceneSpec, LoraSpec
from .safety import apply_cinema_safe_overrides
from .stitch import blend_overlap
from .encode import encode_png_sequence_to_mp4, encode_prores_master, encode_h265_main10
from .postprocess import PostProcessConfig, postprocess_pipeline
from .report import export_shotlist_csv, build_thumbnails, create_html_report
from .utils import safe_makedirs, env_report, write_json, read_json
from .style_packs import get_pack, apply_pack
from .clip_cache import (
    clip_cache_root,
    compute_clip_cache_key,
    load_entry as _clip_cache_load_entry,
    load_frames as _clip_cache_load_frames,
    load_last_frame as _clip_cache_load_last_frame,
    save_clip as _clip_cache_save_clip,
    maybe_prune_cache,
    ClipCacheEntry,
)

from .resource_manager import get_resource_manager, cuda_cleanup, cuda_mem_info

import torch
try:
    from huggingface_hub import hf_hub_download  # type: ignore
except Exception:
    hf_hub_download = None

try:
    from diffusers import WanPipeline, AutoencoderKLWan
except Exception:
    WanPipeline = None
    AutoencoderKLWan = None
try:
    from diffusers import WanImageToVideoPipeline
except Exception:
    WanImageToVideoPipeline = None

try:
    from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline
except Exception:
    CogVideoXPipeline = None
    CogVideoXImageToVideoPipeline = None

try:
    from diffusers import LTXPipeline, LTXImageToVideoPipeline
except Exception:
    LTXPipeline = None
    LTXImageToVideoPipeline = None


# LTX-2 (audio+video) pipelines landed in diffusers >= v0.36.0
try:
    from diffusers import LTX2Pipeline  # T2V (returns video + audio)
except Exception:
    LTX2Pipeline = None

# I2V pipeline import path may vary across diffusers versions.
try:
    from diffusers import LTX2ImageToVideoPipeline
except Exception:
    try:
        from diffusers.pipelines.ltx2 import LTX2ImageToVideoPipeline  # type: ignore
    except Exception:
        LTX2ImageToVideoPipeline = None

try:
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
except Exception:
    UniPCMultistepScheduler = None

try:
    from transformers import CLIPVisionModel
except Exception:
    CLIPVisionModel = None

ProgressCallback = Callable[[float, str], None]
LogCallback = Callable[[str], None]
CancelCheck = Callable[[], bool]

# Optional per-clip callbacks (used by the UI to update the timeline during a full render).
ClipStateCallback = Callable[[int, str, bool], None]      # (scene_idx, state, cache_hit)
ClipProgressCallback = Callable[[int, float, str], None]  # (scene_idx, p01, message)
ClipAssetCallback = Callable[[int, str, str], None]        # (scene_idx, kind, path)


def _ensure_cache_preview_mp4(entry: ClipCacheEntry, cfg: ProjectConfig, log: Optional[LogCallback] = None) -> Optional[str]:
    """Create/reuse an mp4 preview in the clip cache entry (best-effort)."""
    try:
        entry_dir = os.path.dirname(str(entry.frames_dir or '').rstrip('/'))
        if not entry_dir:
            return None
        out_mp4 = os.path.join(entry_dir, 'preview.mp4')
        if os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 1024:
            return out_mp4
        crf = int(getattr(getattr(cfg, 'studio', None), 'clip_preview_crf', 23) or 23)
        preset = str(getattr(getattr(cfg, 'studio', None), 'clip_preview_preset', 'veryfast') or 'veryfast')
        fps = int(getattr(entry, 'fps', 0) or getattr(cfg, 'fps', 24) or 24)
        try:
            if log is not None:
                log(f"[ClipPreview] encoding preview.mp4 (crf={crf}, preset={preset}) → {out_mp4}")
        except Exception:
            pass
        encode_png_sequence_to_mp4(str(entry.frames_dir), out_mp4, fps=fps, crf=crf, preset=preset)
        return out_mp4 if os.path.exists(out_mp4) else None
    except Exception as e:
        try:
            if log is not None:
                log(f"[ClipPreview] skipped: {e}")
        except Exception:
            pass
        return None

def _ensure_form_mk1(n: int, m: int) -> int:
    """Return nearest (m*k + 1) >= n."""
    if n <= 1:
        return 1
    k = (n - 1 + (m - 1)) // m
    return m * k + 1

def _ensure_form_4k1(n: int) -> int:
    return _ensure_form_mk1(n, 4)

def _ensure_form_8k1(n: int) -> int:
    return _ensure_form_mk1(n, 8)

def seconds_to_num_frames(seconds: float, fps: int, backend: str = "wan") -> int:
    """Convert duration to a frame count (including first frame), with backend constraints."""
    target = int(round(float(seconds) * int(fps))) + 1
    if backend in ("ltx", "ltx2"):
        return _ensure_form_8k1(target)
    # CogVideoX varies; keep 4k+1 as safe default unless pipeline says otherwise
    return _ensure_form_4k1(target)

def normalize_hw_for_backend(pipe, width: int, height: int, backend: str = "wan") -> Tuple[int, int]:
    """Normalize (W,H) to satisfy the backend/pipeline VAE constraints.

    - wan: infer mod from pipeline when possible (fallback 16)
    - cogvideox: commonly requires multiple of 16
    - ltx: commonly requires multiple of 32
    """
    width = int(width); height = int(height)
    if backend in ("ltx", "ltx2"):
        mod_value = 32
    elif backend == "cogvideox":
        mod_value = 16
    else:
        try:
            mod_value = int(getattr(pipe, "vae_scale_factor_spatial")) * int(pipe.transformer.config.patch_size[1])
        except Exception:
            mod_value = 16
    h = (height // mod_value) * mod_value
    w = (width // mod_value) * mod_value
    return max(mod_value, w), max(mod_value, h)

# Backward-compatible name
normalize_hw_for_vae = normalize_hw_for_backend

def resize_image_for_pipe(img: Image.Image, pipe, target_area: int) -> Tuple[Image.Image, int, int]:
    aspect = img.height / img.width
    try:
        mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    except Exception:
        mod_value = 16
    height = int(round(math.sqrt(target_area * aspect)))
    width = int(round(math.sqrt(target_area / aspect)))
    height = (height // mod_value) * mod_value
    width = (width // mod_value) * mod_value
    img = img.resize((width, height), Image.LANCZOS)
    return img, height, width


def resize_image_for_backend(img: Image.Image, pipe, target_w: int, target_h: int, backend: str = 'wan') -> Tuple[Image.Image, int, int]:
    """Resize/letterbox-free image to backend constraints close to (target_w,target_h)."""
    target_w = int(target_w); target_h = int(target_h)
    # Keep aspect by fitting inside target
    aspect = img.height / max(1, img.width)
    # Start with target_w, compute h
    w = target_w
    h = int(round(w * aspect))
    if h > target_h:
        h = target_h
        w = int(round(h / max(1e-6, aspect)))
    w, h = normalize_hw_for_backend(pipe, w, h, backend=backend)
    img2 = img.resize((w, h), Image.LANCZOS)
    return img2, h, w

def _list_frame_indices(frames_dir: str) -> List[int]:
    files = glob.glob(os.path.join(frames_dir, "frame_*.png"))
    idxs: List[int] = []
    for p in files:
        bn = os.path.basename(p)
        m = bn.replace("frame_", "").replace(".png", "")
        try:
            idxs.append(int(m))
        except Exception:
            pass
    return sorted(idxs)

def _read_tail_frames(frames_dir: str, count: int) -> List[np.ndarray]:
    idxs = _list_frame_indices(frames_dir)
    if not idxs:
        return []
    tail = idxs[-count:]
    out: List[np.ndarray] = []
    for i in tail:
        p = os.path.join(frames_dir, f"frame_{i:06d}.png")
        if os.path.exists(p):
            out.append(np.array(Image.open(p).convert("RGB"), dtype=np.uint8))
    return out


def _save_ui_preview_png(frame: np.ndarray, out_path: str, *, max_w: int = 512) -> Optional[str]:
    """Save a small preview PNG for the UI (best-effort).

    Returns out_path on success, None on failure.
    """
    try:
        im = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
        if max_w and im.width > int(max_w):
            w = int(max_w)
            h = max(1, int(round(im.height * (w / float(im.width)))))
            im = im.resize((w, h), Image.LANCZOS)
        safe_makedirs(os.path.dirname(out_path))
        im.save(out_path, "PNG", optimize=True)
        return out_path
    except Exception:
        return None

class FrameWriter:
    def __init__(self, frames_dir: str, max_overlap: int, default_overlap: int, default_blend: str, default_ease: str, log: Optional[LogCallback] = None):
        self.frames_dir = frames_dir
        self.max_overlap = max(0, int(max_overlap))
        self.default_overlap = max(0, int(default_overlap))
        self.default_blend = str(default_blend)
        self.default_ease = str(default_ease)
        self.log = log or (lambda s: None)
        safe_makedirs(self.frames_dir)

        self.frame_index = 1
        self._tail: List[np.ndarray] = []
    def restore_from_existing(self) -> int:
        idxs = _list_frame_indices(self.frames_dir)
        if not idxs:
            return 0
        last = idxs[-1]
        self.frame_index = last + 1
        self._tail = _read_tail_frames(self.frames_dir, self.max_overlap)
        return last

    def _write_frame(self, frame: np.ndarray) -> None:
        p = os.path.join(self.frames_dir, f"frame_{self.frame_index:06d}.png")
        Image.fromarray(frame).save(p, "PNG")
        self.frame_index += 1

    def _overwrite_frame(self, index: int, frame: np.ndarray) -> None:
        """Overwrite an already-written frame on disk.

        Important: overlap stitching must *not* append duplicate frames.
        We rewrite the last `overlap` frames instead, then append only the
        truly new frames. This prevents visible stutter/jitter at chunk
        boundaries.
        """
        if index <= 0:
            return
        p = os.path.join(self.frames_dir, f"frame_{index:06d}.png")
        Image.fromarray(frame).save(p, "PNG")

    def _tail_last(self, ov: int) -> List[np.ndarray]:
        if ov <= 0:
            return []
        return self._tail[-ov:] if len(self._tail) >= ov else list(self._tail)

    def _remember_tail(self, frames: List[np.ndarray]) -> None:
        if self.max_overlap <= 0:
            self._tail = []
            return
        self._tail = (self._tail + frames)[-self.max_overlap:]

    def push_segment(self, frames: List[np.ndarray], overlap: Optional[int] = None, blend: Optional[str] = None, ease: Optional[str] = None) -> None:
        if not frames:
            return

        ov = self.default_overlap if overlap is None else max(0, int(overlap))
        blend_mode = self.default_blend if blend is None else str(blend)
        ease_mode = self.default_ease if ease is None else str(ease)

        if ov == 0 or not self._tail:
            for fr in frames:
                self._write_frame(fr)
            self._remember_tail(frames)
            return

        tail = self._tail_last(ov)
        if len(tail) < ov:
            for fr in frames:
                self._write_frame(fr)
            self._remember_tail(frames)
            return

        # Stitching MUST NOT append overlap frames; instead we overwrite the
        # last ov frames already written, then append only the non-overlap tail.
        # This fixes visible stutter and duplicated motion at chunk boundaries.

        # Where to start overwriting in the already-written sequence.
        overwrite_start = self.frame_index - ov
        if overwrite_start <= 0:
            # Safety fallback (shouldn't happen): just append.
            for fr in frames:
                self._write_frame(fr)
            self._remember_tail(frames)
            return

        if len(frames) <= ov:
            head = frames
            tail2 = tail[-len(head):]
            blended = blend_overlap(tail2, head, mode=blend_mode, ease=ease_mode)
            for i, fr in enumerate(blended):
                self._overwrite_frame(overwrite_start + i, fr)
            # No new frames appended.
            self._remember_tail(frames)
            return

        head = frames[:ov]
        blended = blend_overlap(tail, head, mode=blend_mode, ease=ease_mode)
        for i, fr in enumerate(blended):
            self._overwrite_frame(overwrite_start + i, fr)

        # Append only the truly new frames
        middle = frames[ov:]
        for fr in middle:
            self._write_frame(fr)

        self._remember_tail(frames)


class ClipAssembler:
    """In-memory per-clip frame assembler.

    We use this to build a clean clip cache (without montage boundary blending).
    It mirrors FrameWriter.push_segment, but keeps frames in RAM.
    """

    def __init__(self, max_overlap: int, default_overlap: int, default_blend: str, default_ease: str):
        self.max_overlap = max(0, int(max_overlap))
        self.default_overlap = max(0, int(default_overlap))
        self.default_blend = str(default_blend)
        self.default_ease = str(default_ease)
        self.frames: List[np.ndarray] = []
        self._tail: List[np.ndarray] = []

    def _tail_last(self, ov: int) -> List[np.ndarray]:
        if ov <= 0:
            return []
        return self._tail[-ov:] if len(self._tail) >= ov else list(self._tail)

    def _remember_tail(self, frames: List[np.ndarray]) -> None:
        if self.max_overlap <= 0:
            self._tail = []
            return
        self._tail = (self._tail + frames)[-self.max_overlap:]

    def push_segment(self, frames: List[np.ndarray], overlap: Optional[int] = None, blend: Optional[str] = None, ease: Optional[str] = None) -> None:
        if not frames:
            return

        ov = self.default_overlap if overlap is None else max(0, int(overlap))
        blend_mode = self.default_blend if blend is None else str(blend)
        ease_mode = self.default_ease if ease is None else str(ease)

        if ov == 0 or not self._tail:
            self.frames.extend(frames)
            self._remember_tail(frames)
            return

        tail = self._tail_last(ov)
        if len(tail) < ov:
            self.frames.extend(frames)
            self._remember_tail(frames)
            return

        # Replace the last ov frames with blended overlap
        head = frames[:ov] if len(frames) >= ov else frames
        tail2 = tail[-len(head):]
        blended = blend_overlap(tail2, head, mode=blend_mode, ease=ease_mode)
        if blended:
            self.frames[-len(blended):] = blended

        # Append only non-overlap remainder
        if len(frames) > ov:
            self.frames.extend(frames[ov:])

        self._remember_tail(frames)

class WanEngine:



    def _to_uint8(self, fr):
        """Convert a frame array/tensor to uint8 HWC.

        Handles common model output ranges:
          - float in [0, 1]  -> *255
          - float in [-1, 1] -> (x+1)*127.5
          - float/int in [0, 255] -> clip
        """
        import numpy as np
        fr = np.asarray(fr)
        if fr.dtype == np.uint8:
            return fr
        try:
            if np.issubdtype(fr.dtype, np.floating):
                mn = float(np.nanmin(fr))
                mx = float(np.nanmax(fr))
                if mx <= 1.01 and mn >= -0.01:
                    fr = fr * 255.0
                elif mx <= 1.01 and mn >= -1.01:
                    fr = (fr + 1.0) * 127.5
        except Exception:
            pass
        fr = np.clip(fr, 0, 255).astype(np.uint8)
        return fr

    def unload(self, log=None):
        """Free pipeline and clear CUDA cache so user can retry without restarting."""
        try:
            # Unregister via resource manager first (drops references deterministically).
            try:
                # Only drop the tracked resource entry; the actual objects are
                # cleared below.
                self._rm.unregister("i2v_pipe")
            except Exception:
                pass
            self._unload_i2v_internal()
        except Exception:
            pass
        try:
            self.model_id = None
            self.backend = None
            self.mode = None
        except Exception:
            pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        try:
            cuda_cleanup(aggressive=False)
        except Exception:
            pass
        if log is not None:
            try:
                self._log(log, "Pipeline unloaded / CUDA cache cleared.")
            except Exception:
                pass

    def unload_all(self) -> None:
        """Unload *all* heavy resources and clear CUDA caches.

        Intended for a UI "panic button" when VRAM is fragmented or multiple
        pipelines were loaded during a session.
        """
        # 1) Unload init pipelines first (Flux/SDXL/etc.)
        try:
            self._rm.unload(kind="init", aggressive=True)
        except Exception:
            pass

        # 2) Unload I2V pipeline
        try:
            # Drop tracked entry (if any) then drop internal refs.
            try:
                self._rm.unregister("i2v_pipe")
            except Exception:
                pass
            self._unload_i2v_internal()
        except Exception:
            pass

        # 3) Unload any remaining tracked resources
        try:
            self._rm.unload(aggressive=True)
        except Exception:
            pass

        # 4) Final cleanup
        try:
            self.model_id = None
            self.backend = None
            self.mode = None
        except Exception:
            pass
        try:
            cuda_cleanup(aggressive=True)
        except Exception:
            pass

    def _unload_i2v_internal(self) -> None:
        """Drop I2V pipeline refs without calling ResourceManager (no recursion)."""
        try:
            self.pipe = None
        except Exception:
            pass
        try:
            self._scheduler_use_unipc = None
            self._scheduler_flow_shift = None
        except Exception:
            pass
        try:
            self.model_id = None
            self.backend = None
            self.mode = None
        except Exception:
            pass
        try:
            import gc as _gc
            _gc.collect()
        except Exception:
            pass
        try:
            cuda_cleanup(aggressive=True)
        except Exception:
            pass

    def _normalize_pipe_frames(self, out, log=None):
        """Normalize diffusers pipeline output frames to a List[np.ndarray] of shape (H,W,3) uint8."""
        import numpy as np
        try:
            import torch
            if isinstance(out, torch.Tensor):
                out = out.detach().cpu().numpy()
        except Exception:
            pass

        # Some pipelines return list per batch
        if isinstance(out, (list, tuple)):
            # If batch list
            if len(out) == 0:
                return []
            # If elements are tensors/arrays, take first batch
            if len(out) == 1:
                out0 = out[0]
            else:
                out0 = out[0]
            out = out0
            try:
                import torch
                if isinstance(out, torch.Tensor):
                    out = out.detach().cpu().numpy()
            except Exception:
                pass

        if isinstance(out, np.ndarray):
            arr = out
            # Common shapes:
            # (F,H,W,C), (B,F,H,W,C), (H,W,C), (B,H,W,C)
            if arr.ndim == 5:
                # (B,F,H,W,C) -> take batch 0
                arr = arr[0]
            if arr.ndim == 4 and arr.shape[-1] == 3:
                # assume (F,H,W,C)
                frames = [arr[i] for i in range(arr.shape[0])]
            elif arr.ndim == 4 and arr.shape[1] == 3 and arr.shape[-1] != 3:
                # maybe (F,3,H,W) -> transpose
                frames = [np.transpose(arr[i], (1,2,0)) for i in range(arr.shape[0])]
            elif arr.ndim == 3:
                # single frame
                frames = [arr]
            else:
                # try squeeze then interpret
                arr2 = np.squeeze(arr)
                if arr2.ndim == 3 and arr2.shape[-1] == 3:
                    frames = [arr2]
                elif arr2.ndim == 4 and arr2.shape[-1] == 3:
                    frames = [arr2[i] for i in range(arr2.shape[0])]
                else:
                    raise TypeError(f"Unsupported frames array shape: {getattr(out,'shape',None)}")
        else:
            # iterable of frames
            frames = list(out)

        fixed = []
        for fr in frames:
            try:
                import torch
                if isinstance(fr, torch.Tensor):
                    fr = fr.detach().cpu().numpy()
            except Exception:
                pass
            fr = np.asarray(fr)
            # squeeze accidental batch dims
            if fr.ndim == 4 and fr.shape[-1] == 3 and fr.shape[0] == 1:
                fr = fr[0]
            if fr.ndim != 3:
                fr = np.squeeze(fr)
            if fr.ndim == 3 and fr.shape[-1] == 3:
                pass
            elif fr.ndim == 3 and fr.shape[0] == 3 and fr.shape[-1] != 3:
                # CHW -> HWC
                fr = np.transpose(fr, (1,2,0))
            else:
                raise TypeError(f"Unsupported frame shape: {fr.shape}")
            if fr.dtype != np.uint8:
                fr = self._to_uint8(fr)
            fixed.append(fr)
        return fixed

    def _last_frame_image(self, frames, log=None):
        """Return PIL.Image of the last frame from various frame container formats."""
        from PIL import Image
        import numpy as np
        if not frames:
            raise ValueError("No frames produced.")
        fr = frames[-1]
        fr = np.asarray(fr)
        if fr.ndim == 4 and fr.shape[-1] == 3:
            # likely (F,H,W,C) -> take last
            fr = fr[-1]
        fr = np.squeeze(fr)
        if fr.ndim != 3 or fr.shape[-1] != 3:
            raise TypeError(f"Cannot convert last frame to image, shape={getattr(fr,'shape',None)}")
        if fr.dtype != np.uint8:
            fr = self._to_uint8(fr)
        return Image.fromarray(fr)

    def _patch_sequential_offload_meta_inputs(self, pipe, log=None):
        """Workaround for 'Cannot copy out of meta tensor; no data!' after offload.

        Some Diffusers video pipelines (notably Wan I2V in some versions) derive the
        prompt-encoding 'device' from a module's *current* parameter device. Under
        accelerate sequential CPU offload, that module can temporarily sit on the
        `meta` device. The pipeline then moves *inputs* to `meta`, which later crashes
        when Accelerate tries to send them to the execution device.

        Fix: monkey-patch pipe.encode_prompt so that if it receives a `device` equal to
        `meta`, we override it with a real device (cuda if available else cpu).
        """
        try:
            import torch, types
        except Exception:
            return

        if getattr(pipe, "_ws_patch_meta_encode_prompt", False):
            return
        if not hasattr(pipe, "encode_prompt"):
            return

        # Only patch when we detect 'meta' anywhere on the text encoder, otherwise leave it alone.
        try:
            te = getattr(pipe, "text_encoder", None)
            if te is not None:
                any_meta = False
                for p in te.parameters():
                    if getattr(p, "is_meta", False) or (hasattr(p, "device") and p.device.type == "meta"):
                        any_meta = True
                        break
                if not any_meta:
                    return
        except Exception:
            # If we cannot inspect, still patch (safe).
            pass

        orig = pipe.encode_prompt  # bound method
        safe_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        def _wrapped(self, *args, **kwargs):
            # device may be passed positionally or as kw.
            try:
                if "device" in kwargs:
                    dev = kwargs.get("device")
                    if isinstance(dev, torch.device) and dev.type == "meta":
                        kwargs["device"] = safe_device
                elif len(args) >= 2:
                    dev = args[1]
                    if isinstance(dev, torch.device) and dev.type == "meta":
                        args = list(args)
                        args[1] = safe_device
                        args = tuple(args)
            except Exception:
                pass
            return orig(*args, **kwargs)

        pipe.encode_prompt = types.MethodType(_wrapped, pipe)
        setattr(pipe, "_ws_patch_meta_encode_prompt", True)
        if log is not None:
            try:
                self._log(log, "[Wan] Patched encode_prompt: device meta→cuda to avoid meta-tensor crash under sequential offload.")
            except Exception:
                pass

    def _apply_device_strategy(self, pipe, cfg, device, log=None):
        """Apply a VRAM-safe device strategy.
        Priority:
          - sequential_cpu_offload (lowest VRAM, slowest)
          - model_cpu_offload
          - full .to(cuda) (fastest)
        If full .to(cuda) OOMs, automatically fall back to sequential offload.
        """
        import torch

        def _pipe_dtype(p):
            """Best-effort dtype detection.

            Diffusers pipelines don't consistently expose `dtype`. Some use
            `torch_dtype`, and some only have sub-modules with dtypes.
            We use this to avoid ever moving fp16 pipelines to CPU, which can
            trigger warnings and (worse) runtime failures.
            """
            dt = getattr(p, "dtype", None) or getattr(p, "torch_dtype", None)
            if dt is not None:
                return dt
            for sub in ("transformer", "unet", "vae", "text_encoder", "text_encoder_2"):
                m = getattr(p, sub, None)
                if m is None:
                    continue
                dt2 = getattr(m, "dtype", None)
                if dt2 is not None:
                    return dt2
                try:
                    prm = next(m.parameters(), None)
                    if prm is not None:
                        return prm.dtype
                except Exception:
                    pass
            return None

        pipe_dtype = _pipe_dtype(pipe)

        # Heuristic guard: if the GPU is already heavily used, sequential offload is safer.
        # (Prevents mid-render OOMs and avoids accelerate hook corruption when switching strategies.)
        force_seq = False
        try:
            import torch
            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info()
                free_gb = float(free_b) / (1024**3)
                total_gb = float(total_b) / (1024**3)
                # If less than ~7GB free, model+activations frequently OOM for large video models.
                if free_gb < 7.0:
                    force_seq = True
                    if log:
                        self._log(log, f"[VRAM] {free_gb:.1f}GB free / {total_gb:.1f}GB total → forcing sequential CPU offload for stability.")
        except Exception:
            pass

        # If we will use offload hooks, DO NOT force a pre-move to CPU.
        # (It is not needed, and can be harmful for fp16 pipelines when dtype is unknown.)
        want_offload = bool(force_seq or getattr(cfg, "enable_sequential_cpu_offload", False) or getattr(cfg, "enable_model_cpu_offload", False))
        if not want_offload:
            # Start from CPU to avoid partial CUDA allocations before strategy choice.
            # Never move fp16 pipelines to CPU.
            try:
                # If dtype is unknown, assume it *might* be fp16 and avoid CPU moves.
                if pipe_dtype is None or pipe_dtype == torch.float16:
                    pass
                else:
                    pipe.to("cpu")
            except Exception:
                pass

        if force_seq or getattr(cfg, "enable_sequential_cpu_offload", False):
            try:
                pipe.enable_sequential_cpu_offload()
                if log: self._log(log, "Sequential CPU offload activé.")
                # Offload can temporarily place some modules on the `meta` device.
                # Patch known pipelines to avoid moving *inputs* to meta.
                self._patch_sequential_offload_meta_inputs(pipe, log)
                return
            except Exception as e:
                if log: self._log(log, f"Sequential CPU offload non supporté ({e})")

        if getattr(cfg, "enable_model_cpu_offload", False):
            try:
                pipe.enable_model_cpu_offload()
                if log: self._log(log, "CPU offload activé (model).")
                return
            except Exception as e:
                if log: self._log(log, f"CPU offload non supporté ({e})")
                # Some diffusers/accelerate combos can leave modules on the meta device.
                # Instead of falling through to pipe.to(cuda) (which will crash),
                # try sequential offload as a safe fallback.
                try:
                    if "meta tensor" in str(e).lower() and hasattr(pipe, "enable_sequential_cpu_offload"):
                        pipe.enable_sequential_cpu_offload()
                        if log: self._log(log, "Sequential CPU offload activé (fallback après meta-tensor).")
                        self._patch_sequential_offload_meta_inputs(pipe, log)
                        return
                except Exception:
                    pass

        # Full to(device) with fallback
        try:
            pipe.to(device)
            if log: self._log(log, f"Pipeline sur {device}.")
            return
        except torch.OutOfMemoryError as e:
            if log: self._log(log, f"OOM during pipe.to({device}). Fallback to sequential CPU offload.")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            # No need to move pipeline to CPU here; sequential offload will handle placement.
            try:
                pipe.enable_sequential_cpu_offload()
                if log: self._log(log, "Sequential CPU offload activé (fallback).")
                self._patch_sequential_offload_meta_inputs(pipe, log)
                return
            except Exception as e2:
                if log: self._log(log, f"Fallback sequential offload failed: {e2}")
                raise e


    def _filter_pipe_kwargs(self, pipe, call_kwargs: dict, log=None) -> dict:
        """Filter kwargs to match the current pipeline __call__ signature.
    
        Diffusers pipeline signatures can change between versions/models. We drop
        unsupported keys instead of crashing, and we de-dup compat logs so longform
        renders don't spam the console.
        """
        try:
            import inspect
            sig = inspect.signature(pipe.__call__)
            # If pipeline accepts **kwargs, nothing to filter.
            for p in sig.parameters.values():
                if p.kind == inspect.Parameter.VAR_KEYWORD:
                    return call_kwargs
            allowed = set(sig.parameters.keys())
        except Exception:
            return call_kwargs
    
        # Ensure compat log state exists
        if not hasattr(self, "_compat_logged"):
            self._compat_logged = set()
    
        # Special-case: boundary_ratio is often a config attribute, not a __call__ kwarg.
        if "boundary_ratio" in call_kwargs and "boundary_ratio" not in allowed:
            try:
                if hasattr(getattr(pipe, "config", None), "boundary_ratio"):
                    setattr(pipe.config, "boundary_ratio", float(call_kwargs["boundary_ratio"]))
            except Exception:
                pass

        # Drop unsupported keys
        dropped = [k for k in list(call_kwargs.keys()) if k not in allowed]
        if dropped:
            for k in dropped:
                call_kwargs.pop(k, None)
            key = ("dropped", tuple(sorted(dropped)))
            if key not in self._compat_logged:
                self._compat_logged.add(key)
                if log is not None:
                    try:
                        self._log(log, f"[compat] Dropped unsupported pipeline args: {', '.join(dropped)}")
                    except Exception:
                        pass
    
        # Special-case: Wan2.2 dual-stage. `guidance_scale_2` only works when a second transformer is
        # present AND a `boundary_ratio` is set on the pipeline config.
        # Note: `boundary_ratio` is often a config attribute (pipe.config.boundary_ratio), not a __call__ kwarg.
        if "guidance_scale_2" in call_kwargs:
            try:
                br_cfg = getattr(getattr(pipe, "config", None), "boundary_ratio", None)
            except Exception:
                br_cfg = None
            has_t2 = hasattr(pipe, "transformer_2") or hasattr(pipe, "transformer2")
            if br_cfg is None or not has_t2:
                call_kwargs.pop("guidance_scale_2", None)
                key = ("drop_gs2",)
                if key not in self._compat_logged:
                    self._compat_logged.add(key)
                    if log is not None:
                        try:
                            reason = "boundary_ratio is None" if br_cfg is None else "transformer_2 missing"
                            self._log(log, f"[compat] Dropped guidance_scale_2 ({reason}).")
                        except Exception:
                            pass
    
        return call_kwargs
    def __init__(self):
        self.pipe = None
        self.mode = None
        self.model_id = None
        self.backend = None

        # Shared resource manager (VRAM hygiene / unload ordering).
        self._rm = get_resource_manager()

        self._compat_logged = set()
        # Filled by render_once; used to re-apply LoRAs when we must reload a pipeline.
        self._cache_dir: Optional[str] = None

        # Track scheduler state so per-clip overrides can be applied without full reloads.
        self._scheduler_use_unipc: Optional[bool] = None
        self._scheduler_flow_shift: Optional[float] = None

    def _vram_log(self, log: Optional[LogCallback], prefix: str = "[VRAM]") -> None:
        try:
            free, total = cuda_mem_info()
            if free and total and log is not None:
                self._log(log, f"{prefix} {free:.2f}GB free / {total:.2f}GB total")
        except Exception:
            pass
    def _is_oom_error(self, e: Exception) -> bool:
        try:
            if isinstance(e, torch.OutOfMemoryError):
                return True
        except Exception:
            pass
        try:
            s = str(e).lower()
            return ('out of memory' in s) or ('cuda oom' in s)
        except Exception:
            return False

    def _recommend_max_frames_per_call(self, cfg: ProjectConfig, eff_cfg: ProjectConfig, log: Optional[LogCallback]) -> Optional[int]:
        """Heuristic cap on frames per diffusion call to avoid VRAM spikes.

        We prefer reducing *segment length* (more segments) over reducing resolution
        to keep visual quality. Users can override via cfg.max_frames_per_segment.
        """
        try:
            if not bool(getattr(cfg, 'auto_vram_optimizations', True)):
                return None
        except Exception:
            return None

        try:
            user_cap = int(getattr(cfg, 'max_frames_per_segment', 0) or 0)
        except Exception:
            user_cap = 0
        if user_cap and user_cap > 0:
            return max(1, user_cap)

        backend = str(getattr(eff_cfg, 'backend', getattr(cfg, 'backend', 'wan')) or 'wan')
        model_id = str(getattr(eff_cfg, 'model_id', getattr(cfg, 'model_id', '')) or '')

        # Base calibration (empirical, safe defaults for 24GB cards).
        if 'A14B' in model_id:
            base = 33
        elif '14B' in model_id:
            base = 41
        elif '5B' in model_id:
            base = 61
        else:
            base = 61

        # Scale with resolution (frames roughly inversely proportional to pixels).
        base_pixels = 960 * 544
        try:
            pix = int(getattr(cfg, 'width', 960)) * int(getattr(cfg, 'height', 544))
            if pix <= 0:
                pix = base_pixels
        except Exception:
            pix = base_pixels
        ratio = float(base_pixels) / float(max(1, pix))
        cap = int(round(base * ratio))

        # Scale with available free VRAM (very rough, but helps when other apps use VRAM).
        try:
            free, total = cuda_mem_info()
            if free and total:
                if free < 6.0:
                    cap = int(cap * 0.60)
                elif free < 8.0:
                    cap = int(cap * 0.75)
                elif free < 10.0:
                    cap = int(cap * 0.90)
        except Exception:
            pass

        # Clamp + backend form constraints.
        cap = max(17, min(cap, 121))
        try:
            if backend == 'wan':
                cap = _ensure_form_4k1(int(cap))
            elif backend == 'ltx':
                cap = _ensure_form_8k1(int(cap))
        except Exception:
            pass

        # Inform once per session (avoid spam).
        try:
            key = ('cap_frames', backend, ('A14B' if 'A14B' in model_id else 'other'), int(getattr(cfg,'width',0)), int(getattr(cfg,'height',0)))
            if key not in self._compat_logged:
                self._compat_logged.add(key)
                if log is not None:
                    self._log(log, f"[VRAM] Auto segment cap: max {cap} frames/call (backend={backend})")
        except Exception:
            pass

        return int(cap)

    def _log(self, log: Optional[LogCallback], msg: str) -> None:
        if log:
            log(msg)

    def _download_lora(self, lora: LoraSpec, cache_dir: str) -> str:
        """Return a local file path for the LoRA, or an empty string if incomplete.

        UI can contain placeholder rows (e.g. user clicked "Add LoRA") that have a
        name but no source yet. We treat these as disabled instead of crashing.
        """
        if lora.local_path:
            lp = os.path.abspath(lora.local_path)
            if os.path.exists(lp):
                return lp
            # Local path provided but missing → treat as disabled.
            return ""
        if not (lora.repo_id and lora.weight_name):
            # Missing source fields → treat as disabled.
            return ""
        if hf_hub_download is None:
            raise RuntimeError("huggingface_hub is required to download LoRAs. Please install it: pip install huggingface_hub")
        return hf_hub_download(repo_id=lora.repo_id, filename=lora.weight_name, cache_dir=cache_dir)

    def _load_pipeline(self, cfg: ProjectConfig, log: Optional[LogCallback]) -> None:
        backend = getattr(cfg, "backend", "wan") or "wan"

        if backend == 'lingbot':
            # LingBot runs in a subprocess; no diffusers pipeline to load here.
            self.backend = 'lingbot'
            self.pipe = None
            return


        # --- VRAM hygiene (pre-load) ---
        try:
            if bool(getattr(getattr(cfg, 'studio', None), 'vram_manager_enabled', True)):
                target = float(getattr(getattr(cfg, 'studio', None), 'vram_target_free_gb', 5.0) or 5.0)
                self._rm.ensure_free(target, log=log, reason="before pipeline load", aggressive=True)
        except Exception:
            pass

        # Lazy-import guard: allow UI to start even if diffusers is missing.
        if backend == 'wan' and WanPipeline is None:
            raise RuntimeError('diffusers is required for Wan backend. Install: pip install diffusers')
        if backend == 'ltx' and LTXPipeline is None:
            raise RuntimeError('diffusers is required for LTX backend. Install: pip install diffusers')
        if backend == 'ltx2' and LTX2Pipeline is None:
            raise RuntimeError('diffusers is required for LTX2 backend. Install: pip install diffusers')
        if backend == 'cogvideox' and CogVideoXPipeline is None:
            raise RuntimeError('diffusers is required for CogVideoX backend. Install: pip install diffusers')

        # --- Safety: ensure backend matches model_id ---
        # A mismatched (model_id, backend) can happen when switching models
        # in the UI or when loading older autosaves. This causes confusing
        # diffusers errors (missing components, wrong pipeline class, etc.).
        # We auto-correct based on model_id unless the user is explicitly
        # using a local path.
        def _infer_backend_from_model_id(mid: str) -> Optional[str]:
            try:
                s = (mid or "").strip().upper()
            except Exception:
                s = ""
            if not s:
                return None
            # If it's a local path, don't guess.
            if os.path.sep in (mid or "") and (mid or "").startswith(("/", "./", "../")):
                return None
            if "LTX-2" in s or "LTX2" in s:
                return "ltx2"
            if "LTX-VIDEO" in s or s.startswith("LTX/") or "LTX_VIDEO" in s:
                return "ltx"
            if "COGVIDEOX" in s:
                return "cogvideox"
            if "WAN" in s:
                return "wan"
            return None

        inferred = _infer_backend_from_model_id(getattr(cfg, "model_id", "") or "")
        if inferred and inferred != backend:
            self._log(log, f"⚠️ Backend '{backend}' incompatible avec model_id '{cfg.model_id}'. Auto-fix → '{inferred}'.")
            backend = inferred
            try:
                cfg.backend = inferred  # persist for this run
            except Exception:
                pass
        if (
            self.pipe is not None
            and self.model_id == cfg.model_id
            and self.mode == cfg.mode
            and self.backend == backend
        ):
            return

        self._log(log, f"Chargement modèle: {cfg.model_id} backend={backend} mode={cfg.mode}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Prefer BF16 on modern GPUs, otherwise FP16.
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        def _fp_kwargs(**kw):
            # Try torch_dtype first, then dtype for older custom pipelines.
            return kw

        def _from_pretrained(cls, *args, **kwargs):
            try:
                return cls.from_pretrained(*args, torch_dtype=dtype, **kwargs)
            except TypeError:
                return cls.from_pretrained(*args, dtype=dtype, **kwargs)

        pipe = None

        if backend == "wan":
            vae_dtype = torch.float32 if cfg.vae_decode_fp32 else dtype
            try:
                vae = AutoencoderKLWan.from_pretrained(cfg.model_id, subfolder="vae", torch_dtype=vae_dtype)
            except TypeError:
                vae = AutoencoderKLWan.from_pretrained(cfg.model_id, subfolder="vae", dtype=vae_dtype)

            if cfg.mode == "T2V":
                pipe = _from_pretrained(WanPipeline, cfg.model_id, vae=vae)
            else:
                if WanImageToVideoPipeline is None:
                    raise RuntimeError("WanImageToVideoPipeline indisponible: mets à jour diffusers.")
                try:
                    pipe = _from_pretrained(WanImageToVideoPipeline, cfg.model_id, vae=vae)
                except Exception as e:
                    self._log(log, f"I2V direct échoue: {e}. Tentative image_encoder explicite…")
                    if CLIPVisionModel is None:
                        raise RuntimeError("transformers/CLIPVisionModel manquant pour Wan I2V")
                    image_encoder = CLIPVisionModel.from_pretrained(cfg.model_id, subfolder="image_encoder", torch_dtype=torch.float32)
                    pipe = _from_pretrained(WanImageToVideoPipeline, cfg.model_id, vae=vae, image_encoder=image_encoder)

            if cfg.use_unipc:
                if UniPCMultistepScheduler is None:
                    raise RuntimeError("UniPC scheduler indisponible.")
                try:
                    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=cfg.flow_shift)
                except Exception:
                    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

            # Wan2.2 dual-stage: boundary_ratio is typically stored on the pipeline config.
            # We set it best-effort so guidance_scale_2 can work when transformer_2 is present.
            if cfg.boundary_ratio is not None:
                try:
                    if hasattr(pipe, "config") and hasattr(pipe.config, "boundary_ratio"):
                        pipe.config.boundary_ratio = float(cfg.boundary_ratio)
                    elif hasattr(pipe, "boundary_ratio"):
                        pipe.boundary_ratio = float(cfg.boundary_ratio)
                except Exception:
                    pass

            # Warn once if dual-stage components aren't present.
            try:
                br_cfg = getattr(getattr(pipe, "config", None), "boundary_ratio", None)
            except Exception:
                br_cfg = None
            has_t2 = hasattr(pipe, "transformer_2") or hasattr(pipe, "transformer2")
            if log is not None and (br_cfg is None or not has_t2):
                reason = []
                if br_cfg is None:
                    reason.append("boundary_ratio missing")
                if not has_t2:
                    reason.append("transformer_2 missing")
                self._log(log, f"[Wan] Dual-stage inactive ({', '.join(reason)}). Using single-stage denoising.")

        elif backend == "cogvideox":
            if cfg.mode == "I2V":
                if CogVideoXImageToVideoPipeline is None:
                    raise RuntimeError("CogVideoX I2V pipeline indisponible: mets à jour diffusers (CogVideoX support).")
                pipe = _from_pretrained(CogVideoXImageToVideoPipeline, cfg.model_id)
            else:
                if CogVideoXPipeline is None:
                    raise RuntimeError("CogVideoX pipeline indisponible: mets à jour diffusers (CogVideoX support).")
                pipe = _from_pretrained(CogVideoXPipeline, cfg.model_id)

        elif backend == "ltx":
            if cfg.mode == "I2V":
                if LTXImageToVideoPipeline is None:
                    raise RuntimeError("LTX-Video I2V pipeline indisponible: mets à jour diffusers (LTX support).")
                pipe = _from_pretrained(LTXImageToVideoPipeline, cfg.model_id)
            else:
                if LTXPipeline is None:
                    raise RuntimeError("LTX-Video pipeline indisponible: mets à jour diffusers (LTX support).")
                pipe = _from_pretrained(LTXPipeline, cfg.model_id)
        elif backend == "ltx2":
            if cfg.mode == "I2V":
                cls = LTX2ImageToVideoPipeline
                if cls is None:
                    # Try a late import so we can surface the REAL reason (missing deps, wrong env, etc.)
                    import sys
                    import diffusers as _diffusers
                    err = None
                    try:
                        from diffusers.pipelines.ltx2 import LTX2ImageToVideoPipeline as _Cls  # type: ignore
                        cls = _Cls
                    except Exception as e:
                        err = e
                    if cls is None:
                        raise RuntimeError(
                            "LTX-2 I2V pipeline indisponible (diffusers ne fournit pas diffusers.pipelines.ltx2 dans cet environnement). "
                            f"python={sys.executable} diffusers={getattr(_diffusers,'__version__','?')}. "
                            f"Erreur import: {type(err).__name__ if err else 'Unknown'}: {err}. "
                            "Fix: installe Diffusers avec support LTX-2 (>=0.37.0.dev0, branche main): `pip install -U git+https://github.com/huggingface/diffusers.git@main`."
                        )
                pipe = _from_pretrained(cls, cfg.model_id)
            else:
                cls = LTX2Pipeline
                if cls is None:
                    import sys
                    import diffusers as _diffusers
                    err = None
                    try:
                        from diffusers import LTX2Pipeline as _Cls  # type: ignore
                        cls = _Cls
                    except Exception as e:
                        err = e
                    if cls is None:
                        raise RuntimeError(
                            "LTX-2 pipeline indisponible (LTX2Pipeline absent dans diffusers installé). "
                            f"python={sys.executable} diffusers={getattr(_diffusers,'__version__','?')}. "
                            f"Erreur import: {type(err).__name__ if err else 'Unknown'}: {err}. "
                            "Fix: installe Diffusers avec support LTX-2 (>=0.37.0.dev0, branche main): `pip install -U git+https://github.com/huggingface/diffusers.git@main`."
                        )
                pipe = _from_pretrained(cls, cfg.model_id)


        else:
            raise ValueError(f"Backend inconnu: {backend}")

        assert pipe is not None

        # ---- VRAM savers (best-effort) ----
        # Video VAEs can be very memory hungry during encode/decode. These toggles
        # reduce peak VRAM on many diffusers pipelines, and are safe to ignore when
        # unsupported.
        try:
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
        except Exception as e:
            try:
                self._log(log, f"[VRAM] VAE slicing/tiling unavailable: {e}")
            except Exception:
                pass

        self._apply_device_strategy(pipe, cfg, device, log)

        self.pipe = pipe
        self.mode = cfg.mode
        self.model_id = cfg.model_id
        self.backend = backend

        # Record scheduler state for later per-clip overrides
        try:
            if str(backend).lower() == 'wan':
                self._scheduler_use_unipc = bool(getattr(cfg, 'use_unipc', False))
                self._scheduler_flow_shift = float(getattr(cfg, 'flow_shift', 5.0) or 0.0) if self._scheduler_use_unipc else None
            else:
                self._scheduler_use_unipc = None
                self._scheduler_flow_shift = None
        except Exception:
            pass

        # Track current heavy pipeline for deterministic unload.
        try:
            self._rm.register(
                "i2v_pipe",
                self.pipe,
                kind="i2v",
                unload=self._unload_i2v_internal,
            )
        except Exception:
            pass


    def _apply_wan_scheduler_if_needed(self, cfg: ProjectConfig, log: Optional[LogCallback]) -> None:
        """Apply Wan scheduler settings (UniPC + flow_shift) even when reusing a pipeline.

        Note: switching *off* UniPC requires a pipeline reload to restore the original scheduler.
        """
        try:
            pipe = self.pipe
            if pipe is None:
                return
            backend = str(getattr(self, 'backend', getattr(cfg, 'backend', 'wan')) or 'wan').lower()
            if backend != 'wan':
                return

            want_unipc = bool(getattr(cfg, 'use_unipc', False))
            want_shift = float(getattr(cfg, 'flow_shift', 5.0) or 0.0)

            cur_unipc = self._scheduler_use_unipc
            cur_shift = self._scheduler_flow_shift

            # If we don't know current state, infer from scheduler class best-effort.
            if cur_unipc is None:
                try:
                    cur_unipc = bool(UniPCMultistepScheduler) and isinstance(getattr(pipe, 'scheduler', None), UniPCMultistepScheduler)
                except Exception:
                    cur_unipc = False

            if (want_unipc == bool(cur_unipc)) and ((not want_unipc) or (cur_shift is not None and abs(float(cur_shift) - float(want_shift)) < 1e-6)):
                return

            if want_unipc:
                if UniPCMultistepScheduler is None:
                    raise RuntimeError('UniPC scheduler indisponible (diffusers trop ancien).')
                try:
                    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=want_shift)
                except Exception:
                    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
                self._scheduler_use_unipc = True
                self._scheduler_flow_shift = float(want_shift)
                if log is not None:
                    self._log(log, f'[Scheduler] UniPC enabled (flow_shift={want_shift})')
                return

            # Switching OFF UniPC: safest is a reload (restores default scheduler).
            if bool(cur_unipc):
                if log is not None:
                    self._log(log, '[Scheduler] UniPC → default: reloading pipeline to restore default scheduler…')
                # Unload I2V and reload with desired config.
                self.unload(log=None)
                self._load_pipeline(cfg, log)

            self._scheduler_use_unipc = False
            self._scheduler_flow_shift = None
        except Exception as e:
            try:
                if log is not None:
                    self._log(log, f'[Scheduler] Warning: cannot apply scheduler override: {e}')
            except Exception:
                pass


    def _lora_compat_skip(self, cfg: ProjectConfig, lora: LoraSpec) -> Optional[str]:
        """Return a reason string if the LoRA should be skipped for current model/mode."""
        hint = " ".join([str(lora.name or ""), str(lora.repo_id or ""), str(lora.weight_name or ""), str(lora.local_path or "")]).upper()
        mid = str(cfg.model_id or "").upper()
        mode = str(cfg.mode or "").upper()

        # Common mismatch: A14B LoRAs on 5B model.
        if "A14B" in hint and "A14B" not in mid:
            return "LoRA A14B détectée mais modèle courant n'est pas A14B (ex: 5B) → incompatible."

        # Common mismatch: I2V-only LoRAs used in T2V mode.
        if "I2V" in hint and mode == "T2V":
            return "LoRA I2V détectée mais mode courant est T2V → désactive ou passe en I2V."

        return None

    def _guess_lora_prefix(self, lora_path: str) -> Any:
        """Best-effort guess for diffusers `prefix` when loading LoRA.

        Some LoRA checkpoints are saved without a top-level 'transformer.' prefix for Wan.
        Returning None lets diffusers try to match keys without filtering.
        """
        try:
            from safetensors.torch import safe_open
            with safe_open(lora_path, framework="pt") as f:
                keys = list(f.keys())
            # Fast scan
            # Prefer leaving prefix unset for Wan. Some LoRAs are saved with
            # keys starting with "transformer." but still don't match the exact
            # module names expected by the pipeline, causing noisy warnings.
            # We only force prefix for "unet."-style checkpoints.
            for k in keys[:2000]:
                if k.startswith("unet."):
                    return "unet"
            # If LoRA keys exist but no component prefix, use prefix=None
            if any(".lora_" in k or "lora_" in k for k in keys[:2000]):
                return None
        except Exception:
            return None
        # Default for WAN pipelines: do not filter keys
        return None


    def _apply_loras(self, cfg: ProjectConfig, log: Optional[LogCallback], cache_dir: str) -> None:
        if not cfg.loras:
            return
        pipe = self.pipe
        assert pipe is not None

        # Backend name is used for compatibility decisions (e.g. Wan LoRA prefix handling)
        backend = (getattr(cfg, "backend", None) or getattr(self, "backend", None) or "wan").lower()

        try:
            pipe.unload_lora_weights()
        except Exception:
            pass

        names: List[str] = []
        weights: List[float] = []
        loaded_entries: List[Dict[str, Any]] = []  # for sequential fuse fallback

        for lora in cfg.loras:

            # Guard: placeholder/incomplete specs (e.g. "new_lora" row with no file/repo)
            if not (getattr(lora, "local_path", None) or (getattr(lora, "repo_id", None) and getattr(lora, "weight_name", None))):
                self._log(log, f"[LoRA] Skip: {lora.name or '(unnamed)'} → missing local_path or (repo_id+weight_name).")
                continue

            # Optional compatibility guard: don't apply a LoRA to the wrong backend
            tb = getattr(lora, "target_backend", None)
            if tb and str(tb).lower() != backend:
                self._log(log, f"[LoRA] Skip: {lora.name or '(unnamed)'} → incompatible backend (need '{tb}', current '{backend}').")
                continue

            if getattr(lora, "local_path", None) and not os.path.exists(str(lora.local_path)):
                self._log(log, f"[LoRA] Skip: {lora.name or '(unnamed)'} → local_path introuvable: {lora.local_path}")
                continue

            # Compat guard BEFORE downloading/loading
            reason = self._lora_compat_skip(cfg, lora)
            if reason:
                self._log(log, f"[LoRA] Skip: {lora.name} → {reason}")
                continue

            # Guard: LoRAs meant for transformer_2 require a dual-stage pipeline
            if getattr(lora, "load_into_transformer_2", False):
                has_t2 = hasattr(pipe, "transformer_2") or hasattr(pipe, "transformer2")
                if not has_t2:
                    self._log(log, f"[LoRA] Skip: {lora.name} → transformer_2 absent (dual-noise LoRA).")
                    continue

            path = self._download_lora(lora, cache_dir)
            if not path:
                self._log(log, f"[LoRA] Skip: {lora.name or '(unnamed)'} → source LoRA invalide.")
                continue
            self._log(log, f"LoRA load: {lora.name} w={lora.weight} t2={lora.load_into_transformer_2} file={os.path.basename(path)}")
            lora_dir = os.path.dirname(path)
            lora_file = os.path.basename(path)

            import inspect
            sig = inspect.signature(pipe.load_lora_weights).parameters
            kwargs: Dict[str, Any] = {}

            # IMPORTANT (Wan2.x): diffusers may default `prefix='transformer'` which can emit noisy warnings
            # like: "No LoRA keys associated to WanTransformer3DModel found with prefix='transformer'".
            # For Wan pipelines, we force prefix=None (no filtering) so keys like "blocks.*" can match.
            if "prefix" in sig:
                if backend == "wan":
                    kwargs["prefix"] = None
                else:
                    # For non-Wan pipelines, only set a prefix when we are confident (e.g. "unet.")
                    pref = self._guess_lora_prefix(path)
                    if pref is not None:
                        kwargs["prefix"] = pref

            if lora.load_into_transformer_2 and "load_into_transformer_2" in sig:
                kwargs["load_into_transformer_2"] = True

            import warnings
            import contextlib
            import torch
            # Some PEFT/diffusers versions create LoRA parameters then flip requires_grad.
            # If we are inside torch.inference_mode() (directly or indirectly), this can throw:
            # "Setting requires_grad=True on inference tensor ...".
            # Force-disable inference_mode during adapter injection (still no-grad for inference).
            inf_ctx = torch.inference_mode(False) if hasattr(torch, "inference_mode") else contextlib.nullcontext()
            try:
                with inf_ctx, warnings.catch_warnings(record=True) as wlist:
                    warnings.simplefilter("always")
                    pipe.load_lora_weights(lora_dir, weight_name=lora_file, adapter_name=lora.name, **kwargs)

                # Older diffusers emitted this as a `warnings.warn(...)`; newer ones may log it.
                # We already force prefix=None for Wan, but keep a safety retry for other backends.
                need_retry = False
                for w in wlist:
                    msg = str(getattr(w, "message", ""))
                    if "No LoRA keys associated to" in msg and "prefix='transformer'" in msg:
                        if "prefix" in kwargs and kwargs.get("prefix") is not None:
                            need_retry = True
                        break

                if need_retry:
                    self._log(log, f"[LoRA] Warning: keys not found with prefix={kwargs.get('prefix')!r}. Retrying with prefix=None.")
                    kwargs2 = dict(kwargs)
                    kwargs2["prefix"] = None
                    try:
                        pipe.unload_lora_weights()
                    except Exception:
                        pass
                    pipe.load_lora_weights(lora_dir, weight_name=lora_file, adapter_name=lora.name, **kwargs2)
            except Exception as e:
                msg = str(e)
                self._log(log, f"❌ LoRA incompatible avec le modèle courant: {lora.name}.")
                if "size mismatch" in msg or "shape" in msg:
                    self._log(log, "   → Très probable: LoRA entraînée sur un autre backbone (ex: A14B) que ton modèle actuel (ex: 5B).")
                    self._log(log, "   → Solution: passe sur un modèle compatible (A14B) OU retire ce LoRA.")
                self._log(log, f"   Détails: {msg.splitlines()[0] if msg else repr(e)}")
                try:
                    pipe.unload_lora_weights()
                except Exception:
                    pass
                return
            names.append(lora.name)
            weights.append(float(lora.weight))
            try:
                loaded_entries.append({
                    'name': str(lora.name),
                    'weight': float(lora.weight),
                    'lora_dir': str(lora_dir),
                    'lora_file': str(lora_file),
                    'kwargs': dict(kwargs),
                })
            except Exception:
                pass

        # Activate adapters if supported by this diffusers/peft backend.
        # diffusers LoRA API has changed a few times. For some pipelines
        # `get_list_adapters()` returns a dict like {"transformer": ["a", "b"]}.
        # Using `set(dict)` would incorrectly return only the keys.
        available: set[str] = set()
        adapters_raw: Any = None
        if hasattr(pipe, "get_list_adapters"):
            try:
                adapters_raw = pipe.get_list_adapters()
            except Exception:
                adapters_raw = None

        if isinstance(adapters_raw, dict):
            # Flatten values
            for v in adapters_raw.values():
                if isinstance(v, (list, tuple, set)):
                    available.update(str(x) for x in v)
                elif v:
                    available.add(str(v))
        elif isinstance(adapters_raw, (list, tuple, set)):
            available.update(str(x) for x in adapters_raw)
        elif isinstance(adapters_raw, str) and adapters_raw:
            available.add(adapters_raw)

        # diffusers versions differ:
        # - some return adapter names directly
        # - some return component names like ['transformer', 'unet']
        #   and require get_list_adapters(component) to obtain the adapter names.
        KNOWN_COMPONENTS = {
            "transformer",
            "transformer_2",
            "unet",
            "text_encoder",
            "text_encoder_2",
        }
        if available and available.issubset(KNOWN_COMPONENTS) and hasattr(pipe, "get_list_adapters"):
            expanded: set[str] = set()
            for comp in sorted(available):
                comp_raw: Any = None
                # Try both calling conventions: get_list_adapters(component=...) and get_list_adapters(...)
                try:
                    comp_raw = pipe.get_list_adapters(component=comp)  # type: ignore
                except TypeError:
                    try:
                        comp_raw = pipe.get_list_adapters(comp)  # type: ignore
                    except Exception:
                        comp_raw = None
                except Exception:
                    comp_raw = None

                if isinstance(comp_raw, dict):
                    for v in comp_raw.values():
                        if isinstance(v, (list, tuple, set)):
                            expanded.update(str(x) for x in v)
                        elif v:
                            expanded.add(str(v))
                elif isinstance(comp_raw, (list, tuple, set)):
                    expanded.update(str(x) for x in comp_raw)
                elif isinstance(comp_raw, str) and comp_raw:
                    expanded.add(comp_raw)

            if expanded:
                available = expanded

        if not available:
            # Some pipelines/backends load LoRA weights but don't register adapters (or PEFT missing).
            # In that case, calling set_adapters raises. We fall back to fuse_lora if available.
            self._log(log, "[LoRA] Aucun adapter enregistré (get_list_adapters=∅).")
            self._log(log, "      → Pour le contrôle multi-LoRA: `pip install -U peft transformers` puis relance.")

            if hasattr(pipe, "fuse_lora"):
                # If multiple LoRAs were requested and we don't have adapter support,
                # do a robust sequential fuse: load → fuse → unload (repeat).
                if len(loaded_entries) > 1:
                    self._log(log, f"[LoRA] Fallback: sequential fuse ({len(loaded_entries)} LoRAs)…")
                    try:
                        pipe.unload_lora_weights()
                    except Exception:
                        pass
                    import warnings
                    import contextlib
                    import torch
                    inf_ctx = torch.inference_mode(False) if hasattr(torch, "inference_mode") else contextlib.nullcontext()
                    try:
                        for ent in loaded_entries:
                            try:
                                with inf_ctx, warnings.catch_warnings(record=True):
                                    warnings.simplefilter("always")
                                    pipe.load_lora_weights(
                                        ent['lora_dir'],
                                        weight_name=ent['lora_file'],
                                        adapter_name=ent.get('name', 'lora'),
                                        **(ent.get('kwargs') or {})
                                    )
                                pipe.fuse_lora(lora_scale=float(ent.get('weight', 1.0) or 1.0))
                            except Exception as e:
                                self._log(log, f"[LoRA] sequential fuse failed for {ent.get('name')}: {e}")
                            finally:
                                try:
                                    pipe.unload_lora_weights()
                                except Exception:
                                    pass
                        self._log(log, "[LoRA] sequential fuse done.")
                    except Exception as e:
                        self._log(log, f"[LoRA] sequential fuse fatal: {e}. LoRA désactivé.")
                        try:
                            pipe.unload_lora_weights()
                        except Exception:
                            pass
                else:
                    # Single-LoRA: fuse in-place.
                    try:
                        scale = float(weights[0]) if len(weights) == 1 else 1.0
                        pipe.fuse_lora(lora_scale=scale)
                        self._log(log, f"[LoRA] fuse_lora OK (scale={scale}).")
                    except Exception as e:
                        self._log(log, f"[LoRA] fuse_lora échoué: {e}. LoRA désactivé.")
                        try:
                            pipe.unload_lora_weights()
                        except Exception:
                            pass
            else:
                self._log(log, "[LoRA] Backend LoRA sans adapters/fuse_lora → LoRA ignoré.")
            return

        missing = set(names) - available
        if missing:
            self._log(log, f"[LoRA] Adapters demandés manquants: {sorted(missing)} ; présents: {sorted(available)}")
            # Best-effort remap: if counts match, align in sorted order.
            if len(available) == len(names):
                names = sorted(list(available))
                self._log(log, f"[LoRA] Remap automatique → {names}")
            else:
                # Use whatever is available
                names = sorted(list(available))
                weights = [1.0] * len(names)

        try:
            pipe.set_adapters(names, adapter_weights=weights)
        except TypeError:
            # older diffusers may not support adapter_weights kw
            pipe.set_adapters(names)
        except Exception as e:
            self._log(log, f"[LoRA] set_adapters échoué: {e}. Tentative fuse_lora…")
            if hasattr(pipe, "fuse_lora"):
                try:
                    scale = float(weights[0]) if len(weights) == 1 else 1.0
                    pipe.fuse_lora(lora_scale=scale)
                except Exception:
                    pass

    def _reload_pipeline_sequential_for_oom(self, cfg: ProjectConfig, log: Optional[LogCallback]) -> ProjectConfig:
        """Safely reload the current pipeline with sequential CPU offload.

        Why: switching from model_cpu_offload -> sequential_cpu_offload on an already-hooked
        pipeline can leave modules on the `meta` device and crash later with:
        "Cannot copy out of meta tensor; no data!".
        Reloading is slower but deterministic and robust.
        """
        import copy

        cfg2 = copy.copy(cfg)
        setattr(cfg2, "enable_sequential_cpu_offload", True)
        setattr(cfg2, "enable_model_cpu_offload", False)

        self._log(log, "[OOM] Reload pipeline with sequential CPU offload (safe)…")
        self.unload(log)
        self._load_pipeline(cfg2, log)

        # Re-apply LoRAs if any
        try:
            cache_dir = self._cache_dir
            if cache_dir:
                self._apply_loras(cfg2, log, cache_dir)
        except Exception as e:
            self._log(log, f"[OOM] LoRA re-apply failed after reload: {e}")
        return cfg2

    def generate_segment(
        self,
        cfg: ProjectConfig,
        scene: SceneSpec,
        num_frames: int,
        conditioning_image: Optional[Image.Image],
        seed_offset: int,
        progress: Optional[ProgressCallback],
        log: Optional[LogCallback],
        is_cancelled: Optional[Callable[[], bool]],
    ) -> List[np.ndarray]:
        pipe = self.pipe
        if pipe is None:
            # Recover from unexpected unloads before rendering a segment.
            self._log(log, "[SAFE] Pipeline missing at segment start -> reloading.")
            self._load_pipeline(cfg, log)
            pipe = self.pipe
        if pipe is None:
            raise RuntimeError("Pipeline not initialized (backend load failed).")

        backend = (getattr(cfg, "backend", None) or getattr(self, "backend", None) or "wan")
        if backend == 'lingbot':
            return self._generate_segment_lingbot(cfg, scene, num_frames, conditioning_image, seed_offset, progress, log, is_cancelled)

        # Base params (scene overrides win)
        steps = scene.num_inference_steps if scene.num_inference_steps is not None else cfg.num_inference_steps
        g1 = scene.guidance_scale if scene.guidance_scale is not None else cfg.guidance_scale
        g2 = scene.guidance_scale_2 if scene.guidance_scale_2 is not None else cfg.guidance_scale_2
        br = scene.boundary_ratio if scene.boundary_ratio is not None else cfg.boundary_ratio

        # Prompt injection (Phase C): scene + characters + location + merged negatives
        try:
            from .cinema_prompts import build_scene_prompts
            p_inj, n_inj = build_scene_prompts(cfg, scene)
            effective_prompt = (p_inj or '').strip() or 'Cinematic shot, gentle camera motion.'
            effective_negative = (n_inj or '').strip()
        except Exception:
            negative = scene.negative_prompt if scene.negative_prompt is not None else cfg.negative_prompt
            effective_prompt = (scene.prompt or '').strip() or 'Cinematic shot, gentle camera motion.'
            effective_negative = (negative or '').strip()

        # T2V is more prone to temporal artifacts. Add light guardrails unless user already specified them.
        if str(getattr(cfg, 'mode', '')).upper() == 'T2V':
            lp = effective_prompt.lower()
            ln = effective_negative.lower()
            if 'single continuous shot' not in lp:
                effective_prompt = effective_prompt + ", single continuous shot, consistent subject appearance, stable camera, coherent motion"
            # Extend negative with common video failures
            if 'flicker' not in ln:
                extra_neg = "flicker, temporal jitter, frame tearing, warping, unstable text"
                effective_negative = (effective_negative + (", " if effective_negative else "") + extra_neg).strip()

        # Strict coherence: reduce drift (identity/background) across frames and chunked segments.
        if bool(getattr(cfg, 'strict_clip_coherence', False)):
            lp = (effective_prompt or '').lower()
            if 'consistent subject' not in lp and 'consistent identity' not in lp:
                effective_prompt = (effective_prompt + ", consistent subject identity, consistent clothing, consistent background, no sudden scene changes, stable geometry, coherent motion").strip()
            ln = (effective_negative or '').lower()
            extra_neg = "identity drift, face changing, body morphing, background changing, object popping, teleporting, sudden cuts"
            if extra_neg.split(",")[0] not in ln:
                effective_negative = (effective_negative + (", " if effective_negative else "") + extra_neg).strip()


        # Optional style packs (global or per-shot)
        pack_name = (getattr(scene, "style_pack", None) or getattr(cfg, "global_style_pack", None) or "").strip()
        if pack_name and pack_name.lower() in ("(inherit)", "inherit", "none", "(none)"):
            pack_name = ""
        if pack_name:
            try:
                pack = get_pack(pack_name)
                # Apply prompt/negative additions
                p2, n2 = apply_pack(effective_prompt, effective_negative, pack)
                effective_prompt = p2 or effective_prompt
                effective_negative = n2

                # Apply suggested settings only if not overridden per scene
                if pack.suggest and scene.num_inference_steps is None and isinstance(pack.suggest.get("steps"), (int, float)):
                    steps = int(pack.suggest["steps"])
                if pack.suggest and scene.guidance_scale is None and isinstance(pack.suggest.get("guidance_scale"), (int, float)):
                    g1 = float(pack.suggest["guidance_scale"])

                if log is not None:
                    self._log(log, f"StylePack: {pack_name}")
            except Exception as e:
                if log is not None:
                    self._log(log, f"[StylePack] warning: {e}")

        # --- Realism guardrails ---
        # Users often end up with a stray token like "anime" or "cartoon" in the prompt
        # (from a storyboard model, a style pack, or copy/paste). If the user is in a
        # realistic/cinema workflow, we proactively push those tokens into the negative prompt.
        # This is conservative: we only do it when a cinematic/realism style pack is selected
        # OR when the global negative prompt already contains realism guards.
        def _realism_guard(prompt: str, neg: str) -> tuple[str, str]:
            p = (prompt or "").strip()
            n = (neg or "").strip()
            lp = p.lower()
            ln = n.lower()
            cinematic_context = (
                (pack_name and any(k in pack_name.lower() for k in ("cinema", "cinematic", "realism", "photoreal")))
                or any(k in ln for k in ("cartoon", "anime", "pixar", "toon"))
            )
            if not cinematic_context:
                return p, n

            banned = [
                "cartoon", "anime", "illustration", "comic", "toon", "toon shading",
                "pixar", "3d render", "cgi", "cel shaded"
            ]
            found = [b for b in banned if b in lp]
            if found:
                # Remove simple occurrences to reduce accidental style bleed.
                for b in found:
                    p = p.replace(b, " ").replace(b.title(), " ").replace(b.upper(), " ")
                extra = ", ".join(sorted(set(found)))
                if extra and extra not in ln:
                    n = (n + (", " if n else "") + extra).strip()
                if log is not None:
                    self._log(log, f"[Prompt] realism guard: moved style tokens to negative: {extra}")
            return " ".join(p.split()), n

        effective_prompt, effective_negative = _realism_guard(effective_prompt, effective_negative)

        # --- Prompt token safety (V24) ---
        # Many CLIP-based text encoders cap at 77 tokens. Diffusers will silently truncate,
        # which can lead to drift/inconsistency because the tail of the prompt is ignored.
        # When enabled, we proactively trim prompts using the pipeline tokenizer(s).
        if bool(getattr(cfg, "strict_prompt_max_tokens", False)):
            try:
                strategy = str(getattr(cfg, "prompt_trim_strategy", "head_tail") or "head_tail").strip().lower()

                def _trim_one(text: str, tok, tag: str) -> str:
                    if not text or tok is None:
                        return text
                    max_len = int(getattr(tok, "model_max_length", 77) or 77)
                    try:
                        enc = tok(text, truncation=False, return_tensors="pt")
                        ids = enc.get("input_ids")
                        if ids is None:
                            return text
                        ids0 = ids[0].tolist()
                    except Exception:
                        return text

                    if len(ids0) <= max_len:
                        return text

                    bos = getattr(tok, "bos_token_id", None)
                    eos = getattr(tok, "eos_token_id", None)
                    has_wrap = bool(bos is not None and eos is not None and len(ids0) >= 2 and ids0[0] == bos and ids0[-1] == eos)
                    core = ids0[1:-1] if has_wrap else ids0
                    budget = max_len - (2 if has_wrap else 0)
                    if budget <= 1:
                        budget = max_len

                    if strategy in ("head", "start"):
                        kept = core[:budget]
                    else:
                        head = max(1, budget // 2)
                        tail = max(1, budget - head)
                        kept = core[:head] + core[-tail:]

                    new_ids = ([bos] + kept + [eos]) if has_wrap else kept
                    try:
                        new_text = tok.decode(new_ids, skip_special_tokens=True)
                        new_text = " ".join((new_text or "").split())
                    except Exception:
                        return text

                    if log is not None:
                        try:
                            self._log(log, f"[Prompt] {tag} trimmed {len(ids0)} -> {max_len} tokens ({strategy})")
                        except Exception:
                            pass
                    return new_text or text

                # Apply sequentially to satisfy multiple tokenizers (e.g. SDXL has tokenizer + tokenizer_2).
                t = effective_prompt
                t = _trim_one(t, getattr(pipe, "tokenizer", None), "prompt")
                t = _trim_one(t, getattr(pipe, "tokenizer_2", None), "prompt2")
                effective_prompt = t

                tn = effective_negative
                tn = _trim_one(tn, getattr(pipe, "tokenizer", None), "neg")
                tn = _trim_one(tn, getattr(pipe, "tokenizer_2", None), "neg2")
                effective_negative = tn
            except Exception as e:
                if log is not None:
                    try:
                        self._log(log, f"[Prompt] trim skipped (error): {e}")
                    except Exception:
                        pass

        if log is not None:
            self._log(log, f"Prompt: {effective_prompt[:180]}")
            if effective_negative:
                self._log(log, f"Neg: {effective_negative[:180]}")

        width, height = normalize_hw_for_backend(pipe, cfg.width, cfg.height, backend=backend)

        # Frame constraints per backend (best-effort)
        if backend == "wan":
            num_frames = _ensure_form_4k1(int(num_frames))
        elif backend in ("ltx", "ltx2"):
            num_frames = _ensure_form_8k1(int(num_frames))
        else:
            num_frames = int(num_frames)

        base_seed = scene.seed if scene.seed is not None else cfg.base_seed
        seed = int(base_seed) + (int(seed_offset) * 1337 if cfg.vary_seed_per_segment else 0)
        gen_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        steps_total = int(steps)

        def cb_on_step_end(*args, **_kw):
            # Diffusers callback signature varies by version.
            if len(args) == 4:
                _pipe, step, _timestep, kwargs = args
            elif len(args) == 3:
                step, _timestep, kwargs = args
            else:
                step = int(args[-3]) if len(args) >= 3 else 0
                kwargs = args[-1] if len(args) >= 1 else {}
            if is_cancelled and is_cancelled():
                raise RuntimeError("Annulé par l'utilisateur")
            if progress:
                progress((int(step) + 1) / max(1, steps_total), f"Denoise {int(step)+1}/{steps_total}")
            return kwargs

        # Many pipelines accept a subset of these; we'll filter via signature
        anchor_img_for_first_frame: Optional[Image.Image] = None
        call_kwargs: Dict[str, Any] = dict(
            prompt=effective_prompt,
            negative_prompt=effective_negative or None,
            width=int(width),
            height=int(height),
            num_frames=int(num_frames),
            num_inference_steps=int(steps),
            guidance_scale=float(g1),
            generator=generator,
            callback_on_step_end=cb_on_step_end,
        )

        # Wan2.2 dual-stage controls
        if backend == "wan":
            # boundary_ratio is typically a config attribute, not a __call__ kwarg.
            if br is not None:
                try:
                    if hasattr(pipe, "config") and hasattr(pipe.config, "boundary_ratio"):
                        pipe.config.boundary_ratio = float(br)
                    elif hasattr(pipe, "boundary_ratio"):
                        pipe.boundary_ratio = float(br)
                except Exception:
                    pass

            try:
                br_cfg = getattr(getattr(pipe, "config", None), "boundary_ratio", None)
            except Exception:
                br_cfg = None
            has_t2 = hasattr(pipe, "transformer_2") or hasattr(pipe, "transformer2")

            if g2 is not None:
                if br_cfg is None or not has_t2:
                    key = ("drop_gs2_runtime",)
                    if key not in self._compat_logged:
                        self._compat_logged.add(key)
                        if log is not None:
                            reason = "boundary_ratio missing" if br_cfg is None else "transformer_2 missing"
                            self._log(log, f"[Wan] guidance_scale_2 ignoré ({reason}).")
                else:
                    call_kwargs["guidance_scale_2"] = float(g2)

            call_kwargs["output_type"] = "np"
        elif backend == "ltx2":
            # LTX-2 returns (video, audio) when return_dict=False; we only consume video for now.
            call_kwargs["output_type"] = "np"
            call_kwargs["return_dict"] = False
            call_kwargs["frame_rate"] = float(getattr(cfg, "fps", 24))
        else:
            # Safer default across pipelines
            call_kwargs["output_type"] = "pil"

        if cfg.mode == "I2V":
            if conditioning_image is None:
                cand = None
                if getattr(scene, "use_character_ref", False) and getattr(cfg, "character_image_path", ""):
                    cand = cfg.character_image_path
                if not cand and getattr(cfg, "input_image_path", ""):
                    cand = cfg.input_image_path
                if not cand and getattr(cfg, "character_image_path", ""):
                    cand = cfg.character_image_path
                if cand and os.path.exists(cand):
                    conditioning_image = Image.open(cand).convert("RGB")
                else:
                    raise ValueError("Mode I2V: fournir une image initiale (ou définir une Character Reference).")

            img, h2, w2 = resize_image_for_backend(conditioning_image, pipe, target_w=int(width), target_h=int(height), backend=backend)
            anchor_img_for_first_frame = img
            call_kwargs["image"] = img
            call_kwargs["height"] = int(h2)
            call_kwargs["width"] = int(w2)

        call_kwargs = self._filter_pipe_kwargs(pipe, call_kwargs, log)

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        def _run_pipe_call(_pipe):
            # IMPORTANT: keep pipeline calls in no-grad; but DO NOT wrap pipeline
            # (re)loading / LoRA injection in inference_mode, or you can end up with
            # "inference tensors" that later break LoRA injection and/or accelerate offload.
            with torch.no_grad():
                if backend == "ltx2":
                    res = _pipe(**call_kwargs)
                    # Tuple: (video, audio) or (video,)
                    if isinstance(res, tuple):
                        return res[0] if len(res) >= 1 else None
                    return getattr(res, "frames", None) or getattr(res, "videos", None) or getattr(res, "video", None) or getattr(res, "images", None) or res
                return _pipe(**call_kwargs).frames

        try:
            out = _run_pipe_call(pipe)
        except torch.OutOfMemoryError as oom:
            self._log(log, f"OOM during segment generation: {oom}.")

            # If we're already in sequential offload, there's nothing safer to do automatically.
            if getattr(cfg, "enable_sequential_cpu_offload", False):
                raise

            # IMPORTANT: do NOT switch offload modes on the existing pipeline.
            # It can leave modules on `meta` and crash later (meta tensor error).
            cfg = self._reload_pipeline_sequential_for_oom(cfg, log)
            pipe = self.pipe
            assert pipe is not None

            # Re-filter call kwargs for the reloaded pipeline.
            call_kwargs = self._filter_pipe_kwargs(pipe, call_kwargs, log)

            out = _run_pipe_call(pipe)
        except RuntimeError as e:
            # Some scheduler/model combos can throw shape mismatch errors at runtime.
            # The most common one for Wan + UniPC is:
            #   "The size of tensor a (...) must match the size of tensor b (...)"
            # In that case, auto-fallback to the default scheduler (reload pipeline)
            # rather than crashing the entire render.
            msg = str(e)
            if "Annulé" in msg or "Cancelled" in msg:
                raise

            is_wan = (backend == "wan")
            # NOTE: We cannot rely only on cfg.use_unipc here.
            # Users can switch presets or reuse pipelines and end up with a UniPC scheduler
            # still attached even if cfg.use_unipc is now False.
            looks_like_shape_mismatch = ("size of tensor" in msg and "must match" in msg)
            cur_scheduler = getattr(pipe, "scheduler", None)
            cur_is_unipc = bool(UniPCMultistepScheduler) and isinstance(cur_scheduler, UniPCMultistepScheduler)
            want_unipc = bool(getattr(cfg, "use_unipc", False))

            # Trigger fallback if either the config *wants* UniPC or the current scheduler *is* UniPC.
            if is_wan and (want_unipc or cur_is_unipc) and looks_like_shape_mismatch:
                try:
                    self._log(log, f"[Scheduler] UniPC runtime error -> fallback to default scheduler and retry once: {msg}")
                except Exception:
                    pass

                cfg2 = replace(cfg, use_unipc=False)
                try:
                    self.unload(log=None)
                    self._load_pipeline(cfg2, log)
                    pipe = self.pipe
                    assert pipe is not None
                    call_kwargs = self._filter_pipe_kwargs(pipe, call_kwargs, log)
                    out = _run_pipe_call(pipe)
                except Exception:
                    raise e

            elif is_wan and looks_like_shape_mismatch and bool(getattr(cfg, "cinema_safe_mode", False)):
                # In Cinema SAFE mode, be extra aggressive: even if UniPC wasn't detected,
                # a Wan scheduler can still end up in a bad state. Hard-reload and retry once.
                try:
                    self._log(log, f"[SAFE][Wan] shape-mismatch -> hard reload pipeline and retry once: {msg}")
                except Exception:
                    pass
                try:
                    cfg2 = cfg
                    if hasattr(cfg, "flow_shift"):
                        cfg2 = replace(cfg2, flow_shift=0.0)
                    if hasattr(cfg2, "use_unipc"):
                        cfg2 = replace(cfg2, use_unipc=False)
                    self.unload(log=None)
                    self._load_pipeline(cfg2, log)
                    pipe = self.pipe
                    assert pipe is not None
                    call_kwargs = self._filter_pipe_kwargs(pipe, call_kwargs, log)
                    out = _run_pipe_call(pipe)
                except Exception:
                    raise e
            else:
                raise
        finally:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        frames: List[np.ndarray] = self._normalize_pipe_frames(out, log)

        # Strict coherence: force frame 0 to exactly match the conditioning image (I2V).
        # This eliminates subtle pops at chunk boundaries (even with overlap blending).
        try:
            if frames and anchor_img_for_first_frame is not None and (
                bool(getattr(cfg, 'force_first_frame_to_conditioning', False)) or bool(getattr(cfg, 'strict_clip_coherence', False))
            ):
                frames[0] = np.array(anchor_img_for_first_frame.convert("RGB"), dtype=np.uint8)
        except Exception:
            pass

        return frames

    def _write_state(self, state_path: str, state: Dict[str, Any]) -> None:
        write_json(state_path, state)

    def _load_state(self, state_path: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(state_path):
            return None
        try:
            return read_json(state_path)
        except Exception:
            return None

    def render_once(
        self,
        cfg: ProjectConfig,
        progress_global: Optional[ProgressCallback],
        log: Optional[LogCallback],
        is_cancelled: Optional[Callable[[], bool]],
        clip_state: Optional[ClipStateCallback] = None,
        clip_progress: Optional[ClipProgressCallback] = None,
        clip_asset: Optional[ClipAssetCallback] = None,
        run_name_suffix: str = "",
        resume_from: Optional[str] = None,
    ) -> str:
        safe_makedirs(cfg.output_dir)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{cfg.project_name}{run_name_suffix}_{timestamp}"
        proj_dir = os.path.join(cfg.output_dir, run_name) if resume_from is None else os.path.abspath(resume_from)
        safe_makedirs(proj_dir)

        frames_dir = os.path.join(proj_dir, "frames")
        cache_dir = os.path.join(proj_dir, "hf_cache")
        safe_makedirs(frames_dir); safe_makedirs(cache_dir)

        # Remember for safe in-run pipeline reloads (OOM fallback).
        self._cache_dir = cache_dir

        cfg.save(os.path.join(proj_dir, "project.json"))
        write_json(os.path.join(proj_dir, "env.json"), env_report())

        if cfg.studio.export_shotlist_csv:
            export_shotlist_csv(cfg, os.path.join(proj_dir, "shotlist.csv"))


        # --- Ciné stable / no-overlap strict ---
        # Some stitching modes (notably optical-flow based) can create heavy warp/halo artefacts,
        # especially on high motion. When no_overlap_strict is enabled we hard force:
        #   overlap_frames=0 and blend_mode='cut'
        # This guarantees there is no flow/crossfade contamination at segment boundaries.
        no_overlap_strict = bool(getattr(cfg, 'no_overlap_strict', False) or getattr(cfg, 'no_overlap', False))
        eff_overlap = 0 if no_overlap_strict else int(getattr(cfg, 'overlap_frames', 0) or 0)
        eff_blend = 'cut' if no_overlap_strict else str(getattr(cfg, 'blend_mode', 'cut') or 'cut')
        eff_scene_trans_frames = 0 if no_overlap_strict else int(getattr(cfg, 'scene_transition_frames', 0) or 0)
        eff_scene_trans_mode = 'cut' if no_overlap_strict else str(getattr(cfg, 'scene_transition_mode', 'cut') or 'cut')

        # Note: per-scene transition_* are respected unless no_overlap_strict is enabled.
        per_scene_trans_frames = [0 if no_overlap_strict else int(getattr(s, 'transition_frames', 0) or 0) for s in (cfg.scenes or [])]
        max_overlap = max(int(eff_overlap), int(eff_scene_trans_frames), max(per_scene_trans_frames or [0]))

        writer = FrameWriter(
            frames_dir,
            max_overlap=max_overlap,
            default_overlap=int(eff_overlap),
            default_blend=str(eff_blend),
            default_ease=str(getattr(cfg, 'scene_transition_ease', 'smoothstep') or 'smoothstep'),
            log=log,
        )

        # Per-clip render cache (stable across runs).
        clip_cache_enabled = bool(getattr(cfg, 'clip_cache_enabled', True))
        clip_cache_root_dir = None
        resume_from_folder = bool(resume_from is not None and bool(getattr(cfg.studio, 'crash_safe_resume', False)))
        try:
            if clip_cache_enabled:
                clip_cache_root_dir = clip_cache_root(cfg)
                maybe_prune_cache(cfg, clip_cache_root_dir, log=log)
        except Exception:
            clip_cache_root_dir = None
            clip_cache_enabled = False

        # Optional: clear stale CUDA allocations before starting (helps after previous runs / model swaps).
        try:
            if bool(getattr(cfg, 'auto_vram_optimizations', True)):
                cuda_cleanup(aggressive=True)
                self._vram_log(log, prefix="[VRAM] After pre-clean")
        except Exception:
            pass

        chunk_frames = seconds_to_num_frames(cfg.chunk_seconds, cfg.fps, backend=getattr(cfg, "backend", "wan"))
        overlap = int(eff_overlap)
        if overlap >= chunk_frames:
            raise ValueError("overlap_frames doit être < chunk_frames.")

        total_seconds = sum(max(0.0, s.seconds) for s in cfg.scenes)
        total_frames = int(round(total_seconds * cfg.fps)) + 1
        effective_new = max(1, chunk_frames - overlap)
        approx_segments = max(1, math.ceil((total_frames - 1) / effective_new))

        state_path = os.path.join(proj_dir, "state.json")

        # resume defaults
        scene_i0 = 0
        seg_idx = 0
        need_remaining = None
        first_segment = True
        prev_scene_dict = None
        conditioning_image: Optional[Image.Image] = None
        loc_memory: Dict[str, str] = {}

        # Ultra isolation: hard reset pipelines/VRAM between clips for maximum independence.
        ultra_isolation = bool(getattr(cfg, 'ultra_isolated_clips', False) or getattr(cfg, 'ultra_clip_isolation', False))
        if ultra_isolation:
            try:
                self._log(log, '[Isolation] Ultra clip isolation ENABLED (full pipeline reset between clips).')
            except Exception:
                pass
            # Start from a clean slate (in case the UI previously loaded a model).
            try:
                self.unload_all()
            except Exception:
                pass
            try:
                cuda_cleanup(aggressive=True)
                self._vram_log(log, prefix='[VRAM] After isolation pre-clean')
            except Exception:
                pass

        if resume_from is not None and cfg.studio.crash_safe_resume:
            last_written = writer.restore_from_existing()
            st = self._load_state(state_path)
            if st:
                scene_i0 = int(st.get("scene_i", 0))
                seg_idx = int(st.get("seg_idx", 0))
                need_remaining = st.get("need_remaining", None)
                first_segment = bool(st.get("first_segment", True))
                prev_scene_dict = st.get("prev_scene", None)
                ci_path = st.get("conditioning_image_path", None)
                if ci_path and os.path.exists(ci_path):
                    conditioning_image = Image.open(ci_path).convert("RGB")
                elif last_written > 0:
                    conditioning_image = Image.open(os.path.join(frames_dir, f"frame_{last_written:06d}.png")).convert("RGB")
            else:
                if last_written > 0:
                    conditioning_image = Image.open(os.path.join(frames_dir, f"frame_{last_written:06d}.png")).convert("RGB")
            self._log(log, f"RESUME: dir={proj_dir} scene_i={scene_i0} seg_idx={seg_idx}")

        self._log(log, f"Sortie: {proj_dir}")
        self._log(log, f"Base: {cfg.width}x{cfg.height}@{cfg.fps} chunk_frames={chunk_frames} overlap={overlap} segments≈{approx_segments}")

        prev_scene: Optional[SceneSpec] = SceneSpec.from_dict(prev_scene_dict) if isinstance(prev_scene_dict, dict) else None

        for scene_i in range(scene_i0, len(cfg.scenes)):
            if is_cancelled and is_cancelled():
                raise RuntimeError("Annulé par l'utilisateur")


            scene = cfg.scenes[scene_i]

            # Between clips: hard reset pipelines/VRAM so no state can leak across clips.
            try:
                if ultra_isolation and int(scene_i) > int(scene_i0):
                    self._log(log, f'[Isolation] Reset pipelines before clip {scene_i+1}/{len(cfg.scenes)} ({scene.label})')
                    self.unload_all()
                    cuda_cleanup(aggressive=True)
                    try:
                        self._vram_log(log, prefix='[VRAM] After clip-boundary reset')
                    except Exception:
                        pass
            except Exception:
                pass

            # Notify UI (best-effort) that this clip is starting.
            try:
                if clip_state:
                    clip_state(int(scene_i), 'rendering', False)
                if clip_progress:
                    clip_progress(int(scene_i), 0.0, 'start')
            except Exception:
                pass
            
            conditioning_path_hint: Optional[str] = None
            # --- Premiere-style per-clip overrides (backend/model/mode/input) ---
            import copy
            eff_cfg = copy.copy(cfg)
            eff_cfg.backend = scene.backend_override or getattr(cfg, 'backend', 'wan')
            eff_cfg.model_id = scene.model_id_override or cfg.model_id
            eff_cfg.mode = scene.mode_override or cfg.mode
            eff_cfg.input_image_path = scene.input_image_path_override or cfg.input_image_path
            # Per-clip scheduler overrides (inherit by default)
            try:
                uo = getattr(scene, 'use_unipc_override', None)
                if uo is None:
                    eff_cfg.use_unipc = bool(getattr(cfg, 'use_unipc', False))
                    eff_cfg.flow_shift = float(getattr(cfg, 'flow_shift', 5.0) or 0.0)
                else:
                    eff_cfg.use_unipc = bool(uo)
                    if bool(uo):
                        fs = getattr(scene, 'flow_shift_override', None)
                        eff_cfg.flow_shift = float(fs if fs is not None else getattr(cfg, 'flow_shift', 5.0) or 0.0)
                    else:
                        eff_cfg.flow_shift = float(getattr(cfg, 'flow_shift', 5.0) or 0.0)
            except Exception:
                pass
            # Cinema SAFE mode: force the most compatible scheduler settings.
            # UniPC is known to produce shape-mismatch errors with some Wan scheduler/model combos.
            try:
                if bool(getattr(cfg, 'cinema_safe_mode', False)) and str(getattr(eff_cfg, 'backend', 'wan')) == 'wan':
                    eff_cfg.use_unipc = False
                    eff_cfg.flow_shift = 0.0
            except Exception:
                pass

            # Stability / coherence flags
            # Ultra isolation forces the strictest settings (clip independence + anti-drift).
            strict_coherence = bool(getattr(cfg, 'strict_clip_coherence', False) or ultra_isolation)
            reset_between_clips = bool(getattr(cfg, 'reset_continuity_between_clips', False) or ultra_isolation)
            force_first = bool(getattr(cfg, 'force_first_frame_to_conditioning', False) or ultra_isolation)
            cut_strict = bool(getattr(cfg, 'cut_strict_between_clips', False) or ultra_isolation)

            # Apply enforced flags to the effective per-clip config (so generate_segment sees them).
            try:
                eff_cfg.strict_clip_coherence = strict_coherence
                eff_cfg.reset_continuity_between_clips = reset_between_clips
                eff_cfg.force_first_frame_to_conditioning = force_first
                eff_cfg.cut_strict_between_clips = cut_strict
                # No-overlap strict: enforce cut stitching regardless of presets.
                if bool(no_overlap_strict):
                    eff_cfg.no_overlap_strict = True
                    eff_cfg.overlap_frames = 0
                    eff_cfg.blend_mode = 'cut'
                    eff_cfg.scene_transition_mode = 'cut'
                    eff_cfg.scene_transition_frames = 0
                if ultra_isolation:
                    # Ensure clip boundaries are hard cuts.
                    eff_cfg.scene_transition_mode = 'cut'
                    eff_cfg.scene_transition_frames = 0
            except Exception:
                pass

            # In strict mode we want the seed offset to actually affect generation;
            # generate_segment only applies seed_offset when cfg.vary_seed_per_segment is True.
            if strict_coherence:
                try:
                    eff_cfg.vary_seed_per_segment = True
                except Exception:
                    pass

            # Coherence policy:
            # - Keep continuity (conditioning = last frame) by default.
            # - Reset continuity if:
            #     * scene.hard_cut
            #     * per-scene input image override
            #     * project reset_continuity_between_clips (clip independence)
            #     * project cut_strict_between_clips (when you want hard cuts, you usually want no conditioning spill)
            reset_continuity = bool(getattr(scene, 'hard_cut', False))
            if reset_between_clips and int(scene_i) > 0:
                reset_continuity = True
            if bool(cut_strict) and int(scene_i) > 0:
                reset_continuity = True

            force_cut_boundary = bool(reset_continuity)

            # If we reset continuity between clips, also clear init-frame location memory
            # so Flux2 won't reuse the previous clip's "last location" as a hidden anchor.
            if reset_continuity and reset_between_clips:
                try:
                    loc_memory.clear()
                except Exception:
                    pass

            if reset_continuity:
                conditioning_image = None
                # If project/scene has an input image, use it as a fresh anchor for this scene.
                try:
                    ip = getattr(eff_cfg, 'input_image_path', None) or ''
                    if ip and os.path.exists(ip):
                        conditioning_image = Image.open(ip).convert('RGB')
                        conditioning_path_hint = ip
                except Exception as e:
                    self._log(log, f"⚠️ Cannot load input image for hard_cut: {e}")

            # If this clip provides its own input image override, it always resets continuity.
            if scene.input_image_path_override and os.path.exists(scene.input_image_path_override):
                try:
                    conditioning_image = Image.open(scene.input_image_path_override).convert("RGB")
                    reset_continuity = True
                    conditioning_path_hint = scene.input_image_path_override
                    force_cut_boundary = True
                except Exception as e:
                    self._log(log, f"⚠️ Cannot load scene input override: {e}")

            # Flux2 auto-init (only when a fresh anchor is needed)
            try:
                init_cfg = getattr(cfg, 'init_image', None)
                need_anchor = (str(getattr(eff_cfg, 'mode', '')).upper() == 'I2V') and (conditioning_image is None) and (not scene.input_image_path_override)
                if init_cfg and getattr(init_cfg, 'enabled', False) and need_anchor:
                    from .flux2_init import generate_init_image_for_scene
                    loc_name = (getattr(scene, 'location', '') or '').strip()
                    mem = None
                    if loc_name and loc_name in loc_memory:
                        mem = loc_memory.get(loc_name)
                    if mem is None and (not strict_coherence):
                        mem = loc_memory.get('__last__')
                    force = (str(getattr(init_cfg, 'policy', 'necessary')) == 'always_refine')
                    res = generate_init_image_for_scene(scene, cfg, force=force, last_location_memory_path=mem, log=log)
                    if res and getattr(res, 'path', None) and os.path.exists(res.path):
                        conditioning_image = Image.open(res.path).convert('RGB')
                        conditioning_path_hint = res.path
                        # Persist on scene for UI (best-effort)
                        try:
                            scene.init_frame_path = str(res.path)
                        except Exception:
                            try:
                                setattr(scene, 'init_frame_path', str(res.path))
                            except Exception:
                                pass
                        if loc_name:
                            loc_memory[loc_name] = res.path
                        loc_memory['__last__'] = res.path
                        self._log(log, f"[InitFrame] using: {res.path}")
                        try:
                            if clip_asset:
                                clip_asset(int(scene_i), 'init', str(res.path))
                        except Exception:
                            pass
                        try:
                            if clip_progress:
                                clip_progress(int(scene_i), 0.0, 'init_frame')
                        except Exception:
                            pass
            except Exception as e:
                self._log(log, f"⚠️ InitFrame generation skipped/failed: {e}")

            # If the clip opts-in, use global character reference to anchor identity (fallback, I2V only)
            if getattr(eff_cfg, 'mode', '').upper() == 'I2V' and (not scene.input_image_path_override) and getattr(scene, 'use_character_ref', False) and conditioning_image is None:
                char_p = getattr(cfg, 'character_image_path', '') or ''
                if char_p and os.path.exists(char_p):
                    try:
                        conditioning_image = Image.open(char_p).convert('RGB')
                        conditioning_path_hint = char_p
                    except Exception as e:
                        self._log(log, f'⚠️ Cannot load character ref: {e}')

            scene_frames = seconds_to_num_frames(scene.seconds, cfg.fps, backend=getattr(eff_cfg, 'backend', 'wan'))
            # Chunk length for this backend (used for seed-lock heuristics)
            try:
                _chunk_frames_scene0 = seconds_to_num_frames(cfg.chunk_seconds, cfg.fps, backend=getattr(eff_cfg, 'backend', 'wan'))
            except Exception:
                _chunk_frames_scene0 = seconds_to_num_frames(cfg.chunk_seconds, cfg.fps, backend='wan')

            # Seed lock within a clip reduces drift/flicker across segments.
            # In strict mode, we always lock seed within the clip.
            lock_seed = bool(strict_coherence) or (bool(getattr(cfg, 'vary_seed_per_segment', False)) and int(scene_frames) > int(_chunk_frames_scene0))
            if lock_seed and log is not None:
                try:
                    self._log(log, '[Seed] Seed locked within this clip to reduce temporal drift.')
                except Exception:
                    pass

            if need_remaining is None:
                need = scene_frames
                first_segment = True
            else:
                need = int(need_remaining)
                need_remaining = None

            # Segment counters (for UI status). Best-effort estimate; OOM caps may increase real segments.
            try:
                _eff_new = max(1, int(_chunk_frames_scene0) - int(eff_overlap))
                seg_total_est = max(1, int(math.ceil(max(0, int(scene_frames) - 1) / float(_eff_new))))
            except Exception:
                seg_total_est = 1
            seg_local = 0
            last_preview_path_for_scene: Optional[str] = None
            live_preview_dir = os.path.join(proj_dir, 'assets', 'live_previews')

            boundary_overlap = None
            boundary_blend = None
            boundary_ease = None
            if prev_scene is not None:
                # Montage transition BETWEEN clips (scene boundary).
                # Global setting lives in ProjectConfig.scene_transition_mode.
                # Per-clip override lives in SceneSpec.transition_mode.
                boundary_blend = str(getattr(scene, "transition_mode", None) or getattr(eff_cfg, 'scene_transition_mode', cfg.scene_transition_mode))
                # Global strict-cut (or ultra isolation): force hard cut at clip boundaries unless the clip explicitly overrides.
                if bool(cut_strict) and getattr(scene, 'transition_mode', None) is None:
                    boundary_blend = 'cut'
                # If continuity was reset (hard_cut / input override), always force a hard cut at boundary.
                if 'force_cut_boundary' in locals() and bool(force_cut_boundary):
                    boundary_blend = 'cut'
                boundary_overlap = int(scene.transition_frames) if scene.transition_frames is not None else int(getattr(eff_cfg, 'scene_transition_frames', cfg.scene_transition_frames))
                if str(boundary_blend) == "cut":
                    boundary_overlap = 0
                boundary_ease = str(scene.transition_ease) if scene.transition_ease is not None else str(getattr(eff_cfg, 'scene_transition_ease', cfg.scene_transition_ease))


            clip_cache_enabled_for_scene = bool(clip_cache_enabled and clip_cache_root_dir and not (resume_from_folder and scene_i == scene_i0))
            # --- Per-clip render cache (optional) ---
            # If a clip with the same effective inputs/settings was already rendered,
            # we reuse its PNG frames and skip generation. This is independent from the
            # init-frame cache; it speeds up iteration dramatically.
            clip_cache_key = None
            clip_assembler = None
            try:
                setattr(scene, '_clip_cache_hit', False)
            except Exception:
                pass
            if clip_cache_enabled_for_scene and clip_cache_root_dir:
                try:
                    # Cache key should change when seed strategy changes.
                    base_seed_for_cache = None
                    try:
                        base_seed_for_cache = int(scene.seed) if scene.seed is not None else int(getattr(eff_cfg, 'base_seed', 0) or 0)
                    except Exception:
                        base_seed_for_cache = None

                    # Capture key generation inputs that affect output but are not already embedded
                    # in prompts / conditioning.
                    _extra = {
                        "lock_seed": bool(lock_seed),
                        "ultra_isolation": bool(ultra_isolation),
                        "vary_seed_per_segment": bool(getattr(eff_cfg, 'vary_seed_per_segment', False)),
                        "seed_stride": 1337,
                        "style_pack": (getattr(scene, 'style_pack', None) or getattr(eff_cfg, 'global_style_pack', None) or ""),
                    }
                    try:
                        # include LoRA stack signature
                        loras = getattr(eff_cfg, 'loras', None) or []
                        lsig = []
                        for l in loras:
                            try:
                                if hasattr(l, 'enabled') and not bool(getattr(l, 'enabled')):
                                    continue
                            except Exception:
                                pass
                            lsig.append({
                                "name": str(getattr(l, 'name', '') or ''),
                                "weight": float(getattr(l, 'weight', 0.0) or 0.0),
                                "t2": bool(getattr(l, 't2', False)),
                                "repo_id": str(getattr(l, 'repo_id', '') or ''),
                                "weight_name": str(getattr(l, 'weight_name', '') or ''),
                                "local_path": str(getattr(l, 'local_path', '') or ''),
                            })
                        _extra["loras"] = lsig
                    except Exception:
                        pass

                    clip_cache_key = compute_clip_cache_key(
                        cfg,
                        scene,
                        effective_backend=str(getattr(eff_cfg, 'backend', 'wan')),
                        effective_model_id=str(getattr(eff_cfg, 'model_id', '')),
                        effective_mode=str(getattr(eff_cfg, 'mode', '')),
                        conditioning_image=conditioning_image,
                        conditioning_path_hint=conditioning_path_hint,
                        init_frame_key_hint=getattr(scene, 'init_frame_path', None) or None,
                        effective_seed=base_seed_for_cache,
                        extra=_extra,
                    )
                except Exception as e:
                    clip_cache_key = None
                    self._log(log, f"[ClipCache] key error: {e}")

            if clip_cache_key and clip_cache_enabled and clip_cache_root_dir:
                try:
                    entry = _clip_cache_load_entry(clip_cache_root_dir, clip_cache_key)
                    if entry is not None:
                        self._log(log, f"[ClipCache] HIT {clip_cache_key[:16]} | scene {scene_i+1}/{len(cfg.scenes)}: {scene.label}")
                        try:
                            setattr(scene, '_clip_cache_hit', True)
                        except Exception:
                            pass
                        # Stream frames in small batches to keep memory bounded.
                        batch: List[np.ndarray] = []
                        files = [p for p in os.listdir(entry.frames_dir) if p.startswith('frame_') and p.endswith('.png')]
                        files.sort()
                        # Emit a quick UI preview from cached frames (best-effort).
                        try:
                            if files:
                                _lp = os.path.join(entry.frames_dir, files[-1])
                                if os.path.exists(_lp):
                                    arr = np.array(Image.open(_lp).convert('RGB'), dtype=np.uint8)
                                    _live_dir = os.path.join(proj_dir, 'assets', 'live_previews')
                                    _prev_out = os.path.join(_live_dir, f"scene_{int(scene_i):03d}_cache.png")
                                    ok = _save_ui_preview_png(arr, _prev_out, max_w=512)
                                    if ok:
                                        try:
                                            scene.preview_frame_path = str(ok)
                                        except Exception:
                                            try:
                                                setattr(scene, 'preview_frame_path', str(ok))
                                            except Exception:
                                                pass
                                        try:
                                            if clip_asset:
                                                clip_asset(int(scene_i), 'preview', str(ok))
                                        except Exception:
                                            pass
                        except Exception:
                            pass
                        used_boundary = int(boundary_overlap or 0) if prev_scene is not None else 0
                        first_batch = True
                        for fn in files:
                            if is_cancelled and is_cancelled():
                                raise RuntimeError("Annulé par l'utilisateur")
                            p = os.path.join(entry.frames_dir, fn)
                            try:
                                batch.append(np.array(Image.open(p).convert('RGB'), dtype=np.uint8))
                            except Exception:
                                continue
                            if len(batch) >= 64:
                                if prev_scene is not None and first_batch:
                                    writer.push_segment(batch, overlap=used_boundary, blend=boundary_blend, ease=boundary_ease)
                                else:
                                    writer.push_segment(batch, overlap=0)
                                first_batch = False
                                batch = []
                        if batch:
                            if prev_scene is not None and first_batch:
                                writer.push_segment(batch, overlap=used_boundary, blend=boundary_blend, ease=boundary_ease)
                            else:
                                writer.push_segment(batch, overlap=0)

                        # Continuity update from cached last frame
                        # (Disabled in ultra isolation mode: no cross-clip state carry.)
                        lf = _clip_cache_load_last_frame(entry)
                        if lf is not None and (not ultra_isolation):
                            conditioning_image = lf
                            ci_path = os.path.join(proj_dir, 'conditioning_last.png')
                            try:
                                conditioning_image.save(ci_path, 'PNG')
                            except Exception:
                                pass

                        # Best-effort progress advance
                        try:
                            chunk_frames_est = seconds_to_num_frames(cfg.chunk_seconds, cfg.fps, backend=getattr(eff_cfg, 'backend', 'wan'))
                            eff_new = max(1, int(chunk_frames_est) - int(overlap))
                            seg_idx += max(1, math.ceil((int(scene_frames) - 1) / eff_new))
                        except Exception:
                            seg_idx += 1

                        if progress_global:
                            progress_global(min(0.999, seg_idx / max(1, approx_segments)), f"Clip cache HIT ({scene.label})")

                        # Notify UI about cache completion.
                        try:
                            if clip_progress:
                                clip_progress(int(scene_i), 1.0, 'cache_hit')
                            if clip_state:
                                clip_state(int(scene_i), 'done', True)
                        except Exception:
                            pass

                        # Persist cache key + stable preview assets for the Studio timeline.
                        try:
                            setattr(scene, '_clip_cache_key', str(clip_cache_key))
                        except Exception:
                            pass
                        try:
                            if getattr(entry, 'last_frame_path', None) and os.path.exists(str(entry.last_frame_path)):
                                try:
                                    scene.preview_frame_path = str(entry.last_frame_path)
                                except Exception:
                                    setattr(scene, 'preview_frame_path', str(entry.last_frame_path))
                                try:
                                    if clip_asset:
                                        clip_asset(int(scene_i), 'preview', str(entry.last_frame_path))
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        # Optional: auto-build a per-clip MP4 preview from the cache entry.
                        try:
                            if bool(getattr(cfg.studio, 'auto_generate_clip_previews', False)):
                                mp4 = _ensure_cache_preview_mp4(entry, cfg, log)
                                if mp4:
                                    try:
                                        scene.proxy_video_path = str(mp4)
                                    except Exception:
                                        setattr(scene, 'proxy_video_path', str(mp4))
                                    try:
                                        if clip_asset:
                                            clip_asset(int(scene_i), 'proxy_video', str(mp4))
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                        prev_scene = scene
                        continue
                except Exception as e:
                    self._log(log, f"[ClipCache] skip (error): {e}")

            # Load pipeline for this clip/backend if needed, and apply LoRAs once loaded
            self._load_pipeline(eff_cfg, log)
            self._apply_wan_scheduler_if_needed(eff_cfg, log)
            self._apply_loras(eff_cfg, log, cache_dir)

            # Build per-clip cache frames (no montage boundary blending)
            if clip_cache_key and clip_cache_enabled and clip_cache_root_dir:
                clip_assembler = ClipAssembler(
                    max_overlap=max_overlap,
                    default_overlap=int(cfg.overlap_frames),
                    default_blend=str(scene.blend_mode or cfg.blend_mode),
                    default_ease=str(cfg.scene_transition_ease),
                )

            # --- T2V improvements ---
            # Long T2V shots drift across segments. If a scene spans multiple segments,
            # generate segment 1 in T2V, then continue in I2V from segment 2 onward using the last frame as anchor.

            hybrid_t2v = str(getattr(eff_cfg, 'mode', '')).upper() == 'T2V' and scene_frames > int(_chunk_frames_scene0)
            hybrid_switched = False
            while need > 0:
                seg_idx += 1

                if hybrid_t2v and (not first_segment) and (not hybrid_switched):
                    # Switch to I2V continuation and anchor with the last frame for coherence.
                    eff_cfg.mode = 'I2V'
                    self._log(log, '[T2V] Switching to I2V continuation (anchor with last frame) for temporal consistency…')
                    self._load_pipeline(eff_cfg, log)
                    self._apply_wan_scheduler_if_needed(eff_cfg, log)
                    self._apply_loras(eff_cfg, log, cache_dir)
                    hybrid_switched = True
                if progress_global:
                    progress_global(min(0.999, (seg_idx - 1) / max(1, approx_segments)),
                                    f"Segment {seg_idx}/{approx_segments} (scene {scene_i+1}/{len(cfg.scenes)})")

                chunk_frames_scene = seconds_to_num_frames(cfg.chunk_seconds, cfg.fps, backend=getattr(eff_cfg, "backend", "wan"))

                if overlap >= chunk_frames_scene:

                    raise ValueError("overlap_frames doit être < chunk_frames (backend-aware).")


                bkend = getattr(eff_cfg, "backend", "wan")

                cap = self._recommend_max_frames_per_call(cfg, eff_cfg, log)

                retry_left = int(getattr(cfg, "oom_retry_max_attempts", 2) or 2)


                while True:

                    num_frames = min(int(chunk_frames_scene), int(need))

                    if cap is not None:

                        num_frames = min(int(num_frames), int(cap))

                    if bkend == "wan":

                        num_frames = _ensure_form_4k1(int(num_frames))

                    elif bkend == "ltx":

                        num_frames = _ensure_form_8k1(int(num_frames))

                    else:

                        num_frames = int(num_frames)


                    self._log(log, f"Gen seg {seg_idx}: {num_frames} frames | scene {scene_i+1} ({scene.label})")

                    try:

                        frames = self.generate_segment(

                            cfg=eff_cfg,

                            scene=scene,

                            num_frames=num_frames,

                            conditioning_image=conditioning_image,

                            seed_offset=(scene_i if lock_seed else seg_idx),

                            progress=(lambda p, m: progress_global(min(0.999, ((seg_idx - 1) + p) / max(1, approx_segments)), m) if progress_global else None),

                            log=log,

                            is_cancelled=is_cancelled,

                        )

                        break

                    except Exception as e:

                        if self._is_oom_error(e) and retry_left > 0:

                            retry_left -= 1

                            try:

                                cuda_cleanup(aggressive=True)

                            except Exception:

                                pass

                            # reduce frames-per-call and retry (more segments, lower peak VRAM)

                            try:

                                new_cap = max(17, int(int(num_frames) * 0.70))

                                if cap is not None:

                                    new_cap = min(new_cap, int(cap))

                                if bkend == "wan":

                                    new_cap = _ensure_form_4k1(int(new_cap))

                                elif bkend == "ltx":

                                    new_cap = _ensure_form_8k1(int(new_cap))

                                cap = int(new_cap)

                            except Exception:

                                cap = max(17, int(int(num_frames) * 0.70))

                            self._log(log, f"[OOM] retry: reducing frames/call -> {cap} (remaining retries={retry_left})")

                            continue

                        raise

                conditioning_image = self._last_frame_image(frames, log)
                ci_path = os.path.join(proj_dir, "conditioning_last.png")
                conditioning_image.save(ci_path, "PNG")

                # Push frames with the right overlap policy.
                # NOTE: FrameWriter overwrites the previous overlap region (it does NOT append duplicates).
                # Therefore, the number of newly-added timeline frames is:
                #   - num_frames                  if there was no prior tail to overlap with
                #   - (num_frames - used_overlap) otherwise
                had_tail = bool(writer._tail)
                if prev_scene is not None and first_segment:
                    used_overlap = int(boundary_overlap or 0)
                    used_overlap = min(used_overlap, max(0, int(num_frames) - 1))
                    writer.push_segment(frames, overlap=used_overlap, blend=boundary_blend, ease=boundary_ease)
                else:
                    # Use the effective overlap/blend (no-overlap strict can override cfg values).
                    used_overlap = int(eff_overlap)
                    used_overlap = min(used_overlap, max(0, int(num_frames) - 1))
                    writer.push_segment(frames, overlap=used_overlap, blend=str(eff_blend), ease=str(cfg.scene_transition_ease))


                # Build per-clip cache frames (no montage boundary blending)
                if clip_assembler is not None:
                    try:
                        used_int = min(int(eff_overlap), max(0, int(num_frames) - 1))
                        clip_assembler.push_segment(frames, overlap=used_int, blend=str(eff_blend), ease=str(cfg.scene_transition_ease))
                    except Exception:
                        pass

                added = int(num_frames) if (not had_tail or used_overlap <= 0) else max(0, (int(num_frames) - int(used_overlap)))
                if added <= 0: added = 1
                need -= added
                first_segment = False

                # Segment completed: emit a live UI preview frame (best-effort, non-fatal).
                seg_local += 1
                try:
                    if isinstance(frames, list) and frames:
                        prev_path = last_preview_path_for_scene
                        preview_path = os.path.join(live_preview_dir, f"scene_{int(scene_i):03d}_seg_{int(seg_local):03d}.png")
                        ok = _save_ui_preview_png(frames[-1], preview_path, max_w=512)
                        if ok:
                            # remove previous preview to avoid accumulating files
                            try:
                                if prev_path and prev_path != ok and os.path.exists(prev_path) and os.path.dirname(os.path.abspath(prev_path)) == os.path.abspath(live_preview_dir):
                                    os.remove(prev_path)
                            except Exception:
                                pass
                            last_preview_path_for_scene = ok
                            try:
                                scene.preview_frame_path = str(ok)
                            except Exception:
                                try:
                                    setattr(scene, 'preview_frame_path', str(ok))
                                except Exception:
                                    pass
                            try:
                                if clip_asset:
                                    clip_asset(int(scene_i), 'preview', str(ok))
                            except Exception:
                                pass
                except Exception:
                    pass

                # Per-clip progress (0..1) for UI timeline.
                try:
                    if clip_progress:
                        denom = max(1, int(scene_frames) - 1)
                        done_frames = denom - max(0, int(need))
                        p01 = float(done_frames) / float(denom)
                        if p01 < 0.0:
                            p01 = 0.0
                        if p01 > 0.999:
                            p01 = 0.999
                        clip_progress(int(scene_i), float(p01), f"seg {int(seg_local)}/{int(seg_total_est)}")
                except Exception:
                    pass

                if cfg.studio.crash_safe_resume:
                    self._write_state(state_path, {
                        "scene_i": scene_i,
                        "seg_idx": seg_idx,
                        "need_remaining": max(0, need),
                        "first_segment": first_segment,
                        "prev_scene": scene.to_dict(),
                        "conditioning_image_path": ci_path,
                        "last_frame_index": writer.frame_index - 1,
                    })



            # Save per-clip cache on successful generation
            if clip_cache_key and clip_cache_enabled and clip_cache_root_dir and clip_assembler is not None:
                try:
                    if getattr(clip_assembler, 'frames', None):
                        entry = _clip_cache_save_clip(clip_cache_root_dir, clip_cache_key, clip_assembler.frames, fps=int(cfg.fps), extra_meta={
                            'scene_label': str(getattr(scene, 'label', '')),
                            'scene_i': int(scene_i),
                            'backend': str(getattr(eff_cfg, 'backend', 'wan')),
                            'model_id': str(getattr(eff_cfg, 'model_id', '')),
                            'mode': str(getattr(eff_cfg, 'mode', '')),
                        })
                        self._log(log, f"[ClipCache] SAVED {clip_cache_key[:16]} ({len(clip_assembler.frames)} frames) | scene {scene_i+1}/{len(cfg.scenes)}")

                        # Persist cache key + stable preview assets for the Studio timeline.
                        try:
                            setattr(scene, '_clip_cache_key', str(clip_cache_key))
                        except Exception:
                            pass
                        try:
                            if getattr(entry, 'last_frame_path', None) and os.path.exists(str(entry.last_frame_path)):
                                try:
                                    scene.preview_frame_path = str(entry.last_frame_path)
                                except Exception:
                                    setattr(scene, 'preview_frame_path', str(entry.last_frame_path))
                                try:
                                    if clip_asset:
                                        clip_asset(int(scene_i), 'preview', str(entry.last_frame_path))
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        # Optional: auto-build a per-clip MP4 preview from the cache entry.
                        try:
                            if bool(getattr(cfg.studio, 'auto_generate_clip_previews', False)):
                                mp4 = _ensure_cache_preview_mp4(entry, cfg, log)
                                if mp4:
                                    try:
                                        scene.proxy_video_path = str(mp4)
                                    except Exception:
                                        setattr(scene, 'proxy_video_path', str(mp4))
                                    try:
                                        if clip_asset:
                                            clip_asset(int(scene_i), 'proxy_video', str(mp4))
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                except Exception as e:
                    self._log(log, f"[ClipCache] save failed: {e}")

            # Notify UI (best-effort) that this clip completed.
            try:
                if clip_progress:
                    clip_progress(int(scene_i), 1.0, 'done')
                if clip_state:
                    clip_state(int(scene_i), 'done', False)
            except Exception:
                pass

            prev_scene = scene

        last_frame_index = writer.frame_index - 1
        thumbs: Dict[str, str] = {}
        if cfg.studio.write_report:
            thumbs = build_thumbnails(frames_dir, os.path.join(proj_dir, "report_assets"), last_frame_index)

        raw_mp4 = os.path.join(proj_dir, "raw.mp4")

        # --- Frame safety (V24) ---
        # Rarely, diffusion pipelines can output a single "glitch" frame (psychedelic/warped)
        # that stands out between two normal frames. In cinema workflows this is very noticeable.
        # When enabled, we conservatively repair such isolated frames by blending neighbors.
        if bool(getattr(cfg, "frame_safety_enabled", False)):
            try:
                from .frame_safety import repair_glitch_frames
                repaired = repair_glitch_frames(
                    frames_dir,
                    threshold_factor=float(getattr(cfg, "frame_safety_threshold_factor", 10.0)),
                    neighbor_similarity_factor=float(getattr(cfg, "frame_safety_neighbor_similarity_factor", 0.35)),
                    log=(lambda msg: self._log(log, msg)) if log is not None else None,
                )
                if repaired and log is not None:
                    self._log(log, f"[FrameSafety] repaired {int(repaired)} frame(s) before encoding")
            except Exception as e:
                # Best-effort only
                if log is not None:
                    self._log(log, f"[FrameSafety] skipped (error): {e}")

        self._log(log, "Encodage raw.mp4...")
        encode_png_sequence_to_mp4(frames_dir, raw_mp4, fps=cfg.fps, crf=18, preset="slow")

        outputs: Dict[str, str] = {"raw": raw_mp4}
        out_path = raw_mp4

        if cfg.post.enabled:
            self._log(log, "Post-prod premium...")
            out_pp_dir = os.path.join(proj_dir, "post")
            pp_cfg = PostProcessConfig(
                enabled=True,
                tools_dir=cfg.post.tools_dir,
                use_rife_if_available=cfg.post.use_rife_if_available,
                use_realesrgan_if_available=cfg.post.use_realesrgan_if_available,
                fps_in=int(cfg.fps),
                target_fps=int(cfg.post.target_fps),
                out_width=int(cfg.post.out_width),
                out_height=int(cfg.post.out_height),
                deflicker=bool(cfg.post.deflicker),
                denoise=bool(cfg.post.denoise),
                denoise_strength=(float(cfg.post.denoise_luma), float(cfg.post.denoise_chroma),
                                  float(cfg.post.denoise_temporal), float(cfg.post.denoise_temporal_chroma)),
                sharpen=bool(cfg.post.sharpen),
                unsharp=(int(cfg.post.unsharp_mx), int(cfg.post.unsharp_my), float(cfg.post.unsharp_amount)),
                export_master_prores=bool(cfg.post.export_master_prores),
                export_delivery_h265_main10=bool(cfg.post.export_delivery_h265_main10),
                h265_crf=int(cfg.post.h265_crf),
                h265_preset=str(cfg.post.h265_preset),
            )
            rep = postprocess_pipeline(raw_mp4, out_pp_dir, cfg=pp_cfg, log=log)
            write_json(os.path.join(proj_dir, "postprocess_report.json"), rep)

            intermediate = rep["exports"]["intermediate"]
            out_fps = int(rep["exports"].get("fps", cfg.fps))
            outputs["intermediate"] = intermediate


            # Choose primary deliver (what the UI will open by default)
            primary = str(getattr(cfg.post, "primary_deliver", "prores") or "prores").lower().strip()

            # Tag exports with target canvas (helps when doing 4K upscales in post)
            try:
                tw = int(getattr(cfg.post, "out_width", 0) or 0)
                th = int(getattr(cfg.post, "out_height", 0) or 0)
            except Exception:
                tw, th = 0, 0
            tag = f"{tw}x{th}" if (tw and th) else "native"

            # Ensure requested primary exists (auto-enable exports if needed)
            if primary == "h265":
                cfg.post.export_delivery_h265_main10 = True
            if primary == "prores":
                cfg.post.export_master_prores = True

            # Exports (may be enabled independently)
            if cfg.post.export_master_prores:
                master = os.path.join(proj_dir, f"master_{tag}_prores.mov")
                self._log(log, "Export ProRes master...")
                encode_prores_master(intermediate, master, fps=out_fps, profile=str(cfg.post.prores_profile))
                outputs["master_prores"] = master

            if cfg.post.export_delivery_h265_main10:
                deliver = os.path.join(proj_dir, f"deliver_{tag}_h265_main10.mp4")
                self._log(log, "Export H.265 Main10 delivery...")
                encode_h265_main10(intermediate, deliver, fps=out_fps, crf=int(cfg.post.h265_crf), preset=str(cfg.post.h265_preset))
                outputs["deliver_h265_main10"] = deliver

            # Primary output path
            if primary == "h265" and outputs.get("deliver_h265_main10"):
                out_path = outputs["deliver_h265_main10"]
            elif primary == "intermediate":
                out_path = intermediate
            else:
                out_path = outputs.get("master_prores", intermediate)

            # Preview-friendly media path (Qt plays mp4 more reliably than ProRes on some systems)
            outputs["preview_media"] = outputs.get("deliver_h265_main10", intermediate)

        if cfg.studio.write_report:
            report_path = create_html_report(cfg, proj_dir, outputs=outputs, thumbs=thumbs, extra={"last_frame_index": last_frame_index})
            outputs["report"] = report_path

        if cfg.studio.crash_safe_resume:
            try:
                self._write_state(state_path, {"completed": True, "outputs": outputs, "last_frame_index": last_frame_index})
            except Exception:
                pass

        if not cfg.keep_png_frames:
            self._log(log, "Nettoyage frames PNG...")
            shutil.rmtree(frames_dir, ignore_errors=True)

        if progress_global:
            progress_global(1.0, "Terminé")

        return out_path

    def resume_from_folder(
        self,
        folder: str,
        progress_global: Optional[ProgressCallback],
        log: Optional[LogCallback],
        is_cancelled: Optional[Callable[[], bool]],
        clip_state: Optional[ClipStateCallback] = None,
        clip_progress: Optional[ClipProgressCallback] = None,
        clip_asset: Optional[ClipAssetCallback] = None,
    ) -> str:
        folder = os.path.abspath(folder)
        proj_json = os.path.join(folder, "project.json")
        if not os.path.exists(proj_json):
            raise ValueError("Dossier invalide: project.json manquant.")
        cfg = ProjectConfig.load(proj_json)
        return self.render_once(
            cfg,
            progress_global,
            log,
            is_cancelled,
            clip_state=clip_state,
            clip_progress=clip_progress,
            clip_asset=clip_asset,
            resume_from=folder,
        )



    def _generate_segment_lingbot(
        self,
        cfg: ProjectConfig,
        scene: SceneSpec,
        num_frames: int,
        conditioning_image: Optional[Image.Image],
        seed_offset: int,
        progress: Optional[ProgressCallback],
        log: Optional[LogCallback],
        is_cancelled: Optional[Callable[[], bool]],
    ) -> List[np.ndarray]:
        """Generate frames using LingBot-World (base-cam) in a subprocess."""
        import tempfile
        import os
        from PIL import Image as _PILImage

        # Prompt injection (best-effort)
        try:
            from .cinema_prompts import build_scene_prompts
            p_inj, _n_inj = build_scene_prompts(cfg, scene)
            prompt = (p_inj or scene.prompt or getattr(cfg, "prompt", "") or "").strip()
        except Exception:
            prompt = (scene.prompt or getattr(cfg, "prompt", "") or "").strip()

        if not prompt:
            prompt = "Cinematic shot, controlled camera motion."

        # Seed logic consistent with other backends
        base_seed = scene.seed if scene.seed is not None else cfg.base_seed
        seed = int(base_seed) + (int(seed_offset) * 1337 if cfg.vary_seed_per_segment else 0)

        ling = getattr(cfg, "lingbot", None)
        if ling is None or not getattr(ling, "enabled", True):
            raise RuntimeError("LingBot backend selected but cfg.lingbot.enabled is False")

        internal_fps = int(getattr(ling, "fps_internal", 16) or 16)
        seconds = float(num_frames) / float(getattr(cfg, "fps", 24) or 24)
        internal_frames = int(round(seconds * float(internal_fps)))
        from .camera_paths import ensure_4n1
        internal_frames = ensure_4n1(internal_frames)

        # Prepare init image (force 832x480 landscape for base-cam)
        W0, H0 = 832, 480
        if conditioning_image is None:
            try:
                from .flux2_init import generate_init_image_for_scene
                res = generate_init_image_for_scene(scene, cfg, log=log)
                conditioning_image = _PILImage.open(res.path).convert("RGB")
            except Exception as e:
                raise RuntimeError(f"LingBot requires an init image. Provide an image or enable Auto Init-Frame. ({e})")

        img = conditioning_image.convert("RGB") if hasattr(conditioning_image, "convert") else conditioning_image

        # center-crop + resize to exact W0xH0
        try:
            iw, ih = img.size
            target_ar = W0 / H0
            cur_ar = iw / ih
            if cur_ar > target_ar:
                new_w = int(ih * target_ar)
                x0 = max(0, (iw - new_w) // 2)
                img = img.crop((x0, 0, x0 + new_w, ih))
            elif cur_ar < target_ar:
                new_h = int(iw / target_ar)
                y0 = max(0, (ih - new_h) // 2)
                img = img.crop((0, y0, iw, y0 + new_h))
            img = img.resize((W0, H0), resample=_PILImage.Resampling.LANCZOS)
        except Exception:
            try:
                img = img.resize((W0, H0))
            except Exception:
                pass

        preset = (getattr(scene, "camera_path_preset", None) or "static").strip()
        fov = float(getattr(scene, "camera_fov_deg", 50.0) or 50.0)
        amount = float(getattr(scene, "camera_path_amount", 1.0) or 1.0)
        strength = float(getattr(scene, "camera_path_strength", 1.0) or 1.0)

        tmp = tempfile.mkdtemp(prefix="wan_lingbot_")
        try:
            init_path = os.path.join(tmp, "init.png")
            img.save(init_path)

            action_dir = os.path.join(tmp, "action_path")
            from .camera_paths import write_action_path
            write_action_path(action_dir, preset=preset, frame_num=internal_frames, width=W0, height=H0, fov_deg=fov, amount=amount, strength=strength)

            out_mp4 = os.path.join(tmp, "lingbot_out.mp4")

            if progress:
                progress(0.02, "LingBot: génération…")

            from .lingbot_runner import run_lingbot_generate, decode_mp4_to_frames, resample_frames
            third_party_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "third_party"))
            run_lingbot_generate(
                third_party_dir=third_party_dir,
                ckpt_dir=getattr(ling, "ckpt_dir", ""),
                prompt=prompt,
                image_path=init_path,
                action_path=action_dir,
                size=str(getattr(ling, "size", "832*480") or "832*480"),
                frame_num=int(internal_frames),
                sample_steps=int(getattr(ling, "sample_steps", 24) or 24),
                sample_shift=float(getattr(ling, "sample_shift", 3.0) or 3.0),
                guide_scale=float(getattr(ling, "sample_guide_scale", 5.0) or 5.0),
                seed=int(seed),
                offload_model=bool(getattr(ling, "offload_model", True)),
                t5_cpu=bool(getattr(ling, "t5_cpu", True)),
                ulysses_size=int(getattr(ling, "ulysses_size", 1) or 1),
                save_file=out_mp4,
                log=log,
                is_cancelled=is_cancelled,
            )

            if progress:
                progress(0.85, "LingBot: décodage…")

            frames = decode_mp4_to_frames(out_mp4, log=log)
            frames = resample_frames(frames, int(num_frames))

            if progress:
                progress(0.98, "LingBot: ok")

            return [self._to_uint8(fr) for fr in frames]
        finally:
            try:
                import shutil
                shutil.rmtree(tmp, ignore_errors=True)
            except Exception:
                pass

    def render_project(
        self,
        cfg: ProjectConfig,
        progress_global: Optional[ProgressCallback],
        log: Optional[LogCallback],
        is_cancelled: Optional[Callable[[], bool]],
        clip_state: Optional[ClipStateCallback] = None,
        clip_progress: Optional[ClipProgressCallback] = None,
        clip_asset: Optional[ClipAssetCallback] = None,
    ) -> List[str]:
        outs: List[str] = []
        # Enforce crash-proof settings at runtime when enabled.
        cfg = apply_cinema_safe_overrides(cfg, log=log, where='render_project')
        if not cfg.batch.enabled or cfg.batch.variants <= 1:
            try:
                outs.append(self.render_once(cfg, progress_global, log, is_cancelled, clip_state=clip_state, clip_progress=clip_progress, clip_asset=clip_asset))
            except Exception:
                self.unload(log)
                raise
            return outs

        base = int(cfg.base_seed)
        for i in range(int(cfg.batch.variants)):
            if is_cancelled and is_cancelled():
                raise RuntimeError("Annulé par l'utilisateur")
            cfg2 = ProjectConfig.from_dict(cfg.to_dict())
            cfg2.base_seed = base + i * int(cfg.batch.seed_step)
            suffix = f"_{cfg.batch.suffix}{i+1}"
            if progress_global:
                progress_global(0.0, f"Variant {i+1}/{cfg.batch.variants}")
            try:
                outs.append(self.render_once(cfg2, progress_global, log, is_cancelled, clip_state=clip_state, clip_progress=clip_progress, clip_asset=clip_asset, run_name_suffix=suffix))
            except Exception:
                self.unload(log)
                raise
        return outs
