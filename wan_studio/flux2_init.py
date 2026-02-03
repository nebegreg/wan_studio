from __future__ import annotations

import hashlib
import inspect
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from .config import ProjectConfig, SceneSpec
from .init_frame_subprocess import resolve_hf_cache_dir, resolve_model_overrides, build_job, run_init_frame_job, clamp_to_multiple_of_8
from .cinema_prompts import build_scene_prompts
from .init_frame_cache import CacheItem, load_cache, save_cache
from .resource_manager import get_resource_manager, cuda_cleanup


@dataclass
class InitGenResult:
    path: str
    cache_key: str
    cache_hit: bool


@dataclass
class InitCacheStatus:
    cache_key: str
    cached_path: Optional[str]
    exists: bool


@dataclass
class RefBundle:
    """Separated reference image groups (best-effort, adapter-aware)."""

    character_refs: List[str]
    location_refs: List[str]
    style_refs: List[str]
    extra_refs: List[str]

    @property
    def all_refs(self) -> List[str]:
        return _dedup_paths(list(self.character_refs) + list(self.location_refs) + list(self.style_refs) + list(self.extra_refs))


def _log(log, msg: str):
    try:
        if callable(log):
            log(msg)
    except Exception:
        pass


def _safe_abspath(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _resolve_model_id_for_preset(init_cfg: Optional[object], preset: str) -> str:
    overrides = resolve_model_overrides(init_cfg)
    preset = str(preset or "").strip()
    env_map = {
        "flux2_bnb4bit": "WAN_INIT_FLUX2_BNB4BIT_MODEL",
        "flux2": "WAN_INIT_FLUX2_MODEL",
        "zimage_turbo": "WAN_INIT_ZIMAGE_TURBO_MODEL",
        "zimage": "WAN_INIT_ZIMAGE_MODEL",
        "sdxl_base": "WAN_INIT_SDXL_BASE_MODEL",
        "sdxl_turbo": "WAN_INIT_SDXL_TURBO_MODEL",
        "sdxl_lightning_4step": "WAN_INIT_SDXL_BASE_MODEL",
        "flux1_schnell": "WAN_INIT_FLUX1_SCHNELL_MODEL",
    }
    override_map = {
        "flux2_bnb4bit": "flux2_bnb4bit_model_id",
        "flux2": "flux2_model_id",
        "zimage_turbo": "zimage_turbo_model_id",
        "zimage": "zimage_model_id",
        "sdxl_base": "sdxl_base_model_id",
        "sdxl_turbo": "sdxl_turbo_model_id",
        "sdxl_lightning_4step": "sdxl_base_model_id",
        "flux1_schnell": "flux1_schnell_model_id",
    }
    ov_key = override_map.get(preset)
    if ov_key and overrides.get(ov_key):
        return overrides[ov_key]
    env_key = env_map.get(preset)
    if env_key:
        return os.environ.get(env_key, "").strip()
    return ""


def _project_root(cfg: ProjectConfig) -> str:
    return _safe_abspath(os.path.join(cfg.output_dir, cfg.project_name))


def _resolve_cache_dirs(cfg: ProjectConfig) -> Tuple[str, str]:
    root = _project_root(cfg)
    init_cfg = getattr(cfg, "init_image", None)
    if init_cfg and getattr(init_cfg, "cache_location", "assets") == "project":
        cache_root = os.path.join(root, getattr(init_cfg, "project_cache_dirname", ".wan_cache"))
        img_dir = os.path.join(cache_root, "init_frames")
        cache_json = os.path.join(cache_root, "init_frames_cache.json")
    else:
        img_dir = os.path.join(root, "assets", "init_frames")
        cache_json = os.path.join(img_dir, "init_frames_cache.json")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cache_json), exist_ok=True)
    return img_dir, cache_json


def get_init_cache_dirs(cfg: ProjectConfig) -> Tuple[str, str]:
    """Public helper for UI/tools.

    Returns (init_frames_dir, cache_json_path) based on cfg.init_image.cache_location.
    """
    return _resolve_cache_dirs(cfg)


def _hash_file_hint(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"
    except Exception:
        return os.path.abspath(path)


def _compute_key(parts: Sequence[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def _dedup_paths(paths: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for p in paths:
        if not p:
            continue
        ap = _safe_abspath(p)
        if ap in seen:
            continue
        if not os.path.exists(ap):
            continue
        seen.add(ap)
        out.append(ap)
    return out


def _effective_init_preset(scene: SceneSpec, cfg: ProjectConfig, preset_override: Optional[str] = None) -> str:
    init_cfg = getattr(cfg, "init_image", None)
    if preset_override:
        return str(preset_override).strip() or "zimage_turbo"
    sc = (getattr(scene, "init_preset_override", None) or "").strip()
    if sc:
        return sc
    return str(getattr(init_cfg, "preset", "zimage_turbo") or "zimage_turbo").strip()


def _effective_init_policy(scene: SceneSpec, cfg: ProjectConfig, policy_override: Optional[str] = None) -> str:
    init_cfg = getattr(cfg, "init_image", None)
    if policy_override:
        return str(policy_override).strip() or "necessary"
    sc = (getattr(scene, "init_policy_override", None) or "").strip()
    if sc:
        return sc
    return str(getattr(init_cfg, "policy", "necessary") or "necessary").strip()


def _gather_refs(
    scene: SceneSpec,
    cfg: ProjectConfig,
    *,
    last_location_memory_path: Optional[str] = None,
) -> RefBundle:
    init_cfg = getattr(cfg, "init_image", None)
    character_refs: List[str] = []
    location_refs: List[str] = []
    style_refs: List[str] = []
    extra_refs: List[str] = []

    # Character refs from bible
    for nm in (getattr(scene, "characters", []) or []):
        c = _find_character_spec(cfg, nm)
        if c:
            character_refs.extend(list(getattr(c, "ref_images", []) or []))

    # Location refs from library
    loc = (getattr(scene, "location", "") or "").strip()
    if loc:
        l = _find_location_spec(cfg, loc)
        if l:
            location_refs.extend(list(getattr(l, "ref_images", []) or []))

    # Global style refs
    try:
        style_refs.extend(list(getattr(init_cfg, "style_ref_images", []) or []))
    except Exception:
        pass

    # Optional global character ref (if enabled for this scene)
    try:
        if getattr(scene, "use_character_ref", False):
            cp = getattr(cfg, "character_image_path", None)
            if cp:
                character_refs.append(cp)
    except Exception:
        pass

    # Location memory
    if getattr(init_cfg, "use_last_location_memory", True) and last_location_memory_path:
        location_refs.append(last_location_memory_path)

    # Existing image as reference
    if getattr(init_cfg, "use_existing_as_reference", True):
        for rp in [
            getattr(scene, "init_frame_path", None),
            getattr(scene, "input_image_path_override", None),
            getattr(cfg, "input_image_path", None),
        ]:
            if rp:
                extra_refs.append(rp)

    # De-dupe + clamp total refs
    character_refs = _dedup_paths(character_refs)
    location_refs = _dedup_paths(location_refs)
    style_refs = _dedup_paths(style_refs)
    extra_refs = _dedup_paths(extra_refs)

    max_refs = int(getattr(init_cfg, "max_refs", 6) or 6)
    all_refs = _dedup_paths(character_refs + location_refs + style_refs + extra_refs)
    if len(all_refs) > max_refs:
        # Preserve group priority: character > location > style > extra
        kept: List[str] = []
        for group in (character_refs, location_refs, style_refs, extra_refs):
            for p in group:
                if p in kept:
                    continue
                if len(kept) >= max_refs:
                    break
                kept.append(p)
            if len(kept) >= max_refs:
                break
        # Rebuild groups filtered by kept
        character_refs = [p for p in character_refs if p in kept]
        location_refs = [p for p in location_refs if p in kept]
        style_refs = [p for p in style_refs if p in kept]
        extra_refs = [p for p in extra_refs if p in kept]

    return RefBundle(
        character_refs=character_refs,
        location_refs=location_refs,
        style_refs=style_refs,
        extra_refs=extra_refs,
    )


def compute_init_cache_key(
    scene: SceneSpec,
    cfg: ProjectConfig,
    *,
    preset_override: Optional[str] = None,
    policy_override: Optional[str] = None,
    last_location_memory_path: Optional[str] = None,
    log=None,
) -> Tuple[str, RefBundle, str, str, int, int, int, float]:
    """Return (key, refs, preset, policy, w, h, steps, guidance_scale)."""
    init_cfg = getattr(cfg, "init_image", None)
    if not init_cfg or not getattr(init_cfg, "enabled", True):
        raise RuntimeError("Init image generation is disabled")

    prompt, negative = build_scene_prompts(cfg, scene)
    # Width/height: when init_cfg.width/height are 0, follow the project base resolution.
    w = int(getattr(init_cfg, "width", 0) or getattr(cfg, "width", 1024) or 1024)
    h = int(getattr(init_cfg, "height", 0) or getattr(cfg, "height", 1024) or 1024)
    # Align to multiples of 8 (common requirement across diffusion video/image models).
    w = max(256, (w // 8) * 8)
    h = max(256, (h // 8) * 8)
    steps = int(getattr(init_cfg, "steps", 28))
    gs = float(getattr(init_cfg, "guidance_scale", 4.0))
    preset = _effective_init_preset(scene, cfg, preset_override)
    policy = _effective_init_policy(scene, cfg, policy_override)

    model_id = _resolve_model_id_for_preset(init_cfg, preset)

    # Cinema SAFE mode: cap Flux2 init resolution/steps to avoid CUDA OOM (preserve aspect ratio).
    # Note: this affects ONLY init-frame generation (conditioning), not the final render resolution.
    try:
        if bool(getattr(cfg, 'cinema_safe_mode', False)) and str(preset).startswith('flux2'):
            # Dynamic OOM guard based on *current* free VRAM.
            free_gb = 0.0
            try:
                import torch
                if torch.cuda.is_available():
                    free, _total = torch.cuda.mem_get_info()
                    free_gb = float(free) / (1024**3)
            except Exception:
                free_gb = 0.0

            # Conservative step/guidance caps: init frames don't need huge step counts.
            steps = min(int(steps), 18 if free_gb >= 16.0 else 14)
            gs = min(float(gs), 5.0)

            # Area caps tuned for Flux2 init-frame stability:
            # 960x544 ≈ 522k pixels (good target for 24GB). Bigger often OOMs under Flux2.
            if free_gb >= 20.0:
                max_area = 520_000
            elif free_gb >= 16.0:
                max_area = 420_000
            elif free_gb >= 12.0:
                max_area = 320_000
            else:
                max_area = 240_000

            area = int(w) * int(h)
            if area > int(max_area):
                import math
                scale = math.sqrt(float(max_area) / float(max(1, area)))
                w2 = max(256, int((w * scale) // 8) * 8)
                h2 = max(256, int((h * scale) // 8) * 8)
                if log:
                    try:
                        log(f"[SAFE] init caps: {w}x{h} -> {w2}x{h2} (Flux2 VRAM guard, free≈{free_gb:.1f}GB)")
                    except Exception:
                        pass
                w, h = w2, h2
    except Exception:
        # Never fail init cache key computation because of safety heuristics.
        pass

    refs = _gather_refs(scene, cfg, last_location_memory_path=last_location_memory_path)

    key_parts = [
        "preset=" + preset,
        f"{w}x{h}",
        f"steps={steps}",
        f"gs={gs}",
        "prompt=" + (prompt or ""),
        "neg=" + (negative or ""),
    ]
    if model_id:
        key_parts.append("model_id=" + model_id)
    # Flags that affect the generated image MUST be part of the cache key.
    try:
        key_parts.append(f"ip_adapter={bool(getattr(init_cfg,'enable_ip_adapter',False))}")
        key_parts.append(f"ip_model={str(getattr(init_cfg,'ip_adapter_model_id','') or '')}")
        key_parts.append(f"ip_weight={str(getattr(init_cfg,'ip_adapter_weight_name','') or '')}")
        key_parts.append(f"ip_c={float(getattr(init_cfg,'ip_adapter_scale_character',0.8) or 0.8):.3f}")
        key_parts.append(f"ip_l={float(getattr(init_cfg,'ip_adapter_scale_location',0.6) or 0.6):.3f}")
        key_parts.append(f"ip_s={float(getattr(init_cfg,'ip_adapter_scale_style',0.5) or 0.5):.3f}")
    except Exception:
        pass
    try:
        key_parts.append(f"controlnet={bool(getattr(init_cfg,'enable_controlnet',False))}")
        key_parts.append(f"cn_type={str(getattr(init_cfg,'controlnet_type','') or '')}")
        key_parts.append(f"cn_id={str(getattr(init_cfg,'controlnet_model_id','') or '')}")
        key_parts.append(f"cn_scale={float(getattr(init_cfg,'controlnet_scale',0.75) or 0.75):.3f}")
    except Exception:
        pass
    try:
        key_parts.append(f"refine_strength={float(getattr(init_cfg,'refine_strength',0.35) or 0.35):.3f}")
    except Exception:
        pass
    for r in refs.character_refs:
        key_parts.append("char_ref=" + _hash_file_hint(r))
    for r in refs.location_refs:
        key_parts.append("loc_ref=" + _hash_file_hint(r))
    for r in refs.style_refs:
        key_parts.append("style_ref=" + _hash_file_hint(r))
    for r in refs.extra_refs:
        key_parts.append("ref=" + _hash_file_hint(r))

    key = _compute_key(key_parts)
    return key, refs, preset, policy, w, h, steps, gs


def get_init_cache_status(
    scene: SceneSpec,
    cfg: ProjectConfig,
    *,
    preset_override: Optional[str] = None,
    policy_override: Optional[str] = None,
    last_location_memory_path: Optional[str] = None,
) -> InitCacheStatus:
    img_dir, cache_json = _resolve_cache_dirs(cfg)
    key, _refs, _preset, _policy, _w, _h, _steps, _gs = compute_init_cache_key(
        scene,
        cfg,
        preset_override=preset_override,
        policy_override=policy_override,
        last_location_memory_path=last_location_memory_path,
        log=log,
)

    out_path = os.path.join(img_dir, f"init_{key[:16]}.png")
    cache = load_cache(cache_json)
    item = cache.get(key)
    cand = None
    if item and getattr(item, "path", None):
        cand = str(item.path)
    if cand and os.path.exists(cand):
        return InitCacheStatus(cache_key=key, cached_path=cand, exists=True)
    if os.path.exists(out_path):
        return InitCacheStatus(cache_key=key, cached_path=os.path.abspath(out_path), exists=True)
    return InitCacheStatus(cache_key=key, cached_path=cand or os.path.abspath(out_path), exists=False)


def try_use_cached_init_frame(
    scene: SceneSpec,
    cfg: ProjectConfig,
    *,
    preset_override: Optional[str] = None,
    last_location_memory_path: Optional[str] = None,
    log=None,
) -> InitGenResult:
    """Force cache usage for this call if a cached init frame exists."""
    st = get_init_cache_status(
        scene,
        cfg,
        preset_override=preset_override,
        policy_override="necessary",
        last_location_memory_path=last_location_memory_path,
    )
    if st.exists and st.cached_path and os.path.exists(st.cached_path):
        scene.init_frame_path = os.path.abspath(st.cached_path)
        scene.init_frame_hash = st.cache_key
        _log(log, f"[InitFrame] cache HIT -> {scene.init_frame_path}")
        return InitGenResult(path=scene.init_frame_path, cache_key=st.cache_key, cache_hit=True)
    raise FileNotFoundError("Init frame cache miss")


def _find_character_spec(cfg: ProjectConfig, name: str):
    n = (name or "").strip().lower()
    for c in getattr(cfg, "characters", []) or []:
        if (getattr(c, "name", "") or "").strip().lower() == n:
            return c
    return None


def _find_location_spec(cfg: ProjectConfig, name: str):
    n = (name or "").strip().lower()
    for l in getattr(cfg, "locations", []) or []:
        if (getattr(l, "name", "") or "").strip().lower() == n:
            return l
    return None


_PIPE = None


def _has_meta_tensor_error(e: BaseException) -> bool:
    s = str(e).lower()
    return ("meta tensor" in s) or ("copy out of meta" in s) or ("cannot copy out of meta" in s)


def _patch_meta_encode_prompt(pipe, safe_device, log=None) -> None:
    """Patch encode_prompt to avoid passing device=meta under accelerate offload.

    Some diffusers/accelerate combinations can temporarily put submodules on the
    `meta` device. If the pipeline then tries to move *inputs* to meta, PyTorch
    errors with: "Cannot copy out of meta tensor; no data!".
    """
    try:
        import types
        import torch

        if not hasattr(pipe, "encode_prompt"):
            return
        if getattr(pipe, "_ws_patch_meta_encode_prompt", False):
            return

        orig = pipe.encode_prompt

        def _wrapped(self, *args, **kwargs):
            try:
                # kw device
                if "device" in kwargs:
                    dev = kwargs.get("device")
                    if isinstance(dev, torch.device) and dev.type == "meta":
                        kwargs["device"] = safe_device
                # positional device is commonly 2nd arg
                elif len(args) >= 2:
                    dev = args[1]
                    if isinstance(dev, torch.device) and dev.type == "meta":
                        aa = list(args)
                        aa[1] = safe_device
                        args = tuple(aa)
            except Exception:
                pass
            return orig(*args, **kwargs)

        pipe.encode_prompt = types.MethodType(_wrapped, pipe)
        setattr(pipe, "_ws_patch_meta_encode_prompt", True)
        _log(log, "[Flux2] Patched encode_prompt: device meta→cuda (avoids meta-tensor crash under offload).")
    except Exception:
        return




def _is_oom_error(e: BaseException) -> bool:
    try:
        import torch
        if isinstance(e, getattr(torch.cuda, 'OutOfMemoryError', ())):
            return True
    except Exception:
        pass
    msg = str(e).lower()
    return ('out of memory' in msg) or ('cuda oom' in msg) or ('cublas' in msg and 'alloc' in msg)


def _is_device_mismatch_error(e: BaseException) -> bool:
    """Heuristic for CPU/CUDA mixing errors.

    Seen with some accelerate/offload combinations when Flux2 is partially on CPU.
    """
    msg = str(e).lower()
    return (
        ('expected all tensors to be on the same device' in msg)
        or ('found at least two devices' in msg)
        or ('cuda:0 and cpu' in msg)
        or ('cpu and cuda' in msg)
    )

def unload_flux2_pipe(log=None) -> None:
    """Force unload the global Flux2 pipeline to free VRAM."""
    global _PIPE
    try:
        _PIPE = None
    except Exception:
        pass
    try:
        cuda_cleanup()
    except Exception:
        pass
    try:
        rm = get_resource_manager()
        rm.unregister("flux2_pipe")
    except Exception:
        pass


def _filter_kwargs(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs


def _load_flux2_pipe(log=None, *, force_cuda: bool = False):
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    _log(log, "[Flux2] loading FLUX.2-dev-bnb-4bit (first time)...")

    import torch

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    try:
        from diffusers import Flux2Pipeline, Flux2Transformer2DModel
    except Exception as e:
        raise RuntimeError(f"[Flux2] diffusers Flux2Pipeline not available: {e}")

    text_encoder_cls = None
    try:
        from transformers import Mistral3ForConditionalGeneration as _M3
        text_encoder_cls = _M3
    except Exception:
        try:
            from transformers import MistralForConditionalGeneration as _M
            text_encoder_cls = _M
        except Exception:
            text_encoder_cls = None

    # Base (VAE + config) comes from the official repo; quantized weights live in diffusers org.
    # Note: FLUX.2-dev is gated on Hugging Face. Users must accept the license and login via `hf auth login`.
    model_id = os.getenv("WAN_STUDIO_FLUX2_BASE_ID", "black-forest-labs/FLUX.2-dev")
    transformer_id = os.getenv("WAN_STUDIO_FLUX2_TRANSFORMER_ID", "diffusers/FLUX.2-dev-bnb-4bit")
    transformer_subfolder = os.getenv("WAN_STUDIO_FLUX2_TRANSFORMER_SUBFOLDER", "transformer")

    # IMPORTANT:
    # Some accelerate/device_map combos can leave modules or *outputs* on the `meta` device
    # under offload, which later crashes with:
    #   "Cannot copy out of meta tensor; no data!"
    # We therefore default to a "safe" load path:
    #   - load weights normally (no device_map)
    #   - move the full pipeline to CUDA when available
    #   - only fall back to offload on OOM
    use_cuda = torch.cuda.is_available()
    try:
        dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else (torch.float16 if use_cuda else torch.float32)
    except Exception:
        dtype = torch.float16 if use_cuda else torch.float32

    transformer = Flux2Transformer2DModel.from_pretrained(
        transformer_id,
        subfolder=str(transformer_subfolder),
        torch_dtype=dtype,
    )

    if text_encoder_cls is None:
        pipe = Flux2Pipeline.from_pretrained(
            model_id,
            transformer=transformer,
            torch_dtype=dtype,
        )
    else:
        text_encoder = text_encoder_cls.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=dtype,
        )
        pipe = Flux2Pipeline.from_pretrained(
            model_id,
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=dtype,
        )

    # Prefer CUDA for stability (avoids meta tensors). Only offload when not forced.
    if use_cuda:
        try:
            pipe.to("cuda")
            # Memory guards (optional depending on pipeline implementation)
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
            try:
                pipe.enable_vae_tiling()
            except Exception:
                pass
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        except Exception:
            # If move fails (e.g., VRAM pressure), we'll try offload below
            # unless force_cuda is requested.
            if force_cuda:
                raise
            pass

    # Keep VRAM usage low (only if we couldn't move to CUDA and not forced).
    if (not force_cuda) and use_cuda and getattr(getattr(pipe, "device", None), "type", "") != "cuda":
        offload_ok = False
        try:
            if hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
                offload_ok = True
        except Exception:
            offload_ok = False
        if not offload_ok:
            try:
                if hasattr(pipe, "enable_sequential_cpu_offload"):
                    pipe.enable_sequential_cpu_offload()
                    offload_ok = True
            except Exception:
                offload_ok = False
        if not offload_ok:
            _log(log, "[Flux2] WARNING: could not enable offload; staying on CPU (slow).")

    # Offload can transiently put parts of the pipeline on `meta`. Patch encode_prompt so we
    # never try to move *inputs* to meta (crash: "Cannot copy out of meta tensor; no data!").
    try:
        if use_cuda:
            import torch
            _patch_meta_encode_prompt(pipe, torch.device("cuda"), log=log)
    except Exception:
        pass

    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    _PIPE = pipe
    try:
        rm = get_resource_manager()
        rm.register("flux2_pipe", _PIPE, kind="init", unload=lambda: unload_flux2_pipe(log))
    except Exception:
        pass
    _log(log, "[Flux2] ready")
    return _PIPE


def _generate_init_fallback(
    *,
    scene: SceneSpec,
    cfg: ProjectConfig,
    refs: RefBundle,
    prompt: str,
    negative: str,
    preset: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: Optional[int],
    out_path: str,
    log=None,
) -> str:
    """Fallback init-frame generator (SDXL/FLUX.1) used when Flux2 fails.

    This keeps the app resilient: an init-frame is *better than none* for I2V
    stability, and Flux2 can occasionally fail under offload/driver combos.
    """
    init_cfg = getattr(cfg, "init_image", None)
    from .ai_image_init import generate_init_image

    # Flatten refs.
    ref_paths = list(refs.all_refs)

    # Best-effort anchor for img2img refine on SDXL/Flux1 pipelines.
    anchor_path = None
    if refs.extra_refs:
        anchor_path = refs.extra_refs[0]
    elif refs.character_refs:
        anchor_path = refs.character_refs[0]
    elif refs.location_refs:
        anchor_path = refs.location_refs[0]
    elif refs.style_refs:
        anchor_path = refs.style_refs[0]

    enable_ip = bool(getattr(init_cfg, "enable_ip_adapter", False)) if init_cfg else False
    ip_model_id = str(getattr(init_cfg, "ip_adapter_model_id", "") or "").strip() if init_cfg else ""
    ip_sub = str(getattr(init_cfg, "ip_adapter_subfolder", "") or "").strip() if init_cfg else ""
    ip_weight = str(getattr(init_cfg, "ip_adapter_weight_name", "") or "").strip() if init_cfg else ""

    # Single scale for legacy pipelines.
    ip_scale = None
    try:
        if ref_paths and enable_ip and init_cfg is not None:
            if refs.character_refs:
                ip_scale = float(getattr(init_cfg, "ip_adapter_scale_character", 0.8) or 0.8)
            elif refs.location_refs:
                ip_scale = float(getattr(init_cfg, "ip_adapter_scale_location", 0.6) or 0.6)
            else:
                ip_scale = float(getattr(init_cfg, "ip_adapter_scale_style", 0.5) or 0.5)
    except Exception:
        ip_scale = None

    res = generate_init_image(
        prompt=prompt,
        negative_prompt=negative,
        preset=str(preset),
        width=int(width),
        height=int(height),
        steps=int(steps),
        guidance_scale=float(guidance_scale),
        seed=seed,
        out_dir=os.path.dirname(out_path),
        filename=os.path.basename(out_path),
        cache_dir=resolve_hf_cache_dir(cfg),
        ref_image_paths=ref_paths,
        enable_ip_adapter=enable_ip,
        ip_adapter_model_id=ip_model_id,
        ip_adapter_subfolder=ip_sub,
        ip_adapter_weight_name=ip_weight,
        ip_adapter_scale=ip_scale,
        enable_controlnet=bool(getattr(init_cfg, "enable_controlnet", False)) if init_cfg else False,
        controlnet_type=str(getattr(init_cfg, "controlnet_type", "canny") or "canny") if init_cfg else "canny",
        controlnet_model_id=str(getattr(init_cfg, "controlnet_model_id", "") or "") if init_cfg else "",
        controlnet_scale=float(getattr(init_cfg, "controlnet_scale", 0.75) or 0.75) if init_cfg else 0.75,
        controlnet_image_paths=list(refs.location_refs or refs.extra_refs or refs.character_refs or []),
        anchor_image_path=anchor_path,
        refine_strength=float(getattr(init_cfg, "refine_strength", 0.35) or 0.35) if init_cfg else 0.35,
        log=log,
    )
    return getattr(res, "path", None) or out_path


def generate_init_image_for_scene(
    scene: SceneSpec,
    cfg: ProjectConfig,
    *,
    force: bool = False,
    preset_override: Optional[str] = None,
    policy_override: Optional[str] = None,
    use_cache_only: bool = False,
    last_location_memory_path: Optional[str] = None,
    log=None,
) -> InitGenResult:
    init_cfg = getattr(cfg, "init_image", None)
    if not init_cfg or not getattr(init_cfg, "enabled", True):
        raise RuntimeError("Init image generation is disabled")

    key, refs, preset, policy, w, h, steps, gs = compute_init_cache_key(
        scene,
        cfg,
        preset_override=preset_override,
        policy_override=policy_override,
        last_location_memory_path=last_location_memory_path,
        log=log,
    )
    prompt, negative = build_scene_prompts(cfg, scene)
    seed = getattr(init_cfg, "seed", None)

    img_dir, cache_json = _resolve_cache_dirs(cfg)
    out_path = os.path.join(img_dir, f"init_{key[:16]}.png")

    cache = load_cache(cache_json)
    item = cache.get(key)

    if use_cache_only:
        return try_use_cached_init_frame(scene, cfg, preset_override=preset_override, last_location_memory_path=last_location_memory_path, log=log)

    if item and os.path.exists(item.path) and policy != "always_refine" and not force:
        scene.init_frame_path = item.path
        scene.init_frame_hash = key
        return InitGenResult(path=item.path, cache_key=key, cache_hit=True)

    _log(log, f"[InitFrame] generating ({preset}) -> {out_path}")

    ref_paths = {
        "character_refs": list(refs.character_refs),
        "location_refs": list(refs.location_refs),
        "style_refs": list(refs.style_refs),
        "extra_refs": list(refs.extra_refs),
    }
    anchor_path = None
    for group in (refs.extra_refs, refs.character_refs, refs.location_refs, refs.style_refs):
        if group:
            anchor_path = group[0]
            break

    model_overrides = resolve_model_overrides(init_cfg)
    cache_dir = resolve_hf_cache_dir(cfg)

    job = build_job(
        preset=preset,
        prompt=prompt,
        negative_prompt=negative,
        width=w,
        height=h,
        steps=steps,
        guidance_scale=gs,
        seed=seed,
        out_path=out_path,
        cache_dir=cache_dir,
        ref_paths=ref_paths,
        enable_ip_adapter=bool(getattr(init_cfg, "enable_ip_adapter", False)),
        ip_adapter_model_id=str(getattr(init_cfg, "ip_adapter_model_id", "") or ""),
        ip_adapter_subfolder=str(getattr(init_cfg, "ip_adapter_subfolder", "") or ""),
        ip_adapter_weight_name=str(getattr(init_cfg, "ip_adapter_weight_name", "") or ""),
        ip_adapter_scale=float(getattr(init_cfg, "ip_adapter_scale_location", 0.6) or 0.6),
        enable_controlnet=bool(getattr(init_cfg, "enable_controlnet", False)),
        controlnet_type=str(getattr(init_cfg, "controlnet_type", "canny") or "canny"),
        controlnet_model_id=str(getattr(init_cfg, "controlnet_model_id", "") or ""),
        controlnet_scale=float(getattr(init_cfg, "controlnet_scale", 0.75) or 0.75),
        controlnet_image_paths=list(refs.location_refs or refs.extra_refs or refs.character_refs or []),
        anchor_image_path=anchor_path,
        refine_strength=float(getattr(init_cfg, "refine_strength", 0.35) or 0.35),
        model_overrides=model_overrides,
        max_sequence_length=512,
    )

    w, h = clamp_to_multiple_of_8(w, h)
    result = run_init_frame_job(job, log=log, target_size=(w, h))
    if not result.ok or not result.path:
        raise RuntimeError(result.error or "Init frame generation failed")

    cache[key] = CacheItem(path=_safe_abspath(result.path), seed=int(result.seed or 0), created_ts=int(time.time()))
    save_cache(cache_json, cache)

    scene.init_frame_path = _safe_abspath(result.path)
    scene.init_frame_hash = key
    return InitGenResult(path=scene.init_frame_path, cache_key=key, cache_hit=False)
