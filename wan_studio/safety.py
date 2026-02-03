from __future__ import annotations

from dataclasses import replace
from typing import Optional, Callable, Any

try:
    from .config import ProjectConfig
except Exception:  # pragma: no cover
    ProjectConfig = Any  # type: ignore

LogFn = Optional[Callable[[str], None]]


def apply_cinema_safe_overrides(cfg: ProjectConfig, log: LogFn = None, where: str = "") -> ProjectConfig:
    """Return a runtime-safe cfg (does not mutate the original dataclass).

    This is the "make it not crash" master switch:
    - disables UniPC for Wan (shape-mismatch crash guard)
    - forces no-overlap + cut stitching
    - forces hard cut transitions
    - increases OOM retries
    - enables sequential offload to lower peak VRAM

    The UI may still show the user's values, but the engine will enforce these at runtime
    when cinema_safe_mode=True.
    """
    try:
        enabled = bool(getattr(cfg, "cinema_safe_mode", False))
    except Exception:
        enabled = False

    if not enabled:
        return cfg

    try:
        if log is not None:
            p = f" ({where})" if where else ""
            log(f"[SAFE] Cinema SAFE mode ON{p}: enforcing anti-crash settings.")
    except Exception:
        pass

    # Prepare overrides (only top-level dataclass fields; nested objects are handled below).
    overrides = dict(
        no_overlap_strict=True,
        overlap_frames=0,
        blend_mode="cut",
        scene_transition_mode="cut",
        scene_transition_frames=0,
        cut_strict_between_clips=True,
        reset_continuity_between_clips=True,
        strict_clip_coherence=True,
        force_first_frame_to_conditioning=True,
        strict_prompt_max_tokens=True,
        frame_safety_enabled=True,
        auto_vram_optimizations=True,
        # UniPC on Wan has been a common source of shape-mismatch crashes
        use_unipc=False,
        # Reduce peak VRAM
        enable_model_cpu_offload=True,
        enable_sequential_cpu_offload=True,
        # FP32 VAE decode increases VRAM usage a lot; keep quality high elsewhere.
        vae_decode_fp32=False,
        # Retry more aggressively when an OOM is detected
        oom_retry_max_attempts=max(int(getattr(cfg, "oom_retry_max_attempts", 2) or 2), 4),
    )

    # Apply overrides safely (ignore unknown keys if cfg class differs).
    kwargs = {}
    for k, v in overrides.items():
        if hasattr(cfg, k):
            kwargs[k] = v

    cfg2 = replace(cfg, **kwargs) if kwargs else cfg

    # Nested init-image caps (best-effort)
    try:
        init = getattr(cfg2, "init_image", None)
        if init is not None:
            # keep user's preset choice, but make it less likely to OOM on Flux2
            if str(getattr(init, "preset", "")).startswith("flux2"):
                # Conservative step cap for init frames in safe mode
                if hasattr(init, "steps"):
                    init.steps = min(int(getattr(init, "steps", 18) or 18), 20)
                if hasattr(init, "guidance_scale"):
                    init.guidance_scale = min(float(getattr(init, "guidance_scale", 4.0) or 4.0), 5.0)
    except Exception:
        pass

    return cfg2
