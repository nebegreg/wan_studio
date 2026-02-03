"""Local text-to-image helpers to generate an init frame for I2V.

Goal: give users a quick way to synthesize a first frame (character/background)
that can be used as conditioning for I2V clips.

Design constraints:
- Keep dependencies minimal (diffusers/torch already in the project).
- Prefer SDXL + SDXL-Lightning (fast) because it runs well on 24GB VRAM.
- Optional FLUX support if the user has access to the weights.

This module is intentionally conservative: it loads a pipeline, generates one
image, then frees GPU memory.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any

import torch


@dataclass
class ImageGenResult:
    path: str
    seed: int
    width: int
    height: int
    steps: int
    preset: str


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _maybe_mkdir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _log(log: Optional[Callable[[str], None]], msg: str):
    if callable(log):
        log(msg)


def _resolve_model_override(model_overrides: Optional[Dict[str, str]], key: str, env_key: str, default: str) -> str:
    try:
        val = (model_overrides or {}).get(key) or ""
    except Exception:
        val = ""
    if not val:
        val = os.environ.get(env_key, "") or ""
    return val.strip() or default

def _make_control_image_pil(img):
    """Best-effort edge map using PIL only (no opencv dependency).

    This is *not* a true Canny detector, but it provides a useful structural guide
    for ControlNet when opencv/controlnet-aux aren't available.
    """
    try:
        from PIL import Image, ImageFilter, ImageOps
        im = img.convert('RGB')
        ed = im.filter(ImageFilter.FIND_EDGES)
        ed = ImageOps.autocontrast(ed.convert('L'))
        ed = ed.convert('RGB')
        return ed
    except Exception:
        return img


def generate_init_image(
    *,
    prompt: str,
    negative_prompt: str = "",
    preset: str = "sdxl_lightning_4step",
    width: int = 1024,
    height: int = 1024,
    steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    seed: Optional[int] = None,
    out_dir: str = "./outputs/assets/init_images",
    filename: Optional[str] = None,
    cache_dir: Optional[str] = None,
    device: Optional[str] = None,
    # Optional "cinema-grade" reference conditioning (best-effort)
    ref_image_paths: Optional[list[str]] = None,
    enable_ip_adapter: bool = False,
    ip_adapter_model_id: str = "",
    ip_adapter_subfolder: str = "",
    ip_adapter_weight_name: str = "",
    ip_adapter_scale: Optional[float] = None,
    # Optional ControlNet (best-effort)
    enable_controlnet: bool = False,
    controlnet_type: str = 'canny',
    controlnet_model_id: str = '',
    controlnet_scale: float = 0.75,
    controlnet_image_paths: Optional[list[str]] = None,
    # Optional img2img refine (best-effort)
    anchor_image_path: Optional[str] = None,
    refine_strength: float = 0.35,
    log: Optional[Callable[[str], None]] = None,
    model_overrides: Optional[Dict[str, str]] = None,
    max_sequence_length: Optional[int] = None,
) -> ImageGenResult:
    """Generate a single PNG image and return its path.

    Presets:
      - sdxl_lightning_4step (default): SDXL base + ByteDance SDXL-Lightning LoRA
      - sdxl_base: SDXL base
      - sdxl_turbo: SDXL Turbo
      - flux1_schnell: Black Forest Labs FLUX.1-schnell (optional)

    Note: some models (FLUX) require accepting a license on Hugging Face.
    """

    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Prompt vide")

    device = device or _default_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Seed
    if seed is None:
        seed = int.from_bytes(os.urandom(2), "little")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    # Output path
    out_dir = _maybe_mkdir(out_dir)
    if not filename:
        filename = f"init_{preset}_{_now_tag()}_{seed}.png"
    out_path = os.path.join(out_dir, filename)

    pipe = None
    # Optional anchor image (img2img refine)
    anchor_img = None
    if anchor_image_path and os.path.exists(anchor_image_path):
        try:
            from PIL import Image
            anchor_img = Image.open(anchor_image_path).convert('RGB')
        except Exception:
            anchor_img = None

    # Optional ControlNet control image (best-effort)
    control_img = None
    if enable_controlnet:
        cand_paths = controlnet_image_paths or ref_image_paths or []
        for cp in cand_paths:
            if cp and os.path.exists(cp):
                try:
                    from PIL import Image
                    control_img = Image.open(cp).convert('RGB')
                    break
                except Exception:
                    control_img = None
        if control_img is not None and (controlnet_type or 'canny').lower().startswith('canny'):
            control_img = _make_control_image_pil(control_img)

    try:
        if preset in ("zimage_turbo", "zimage"):
            from diffusers import AutoPipelineForText2Image

            model_id = _resolve_model_override(
                model_overrides,
                "zimage_turbo_model_id" if preset == "zimage_turbo" else "zimage_model_id",
                "WAN_INIT_ZIMAGE_TURBO_MODEL" if preset == "zimage_turbo" else "WAN_INIT_ZIMAGE_MODEL",
                "Tongyi-MAI/Z-Image-Turbo" if preset == "zimage_turbo" else "Tongyi-MAI/Z-Image",
            )
            _log(log, f"[InitImage] Loading Z-Image: {model_id}")
            pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype, cache_dir=cache_dir)
            pipe = pipe.to(device)

            kwargs = dict(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=int(steps or 8),
                guidance_scale=float(guidance_scale or 3.5),
                generator=generator,
            )
            if max_sequence_length is not None:
                try:
                    if "max_sequence_length" in pipe.__call__.__code__.co_varnames:
                        kwargs["max_sequence_length"] = int(max_sequence_length)
                except Exception:
                    pass

            out = pipe(**kwargs)
            img = out.images[0]
            img.save(out_path)
            return ImageGenResult(path=out_path, seed=int(seed), width=width, height=height, steps=int(steps or 8), preset=preset)

        if preset == "sdxl_lightning_4step":
            from diffusers import EulerDiscreteScheduler
            # Choose pipeline: txt2img / img2img / ControlNet (best-effort)
            StableDiffusionXLPipeline = None
            StableDiffusionXLImg2ImgPipeline = None
            StableDiffusionXLControlNetPipeline = None
            try:
                from diffusers import StableDiffusionXLPipeline as _T
                StableDiffusionXLPipeline = _T
            except Exception:
                pass
            try:
                from diffusers import StableDiffusionXLImg2ImgPipeline as _I
                StableDiffusionXLImg2ImgPipeline = _I
            except Exception:
                pass
            try:
                from diffusers import StableDiffusionXLControlNetPipeline as _C
                StableDiffusionXLControlNetPipeline = _C
            except Exception:
                pass

            base_id = _resolve_model_override(
                model_overrides,
                "sdxl_base_model_id",
                "WAN_INIT_SDXL_BASE_MODEL",
                "stabilityai/stable-diffusion-xl-base-1.0",
            )
            lora_repo = _resolve_model_override(
                model_overrides,
                "sdxl_lightning_lora_id",
                "WAN_INIT_SDXL_LIGHTNING_LORA",
                "ByteDance/SDXL-Lightning",
            )
            lora_file = _resolve_model_override(
                model_overrides,
                "sdxl_lightning_lora_file",
                "WAN_INIT_SDXL_LIGHTNING_LORA_FILE",
                "sdxl_lightning_4step_lora.safetensors",
            )

            _log(log, f"[InitImage] Loading SDXL base: {base_id}")
            controlnet = None
            if enable_controlnet and StableDiffusionXLControlNetPipeline is not None:
                # Default SDXL canny controlnet
                cn_id = (controlnet_model_id or '').strip()
                if not cn_id:
                    cn_id = 'diffusers/controlnet-canny-sdxl-1.0'
                try:
                    from diffusers import ControlNetModel
                    controlnet = ControlNetModel.from_pretrained(cn_id, torch_dtype=dtype, cache_dir=cache_dir)
                    _log(log, f'[InitImage] ControlNet: {cn_id}')
                except Exception as e:
                    controlnet = None
                    _log(log, f'[InitImage] ControlNet load skipped: {e}')

            if controlnet is not None and StableDiffusionXLControlNetPipeline is not None and control_img is not None:
                pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                    base_id,
                    controlnet=controlnet,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    cache_dir=cache_dir,
                )
            elif anchor_img is not None and StableDiffusionXLImg2ImgPipeline is not None:
                pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    base_id,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    cache_dir=cache_dir,
                )
            else:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    base_id,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    cache_dir=cache_dir,
                )

            # Recommended scheduler for SDXL-Lightning.
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config, timestep_spacing="trailing"
            )
            pipe.to(device)

            # Lightning is a LoRA; we load + fuse.
            _log(log, f"[InitImage] Loading Lightning LoRA: {lora_repo}/{lora_file}")
            pipe.load_lora_weights(lora_repo, weight_name=lora_file, cache_dir=cache_dir)
            try:
                pipe.fuse_lora()
            except Exception:
                # fuse_lora isn't strictly required; fallback to runtime LoRA.
                pass

            steps = int(steps or 4)
            guidance_scale = float(guidance_scale if guidance_scale is not None else 0.0)

        elif preset == "sdxl_base":
            # Choose pipeline: txt2img / img2img / ControlNet (best-effort)
            StableDiffusionXLPipeline = None
            StableDiffusionXLImg2ImgPipeline = None
            StableDiffusionXLControlNetPipeline = None
            try:
                from diffusers import StableDiffusionXLPipeline as _T
                StableDiffusionXLPipeline = _T
            except Exception:
                pass
            try:
                from diffusers import StableDiffusionXLImg2ImgPipeline as _I
                StableDiffusionXLImg2ImgPipeline = _I
            except Exception:
                pass
            try:
                from diffusers import StableDiffusionXLControlNetPipeline as _C
                StableDiffusionXLControlNetPipeline = _C
            except Exception:
                pass

            base_id = _resolve_model_override(
                model_overrides,
                "sdxl_base_model_id",
                "WAN_INIT_SDXL_BASE_MODEL",
                "stabilityai/stable-diffusion-xl-base-1.0",
            )
            _log(log, f"[InitImage] Loading SDXL base: {base_id}")
            controlnet = None
            if enable_controlnet and StableDiffusionXLControlNetPipeline is not None:
                # Default SDXL canny controlnet
                cn_id = (controlnet_model_id or '').strip()
                if not cn_id:
                    cn_id = 'diffusers/controlnet-canny-sdxl-1.0'
                try:
                    from diffusers import ControlNetModel
                    controlnet = ControlNetModel.from_pretrained(cn_id, torch_dtype=dtype, cache_dir=cache_dir)
                    _log(log, f'[InitImage] ControlNet: {cn_id}')
                except Exception as e:
                    controlnet = None
                    _log(log, f'[InitImage] ControlNet load skipped: {e}')

            if controlnet is not None and StableDiffusionXLControlNetPipeline is not None and control_img is not None:
                pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                    base_id,
                    controlnet=controlnet,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    cache_dir=cache_dir,
                )
            elif anchor_img is not None and StableDiffusionXLImg2ImgPipeline is not None:
                pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    base_id,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    cache_dir=cache_dir,
                )
            else:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    base_id,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    cache_dir=cache_dir,
                )
            pipe.to(device)

            steps = int(steps or 30)
            guidance_scale = float(guidance_scale if guidance_scale is not None else 5.0)

        elif preset == "sdxl_turbo":
            from diffusers import AutoPipelineForText2Image

            model_id = _resolve_model_override(
                model_overrides,
                "sdxl_turbo_model_id",
                "WAN_INIT_SDXL_TURBO_MODEL",
                "stabilityai/sdxl-turbo",
            )
            _log(log, f"[InitImage] Loading SDXL Turbo: {model_id}")
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                cache_dir=cache_dir,
            )
            pipe.to(device)

            steps = int(steps or 2)
            guidance_scale = float(guidance_scale if guidance_scale is not None else 0.0)

        elif preset in ("flux2_bnb4bit", "flux2"):
            # Optional: requires access to the weights.
            try:
                from diffusers import Flux2Pipeline
            except Exception as e:
                raise RuntimeError(
                    "Flux2Pipeline non disponible dans diffusers. Mets à jour diffusers (main/dev) ou utilise SDXL."
                ) from e

            # Default model IDs.
            model_id = _resolve_model_override(
                model_overrides,
                "flux2_bnb4bit_model_id" if preset == "flux2_bnb4bit" else "flux2_model_id",
                "WAN_INIT_FLUX2_BNB4BIT_MODEL" if preset == "flux2_bnb4bit" else "WAN_INIT_FLUX2_MODEL",
                "black-forest-labs/FLUX.2-dev-bnb-4bit" if preset == "flux2_bnb4bit" else "black-forest-labs/FLUX.2-dev",
            )
            _log(log, f"[InitImage] Loading FLUX2: {model_id}")
            pipe = Flux2Pipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                cache_dir=cache_dir,
            )
            pipe.to(device)
            steps = int(steps or 28)
            guidance_scale = float(guidance_scale if guidance_scale is not None else 4.5)

        elif preset == "flux1_schnell":
            # Optional: requires access to the weights.
            try:
                from diffusers import FluxPipeline
            except Exception as e:
                raise RuntimeError(
                    "FluxPipeline non disponible dans diffusers. Mets à jour diffusers (main/dev) ou utilise SDXL."
                ) from e

            model_id = _resolve_model_override(
                model_overrides,
                "flux1_schnell_model_id",
                "WAN_INIT_FLUX1_SCHNELL_MODEL",
                "black-forest-labs/FLUX.1-schnell",
            )
            _log(log, f"[InitImage] Loading FLUX.1-schnell: {model_id}")
            pipe = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                cache_dir=cache_dir,
            )
            pipe.to(device)
            steps = int(steps or 4)
            guidance_scale = float(guidance_scale if guidance_scale is not None else 3.5)

        else:
            raise ValueError(f"Preset inconnu: {preset}")

        # Memory helpers
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

        # Optional: IP-Adapter (best-effort) for reference images
        ip_adapter_images = []
        if enable_ip_adapter and ref_image_paths:
            try:
                from PIL import Image
                for p in (ref_image_paths or []):
                    try:
                        if p and os.path.exists(p):
                            ip_adapter_images.append(Image.open(p).convert('RGB'))
                    except Exception:
                        pass
            except Exception:
                ip_adapter_images = []

        if enable_ip_adapter and ip_adapter_images and hasattr(pipe, 'load_ip_adapter'):
            try:
                kwargs_load: Dict[str, Any] = {}
                if (ip_adapter_model_id or '').strip():
                    kwargs_load['pretrained_model_name_or_path'] = ip_adapter_model_id
                if (ip_adapter_subfolder or '').strip():
                    kwargs_load['subfolder'] = ip_adapter_subfolder
                if (ip_adapter_weight_name or '').strip():
                    kwargs_load['weight_name'] = ip_adapter_weight_name
                if kwargs_load:
                    try:
                        pipe.load_ip_adapter(**kwargs_load)
                    except TypeError:
                        pipe.load_ip_adapter()
                else:
                    pipe.load_ip_adapter()
                _log(log, '[InitImage] IP-Adapter enabled')
                if hasattr(pipe, 'set_ip_adapter_scale') and ip_adapter_scale is not None:
                    try:
                        pipe.set_ip_adapter_scale(float(ip_adapter_scale))
                    except Exception:
                        pass
            except Exception as e:
                _log(log, f"[InitImage] IP-Adapter load skipped: {e}")

        # Inference
        _log(log, f"[InitImage] Generating {width}x{height}, steps={steps}, seed={seed}")
        call_kwargs: Dict[str, Any] = dict(
            prompt=prompt,
            negative_prompt=(negative_prompt or "") if preset.startswith("sdxl") else None,
            width=int(width),
            height=int(height),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        )
        # ControlNet (SDXL): control image passed as `image`
        if enable_controlnet and control_img is not None:
            call_kwargs['image'] = control_img
            call_kwargs['controlnet_conditioning_scale'] = float(controlnet_scale or 0.75)
        # Img2Img (SDXL): anchor image passed as `image` + strength
        if (not enable_controlnet) and anchor_img is not None:
            call_kwargs['image'] = anchor_img
            call_kwargs['strength'] = float(refine_strength or 0.35)
        # IP-Adapter image conditioning (only if the pipeline supports it)
        if enable_ip_adapter and ip_adapter_images:
            call_kwargs['ip_adapter_image'] = ip_adapter_images if len(ip_adapter_images) > 1 else ip_adapter_images[0]

        with torch.inference_mode():
            try:
                res = pipe(**call_kwargs)
            except TypeError:
                # Some pipelines reject unknown kwargs; retry without ip_adapter_image
                call_kwargs.pop('ip_adapter_image', None)
                res = pipe(**call_kwargs)

        img = res.images[0]
        img.save(out_path)
        _log(log, f"[InitImage] Saved: {out_path}")

        return ImageGenResult(
            path=os.path.abspath(out_path),
            seed=int(seed),
            width=int(width),
            height=int(height),
            steps=int(steps),
            preset=str(preset),
        )

    finally:
        # Free memory ASAP (important when the video pipeline is large).
        try:
            if pipe is not None:
                # Free the pipeline instead of moving fp16 weights to CPU (can spam warnings / break on CPU-only).
                del pipe
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
