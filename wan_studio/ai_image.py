"""AI still image generator.

Purpose: generate a reference image to initialize I2V pipelines.

We keep this module decoupled from the main video engine so it can be used
from both Model tab (global init image) and Studio clip inspector (per-clip
init override).

Implementation notes:
- Uses diffusers pipelines if available.
- Tries FLUX first when the model id looks like a FLUX repo.
- Falls back to AutoPipelineForText2Image.
- Filters kwargs according to the pipeline's __call__ signature for
  robustness across diffusers versions.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ImageGenResult:
    ok: bool
    path: Optional[str] = None
    message: str = ""


def _filter_kwargs(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs


def generate_image(
    *,
    prompt: str,
    negative_prompt: str = "",
    model_id: str,
    out_path: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    guidance: float = 4.0,
    seed: Optional[int] = None,
    dtype: str = "auto",
    enable_cpu_offload: bool = True,
) -> ImageGenResult:
    """Generate a PNG reference image.

    Parameters
    ----------
    dtype:
        "auto" tries bf16 then fp16, depending on GPU support.
    """
    prompt = (prompt or "").strip()
    model_id = (model_id or "").strip()
    out_path = str(out_path)

    if not prompt:
        return ImageGenResult(False, None, "Prompt vide")
    if not model_id:
        return ImageGenResult(False, None, "Model ID vide")

    try:
        import torch
        from diffusers import AutoPipelineForText2Image

        # dtype selection
        torch_dtype = None
        if dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype == "fp16":
            torch_dtype = torch.float16
        else:
            # auto
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            if torch.cuda.is_available():
                # Many consumer GPUs handle fp16 better; bf16 is ok on Ada but keep fallback.
                try:
                    if not torch.cuda.is_bf16_supported():
                        torch_dtype = torch.float16
                except Exception:
                    torch_dtype = torch.float16

        pipe = None
        # Prefer FLUX pipeline if available and user selected a FLUX model.
        if "flux" in model_id.lower():
            try:
                from diffusers import FluxPipeline  # type: ignore
                pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
            except Exception:
                pipe = None

        if pipe is None:
            # Generic path
            pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch_dtype)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if enable_cpu_offload and device == "cuda":
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                try:
                    pipe.to(device)
                except Exception:
                    pass
        else:
            try:
                pipe.to(device)
            except Exception:
                pass

        # Seed
        generator = None
        if seed is not None and device == "cuda":
            try:
                generator = torch.Generator(device=device).manual_seed(int(seed))
            except Exception:
                generator = None

        kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "width": int(width),
            "height": int(height),
            "num_inference_steps": int(steps),
            "guidance_scale": float(guidance),
            "generator": generator,
        }
        kwargs = _filter_kwargs(pipe.__call__, kwargs)

        # Run
        with torch.inference_mode():
            out = pipe(**kwargs)

        # Diffusers returns either images or .images
        img = None
        if hasattr(out, "images"):
            img = out.images[0]
        elif isinstance(out, (list, tuple)) and out:
            img = out[0]
        if img is None:
            return ImageGenResult(False, None, "Sortie pipeline inconnue")

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)

        # cleanup
        try:
            del pipe
        except Exception:
            pass
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

        return ImageGenResult(True, out_path, "")
    except Exception as e:
        return ImageGenResult(False, None, str(e))
