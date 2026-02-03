from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional


def _is_oom_error(e: BaseException) -> bool:
    try:
        import torch
        if isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())):
            return True
    except Exception:
        pass
    msg = str(e).lower()
    return ("out of memory" in msg) or ("cuda oom" in msg) or ("cublas" in msg and "alloc" in msg)


def _is_device_mismatch_error(e: BaseException) -> bool:
    msg = str(e).lower()
    return (
        ("expected all tensors to be on the same device" in msg)
        or ("found at least two devices" in msg)
        or ("cuda:0 and cpu" in msg)
        or ("cpu and cuda" in msg)
    )


def _write_result(payload: Dict[str, Any]) -> None:
    print("WAN_INIT_RESULT:", json.dumps(payload, ensure_ascii=False))


def _resolve_model_override(model_overrides: Dict[str, str], key: str, env_key: str, default: str) -> str:
    val = (model_overrides or {}).get(key) or ""
    if not val:
        val = os.environ.get(env_key, "")
    return val.strip() or default


def _generate_flux2(job: Dict[str, Any]) -> Dict[str, Any]:
    from PIL import Image
    import torch
    from .flux2_init import _load_flux2_pipe, _filter_kwargs, _has_meta_tensor_error, unload_flux2_pipe

    prompt = job.get("prompt", "")
    negative = job.get("negative_prompt", "")
    w = int(job.get("width") or 1024)
    h = int(job.get("height") or 1024)
    steps = int(job.get("steps") or 42)
    gs = float(job.get("guidance_scale") or 4.5)
    seed = job.get("seed")
    out_path = job.get("out_path")
    cache_dir = job.get("cache_dir")
    ref_paths = job.get("ref_paths") or {}
    init_cfg = job

    model_overrides = job.get("model_overrides") or {}
    preset = str(job.get("preset") or "")
    if preset == "flux2":
        model_id = _resolve_model_override(model_overrides, "flux2_model_id", "WAN_INIT_FLUX2_MODEL", "black-forest-labs/FLUX.2-dev")
    else:
        model_id = _resolve_model_override(model_overrides, "flux2_bnb4bit_model_id", "WAN_INIT_FLUX2_BNB4BIT_MODEL", "black-forest-labs/FLUX.2-dev-bnb-4bit")

    try:
        if model_id:
            os.environ["WAN_STUDIO_FLUX2_BASE_ID"] = model_id
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "little")
        exec_dev = "cpu"
        try:
            pipe = _load_flux2_pipe(None)
        except Exception:
            pipe = _load_flux2_pipe(None, force_cuda=True)
        if torch.cuda.is_available() and getattr(getattr(pipe, "device", None), "type", "") == "cuda":
            exec_dev = "cuda"
        generator = torch.Generator(device=exec_dev).manual_seed(int(seed))

        def _load_imgs(paths):
            out = []
            for rp in (paths or []):
                try:
                    out.append(Image.open(rp).convert("RGB"))
                except Exception:
                    pass
            return out

        char_imgs = _load_imgs(ref_paths.get("character_refs"))
        loc_imgs = _load_imgs(ref_paths.get("location_refs"))
        style_imgs = _load_imgs(ref_paths.get("style_refs"))
        extra_imgs = _load_imgs(ref_paths.get("extra_refs"))
        all_imgs = char_imgs + loc_imgs + style_imgs + extra_imgs

        ip_adapter_images = None
        ip_adapter_scales = None
        try:
            if bool(init_cfg.get("enable_ip_adapter", False)) and hasattr(pipe, "load_ip_adapter"):
                kwargs_load = {}
                if init_cfg.get("ip_adapter_model_id"):
                    kwargs_load["pretrained_model_name_or_path"] = init_cfg.get("ip_adapter_model_id")
                if init_cfg.get("ip_adapter_subfolder"):
                    kwargs_load["subfolder"] = init_cfg.get("ip_adapter_subfolder")
                if init_cfg.get("ip_adapter_weight_name"):
                    kwargs_load["weight_name"] = init_cfg.get("ip_adapter_weight_name")
                if kwargs_load:
                    pipe.load_ip_adapter(**_filter_kwargs(pipe.load_ip_adapter, kwargs_load))
                else:
                    pipe.load_ip_adapter()

                csc = float(init_cfg.get("ip_adapter_scale_character", 0.8) or 0.8)
                lsc = float(init_cfg.get("ip_adapter_scale_location", 0.6) or 0.6)
                ssc = float(init_cfg.get("ip_adapter_scale_style", 0.5) or 0.5)
                ip_adapter_images = []
                ip_adapter_scales = []
                for im in char_imgs:
                    ip_adapter_images.append(im)
                    ip_adapter_scales.append(csc)
                for im in loc_imgs:
                    ip_adapter_images.append(im)
                    ip_adapter_scales.append(lsc)
                for im in style_imgs:
                    ip_adapter_images.append(im)
                    ip_adapter_scales.append(ssc)
                for im in extra_imgs:
                    ip_adapter_images.append(im)
                    ip_adapter_scales.append(lsc)
                if hasattr(pipe, "set_ip_adapter_scale") and ip_adapter_scales:
                    try:
                        pipe.set_ip_adapter_scale(ip_adapter_scales if len(ip_adapter_scales) > 1 else float(ip_adapter_scales[0]))
                    except Exception:
                        pass
        except Exception:
            ip_adapter_images = None
            ip_adapter_scales = None

        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative,
            width=w,
            height=h,
            num_inference_steps=steps,
            guidance_scale=gs,
            generator=generator,
        )

        try:
            if bool(init_cfg.get("enable_controlnet", False)) and loc_imgs:
                ctrl = loc_imgs[0]
                kwargs["control_image"] = ctrl
                kwargs["controlnet_conditioning_image"] = ctrl
                kwargs["controlnet_image"] = ctrl
                cn_scale = float(init_cfg.get("controlnet_scale", 0.75) or 0.75)
                kwargs["controlnet_conditioning_scale"] = cn_scale
                kwargs["controlnet_scale"] = cn_scale
        except Exception:
            pass

        anchor = None
        if extra_imgs:
            anchor = extra_imgs[0]
        elif char_imgs:
            anchor = char_imgs[0]
        elif loc_imgs:
            anchor = loc_imgs[0]
        elif style_imgs:
            anchor = style_imgs[0]
        elif all_imgs:
            anchor = all_imgs[0]
        if anchor is not None:
            kwargs["image"] = anchor
            try:
                kwargs["strength"] = float(init_cfg.get("refine_strength", 0.35) or 0.35)
            except Exception:
                pass

        if ip_adapter_images:
            kwargs["ip_adapter_image"] = ip_adapter_images if len(ip_adapter_images) > 1 else ip_adapter_images[0]

        kwargs = _filter_kwargs(pipe.__call__, kwargs)
        with torch.inference_mode():
            try:
                out = pipe(**kwargs)
            except Exception as e:
                if _has_meta_tensor_error(e):
                    unload_flux2_pipe()
                    pipe2 = _load_flux2_pipe(None)
                    kwargs2 = _filter_kwargs(pipe2.__call__, dict(kwargs))
                    out = pipe2(**kwargs2)
                else:
                    raise

        img = None
        try:
            if getattr(out, "images", None):
                img = out.images[0]
        except Exception:
            img = None
        if img is None:
            raise RuntimeError("[Flux2] No image returned")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img.save(out_path)

        return {
            "ok": True,
            "path": out_path,
            "seed": int(seed),
            "width": w,
            "height": h,
            "preset": preset,
        }
    finally:
        try:
            unload_flux2_pipe()
        except Exception:
            pass


def _generate_other(job: Dict[str, Any]) -> Dict[str, Any]:
    from .ai_image_init import generate_init_image

    model_overrides = job.get("model_overrides") or {}
    preset = str(job.get("preset") or "sdxl_lightning_4step")
    res = generate_init_image(
        prompt=job.get("prompt", ""),
        negative_prompt=job.get("negative_prompt", ""),
        preset=preset,
        width=int(job.get("width") or 1024),
        height=int(job.get("height") or 1024),
        steps=int(job.get("steps") or 28),
        guidance_scale=float(job.get("guidance_scale") or 4.5),
        seed=job.get("seed"),
        out_dir=os.path.dirname(job.get("out_path") or "."),
        filename=os.path.basename(job.get("out_path") or "init.png"),
        cache_dir=job.get("cache_dir"),
        ref_image_paths=list((job.get("ref_paths") or {}).get("extra_refs") or []),
        enable_ip_adapter=bool(job.get("enable_ip_adapter", False)),
        ip_adapter_model_id=str(job.get("ip_adapter_model_id") or ""),
        ip_adapter_subfolder=str(job.get("ip_adapter_subfolder") or ""),
        ip_adapter_weight_name=str(job.get("ip_adapter_weight_name") or ""),
        ip_adapter_scale=job.get("ip_adapter_scale"),
        enable_controlnet=bool(job.get("enable_controlnet", False)),
        controlnet_type=str(job.get("controlnet_type") or "canny"),
        controlnet_model_id=str(job.get("controlnet_model_id") or ""),
        controlnet_scale=float(job.get("controlnet_scale") or 0.75),
        controlnet_image_paths=list(job.get("controlnet_image_paths") or []),
        anchor_image_path=job.get("anchor_image_path"),
        refine_strength=float(job.get("refine_strength") or 0.35),
        model_overrides=model_overrides,
        max_sequence_length=job.get("max_sequence_length"),
    )
    return {
        "ok": True,
        "path": res.path,
        "seed": int(res.seed),
        "width": int(res.width),
        "height": int(res.height),
        "preset": preset,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    args = ap.parse_args()

    job_path = args.job
    try:
        with open(job_path, "r", encoding="utf-8") as f:
            job = json.load(f)
    except Exception as e:
        _write_result({"ok": False, "error": str(e), "error_type": "job"})
        return 2

    preset = str(job.get("preset") or "")
    try:
        if preset in ("flux2_bnb4bit", "flux2"):
            payload = _generate_flux2(job)
        else:
            payload = _generate_other(job)
        _write_result(payload)
        return 0
    except Exception as e:
        err_type = "error"
        if _is_oom_error(e):
            err_type = "oom"
        elif _is_device_mismatch_error(e):
            err_type = "device"
        _write_result(
            {
                "ok": False,
                "error": str(e),
                "error_type": err_type,
                "preset": preset,
                "width": int(job.get("width") or 0),
                "height": int(job.get("height") or 0),
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
