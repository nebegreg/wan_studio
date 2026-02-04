from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Callable, List


LogFn = Optional[Callable[[str], None]]


@dataclass
class InitFrameSubprocessResult:
    ok: bool
    path: Optional[str]
    preset: str
    width: int
    height: int
    seed: Optional[int]
    error: str = ""
    error_type: str = ""
    stdout: str = ""
    stderr: str = ""
    used_preset: str = ""


def _log(log: LogFn, msg: str) -> None:
    try:
        if callable(log):
            log(msg)
    except Exception:
        pass


def resolve_hf_cache_dir(cfg: Any = None, explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return os.path.expanduser(explicit)
    if cfg is not None:
        try:
            cand = getattr(cfg, "hf_cache_dir", None)
            if cand:
                return os.path.expanduser(str(cand))
        except Exception:
            pass
    env = os.environ
    for key in ("WAN_HF_CACHE", "HF_HUB_CACHE", "HF_HOME"):
        val = env.get(key)
        if val:
            if key == "HF_HOME":
                return os.path.join(os.path.expanduser(val), "hub")
            return os.path.expanduser(val)
    return os.path.expanduser("~/.cache/huggingface")


def resolve_model_overrides(init_cfg: Any) -> Dict[str, str]:
    if init_cfg is None:
        return {}
    out = {}
    for key in (
        "flux2_bnb4bit_model_id",
        "flux2_model_id",
        "zimage_turbo_model_id",
        "zimage_model_id",
        "sdxl_base_model_id",
        "sdxl_turbo_model_id",
        "sdxl_lightning_lora_id",
        "sdxl_lightning_lora_file",
        "flux1_schnell_model_id",
    ):
        try:
            val = str(getattr(init_cfg, key, "") or "").strip()
        except Exception:
            val = ""
        if val:
            out[key] = val
    return out


def clamp_to_multiple_of_8(w: int, h: int, min_size: int = 256) -> Tuple[int, int]:
    w = max(min_size, int(w))
    h = max(min_size, int(h))
    w = (w // 8) * 8
    h = (h // 8) * 8
    return max(min_size, w), max(min_size, h)


def ensure_flux2_steps(steps: int) -> int:
    return max(int(steps), 42)


def build_job(
    *,
    preset: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: Optional[int],
    out_path: str,
    cache_dir: Optional[str],
    ref_paths: Optional[Dict[str, Sequence[str]]] = None,
    enable_ip_adapter: bool = False,
    ip_adapter_model_id: str = "",
    ip_adapter_subfolder: str = "",
    ip_adapter_weight_name: str = "",
    ip_adapter_scale: Optional[float] = None,
    enable_controlnet: bool = False,
    controlnet_type: str = "canny",
    controlnet_model_id: str = "",
    controlnet_scale: float = 0.75,
    controlnet_image_paths: Optional[Sequence[str]] = None,
    anchor_image_path: Optional[str] = None,
    refine_strength: float = 0.35,
    model_overrides: Optional[Dict[str, str]] = None,
    max_sequence_length: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "preset": preset,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": int(width),
        "height": int(height),
        "steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "seed": seed,
        "out_path": out_path,
        "cache_dir": cache_dir,
        "ref_paths": ref_paths or {},
        "enable_ip_adapter": bool(enable_ip_adapter),
        "ip_adapter_model_id": ip_adapter_model_id,
        "ip_adapter_subfolder": ip_adapter_subfolder,
        "ip_adapter_weight_name": ip_adapter_weight_name,
        "ip_adapter_scale": ip_adapter_scale,
        "enable_controlnet": bool(enable_controlnet),
        "controlnet_type": controlnet_type,
        "controlnet_model_id": controlnet_model_id,
        "controlnet_scale": float(controlnet_scale),
        "controlnet_image_paths": list(controlnet_image_paths or []),
        "anchor_image_path": anchor_image_path,
        "refine_strength": float(refine_strength),
        "model_overrides": model_overrides or {},
        "max_sequence_length": max_sequence_length,
    }


def _parse_worker_output(stdout: str) -> Dict[str, Any]:
    for line in (stdout or "").splitlines():
        if line.strip().startswith("WAN_INIT_RESULT:"):
            payload = line.split("WAN_INIT_RESULT:", 1)[-1].strip()
            try:
                return json.loads(payload)
            except Exception:
                continue
    return {}


def _call_worker(job: Dict[str, Any]) -> InitFrameSubprocessResult:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(job, f)
        job_path = f.name
    try:
        cmd = [sys.executable, "-m", "wan_studio.init_frame_worker", "--job", job_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        data = _parse_worker_output(stdout)
        ok = bool(data.get("ok", False))
        return InitFrameSubprocessResult(
            ok=ok,
            path=data.get("path") if ok else None,
            preset=str(data.get("preset") or job.get("preset") or ""),
            width=int(data.get("width") or job.get("width") or 0),
            height=int(data.get("height") or job.get("height") or 0),
            seed=data.get("seed"),
            error=str(data.get("error") or ""),
            error_type=str(data.get("error_type") or ""),
            stdout=stdout,
            stderr=stderr,
            used_preset=str(data.get("preset") or job.get("preset") or ""),
        )
    finally:
        try:
            os.unlink(job_path)
        except Exception:
            pass


def _upscale_to_target(path: str, target_w: int, target_h: int, log: LogFn = None) -> None:
    try:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        if img.size == (int(target_w), int(target_h)):
            return
        img = img.resize((int(target_w), int(target_h)), Image.LANCZOS)
        img.save(path)
        _log(log, f"[InitFrame] upscale → {target_w}x{target_h}")
    except Exception as e:
        _log(log, f"[InitFrame] upscale skipped: {e}")


def _fallback_chain(preset: str) -> List[str]:
    chain = ["zimage_turbo", "sdxl_lightning_4step", "sdxl_base"]
    out = []
    for p in [preset] + chain:
        if p and p not in out:
            out.append(p)
    return out


def run_init_frame_job(
    job: Dict[str, Any],
    *,
    log: LogFn = None,
    downscale_retries: int = 2,
    fallback_chain: Optional[Sequence[str]] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> InitFrameSubprocessResult:
    job = dict(job)
    preset = str(job.get("preset") or "").strip()
    chain = list(fallback_chain or _fallback_chain(preset))
    if preset and chain[0] != preset:
        chain.insert(0, preset)

    target_w, target_h = target_size or (int(job.get("width") or 0), int(job.get("height") or 0))
    target_w, target_h = clamp_to_multiple_of_8(target_w, target_h)

    for p in chain:
        job["preset"] = p
        if p in ("flux2_bnb4bit", "flux2"):
            job["steps"] = ensure_flux2_steps(int(job.get("steps") or 42))

        base_w, base_h = clamp_to_multiple_of_8(int(job.get("width") or target_w), int(job.get("height") or target_h))
        job["width"], job["height"] = base_w, base_h

        _log(log, f"[InitFrame] subprocess preset={p} {job['width']}x{job['height']} steps={job.get('steps')}")
        result = _call_worker(job)
        if result.ok and result.path:
            if _looks_black_image(result.path):
                _log(log, f"[InitFrame] black image detected for preset={p}; trying fallback.")
            else:
                if (base_w, base_h) != (target_w, target_h):
                    _upscale_to_target(result.path, target_w, target_h, log=log)
                    result.width = target_w
                    result.height = target_h
                return result

        if result.error_type == "oom":
            for i in range(downscale_retries):
                scale = 0.82 if i == 0 else 0.70
                nw, nh = clamp_to_multiple_of_8(int(base_w * scale), int(base_h * scale))
                if (nw, nh) == (base_w, base_h):
                    continue
                job["width"], job["height"] = nw, nh
                _log(log, f"[InitFrame] OOM retry {i+1}/{downscale_retries}: {base_w}x{base_h} → {nw}x{nh}")
                retry = _call_worker(job)
                if retry.ok and retry.path:
                    if _looks_black_image(retry.path):
                        _log(log, f"[InitFrame] black image detected after OOM retry for preset={p}; trying fallback.")
                        break
                    if (nw, nh) != (target_w, target_h):
                        _upscale_to_target(retry.path, target_w, target_h, log=log)
                        retry.width = target_w
                        retry.height = target_h
                    return retry
            _log(log, f"[InitFrame] preset {p} failed after OOM retries; trying fallback.")
        else:
            _log(log, f"[InitFrame] preset {p} failed: {result.error or 'unknown error'} → fallback")

    return InitFrameSubprocessResult(
        ok=False,
        path=None,
        preset=preset,
        width=target_w,
        height=target_h,
        seed=None,
        error=f"InitFrame failed for presets: {', '.join(chain)}",
        error_type="failed",
        stdout="",
        stderr="",
        used_preset="",
    )
