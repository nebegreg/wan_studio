"""Quick environment sanity checks.

Run:
  python doctor.py

This prints:
  - Torch/CUDA availability
  - GPU name + VRAM
  - ffmpeg presence
  - RIFE + models presence
  - Piper + default voice presence
"""

from __future__ import annotations

import os
import shutil
import subprocess


def _which(exe: str) -> str | None:
    return shutil.which(exe)


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        out = (p.stdout or "") + (p.stderr or "")
        return p.returncode, out.strip()
    except Exception as e:
        return 999, str(e)


def main() -> int:
    print("== doctor ==")

    # Python + pip
    try:
        import sys
        print("python:", sys.version.splitlines()[0])
        print("python_exe:", sys.executable)
    except Exception:
        pass
    try:
        import pip  # type: ignore
        print("pip:", getattr(pip, "__version__", "?"))
    except Exception as e:
        print("pip: MISSING (fix: python -m ensurepip --upgrade)")

    # Torch/CUDA
    try:
        import torch

        print("torch:", torch.__version__)
        print("cuda_available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("gpu:", torch.cuda.get_device_name(0))
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print("vram_gb:", round(vram, 2))
    except Exception as e:
        print("torch: ERROR", e)

    # HF stack versions
    for pkg in ("diffusers", "transformers", "accelerate", "safetensors", "huggingface_hub", "bitsandbytes"):
        try:
            mod = __import__(pkg)
            print(f"{pkg}:", getattr(mod, "__version__", "?"))
        except Exception as e:
            print(f"{pkg}: ERROR", e)

    # Diffusers pipeline imports (quick smoke tests)
    try:
        from diffusers import WanPipeline
        print("diffusers.WanPipeline: OK")
    except Exception as e:
        print("diffusers.WanPipeline: ERROR", e)
    try:
        from diffusers import WanImageToVideoPipeline
        print("diffusers.WanImageToVideoPipeline: OK")
    except Exception as e:
        print("diffusers.WanImageToVideoPipeline: ERROR", e)
    try:
        from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline
        print("diffusers.CogVideoXPipeline: OK")
        print("diffusers.CogVideoXImageToVideoPipeline: OK")
    except Exception as e:
        print("diffusers.CogVideoX*: ERROR", e)
    try:
        from diffusers import LTXPipeline, LTXImageToVideoPipeline
        print("diffusers.LTXPipeline: OK")
        print("diffusers.LTXImageToVideoPipeline: OK")
    except Exception as e:
        print("diffusers.LTX*: ERROR", e)
    # LTX-2 import path in diffusers
    try:
        from diffusers import LTX2Pipeline
        print("diffusers.LTX2Pipeline: OK")
    except Exception as e:
        print("diffusers.LTX2Pipeline: ERROR", e)
    try:
        from diffusers.pipelines.ltx2 import LTX2ImageToVideoPipeline
        print("diffusers.pipelines.ltx2.LTX2ImageToVideoPipeline: OK")
    except Exception as e:
        print("diffusers.pipelines.ltx2.LTX2ImageToVideoPipeline: ERROR", e)

    # FLUX.2 (init frame)
    try:
        from diffusers import Flux2Pipeline, Flux2Transformer2DModel
        print("diffusers.Flux2Pipeline: OK")
        print("diffusers.Flux2Transformer2DModel: OK")
    except Exception as e:
        print("diffusers.Flux2*: ERROR", e)

    # Z-Image / AutoPipeline
    try:
        from diffusers import AutoPipelineForText2Image
        print("diffusers.AutoPipelineForText2Image: OK")
    except Exception as e:
        print("diffusers.AutoPipelineForText2Image: ERROR", e)

    # ffmpeg / ffprobe
    ffmpeg = _which("ffmpeg")
    ffprobe = _which("ffprobe")
    print("ffmpeg:", ffmpeg or "MISSING")
    print("ffprobe:", ffprobe or "MISSING")
    if ffmpeg:
        rc, out = _run([ffmpeg, "-version"])
        print("ffmpeg_version:", out.splitlines()[0] if out else f"rc={rc}")

    # RIFE
    rife = os.path.abspath("tools/rife/rife-ncnn-vulkan")
    if os.path.isfile(rife) and os.access(rife, os.X_OK):
        print("rife:", rife)
        models = []
        for name in os.listdir("tools/rife") if os.path.isdir("tools/rife") else []:
            if name.startswith("rife-v"):
                models.append(name)
        print("rife_models:", sorted(models))
    else:
        print("rife:", "MISSING (run: bash get_tools.sh ./tools)")

    # Piper
    piper = os.path.abspath("tools/piper/piper")
    if not (os.path.isfile(piper) and os.access(piper, os.X_OK)):
        piper = _which("piper") or ""
    print("piper:", piper if piper else "MISSING")
    voice = os.path.abspath("tools/piper/voices/fr_FR-upmc-medium.onnx")
    print("piper_voice_fr_FR-upmc-medium:", "OK" if os.path.isfile(voice) else "MISSING")

    print("== done ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
