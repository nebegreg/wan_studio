from __future__ import annotations

import os
import sys
import subprocess
import wave
from typing import Optional

def _write_silence_wav(path: str, seconds: float, sample_rate: int = 32000, channels: int = 1):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    nframes = int(max(0.0, float(seconds)) * int(sample_rate))
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(int(channels))
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(b'\x00\x00' * int(channels) * nframes)

def _pip_install_audiocraft() -> bool:
    """Best-effort install of audiocraft (AudioGen/MusicGen).

    We keep this optional to avoid breaking offline/minimal environments.
    """
    try:
        # Use python -m pip to avoid relying on `pip` executable
        cmd = [sys.executable, "-m", "pip", "install", "git+https://github.com/facebookresearch/audiocraft.git"]
        subprocess.check_call(cmd)
        return True
    except Exception:
        return False

def ensure_audiocraft(try_install: bool = True) -> bool:
    try:
        import audiocraft  # noqa
        return True
    except Exception:
        if try_install:
            if _pip_install_audiocraft():
                try:
                    import audiocraft  # noqa
                    return True
                except Exception:
                    return False
        return False

def generate_vfx_wav(
    *,
    prompt: str,
    out_wav: str,
    seconds: float,
    model_name: str = "facebook/audiogen-medium",
    seed: Optional[int] = None,
    try_install: bool = True,
) -> str:
    """Generate a VFX/SFX wav using AudioGen (audiocraft) if available.

    If audiocraft is missing (and install fails), writes silence instead and returns out_wav.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        _write_silence_wav(out_wav, seconds=float(seconds), sample_rate=32000, channels=1)
        return out_wav

    if not ensure_audiocraft(try_install=try_install):
        _write_silence_wav(out_wav, seconds=float(seconds), sample_rate=32000, channels=1)
        return out_wav

    try:
        from audiocraft.models import AudioGen
        from audiocraft.data.audio import audio_write
        import torch

        dur = float(max(0.5, min(30.0, float(seconds))))
        ag = AudioGen.get_pretrained(model_name)
        ag.set_generation_params(duration=dur)
        if seed is not None:
            try:
                torch.manual_seed(int(seed))
            except Exception:
                pass

        wav = ag.generate([prompt])[0].cpu()
        os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)
        tmp_base = os.path.splitext(out_wav)[0]
        audio_write(tmp_base, wav, ag.sample_rate, strategy='loudness')
        gen = tmp_base + ".wav"
        if gen != out_wav:
            try:
                os.replace(gen, out_wav)
            except Exception:
                out_wav = gen
        return out_wav
    except Exception:
        _write_silence_wav(out_wav, seconds=float(seconds), sample_rate=32000, channels=1)
        return out_wav
