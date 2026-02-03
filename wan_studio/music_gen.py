from __future__ import annotations

import os
import wave
from typing import Optional


def _write_silence_wav(path: str, seconds: float, sample_rate: int = 48000, channels: int = 2):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    nframes = int(max(0.0, float(seconds)) * int(sample_rate))
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(int(channels))
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(b'\x00\x00' * int(channels) * nframes)


def generate_music_wav(
    *,
    prompt: str,
    out_wav: str,
    seconds: float,
    sample_rate: int = 48000,
    model_name: str = 'facebook/musicgen-small',
    seed: Optional[int] = None,
) -> str:
    """Generate a music bed.

    Uses audiocraft MusicGen if installed; otherwise writes silence.
    Always returns a valid wav path.
    """
    try:
        # audiocraft is optional; we avoid hard dependency.
        from audiocraft.models import MusicGen
        from audiocraft.data.audio import audio_write
        import torch

        dur = float(max(1.0, min(120.0, float(seconds))))
        mg = MusicGen.get_pretrained(model_name)
        mg.set_generation_params(duration=dur)
        if seed is not None:
            try:
                torch.manual_seed(int(seed))
            except Exception:
                pass
        wav = mg.generate([prompt])[0].cpu()
        os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)
        # audio_write will add extension; we force wav
        tmp = os.path.splitext(out_wav)[0]
        audio_write(tmp, wav, mg.sample_rate, strategy='loudness')
        # audio_write produces tmp.wav
        gen = tmp + '.wav'
        if gen != out_wav:
            try:
                os.replace(gen, out_wav)
            except Exception:
                out_wav = gen
        return out_wav
    except Exception:
        _write_silence_wav(out_wav, seconds=float(seconds), sample_rate=sample_rate, channels=2)
        return out_wav
