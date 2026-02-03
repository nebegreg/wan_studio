from __future__ import annotations

import os
import re
import shlex
import subprocess
from typing import Optional, List, Dict, Tuple

from .config import ProjectConfig, SceneSpec, AudioSpec


def _which(exe: str) -> Optional[str]:
    from shutil import which

    return which(exe)


def find_ffmpeg() -> str:
    ff = _which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg not found in PATH")
    return ff


def find_ffprobe() -> str:
    fp = _which("ffprobe")
    if not fp:
        raise RuntimeError("ffprobe not found in PATH")
    return fp


def find_piper(tools_dir: str) -> Optional[str]:
    """Find Piper TTS executable.

    We prefer a project-local install under tools/piper/.
    """

    cand = os.path.join(tools_dir, "piper", "piper")
    if os.path.isfile(cand) and os.access(cand, os.X_OK):
        return cand
    cand2 = os.path.join(tools_dir, "piper", "piper-tts")
    if os.path.isfile(cand2) and os.access(cand2, os.X_OK):
        return cand2
    return _which("piper")


def probe_duration_seconds(path: str) -> float:
    fp = find_ffprobe()
    cmd = [
        fp,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    try:
        return float(out)
    except Exception as e:
        raise RuntimeError(f"ffprobe failed for {path}: {out}") from e


def _safe_name(name: str, fallback: str) -> str:
    s = (name or "").strip() or fallback
    return "".join([c if c.isalnum() or c in "-_" else "_" for c in s])


def _shot_id(scene: SceneSpec, index: int) -> str:
    return _safe_name(scene.label, f"SHOT_{index+1:03d}")


def _scene_id_from_label(label: str, index: int) -> str:
    """Infer a scene id like S01 from labels like S01_SH07.

    If not found, returns SCXX.
    """

    s = (label or "").strip()
    m = re.match(r"^(S\d+)", s)
    if m:
        return m.group(1)
    if "_" in s and s.split("_")[0]:
        # Keep first chunk if it looks like a scene token
        head = s.split("_")[0]
        if len(head) <= 6:
            return _safe_name(head, f"SC{index+1:02d}")
    return f"SC{index+1:02d}"


def ensure_audio_dir(project_out_dir: str, audio: AudioSpec) -> str:
    adir = os.path.join(project_out_dir, audio.audio_dir_name)
    os.makedirs(adir, exist_ok=True)
    return adir


def _ensure_wav(in_path: str, out_wav: str, sample_rate: int, channels: int = 2) -> str:
    ff = find_ffmpeg()
    os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)
    cmd = [
        ff,
        "-y",
        "-i",
        in_path,
        "-vn",
        "-ar",
        str(int(sample_rate)),
        "-ac",
        str(int(channels)),
        out_wav,
    ]
    subprocess.check_call(cmd)
    return out_wav


def _make_silence(out_wav: str, seconds: float, sample_rate: int, channels: int = 2) -> str:
    ff = find_ffmpeg()
    os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)
    ch = "stereo" if int(channels) == 2 else "mono"
    cmd = [
        ff,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=channel_layout={ch}:sample_rate={int(sample_rate)}",
        "-t",
        f"{float(seconds):.6f}",
        out_wav,
    ]
    subprocess.check_call(cmd)
    return out_wav


def _make_placeholder_music(out_wav: str, seconds: float, sample_rate: int, style: str = "ambient") -> str:
    """Generate a procedural ambience bed with ffmpeg (lavfi).

    Goals:
      - Always produce an audible, non-intrusive "cinematic bed" without external deps.
      - Keep headroom (limiter) so it can be mixed down cleanly.

    `style` is best-effort and only tweaks the filter chain.
    """
    ff = find_ffmpeg()
    os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)
    sec = max(0.1, float(seconds))
    sr = int(sample_rate)
    style = str(style or "ambient").strip().lower()

    # Ambient pad: pink noise (lowpassed) + sub drone(s) with slow tremolo + mild echo.
    # NOTE: This track will still be gain-staged in the final master mix (music_gain_db).
    if style in ("trailer", "epic"):
        # More energy, still safe.
        filt = (
            "[0:a]volume=-14dB,alowpass=1800,highpass=40[a0];"
            "[1:a]volume=-12dB,atremolo=f=0.10:d=0.6[a1];"
            "[2:a]volume=-18dB,atremolo=f=0.07:d=0.5[a2];"
            "[a0][a1][a2]amix=inputs=3:normalize=0,"
            "aecho=0.7:0.5:90|160:0.25|0.18,alimiter=limit=0.95[a]"
        )
    else:
        # Default: subtle, audible "cinematic ambience".
        filt = (
            "[0:a]volume=-18dB,alowpass=1400,highpass=40[a0];"
            "[1:a]volume=-16dB,atremolo=f=0.07:d=0.55[a1];"
            "[2:a]volume=-22dB,atremolo=f=0.05:d=0.45[a2];"
            "[a0][a1][a2]amix=inputs=3:normalize=0,"
            "aecho=0.6:0.5:120|260:0.22|0.16,alimiter=limit=0.95[a]"
        )
    cmd = [
        ff,
        "-y",
        "-f", "lavfi", "-i", f"anoisesrc=color=pink:sample_rate={sr}",
        "-f", "lavfi", "-i", f"sine=frequency=55:sample_rate={sr}",
        "-f", "lavfi", "-i", f"sine=frequency=110:sample_rate={sr}",
        "-filter_complex",
        filt,
        "-map",
        "[a]",
        "-t",
        f"{sec:.6f}",
        "-ar",
        str(sr),
        "-ac",
        "2",
        out_wav,
    ]
    subprocess.check_call(cmd)
    return out_wav


def _build_atempo_chain(factor: float) -> str:
    """ffmpeg atempo supports 0.5..2.0; chain if needed."""

    parts: List[float] = []
    f = float(factor)
    while f > 2.0:
        parts.append(2.0)
        f /= 2.0
    while f < 0.5:
        parts.append(0.5)
        f /= 0.5
    parts.append(f)
    return ",".join([f"atempo={p:.6f}" for p in parts])


def fit_audio_to_duration(
    in_wav: str,
    out_wav: str,
    target_seconds: float,
    spec: AudioSpec,
    allow_speedup: bool,
) -> str:
    """Fit WAV to an exact duration.

    - Optionally time-stretch (speed up) if longer than target.
    - Pad/trim to match exactly.
    """

    ff = find_ffmpeg()
    dur = max(0.001, probe_duration_seconds(in_wav))
    target = max(0.001, float(target_seconds))

    filters: List[str] = []
    if allow_speedup and spec.auto_fit_to_scene and dur > target:
        factor = dur / target
        factor = min(float(spec.fit_max_speedup), factor)
        filters.append(_build_atempo_chain(factor))

    filters.append(f"aresample={int(spec.sample_rate)}")
    if spec.fit_pad_shorter:
        filters.append("apad")
    filters.append(f"atrim=0:{target:.6f}")
    filters.append("asetpts=N/SR/TB")

    cmd = [
        ff,
        "-y",
        "-i",
        in_wav,
        "-vn",
        "-af",
        ",".join(filters),
        "-ar",
        str(int(spec.sample_rate)),
        "-ac",
        "2",
        out_wav,
    ]
    subprocess.check_call(cmd)
    return out_wav


def tts_piper(
    text: str,
    out_wav: str,
    tools_dir: str,
    voice_model_path: str,
    voice_config_path: Optional[str] = None,
    speaker: Optional[int] = None,
) -> None:
    """Run Piper TTS.

    On some Linux distros/bundles Piper ships shared libs next to the binary
    (e.g. libpiper_phonemize.so.1). When the loader can't find them you'll get
    exit=127 with a "cannot open shared object file" error.

    We fix that by injecting a robust LD_LIBRARY_PATH for the piper process.
    """

    piper = find_piper(tools_dir)
    if not piper:
        raise RuntimeError("Piper not found. Install it with: bash get_tools.sh")

    model = voice_model_path
    if not os.path.isfile(model):
        raise RuntimeError(f"Piper voice model not found: {model}")
    # Some builds need a .json next to .onnx; we just validate if provided.
    if voice_config_path and not os.path.isfile(voice_config_path):
        # Not fatal.
        pass

    os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)

    # Build a safe LD_LIBRARY_PATH for Piper (fixes libpiper_phonemize.so.1 issues).
    env = os.environ.copy()
    pdir = os.path.dirname(os.path.abspath(piper))
    cand_dirs = [pdir, os.path.join(pdir, "lib"), os.path.join(pdir, "libs"), os.path.join(pdir, "lib64")]

    # Also include any directory containing libpiper_phonemize.* found under tools_dir/piper.
    try:
        base = os.path.join(tools_dir, "piper")
        if os.path.isdir(base):
            import glob
            for pat in ("libpiper_phonemize.so*", "libpiper*.so*"):
                for f in glob.glob(os.path.join(base, "**", pat), recursive=True):
                    d = os.path.dirname(os.path.abspath(f))
                    cand_dirs.append(d)
    except Exception:
        pass

    # Deduplicate while preserving order
    seen = set()
    lib_path = []
    for d in cand_dirs:
        if not d or not os.path.isdir(d):
            continue
        if d in seen:
            continue
        seen.add(d)
        lib_path.append(d)

    old_ld = env.get("LD_LIBRARY_PATH", "")
    if old_ld:
        lib_path.append(old_ld)
    env["LD_LIBRARY_PATH"] = ":".join(lib_path)

    cmd = [piper, "--model", model, "--output_file", out_wav]
    if speaker is not None:
        try:
            cmd += ["--speaker", str(int(speaker))]
        except Exception:
            pass

    proc = subprocess.run(
        cmd,
        input=(text.strip() + "\n"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        err = (proc.stderr or "")
        if speaker is not None and ('unrecognized arguments' in err or 'unknown argument' in err) and '--speaker' in err:
            # retry without speaker for single-speaker voices/binaries
            cmd2 = [piper, '--model', model, '--output_file', out_wav]
            proc2 = subprocess.run(
                cmd2,
                input=(text.strip() + "\n"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            if proc2.returncode == 0:
                return
            err = (proc2.stderr or err)

        hint = ""
        if "libpiper" in err and "cannot open shared object file" in err:
            hint = (
                "\n\nHint: Piper's shared libs aren't being found. "
                "This build expects libs next to the binary (or in tools/piper/lib). "
                "Try running: `ldd tools/piper/piper` and ensure libpiper_phonemize.so.1 exists. "
                "(The app now sets LD_LIBRARY_PATH automatically for Piper.)"
            )
        raise RuntimeError(
            f"Piper failed (exit {proc.returncode}).\nCMD: {shlex.join(cmd)}\nSTDERR: {err[-2000:]}{hint}"
        )





def _apply_pitch_rate(
    in_wav: str,
    out_wav: str,
    *,
    sample_rate: int,
    pitch: float = 1.0,
    rate: float = 1.0,
) -> str:
    """Best-effort pitch/rate transform.

    Uses ffmpeg filters; designed to be robust rather than perfect.
    - pitch: >1 increases pitch (child-like); this also affects speed but later fitting corrects duration.
    - rate: atempo factor (0.5..2.0).
    """
    pitch = float(pitch) if pitch is not None else 1.0
    rate = float(rate) if rate is not None else 1.0
    if abs(pitch - 1.0) < 1e-3 and abs(rate - 1.0) < 1e-3:
        return in_wav
    ff = find_ffmpeg()
    os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)

    # Clamp atempo
    if rate < 0.5:
        rate = 0.5
    if rate > 2.0:
        rate = 2.0

    # asetrate changes pitch+speed, aresample returns to base sample rate, atempo adjusts speed.
    # We rely on later fit_audio_to_duration to match the target duration.
    filt = f"asetrate={int(sample_rate)}*{pitch},aresample={int(sample_rate)},atempo={rate}"
    cmd = [ff, "-y", "-i", in_wav, "-vn", "-ar", str(int(sample_rate)), "-ac", "2", "-filter:a", filt, out_wav]
    try:
        subprocess.check_call(cmd)
        return out_wav
    except Exception:
        return in_wav


def _resolve_voice(scene: SceneSpec, cfg: ProjectConfig) -> Tuple[str, Optional[str]]:
    """Resolve voice model/config for scene.

    Priority:
      1) scene.dialogue_voice
      2) Character voice (first in scene.characters)
      3) project default voice
    """
    try:
        from .voice_manager import resolve_voice_for_scene
        return resolve_voice_for_scene(scene, cfg)
    except Exception:
        audio = cfg.audio
        tools_dir = cfg.post.tools_dir if cfg.post and cfg.post.tools_dir else os.path.abspath("./tools")
        voice_model = audio.voice_model_path
        voice_cfg = audio.voice_config_path
        # If the stored paths are absolute from another bundle/folder, try to re-root by basename.
        try:
            if voice_model and not os.path.isfile(voice_model):
                base = os.path.basename(voice_model)
                cand = os.path.join(tools_dir, 'piper', 'voices', base)
                if os.path.isfile(cand):
                    voice_model = cand
                    if os.path.isfile(cand + '.json'):
                        voice_cfg = cand + '.json'
            if voice_cfg and not os.path.isfile(voice_cfg) and voice_model and os.path.isfile(voice_model + '.json'):
                voice_cfg = voice_model + '.json'
        except Exception:
            pass
        if (scene.dialogue_voice or "").strip():
            v = scene.dialogue_voice.strip()
            if os.path.isfile(v):
                voice_model = v
                if os.path.isfile(v + ".json"):
                    voice_cfg = v + ".json"
            else:
                cand = os.path.join(tools_dir, "piper", "voices", v)
                if os.path.isfile(cand):
                    voice_model = cand
                    if os.path.isfile(cand + ".json"):
                        voice_cfg = cand + ".json"
        return voice_model, voice_cfg


def generate_scene_dialogue_audio(
    cfg: ProjectConfig,
    scene_index: int | SceneSpec,
    project_out_dir: str,
) -> Optional[str]:
    """Generate (or reuse) the fitted dialogue WAV for one shot.

    Kept for backward compatibility ("scene" == timeline clip).
    """

    audio = cfg.audio
    if not audio.enabled:
        return None

    if isinstance(scene_index, SceneSpec):
        scene = scene_index
        try:
            idx = cfg.scenes.index(scene)
        except ValueError:
            idx = 0
    else:
        idx = int(scene_index)
        if idx < 0 or idx >= len(cfg.scenes):
            raise IndexError("scene_index out of range")
        scene = cfg.scenes[idx]

    text = (scene.dialogue_text or "").strip()
    if not text:
        return None

    adir = ensure_audio_dir(project_out_dir, audio)
    sdir = os.path.join(adir, audio.shot_stems_dir_name)
    os.makedirs(sdir, exist_ok=True)

    sid = _shot_id(scene, idx)
    raw = os.path.join(sdir, f"{sid}_dialogue_raw.wav")
    fitted = os.path.join(sdir, f"{sid}_dialogue.wav")

    tools_dir = cfg.post.tools_dir if cfg.post and cfg.post.tools_dir else os.path.abspath("./tools")
    voice_model, voice_cfg = _resolve_voice(scene, cfg)

    if audio.tts_engine.lower() == "piper":
        try:
            tts_piper(
                text=text,
                out_wav=raw,
                tools_dir=tools_dir,
                voice_model_path=voice_model,
                voice_config_path=voice_cfg,
                speaker=getattr(scene, "dialogue_speaker", None),
            )
            # Apply pitch/rate (best-effort) before fitting.
            try:
                pitch = float(getattr(scene, "dialogue_pitch", 1.0) or 1.0)
                rate = float(getattr(scene, "dialogue_rate", 1.0) or 1.0)
                if abs(pitch - 1.0) > 1e-3 or abs(rate - 1.0) > 1e-3:
                    raw2 = os.path.join(sdir, f"{sid}_dialogue_raw_pr.wav")
                    raw = _apply_pitch_rate(
                        raw,
                        raw2,
                        sample_rate=int(audio.sample_rate),
                        pitch=pitch,
                        rate=rate,
                    )
            except Exception:
                pass



        except Exception as e:
            # Piper is optional. If it fails, we fall back to silence and allow muxing to continue.
            try:
                print(f"[Audio] Piper TTS failed for {sid}: {e}")
            except Exception:
                pass
            return None

    elif audio.tts_engine.lower() == "none":
        return None
    else:
        raise RuntimeError(f"Unsupported TTS engine: {audio.tts_engine}")

    # Cinematic option: if speech is longer than the shot, extend the shot duration
    # instead of speeding up or trimming.
    allow_speedup = bool(getattr(audio, "auto_fit_to_scene", True))
    try:
        if bool(getattr(audio, "auto_extend_scene_to_dialogue", False)):
            raw_dur = max(0.0, probe_duration_seconds(raw))
            if raw_dur > float(scene.seconds):
                pad = float(getattr(audio, "extend_pad_seconds", 0.20))
                maxs = float(getattr(audio, "extend_max_seconds", 120.0))
                scene.seconds = float(min(maxs, raw_dur + pad))
                allow_speedup = False
    except Exception:
        # Non-fatal: we can still fit to the existing duration.
        pass

    # Fit to the (possibly updated) scene duration.
    fit_audio_to_duration(
        raw,
        fitted,
        target_seconds=float(scene.seconds),
        spec=audio,
        allow_speedup=allow_speedup,
    )
    scene.dialogue_audio_path = fitted
    return fitted



def generate_scene_vfx_audio(
    cfg: ProjectConfig,
    scene_index: int | SceneSpec,
    project_out_dir: str,
) -> Optional[str]:
    """Generate (or reuse) a fitted VFX/SFX WAV for one shot using AudioGen if available.

    If generation is not possible, returns None.
    """
    audio = cfg.audio
    if not audio.enabled:
        return None

    if isinstance(scene_index, SceneSpec):
        scene = scene_index
        try:
            idx = cfg.scenes.index(scene)
        except ValueError:
            idx = 0
    else:
        idx = int(scene_index)
        if idx < 0 or idx >= len(cfg.scenes):
            raise IndexError("scene_index out of range")
        scene = cfg.scenes[idx]

    prompt = (getattr(scene, "vfx_prompt", "") or "").strip()
    if not prompt:
        return None

    adir = ensure_audio_dir(project_out_dir, audio)
    sdir = os.path.join(adir, audio.shot_stems_dir_name)
    os.makedirs(sdir, exist_ok=True)
    sid = _shot_id(scene, idx)
    secs = float(max(0.0, float(scene.seconds)))

    raw = os.path.join(sdir, f"{sid}_vfx_aigen_raw.wav")
    fitted = os.path.join(sdir, f"{sid}_vfx.wav")

    # Generate via AudioGen (optional)
    try:
        from .audio.audiogen_wrapper import generate_vfx_wav
        gen = generate_vfx_wav(
            prompt=prompt,
            out_wav=raw,
            seconds=max(0.5, secs),
            seed=getattr(scene, "vfx_seed", None),
            try_install=True,
        )
    except Exception:
        return None

    try:
        # Normalize to project SR/stereo
        raw2 = os.path.join(sdir, f"{sid}_vfx_raw.wav")
        _ensure_wav(gen, raw2, audio.sample_rate, channels=2)
        fit_audio_to_duration(raw2, fitted, secs, audio, allow_speedup=False)
        scene.vfx_audio_path = fitted
        return fitted
    except Exception:
        return None


def _write_concat_list(paths: List[str], list_path: str) -> None:
    with open(list_path, "w", encoding="utf-8") as f:
        for p in paths:
            safe = p.replace("'", "'\\''")
            f.write(f"file '{safe}'\n")


def _concat_wavs(paths: List[str], out_wav: str, sample_rate: int) -> str:
    """Concatenate WAV files (re-encodes if copy fails)."""

    if not paths:
        raise RuntimeError("No audio clips to concat")
    ff = find_ffmpeg()
    os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)
    list_path = os.path.join(os.path.dirname(out_wav), f"_concat_{os.path.basename(out_wav)}.txt")
    _write_concat_list(paths, list_path)
    cmd_copy = [ff, "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out_wav]
    try:
        subprocess.check_call(cmd_copy)
        return out_wav
    except Exception:
        cmd = [
            ff,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-ar",
            str(int(sample_rate)),
            "-ac",
            "2",
            out_wav,
        ]
        subprocess.check_call(cmd)
        return out_wav


def _film_seconds(cfg: ProjectConfig) -> float:
    return float(sum(max(0.0, float(s.seconds)) for s in cfg.scenes))



def _music_intensity(cfg: ProjectConfig) -> float:
    vals = []
    for s in getattr(cfg, "scenes", []) or []:
        v = getattr(s, "music_intensity", None)
        if v is None:
            continue
        try:
            vals.append(float(v))
        except Exception:
            pass
    if not vals:
        return 0.5
    x = sum(vals) / max(1, len(vals))
    return max(0.0, min(1.0, x))

def build_shot_stems(cfg: ProjectConfig, project_out_dir: str) -> Tuple[List[str], List[str], Dict[str, List[int]]]:
    """Build per-shot dialogue and vfx stems (or silences).

    Returns:
      dialogue_paths, vfx_paths, scene_to_indices
    """

    audio = cfg.audio
    adir = ensure_audio_dir(project_out_dir, audio)
    shots_dir = os.path.join(adir, audio.shot_stems_dir_name)
    os.makedirs(shots_dir, exist_ok=True)

    dlg_paths: List[str] = []
    vfx_paths: List[str] = []
    scene_to_indices: Dict[str, List[int]] = {}

    for i, s in enumerate(cfg.scenes):
        scene_id = _scene_id_from_label(s.label, i)
        scene_to_indices.setdefault(scene_id, []).append(i)

        sid = _shot_id(s, i)
        secs = float(max(0.0, s.seconds))

        # --- Dialogue ---
        if (s.dialogue_text or "").strip():
            if not (s.dialogue_audio_path and os.path.isfile(s.dialogue_audio_path)):
                generate_scene_dialogue_audio(cfg, i, project_out_dir)
            if s.dialogue_audio_path and os.path.isfile(s.dialogue_audio_path):
                dlg_paths.append(s.dialogue_audio_path)
            else:
                sil = os.path.join(shots_dir, f"{sid}_dialogue_silence.wav")
                if not os.path.isfile(sil):
                    _make_silence(sil, secs, audio.sample_rate)
                dlg_paths.append(sil)
        else:
            sil = os.path.join(shots_dir, f"{sid}_dialogue_silence.wav")
            if not os.path.isfile(sil):
                _make_silence(sil, secs, audio.sample_rate)
            dlg_paths.append(sil)

        # --- VFX ---
        # If no VFX file is provided but a prompt exists, try generating via AudioGen.
        if (not (s.vfx_audio_path and os.path.isfile(s.vfx_audio_path))) and (getattr(s, 'vfx_prompt', '') or '').strip():
            try:
                generate_scene_vfx_audio(cfg, i, project_out_dir)
            except Exception:
                pass

        if s.vfx_audio_path and os.path.isfile(s.vfx_audio_path):
            raw = os.path.join(shots_dir, f"{sid}_vfx_raw.wav")
            fitted = os.path.join(shots_dir, f"{sid}_vfx.wav")
            _ensure_wav(s.vfx_audio_path, raw, audio.sample_rate, channels=2)
            # No speed-up for VFX: pad/trim only.
            fit_audio_to_duration(raw, fitted, secs, audio, allow_speedup=False)
            vfx_paths.append(fitted)
        else:
            sil = os.path.join(shots_dir, f"{sid}_vfx_silence.wav")
            if not os.path.isfile(sil):
                _make_silence(sil, secs, audio.sample_rate)
            vfx_paths.append(sil)

    return dlg_paths, vfx_paths, scene_to_indices


def build_dialogue_track(cfg: ProjectConfig, project_out_dir: str) -> str:
    """Build full-length dialogue stem by concatenating shot stems."""

    audio = cfg.audio
    adir = ensure_audio_dir(project_out_dir, audio)
    dlg_paths, _, _ = build_shot_stems(cfg, project_out_dir)
    out = os.path.join(adir, audio.track_filename)
    return _concat_wavs(dlg_paths, out, audio.sample_rate)


def build_vfx_track(cfg: ProjectConfig, project_out_dir: str) -> str:
    """Build full-length VFX stem by concatenating shot stems."""

    audio = cfg.audio
    adir = ensure_audio_dir(project_out_dir, audio)
    _, vfx_paths, _ = build_shot_stems(cfg, project_out_dir)
    out = os.path.join(adir, audio.vfx_track_filename)
    return _concat_wavs(vfx_paths, out, audio.sample_rate)


def build_scene_stems(cfg: ProjectConfig, project_out_dir: str) -> Dict[str, Dict[str, str]]:
    """Export per-scene stems (dialogue/vfx) as concatenated WAV files.

    Returns mapping:
      scene_id -> {"dialogue": path, "vfx": path}
    """

    audio = cfg.audio
    adir = ensure_audio_dir(project_out_dir, audio)
    scenes_dir = os.path.join(adir, audio.scene_stems_dir_name)
    os.makedirs(scenes_dir, exist_ok=True)

    dlg_paths, vfx_paths, scene_to_indices = build_shot_stems(cfg, project_out_dir)
    out_map: Dict[str, Dict[str, str]] = {}

    for scene_id, indices in scene_to_indices.items():
        dlg = [dlg_paths[i] for i in indices]
        vfx = [vfx_paths[i] for i in indices]
        out_d = os.path.join(scenes_dir, f"{scene_id}_dialogue.wav")
        out_v = os.path.join(scenes_dir, f"{scene_id}_vfx.wav")
        _concat_wavs(dlg, out_d, audio.sample_rate)
        _concat_wavs(vfx, out_v, audio.sample_rate)
        out_map[scene_id] = {"dialogue": out_d, "vfx": out_v}

    return out_map



def build_music_track(cfg: ProjectConfig, project_out_dir: str) -> str:
    """Build a film-length music bed (loops if needed).

    Priority:
      1) audio.music_path if provided
      2) auto-generate (MusicGen if available) when audio.music_auto_generate=True
      3) placeholder ambience (when audio.music_placeholder_if_missing=True)
      4) silence
    """

    audio = cfg.audio
    adir = ensure_audio_dir(project_out_dir, audio)
    total = _film_seconds(cfg)
    out = os.path.join(adir, audio.music_track_filename)

    # Source music
    src = None
    if audio.music_path and os.path.isfile(audio.music_path):
        src = audio.music_path
    elif bool(getattr(audio, "music_auto_generate", False)):
        try:
            from .music_presets import build_prompt
            from .music_gen import generate_music_wav
            preset = getattr(audio, "music_style_preset", "cinematic_orchestral")
            prompt = build_prompt(str(preset), intensity=_music_intensity(cfg))
            gen_raw = os.path.join(adir, "_music_generated.wav")
            generate_music_wav(
                prompt=prompt,
                out_wav=gen_raw,
                seconds=min(float(getattr(audio, "music_max_seconds", 60.0)), float(total)),
                sample_rate=int(audio.sample_rate),
                model_name=str(getattr(audio, "music_model_name", "facebook/musicgen-small")),
                seed=getattr(audio, "music_seed", None),
            )
            if os.path.isfile(gen_raw):
                src = gen_raw
        except Exception as e:
            # MusicGen isn't installed / crashed / no GPU etc.
            # Fall back to a simple procedural ambience so the master mix isn't silent.
            try:
                print(f"[Audio] MusicGen failed ({type(e).__name__}: {e}). Using placeholder ambience.")
            except Exception:
                pass
            try:
                _make_placeholder_music(out, total, int(audio.sample_rate), style=str(getattr(audio, "music_placeholder_style", "ambient")))
                return out
            except Exception:
                src = None

    if not src:
        # Fallback: placeholder ambience (if enabled) or silence
        if bool(getattr(audio, "music_placeholder_if_missing", False)):
            try:
                _make_placeholder_music(out, total, int(audio.sample_rate), style=str(getattr(audio, "music_placeholder_style", "ambient")))
                return out
            except Exception:
                pass
        _make_silence(out, total, audio.sample_rate)
        return out

    raw = os.path.join(adir, "_music_raw.wav")
    _ensure_wav(src, raw, audio.sample_rate, channels=2)

    ff = find_ffmpeg()
    cmd = [
        ff,
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        raw,
        "-t",
        f"{total:.6f}",
        "-af",
        f"atrim=0:{total:.6f},asetpts=N/SR/TB",
        "-ar",
        str(int(audio.sample_rate)),
        "-ac",
        "2",
        out,
    ]
    subprocess.check_call(cmd)
    return out



def build_master_mix(cfg: ProjectConfig, project_out_dir: str) -> str:
    """Create master mix (music + dialogue + vfx), with optional ducking."""

    audio = cfg.audio
    adir = ensure_audio_dir(project_out_dir, audio)
    dlg = build_dialogue_track(cfg, project_out_dir)
    vfx = build_vfx_track(cfg, project_out_dir)
    music = build_music_track(cfg, project_out_dir)

    ff = find_ffmpeg()
    out = os.path.join(adir, audio.master_track_filename)
    def _vol(gain_db: float, muted: bool) -> str:
        return "volume=0" if muted else f"volume={gain_db}dB"

    music_vol = _vol(audio.music_gain_db, bool(getattr(audio, "music_muted", False)))
    dlg_vol = _vol(audio.dialogue_gain_db, bool(getattr(audio, "dialogue_muted", False)))
    vfx_vol = _vol(audio.vfx_gain_db, bool(getattr(audio, "vfx_muted", False)))
    master_vol = _vol(float(getattr(audio, "master_gain_db", 0.0)), False)

    if audio.duck_music_under_dialogue:
        sc = (
            f"sidechaincompress=threshold={audio.duck_threshold}:"
            f"ratio={audio.duck_ratio}:attack={audio.duck_attack}:release={audio.duck_release}"
        )
        filt = (
            f"[0:a]{music_vol}[m0];"
            f"[1:a]{dlg_vol},asplit=2[d_sc][d_mix];"
            f"[2:a]{vfx_vol}[v0];"
            f"[m0][d_sc]{sc}[mduck];"
            f"[mduck][d_mix][v0]amix=inputs=3:normalize=0,{master_vol},alimiter=limit=0.95[aout]"
        )
    else:
        filt = (
            f"[0:a]{music_vol}[m0];"
            f"[1:a]{dlg_vol}[d0];"
            f"[2:a]{vfx_vol}[v0];"
            f"[m0][d0][v0]amix=inputs=3:normalize=0,{master_vol},alimiter=limit=0.95[aout]"
        )

    cmd = [
        ff,
        "-y",
        "-i",
        music,
        "-i",
        dlg,
        "-i",
        vfx,
        "-filter_complex",
        filt,
        "-map",
        "[aout]",
        "-ar",
        str(int(audio.sample_rate)),
        "-ac",
        "2",
        out,
    ]
    subprocess.check_call(cmd)
    return out


def mux_audio(video_path: str, audio_path: str, out_path: str, *, audio_codec: str = "aac", aac_bitrate: str = "320k") -> str:
    """Mux a WAV/track into an existing video WITHOUT re-encoding the video.

    - Video is always copied: -c:v copy
    - Audio codec presets:
        - aac (lossy, MP4-friendly) [default bitrate 320k]
        - flac (lossless, MKV-friendly)
        - pcm_s16le (lossless, MKV/MOV-friendly)
        - copy (copy audio stream as-is)

    Note: MP4 generally doesn't support FLAC/PCM; if you request flac/pcm into an .mp4,
    we fall back to AAC.
    """
    ff = find_ffmpeg()

    # Container compatibility guard
    ext = os.path.splitext(out_path)[1].lower()
    if ext in (".mp4", ".m4v", ".mov") and audio_codec not in ("aac", "copy"):
        audio_codec = "aac"

    cmd = [
        ff,
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
    ]

    if audio_codec == "copy":
        cmd += ["-c:a", "copy"]
    elif audio_codec == "aac":
        cmd += ["-c:a", "aac", "-b:a", str(aac_bitrate)]
    else:
        cmd += ["-c:a", str(audio_codec)]

    cmd += ["-shortest", out_path]

    subprocess.check_call(cmd)
    return out_path


def mux_master_audio(video_path: str, cfg: ProjectConfig, project_out_dir: str, out_path: str, *, audio_codec: str = "aac", aac_bitrate: str = "320k") -> str:
    """Build master mix and mux it into a video (copy video)."""
    master = build_master_mix(cfg, project_out_dir)
    return mux_audio(video_path, master, out_path, audio_codec=audio_codec, aac_bitrate=aac_bitrate)



def _format_srt_time(t: float) -> str:
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000.0))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def export_subtitles_srt(cfg: ProjectConfig, project_out_dir: str) -> Optional[str]:
    """Export a simple SRT based on per-shot dialogue_text.

    This is *not* word-level alignment. It's robust, fast, and matches the timeline.
    (WhisperX alignment can be added later as an optional post-step.)
    """

    audio = cfg.audio
    if not audio.export_srt:
        return None
    adir = ensure_audio_dir(project_out_dir, audio)
    out = os.path.join(adir, audio.srt_filename)

    lines: List[str] = []
    t0 = 0.0
    idx = 1
    for s in cfg.scenes:
        dur = float(max(0.0, s.seconds))
        txt = (s.dialogue_text or "").strip()
        if txt:
            start = t0
            end = t0 + dur
            lines.append(str(idx))
            lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
            lines.append(txt)
            lines.append("")
            idx += 1
        t0 += dur

    if not lines:
        # Still write an empty file so users can find it.
        with open(out, "w", encoding="utf-8") as f:
            f.write("")
        return out

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")
    return out


def build_all_audio(cfg: ProjectConfig, project_out_dir: str) -> Dict[str, object]:
    """One-call audio build for the UI.

    Produces:
      - per-shot stems (dialogue/vfx)
      - per-scene stems (optional)
      - global stems (dialogue/vfx/music)
      - master mix
      - subtitles (optional)
    """

    audio = cfg.audio
    if not audio.enabled:
        return {"enabled": False}

    ensure_audio_dir(project_out_dir, audio)
    dlg_track = build_dialogue_track(cfg, project_out_dir)
    vfx_track = build_vfx_track(cfg, project_out_dir)
    music_track = build_music_track(cfg, project_out_dir)
    master = build_master_mix(cfg, project_out_dir)

    scene_stems: Optional[Dict[str, Dict[str, str]]] = None
    if audio.export_scene_stems:
        scene_stems = build_scene_stems(cfg, project_out_dir)

    srt_path: Optional[str] = None
    if audio.export_srt:
        srt_path = export_subtitles_srt(cfg, project_out_dir)

    return {
        "enabled": True,
        "dialogue_track": dlg_track,
        "vfx_track": vfx_track,
        "music_track": music_track,
        "master_track": master,
        "scene_stems": scene_stems,
        "subtitles_srt": srt_path,
    }
