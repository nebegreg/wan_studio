from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from .config import ProjectConfig, CharacterSpec, SceneSpec


def _norm(s: str) -> str:
    return (s or "").strip()


def _infer_gender_from_filename(fn: str) -> str:
    n = fn.lower()
    if any(t in n for t in ("female", "woman", "fem")):
        return "female"
    if any(t in n for t in ("male", "man", "masc")):
        return "male"
    return ""


def _infer_lang_from_filename(fn: str) -> str:
    # Common piper naming: fr_FR-xxx.onnx
    base = os.path.basename(fn)
    parts = base.split("-")
    if parts and "_" in parts[0]:
        return parts[0].split("_")[0].lower()
    # fallback: look for '_en' tokens etc
    low = base.lower()
    for k in ("fr", "en", "es", "de", "it", "pt", "nl", "ru", "ja", "ko", "zh"):
        if f"{k}_" in low or f"_{k}" in low or low.startswith(k + "-"):
            return k
    return ""


def list_piper_voices(tools_dir: str) -> List[Dict[str, str]]:
    voices_dir = os.path.join(os.path.abspath(tools_dir), "piper", "voices")
    out: List[Dict[str, str]] = []
    if not os.path.isdir(voices_dir):
        return out
    for fn in sorted(os.listdir(voices_dir)):
        if not fn.endswith(".onnx"):
            continue
        model_path = os.path.join(voices_dir, fn)
        cfg_path = model_path + ".json" if os.path.isfile(model_path + ".json") else (model_path + ".onnx.json" if os.path.isfile(model_path + ".onnx.json") else "")
        out.append(
            {
                "id": fn,
                "model_path": model_path,
                "config_path": cfg_path,
                "lang": _infer_lang_from_filename(fn),
                "gender": _infer_gender_from_filename(fn),
            }
        )
    return out


def _find_character(cfg: ProjectConfig, name: str) -> Optional[CharacterSpec]:
    n = _norm(name).lower()
    for c in getattr(cfg, "characters", []) or []:
        if _norm(getattr(c, "name", "")).lower() == n:
            return c
    return None


def _resolve_voice_path(tools_dir: str, model_or_id: str) -> Tuple[str, str]:
    v = _norm(model_or_id)
    if not v:
        return "", ""
    # absolute path
    if os.path.isfile(v):
        cfg = v + ".json" if os.path.isfile(v + ".json") else (v + ".onnx.json" if os.path.isfile(v + ".onnx.json") else "")
        return v, cfg

    voices_dir = os.path.join(os.path.abspath(tools_dir), "piper", "voices")

    # exact match (may already include .onnx)
    cand = os.path.join(voices_dir, v)
    if os.path.isfile(cand):
        cfg = cand + ".json" if os.path.isfile(cand + ".json") else (cand + ".onnx.json" if os.path.isfile(cand + ".onnx.json") else "")
        return cand, cfg

    # common: user specifies id without extension (e.g. fr_FR-siwis-medium)
    if not v.endswith(".onnx"):
        cand2 = os.path.join(voices_dir, v + ".onnx")
        if os.path.isfile(cand2):
            cfg = (
                cand2 + ".json"
                if os.path.isfile(cand2 + ".json")
                else (cand2 + ".onnx.json" if os.path.isfile(cand2 + ".onnx.json") else "")
            )
            if not cfg:
                cfg_only = os.path.join(voices_dir, v + ".onnx.json")
                if os.path.isfile(cfg_only):
                    cfg = cfg_only
            return cand2, cfg

    return "", ""


def assign_missing_character_voices(cfg: ProjectConfig) -> int:
    """Best-effort: assign a unique piper voice per character.

    Returns number of assignments performed.
    """
    tools_dir = cfg.post.tools_dir if getattr(cfg, "post", None) and cfg.post.tools_dir else os.path.abspath("./tools")
    voices = list_piper_voices(tools_dir)
    if not voices:
        return 0

    used = set()
    for c in getattr(cfg, "characters", []) or []:
        mp = _norm(getattr(c, "voice_model_path", ""))
        if mp:
            used.add(os.path.abspath(mp))

    def pick(lang: str, gender: str) -> Optional[Dict[str, str]]:
        lang = (lang or "").lower()
        gender = (gender or "").lower()
        # prefer matching lang+gender, then lang, then any
        tiers = [
            [v for v in voices if (not lang or v.get("lang") == lang) and (not gender or v.get("gender") == gender)],
            [v for v in voices if (not lang or v.get("lang") == lang)],
            voices,
        ]
        for tier in tiers:
            for v in tier:
                mp = os.path.abspath(v["model_path"])
                if mp in used:
                    continue
                return v
        # allow reuse if needed
        return tiers[0][0] if tiers[0] else (tiers[1][0] if tiers[1] else (voices[0] if voices else None))

    count = 0
    default_lang = (getattr(cfg.audio, "default_language", "") if getattr(cfg, "audio", None) else "") or ""

    for c in getattr(cfg, "characters", []) or []:
        if _norm(getattr(c, "voice_model_path", "")):
            continue
        lang = _norm(getattr(c, "voice_language", "")) or default_lang
        gender = _norm(getattr(c, "voice_gender", ""))
        v = pick(lang, gender)
        if not v:
            continue
        c.voice_model_path = v.get("model_path", "")
        c.voice_config_path = v.get("config_path", "")
        if not _norm(getattr(c, "voice_language", "")):
            c.voice_language = v.get("lang", "") or lang
        if not _norm(getattr(c, "voice_gender", "")):
            c.voice_gender = v.get("gender", "") or gender
        used.add(os.path.abspath(c.voice_model_path))
        count += 1

    return count


def resolve_voice_for_scene(scene: SceneSpec, cfg: ProjectConfig) -> Tuple[str, Optional[str]]:
    """Resolve voice model/config for a given scene.

    Priority:
      1) scene.dialogue_voice (path or id)
      2) first character in scene.characters with voice set
      3) project default cfg.audio.voice_model_path
    """
    tools_dir = cfg.post.tools_dir if getattr(cfg, "post", None) and cfg.post.tools_dir else os.path.abspath("./tools")

    # explicit override
    if _norm(getattr(scene, "dialogue_voice", "")):
        mp, cp = _resolve_voice_path(tools_dir, scene.dialogue_voice)
        if mp:
            return mp, cp or None

    # character voice
    for nm in getattr(scene, "characters", []) or []:
        c = _find_character(cfg, nm)
        if not c:
            continue
        mp = _norm(getattr(c, "voice_model_path", ""))
        if mp:
            # allow absolute or id
            if os.path.isfile(mp):
                cp = _norm(getattr(c, "voice_config_path", ""))
                if not cp:
                    cp_guess = mp + ".json" if os.path.isfile(mp + ".json") else (mp + ".onnx.json" if os.path.isfile(mp + ".onnx.json") else "")
                    cp = cp_guess
                return mp, cp or None
            mp2, cp2 = _resolve_voice_path(tools_dir, mp)
            if mp2:
                cp = _norm(getattr(c, "voice_config_path", "")) or cp2
                return mp2, (cp or None)

    # global
    return cfg.audio.voice_model_path, cfg.audio.voice_config_path
