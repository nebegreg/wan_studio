from __future__ import annotations

from typing import List, Tuple, Optional

from .config import ProjectConfig, SceneSpec, CharacterSpec, LocationSpec

_BANNED_STYLE_TOKENS = [
    "anime", "cartoon", "illustration", "comic", "manga", "toon", "pixar", "cgi", "3d render",
]


def _norm(s: str) -> str:
    return (s or "").strip()


def _find_character(cfg: ProjectConfig, name: str) -> Optional[CharacterSpec]:
    n = _norm(name).lower()
    for c in getattr(cfg, "characters", []) or []:
        if _norm(getattr(c, "name", "")).lower() == n:
            return c
    return None


def _find_location(cfg: ProjectConfig, name: str) -> Optional[LocationSpec]:
    n = _norm(name).lower()
    for l in getattr(cfg, "locations", []) or []:
        if _norm(getattr(l, "name", "")).lower() == n:
            return l
    return None


def merge_negatives(parts: List[str]) -> str:
    toks: List[str] = []
    seen = set()
    for p in parts:
        for t in _norm(p).replace(";", ",").split(","):
            tt = t.strip()
            if not tt:
                continue
            k = tt.lower()
            if k in seen:
                continue
            seen.add(k)
            toks.append(tt)
    return ", ".join(toks)


def build_scene_prompts(cfg: ProjectConfig, scene: SceneSpec) -> Tuple[str, str]:
    """Prompt injection for Phase C.

    - prompt = scene.prompt + character prompts + location prompt
    - negative = cfg.negative_prompt + scene.negative_prompt + character/location negatives + anti-toon guard
    """
    base = _norm(getattr(scene, "prompt", ""))
    extras: List[str] = []

    # characters
    for nm in (getattr(scene, "characters", []) or []):
        c = _find_character(cfg, nm)
        if c is None:
            if _norm(nm):
                extras.append(_norm(nm))
            continue
        extras.append(_norm(c.prompt) or _norm(c.description) or _norm(c.name))

    # location
    loc = _norm(getattr(scene, "location", ""))
    if loc:
        l = _find_location(cfg, loc)
        extras.append(_norm(getattr(l, "prompt", "")) or _norm(getattr(l, "description", "")) or loc)

    # shot meta (from storyboard)
    meta = []
    st = _norm(getattr(scene, 'shot_type', ''))
    cm = _norm(getattr(scene, 'camera_move', ''))
    mood = _norm(getattr(scene, 'mood', ''))
    mi = _norm(getattr(scene, 'music_intensity', ''))
    if st:
        meta.append(f"{st} shot")
    if cm:
        meta.append(f"camera move: {cm}")
    if mood:
        meta.append(f"mood: {mood}")
    if mi:
        meta.append(f"music intensity: {mi}")
    if meta:
        extras.append(' | '.join(meta))

    prompt = base
    if extras:
        prompt = (prompt + "\n\n" + "\n".join([e for e in extras if e])).strip()

    neg_parts = [
        _norm(getattr(cfg, "negative_prompt", "")),
        _norm(getattr(scene, "negative_prompt", "")),
    ]

    for nm in (getattr(scene, "characters", []) or []):
        c = _find_character(cfg, nm)
        if c and _norm(getattr(c, "negative_prompt", "")):
            neg_parts.append(_norm(c.negative_prompt))

    if loc:
        l = _find_location(cfg, loc)
        if l and _norm(getattr(l, "negative_prompt", "")):
            neg_parts.append(_norm(l.negative_prompt))

    neg_parts.append(", ".join(_BANNED_STYLE_TOKENS))
    negative = merge_negatives([p for p in neg_parts if p])
    return prompt, negative
