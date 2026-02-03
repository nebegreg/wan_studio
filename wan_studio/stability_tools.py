from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class DriftRisk:
    """Lightweight stability score used by the GUI (timeline badges + Doctor).

    score: 0..100 (higher = riskier)
    icon: 🟢 (low), 🟡 (medium), 🔴 (high)
    """

    score: int
    icon: str
    title: str
    details: str


def compute_drift_risk(*, cfg, scene, scene_index: int = 0) -> DriftRisk:
    """Heuristic drift/instability risk.

    Goal: catch the common causes of:
      - clip-to-clip bleed (conditioning/memory spill)
      - chunk seam pops (frame-0 mismatch)
      - long-shot drift (identity/background morph)

    This is intentionally conservative and explainable (reasons are returned).
    """

    score = 0
    reasons: List[str] = []

    # --- Anchors / conditioning ---
    # In I2V, lack of a stable anchor is the #1 drift driver.
    try:
        mode = str(getattr(scene, "mode_override", None) or getattr(cfg, "mode", "")).upper()
    except Exception:
        mode = "I2V"

    if mode == "I2V":
        has_anchor = False
        for k in ("input_image_path_override", "init_frame_path"):
            if getattr(scene, k, None):
                has_anchor = True
                break
        if not has_anchor and getattr(cfg, "input_image_path", None):
            has_anchor = True

        if not has_anchor:
            score += 35
            reasons.append("I2V sans image d’ancrage (input/ init-frame) → drift probable")

    # --- Clip-to-clip spill ---
    if scene_index > 0:
        if not bool(getattr(cfg, "reset_continuity_between_clips", False)):
            score += 18
            reasons.append("Reset continuity désactivé → bleed entre clips")

    # Ultra isolation: the engine unloads/reloads pipelines between clips.
    # This reduces the chance of hidden state carry (LoRA/adapters/memory).
    if bool(getattr(cfg, 'ultra_isolated_clips', False)) and scene_index > 0:
        score = max(0, score - 12)
        reasons.append('Ultra isolation activée → reset pipeline/VRAM entre clips (anti-contamination)')

    # Per-clip hard cut helps even if global is off
    if bool(getattr(scene, "hard_cut", False)):
        score = max(0, score - 6)
        reasons.append("Hard cut activé (bon point)")

    # --- Chunking / seams ---
    try:
        chunk_seconds = float(getattr(cfg, "chunk_seconds", 0.0) or 0.0)
    except Exception:
        chunk_seconds = 0.0

    if chunk_seconds >= 6.0:
        score += 20
        reasons.append(f"Chunk long ({chunk_seconds:.1f}s) → drift plus probable")
    elif chunk_seconds >= 4.0:
        score += 10
        reasons.append(f"Chunk moyen ({chunk_seconds:.1f}s) → drift possible")
    elif chunk_seconds <= 0.0:
        # no chunking: not necessarily bad, but for long shots it can drift.
        score += 6
        reasons.append("Pas de chunking (shot long) → drift possible")

    try:
        overlap = int(getattr(cfg, "overlap_frames", 0) or 0)
    except Exception:
        overlap = 0
    if overlap < 6:
        score += 14
        reasons.append(f"Overlap faible ({overlap}f) → raccords plus visibles")
    elif overlap > 24:
        score += 6
        reasons.append(f"Overlap très élevé ({overlap}f) → ghosting/artefacts possibles")

    if not bool(getattr(cfg, "force_first_frame_to_conditioning", False)):
        score += 12
        reasons.append("First-frame lock désactivé → pops possibles au début de segment")

    if not bool(getattr(cfg, "strict_clip_coherence", False)):
        score += 18
        reasons.append("Strict clip coherence désactivé → drift intra-clip")

    # --- Sampling sanity ---
    steps = getattr(scene, "num_inference_steps", None)
    if steps is None:
        steps = getattr(cfg, "num_inference_steps", 0)
    try:
        steps = int(steps or 0)
    except Exception:
        steps = 0
    if steps and steps < 12:
        score += 10
        reasons.append(f"Steps bas ({steps}) → instabilité/artefacts")

    g = getattr(scene, "guidance_scale", None)
    if g is None:
        g = getattr(cfg, "guidance_scale", None)
    try:
        g = float(g) if g is not None else 0.0
    except Exception:
        g = 0.0
    if g >= 7.0:
        score += 8
        reasons.append(f"Guidance élevée ({g:.1f}) → sur-contrainte / flicker")
    elif 0.0 < g <= 2.0:
        score += 6
        reasons.append(f"Guidance faible ({g:.1f}) → drift/variance")

    # --- Seeds ---
    if getattr(scene, "seed", None) is None and not bool(getattr(cfg, "strict_clip_coherence", False)):
        score += 6
        reasons.append("Seed non verrouillé → variance plus élevée")

    # Clamp
    score = max(0, min(100, int(round(score))))
    if score <= 20:
        icon = "🟢"
        title = "Faible risque"
    elif score <= 45:
        icon = "🟡"
        title = "Risque moyen"
    else:
        icon = "🔴"
        title = "Risque élevé"

    details = "\n".join(reasons) if reasons else "Paramètres OK."
    return DriftRisk(score=score, icon=icon, title=title, details=details)
