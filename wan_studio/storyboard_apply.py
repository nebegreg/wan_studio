from __future__ import annotations

from typing import Callable, Iterable, List, Optional

from .config import CharacterSpec, LocationSpec, ProjectConfig, SceneSpec


def apply_ai_scenes_in_place(
    cfg: ProjectConfig,
    scenes: Iterable[SceneSpec],
    *,
    log: Optional[Callable[[str], None]] = None,
) -> None:
    """Apply storyboard scenes while keeping object identity when possible.

    This is the non-Qt core used by the UI. It updates existing SceneSpec objects
    in-place so timeline/audio references remain valid.

    It also:
    - Respects per-scene locks: lock_prompt, lock_dialogue, lock_cast, lock_location, lock_shot, lock_music
    - Ensures newly referenced characters/locations are added to the project libraries
    - Clears per-shot generated dialogue audio when dialogue text changes (and not locked)
    """

    def _log(msg: str):
        if log:
            try:
                log(msg)
            except Exception:
                pass

    incoming = list(scenes or [])

    if not getattr(cfg, "scenes", None):
        cfg.scenes = incoming
    else:
        old = list(cfg.scenes)
        by_label = {}
        for s in old:
            lab = getattr(s, "label", None)
            if lab:
                by_label.setdefault(str(lab), []).append(s)

        new_list: List[SceneSpec] = []
        for i, ns in enumerate(incoming):
            tgt = None
            lab = getattr(ns, "label", None)
            if lab and str(lab) in by_label and by_label[str(lab)]:
                tgt = by_label[str(lab)].pop(0)
            elif i < len(old):
                tgt = old[i]

            if tgt is None:
                new_list.append(ns)
                continue

            prev_dialogue = getattr(tgt, "dialogue_text", "") or ""

            fields = (
                "label","tags","notes","seconds","prompt","negative_prompt",
                "characters","location","hard_cut","shot_type","camera_move","mood","music_intensity",
                "model_id_override","mode_override","input_image_path_override",
                "use_character_ref","style_pack","backend_override",
                "seed","num_inference_steps","guidance_scale","guidance_scale_2","boundary_ratio",
                "blend_mode","transition_mode","transition_frames","transition_ease",
                "dialogue_text","dialogue_language","dialogue_voice",
                "vfx_audio_path",
            )

            for k in fields:
                if not hasattr(ns, k):
                    continue

                # lock-aware skip
                try:
                    if k in ("prompt","negative_prompt","tags","notes") and bool(getattr(tgt, "lock_prompt", False)):
                        continue
                    if k in ("dialogue_text","dialogue_language","dialogue_voice") and bool(getattr(tgt, "lock_dialogue", False)):
                        continue
                    if k in ("characters",) and bool(getattr(tgt, "lock_cast", False)):
                        continue
                    if k in ("location",) and bool(getattr(tgt, "lock_location", False)):
                        continue
                    if k in ("shot_type","camera_move","mood","hard_cut") and bool(getattr(tgt, "lock_shot", False)):
                        continue
                    if k in ("music_intensity",) and bool(getattr(tgt, "lock_music", False)):
                        continue
                except Exception:
                    pass

                setattr(tgt, k, getattr(ns, k))

            # If dialogue changed (and not locked), clear generated dialogue audio
            if not bool(getattr(tgt, "lock_dialogue", False)) and (getattr(tgt, "dialogue_text", "") or "") != prev_dialogue:
                try:
                    tgt.dialogue_audio_path = None
                except Exception:
                    pass

            new_list.append(tgt)

        cfg.scenes = new_list

    # Ensure Character Bible / Location Library contain any newly referenced names
    try:
        added_c = 0
        added_l = 0
        chars = getattr(cfg, "characters", None) or []
        locs = getattr(cfg, "locations", None) or []
        existing_c = { (getattr(c, "name", "") or "").strip().lower(): c for c in chars if (getattr(c, "name", "") or "").strip() }
        existing_l = { (getattr(l, "name", "") or "").strip().lower(): l for l in locs if (getattr(l, "name", "") or "").strip() }

        for s in (cfg.scenes or []):
            for nm in (getattr(s, "characters", []) or []):
                key = (str(nm) or "").strip()
                if not key:
                    continue
                lk = key.lower()
                if lk not in existing_c:
                    chars.append(CharacterSpec(name=key))
                    existing_c[lk] = chars[-1]
                    added_c += 1

            loc = (getattr(s, "location", "") or "").strip()
            if loc:
                lk = loc.lower()
                if lk not in existing_l:
                    locs.append(LocationSpec(name=loc))
                    existing_l[lk] = locs[-1]
                    added_l += 1

        cfg.characters = chars
        cfg.locations = locs
        if added_c or added_l:
            _log(f"Bible auto: +{added_c} personnage(s), +{added_l} lieu(x).")
    except Exception:
        pass

    # Auto-assign missing character voices (best-effort)
    try:
        from .voice_manager import assign_missing_character_voices
        n = assign_missing_character_voices(cfg)
        if n:
            _log(f"Voices auto-assign: {n} personnage(s).")
    except Exception:
        pass
