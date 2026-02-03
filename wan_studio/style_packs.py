from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Built-in packs. You can extend them by dropping a JSON file at:
#   ./style_packs.json
# with the same schema.
_DEFAULT_PACKS: Dict[str, dict] = {
    "Cinematic Product (Realism)": {
        "prompt_prefix": "cinematic product commercial, ultra realistic, 35mm film look, shallow depth of field, high dynamic range, natural color grading, film grain, crisp focus, ",
        "prompt_suffix": ", professional advertising shot, studio quality lighting, bokeh, smooth camera movement",
        "negative_add": "lowres, blurry, out of focus, bad anatomy, deformed, extra limbs, extra fingers, bad hands, watermark, text, logo, jpeg artifacts, oversharpen, overexposed, underexposed, noisy, flicker, temporal artifacts",
        "suggest": {"guidance_scale": 4.7, "steps": 20},
    },
    "Cinematic (General Realism)": {
        "prompt_prefix": "cinematic, ultra realistic, natural lighting, 35mm, shallow depth of field, film grain, ",
        "prompt_suffix": ", smooth camera motion, consistent subject, clean edges",
        "negative_add": "lowres, blurry, bad anatomy, deformed, extra limbs, extra fingers, watermark, text, logo, jpeg artifacts, flicker, temporal artifacts",
        "suggest": {"guidance_scale": 4.2, "steps": 18},
    },
    "Cinematic Ultra (Photoreal Max)": {
        "prompt_prefix": "cinematic, ultra photorealistic, natural lighting, 35mm film, shallow depth of field, high dynamic range, accurate skin texture, realistic materials, film grain, ",
        "prompt_suffix": ", smooth camera motion, stable composition, consistent subject identity, clean edges, no artifacts",
        "negative_add": "lowres, blurry, out of focus, oversharpen, noisy, jpeg artifacts, watermark, text, logo, cgi, 3d render, plastic skin, waxy, doll, cartoon, anime, illustration, comic, pixar, flicker, temporal jitter, warping, frame tearing",
        "suggest": {"guidance_scale": 4.8, "steps": 22},
    },
    "Anime / Cartoon": {
        "prompt_prefix": "high quality anime, clean lineart, vibrant colors, ",
        "prompt_suffix": ", smooth animation, consistent character, clean edges",
        "negative_add": "photo, realistic skin texture, noise, grain, lowres, blurry, watermark, text, logo, flicker, temporal artifacts",
        "suggest": {"guidance_scale": 3.0, "steps": 12},
    },
    "Illustration / Concept Art": {
        "prompt_prefix": "high quality illustration, concept art, painterly, ",
        "prompt_suffix": ", cinematic composition, consistent subject, smooth motion",
        "negative_add": "photo, lowres, blurry, watermark, text, logo, flicker, temporal artifacts",
        "suggest": {"guidance_scale": 3.5, "steps": 14},
    },
}

@dataclass
class StylePack:
    name: str
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    negative_add: str = ""
    suggest: Dict[str, float] | None = None

def _load_json(path: str) -> Dict[str, dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def list_packs(extra_path: Optional[str] = None) -> List[str]:
    packs = dict(_DEFAULT_PACKS)
    if extra_path and os.path.isfile(extra_path):
        packs.update(_load_json(extra_path))
    return sorted(packs.keys())

def get_pack(name: str, extra_path: Optional[str] = None) -> StylePack:
    packs = dict(_DEFAULT_PACKS)
    if extra_path and os.path.isfile(extra_path):
        packs.update(_load_json(extra_path))
    spec = packs.get(name, {}) if isinstance(packs.get(name, {}), dict) else {}
    return StylePack(
        name=name,
        prompt_prefix=str(spec.get("prompt_prefix", "")),
        prompt_suffix=str(spec.get("prompt_suffix", "")),
        negative_add=str(spec.get("negative_add", "")),
        suggest=spec.get("suggest") if isinstance(spec.get("suggest"), dict) else None,
    )

def apply_pack(prompt: str, negative: str, pack: StylePack) -> Tuple[str, str]:
    p = (pack.prompt_prefix or "") + (prompt or "") + (pack.prompt_suffix or "")
    n = (negative or "").strip()
    add = (pack.negative_add or "").strip()
    if add:
        n = (n + ", " + add) if n else add
    return p.strip(), n.strip()