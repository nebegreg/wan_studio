from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class MusicPreset:
    key: str
    title: str
    prompt: str


# NOTE: we avoid naming living composers; we use descriptive styles and some public-domain references.
PRESETS: Dict[str, MusicPreset] = {
    "cinematic_orchestral": MusicPreset(
        key="cinematic_orchestral",
        title="Cinematic orchestral",
        prompt="cinematic orchestral score, modern film trailer feel, lush strings, brass swells, subtle percussion, wide reverb",
    ),
    "noir_jazz": MusicPreset(
        key="noir_jazz",
        title="Noir jazz",
        prompt="noir jazz underscore, muted trumpet, double bass, brushed drums, smoky club ambience, sparse harmony",
    ),
    "baroque_bach": MusicPreset(
        key="baroque_bach",
        title="Baroque (Bach-like)",
        prompt="baroque chamber music, harpsichord continuo, string ensemble, contrapuntal textures, clean room ambience",
    ),
    "classical_mozart": MusicPreset(
        key="classical_mozart",
        title="Classical (Mozart-like)",
        prompt="classical era orchestral music, light strings, playful woodwinds, clear melodic phrases, balanced dynamics",
    ),
    "impressionist_debussy": MusicPreset(
        key="impressionist_debussy",
        title="Impressionist (Debussy-like)",
        prompt="impressionist orchestral colors, soft piano, shimmering strings, airy woodwinds, floating harmonies",
    ),
    "synthwave_80s": MusicPreset(
        key="synthwave_80s",
        title="Synthwave 80s",
        prompt="synthwave, analog synth pads, arpeggiated bass, vintage drum machine, neon night drive ambience",
    ),
}


def build_prompt(preset_key: str, intensity: float = 0.5) -> str:
    p = PRESETS.get(preset_key or "", PRESETS["cinematic_orchestral"]).prompt
    # Intensity hint: keep in text domain for MusicGen.
    intensity = max(0.0, min(1.0, float(intensity)))
    if intensity < 0.35:
        return p + ", calm, minimal, low intensity"
    if intensity > 0.75:
        return p + ", intense, high energy, driving"
    return p + ", medium intensity"
