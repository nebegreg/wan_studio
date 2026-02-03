from __future__ import annotations

import os


def run() -> None:
    from wan_studio.config import ProjectConfig, SceneSpec, InitImageSpec
    from wan_studio.flux2_init import compute_init_cache_key

    cfg = ProjectConfig(project_name="smoke", output_dir=os.path.abspath("./o"), scenes=[])
    cfg.init_image = InitImageSpec(enabled=True, preset="flux2_bnb4bit", policy="necessary", width=640, height=360, steps=12, guidance_scale=3.5)

    scene = SceneSpec(label="S", seconds=1.0, prompt="A cat in a cinema", negative_prompt="anime")

    k1, *_ = compute_init_cache_key(scene, cfg)
    k2, *_ = compute_init_cache_key(scene, cfg)
    assert k1 == k2, "Init cache key should be stable across repeated calls"
