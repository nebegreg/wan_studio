from __future__ import annotations

import os


def run() -> None:
    from wan_studio.config import ProjectConfig, SceneSpec
    from wan_studio.clip_cache import compute_clip_cache_key

    cfg = ProjectConfig(project_name="smoke", output_dir=os.path.abspath("./o"), scenes=[])
    scene = SceneSpec(label="S", seconds=1.0, prompt="A car chase", negative_prompt="anime", characters=["A"], location="Street")

    k1 = compute_clip_cache_key(cfg, scene, effective_backend="wan", effective_model_id="dummy", effective_mode="T2V", conditioning_image=None)
    k2 = compute_clip_cache_key(cfg, scene, effective_backend="wan", effective_model_id="dummy", effective_mode="T2V", conditioning_image=None)
    assert k1 == k2, "Clip cache key should be stable across repeated calls"
