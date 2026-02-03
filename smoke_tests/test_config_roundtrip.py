from __future__ import annotations

import os

def run() -> None:
    from wan_studio.config import ProjectConfig, SceneSpec

    cfg = ProjectConfig(
        project_name="smoke",
        output_dir=os.path.abspath("./outputs_smoke"),
        scenes=[
            SceneSpec(
                label="shot1",
                seconds=2.0,
                prompt="A hero enters a room",
                negative_prompt="anime, cartoon",
                characters=["Alice", "Bob"],
                location="Paris apartment",
                hard_cut=True,
                dialogue_text="Bonjour.",
                transition_mode="cut",
                init_preset_override="flux2_bnb4bit",
                init_policy_override="necessary",
            )
        ],
    )
    d = cfg.to_dict()
    cfg2 = ProjectConfig.from_dict(d)
    d2 = cfg2.to_dict()
    assert d2 == d, "ProjectConfig roundtrip to_dict/from_dict is not stable"
