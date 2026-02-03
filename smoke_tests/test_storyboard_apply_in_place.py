from __future__ import annotations


def run() -> None:
    from wan_studio.config import ProjectConfig, SceneSpec
    from wan_studio.storyboard_apply import apply_ai_scenes_in_place

    cfg = ProjectConfig(project_name="t", output_dir="./o", scenes=[
        SceneSpec(label="A", seconds=1.0, prompt="old", dialogue_text="hi"),
        SceneSpec(label="B", seconds=1.0, prompt="old2", dialogue_text="yo"),
    ])

    s0 = cfg.scenes[0]
    s1 = cfg.scenes[1]
    # lock prompt on first shot
    s0.lock_prompt = True
    s0.dialogue_audio_path = "some.wav"

    incoming = [
        SceneSpec(label="A", seconds=1.25, prompt="NEW PROMPT", dialogue_text="hi2", characters=["Alice"], location="Loc1"),
        SceneSpec(label="B", seconds=0.75, prompt="NEW2", dialogue_text="yo", characters=["Bob"], location="Loc2"),
    ]

    apply_ai_scenes_in_place(cfg, incoming)

    assert cfg.scenes[0] is s0, "Scene A object identity should be preserved"
    assert cfg.scenes[1] is s1, "Scene B object identity should be preserved"

    # prompt should not change due to lock_prompt
    assert cfg.scenes[0].prompt == "old", "lock_prompt should prevent prompt overwrite"
    # seconds should update even when lock_prompt
    assert abs(cfg.scenes[0].seconds - 1.25) < 1e-6

    # dialogue changed and is not locked -> audio path cleared
    assert cfg.scenes[0].dialogue_text == "hi2"
    assert cfg.scenes[0].dialogue_audio_path is None

    # bible auto-add
    assert any(getattr(c, 'name', '') == 'Alice' for c in (cfg.characters or []))
    assert any(getattr(l, 'name', '') == 'Loc2' for l in (cfg.locations or []))
