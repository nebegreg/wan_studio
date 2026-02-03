"""Small CLI to generate and mux audio.

Examples:
  python -m wan_studio.audio_cli gen --project space.json --out outputs/run1
  python -m wan_studio.audio_cli track --project space.json --out outputs/run1
  python -m wan_studio.audio_cli mux --video outputs/run1/final.mp4 --audio outputs/run1/audio/dialogue_track.wav
"""

from __future__ import annotations

import argparse
import os

from .config import ProjectConfig
from .audio_pipeline import (
    generate_scene_dialogue_audio,
    build_dialogue_track,
    build_all_audio,
    mux_audio,
)


def _load_project(path: str) -> ProjectConfig:
    with open(path, "r", encoding="utf-8") as f:
        return ProjectConfig.from_dict(__import__("json").loads(f.read()))


def main() -> None:
    ap = argparse.ArgumentParser(prog="wan_studio.audio_cli")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_gen = sub.add_parser("gen", help="generate per-scene dialogue WAVs")
    ap_gen.add_argument("--project", required=True)
    ap_gen.add_argument("--out", required=True, help="project output directory")
    ap_gen.add_argument("--scene", type=int, default=-1, help="scene index (default: all)")

    ap_track = sub.add_parser("track", help="build a continuous dialogue track")
    ap_track.add_argument("--project", required=True)
    ap_track.add_argument("--out", required=True)

    ap_mix = sub.add_parser("mix", help="build stems (dialogue/vfx/music) and a master mix")
    ap_mix.add_argument("--project", required=True)
    ap_mix.add_argument("--out", required=True)

    ap_mux = sub.add_parser("mux", help="mux audio into a video")
    ap_mux.add_argument("--video", required=True)
    ap_mux.add_argument("--audio", required=True)
    ap_mux.add_argument("--out", default="")

    args = ap.parse_args()

    if args.cmd in ("gen", "track", "mix"):
        cfg = _load_project(args.project)
        out_dir = args.out
        os.makedirs(out_dir, exist_ok=True)

    if args.cmd == "gen":
        if args.scene >= 0:
            p = generate_scene_dialogue_audio(cfg, args.scene, out_dir)
            print(p or "(no dialogue)")
        else:
            for i in range(len(cfg.scenes)):
                try:
                    p = generate_scene_dialogue_audio(cfg, i, out_dir)
                    if p:
                        print(f"scene {i}: {p}")
                except Exception as e:
                    print(f"scene {i}: ERROR: {e}")
    elif args.cmd == "track":
        t = build_dialogue_track(cfg, out_dir)
        print(t)
    elif args.cmd == "mix":
        info = build_all_audio(cfg, out_dir)
        print(info.get("master_track"))
    elif args.cmd == "mux":
        out = args.out or os.path.splitext(args.video)[0] + "_with_audio.mp4"
        mux_audio(args.video, args.audio, out)
        print(out)


if __name__ == "__main__":
    main()
