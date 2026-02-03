# Changelog — Wan Studio Cinema Clean v1

## Included fixes

### Rendering stability
- Fixed overlap stitching to **overwrite** the overlap region instead of appending duplicate frames (removes stutter / chunk feeling).
- Added a **realism prompt guard** to prevent accidental anime/cartoon style leakage into photoreal/cinematic renders.

### Backends / pipelines
- Added backend/model guard rails: if a `model_id` looks like WAN but backend is `ltx2` (or vice-versa), the engine logs a warning and auto-switches to the compatible backend.

### Post-production
- RIFE interpolation now supports 4x/8x/etc by chaining 2x passes; if the target fps ratio is not a power of two, we fall back to ffmpeg minterpolate.

### Montage transitions
- Global `scene_transition_mode` + per-clip override (`transition_mode`) supporting `cut`, `crossfade`, and `flow`.

### Audio
- Integrated audio pipeline (Piper TTS, stems, music bed, VFX stem, ducking, mux) as documented in README.

### Narrative / coherence stability (hotfix 2026-01-21)
- Added `reset_continuity_between_clips` to prevent **conditioning bleed** between clips (each clip becomes independent).
- Added `strict_clip_coherence` to reduce **identity/background drift** in long, chunked clips (seed lock + prompt/negative guardrails).
- Added `force_first_frame_to_conditioning` to pin frame 0 to the I2V conditioning image (removes subtle chunk seam pops).
