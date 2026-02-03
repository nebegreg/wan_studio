from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import List, Optional, Literal, Dict, Any
import os
import json

ModelMode = Literal["T2V", "I2V"]
BlendMode = Literal["cut", "flow", "crossfade"]
EaseMode = Literal["linear", "smoothstep", "ease_in_out"]
BackendName = Literal["wan", "cogvideox", "ltx", "ltx2", "lingbot"]


@dataclass
class TakeSpec:
    """A lightweight 'retake' record for a scene.

    We keep this intentionally simple so it survives JSON round-trips and
    doesn't break older projects.
    """

    name: str
    kind: str = "final"  # "proxy" | "final" | "custom"
    video_path: str = ""
    preview_path: Optional[str] = None
    seed: Optional[int] = None
    created_at: str = ""  # ISO-ish timestamp
    note: str = ""
    rating: int = 0
    tags: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TakeSpec":
        return TakeSpec(**_filter_dataclass_kwargs(TakeSpec, d or {}))


@dataclass
class AudioSpec:
    """Project-level audio settings.

    We keep this intentionally lightweight and resilient:
    - If Piper is available (tools/piper/piper or piper in PATH), we use it for TTS.
    - Otherwise, audio generation is disabled, but muxing still works.

    Notes:
    - WhisperX auto-align is optional (heavy). If installed, we can generate SRT.
      If not, we fall back to duration-fit only.
    """

    enabled: bool = True

    # TTS engine selection. Currently: "piper" or "none".
    tts_engine: str = "piper"

    # Default voice model (Piper .onnx) and config (.onnx.json).
    # These live under tools/piper/voices by default.
    voice_model_path: str = os.path.abspath("./tools/piper/voices/fr_FR-upmc-medium.onnx")
    voice_config_path: str = os.path.abspath("./tools/piper/voices/fr_FR-upmc-medium.onnx.json")

    # Sample rate to standardize audio clips before concat/mux.
    sample_rate: int = 48000

    # Default language tag used for metadata + (optional) alignment.
    default_language: str = "fr"

    # Auto-fit speech to the scene duration.
    auto_fit_to_scene: bool = True
    fit_max_speedup: float = 1.35  # don't go chipmunk
    fit_pad_shorter: bool = True

    # --- Cinematic workflow helpers ---
    # If enabled, when a generated dialogue audio is longer than the shot duration,
    # we extend the shot duration to match the speech (instead of speed-up/trim).
    auto_extend_scene_to_dialogue: bool = False
    extend_pad_seconds: float = 0.20
    extend_max_seconds: float = 120.0

    # Convenience: when applying an AI storyboard, optionally auto-generate audio
    # and build the full audio bundle (tracks + master mix + SRT).
    auto_generate_on_storyboard_apply: bool = False
    auto_build_mix_on_storyboard_apply: bool = False

    # Optional: auto-align captions with WhisperX when available.
    auto_align_captions: bool = False
    whisperx_model: str = "large-v3"  # only used if whisperx is installed

    # Where to write per-scene audio + the full track.
    audio_dir_name: str = "audio"
    track_filename: str = "dialogue_track.wav"

    # Export stems for editing in an NLE (DaVinci/Premiere/etc.).
    export_shot_stems: bool = True
    export_scene_stems: bool = True
    shot_stems_dir_name: str = "shots"
    scene_stems_dir_name: str = "scenes"

    # Optional subtitles export.
    export_srt: bool = True
    srt_filename: str = "subtitles.srt"

    # --- Additional stems + mixdown (V17+) ---
    # Optional global music bed that spans the whole film.
    music_path: Optional[str] = None

    # Auto music generation (optional). Uses audiocraft MusicGen if installed;
    # otherwise falls back to a safe placeholder.
    music_auto_generate: bool = False
    music_style_preset: str = "cinematic_orchestral"
    music_model_name: str = "facebook/musicgen-small"
    music_max_seconds: float = 60.0
    music_seed: Optional[int] = None
    music_track_filename: str = "music_track.wav"

    # If True and no music is provided/generated, create a subtle placeholder ambience
    # so exports are never silently missing a music bed.
    music_placeholder_if_missing: bool = True
    music_placeholder_style: str = "ambient"  # ambient | trailer (placeholder generator flavor)

    # Optional VFX stem. If scenes define vfx_audio_path, we build a continuous track.
    vfx_track_filename: str = "vfx_track.wav"

    # Final mix (music + dialogue + vfx).
    master_track_filename: str = "master_mix.wav"

    # Simple gain staging (dB).
    music_gain_db: float = -14.0
    dialogue_gain_db: float = -3.0
    vfx_gain_db: float = -6.0
    # Master gain applied after mixing (dB).
    master_gain_db: float = 0.0

    # Simple per-bus mute toggles (useful for quick exports / debugging).
    music_muted: bool = False
    dialogue_muted: bool = False
    vfx_muted: bool = False


    # Ducking: reduce music when dialogue is present.
    duck_music_under_dialogue: bool = True
    # These parameters are fed to ffmpeg sidechaincompress.
    duck_threshold: float = 0.08
    duck_ratio: float = 10.0
    duck_attack: float = 0.02
    duck_release: float = 0.25

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AudioSpec":
        return AudioSpec(**_filter_dataclass_kwargs(AudioSpec, d))

@dataclass
class LoraSpec:
    name: str
    weight: float = 1.0
    repo_id: Optional[str] = None
    weight_name: Optional[str] = None
    local_path: Optional[str] = None
    load_into_transformer_2: bool = False

    target_backend: Optional[str] = None  # e.g. 'wan', 'ltx', 'cog'
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LoraSpec":
        return LoraSpec(**_filter_dataclass_kwargs(LoraSpec, d))



@dataclass
class CharacterSpec:
    name: str = ""
    description: str = ""
    prompt: str = ""            # injected when character is present
    negative_prompt: str = ""   # optional character-specific negatives
    ref_images: List[str] = field(default_factory=list)

    # Voice identity (optional). If empty, global project voice is used.
    # voice_model_path can be either an absolute path to a .onnx or a filename
    # that will be resolved under tools/piper/voices.
    voice_gender: str = ""       # "female" | "male" | "neutral" | ""
    voice_language: str = ""     # "fr" | "en" | ...
    voice_model_path: str = ""   # .onnx path or id
    voice_config_path: str = ""  # .onnx.json path (optional)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CharacterSpec":
        dd = dict(d or {})
        # legacy: ref_image -> ref_images
        if "ref_images" not in dd and dd.get("ref_image"):
            dd["ref_images"] = [dd.get("ref_image")]
        if isinstance(dd.get("ref_images"), str):
            dd["ref_images"] = [dd.get("ref_images")]
        return CharacterSpec(**_filter_dataclass_kwargs(CharacterSpec, dd))


@dataclass
class LocationSpec:

    name: str = ""
    description: str = ""
    prompt: str = ""            # injected when location is used
    negative_prompt: str = ""   # optional location-specific negatives
    ref_images: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LocationSpec":
        dd = dict(d or {})
        if "ref_images" not in dd and dd.get("ref_image"):
            dd["ref_images"] = [dd.get("ref_image")]
        if isinstance(dd.get("ref_images"), str):
            dd["ref_images"] = [dd.get("ref_images")]
        return LocationSpec(**_filter_dataclass_kwargs(LocationSpec, dd))


@dataclass
class InitImageSpec:
    # Auto init-frame generation is optional.
    # I2V normally expects an explicit init image; auto-generation should only
    # happen when the user enables it.
    enabled: bool = False
    preset: str = "flux2_bnb4bit"   # default init-frame preset
    # Default per spec: ALWAYS refine the init frame even if one already exists.
    # (User can switch to "necessary" to rely on cache hits.)
    policy: str = "necessary"   # necessary | always_refine
    cache_location: str = "assets"  # assets | project
    project_cache_dirname: str = ".wan_cache"

    # 0 means: use the project resolution (cfg.width/cfg.height).
    width: int = 0
    height: int = 0
    steps: int = 32
    guidance_scale: float = 4.5
    seed: Optional[int] = None

    use_existing_as_reference: bool = True
    use_last_location_memory: bool = False
    max_refs: int = 6

    # --- "cinema-grade" reference separation ---
    # (best-effort; pipelines that don't support adapters simply ignore these)
    style_ref_images: List[str] = field(default_factory=list)

    # --- optional IP-Adapter / adapter controls (best-effort) ---
    enable_ip_adapter: bool = False
    ip_adapter_model_id: str = ""          # optional repo id (if empty, use pipeline defaults)
    ip_adapter_subfolder: str = ""         # optional
    ip_adapter_weight_name: str = ""       # optional
    ip_adapter_scale_character: float = 0.8
    ip_adapter_scale_location: float = 0.6
    ip_adapter_scale_style: float = 0.5

    # --- optional ControlNet / context refine (best-effort) ---
    enable_controlnet: bool = False
    controlnet_type: str = 'canny'   # canny | depth | pose (best-effort)
    controlnet_model_id: str = ''    # optional HF repo id; if empty, use sane defaults
    controlnet_scale: float = 0.75
    refine_strength: float = 0.35    # img2img strength for refine-capable pipelines

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "InitImageSpec":
        return InitImageSpec(**_filter_dataclass_kwargs(InitImageSpec, d or {}))

@dataclass
class LingBotSpec:
    """LingBot-World (base-cam) backend settings.

    Profile A (stable on RTX 4090 / 24GB): 832×480 landscape, subprocess, offload, t5_cpu.
    """
    enabled: bool = True

    # Local directory or HuggingFace repo id.
    ckpt_dir: str = "robbyant/lingbot-world-base-cam"

    # Must be landscape 832×480 for base-cam intrinsics normalization in upstream code.
    size: str = "832*480"

    # Internal fps for i2v-A14B
    fps_internal: int = 16

    # Sampling
    sample_steps: int = 24
    sample_shift: float = 3.0
    sample_guide_scale: float = 5.0

    # Memory helpers
    offload_model: bool = True
    t5_cpu: bool = True
    ulysses_size: int = 1

@dataclass
class SceneSpec:
    # --- studio metadata ---
    label: str = ""
    tags: str = ""       # comma-separated, e.g. "EXT,NIGHT,WS"
    notes: str = ""      # free text notes

    # --- content ---
    seconds: float = 5.0
    prompt: str = "Cinematic shot, gentle camera motion."
    negative_prompt: Optional[str] = None

    # --- cinema phase C ---
    characters: List[str] = field(default_factory=list)   # names from Character Bible
    location: str = ""                                   # name from Location Library
    hard_cut: bool = False                               # reset continuity + force cut boundary

    # Optional structured shot info (from storyboard)
    shot_type: str = ""
    camera_move: str = ""
    mood: str = ""

    # --- LingBot base-cam camera (clip-centric) ---
    camera_path_preset: str = "static"
    camera_fov_deg: float = 50.0
    camera_path_amount: float = 1.0
    camera_path_strength: float = 1.0
    music_intensity: str = ""

    # --- locks (prevent storyboard apply from overwriting manual edits) ---
    lock_prompt: bool = False
    lock_dialogue: bool = False
    lock_cast: bool = False
    lock_location: bool = False
    lock_shot: bool = False
    lock_music: bool = False

    # --- init frame (Flux2 auto-init) ---
    init_frame_path: Optional[str] = None
    init_frame_hash: Optional[str] = None

    # --- live preview (generated progressively during render) ---
    # This is runtime-friendly and optional; projects can ignore it.
    preview_frame_path: Optional[str] = None

    # Init-frame UI overrides (per clip)
    init_preset_override: Optional[str] = None   # e.g. sdxl_base / flux1_schnell / flux2_bnb4bit
    init_policy_override: Optional[str] = None   # necessary | always_refine

    # --- per-clip overrides (Premiere-style) ---
    model_id_override: Optional[str] = None
    mode_override: Optional[ModelMode] = None
    input_image_path_override: Optional[str] = None  # if set, resets conditioning for this scene

    # --- v14 studio controls ---
    use_character_ref: bool = False
    style_pack: Optional[str] = None
    backend_override: Optional[BackendName] = None
    proxy_video_path: Optional[str] = None
    final_video_path: Optional[str] = None

    # --- Retakes / takes manager ---
    # Each take records an output video (proxy/final/custom) and an optional preview PNG.
    takes: List[TakeSpec] = field(default_factory=list)
    active_proxy_take: Optional[str] = None
    active_final_take: Optional[str] = None

    # --- audio (per-shot) ---
    dialogue_text: str = ""
    dialogue_language: str = ""
    dialogue_voice: str = ""
    dialogue_speaker: Optional[int] = None
    dialogue_pitch: float = 1.0
    dialogue_rate: float = 1.0
    dialogue_audio_path: Optional[str] = None
    vfx_audio_path: Optional[str] = None
    vfx_prompt: str = ""
    vfx_seed: Optional[int] = None

    # --- overrides ---
    seed: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    guidance_scale_2: Optional[float] = None
    boundary_ratio: Optional[float] = None

    # --- per-clip scheduler overrides (V17+) ---
    # None = inherit project setting
    use_unipc_override: Optional[bool] = None
    flow_shift_override: Optional[float] = None

    # --- blending hints ---
    blend_mode: Optional[BlendMode] = None
    transition_mode: Optional[BlendMode] = None
    transition_frames: Optional[int] = None
    transition_ease: Optional[EaseMode] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SceneSpec":
        dd = dict(d or {})

        # legacy: character -> characters
        if "characters" not in dd and "character" in dd:
            dd["characters"] = dd.get("character")
        ch = dd.get("characters")
        if isinstance(ch, str):
            raw = ch.replace(";", ",")
            dd["characters"] = [t.strip() for t in raw.split(",") if t.strip()]
        elif isinstance(ch, list):
            dd["characters"] = [str(x).strip() for x in ch if str(x).strip()]
        else:
            dd["characters"] = dd.get("characters") or []

        # location aliases
        if not dd.get("location"):
            for k in ("background", "set", "decor", "place"):
                if dd.get(k):
                    dd["location"] = dd.get(k)
                    break

        # hard cut aliases
        if "hard_cut" not in dd and "hardCut" in dd:
            dd["hard_cut"] = dd.get("hardCut")
        if "hard_cut" in dd:
            try:
                dd["hard_cut"] = bool(dd.get("hard_cut"))
            except Exception:
                dd["hard_cut"] = False

        # boolify locks
        for k in ("lock_prompt","lock_dialogue","lock_cast","lock_location","lock_shot","lock_music"):
            if k in dd:
                try:
                    dd[k] = bool(dd.get(k))
                except Exception:
                    dd[k] = False

        # dialogue alias
        # dialogue alias
        if "dialogue_text" not in dd and "dialogue" in dd:
            dd["dialogue_text"] = dd.get("dialogue") or ""

        # numeric -> str
        if "music_intensity" in dd and dd["music_intensity"] is not None and not isinstance(dd["music_intensity"], str):
            dd["music_intensity"] = str(dd["music_intensity"])

        # takes
        try:
            raw_takes = dd.get("takes") or []
            parsed: List[TakeSpec] = []
            if isinstance(raw_takes, list):
                for t in raw_takes:
                    if isinstance(t, TakeSpec):
                        parsed.append(t)
                    elif isinstance(t, dict):
                        parsed.append(TakeSpec.from_dict(t))
            dd["takes"] = parsed
        except Exception:
            dd["takes"] = []

        # active take fields
        for k in ("active_proxy_take", "active_final_take"):
            if k in dd and dd[k] is not None:
                try:
                    dd[k] = str(dd[k])
                except Exception:
                    dd[k] = None

        return SceneSpec(**_filter_dataclass_kwargs(SceneSpec, dd))

@dataclass
class PostProcessSpec:
    enabled: bool = True
    tools_dir: str = os.path.abspath("./tools")
    use_rife_if_available: bool = True
    use_realesrgan_if_available: bool = True

    target_fps: int = 24
    out_width: int = 1920
    out_height: int = 1080

    deflicker: bool = True
    denoise: bool = True
    denoise_luma: float = 1.5
    denoise_chroma: float = 1.0
    denoise_temporal: float = 3.0
    denoise_temporal_chroma: float = 2.0
    sharpen: bool = True
    unsharp_mx: int = 5
    unsharp_my: int = 5
    unsharp_amount: float = 0.8

    export_master_prores: bool = True
    export_delivery_h265_main10: bool = False
    primary_deliver: str = "prores"  # prores|h265|intermediate
    prores_profile: str = "422hq"     # proxy|lt|422|422hq|4444|4444xq
    h265_crf: int = 18
    h265_preset: str = "slow"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PostProcessSpec":
        return PostProcessSpec(**_filter_dataclass_kwargs(PostProcessSpec, d))
@dataclass
class BatchSpec:
    enabled: bool = False
    variants: int = 1
    seed_step: int = 10000
    suffix: str = "v"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "BatchSpec":
        return BatchSpec(**_filter_dataclass_kwargs(BatchSpec, d))

@dataclass
class StudioSpec:
    write_report: bool = True
    export_shotlist_csv: bool = True
    crash_safe_resume: bool = True

    # Playback choice inside the Studio (Premiere) viewer.
    prefer_proxy_playback: bool = True

    # --- VRAM / resource manager ---
    vram_manager_enabled: bool = True
    # If free VRAM drops under this threshold, we try to cleanup/unload init pipelines.
    vram_min_free_gb: float = 2.0
    # More aggressive target used before loading the big I2V pipeline.
    vram_target_free_gb: float = 5.0
    # Unload Flux/SDXL init-image pipeline after generating the init frame.
    unload_init_pipeline_after_use: bool = True

    # --- Studio (Premiere) timeline conveniences ---
    # When enabled, the engine will auto-create a lightweight per-clip MP4 preview
    # from the clip cache and wire it to SceneSpec.proxy_video_path. This makes
    # already-rendered clips instantly playable in the timeline (double-click / right-click).
    auto_generate_clip_previews: bool = True

    # ffmpeg encoding knobs for the preview MP4 (smaller/faster than raw).
    clip_preview_crf: int = 23
    clip_preview_preset: str = "veryfast"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "StudioSpec":
        return StudioSpec(**_filter_dataclass_kwargs(StudioSpec, d))


@dataclass
class ProjectConfig:
    # Backend controls: "wan" (Wan2.2 diffusers), "cogvideox" (THUDM CogVideoX), "ltx" (Lightricks LTX-Video)
    backend: BackendName = "wan"
    model_id: str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    mode: ModelMode = "I2V"
    input_image_path: Optional[str] = None

    # Global character reference image (used for I2V when a clip opts-in, or as a fallback).
    character_image_path: Optional[str] = None

    # Global style pack (see style_packs.py). Empty/None disables.
    global_style_pack: str = "Cinematic (General Realism)"

    # Default negative prompt geared toward photoreal / cinema.
    negative_prompt: str = (
        "blurry, low quality, distorted, deformed, static, poorly drawn, "
        "cartoon, anime, illustration, comic, toon shading, 3d render, cgi, pixar, "
        "flicker, temporal jitter, frame tearing, warping, unstable text"
    )

    scenes: List[SceneSpec] = field(default_factory=lambda: [
        SceneSpec(label="S01_SH01", tags="EXT,NIGHT,WS", seconds=10.0, prompt="Cinematic shot of a character walking through neon-lit rain at night, shallow depth of field."),
        SceneSpec(label="S01_SH02", tags="INT,NIGHT,MS", seconds=10.0, prompt="The character enters a warm cafe, soft lighting, smooth dolly in, cinematic composition."),
    ])

    output_dir: str = os.path.abspath("./outputs")
    project_name: str = "project"
    fps: int = 24
    width: int = 1280
    height: int = 720

    chunk_seconds: float = 3.0
    # --- Segment stitching (inside a clip) ---
    # "Ciné stable" default: NO overlap, NO blending.
    # This removes flow/blend artefacts and avoids temporal contamination at segment boundaries.
    overlap_frames: int = 0
    blend_mode: BlendMode = "cut"

    # Hard guard: if enabled, the engine will force overlap_frames=0 and blend_mode='cut'
    # at render time, even if a preset/project tries to set a different value.
    no_overlap_strict: bool = True


    # Master switch: lock the app into a crash-proof / anti-OOM cinema mode.
    # When enabled, the engine (and UI) will enforce safe defaults:
    # - no overlap + cut stitching
    # - UniPC disabled for Wan (prevents shape-mismatch crashes)
    # - sequential CPU offload enabled (lower peak VRAM)
    # You can disable it if you know what you're doing.
    cinema_safe_mode: bool = True
    # --- Quality / robustness helpers ---
    # If enabled, scan rendered frames and automatically repair isolated "glitch" frames
    # (rare diffusion failures) by blending neighboring frames. This is conservative and
    # should NOT affect legitimate scene cuts.
    frame_safety_enabled: bool = True
    frame_safety_threshold_factor: float = 10.0
    frame_safety_neighbor_similarity_factor: float = 0.35

    # Prompt token safety (helps avoid silent CLIP truncation at 77 tokens).
    strict_prompt_max_tokens: bool = True
    prompt_trim_strategy: str = "head_tail"  # head | head_tail

    scene_transition_mode: BlendMode = "cut"
    scene_transition_frames: int = 0
    scene_transition_ease: EaseMode = "smoothstep"

    # If enabled, default montage boundary is a hard cut unless a clip overrides transition_mode.
    cut_strict_between_clips: bool = True

    # --- Stability / coherence ---
    # If enabled, every clip starts from a fresh anchor (no last-frame conditioning spill).
    # Also disables init-frame "location memory" fallback across clips.
    reset_continuity_between_clips: bool = True

    # If enabled, adds strict anti-drift guardrails during a clip (chunked shots):
    # - locks seed within a clip (but varies per clip)
    # - forces frame 0 to exactly match the conditioning image (prevents seam jumps)
    # - injects prompt/negative guardrails to reduce identity/background drift
    strict_clip_coherence: bool = True

    # Force the first frame of every generated segment to match the I2V conditioning image.
    # (Useful to eliminate subtle boundary pops even when blending overlap.)
    force_first_frame_to_conditioning: bool = True

    # Ultra isolation mode: ensure *absolute* clip independence by hard-resetting
    # heavy pipelines between clips (I2V + init pipelines) and clearing CUDA caches.
    # This is slower, but minimizes any chance of cross-clip contamination.
    ultra_isolated_clips: bool = False

    keep_png_frames: bool = False

    num_inference_steps: int = 18
    guidance_scale: float = 4.2
    guidance_scale_2: float = 2.5
    boundary_ratio: float = 0.90

    use_unipc: bool = False
    flow_shift: float = 5.0

    enable_model_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    vae_decode_fp32: bool = True

    loras: List[LoraSpec] = field(default_factory=list)

    # Phase C libraries
    characters: List[CharacterSpec] = field(default_factory=list)
    locations: List[LocationSpec] = field(default_factory=list)

    # Flux2 auto init-frame generation
    init_image: InitImageSpec = field(default_factory=InitImageSpec)


    # LingBot-World base-cam backend
    lingbot: LingBotSpec = field(default_factory=LingBotSpec)
    # --- clip render cache (skip re-render if unchanged) ---
    clip_cache_enabled: bool = True
    clip_cache_dirname: str = ".wan_cache"

    cache_max_gb: float = 0.0  # 0 = unlimited; applies to init+clip caches
    prune_cache_on_start: bool = True

    base_seed: int = 12345
    vary_seed_per_segment: bool = False

    # --- VRAM / OOM resilience ---
    # If enabled, the engine will adapt segment length on-the-fly to avoid CUDA OOM
    # (more segments, same resolution/steps).
    auto_vram_optimizations: bool = True

    # Hard cap on frames per diffusion call (0 = auto).
    max_frames_per_segment: int = 0

    # How many times to retry a segment after an OOM by reducing frames-per-call.
    oom_retry_max_attempts: int = 2

    batch: BatchSpec = field(default_factory=BatchSpec)
    studio: StudioSpec = field(default_factory=StudioSpec)
    post: PostProcessSpec = field(default_factory=PostProcessSpec)

    audio: AudioSpec = field(default_factory=AudioSpec)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["scenes"] = [s.to_dict() for s in self.scenes]
        d["loras"] = [l.to_dict() for l in self.loras]
        d["post"] = self.post.to_dict()
        d["batch"] = self.batch.to_dict()
        d["studio"] = self.studio.to_dict()
        d["audio"] = self.audio.to_dict()

        # Phase C
        d["characters"] = [c.to_dict() for c in (self.characters or [])]
        d["locations"] = [l.to_dict() for l in (self.locations or [])]
        d["init_image"] = self.init_image.to_dict() if self.init_image else InitImageSpec().to_dict()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ProjectConfig":
        dd = dict(d or {})

        # legacy key aliases
        if "character_bible" in dd and "characters" not in dd:
            dd["characters"] = dd.get("character_bible") or []
        if "location_library" in dd and "locations" not in dd:
            dd["locations"] = dd.get("location_library") or []
        if "init" in dd and "init_image" not in dd:
            dd["init_image"] = dd.get("init") or {}
        if "cut_strict" in dd and "cut_strict_between_clips" not in dd:
            dd["cut_strict_between_clips"] = bool(dd.get("cut_strict"))

        scenes = [SceneSpec.from_dict(x) for x in (dd.get("scenes") or []) if isinstance(x, dict)]
        loras = [LoraSpec.from_dict(x) for x in (dd.get("loras") or []) if isinstance(x, dict)]
        post = PostProcessSpec.from_dict(dd.get("post", {})) if isinstance(dd.get("post", {}), dict) else PostProcessSpec()
        batch = BatchSpec.from_dict(dd.get("batch", {})) if isinstance(dd.get("batch", {}), dict) else BatchSpec()
        studio = StudioSpec.from_dict(dd.get("studio", {})) if isinstance(dd.get("studio", {}), dict) else StudioSpec()
        audio = AudioSpec.from_dict(dd.get("audio", {})) if isinstance(dd.get("audio", {}), dict) else AudioSpec()

        characters = [CharacterSpec.from_dict(x) for x in (dd.get("characters") or []) if isinstance(x, dict)]
        locations = [LocationSpec.from_dict(x) for x in (dd.get("locations") or []) if isinstance(x, dict)]
        init_image = InitImageSpec.from_dict(dd.get("init_image", {})) if isinstance(dd.get("init_image", {}), dict) else InitImageSpec()

        d2 = dict(dd)
        d2["scenes"] = scenes
        d2["loras"] = loras
        d2["post"] = post
        d2["batch"] = batch
        d2["studio"] = studio
        d2["audio"] = audio
        d2["characters"] = characters
        d2["locations"] = locations
        d2["init_image"] = init_image

        return ProjectConfig(**_filter_dataclass_kwargs(ProjectConfig, d2))

    @staticmethod
    def from_json(s: str) -> "ProjectConfig":
        return ProjectConfig.from_dict(json.loads(s))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @staticmethod
    def load(path: str) -> "ProjectConfig":
        with open(path, "r", encoding="utf-8") as f:
            return ProjectConfig.from_json(f.read())


def _filter_dataclass_kwargs(cls, d: dict) -> dict:
    """Forward/backward compatible load: keep only keys that exist in the dataclass."""
    try:
        allowed = {f.name for f in fields(cls)}
    except Exception:
        return dict(d)
    return {k: v for k, v in d.items() if k in allowed}