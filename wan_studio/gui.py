from __future__ import annotations

import os
import json
import traceback
from typing import Optional, List

from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget

try:
    import qdarkstyle
except Exception:
    qdarkstyle = None

from .config import ProjectConfig, SceneSpec, LoraSpec, InitImageSpec
from .safety import apply_cinema_safe_overrides
from .audio_pipeline import (
    generate_scene_dialogue_audio,
    generate_scene_vfx_audio,
    build_all_audio,
    build_master_mix,
    mux_audio,
    find_piper,
)
from .engine import WanEngine
from .director_dialog import DirectorDialog
from .studio_premiere import StudioPremiereTab
from .mixer_dock import AudioMixerDock
from .ai_image_dialog import ImageInitDialog
from .postprocess import detect_tools
from .media import make_inputs_dir, unique_path, exr_to_png, extract_frame_ffmpeg
from .style_packs import list_packs
from . import __version__ as WS_VERSION
from .build_info import BUILD_TAG

MODEL_PRESETS = [
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    'zai-org/CogVideoX-5b',
    'zai-org/CogVideoX-5b-I2V',
    'Lightricks/LTX-Video',
    'Lightricks/LTX-Video-0.9.7-distilled',
    'Lightricks/LTX-2',
]


class _FuncWorker(QtCore.QObject):
    # Backward-compatible signals:
    # - some callers connect to .finished
    # - some callers connect to .done
    finished = QtCore.Signal(object)
    done = QtCore.Signal(object)
    error = QtCore.Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    @QtCore.Slot()
    def cancel_current_clip(self):
        """Cancel only the clip currently being rendered (best-effort)."""
        if self._current_idx is None:
            return
        try:
            self._cancel_clip_indices.add(int(self._current_idx))
        except Exception:
            pass
        self._cancel_current = True
        self._cancel_current_idx = int(self._current_idx)
        try:
            self.log.emit(f'Annulation du clip {int(self._current_idx)+1} demandée…')
        except Exception:
            pass

    @QtCore.Slot(int)
    def request_cancel_clip(self, idx: int):
        """Request cancellation for a specific clip index."""
        try:
            idx = int(idx)
        except Exception:
            return
        self._cancel_clip_indices.add(idx)
        if self._current_idx is not None and int(self._current_idx) == idx:
            self._cancel_current = True
            self._cancel_current_idx = idx
            try:
                self.log.emit(f'Annulation du clip {idx+1} demandée…')
            except Exception:
                pass

    @QtCore.Slot()
    def run(self):
        try:
            res = self.fn()
            # Emit both for compatibility with older/newer call sites
            self.finished.emit(res)
            try:
                self.done.emit(res)
            except Exception:
                pass
        except Exception:
            self.error.emit(traceback.format_exc())

def default_lora_packs() -> dict[str, list[LoraSpec]]:
    # NOTE: these repo_ids/filenames are placeholders you can customize.
    # The UI is designed so you can paste exact HF repo + filename you use.
    return {
        "SVI 2.0 Pro (I2V A14B) - dual noise": [
            LoraSpec(
                name="svi_high",
                repo_id="vita-video-gen/svi-model",
                weight_name="version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors",
                weight=1.0,
                load_into_transformer_2=False,
            ),
            LoraSpec(
                name="svi_low",
                repo_id="vita-video-gen/svi-model",
                weight_name="version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors",
                weight=1.0,
                load_into_transformer_2=True,
            ),
        ],
        "LightX2V 4-step (I2V A14B) - dual noise": [
            LoraSpec(
                name="lx2v_high",
                repo_id="lightx2v/Wan2.2-Distill-Loras",
                weight_name="wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors",
                weight=1.0,
                load_into_transformer_2=False,
            ),
            LoraSpec(
                name="lx2v_low",
                repo_id="lightx2v/Wan2.2-Distill-Loras",
                weight_name="wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors",
                weight=1.0,
                load_into_transformer_2=True,
            ),
        ],

        # --- Recommended community/official packs (curated) ---
        "Lightning 4-step (I2V A14B) - Seko V1": [
            LoraSpec(
                name="lightning_high",
                repo_id="lightx2v/Wan2.2-Lightning",
                weight_name="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
                weight=1.0,
                load_into_transformer_2=False,
                target_backend="wan",
            ),
            LoraSpec(
                name="lightning_low",
                repo_id="lightx2v/Wan2.2-Lightning",
                weight_name="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
                weight=1.0,
                load_into_transformer_2=True,
                target_backend="wan",
            ),
        ],
        "Camera: Arc shot (Wan2.2 A14B)": [
            LoraSpec(
                name="cam_arcshot",
                repo_id="wangkanai/wan22-fp8-i2v-loras",
                weight_name="loras/wan/wan22-camera-arcshot-rank16-i2v-a14b-high.safetensors",
                weight=0.9,
                load_into_transformer_2=False,
                target_backend="wan",
            ),
        ],
        "Camera: Drone (Wan2.2 A14B)": [
            LoraSpec(
                name="cam_drone",
                repo_id="wangkanai/wan22-fp8-i2v-loras",
                weight_name="loras/wan/wan22-camera-drone-rank16-i2v-a14b.safetensors",
                weight=0.9,
                load_into_transformer_2=False,
                target_backend="wan",
            ),
        ],
        "Camera: Rotation (Wan2.2 A14B)": [
            LoraSpec(
                name="cam_rotation",
                repo_id="wangkanai/wan22-fp8-i2v-loras",
                weight_name="loras/wan/wan22-camera-rotation-rank16-i2v-a14b.safetensors",
                weight=0.9,
                load_into_transformer_2=False,
                target_backend="wan",
            ),
        ],
        "Camera: Earth zoom-out (Wan2.2 A14B)": [
            LoraSpec(
                name="cam_earthzoomout",
                repo_id="wangkanai/wan22-fp8-i2v-loras",
                weight_name="loras/wan/wan22-camera-earthzoomout.safetensors",
                weight=0.9,
                load_into_transformer_2=False,
                target_backend="wan",
            ),
        ],
        "Lighting: Cinematic flare (Wan2.2 A14B)": [
            LoraSpec(
                name="light_cinematicflare_low",
                repo_id="wangkanai/wan22-fp8-i2v-loras",
                weight_name="loras/wan/wan22-light-cinematicflare-i2v-low.safetensors",
                weight=0.8,
                load_into_transformer_2=True,
                target_backend="wan",
            ),
        ],
        "Action: Wink (Wan2.2 A14B)": [
            LoraSpec(
                name="action_wink_low",
                repo_id="wangkanai/wan22-fp8-i2v-loras",
                weight_name="loras/wan/wan22-action-wink-i2v-v1-low.safetensors",
                weight=0.8,
                load_into_transformer_2=True,
                target_backend="wan",
            ),
        ],
        "Camera: Orbit 360 (Wan2.2 A14B) - ostris": [
            LoraSpec(
                name="orbit_high",
                repo_id="ostris/wan22_i2v_14b_orbit_shot_lora",
                weight_name="wan22_14b_i2v_orbit_high_noise.safetensors",
                weight=1.0,
                load_into_transformer_2=False,
                target_backend="wan",
            ),
            LoraSpec(
                name="orbit_low",
                repo_id="ostris/wan22_i2v_14b_orbit_shot_lora",
                weight_name="wan22_14b_i2v_orbit_low_noise.safetensors",
                weight=1.0,
                load_into_transformer_2=True,
                target_backend="wan",
            ),
        ],
        "LTX-Video: Squish effect (v0.9.5)": [
            LoraSpec(
                name="ltxv_squish",
                repo_id="Lightricks/LTX-Video-Squish-LoRA",
                weight_name="ltxv_095_squish_lora.safetensors",
                weight=0.8,
                load_into_transformer_2=False,
                target_backend="ltx",
            ),
        ],
        "LTX-Video: ICLoRA detailer (13B v0.9.8)": [
            LoraSpec(
                name="ltxv_detailer_ic",
                repo_id="Lightricks/LTX-Video-ICLoRA-detailer-13b-0.9.8",
                weight_name="ltxv-098-ic-lora-detailer-diffusers.safetensors",
                weight=0.8,
                load_into_transformer_2=False,
                target_backend="ltx",
            ),
        ],
    }

class PromptBuilderDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prompt Builder (studio)")
        self.resize(900, 600)
        lay = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        lay.addLayout(form)

        self.subject = QtWidgets.QLineEdit("A person")
        self.action = QtWidgets.QLineEdit("walking")
        self.scene = QtWidgets.QLineEdit("through a neon-lit rainy street at night")
        self.camera = QtWidgets.QComboBox()
        self.camera.addItems(["smooth handheld","slow dolly in","dolly out","tracking shot","steadycam","crane shot","static tripod"])
        self.lens = QtWidgets.QComboBox()
        self.lens.addItems(["35mm","50mm","85mm","anamorphic","wide angle"])
        self.lighting = QtWidgets.QComboBox()
        self.lighting.addItems(["cinematic","soft","high contrast","neon","golden hour","moody"])
        self.style = QtWidgets.QLineEdit("cinematic, film grain, shallow depth of field, ultra detailed")
        self.extra = QtWidgets.QLineEdit("subtle motion, natural physics, coherent identity")

        form.addRow("Subject", self.subject)
        form.addRow("Action", self.action)
        form.addRow("Scene", self.scene)
        form.addRow("Camera", self.camera)
        form.addRow("Lens", self.lens)
        form.addRow("Lighting", self.lighting)
        form.addRow("Style", self.style)
        form.addRow("Extra", self.extra)

        self.out = QtWidgets.QPlainTextEdit()
        self.out.setReadOnly(True)
        lay.addWidget(self.out, 1)

        btns = QtWidgets.QHBoxLayout()
        lay.addLayout(btns)
        self.btn_gen = QtWidgets.QPushButton("Generate prompt")
        self.btn_copy = QtWidgets.QPushButton("Copy")
        self.btn_close = QtWidgets.QPushButton("Close")
        btns.addWidget(self.btn_gen)
        btns.addWidget(self.btn_copy)
        btns.addStretch(1)
        btns.addWidget(self.btn_close)

        self.btn_gen.clicked.connect(self._gen)
        self.btn_copy.clicked.connect(self._copy)
        self.btn_close.clicked.connect(self.accept)

        self._gen()

    def _gen(self):
        txt = (
            f"{self.subject.text().strip()} {self.action.text().strip()} {self.scene.text().strip()}, "
            f"{self.camera.currentText()}, {self.lens.currentText()} lens, {self.lighting.currentText()} lighting, "
            f"{self.style.text().strip()}, {self.extra.text().strip()}."
        )
        self.out.setPlainText(txt)

    def _copy(self):
        QtWidgets.QApplication.clipboard().setText(self.out.toPlainText())

    def prompt(self) -> str:
        return self.out.toPlainText().strip()

class ExrImportDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import EXR → PNG (conditioning)")
        self.resize(520, 220)
        lay = QtWidgets.QFormLayout(self)

        self.exposure = QtWidgets.QDoubleSpinBox()
        self.exposure.setRange(-10.0, 10.0)
        self.exposure.setDecimals(2)
        self.exposure.setValue(0.0)

        self.gamma = QtWidgets.QDoubleSpinBox()
        self.gamma.setRange(0.5, 4.0)
        self.gamma.setDecimals(2)
        self.gamma.setValue(2.2)

        self.tonemap = QtWidgets.QComboBox()
        self.tonemap.addItems(["reinhard", "aces", "clip"])

        lay.addRow("Exposure (EV)", self.exposure)
        lay.addRow("Gamma", self.gamma)
        lay.addRow("Tonemap", self.tonemap)

        row = QtWidgets.QHBoxLayout()
        self.btn_ok = QtWidgets.QPushButton("Convert")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        row.addStretch(1)
        row.addWidget(self.btn_cancel)
        row.addWidget(self.btn_ok)
        lay.addRow(row)

        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)


class ModelDownloadWorker(QtCore.QObject):
    log = QtCore.Signal(str)
    finished = QtCore.Signal(str)
    failed = QtCore.Signal(str)

    def __init__(self, repo_id: str, revision: str | None = None, cache_dir: str | None = None, token: str | None = None):
        super().__init__()
        self.repo_id = repo_id
        self.revision = revision
        self.cache_dir = cache_dir
        self.token = token

    @QtCore.Slot()
    def run(self):
        try:
            self.log.emit(f"Downloading model snapshot: {self.repo_id} {('rev='+self.revision) if self.revision else ''}")
            from huggingface_hub import snapshot_download
            local_dir = snapshot_download(
                repo_id=self.repo_id,
                revision=self.revision,
                cache_dir=self.cache_dir,
                token=self.token,
                resume_download=True,
            )
            self.finished.emit(local_dir)
        except Exception as e:
            self.failed.emit(str(e))

class RenderWorker(QtCore.QObject):
    progress = QtCore.Signal(float, str)
    log = QtCore.Signal(str)
    finished = QtCore.Signal(list)
    failed = QtCore.Signal(str)

    # Per-clip live signals (for timeline overlay during a full render)
    clip_state = QtCore.Signal(int, str, bool)      # idx, state, cache_hit
    clip_progress = QtCore.Signal(int, float, str)  # idx, p, msg
    clip_asset = QtCore.Signal(int, str, str)       # idx, kind, path

    def __init__(self, cfg: ProjectConfig, resume_folder: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.resume_folder = resume_folder
        self._cancel = False
        self.engine = WanEngine()

    @QtCore.Slot()
    def run(self):
        import traceback
        try:
            if self.resume_folder:
                out = self.engine.resume_from_folder(
                    self.resume_folder,
                    progress_global=lambda p, m: self.progress.emit(float(p), str(m)),
                    log=lambda s: self.log.emit(str(s)),
                    is_cancelled=lambda: self._cancel,
                    clip_state=lambda idx, st, hit: self.clip_state.emit(int(idx), str(st), bool(hit)),
                    clip_progress=lambda idx, p, msg: self.clip_progress.emit(int(idx), float(p), str(msg)),
                    clip_asset=lambda idx, kind, path: self.clip_asset.emit(int(idx), str(kind), str(path)),
                )
                self.finished.emit([out])
            else:
                outs = self.engine.render_project(
                    self.cfg,
                    progress_global=lambda p, m: self.progress.emit(float(p), str(m)),
                    log=lambda s: self.log.emit(str(s)),
                    is_cancelled=lambda: self._cancel,
                    clip_state=lambda idx, st, hit: self.clip_state.emit(int(idx), str(st), bool(hit)),
                    clip_progress=lambda idx, p, msg: self.clip_progress.emit(int(idx), float(p), str(msg)),
                    clip_asset=lambda idx, kind, path: self.clip_asset.emit(int(idx), str(kind), str(path)),
                )
                self.finished.emit(outs if isinstance(outs, list) else [outs])
        except Exception as e:
            tb = traceback.format_exc()
            msg = f"{e}\n\n{tb}"
            if "CUDA out of memory" in str(e) or "torch.OutOfMemoryError" in tb:
                hint = (
                    "\n\n--- VRAM FIX (recommended) ---\n"
                    "1) In the app: choose model 'Wan2.2-TI2V-5B' (fits better on 24GB).\n"
                    "2) Enable CPU offload, and set 'VAE decode FP32' OFF.\n"
                    "3) Use smaller base res (e.g. 832x480) and fewer steps.\n"
                    "4) Close other GPU apps (check nvidia-smi).\n"
                    "5) Optional: export PYTORCH_ALLOC_CONF=expandable_segments:True\n"
                )
                msg += hint
            self.failed.emit(msg)

    @QtCore.Slot()
    def cancel(self):
        self._cancel = True
        try:
            self.log.emit("Annulation demandée...")
        except Exception:
            pass


class ClipBatchRenderWorker(QtCore.QObject):
    """Render a set of clips independently, updating per-clip proxy/final paths."""
    progress = QtCore.Signal(float, str)
    log = QtCore.Signal(str)
    finished = QtCore.Signal(object)  # dict(kind=str, results=[(idx, path)])
    failed = QtCore.Signal(str)

    # Per-clip live signals (for timeline overlay)
    clip_state = QtCore.Signal(int, str, bool)   # idx, state, cache_hit
    clip_progress = QtCore.Signal(int, float, str)  # idx, p, msg
    clip_log = QtCore.Signal(int, str)
    clip_asset = QtCore.Signal(int, str, str)  # idx, kind, path

    def __init__(self, cfg: ProjectConfig, scene_indices: list[int], kind: str):
        super().__init__()
        self.cfg = cfg
        self.scene_indices = [int(i) for i in scene_indices]
        self.kind = str(kind)
        self._cancel = False
        self._cancel_current = False
        self._cancel_current_idx = None
        self._cancel_clip_indices = set()
        self._current_idx = None
        self.engine = WanEngine()

    @QtCore.Slot()
    def cancel_current_clip(self):
        """Cancel only the clip currently being rendered (best-effort)."""
        if self._current_idx is None:
            return
        try:
            self._cancel_clip_indices.add(int(self._current_idx))
        except Exception:
            pass
        self._cancel_current = True
        self._cancel_current_idx = int(self._current_idx)
        try:
            self.log.emit(f"Annulation du clip {int(self._current_idx)+1} demandée…")
        except Exception:
            pass

    @QtCore.Slot(int)
    def request_cancel_clip(self, idx: int):
        """Request cancellation for a specific clip index."""
        try:
            idx = int(idx)
        except Exception:
            return
        self._cancel_clip_indices.add(idx)
        if self._current_idx is not None and int(self._current_idx) == idx:
            self._cancel_current = True
            self._cancel_current_idx = idx
            try:
                self.log.emit(f"Annulation du clip {idx+1} demandée…")
            except Exception:
                pass

    @QtCore.Slot()
    def run(self):
        import copy, re, os
        try:
            results = []
            total = max(1, len(self.scene_indices))
            for n, idx in enumerate(self.scene_indices, start=1):
                self._current_idx = int(idx)
                if self._cancel:
                    raise RuntimeError("Annulé par l'utilisateur")

                if int(idx) in self._cancel_clip_indices and not (self._cancel_current and self._cancel_current_idx == int(idx)):
                    try:
                        self.clip_state.emit(int(idx), 'skipped', False)
                        self.clip_progress.emit(int(idx), 0.0, 'skipped')
                    except Exception:
                        pass
                    continue

                try:
                    self.clip_state.emit(int(idx), 'rendering', False)
                    self.clip_progress.emit(int(idx), 0.0, 'start')
                except Exception:
                    pass

                sc = copy.deepcopy(self.cfg.scenes[idx])
                cfg2 = copy.deepcopy(self.cfg)
                cfg2.scenes = [sc]

                # Unique per-clip project name
                label = (sc.label or f"CLIP{idx+1}").strip()
                safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", label)[:40]
                suffix = 'PROXY' if self.kind == 'proxy' else 'FINAL'
                cfg2.project_name = f"{self.cfg.project_name}_{suffix}_{idx+1:02d}_{safe}"

                # Apply kind-specific settings
                if self.kind == 'proxy':
                    cfg2.post.enabled = False
                    cfg2.keep_png_frames = False
                    cfg2.studio.write_report = False
                    cfg2.studio.export_shotlist_csv = False

                    # Low-res proxy (keep aspect)
                    base_w = int(getattr(self.cfg, 'width', 960) or 960)
                    base_h = int(getattr(self.cfg, 'height', 544) or 544)
                    aspect = base_h / max(1, base_w)
                    target_w = 640 if base_w >= base_h else 720
                    target_h = int(round(target_w * aspect))
                    # keep reasonable bounds
                    target_w = max(256, min(960, target_w))
                    target_h = max(256, min(960, target_h))
                    # divisible by 16
                    target_w = (target_w // 16) * 16
                    target_h = (target_h // 16) * 16
                    cfg2.width = target_w
                    cfg2.height = target_h

                    cfg2.num_inference_steps = int(min(getattr(cfg2, 'num_inference_steps', 16), 8))
                    cfg2.guidance_scale = float(min(getattr(cfg2, 'guidance_scale', 5.0), 3.5))
                    cfg2.guidance_scale_2 = float(min(getattr(cfg2, 'guidance_scale_2', 5.0), 3.5))

                else:  # final
                    cfg2.post.enabled = True
                    cfg2.post.out_width = 1920
                    cfg2.post.out_height = 1080
                    cfg2.post.export_master_prores = True
                    cfg2.post.export_delivery_h265_main10 = True

                # Run render
                def pg(p, msg):
                    # global progress across clips
                    base = (n - 1) / total
                    span = 1.0 / total
                    try:
                        self.clip_progress.emit(int(idx), float(p), str(msg))
                    except Exception:
                        pass
                    self.progress.emit(base + span * float(p), f"{suffix} {n}/{total}: {msg}")

                try:
                    out = self.engine.render_project(
                        cfg2,
                        progress_global=lambda p, m: pg(p, m),
                        log=lambda s: (self.log.emit(str(s)), self.clip_log.emit(int(idx), str(s))),
                        is_cancelled=lambda: (self._cancel or (self._cancel_current and self._cancel_current_idx == int(idx))),
                        clip_asset=lambda _i, kind, path: self.clip_asset.emit(int(idx), str(kind), str(path)),
                    )
                except Exception as e:
                    # If only the current clip was cancelled, keep going.
                    # NOTE: keep this check defensive; cancellation may bubble up with different messages
                    # depending on the backend/tooling.
                    if (
                        (not self._cancel)
                        and self._cancel_current
                        and (self._cancel_current_idx == int(idx))
                        and ("Annulé par l'utilisateur" in str(e) or "Annulé" in str(e) or "Cancelled" in str(e))
                    ):
                        try:
                            self.clip_state.emit(int(idx), 'cancelled', False)
                        except Exception:
                            pass
                        self._cancel_current = False
                        self._cancel_current_idx = None
                        continue
                    raise

                out_path = out[-1] if isinstance(out, list) and out else (out if isinstance(out, str) else '')
                results.append((idx, out_path))
                if self._cancel_current and self._cancel_current_idx == int(idx):
                    self._cancel_current = False
                    self._cancel_current_idx = None

                try:
                    cache_hit = bool(getattr(cfg2.scenes[0], '_clip_cache_hit', False))
                    self.clip_state.emit(int(idx), 'done', cache_hit)
                except Exception:
                    pass

            self.finished.emit({'kind': self.kind, 'results': results})
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")

    @QtCore.Slot()
    def cancel(self):
        self._cancel = True
        try:
            self.log.emit('Annulation demandée…')
        except Exception:
            pass


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Wan Studio Ultimate+ Studio UI (wan_studio {WS_VERSION} / {BUILD_TAG})")
        self.resize(1680, 1020)

        # Persist UI state (docks, geometry, last tab, etc.)
        self._settings = QtCore.QSettings()

        self.cfg = ProjectConfig()

        # Track the last rendered output so "Reload last" can work even
        # before the first render has completed.
        self._last_output_video: Optional[str] = None

        self.worker: Optional[RenderWorker] = None
        self.thread: Optional[QtCore.QThread] = None
        self.clip_worker: Optional[ClipBatchRenderWorker] = None
        self.clip_thread: Optional[QtCore.QThread] = None
        self.dl_worker: Optional[ModelDownloadWorker] = None
        self.dl_thread: Optional[QtCore.QThread] = None

        self._build_ui()
        self._sync_ui_from_cfg()

        # Demo-friendly default: LingBot Cine A (4090/24GB) unless user previously picked a preset.
        try:
            last_preset = str(self._settings.value("ui/quality_preset_last", "") or "").strip()
            if hasattr(self, "quality") and self.quality is not None:
                if last_preset:
                    self.quality.setCurrentText(last_preset)
                else:
                    self.quality.setCurrentText("LingBot Cine A (4090/24GB)")
                # Apply once to push widgets + LingBot sampler defaults.
                self._apply_quality_preset()
        except Exception:
            pass


        # Guards for table-driven edits (prevents feedback loops / lost edits)
        self._table_loading = False
        self._scene_table_dirty = False

        # Restore saved UI layout if present (never crash on stale state)
        try:
            self._restore_window_state()
        except Exception:
            pass

        # Restore last window/dock layout if present
        self._restore_window_state()

        self.vram_timer = QtCore.QTimer(self)
        self.vram_timer.setInterval(1000)
        self.vram_timer.timeout.connect(self._update_vram)
        self.vram_timer.start()

        self.autosave_timer = QtCore.QTimer(self)
        self.autosave_timer.setInterval(15_000)
        self.autosave_timer.timeout.connect(self._autosave)
        self.autosave_timer.start()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.tabs = QtWidgets.QTabWidget()
        # More "modern" navigation: vertical tabs (Premiere/Resolve-like)
        self.tabs.setTabPosition(QtWidgets.QTabWidget.TabPosition.West)
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)
        root.addWidget(self.tabs, 1)

        # --- Studio docks (Preview + Log) ---
        self._build_docks()

        self.tab_premiere = QtWidgets.QWidget()
        self.tab_quick = QtWidgets.QWidget()
        self.tab_project = QtWidgets.QWidget()
        self.tab_timeline = QtWidgets.QWidget()
        self.tab_model = QtWidgets.QWidget()
        self.tab_lora = QtWidgets.QWidget()
        self.tab_audio = QtWidgets.QWidget()
        self.tab_post = QtWidgets.QWidget()
        self.tab_batch = QtWidgets.QWidget()
        self.tab_studio = QtWidgets.QWidget()
        self.tab_render = QtWidgets.QWidget()

        # Studio-first order (left→right pipeline)
        self.tabs.addTab(self.tab_quick, "Quick Start")
        self.tabs.addTab(self.tab_model, "Modèle & Input")
        self.tabs.addTab(self.tab_timeline, "Timeline")
        self.tabs.addTab(self.tab_audio, "Audio")
        self.tabs.addTab(self.tab_lora, "LoRA / Styles")
        self.tabs.addTab(self.tab_render, "Render")
        self.tabs.addTab(self.tab_premiere, "Studio (Premiere)")
        self.tabs.addTab(self.tab_post, "Post-prod")
        self.tabs.addTab(self.tab_project, "Projet")
        self.tabs.addTab(self.tab_batch, "Batch")
        self.tabs.addTab(self.tab_studio, "Studio+")

        self._build_premiere_tab()
        self._build_quick_tab()
        self._build_project_tab()
        self._build_timeline_tab()
        self._build_audio_tab()
        self._build_model_tab()
        self._build_lora_tab()
        self._build_post_tab()
        self._build_batch_tab()
        self._build_studio_tab()
        self._build_render_tab()
        self._build_menu()
        self._build_toolbar()

    def _scrollify_tab(self, tab: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """Wrap a tab's content in a scroll area so all parameters remain accessible."""
        outer = QtWidgets.QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        sa = QtWidgets.QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        sa.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        sa.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        inner = QtWidgets.QWidget()
        sa.setWidget(inner)
        outer.addWidget(sa, 1)
        return inner

    def _build_toolbar(self):
        """Compact toolbar for the most common actions (friendly, discoverable)."""
        tb = QtWidgets.QToolBar("Main")
        tb.setObjectName("main_toolbar")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tb.setIconSize(QtCore.QSize(18, 18))
        self.addToolBar(tb)

        st = self.style()
        act_new = QtGui.QAction(st.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon), "Nouveau", self)
        act_open = QtGui.QAction(st.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton), "Ouvrir", self)
        act_save = QtGui.QAction(st.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton), "Sauver", self)
        act_render = QtGui.QAction(st.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay), "Render", self)
        act_cancel = QtGui.QAction(st.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaStop), "Cancel", self)
        act_director = QtGui.QAction(st.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MessageBoxInformation), "AI Director", self)

        # Panic button: free VRAM / unload heavy pipelines
        try:
            _clear_icon = st.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TrashIcon)
        except Exception:
            _clear_icon = st.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload)
        act_clear_vram = QtGui.QAction(_clear_icon, "Clear VRAM", self)

        act_new.triggered.connect(self.on_new)
        act_open.triggered.connect(self.on_open)
        act_save.triggered.connect(self.on_save)

        # Render controls (route to the Render tab buttons)
        act_render.triggered.connect(lambda: (self.tabs.setCurrentWidget(self.tab_render), self.on_start()))
        act_cancel.triggered.connect(self.on_cancel)

        act_director.triggered.connect(self.open_ai_director)
        act_clear_vram.triggered.connect(self.on_clear_vram_now)

        tb.addAction(act_new)
        tb.addAction(act_open)
        tb.addAction(act_save)
        tb.addSeparator()
        tb.addAction(act_render)
        tb.addAction(act_cancel)
        tb.addSeparator()
        tb.addAction(act_director)
        tb.addSeparator()
        tb.addAction(act_clear_vram)

        # Quick status (VRAM etc.)
        try:
            self._tb_status = QtWidgets.QLabel(" ")
            self._tb_status.setMinimumWidth(240)
            self._tb_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            tb.addWidget(self._tb_status)
        except Exception:
            self._tb_status = None

    def _build_menu(self):
        m = self.menuBar()
        filem = m.addMenu("Fichier")
        act_new = QtGui.QAction("Nouveau", self)
        act_open = QtGui.QAction("Ouvrir projet (.json)...", self)
        act_save = QtGui.QAction("Sauver projet (.json)...", self)
        act_resume = QtGui.QAction("Resume render from folder…", self)
        act_quit = QtGui.QAction("Quitter", self)

        act_new.triggered.connect(self.on_new)
        act_open.triggered.connect(self.on_open)
        act_save.triggered.connect(self.on_save)
        act_resume.triggered.connect(self.on_resume_from_folder)
        act_quit.triggered.connect(self.close)

        filem.addAction(act_new)
        filem.addAction(act_open)
        filem.addAction(act_save)
        filem.addSeparator()
        filem.addAction(act_resume)
        filem.addSeparator()
        filem.addAction(act_quit)

        tools = m.addMenu("Studio")
        act_pb = QtGui.QAction("Prompt Builder…", self)
        act_pb.triggered.connect(self.open_prompt_builder)
        act_open_report = QtGui.QAction("Open last report (if exists)", self)
        act_open_report.triggered.connect(self.open_last_report)
        act_director = QtGui.QAction("AI Director (Mistral)…", self)
        act_director.triggered.connect(self.open_ai_director)
        act_modeldl = QtGui.QAction("Download / cache current model…", self)
        act_modeldl.triggered.connect(self._download_selected_model)
        act_chars = QtGui.QAction("Character Bible…", self)
        act_chars.triggered.connect(self.open_character_bible)
        act_locs = QtGui.QAction("Locations…", self)
        act_locs.triggered.connect(self.open_locations)

        act_init_settings = QtGui.QAction("Init Image Settings…", self)
        act_init_settings.triggered.connect(self.open_init_image_settings)
        act_cache_mgr = QtGui.QAction("Cache Manager…", self)
        act_cache_mgr.triggered.connect(self.open_cache_manager)

        act_clear_vram = QtGui.QAction("Unload all / Clear VRAM now", self)
        act_clear_vram.triggered.connect(self.on_clear_vram_now)
        tools.addAction(act_pb)
        tools.addAction(act_director)
        tools.addSeparator()
        tools.addAction(act_chars)
        tools.addAction(act_locs)
        tools.addSeparator()
        tools.addAction(act_init_settings)
        tools.addAction(act_cache_mgr)
        tools.addSeparator()
        tools.addAction(act_clear_vram)
        tools.addSeparator()
        tools.addAction(act_modeldl)
        tools.addAction(act_open_report)

        presets = m.addMenu("Presets")
        act_1080 = QtGui.QAction("1080p Premium (2x + 48fps)", self)
        act_fast = QtGui.QAction("Fast Preview", self)
        act_svi = QtGui.QAction("SVI Longform (stable)", self)
        act_1080.triggered.connect(self.apply_preset_1080p)
        act_fast.triggered.connect(self.apply_preset_fast)
        act_svi.triggered.connect(self.apply_preset_svi)
        presets.addAction(act_1080)
        presets.addAction(act_fast)
        presets.addAction(act_svi)


    def _build_premiere_tab(self):
        lay = QtWidgets.QVBoxLayout(self.tab_premiere)
        lay.setContentsMargins(0, 0, 0, 0)
        self.premiere = StudioPremiereTab(
            self,
            # Use getattr() so the app still starts even if optional callbacks
            # are missing (e.g. when running a partial build or an older plugin set).
            image_gen_cb=getattr(self, "_ai_generate_init_image_for_scene", None),
            regen_init_cb=getattr(self, "_regen_init_frame_for_scene", None),
            init_cache_status_cb=getattr(self, "_get_init_cache_status_for_scene", None),
            use_cache_cb=getattr(self, "_use_cached_init_frame_for_scene", None),
        )
        lay.addWidget(self.premiere, 1)
    
        # ---------------- Quick Start (guided) ----------------
    
    def _build_docks(self):
        # Preview dock (last output)
        self.dock_preview = QtWidgets.QDockWidget("Preview (Last Output)", self)
        self.dock_preview.setObjectName("dock_preview")
        self.dock_preview.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        pw = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(pw)
        pl.setContentsMargins(6, 6, 6, 6)

        self.preview_video = QVideoWidget()
        self.preview_video.setMinimumHeight(260)
        pl.addWidget(self.preview_video, 1)

        self.preview_player = QMediaPlayer(self)
        self.preview_audio = QAudioOutput(self)
        self.preview_audio.setVolume(0.8)
        self.preview_player.setAudioOutput(self.preview_audio)
        self.preview_player.setVideoOutput(self.preview_video)

        row = QtWidgets.QHBoxLayout()
        self.btn_prev_play = QtWidgets.QPushButton("Play")
        self.btn_prev_pause = QtWidgets.QPushButton("Pause")
        self.btn_prev_stop = QtWidgets.QPushButton("Stop")
        self.btn_prev_open = QtWidgets.QPushButton("Open file")
        self.btn_prev_reload = QtWidgets.QPushButton("Reload last")
        row.addWidget(self.btn_prev_play)
        row.addWidget(self.btn_prev_pause)
        row.addWidget(self.btn_prev_stop)
        row.addStretch(1)
        row.addWidget(self.btn_prev_reload)
        row.addWidget(self.btn_prev_open)
        pl.addLayout(row)

        self.preview_info = QtWidgets.QLabel("No output loaded yet.")
        self.preview_info.setWordWrap(True)
        pl.addWidget(self.preview_info)

        self.btn_prev_play.clicked.connect(self.preview_player.play)
        self.btn_prev_pause.clicked.connect(self.preview_player.pause)
        self.btn_prev_stop.clicked.connect(self.preview_player.stop)

        # Robust connect: avoid crashing if a hotfix method is missing due to stale bytecode.
        _reload_fn = getattr(self, "_reload_last_output", None)
        if callable(_reload_fn):
            self.btn_prev_reload.clicked.connect(_reload_fn)
        else:
            self.btn_prev_reload.setEnabled(False)

        _open_fn = getattr(self, "_open_last_output_file", None)
        if callable(_open_fn):
            self.btn_prev_open.clicked.connect(_open_fn)
        else:
            self.btn_prev_open.setEnabled(False)

        self.dock_preview.setWidget(pw)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.dock_preview)

        # Log dock
        self.dock_log = QtWidgets.QDockWidget("Log", self)
        self.dock_log.setObjectName("dock_log")
        self.dock_log.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.BottomDockWidgetArea | QtCore.Qt.DockWidgetArea.TopDockWidgetArea
        )
        lw = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(lw)
        ll.setContentsMargins(6, 6, 6, 6)
        self.log_dock_box = QtWidgets.QPlainTextEdit()
        self.log_dock_box.setReadOnly(True)
        ll.addWidget(self.log_dock_box, 1)
        self.dock_log.setWidget(lw)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_log)
        # Audio mixer dock (pro)
        try:
            self.dock_mixer = AudioMixerDock(self)
            self.dock_mixer.setObjectName("dock_mixer")
            self.dock_mixer.setAllowedAreas(
                QtCore.Qt.DockWidgetArea.LeftDockWidgetArea | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
            )
            self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.dock_mixer)
        except Exception:
            self.dock_mixer = None


        # allow user to toggle docks from menu
        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self.dock_preview.toggleViewAction())
        view_menu.addAction(self.dock_log.toggleViewAction())
        if getattr(self, "dock_mixer", None) is not None:
            view_menu.addAction(self.dock_mixer.toggleViewAction())
        view_menu.addSeparator()
        act_reset = QtGui.QAction("Reset layout", self)
        act_reset.triggered.connect(self._reset_layout)
        view_menu.addAction(act_reset)

        # persisted layout
        try:
            self.setDockOptions(
                QtWidgets.QMainWindow.DockOption.AllowTabbedDocks
                | QtWidgets.QMainWindow.DockOption.AllowNestedDocks
                | QtWidgets.QMainWindow.DockOption.AnimatedDocks
            )
        except Exception:
            pass

        self._last_output_video = None

    def _reset_layout(self):
        """Bring back docks + default positions/sizes (useful when user closed/hid panels)."""
        try:
            # Clear persisted UI state
            self._settings.remove("ui/geometry")
            self._settings.remove("ui/windowState")
            self._settings.remove("ui/lastTab")
        except Exception:
            pass

        try:
            self.dock_preview.show()
            self.dock_log.show()
            self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.dock_preview)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_log)
            if getattr(self, "dock_mixer", None) is not None:
                self.dock_mixer.show()
                self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.dock_mixer)
            self.resizeDocks([self.dock_preview], [420], QtCore.Qt.Orientation.Horizontal)
            self.resizeDocks([self.dock_log], [260], QtCore.Qt.Orientation.Vertical)
        except Exception:
            pass

    def _restore_window_state(self):
        try:
            geo = self._settings.value("ui/geometry")
            st = self._settings.value("ui/windowState")
            last_tab = self._settings.value("ui/lastTab")
            if geo is not None:
                self.restoreGeometry(geo)
            if st is not None:
                self.restoreState(st)
            if last_tab is not None:
                try:
                    idx = int(last_tab)
                    if 0 <= idx < self.tabs.count():
                        self.tabs.setCurrentIndex(idx)
                except Exception:
                    pass
        except Exception:
            pass

    def _save_window_state(self):
        try:
            self._settings.setValue("ui/geometry", self.saveGeometry())
            self._settings.setValue("ui/windowState", self.saveState())
            self._settings.setValue("ui/lastTab", self.tabs.currentIndex())
        except Exception:
            pass

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._save_window_state()
        return super().closeEvent(event)


    def _build_quick_tab(self):
        container = self._scrollify_tab(self.tab_quick)
        v = QtWidgets.QVBoxLayout(container)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(10)

        banner = QtWidgets.QLabel(
            "<h2>Workflow (recommandé)</h2>"
            "<ol>"
            "<li><b>Choisir un modèle</b> + mode (I2V ou T2V)</li>"
            "<li><b>Choisir une entrée</b> (I2V: image / EXR / frame vidéo)</li>"
            "<li><b>Écrire un prompt</b> (au moins 1 shot dans Timeline)</li>"
            "<li><b>Choisir un preset</b> (Fast / SVI / 1080p Premium)</li>"
            "<li><b>Render</b> → l'app génère des segments puis les assemble en une vidéo longue</li>"
            "</ol>"
            "<p style='opacity:.85'>Astuce: si tu veux juste 1 clip, garde <b>une seule scène</b> dans Timeline.</p>"
        )
        banner.setWordWrap(True)
        v.addWidget(banner)

        grid = QtWidgets.QGridLayout()
        v.addLayout(grid)

        # Model
        g1 = QtWidgets.QGroupBox("1) Modèle")
        f1 = QtWidgets.QFormLayout(g1)
        self.q_model_id = QtWidgets.QComboBox()
        self.q_model_id.setEditable(True)
        self.q_model_id.addItems(MODEL_PRESETS)
        self.q_mode = QtWidgets.QComboBox()
        self.q_mode.addItems(["I2V", "T2V"])
        f1.addRow("Model ID", self.q_model_id)
        f1.addRow("Mode", self.q_mode)

        # Input
        g2 = QtWidgets.QGroupBox("2) Entrée (I2V)")
        f2 = QtWidgets.QFormLayout(g2)
        self.q_input = QtWidgets.QLineEdit()
        btn_img = QtWidgets.QPushButton("Image…")
        btn_exr = QtWidgets.QPushButton("EXR…")
        btn_vid = QtWidgets.QPushButton("Frame vidéo…")
        btn_ai = QtWidgets.QPushButton("AI Gen…")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.q_input, 1)
        row.addWidget(btn_img)
        row.addWidget(btn_exr)
        row.addWidget(btn_vid)
        row.addWidget(btn_ai)
        f2.addRow("Input", row)
        self.q_preview = QtWidgets.QLabel("Preview: (none)")
        self.q_preview.setMinimumHeight(180)
        self.q_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.q_preview.setStyleSheet("border:1px solid #333; border-radius:12px; padding:8px;")
        f2.addRow(self.q_preview)

        btn_img.clicked.connect(self._choose_input_image_quick)
        btn_exr.clicked.connect(self._import_exr_quick)
        btn_vid.clicked.connect(self._extract_from_video_quick)

        # Prompt
        g3 = QtWidgets.QGroupBox("3) Prompt (shot #1)")
        f3 = QtWidgets.QFormLayout(g3)
        self.q_seconds = QtWidgets.QDoubleSpinBox()
        self.q_seconds.setRange(1.0, 60.0)
        self.q_seconds.setDecimals(2)
        self.q_seconds.setValue(5.0)
        self.q_prompt = QtWidgets.QPlainTextEdit("Cinematic shot, gentle camera motion.")
        btn_pb = QtWidgets.QPushButton("Prompt Builder…")
        btn_dir = QtWidgets.QPushButton("AI Director (Mistral)…")
        f3.addRow("Seconds", self.q_seconds)
        f3.addRow("Prompt", self.q_prompt)
        f3.addRow(btn_pb)
        btn_pb.clicked.connect(self._prompt_builder_to_quick)

        # Preset + render
        g4 = QtWidgets.QGroupBox("4) Preset & Render")
        f4 = QtWidgets.QVBoxLayout(g4)
        self.q_preset = QtWidgets.QComboBox()
        self.q_preset.addItems(["Draft Preview (Low Res)", "Fast Preview", "SVI Longform", "1080p Premium"])
        f4.addWidget(QtWidgets.QLabel("Preset"))
        f4.addWidget(self.q_preset)
        btn_apply = QtWidgets.QPushButton("Apply preset")
        btn_render = QtWidgets.QPushButton("▶ Render now")
        f4.addWidget(btn_apply)
        f4.addWidget(btn_render)
        f4.addStretch(1)
        btn_apply.clicked.connect(self._apply_quick_preset)
        btn_render.clicked.connect(self._render_from_quick)

        grid.addWidget(g1, 0, 0)
        grid.addWidget(g2, 0, 1)
        grid.addWidget(g3, 1, 0)
        grid.addWidget(g4, 1, 1)

        v.addStretch(1)

    def _render_from_quick(self):
        # push quick settings into cfg + timeline scene #1
        self._sync_cfg_from_ui()
        self.cfg.model_id = self.q_model_id.currentText().strip()
        self.cfg.mode = self.q_mode.currentText()
        self.cfg.input_image_path = self.q_input.text().strip() or None

        if not self.cfg.scenes:
            self.cfg.scenes = [SceneSpec()]

        self.cfg.scenes[0].seconds = float(self.q_seconds.value())
        self.cfg.scenes[0].prompt = self.q_prompt.toPlainText().strip() or self.cfg.scenes[0].prompt

        self._sync_ui_from_cfg()
        self.tabs.setCurrentWidget(self.tab_render)
        self.on_start()

    def _apply_quick_preset(self):
        name = self.q_preset.currentText()
        if "Draft" in name:
            self.apply_preset_draft()
        elif "Fast" in name:
            self.apply_preset_fast()
        elif "SVI" in name:
            self.apply_preset_svi()
        else:
            self.apply_preset_1080p()

    def _prompt_builder_to_quick(self):
        d = PromptBuilderDialog(self)
        d.exec()
        txt = d.prompt()
        if txt:
            self.q_prompt.setPlainText(txt)

    def _choose_input_image_quick(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choisir image", ".", "Images (*.png *.jpg *.jpeg *.webp)")
        if f:
            self.q_input.setText(f)
            self._set_preview(self.q_preview, f)

    def _import_exr_quick(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choisir EXR", ".", "EXR (*.exr)")
        if not f:
            return
        d = ExrImportDialog(self)
        if d.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        out_dir = make_inputs_dir(".")
        out_png = unique_path(out_dir, "exr", "png")
        res = exr_to_png(f, out_png, exposure_ev=float(d.exposure.value()), gamma=float(d.gamma.value()), tonemap=str(d.tonemap.currentText()))
        if not res.ok:
            QtWidgets.QMessageBox.critical(self, "EXR import failed", res.message or "Unknown error")
            return
        self.q_input.setText(res.path)
        self._set_preview(self.q_preview, res.path)
        self.append_log(f"EXR imported: {res.path} ({res.message})")

    def _extract_from_video_quick(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choisir vidéo", ".", "Vidéos (*.mp4 *.mov *.mkv *.webm *.avi)")
        if not f:
            return
        t, ok = QtWidgets.QInputDialog.getDouble(self, "Timestamp", "Extraire frame à t (secondes):", 0.0, 0.0, 10_000.0, 2)
        if not ok:
            return
        out_dir = make_inputs_dir(".")
        out_png = unique_path(out_dir, "frame", "png")
        res = extract_frame_ffmpeg(f, float(t), out_png)
        if not res.ok:
            QtWidgets.QMessageBox.critical(self, "Frame extract failed", res.message or "Unknown error")
            return
        self.q_input.setText(res.path)
        self._set_preview(self.q_preview, res.path)
        self.append_log(f"Frame extracted: {res.path}")

    # ---------------- Project tab ----------------
    def _build_project_tab(self):
        container = self._scrollify_tab(self.tab_project)
        lay = QtWidgets.QFormLayout(container)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setVerticalSpacing(10)

        self.out_dir = QtWidgets.QLineEdit(self.cfg.output_dir)
        btn_out = QtWidgets.QPushButton("…")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.out_dir, 1)
        row.addWidget(btn_out)
        lay.addRow("Dossier de sortie", row)
        _fn = getattr(self, "_choose_out_dir", None)
        if callable(_fn):
            btn_out.clicked.connect(_fn)
        else:
            btn_out.setEnabled(False)


        self.project_name = QtWidgets.QLineEdit(self.cfg.project_name)
        lay.addRow("Nom du projet", self.project_name)

        self.negative = QtWidgets.QPlainTextEdit(self.cfg.negative_prompt)
        lay.addRow("Negative prompt global", self.negative)

        self.base_seed = QtWidgets.QSpinBox()
        self.base_seed.setRange(0, 2**31-1)
        self.base_seed.setValue(self.cfg.base_seed)
        lay.addRow("Base seed", self.base_seed)

        self.vary_seed = QtWidgets.QCheckBox()
        self.vary_seed.setChecked(self.cfg.vary_seed_per_segment)
        lay.addRow("Vary seed per segment", self.vary_seed)

        self.trans_mode = QtWidgets.QComboBox()
        self.trans_mode.addItems(["cut", "crossfade", "flow"])
        self.trans_mode.setCurrentText(self.cfg.scene_transition_mode)
        lay.addRow("Scene transition mode", self.trans_mode)

        self.trans_frames = QtWidgets.QSpinBox()
        self.trans_frames.setRange(0, 240)
        self.trans_frames.setValue(self.cfg.scene_transition_frames)
        lay.addRow("Scene transition frames", self.trans_frames)

        self.trans_ease = QtWidgets.QComboBox()
        self.trans_ease.addItems(["linear", "smoothstep", "ease_in_out"])
        self.trans_ease.setCurrentText(self.cfg.scene_transition_ease)

        # Stabilité / Cache (si widgets présents)
        try:
            if hasattr(self, 'cb_cut_strict'):
                self.cb_cut_strict.setChecked(bool(getattr(self.cfg, 'cut_strict_between_clips', True)))
            if hasattr(self, 'cb_reset_cont'):
                self.cb_reset_cont.setChecked(bool(getattr(self.cfg, 'reset_continuity_between_clips', True)))
            if hasattr(self, 'cb_strict_clip'):
                self.cb_strict_clip.setChecked(bool(getattr(self.cfg, 'strict_clip_coherence', True)))
            if hasattr(self, 'cb_force_first'):
                self.cb_force_first.setChecked(bool(getattr(self.cfg, 'force_first_frame_to_conditioning', True)))
            if hasattr(self, 'cb_ultra_iso'):
                self.cb_ultra_iso.setChecked(bool(getattr(self.cfg, 'ultra_isolated_clips', False)))
            if hasattr(self, 'cb_clip_cache'):
                self.cb_clip_cache.setChecked(bool(getattr(self.cfg, 'clip_cache_enabled', True)))
            if hasattr(self, 'cb_prune_cache'):
                self.cb_prune_cache.setChecked(bool(getattr(self.cfg, 'prune_cache_on_start', True)))
            if hasattr(self, 'sp_cache_max'):
                self.sp_cache_max.setValue(float(getattr(self.cfg, 'cache_max_gb', 0.0) or 0.0))
            if hasattr(self, 'le_clip_cache_dirname'):
                self.le_clip_cache_dirname.setText(str(getattr(self.cfg, 'clip_cache_dirname', '.wan_cache') or '.wan_cache'))
        except Exception:
            pass

        lay.addRow("Transition ease", self.trans_ease)

        # --- Stability / Coherence (longform) ---
        gb_stab = QtWidgets.QGroupBox("Stabilité (longform)")
        fl_stab = QtWidgets.QFormLayout(gb_stab)

        self.cb_cut_strict = QtWidgets.QCheckBox("Cut strict entre clips (force hard cut par défaut)")
        self.cb_cut_strict.setChecked(bool(getattr(self.cfg, "cut_strict_between_clips", True)))
        self.cb_cut_strict.setToolTip("Quand activé: par défaut, les transitions entre clips sont des cuts (pas de continuité implicite).")

        self.cb_reset_cont = QtWidgets.QCheckBox("Reset continuity entre clips (clips indépendants)")
        self.cb_reset_cont.setChecked(bool(getattr(self.cfg, "reset_continuity_between_clips", True)))
        self.cb_reset_cont.setToolTip("Quand activé: chaque clip repart d'un nouvel anchor (pas de spill du dernier frame).")

        self.cb_strict_clip = QtWidgets.QCheckBox("Strict clip coherence (anti-drift / anti-hallu)")
        self.cb_strict_clip.setChecked(bool(getattr(self.cfg, "strict_clip_coherence", True)))
        self.cb_strict_clip.setToolTip("Active des garde-fous pendant un clip (chunked): seeds/verrous + prompts guardrails.")

        self.cb_force_first = QtWidgets.QCheckBox("Force 1er frame de chaque segment = conditioning")
        self.cb_force_first.setChecked(bool(getattr(self.cfg, "force_first_frame_to_conditioning", True)))
        self.cb_force_first.setToolTip("Élimine les petits 'pops' aux boundaries de segments, même avec overlap.")

        self.cb_ultra_iso = QtWidgets.QCheckBox("Ultra isolation (reset pipeline/VRAM entre clips)")
        self.cb_ultra_iso.setChecked(bool(getattr(self.cfg, "ultra_isolated_clips", False)))
        self.cb_ultra_iso.setToolTip("Mode le plus strict: unload/reload du modèle entre clips + clear CUDA. Plus lent, mais minimise toute contamination entre clips.")

        fl_stab.addRow(self.cb_cut_strict)
        fl_stab.addRow(self.cb_reset_cont)
        fl_stab.addRow(self.cb_strict_clip)
        fl_stab.addRow(self.cb_force_first)
        fl_stab.addRow(self.cb_ultra_iso)

        stab_note = QtWidgets.QLabel("Tip: pour zéro hallucination entre clips, active Reset continuity + Strict clip coherence. Pour le mode le plus strict, active Ultra isolation (reset pipeline/VRAM entre clips).")
        stab_note.setWordWrap(True)
        fl_stab.addRow(stab_note)

        lay.addRow(gb_stab)

        # --- Cache / VRAM resilience ---
        gb_cache = QtWidgets.QGroupBox("Cache (re-render rapide)")
        fl_cache = QtWidgets.QFormLayout(gb_cache)

        self.cb_clip_cache = QtWidgets.QCheckBox("Activer clip cache (skip re-render si inchangé)")
        self.cb_clip_cache.setChecked(bool(getattr(self.cfg, "clip_cache_enabled", True)))

        self.cb_prune_cache = QtWidgets.QCheckBox("Prune cache au démarrage")
        self.cb_prune_cache.setChecked(bool(getattr(self.cfg, "prune_cache_on_start", True)))

        self.sp_cache_max = QtWidgets.QDoubleSpinBox()
        self.sp_cache_max.setRange(0.0, 5000.0)
        self.sp_cache_max.setDecimals(2)
        self.sp_cache_max.setSingleStep(1.0)
        self.sp_cache_max.setValue(float(getattr(self.cfg, "cache_max_gb", 0.0) or 0.0))
        self.sp_cache_max.setToolTip("0 = illimité. Si >0, l'app prune les plus vieux caches pour rester sous la limite.")

        self.btn_clear_cache = QtWidgets.QPushButton("Clear cache maintenant")
        self.btn_clear_cache.clicked.connect(self.on_clear_cache_now)

        fl_cache.addRow(self.cb_clip_cache)
        fl_cache.addRow(self.cb_prune_cache)
        self.le_clip_cache_dirname = QtWidgets.QLineEdit(str(getattr(self.cfg, 'clip_cache_dirname', '.wan_cache') or '.wan_cache'))
        self.le_clip_cache_dirname.setToolTip("Sous-dossier utilisé pour stocker le clip cache dans output_dir/project_name. Ex: .wan_cache")
        fl_cache.addRow("Clip cache dirname", self.le_clip_cache_dirname)
        fl_cache.addRow("Cache max (GB)", self.sp_cache_max)
        fl_cache.addRow(self.btn_clear_cache)

        lay.addRow(gb_cache)


    # ---------------- Timeline tab ----------------
    def _build_timeline_tab(self):
        v = QtWidgets.QVBoxLayout(self.tab_timeline)

        top = QtWidgets.QHBoxLayout()
        v.addLayout(top)

        self.scene_filter = QtWidgets.QLineEdit()
        self.scene_filter.setPlaceholderText("Filter (label/tags/prompt)…")
        btn_add = QtWidgets.QPushButton("+ Ajouter scène")
        btn_dup = QtWidgets.QPushButton("⎘ Dupliquer")
        btn_del = QtWidgets.QPushButton("− Supprimer")
        btn_up = QtWidgets.QPushButton("↑ Monter")
        btn_down = QtWidgets.QPushButton("↓ Descendre")
        btn_prompt = QtWidgets.QPushButton("Prompt Builder → scène")

        top.addWidget(self.scene_filter, 1)
        top.addWidget(btn_add)
        top.addWidget(btn_dup)
        top.addWidget(btn_del)
        top.addWidget(btn_prompt)
        top.addStretch(1)
        top.addWidget(btn_up)
        top.addWidget(btn_down)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        v.addWidget(splitter, 1)

        self.scene_table = QtWidgets.QTableWidget()
        self.scene_table.setColumnCount(12)
        self.scene_table.setHorizontalHeaderLabels([
            "Label", "Tags", "Sec", "Prompt", "Neg (opt)", "Steps", "CFG1", "CFG2", "Boundary", "Blend (opt)", "TransFrames (opt)", "Ease (opt)"
        ])
        self.scene_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.scene_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.scene_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.scene_table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        self.scene_table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch)
        splitter.addWidget(self.scene_table)

        ins = QtWidgets.QWidget()
        ins_l = QtWidgets.QFormLayout(ins)
        self.ins_label = QtWidgets.QLineEdit()
        self.ins_tags = QtWidgets.QLineEdit()
        self.ins_notes = QtWidgets.QPlainTextEdit()
        self.ins_apply = QtWidgets.QPushButton("Apply to selected")
        self.ins_autofill = QtWidgets.QPushButton("Autofill label Sxx_SHxx")
        ins_l.addRow("Label", self.ins_label)
        ins_l.addRow("Tags", self.ins_tags)
        ins_l.addRow("Notes", self.ins_notes)
        # Last output quick access (global)
        last_box = QtWidgets.QGroupBox("Last output")
        lb_l = QtWidgets.QVBoxLayout(last_box)
        self.tl_last_info = QtWidgets.QLabel("—")
        self.tl_last_info.setWordWrap(True)
        btns2 = QtWidgets.QHBoxLayout()
        self.btn_tl_reload_last = QtWidgets.QPushButton("Reload in Preview dock")
        self.btn_tl_open_last = QtWidgets.QPushButton("Open file")
        btns2.addWidget(self.btn_tl_reload_last)
        btns2.addWidget(self.btn_tl_open_last)
        lb_l.addWidget(self.tl_last_info)
        lb_l.addLayout(btns2)
        ins_l.addRow(last_box)

        _reload_fn = getattr(self, '_reload_last_output', None)
        if callable(_reload_fn):
            self.btn_tl_reload_last.clicked.connect(_reload_fn)
        else:
            self.btn_tl_reload_last.setEnabled(False)
        self.btn_tl_open_last.clicked.connect(self._open_last_output_file)
        rowb = QtWidgets.QHBoxLayout()
        rowb.addWidget(self.ins_apply)
        rowb.addWidget(self.ins_autofill)
        ins_l.addRow(rowb)
        splitter.addWidget(ins)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)

        btn_add.clicked.connect(self.on_add_scene)
        btn_del.clicked.connect(self.on_del_scene)
        btn_up.clicked.connect(lambda: self._move_scene(-1))
        btn_down.clicked.connect(lambda: self._move_scene(1))
        btn_dup.clicked.connect(self.on_dup_scene)
        btn_prompt.clicked.connect(self.on_prompt_to_scene)

        self.scene_table.itemSelectionChanged.connect(self._on_scene_select)
        self.ins_apply.clicked.connect(self._apply_inspector)
        self.ins_autofill.clicked.connect(self._autofill_labels)
        self.scene_filter.textChanged.connect(self._apply_filter)

        # Debounced table -> cfg syncing (keeps Studio timeline + render config consistent)
        self._scene_table_timer = QtCore.QTimer(self)
        self._scene_table_timer.setSingleShot(True)
        self._scene_table_timer.timeout.connect(self._commit_scene_table_edits)
        self.scene_table.itemChanged.connect(self._on_scene_table_item_changed)

        v.addWidget(QtWidgets.QLabel("TransFrames/Ease (opt) s'appliquent à la transition INTO cette scène."))

    # ---------------- Audio tab ----------------
    def _build_audio_tab(self):
        container = self._scrollify_tab(self.tab_audio)
        v = QtWidgets.QVBoxLayout(container)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(10)

        info = QtWidgets.QLabel(
            "Audio (pro): musique globale + dialogues (TTS Piper) + VFX par plan. "
            "Workflow: (1) générer WAV dialogue par plan, (2) importer WAV VFX par plan, (3) sélectionner une musique globale, "
            "(4) build stems + master mix, (5) mux dans une vidéo." 
        )
        info.setWordWrap(True)
        v.addWidget(info)

        # Audio-only preview player (for master/dialogue/music/VFX WAVs)
        self.audio_preview_player = QMediaPlayer(self)
        self.audio_preview_output = QAudioOutput(self)
        self.audio_preview_output.setVolume(0.8)
        self.audio_preview_player.setAudioOutput(self.audio_preview_output)

        # Attach preview player to mixer dock (if present)
        try:
            if getattr(self, "dock_mixer", None) is not None:
                self.dock_mixer.attach_player(self.audio_preview_player)
                self.dock_mixer.sync_from_cfg(self.cfg)
        except Exception:
            pass



        # --- global settings
        g = QtWidgets.QGroupBox("Audio - Global")
        gl = QtWidgets.QFormLayout(g)
        a = self.cfg.audio

        self.audio_enabled = QtWidgets.QCheckBox("Enable audio pipeline")
        self.audio_enabled.setChecked(bool(getattr(a, "enabled", True)))

        self.cb_tts = QtWidgets.QComboBox()
        self.cb_tts.addItems(["piper", "none"])
        self.cb_tts.setCurrentText(getattr(a, "tts_engine", "piper") or "piper")

        self.le_voice_onnx = QtWidgets.QLineEdit(getattr(a, "voice_model_path", "") or "")
        self.le_voice_json = QtWidgets.QLineEdit(getattr(a, "voice_config_path", "") or "")
        btn_voice_onnx = QtWidgets.QPushButton("Browse")
        btn_voice_json = QtWidgets.QPushButton("Browse")
        btn_detect = QtWidgets.QPushButton("Detect Piper")

        self.le_lang = QtWidgets.QLineEdit(getattr(a, "default_language", "fr") or "fr")
        self.cb_fit = QtWidgets.QCheckBox("Auto-fit speech to shot duration")
        self.cb_fit.setChecked(bool(getattr(a, "auto_fit_to_scene", True)))

        # Cinema mode: optionally extend shot duration to match dialogue length.
        self.cb_extend = QtWidgets.QCheckBox("Cinema: extend shot if dialogue is longer")
        self.cb_extend.setChecked(bool(getattr(a, "auto_extend_scene_to_dialogue", False)))
        self.cb_extend.setToolTip(
            "If enabled, when TTS is longer than the shot duration, we extend the shot duration instead of speeding up / trimming speech."
        )

        self.sp_extend_pad = QtWidgets.QDoubleSpinBox()
        self.sp_extend_pad.setRange(0.0, 2.0)
        self.sp_extend_pad.setSingleStep(0.05)
        self.sp_extend_pad.setDecimals(2)
        self.sp_extend_pad.setValue(float(getattr(a, "extend_pad_seconds", 0.20)))

        # Automation when applying AI storyboard
        self.cb_auto_audio_apply = QtWidgets.QCheckBox("Auto-generate dialogue after storyboard apply")
        self.cb_auto_audio_apply.setChecked(bool(getattr(a, "auto_generate_on_storyboard_apply", False)))
        self.cb_auto_build_apply = QtWidgets.QCheckBox("…and build tracks + master mix")
        self.cb_auto_build_apply.setChecked(bool(getattr(a, "auto_build_mix_on_storyboard_apply", False)))
        self.cb_auto_build_apply.setToolTip("Runs the full audio build (dialogue/vfx/music/master + SRT) after generating dialogue audio.")

        self.cb_export_scene_stems = QtWidgets.QCheckBox("Export per-scene stems (dialogue + vfx)")
        self.cb_export_scene_stems.setChecked(bool(getattr(a, "export_scene_stems", True)))

        self.cb_export_srt = QtWidgets.QCheckBox("Export subtitles (SRT) from dialogue text")
        self.cb_export_srt.setChecked(bool(getattr(a, "export_srt", True)))

        self.le_music = QtWidgets.QLineEdit(getattr(a, "music_path", "") or "")
        btn_music = QtWidgets.QPushButton("Browse")
        btn_clear_music = QtWidgets.QPushButton("Clear")

        gl.addRow(self.audio_enabled)
        gl.addRow("TTS engine", self.cb_tts)

        row1 = QtWidgets.QHBoxLayout(); row1.addWidget(self.le_voice_onnx, 1); row1.addWidget(btn_voice_onnx)
        wrow1 = QtWidgets.QWidget(); wrow1.setLayout(row1)
        gl.addRow("Voice model (.onnx)", wrow1)

        row2 = QtWidgets.QHBoxLayout(); row2.addWidget(self.le_voice_json, 1); row2.addWidget(btn_voice_json)
        wrow2 = QtWidgets.QWidget(); wrow2.setLayout(row2)
        gl.addRow("Voice config (.json)", wrow2)

        gl.addRow("Default language", self.le_lang)
        gl.addRow(self.cb_fit)
        row_ext = QtWidgets.QHBoxLayout(); row_ext.addWidget(self.cb_extend, 1); row_ext.addWidget(QtWidgets.QLabel("pad")); row_ext.addWidget(self.sp_extend_pad)
        wrow_ext = QtWidgets.QWidget(); wrow_ext.setLayout(row_ext)
        gl.addRow(wrow_ext)

        gl.addRow(self.cb_auto_audio_apply)
        gl.addRow(self.cb_auto_build_apply)
        gl.addRow(self.cb_export_scene_stems)
        gl.addRow(self.cb_export_srt)

        rowm = QtWidgets.QHBoxLayout(); rowm.addWidget(self.le_music, 1); rowm.addWidget(btn_music); rowm.addWidget(btn_clear_music)
        wrowm = QtWidgets.QWidget(); wrowm.setLayout(rowm)
        gl.addRow("Global music (optional)", wrowm)

        # Auto music generation (MusicGen)
        self.cb_music_autogen = QtWidgets.QCheckBox("Auto-generate music bed (MusicGen if available)")
        self.cb_music_autogen.setChecked(bool(getattr(a, "music_auto_generate", False)))

        self.cb_music_preset = QtWidgets.QComboBox()
        try:
            from .music_presets import PRESETS
            keys = list(PRESETS.keys())
            titles = [PRESETS[k].title for k in keys]
            for k, t in zip(keys, titles):
                self.cb_music_preset.addItem(t, k)
        except Exception:
            for k in ("cinematic_orchestral", "noir_jazz", "baroque_bach", "classical_mozart", "impressionist_debussy", "synthwave_80s"):
                self.cb_music_preset.addItem(k, k)
        # select current
        cur = str(getattr(a, "music_style_preset", "cinematic_orchestral"))
        for i in range(self.cb_music_preset.count()):
            if str(self.cb_music_preset.itemData(i)) == cur:
                self.cb_music_preset.setCurrentIndex(i)
                break

        self.le_music_model = QtWidgets.QLineEdit(str(getattr(a, "music_model_name", "facebook/musicgen-small")) or "facebook/musicgen-small")
        self.sp_music_max = QtWidgets.QDoubleSpinBox()
        self.sp_music_max.setRange(5.0, 600.0)
        self.sp_music_max.setDecimals(1)
        self.sp_music_max.setSingleStep(5.0)
        self.sp_music_max.setValue(float(getattr(a, "music_max_seconds", 60.0)))

        self.sp_music_seed = QtWidgets.QSpinBox()
        self.sp_music_seed.setRange(-1, 2**31 - 1)
        try:
            seed = getattr(a, "music_seed", None)
            self.sp_music_seed.setValue(int(seed) if seed is not None else -1)
        except Exception:
            self.sp_music_seed.setValue(-1)
        self.sp_music_seed.setToolTip("-1 = random")

        gl.addRow(self.cb_music_autogen)
        gl.addRow("Music preset", self.cb_music_preset)
        gl.addRow("Music model", self.le_music_model)
        gl.addRow("Music max seconds", self.sp_music_max)
        gl.addRow("Music seed", self.sp_music_seed)

        gl.addRow(btn_detect)
        v.addWidget(g)

        # --- advanced global settings (mix / stems / alignment)
        g_adv = QtWidgets.QGroupBox("Audio - Advanced")
        gal = QtWidgets.QFormLayout(g_adv)

        # IO + sample rate
        self.sp_audio_sample_rate = QtWidgets.QSpinBox()
        self.sp_audio_sample_rate.setRange(8000, 96000)
        self.sp_audio_sample_rate.setSingleStep(1000)
        self.sp_audio_sample_rate.setValue(int(getattr(a, "sample_rate", 48000) or 48000))

        self.le_audio_dirname = QtWidgets.QLineEdit(str(getattr(a, "audio_dir_name", "audio") or "audio"))
        self.le_audio_track_filename = QtWidgets.QLineEdit(str(getattr(a, "track_filename", "dialogue.wav") or "dialogue.wav"))
        self.le_audio_music_track_filename = QtWidgets.QLineEdit(str(getattr(a, "music_track_filename", "music.wav") or "music.wav"))
        self.le_audio_vfx_track_filename = QtWidgets.QLineEdit(str(getattr(a, "vfx_track_filename", "vfx.wav") or "vfx.wav"))
        self.le_audio_master_track_filename = QtWidgets.QLineEdit(str(getattr(a, "master_track_filename", "master.wav") or "master.wav"))
        self.le_audio_srt_filename = QtWidgets.QLineEdit(str(getattr(a, "srt_filename", "subtitles.srt") or "subtitles.srt"))

        # Stems
        self.cb_export_shot_stems = QtWidgets.QCheckBox("Export per-shot stems (dialogue + vfx)")
        self.cb_export_shot_stems.setChecked(bool(getattr(a, "export_shot_stems", True)))
        self.le_shot_stems_dir = QtWidgets.QLineEdit(str(getattr(a, "shot_stems_dir_name", "shots") or "shots"))
        self.le_scene_stems_dir = QtWidgets.QLineEdit(str(getattr(a, "scene_stems_dir_name", "scenes") or "scenes"))

        # Gains (dB)
        def _gain_spin(v):
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(-60.0, 20.0)
            sp.setSingleStep(1.0)
            sp.setDecimals(1)
            sp.setValue(float(v))
            return sp

        self.sp_music_gain = _gain_spin(getattr(a, "music_gain_db", -10.0))
        self.sp_dialogue_gain = _gain_spin(getattr(a, "dialogue_gain_db", 0.0))
        self.sp_vfx_gain = _gain_spin(getattr(a, "vfx_gain_db", -3.0))
        self.sp_master_gain = _gain_spin(getattr(a, "master_gain_db", 0.0))

        # Quick mute toggles
        self.cb_mute_music = QtWidgets.QCheckBox("Mute music")
        self.cb_mute_dialogue = QtWidgets.QCheckBox("Mute dialogue")
        self.cb_mute_vfx = QtWidgets.QCheckBox("Mute VFX")
        self.cb_mute_music.setChecked(bool(getattr(a, "music_muted", False)))
        self.cb_mute_dialogue.setChecked(bool(getattr(a, "dialogue_muted", False)))
        self.cb_mute_vfx.setChecked(bool(getattr(a, "vfx_muted", False)))

        # Ducking music under dialogue
        self.cb_duck = QtWidgets.QCheckBox("Duck music under dialogue")
        self.cb_duck.setChecked(bool(getattr(a, "duck_music_under_dialogue", True)))
        self.sp_duck_threshold = QtWidgets.QDoubleSpinBox(); self.sp_duck_threshold.setRange(0.0, 1.0); self.sp_duck_threshold.setSingleStep(0.05); self.sp_duck_threshold.setDecimals(2)
        self.sp_duck_threshold.setValue(float(getattr(a, "duck_threshold", 0.18)))
        self.sp_duck_ratio = QtWidgets.QDoubleSpinBox(); self.sp_duck_ratio.setRange(1.0, 50.0); self.sp_duck_ratio.setSingleStep(0.5); self.sp_duck_ratio.setDecimals(1)
        self.sp_duck_ratio.setValue(float(getattr(a, "duck_ratio", 8.0)))
        self.sp_duck_attack = QtWidgets.QDoubleSpinBox(); self.sp_duck_attack.setRange(0.0, 1.0); self.sp_duck_attack.setSingleStep(0.01); self.sp_duck_attack.setDecimals(2)
        self.sp_duck_attack.setValue(float(getattr(a, "duck_attack", 0.02)))
        self.sp_duck_release = QtWidgets.QDoubleSpinBox(); self.sp_duck_release.setRange(0.0, 3.0); self.sp_duck_release.setSingleStep(0.05); self.sp_duck_release.setDecimals(2)
        self.sp_duck_release.setValue(float(getattr(a, "duck_release", 0.25)))

        # Speech auto-fit behavior
        self.sp_fit_max_speedup = QtWidgets.QDoubleSpinBox(); self.sp_fit_max_speedup.setRange(1.0, 2.0); self.sp_fit_max_speedup.setSingleStep(0.05); self.sp_fit_max_speedup.setDecimals(2)
        self.sp_fit_max_speedup.setValue(float(getattr(a, "fit_max_speedup", 1.15)))
        self.cb_fit_pad_shorter = QtWidgets.QCheckBox("If speech is shorter, pad with silence")
        self.cb_fit_pad_shorter.setChecked(bool(getattr(a, "fit_pad_shorter", True)))
        self.sp_extend_max = QtWidgets.QDoubleSpinBox(); self.sp_extend_max.setRange(0.0, 120.0); self.sp_extend_max.setSingleStep(1.0); self.sp_extend_max.setDecimals(1)
        self.sp_extend_max.setValue(float(getattr(a, "extend_max_seconds", 10.0)))

        # WhisperX alignment
        self.cb_auto_align = QtWidgets.QCheckBox("Auto-align subtitles with WhisperX (if available)")
        self.cb_auto_align.setChecked(bool(getattr(a, "auto_align_captions", False)))
        self.le_whisperx_model = QtWidgets.QLineEdit(str(getattr(a, "whisperx_model", "small") or "small"))

        gal.addRow("Sample rate", self.sp_audio_sample_rate)
        gal.addRow("Audio output folder", self.le_audio_dirname)
        gal.addRow("Dialogue track filename", self.le_audio_track_filename)
        gal.addRow("Music track filename", self.le_audio_music_track_filename)
        gal.addRow("VFX track filename", self.le_audio_vfx_track_filename)
        gal.addRow("Master mix filename", self.le_audio_master_track_filename)
        gal.addRow("SRT filename", self.le_audio_srt_filename)

        gal.addRow(self.cb_export_shot_stems)
        gal.addRow("Shot stems dir", self.le_shot_stems_dir)
        gal.addRow("Scene stems dir", self.le_scene_stems_dir)

        gal.addRow("Music gain (dB)", self.sp_music_gain)
        gal.addRow("Dialogue gain (dB)", self.sp_dialogue_gain)
        gal.addRow("VFX gain (dB)", self.sp_vfx_gain)
        gal.addRow("Master gain (dB)", self.sp_master_gain)

        row_mute = QtWidgets.QHBoxLayout()
        row_mute.addWidget(self.cb_mute_music)
        row_mute.addWidget(self.cb_mute_dialogue)
        row_mute.addWidget(self.cb_mute_vfx)
        w_mute = QtWidgets.QWidget(); w_mute.setLayout(row_mute)
        gal.addRow("Mutes", w_mute)

        row_duck = QtWidgets.QHBoxLayout()
        row_duck.addWidget(self.cb_duck)
        row_duck.addWidget(QtWidgets.QLabel('thr'))
        row_duck.addWidget(self.sp_duck_threshold)
        row_duck.addWidget(QtWidgets.QLabel('ratio'))
        row_duck.addWidget(self.sp_duck_ratio)
        row_duck.addWidget(QtWidgets.QLabel('atk'))
        row_duck.addWidget(self.sp_duck_attack)
        row_duck.addWidget(QtWidgets.QLabel('rel'))
        row_duck.addWidget(self.sp_duck_release)
        w_duck = QtWidgets.QWidget(); w_duck.setLayout(row_duck)
        gal.addRow(w_duck)

        row_fit = QtWidgets.QHBoxLayout()
        row_fit.addWidget(QtWidgets.QLabel('max speedup'))
        row_fit.addWidget(self.sp_fit_max_speedup)
        row_fit.addSpacing(12)
        row_fit.addWidget(self.cb_fit_pad_shorter)
        w_fit = QtWidgets.QWidget(); w_fit.setLayout(row_fit)
        gal.addRow("Auto-fit tuning", w_fit)

        row_ext2 = QtWidgets.QHBoxLayout()
        row_ext2.addWidget(QtWidgets.QLabel('extend max (s)'))
        row_ext2.addWidget(self.sp_extend_max)
        w_ext2 = QtWidgets.QWidget(); w_ext2.setLayout(row_ext2)
        gal.addRow("Cinema extend cap", w_ext2)

        gal.addRow(self.cb_auto_align)
        gal.addRow("WhisperX model", self.le_whisperx_model)

        v.addWidget(g_adv)

        btn_voice_onnx.clicked.connect(self._audio_browse_voice_onnx)
        btn_voice_json.clicked.connect(self._audio_browse_voice_json)
        btn_music.clicked.connect(self._audio_browse_music)
        btn_clear_music.clicked.connect(self._audio_clear_music)
        btn_detect.clicked.connect(self._audio_detect_piper)

        # --- per-shot editing
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        v.addWidget(split, 1)

        left = QtWidgets.QWidget(); ll = QtWidgets.QVBoxLayout(left)
        self.audio_scene_list = QtWidgets.QListWidget()
        ll.addWidget(QtWidgets.QLabel("Plans"))
        ll.addWidget(self.audio_scene_list, 1)

        btn_gen_all = QtWidgets.QPushButton("Generate TTS for ALL shots")
        btn_build_tracks = QtWidgets.QPushButton("Build tracks (dialogue + vfx + music + master)")
        btn_rebuild_master = QtWidgets.QPushButton("Rebuild master mix only")
        btn_rebuild_master.setToolTip("Rebuild only the master mix with current mixer settings (gains/duck/mutes).")
        btn_mux_master = QtWidgets.QPushButton("Mux MASTER into a video…")

        ll.addWidget(btn_gen_all)
        ll.addWidget(btn_build_tracks)
        ll.addWidget(btn_rebuild_master)
        ll.addWidget(btn_mux_master)

        # Preview controls
        gb_prev = QtWidgets.QGroupBox("Preview audio")
        hb_prev = QtWidgets.QHBoxLayout(gb_prev)
        self.btn_audio_prev_master = QtWidgets.QPushButton("▶ Master")
        self.btn_audio_prev_dialogue = QtWidgets.QPushButton("▶ Dialogue")
        self.btn_audio_prev_music = QtWidgets.QPushButton("▶ Music")
        self.btn_audio_prev_shot = QtWidgets.QPushButton("▶ Shot")
        self.btn_audio_prev_vfx = QtWidgets.QPushButton("▶ VFX")
        self.btn_audio_prev_stop = QtWidgets.QPushButton("⏹")
        self.sl_audio_prev_vol = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_audio_prev_vol.setRange(0, 100)
        self.sl_audio_prev_vol.setValue(80)
        hb_prev.addWidget(self.btn_audio_prev_master)
        hb_prev.addWidget(self.btn_audio_prev_dialogue)
        hb_prev.addWidget(self.btn_audio_prev_music)
        hb_prev.addWidget(self.btn_audio_prev_shot)
        hb_prev.addWidget(self.btn_audio_prev_vfx)
        hb_prev.addWidget(self.btn_audio_prev_stop)
        hb_prev.addWidget(QtWidgets.QLabel("Vol"))
        hb_prev.addWidget(self.sl_audio_prev_vol, 1)
        ll.addWidget(gb_prev)
        split.addWidget(left)

        right = QtWidgets.QWidget(); rl = QtWidgets.QFormLayout(right)
        self.audio_shot_label = QtWidgets.QLabel("—")
        self.audio_dialogue = QtWidgets.QPlainTextEdit()
        self.le_voice_override = QtWidgets.QLineEdit()

        self.cb_voice_pack_fr = QtWidgets.QComboBox()
        self.btn_install_voice_fr = QtWidgets.QPushButton("Install voice")
        self.btn_install_voice_fr.setToolTip("Télécharge la voix Piper sélectionnée (si absente) dans tools/piper/voices.")
        # pitch/rate/speaker (per-shot)
        self.sp_dialogue_speaker = QtWidgets.QSpinBox()
        self.sp_dialogue_speaker.setRange(-1, 64)
        self.sp_dialogue_speaker.setValue(-1)
        self.sp_dialogue_speaker.setToolTip("-1 = auto / voix mono-speaker")

        def _mk_dbl(default: float):
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(0.5, 2.0)
            sp.setDecimals(2)
            sp.setSingleStep(0.05)
            sp.setValue(float(default))
            return sp

        self.sp_dialogue_pitch = _mk_dbl(1.0)
        self.sp_dialogue_pitch.setToolTip("Pitch (1.0 = normal). >1.0 = plus aigu (enfant).")
        self.sp_dialogue_rate = _mk_dbl(1.0)
        self.sp_dialogue_rate.setToolTip("Rate (1.0 = normal). >1.0 = plus rapide.")

        # VFX IA (AudioGen)
        self.le_vfx_prompt = QtWidgets.QLineEdit()
        self.sp_vfx_seed = QtWidgets.QSpinBox()
        self.sp_vfx_seed.setRange(-1, 2**31-1)
        self.sp_vfx_seed.setValue(-1)
        self.sp_vfx_seed.setToolTip("-1 = random")
        self.btn_gen_vfx_sel = QtWidgets.QPushButton("Generate VFX IA for selected")
        self.btn_gen_vfx_sel.setToolTip("Génère un WAV VFX/SFX via AudioGen (audiocraft) si disponible.")
        self.le_lang_override = QtWidgets.QLineEdit()
        self.le_dialogue_wav = QtWidgets.QLineEdit(); self.le_dialogue_wav.setReadOnly(True)

        self.le_vfx = QtWidgets.QLineEdit();
        btn_vfx = QtWidgets.QPushButton("Browse")
        btn_clear_vfx = QtWidgets.QPushButton("Clear")

        self.btn_gen_sel = QtWidgets.QPushButton("Generate TTS for selected")
        self.audio_status = QtWidgets.QLabel("—")
        self.audio_status.setWordWrap(True)        # Load FR voice pack (Piper) shipped with the app.
        self._piper_fr_pack = []
        try:
            pack_path = os.path.join(os.path.dirname(__file__), "audio", "piper_library_fr.json")
            with open(pack_path, "r", encoding="utf-8") as f:
                pack = json.load(f) or {}
            # normalize into a flat list
            for cat in ("female", "male", "child"):
                for it in (pack.get(cat) or []):
                    if not isinstance(it, dict):
                        continue
                    e = dict(it)
                    e["category"] = cat
                    e.setdefault("pitch", 1.0)
                    e.setdefault("rate", 1.0)
                    self._piper_fr_pack.append(e)
        except Exception:
            self._piper_fr_pack = []

        # Populate combo
        self.cb_voice_pack_fr.clear()
        self.cb_voice_pack_fr.addItem("— (manual)", None)
        for e in self._piper_fr_pack:
            vid = str(e.get("id") or "").strip()
            if not vid:
                continue
            cat = e.get("category", "")
            prefix = {"female": "👩", "male": "👨", "child": "🧒"}.get(cat, "🎙")
            label = f"{prefix} {vid}"
            self.cb_voice_pack_fr.addItem(label, e)

        rl.addRow("Shot", self.audio_shot_label)
        rl.addRow("Dialogue text (TTS)", self.audio_dialogue)
        # Voice selection: FR pack (recommended) or manual id/path
        row_pack = QtWidgets.QHBoxLayout()
        row_pack.addWidget(self.cb_voice_pack_fr, 1)
        row_pack.addWidget(self.btn_install_voice_fr)
        w_pack = QtWidgets.QWidget(); w_pack.setLayout(row_pack)

        row_pr = QtWidgets.QHBoxLayout()
        row_pr.addWidget(QtWidgets.QLabel("speaker"))
        row_pr.addWidget(self.sp_dialogue_speaker)
        row_pr.addSpacing(10)
        row_pr.addWidget(QtWidgets.QLabel("pitch"))
        row_pr.addWidget(self.sp_dialogue_pitch)
        row_pr.addSpacing(10)
        row_pr.addWidget(QtWidgets.QLabel("rate"))
        row_pr.addWidget(self.sp_dialogue_rate)
        w_pr = QtWidgets.QWidget(); w_pr.setLayout(row_pr)

        rl.addRow("Voice pack FR", w_pack)
        rl.addRow("Dialogue voice id/path", self.le_voice_override)
        rl.addRow("Speaker/Pitch/Rate", w_pr)
        rl.addRow("Dialogue language override", self.le_lang_override)
        rl.addRow("Dialogue WAV", self.le_dialogue_wav)
        rl.addRow("VFX prompt (AudioGen)", self.le_vfx_prompt)
        row_seed = QtWidgets.QHBoxLayout()
        row_seed.addWidget(QtWidgets.QLabel("seed"))
        row_seed.addWidget(self.sp_vfx_seed)
        row_seed.addStretch(1)
        row_seed.addWidget(self.btn_gen_vfx_sel)
        w_seed = QtWidgets.QWidget(); w_seed.setLayout(row_seed)
        rl.addRow("VFX IA", w_seed)

        rowv = QtWidgets.QHBoxLayout(); rowv.addWidget(self.le_vfx, 1); rowv.addWidget(btn_vfx); rowv.addWidget(btn_clear_vfx)
        wrowv = QtWidgets.QWidget(); wrowv.setLayout(rowv)
        rl.addRow("VFX/SFX audio (optional)", wrowv)

        rl.addRow(self.btn_gen_sel)
        rl.addRow("Status", self.audio_status)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 2)

        btn_vfx.clicked.connect(self._audio_browse_vfx)
        btn_clear_vfx.clicked.connect(self._audio_clear_vfx)


        self.cb_voice_pack_fr.currentIndexChanged.connect(self._audio_on_voice_pack_changed)
        self.btn_install_voice_fr.clicked.connect(self._audio_install_selected_voice)
        self.btn_gen_vfx_sel.clicked.connect(self._audio_generate_vfx_selected)

        btn_gen_all.clicked.connect(self._audio_generate_all)
        btn_build_tracks.clicked.connect(self._audio_build_all_tracks)
        btn_rebuild_master.clicked.connect(self._audio_rebuild_master_mix)
        btn_mux_master.clicked.connect(self._audio_mux_master)
        # Preview buttons
        self.btn_audio_prev_master.clicked.connect(self._audio_preview_master)
        self.btn_audio_prev_dialogue.clicked.connect(self._audio_preview_dialogue)
        self.btn_audio_prev_music.clicked.connect(self._audio_preview_music)
        self.btn_audio_prev_shot.clicked.connect(self._audio_preview_shot_dialogue)
        self.btn_audio_prev_vfx.clicked.connect(self._audio_preview_shot_vfx)
        self.btn_audio_prev_stop.clicked.connect(self._audio_preview_stop)
        self.sl_audio_prev_vol.valueChanged.connect(lambda v: self.audio_preview_output.setVolume(float(v)/100.0))

        self.btn_gen_sel.clicked.connect(self._audio_generate_selected)
        self.audio_scene_list.currentRowChanged.connect(self._audio_on_scene_selected)

        self._audio_current_idx = None
        self._audio_loading = False
        self._refresh_audio_scene_list(keep_selection=False)

    def _refresh_audio_scene_list(self, *, keep_selection: bool = True):
        """Rebuild the audio shot list without clobbering storyboard/timeline edits.

        IMPORTANT: this must not auto-save the previous scene when the list is refreshed
        programmatically, otherwise storyboard-applied dialogues can be overwritten by
        empty UI fields.
        """
        prev = 0
        if keep_selection and getattr(self, '_audio_current_idx', None) is not None:
            try:
                prev = int(self._audio_current_idx)
            except Exception:
                prev = 0
        if not self.cfg.scenes:
            self._audio_loading = True
            try:
                self.audio_scene_list.blockSignals(True)
                self.audio_scene_list.clear()
                self.audio_scene_list.blockSignals(False)
                self._audio_current_idx = None
                self._audio_on_scene_selected(-1)
            finally:
                self._audio_loading = False
            return

        prev = max(0, min(prev, len(self.cfg.scenes) - 1))

        self._audio_loading = True
        try:
            self.audio_scene_list.blockSignals(True)
            self.audio_scene_list.clear()
            for i, s in enumerate(self.cfg.scenes):
                lbl = (s.label or f"Shot {i+1}")
                if (getattr(s, 'dialogue_text', '') or '').strip():
                    lbl += "  🔊"
                if getattr(s, 'dialogue_audio_path', None):
                    lbl += "  ✅"
                self.audio_scene_list.addItem(lbl)
            self.audio_scene_list.setCurrentRow(prev)
            self.audio_scene_list.blockSignals(False)
            self._audio_current_idx = prev
            self._audio_on_scene_selected(prev)
        finally:
            try:
                self.audio_scene_list.blockSignals(False)
            except Exception:
                pass
            self._audio_loading = False

    def _audio_sync_ui_from_cfg(self):
        """Sync Audio tab widgets from self.cfg (open/new/preset safe)."""
        try:
            a = self.cfg.audio
        except Exception:
            return

        # Global
        try:
            if hasattr(self, 'audio_enabled'):
                self.audio_enabled.setChecked(bool(getattr(a, 'enabled', True)))
            if hasattr(self, 'cb_tts'):
                self.cb_tts.setCurrentText(str(getattr(a, 'tts_engine', 'piper') or 'piper'))
            if hasattr(self, 'le_voice_onnx'):
                self.le_voice_onnx.setText(str(getattr(a, 'voice_model_path', '') or ''))
            if hasattr(self, 'le_voice_json'):
                self.le_voice_json.setText(str(getattr(a, 'voice_config_path', '') or ''))
            if hasattr(self, 'le_lang'):
                self.le_lang.setText(str(getattr(a, 'default_language', 'fr') or 'fr'))
            if hasattr(self, 'cb_fit'):
                self.cb_fit.setChecked(bool(getattr(a, 'auto_fit_to_scene', True)))
            if hasattr(self, 'cb_extend'):
                self.cb_extend.setChecked(bool(getattr(a, 'auto_extend_scene_to_dialogue', False)))
            if hasattr(self, 'sp_extend_pad'):
                self.sp_extend_pad.setValue(float(getattr(a, 'extend_pad_seconds', 0.20) or 0.20))
            if hasattr(self, 'cb_auto_audio_apply'):
                self.cb_auto_audio_apply.setChecked(bool(getattr(a, 'auto_generate_on_storyboard_apply', False)))
            if hasattr(self, 'cb_auto_build_apply'):
                self.cb_auto_build_apply.setChecked(bool(getattr(a, 'auto_build_mix_on_storyboard_apply', False)))
            if hasattr(self, 'cb_export_scene_stems'):
                self.cb_export_scene_stems.setChecked(bool(getattr(a, 'export_scene_stems', True)))
            if hasattr(self, 'cb_export_srt'):
                self.cb_export_srt.setChecked(bool(getattr(a, 'export_srt', True)))
            if hasattr(self, 'le_music'):
                self.le_music.setText(str(getattr(a, 'music_path', '') or ''))

            # MusicGen
            if hasattr(self, 'cb_music_autogen'):
                self.cb_music_autogen.setChecked(bool(getattr(a, 'music_auto_generate', False)))
            if hasattr(self, 'cb_music_preset'):
                cur = str(getattr(a, 'music_style_preset', 'cinematic_orchestral') or 'cinematic_orchestral')
                for i in range(self.cb_music_preset.count()):
                    if str(self.cb_music_preset.itemData(i)) == cur:
                        self.cb_music_preset.setCurrentIndex(i)
                        break
            if hasattr(self, 'le_music_model'):
                self.le_music_model.setText(str(getattr(a, 'music_model_name', 'facebook/musicgen-small') or 'facebook/musicgen-small'))
            if hasattr(self, 'sp_music_max'):
                self.sp_music_max.setValue(float(getattr(a, 'music_max_seconds', 60.0) or 60.0))
            if hasattr(self, 'sp_music_seed'):
                seed = getattr(a, 'music_seed', None)
                self.sp_music_seed.setValue(int(seed) if seed is not None else -1)
        except Exception:
            pass

        # Advanced
        try:
            if hasattr(self, 'sp_audio_sample_rate'):
                self.sp_audio_sample_rate.setValue(int(getattr(a, 'sample_rate', 48000) or 48000))
            if hasattr(self, 'le_audio_dirname'):
                self.le_audio_dirname.setText(str(getattr(a, 'audio_dir_name', 'audio') or 'audio'))
            if hasattr(self, 'le_audio_track_filename'):
                self.le_audio_track_filename.setText(str(getattr(a, 'track_filename', 'dialogue.wav') or 'dialogue.wav'))
            if hasattr(self, 'le_audio_music_track_filename'):
                self.le_audio_music_track_filename.setText(str(getattr(a, 'music_track_filename', 'music.wav') or 'music.wav'))
            if hasattr(self, 'le_audio_vfx_track_filename'):
                self.le_audio_vfx_track_filename.setText(str(getattr(a, 'vfx_track_filename', 'vfx.wav') or 'vfx.wav'))
            if hasattr(self, 'le_audio_master_track_filename'):
                self.le_audio_master_track_filename.setText(str(getattr(a, 'master_track_filename', 'master.wav') or 'master.wav'))
            if hasattr(self, 'le_audio_srt_filename'):
                self.le_audio_srt_filename.setText(str(getattr(a, 'srt_filename', 'subtitles.srt') or 'subtitles.srt'))

            if hasattr(self, 'cb_export_shot_stems'):
                self.cb_export_shot_stems.setChecked(bool(getattr(a, 'export_shot_stems', True)))
            if hasattr(self, 'le_shot_stems_dir'):
                self.le_shot_stems_dir.setText(str(getattr(a, 'shot_stems_dir_name', 'shots') or 'shots'))
            if hasattr(self, 'le_scene_stems_dir'):
                self.le_scene_stems_dir.setText(str(getattr(a, 'scene_stems_dir_name', 'scenes') or 'scenes'))

            if hasattr(self, 'sp_music_gain'):
                self.sp_music_gain.setValue(float(getattr(a, 'music_gain_db', -10.0) or -10.0))
            if hasattr(self, 'sp_dialogue_gain'):
                self.sp_dialogue_gain.setValue(float(getattr(a, 'dialogue_gain_db', 0.0) or 0.0))
            if hasattr(self, 'sp_vfx_gain'):
                self.sp_vfx_gain.setValue(float(getattr(a, 'vfx_gain_db', -3.0) or -3.0))
                if hasattr(self, 'sp_master_gain'):
                    self.sp_master_gain.setValue(float(getattr(a, 'master_gain_db', 0.0) or 0.0))
                if hasattr(self, 'cb_mute_music'):
                    self.cb_mute_music.setChecked(bool(getattr(a, 'music_muted', False)))
                if hasattr(self, 'cb_mute_dialogue'):
                    self.cb_mute_dialogue.setChecked(bool(getattr(a, 'dialogue_muted', False)))
                if hasattr(self, 'cb_mute_vfx'):
                    self.cb_mute_vfx.setChecked(bool(getattr(a, 'vfx_muted', False)))

            if hasattr(self, 'cb_duck'):
                self.cb_duck.setChecked(bool(getattr(a, 'duck_music_under_dialogue', True)))
            if hasattr(self, 'sp_duck_threshold'):
                self.sp_duck_threshold.setValue(float(getattr(a, 'duck_threshold', 0.18) or 0.18))
            if hasattr(self, 'sp_duck_ratio'):
                self.sp_duck_ratio.setValue(float(getattr(a, 'duck_ratio', 8.0) or 8.0))
            if hasattr(self, 'sp_duck_attack'):
                self.sp_duck_attack.setValue(float(getattr(a, 'duck_attack', 0.02) or 0.02))
            if hasattr(self, 'sp_duck_release'):
                self.sp_duck_release.setValue(float(getattr(a, 'duck_release', 0.25) or 0.25))

            if hasattr(self, 'sp_fit_max_speedup'):
                self.sp_fit_max_speedup.setValue(float(getattr(a, 'fit_max_speedup', 1.15) or 1.15))
            if hasattr(self, 'cb_fit_pad_shorter'):
                self.cb_fit_pad_shorter.setChecked(bool(getattr(a, 'fit_pad_shorter', True)))
            if hasattr(self, 'sp_extend_max'):
                self.sp_extend_max.setValue(float(getattr(a, 'extend_max_seconds', 10.0) or 10.0))

            if hasattr(self, 'cb_auto_align'):
                self.cb_auto_align.setChecked(bool(getattr(a, 'auto_align_captions', False)))
            if hasattr(self, 'le_whisperx_model'):
                self.le_whisperx_model.setText(str(getattr(a, 'whisperx_model', 'small') or 'small'))
        except Exception:
            pass


    def _audio_apply_globals(self):
        a = self.cfg.audio
        a.enabled = bool(self.audio_enabled.isChecked())
        a.tts_engine = self.cb_tts.currentText().strip() or a.tts_engine
        a.voice_model_path = self.le_voice_onnx.text().strip() or a.voice_model_path
        a.voice_config_path = self.le_voice_json.text().strip() or a.voice_config_path
        a.default_language = self.le_lang.text().strip() or a.default_language
        a.auto_fit_to_scene = bool(self.cb_fit.isChecked())
        # cinema helpers
        try:
            a.auto_extend_scene_to_dialogue = bool(self.cb_extend.isChecked())
            a.extend_pad_seconds = float(self.sp_extend_pad.value())
        except Exception:
            pass
        try:
            a.auto_generate_on_storyboard_apply = bool(self.cb_auto_audio_apply.isChecked())
            a.auto_build_mix_on_storyboard_apply = bool(self.cb_auto_build_apply.isChecked())
            if a.auto_build_mix_on_storyboard_apply:
                a.auto_generate_on_storyboard_apply = True
        except Exception:
            pass
        a.export_scene_stems = bool(self.cb_export_scene_stems.isChecked())
        a.export_srt = bool(self.cb_export_srt.isChecked())
        a.music_path = self.le_music.text().strip() or None
        # auto music gen
        try:
            a.music_auto_generate = bool(getattr(self, 'cb_music_autogen', None) and self.cb_music_autogen.isChecked())
        except Exception:
            pass
        try:
            if getattr(self, 'cb_music_preset', None):
                a.music_style_preset = str(self.cb_music_preset.currentData() or self.cb_music_preset.currentText()).strip() or getattr(a, 'music_style_preset', 'cinematic_orchestral')
        except Exception:
            pass
        try:
            if getattr(self, 'le_music_model', None):
                a.music_model_name = self.le_music_model.text().strip() or getattr(a, 'music_model_name', 'facebook/musicgen-small')
        except Exception:
            pass
        try:
            if getattr(self, 'sp_music_max', None):
                a.music_max_seconds = float(self.sp_music_max.value())
        except Exception:
            pass
        try:
            if getattr(self, 'sp_music_seed', None):
                v = int(self.sp_music_seed.value())
                a.music_seed = None if v < 0 else v
        except Exception:
            pass

        # --- Advanced audio settings (V17+) ---
        try:
            if hasattr(self, 'sp_audio_sample_rate'):
                a.sample_rate = int(self.sp_audio_sample_rate.value())
        except Exception:
            pass
        try:
            if hasattr(self, 'le_audio_dirname'):
                a.audio_dir_name = self.le_audio_dirname.text().strip() or getattr(a, 'audio_dir_name', 'audio')
            if hasattr(self, 'le_audio_track_filename'):
                a.track_filename = self.le_audio_track_filename.text().strip() or getattr(a, 'track_filename', 'dialogue.wav')
            if hasattr(self, 'le_audio_music_track_filename'):
                a.music_track_filename = self.le_audio_music_track_filename.text().strip() or getattr(a, 'music_track_filename', 'music.wav')
            if hasattr(self, 'le_audio_vfx_track_filename'):
                a.vfx_track_filename = self.le_audio_vfx_track_filename.text().strip() or getattr(a, 'vfx_track_filename', 'vfx.wav')
            if hasattr(self, 'le_audio_master_track_filename'):
                a.master_track_filename = self.le_audio_master_track_filename.text().strip() or getattr(a, 'master_track_filename', 'master.wav')
            if hasattr(self, 'le_audio_srt_filename'):
                a.srt_filename = self.le_audio_srt_filename.text().strip() or getattr(a, 'srt_filename', 'subtitles.srt')
        except Exception:
            pass
        try:
            if hasattr(self, 'cb_export_shot_stems'):
                a.export_shot_stems = bool(self.cb_export_shot_stems.isChecked())
            if hasattr(self, 'le_shot_stems_dir'):
                a.shot_stems_dir_name = self.le_shot_stems_dir.text().strip() or getattr(a, 'shot_stems_dir_name', 'shots')
            if hasattr(self, 'le_scene_stems_dir'):
                a.scene_stems_dir_name = self.le_scene_stems_dir.text().strip() or getattr(a, 'scene_stems_dir_name', 'scenes')
        except Exception:
            pass
        try:
            if hasattr(self, 'sp_music_gain'):
                a.music_gain_db = float(self.sp_music_gain.value())
            if hasattr(self, 'sp_dialogue_gain'):
                a.dialogue_gain_db = float(self.sp_dialogue_gain.value())
            if hasattr(self, 'sp_vfx_gain'):
                a.vfx_gain_db = float(self.sp_vfx_gain.value())
            if hasattr(self, 'sp_master_gain'):
                a.master_gain_db = float(self.sp_master_gain.value())
            if hasattr(self, 'cb_mute_music'):
                a.music_muted = bool(self.cb_mute_music.isChecked())
            if hasattr(self, 'cb_mute_dialogue'):
                a.dialogue_muted = bool(self.cb_mute_dialogue.isChecked())
            if hasattr(self, 'cb_mute_vfx'):
                a.vfx_muted = bool(self.cb_mute_vfx.isChecked())
        except Exception:
            pass
        try:
            if hasattr(self, 'cb_duck'):
                a.duck_music_under_dialogue = bool(self.cb_duck.isChecked())
            if hasattr(self, 'sp_duck_threshold'):
                a.duck_threshold = float(self.sp_duck_threshold.value())
            if hasattr(self, 'sp_duck_ratio'):
                a.duck_ratio = float(self.sp_duck_ratio.value())
            if hasattr(self, 'sp_duck_attack'):
                a.duck_attack = float(self.sp_duck_attack.value())
            if hasattr(self, 'sp_duck_release'):
                a.duck_release = float(self.sp_duck_release.value())
        except Exception:
            pass
        try:
            if hasattr(self, 'sp_fit_max_speedup'):
                a.fit_max_speedup = float(self.sp_fit_max_speedup.value())
            if hasattr(self, 'cb_fit_pad_shorter'):
                a.fit_pad_shorter = bool(self.cb_fit_pad_shorter.isChecked())
        except Exception:
            pass
        try:
            if hasattr(self, 'sp_extend_max'):
                a.extend_max_seconds = float(self.sp_extend_max.value())
        except Exception:
            pass
        try:
            if hasattr(self, 'cb_auto_align'):
                a.auto_align_captions = bool(self.cb_auto_align.isChecked())
            if hasattr(self, 'le_whisperx_model'):
                a.whisperx_model = self.le_whisperx_model.text().strip() or getattr(a, 'whisperx_model', 'small')
        except Exception:
            pass

    def _audio_save_scene_fields(self):
        if getattr(self, '_audio_loading', False):
            return
        if self._audio_current_idx is None:
            return
        idx = int(self._audio_current_idx)
        if idx < 0 or idx >= len(self.cfg.scenes):
            return
        s = self.cfg.scenes[idx]
        s.dialogue_text = self.audio_dialogue.toPlainText().strip() or ""
        s.dialogue_voice = self.le_voice_override.text().strip() or ""
        s.dialogue_language = self.le_lang_override.text().strip() or ""
        s.vfx_audio_path = self.le_vfx.text().strip() or None

        # extra dialogue controls
        try:
            spk = int(self.sp_dialogue_speaker.value())
            s.dialogue_speaker = spk if spk >= 0 else None
        except Exception:
            s.dialogue_speaker = None
        try:
            s.dialogue_pitch = float(self.sp_dialogue_pitch.value())
        except Exception:
            s.dialogue_pitch = 1.0
        try:
            s.dialogue_rate = float(self.sp_dialogue_rate.value())
        except Exception:
            s.dialogue_rate = 1.0
        try:
            s.vfx_prompt = self.le_vfx_prompt.text().strip() or ""
        except Exception:
            s.vfx_prompt = ""
        try:
            vs = int(self.sp_vfx_seed.value())
            s.vfx_seed = None if vs < 0 else vs
        except Exception:
            s.vfx_seed = None


    def _audio_on_scene_selected(self, row: int):
        # Save previous only for user-driven selection changes.
        if not getattr(self, '_audio_loading', False):
            try:
                self._audio_save_scene_fields()
            except Exception:
                pass

        self._audio_current_idx = row
        if row is None or row < 0 or row >= len(self.cfg.scenes):
            self.audio_shot_label.setText("—")
            self.audio_dialogue.setPlainText("")
            self.le_voice_override.setText("")
            self.le_lang_override.setText("")
            self.le_dialogue_wav.setText("")
            self.le_vfx.setText("")
            try:
                self.cb_voice_pack_fr.setCurrentIndex(0)
                self.sp_dialogue_speaker.setValue(-1)
                self.sp_dialogue_pitch.setValue(1.0)
                self.sp_dialogue_rate.setValue(1.0)
                self.le_vfx_prompt.setText("")
                self.sp_vfx_seed.setValue(-1)
            except Exception:
                pass
            return

        s = self.cfg.scenes[row]
        self.audio_shot_label.setText(s.label or f"Shot {row+1}")
        self.audio_dialogue.setPlainText(getattr(s, 'dialogue_text', '') or '')
        self.le_voice_override.setText(getattr(s, 'dialogue_voice', '') or '')
        self.le_lang_override.setText(getattr(s, 'dialogue_language', '') or '')
        self.le_dialogue_wav.setText(getattr(s, 'dialogue_audio_path', '') or '')
        self.le_vfx.setText(getattr(s, 'vfx_audio_path', '') or '')

        # New controls
        try:
            self.le_vfx_prompt.setText(getattr(s, 'vfx_prompt', '') or '')
        except Exception:
            self.le_vfx_prompt.setText("")
        try:
            vs = getattr(s, 'vfx_seed', None)
            self.sp_vfx_seed.setValue(int(vs) if vs is not None else -1)
        except Exception:
            self.sp_vfx_seed.setValue(-1)

        try:
            spk = getattr(s, 'dialogue_speaker', None)
            self.sp_dialogue_speaker.setValue(int(spk) if spk is not None else -1)
        except Exception:
            self.sp_dialogue_speaker.setValue(-1)
        try:
            self.sp_dialogue_pitch.setValue(float(getattr(s, 'dialogue_pitch', 1.0) or 1.0))
        except Exception:
            self.sp_dialogue_pitch.setValue(1.0)
        try:
            self.sp_dialogue_rate.setValue(float(getattr(s, 'dialogue_rate', 1.0) or 1.0))
        except Exception:
            self.sp_dialogue_rate.setValue(1.0)

        # Voice pack selection (if matching id)
        try:
            cur = (getattr(s, 'dialogue_voice', '') or '').strip()
            best = 0
            if cur:
                for i in range(self.cb_voice_pack_fr.count()):
                    e = self.cb_voice_pack_fr.itemData(i)
                    if isinstance(e, dict) and str(e.get('id') or '').strip() == cur:
                        best = i
                        break
            # block signals to avoid overwriting user edits on load
            self.cb_voice_pack_fr.blockSignals(True)
            self.cb_voice_pack_fr.setCurrentIndex(best)
        except Exception:
            pass
        finally:
            try:
                self.cb_voice_pack_fr.blockSignals(False)
            except Exception:
                pass

    def _audio_preview_play(self, path: str, label: str = "audio"):
        try:
            if not path or not os.path.isfile(path):
                QtWidgets.QMessageBox.information(self, "Audio preview", f"Fichier introuvable: {path or '(none)'}")
                return
            url = QtCore.QUrl.fromLocalFile(os.path.abspath(path))
            self.audio_preview_player.stop()
            self.audio_preview_player.setSource(url)
            self.audio_preview_player.play()
            try:
                if getattr(self, "dock_mixer", None) is not None:
                    self.dock_mixer.on_preview_started(path)
            except Exception:
                pass
            try:
                self.audio_status.setText(f"Preview: {label} → {path}")
            except Exception:
                pass
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Audio preview", str(e))

    def _audio_preview_stop(self):
        try:
            self.audio_preview_player.stop()
        except Exception:
            pass

    def _audio_preview_master(self):
        adir = os.path.join(self.cfg.output_dir, self.cfg.audio.audio_dir_name)
        p = os.path.join(adir, self.cfg.audio.master_track_filename)
        self._audio_preview_play(p, "master")

    def _audio_preview_dialogue(self):
        adir = os.path.join(self.cfg.output_dir, self.cfg.audio.audio_dir_name)
        p = os.path.join(adir, self.cfg.audio.track_filename)
        self._audio_preview_play(p, "dialogue")

    def _audio_preview_music(self):
        adir = os.path.join(self.cfg.output_dir, self.cfg.audio.audio_dir_name)
        p = os.path.join(adir, self.cfg.audio.music_track_filename)
        self._audio_preview_play(p, "music")

    def _audio_preview_shot_dialogue(self):
        # Prefer selected shot's dialogue WAV if present
        idx = getattr(self, '_audio_current_idx', None)
        p = None
        if idx is not None and 0 <= int(idx) < len(self.cfg.scenes):
            s = self.cfg.scenes[int(idx)]
            p = getattr(s, 'dialogue_audio_path', None)
        if not p:
            # fallback to global dialogue
            adir = os.path.join(self.cfg.output_dir, self.cfg.audio.audio_dir_name)
            p = os.path.join(adir, self.cfg.audio.track_filename)
        self._audio_preview_play(p, "shot dialogue")

    def _audio_preview_shot_vfx(self):
        idx = getattr(self, '_audio_current_idx', None)
        p = None
        if idx is not None and 0 <= int(idx) < len(self.cfg.scenes):
            s = self.cfg.scenes[int(idx)]
            p = getattr(s, 'vfx_audio_path', None)
        if not p:
            # fallback to global VFX track
            adir = os.path.join(self.cfg.output_dir, self.cfg.audio.audio_dir_name)
            p = os.path.join(adir, self.cfg.audio.vfx_track_filename)
        self._audio_preview_play(p, "shot vfx")


    def _audio_detect_piper(self):
        exe = find_piper(getattr(self.cfg.post, "tools_dir", os.path.abspath("./tools")))
        if exe:
            self.audio_status.setText(f"Piper detected: {exe}")
        else:
            self.audio_status.setText("Piper not found. Run: python get_tools.py piper")

    def _run_worker(self, fn, done_text: str, on_done=None):
        # minimal async helper
        thread = QtCore.QThread(self)
        worker = _FuncWorker(fn)
        worker.moveToThread(thread)

        def _ok(res):
            thread.quit()
            thread.wait(50)
            self.audio_status.setText(done_text)
            self._refresh_audio_scene_list(keep_selection=True)
            if callable(on_done):
                try:
                    on_done(res)
                except Exception:
                    pass

        def _err(tb):
            thread.quit()
            thread.wait(50)
            self.audio_status.setText("Audio task failed. See console.")
            print(tb)
            QtWidgets.QMessageBox.critical(self, "Audio error", tb)

        worker.finished.connect(_ok)
        worker.error.connect(_err)
        thread.started.connect(worker.run)
        thread.start()

    def _audio_generate_selected(self):
        self._audio_apply_globals()
        self._audio_save_scene_fields()
        idx = self._audio_current_idx
        if idx is None or idx < 0:
            return
        scene_idx = int(idx)
        s = self.cfg.scenes[scene_idx]
        if not (getattr(s, 'dialogue_text', '') or '').strip():
            QtWidgets.QMessageBox.information(self, "Audio", "No dialogue_text for this shot.")
            return

        project_out = self.cfg.output_dir
        def _job():
            path = generate_scene_dialogue_audio(self.cfg, scene_idx, project_out)
            s.dialogue_audio_path = path
            return path

        self._run_worker(_job, "TTS generated for selected shot.")

    def _audio_generate_all(self):
        self._audio_apply_globals()
        self._audio_save_scene_fields()
        project_out = self.cfg.output_dir

        def _job():
            for i, s in enumerate(self.cfg.scenes):
                if (getattr(s, 'dialogue_text', '') or '').strip():
                    s.dialogue_audio_path = generate_scene_dialogue_audio(self.cfg, i, project_out)
            return True

        self._run_worker(_job, "TTS generated for all shots.")

    def _audio_build_all_tracks(self):
        self._audio_apply_globals()
        self._audio_save_scene_fields()

        project_out = self.cfg.output_dir

        def _job():
            info = build_all_audio(self.cfg, project_out)
            return info

        def _done(info):
            try:
                adir = os.path.join(project_out, self.cfg.audio.audio_dir_name)
                msg = f"Audio built in: {adir}"
                if isinstance(info, dict) and info.get("master_track"):
                    msg = f"Master mix: {info.get('master_track')}"
                QtWidgets.QMessageBox.information(self, "Audio", msg)
            except Exception:
                QtWidgets.QMessageBox.information(self, "Audio", "Audio build done.")

        self._run_worker(_job, "Audio build done.", on_done=_done)


    def _audio_rebuild_master_mix(self):
        """Rebuild only the master mix using current mixer settings.

        This is useful when tweaking gains/ducking/mutes without regenerating TTS/VFX.
        """
        self._audio_apply_globals()
        self._audio_save_scene_fields()

        project_out = self.cfg.output_dir

        def _job():
            return build_master_mix(self.cfg, project_out)

        def _done(path):
            try:
                QtWidgets.QMessageBox.information(self, "Audio", f"Master mix rebuilt:\n{path}")
            except Exception:
                QtWidgets.QMessageBox.information(self, "Audio", "Master mix rebuilt.")
            # refresh timeline thumbnails/waveforms
            try:
                if hasattr(self, "premiere") and self.premiere is not None:
                    self.premiere.reload_from_cfg(keep_selection=True)
            except Exception:
                pass

        self._run_worker(_job, "Master mix rebuilt.", on_done=_done)

    def _audio_mux_master(self):
        self._audio_apply_globals()
        self._audio_save_scene_fields()

        # Choose video
        video, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select video", self.cfg.output_dir, "Video (*.mp4 *.mov *.mkv)")
        if not video:
            return
        # Use master mix if present, otherwise fallback to dialogue track.
        adir = os.path.join(self.cfg.output_dir, self.cfg.audio.audio_dir_name)
        master = os.path.join(adir, self.cfg.audio.master_track_filename)
        dialogue = os.path.join(adir, self.cfg.audio.track_filename)
        audio = master if os.path.isfile(master) else (dialogue if os.path.isfile(dialogue) else "")
        if not audio:
            QtWidgets.QMessageBox.information(self, "Audio", "No master/dialogue track found. Build tracks first.")
            return
        # Choose mux preset
        presets = [
            ("MKV (FLAC lossless) — recommandé", "mkv_flac"),
            ("MKV (PCM lossless)", "mkv_pcm"),
            ("MP4 (AAC 320k)", "mp4_aac"),
        ]
        labels = [p[0] for p in presets]
        choice, ok = QtWidgets.QInputDialog.getItem(self, "Mux audio", "Format/codec", labels, 0, False)
        if not ok:
            return
        preset = dict(presets).get(choice, "mkv_flac")
        if preset == "mp4_aac":
            default_out = os.path.join(self.cfg.output_dir, "muxed_audio.mp4")
            filt = "Video (*.mp4)"
        else:
            default_out = os.path.join(self.cfg.output_dir, "muxed_audio.mkv")
            filt = "Video (*.mkv)"

        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Output video", default_out, filt)
        if not out_path:
            return
        mux_kwargs = {}
        if preset == "mkv_flac":
            mux_kwargs = {"audio_codec": "flac"}
        elif preset == "mkv_pcm":
            mux_kwargs = {"audio_codec": "pcm_s16le"}
        else:
            mux_kwargs = {"audio_codec": "aac", "aac_bitrate": "320k"}


        def _job():
            return mux_audio(video, audio, out_path, **mux_kwargs)

        self._run_worker(_job, f"Muxed to: {out_path}")

    def _audio_browse_voice_onnx(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Piper voice .onnx", os.path.dirname(self.le_voice_onnx.text() or "./"), "ONNX (*.onnx)")
        if p:
            self.le_voice_onnx.setText(p)

    def _audio_browse_voice_json(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Piper voice .json", os.path.dirname(self.le_voice_json.text() or "./"), "JSON (*.json)")
        if p:
            self.le_voice_json.setText(p)

    def _audio_browse_music(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select music file", os.path.dirname(self.le_music.text() or "./"), "Audio (*.wav *.mp3 *.m4a *.aac)")
        if p:
            self.le_music.setText(p)

    def _audio_clear_music(self):
        self.le_music.setText("")

    def _audio_browse_vfx(self):
        if self._audio_current_idx is None or self._audio_current_idx < 0:
            return
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select VFX audio", os.path.dirname(self.le_vfx.text() or "./"), "Audio (*.wav *.mp3 *.m4a *.aac)")
        if p:
            self.le_vfx.setText(p)

    def _audio_clear_vfx(self):
        self.le_vfx.setText("")


    def _audio_on_voice_pack_changed(self, idx: int):
        # Update manual overrides + pitch/rate/speaker when selecting a pack entry
        try:
            if getattr(self, '_audio_loading', False):
                return
            e = self.cb_voice_pack_fr.itemData(int(idx)) if hasattr(self, 'cb_voice_pack_fr') else None
            if not isinstance(e, dict):
                return
            vid = str(e.get("id") or "").strip()
            if vid:
                self.le_voice_override.setText(vid)
            # speaker (optional)
            try:
                spk = e.get("speaker", None)
                self.sp_dialogue_speaker.setValue(int(spk) if spk is not None else -1)
            except Exception:
                self.sp_dialogue_speaker.setValue(-1)
            # pitch/rate
            try:
                self.sp_dialogue_pitch.setValue(float(e.get("pitch", 1.0) or 1.0))
            except Exception:
                self.sp_dialogue_pitch.setValue(1.0)
            try:
                self.sp_dialogue_rate.setValue(float(e.get("rate", 1.0) or 1.0))
            except Exception:
                self.sp_dialogue_rate.setValue(1.0)
            # default language helper
            try:
                if not (self.le_lang_override.text() or "").strip():
                    self.le_lang_override.setText("fr")
            except Exception:
                pass
        except Exception:
            pass

    def _audio_install_selected_voice(self):
        try:
            e = self.cb_voice_pack_fr.itemData(int(self.cb_voice_pack_fr.currentIndex()))
            if not isinstance(e, dict):
                QtWidgets.QMessageBox.information(self, "Piper voice", "Sélectionne une voix dans le pack FR.")
                return
            vid = str(e.get("id") or "").strip()
            if not vid:
                QtWidgets.QMessageBox.information(self, "Piper voice", "Voix invalide.")
                return
            tools_dir = self.cfg.post.tools_dir if getattr(self.cfg, "post", None) and getattr(self.cfg.post, "tools_dir", None) else os.path.abspath("./tools")

            def _job():
                # Import from project root (get_tools.py is alongside app.py)
                try:
                    import get_tools
                    get_tools.install_piper_voice(tools_dir, vid)
                    return True
                except Exception:
                    # fallback: run get_tools.py to ensure piper exists
                    import subprocess, sys
                    subprocess.check_call([sys.executable, os.path.join(os.path.abspath("."), "get_tools.py"), tools_dir])
                    try:
                        import get_tools
                        get_tools.install_piper_voice(tools_dir, vid)
                        return True
                    except Exception:
                        return False

            def _done(ok):
                if ok:
                    QtWidgets.QMessageBox.information(self, "Piper voice", f"Voix installée: {vid}")
                else:
                    QtWidgets.QMessageBox.warning(self, "Piper voice", f"Installation échouée: {vid}\nVérifie ta connexion / droits d'écriture.")
            self._run_worker(_job, f"Voice install done: {vid}", on_done=_done)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Piper voice", str(e))

    def _audio_generate_vfx_selected(self):
        self._audio_apply_globals()
        self._audio_save_scene_fields()
        idx = self._audio_current_idx
        if idx is None or idx < 0:
            return
        scene_idx = int(idx)
        s = self.cfg.scenes[scene_idx]
        if not (getattr(s, 'vfx_prompt', '') or '').strip():
            QtWidgets.QMessageBox.information(self, "Audio", "No VFX prompt for this shot.")
            return

        project_out = self.cfg.output_dir

        def _job():
            path = generate_scene_vfx_audio(self.cfg, scene_idx, project_out)
            if path:
                s.vfx_audio_path = path
            return path

        def _done(path):
            try:
                if path:
                    self.le_vfx.setText(path)
            except Exception:
                pass

        self._run_worker(_job, "VFX IA generated for selected shot.", on_done=_done)


    def _audio_select_scene_index(self, idx: int, *, focus_audio_tab: bool = False):
        """Select a shot in the Audio tab programmatically.

        This is used to keep the Audio UI in sync with the timeline/inspector.
        We intentionally avoid auto-saving fields when the selection is driven
        by code (not the user).
        """
        try:
            if idx is None:
                return
            if not hasattr(self, "audio_scene_list"):
                return
            if not self.cfg.scenes:
                return
            i = max(0, min(int(idx), len(self.cfg.scenes) - 1))
            self._audio_loading = True
            try:
                self.audio_scene_list.setCurrentRow(i)
            finally:
                self._audio_loading = False
            # Ensure fields are loaded even if signals were blocked.
            try:
                self._audio_on_scene_selected(i)
            except Exception:
                pass
            if focus_audio_tab:
                try:
                    self.tabs.setCurrentWidget(self.tab_audio)
                except Exception:
                    pass
        except Exception:
            pass

    def _maybe_auto_audio_after_storyboard_apply(self):
        """Optional automation: after AI storyboard apply, generate TTS and/or build the master mix."""
        try:
            a = self.cfg.audio
            if not bool(getattr(a, "enabled", True)):
                return
            if not bool(getattr(a, "auto_generate_on_storyboard_apply", False)):
                return
        except Exception:
            return

        # Apply latest UI globals if the tab exists.
        try:
            self._audio_apply_globals()
        except Exception:
            pass

        # Save any current per-shot audio edits before generating.
        try:
            self._audio_save_scene_fields()
        except Exception:
            pass

        project_out = self.cfg.output_dir

        def _job():
            # 1) generate all dialogue audio
            for i, s in enumerate(self.cfg.scenes):
                if (getattr(s, "dialogue_text", "") or "").strip():
                    s.dialogue_audio_path = generate_scene_dialogue_audio(self.cfg, i, project_out)
            # 2) optionally build tracks + master
            if bool(getattr(self.cfg.audio, "auto_build_mix_on_storyboard_apply", False)):
                info = build_all_audio(self.cfg, project_out)
                return info
            return {"enabled": True}

        def _done(info):
            # If we extended shot durations, update timeline + scene table.
            try:
                if hasattr(self, "premiere") and self.premiere:
                    self.premiere.reload_from_cfg(keep_selection=True)
            except Exception:
                pass
            try:
                self._reload_scene_table()
            except Exception:
                pass
            try:
                self._refresh_audio_scene_list(keep_selection=True)
            except Exception:
                pass

            # User-friendly status.
            try:
                master = None
                if isinstance(info, dict):
                    master = info.get("master_track")
                if master:
                    QtWidgets.QMessageBox.information(self, "Audio", f"Dialogue + Master mix built:\n{master}")
                else:
                    QtWidgets.QMessageBox.information(self, "Audio", "Dialogue audio generated from storyboard.")
            except Exception:
                pass

        self._run_worker(_job, "Audio auto-build done.", on_done=_done)

    # ---------------- Model tab (adds EXR/video extraction) ----------------
    def _build_model_tab(self):
        container = self._scrollify_tab(self.tab_model)
        lay = QtWidgets.QFormLayout(container)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setVerticalSpacing(10)

        self.model_id = QtWidgets.QComboBox()
        self.model_id.setEditable(True)
        self.model_id.addItems(MODEL_PRESETS)
        self.model_id.setCurrentText(self.cfg.model_id)
        lay.addRow("Model ID", self.model_id)
        # Download / cache model snapshot (HuggingFace)
        row_dl = QtWidgets.QHBoxLayout()
        self.btn_model_dl = QtWidgets.QPushButton("Download / Update model")
        self.btn_model_check = QtWidgets.QPushButton("Verify cache")
        row_dl.addWidget(self.btn_model_dl)
        row_dl.addWidget(self.btn_model_check)
        row_dl.addStretch(1)
        wrow = QtWidgets.QWidget()
        wrow.setLayout(row_dl)
        lay.addRow("Cache", wrow)

        self.model_dl_prog = QtWidgets.QProgressBar()
        self.model_dl_prog.setRange(0, 1)  # idle
        self.model_dl_status = QtWidgets.QLabel("Model cache: idle")
        self.model_dl_status.setWordWrap(True)
        lay.addRow(self.model_dl_prog)
        lay.addRow(self.model_dl_status)

        self.btn_model_dl.clicked.connect(self._download_selected_model)
        self.btn_model_check.clicked.connect(self._verify_selected_model_cache)

        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["I2V", "T2V"])
        self.mode.setCurrentText(self.cfg.mode)
        if hasattr(self, 'backend'):
            self.backend.setCurrentText(getattr(self.cfg, 'backend', 'wan') or 'wan')
        lay.addRow("Mode", self.mode)

        self.backend = QtWidgets.QComboBox()
        self.backend.addItems(['wan', 'cogvideox', 'ltx', 'ltx2', 'lingbot'])
        self.backend.setCurrentText(getattr(self.cfg, 'backend', 'wan') or 'wan')
        self.backend.setToolTip('Choisit le backend vidéo par défaut. Chaque clip peut override dans Studio (Premiere).')
        lay.addRow('Backend', self.backend)

        self.input_img = QtWidgets.QLineEdit(self.cfg.input_image_path or "")
        btn_img = QtWidgets.QPushButton("Image…")
        btn_exr = QtWidgets.QPushButton("EXR…")
        btn_vid = QtWidgets.QPushButton("Frame vidéo…")
        btn_ai = QtWidgets.QPushButton("AI Gen…")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.input_img, 1)
        row.addWidget(btn_img)
        row.addWidget(btn_exr)
        row.addWidget(btn_vid)
        row.addWidget(btn_ai)
        lay.addRow("Image init (I2V)", row)
        btn_img.clicked.connect(self._choose_input_image)
        btn_exr.clicked.connect(self._import_exr_modeltab)
        btn_vid.clicked.connect(self._extract_frame_modeltab)
        btn_ai.clicked.connect(self._open_ai_image_global)

        # Global character reference (fallback for I2V)
        self.char_img = QtWidgets.QLineEdit(getattr(self.cfg, "character_image_path", "") or "")
        btn_char = QtWidgets.QPushButton("Character…")
        rowc = QtWidgets.QHBoxLayout()
        rowc.addWidget(self.char_img, 1)
        rowc.addWidget(btn_char)
        lay.addRow("Character ref (I2V)", rowc)
        btn_char.clicked.connect(self._choose_character_image)

        self.model_hint = QtWidgets.QLabel(
            "I2V = <b>image → vidéo</b> (image init obligatoire). "
            "T2V = <b>texte → vidéo</b> (pas d'image init). "
            "Pour longform: Timeline = plusieurs scènes, l'app génère des segments puis les assemble."
        )
        self.model_hint.setWordWrap(True)
        lay.addRow(self.model_hint)

        # Global style pack
        self.global_style = QtWidgets.QComboBox()
        self.global_style.addItem("(none)")
        for name in list_packs():
            self.global_style.addItem(name)
        cur_style = (getattr(self.cfg, "global_style_pack", "") or "").strip()
        self.global_style.setCurrentText(cur_style if cur_style else "(none)")
        lay.addRow("Global style pack", self.global_style)

        # Quality preset
        self.quality = QtWidgets.QComboBox()
        self.quality.addItem("Custom")
        self.quality.addItem("LingBot Cine A (4090/24GB)")
        self.quality.addItem("Draft super fast")
        self.quality.addItem("Draft Preview (fast)")
        self.quality.addItem("LookDev (balanced)")
        self.quality.addItem("Final (base+upscale)")
        self.quality.addItem("Cinema Ultra (photoreal max)")
        self.quality.addItem("Cinema Final+ (deflicker/denoise)")
        self.quality.addItem("Cinema 4K Master (2-stage)")
        btn_q = QtWidgets.QPushButton("Apply")
        rowq = QtWidgets.QHBoxLayout()
        rowq.addWidget(self.quality, 1)
        rowq.addWidget(btn_q)
        lay.addRow("Quality preset", rowq)
        btn_q.clicked.connect(self._apply_quality_preset)

        self.fps = QtWidgets.QSpinBox(); self.fps.setRange(1, 60); self.fps.setValue(self.cfg.fps)
        lay.addRow("FPS", self.fps)

        self.w = QtWidgets.QSpinBox(); self.w.setRange(128, 4096); self.w.setValue(self.cfg.width)
        self.h = QtWidgets.QSpinBox(); self.h.setRange(128, 4096); self.h.setValue(self.cfg.height)
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("W")); row2.addWidget(self.w)
        row2.addWidget(QtWidgets.QLabel("H")); row2.addWidget(self.h)
        lay.addRow("Résolution (base)", row2)

        self.steps = QtWidgets.QSpinBox(); self.steps.setRange(1, 80); self.steps.setValue(self.cfg.num_inference_steps)
        lay.addRow("Steps (global)", self.steps)

        self.cfg1 = QtWidgets.QDoubleSpinBox(); self.cfg1.setRange(0, 20); self.cfg1.setDecimals(2); self.cfg1.setValue(self.cfg.guidance_scale)
        self.cfg2 = QtWidgets.QDoubleSpinBox(); self.cfg2.setRange(0, 20); self.cfg2.setDecimals(2); self.cfg2.setValue(self.cfg.guidance_scale_2)
        r = QtWidgets.QHBoxLayout()
        r.addWidget(QtWidgets.QLabel("CFG1")); r.addWidget(self.cfg1)
        r.addWidget(QtWidgets.QLabel("CFG2")); r.addWidget(self.cfg2)
        lay.addRow("Guidance (global)", r)

        self.boundary = QtWidgets.QDoubleSpinBox(); self.boundary.setRange(0, 1); self.boundary.setDecimals(2); self.boundary.setSingleStep(0.01); self.boundary.setValue(self.cfg.boundary_ratio)
        lay.addRow("Boundary ratio (global)", self.boundary)

        self.chunk_s = QtWidgets.QDoubleSpinBox(); self.chunk_s.setRange(1, 20); self.chunk_s.setDecimals(2); self.chunk_s.setValue(self.cfg.chunk_seconds)
        lay.addRow("Chunk seconds", self.chunk_s)

        # Ciné stable: force 0 overlap + cut stitching (prevents flow/blend artefacts)
        # --- Cinema SAFE mode ---
        self.cb_cinema_safe = QtWidgets.QCheckBox("🛡️ SAFE DEMO MODE (ciné) — zéro crash / anti-OOM")
        self.cb_cinema_safe.setChecked(bool(getattr(self.cfg, 'cinema_safe_mode', True)))
        self.cb_cinema_safe.setToolTip("Force des réglages stables: cut/no-overlap, UniPC OFF (Wan), offload séquentiel ON, retries OOM. Plus lent mais très robuste.")
        self.cb_cinema_safe.toggled.connect(self._update_cinema_safe_ui)
        lay.addRow(self.cb_cinema_safe)
        self.cb_no_overlap_strict = QtWidgets.QCheckBox("No-overlap strict (ciné stable: pas de blend entre segments)")
        self.cb_no_overlap_strict.setChecked(bool(getattr(self.cfg, 'no_overlap_strict', True)))
        try:
            self.cb_cinema_safe.setChecked(bool(getattr(self.cfg, 'cinema_safe_mode', True)))
        except Exception:
            pass
        self.cb_no_overlap_strict.setToolTip("Quand activé: overlap=0 et blend='cut' sont forcés. \nUtile si tu vois des halos/warp/glitches aux boundaries de segments.")
        lay.addRow(self.cb_no_overlap_strict)

        self.overlap = QtWidgets.QSpinBox(); self.overlap.setRange(0, 256); self.overlap.setValue(self.cfg.overlap_frames)
        lay.addRow("Overlap frames (segments)", self.overlap)

        self.blend = QtWidgets.QComboBox(); self.blend.addItems(["cut","flow","crossfade"]); self.blend.setCurrentText(self.cfg.blend_mode)
        lay.addRow("Blend mode (segments)", self.blend)

        # Apply initial UI state + react to user toggles
        try:
            self.cb_no_overlap_strict.toggled.connect(self._update_no_overlap_ui)
            self._update_cinema_safe_ui()
        except Exception:
            pass



        # --- Scheduler (advanced) ---
        self.cb_unipc = QtWidgets.QCheckBox("Use UniPC scheduler (moins d'artefacts sur certains modèles)")
        self.cb_unipc.setChecked(bool(getattr(self.cfg, 'use_unipc', False)))
        lay.addRow(self.cb_unipc)

        self.sp_flow_shift = QtWidgets.QDoubleSpinBox(); self.sp_flow_shift.setRange(0.0, 20.0); self.sp_flow_shift.setDecimals(2)
        self.sp_flow_shift.setValue(float(getattr(self.cfg, 'flow_shift', 5.0) or 0.0))
        self.sp_flow_shift.setToolTip("Paramètre UniPC 'flow_shift' (souvent 0–8). N'agit que si UniPC activé.")
        lay.addRow("UniPC flow_shift", self.sp_flow_shift)

        self.offload = QtWidgets.QCheckBox(); self.offload.setChecked(self.cfg.enable_model_cpu_offload)
        lay.addRow("CPU offload", self.offload)

        self.seq_offload = QtWidgets.QCheckBox(); self.seq_offload.setChecked(getattr(self.cfg, "enable_sequential_cpu_offload", False))
        lay.addRow("Sequential CPU offload (lowest VRAM, slower)", self.seq_offload)

        self.fp32vae = QtWidgets.QCheckBox(); self.fp32vae.setChecked(self.cfg.vae_decode_fp32)
        lay.addRow("VAE decode FP32", self.fp32vae)

        self.keep_frames = QtWidgets.QCheckBox(); self.keep_frames.setChecked(self.cfg.keep_png_frames)
        lay.addRow("Garder frames PNG", self.keep_frames)

        # --- Init-frame global settings (Flux2/SDXL) ---
        gb_init = QtWidgets.QGroupBox("Init-frame auto (Flux2/SDXL)")
        gl_init = QtWidgets.QFormLayout(gb_init)

        # Ensure cfg.init_image exists
        if getattr(self.cfg, 'init_image', None) is None:
            try:
                self.cfg.init_image = InitImageSpec()
            except Exception:
                pass
        init = getattr(self.cfg, 'init_image', None)

        self.cb_init_enabled = QtWidgets.QCheckBox("Enable auto init-frame (quand I2V sans image)")
        self.cb_init_enabled.setChecked(bool(getattr(init, 'enabled', True) if init else True))

        self.cb_init_preset = QtWidgets.QComboBox(); self.cb_init_preset.setEditable(True)
        self.cb_init_preset.addItems([
            'zimage_turbo', 'zimage',
            'flux2_bnb4bit', 'flux2',
            'sdxl_lightning_4step', 'sdxl_base', 'sdxl_turbo',
            'flux1_schnell'
        ])
        self.cb_init_preset.setCurrentText(str(getattr(init, 'preset', 'zimage_turbo') if init else 'zimage_turbo'))

        self.cb_init_policy = QtWidgets.QComboBox()
        self.cb_init_policy.addItem('necessary', 'necessary')
        self.cb_init_policy.addItem('always_refine', 'always_refine')
        cur_pol = str(getattr(init, 'policy', 'necessary') if init else 'necessary')
        for i in range(self.cb_init_policy.count()):
            if str(self.cb_init_policy.itemData(i)) == cur_pol:
                self.cb_init_policy.setCurrentIndex(i)
                break

        self.cb_init_cache_loc = QtWidgets.QComboBox()
        self.cb_init_cache_loc.addItem('assets (dans le projet)', 'assets')
        self.cb_init_cache_loc.addItem('project (.wan_cache)', 'project')
        cur_loc = str(getattr(init, 'cache_location', 'assets') if init else 'assets')
        for i in range(self.cb_init_cache_loc.count()):
            if str(self.cb_init_cache_loc.itemData(i)) == cur_loc:
                self.cb_init_cache_loc.setCurrentIndex(i)
                break

        self.le_init_cache_dirname = QtWidgets.QLineEdit(str(getattr(init, 'project_cache_dirname', '.wan_cache') if init else '.wan_cache'))
        self.le_init_cache_dirname.setToolTip("Utilisé uniquement si cache_location=project")

        base_w = int((getattr(init, 'width', 0) if init else 0) or getattr(self.cfg, 'width', 1024) or 1024)
        base_h = int((getattr(init, 'height', 0) if init else 0) or getattr(self.cfg, 'height', 1024) or 1024)
        self.init_w = QtWidgets.QSpinBox(); self.init_w.setRange(0, 4096); self.init_w.setValue(base_w)
        self.init_h = QtWidgets.QSpinBox(); self.init_h.setRange(0, 4096); self.init_h.setValue(base_h)
        self.init_w.setToolTip("0 = utiliser la résolution projet")
        self.init_h.setToolTip("0 = utiliser la résolution projet")
        row_wh = QtWidgets.QHBoxLayout(); row_wh.addWidget(QtWidgets.QLabel('W')); row_wh.addWidget(self.init_w); row_wh.addWidget(QtWidgets.QLabel('H')); row_wh.addWidget(self.init_h)
        wrow_wh = QtWidgets.QWidget(); wrow_wh.setLayout(row_wh)

        self.init_steps = QtWidgets.QSpinBox(); self.init_steps.setRange(1, 120); self.init_steps.setValue(int(getattr(init, 'steps', 28) if init else 28))
        self.init_gs = QtWidgets.QDoubleSpinBox(); self.init_gs.setRange(0.0, 20.0); self.init_gs.setDecimals(2); self.init_gs.setValue(float(getattr(init, 'guidance_scale', 4.5) if init else 4.5))

        self.cb_init_use_last_loc = QtWidgets.QCheckBox('Use last location memory (continuité décor)')
        self.cb_init_use_last_loc.setChecked(bool(getattr(init, 'use_last_location_memory', False) if init else False))

        self.sp_init_max_refs = QtWidgets.QSpinBox(); self.sp_init_max_refs.setRange(0, 32); self.sp_init_max_refs.setValue(int(getattr(init, 'max_refs', 6) if init else 6))

        # Optional IP-Adapter / ControlNet (best-effort)
        self.cb_init_ip = QtWidgets.QCheckBox('Enable IP-Adapter (best-effort)')
        self.cb_init_ip.setChecked(bool(getattr(init, 'enable_ip_adapter', False) if init else False))
        self.le_ip_model = QtWidgets.QLineEdit(str(getattr(init, 'ip_adapter_model_id', '') if init else ''))
        self.le_ip_sub = QtWidgets.QLineEdit(str(getattr(init, 'ip_adapter_subfolder', '') if init else ''))
        self.le_ip_weight = QtWidgets.QLineEdit(str(getattr(init, 'ip_adapter_weight_name', '') if init else ''))
        self.sp_ip_char = QtWidgets.QDoubleSpinBox(); self.sp_ip_char.setRange(0.0, 2.0); self.sp_ip_char.setDecimals(2); self.sp_ip_char.setValue(float(getattr(init, 'ip_adapter_scale_character', 0.8) if init else 0.8))
        self.sp_ip_loc = QtWidgets.QDoubleSpinBox(); self.sp_ip_loc.setRange(0.0, 2.0); self.sp_ip_loc.setDecimals(2); self.sp_ip_loc.setValue(float(getattr(init, 'ip_adapter_scale_location', 0.6) if init else 0.6))
        self.sp_ip_style = QtWidgets.QDoubleSpinBox(); self.sp_ip_style.setRange(0.0, 2.0); self.sp_ip_style.setDecimals(2); self.sp_ip_style.setValue(float(getattr(init, 'ip_adapter_scale_style', 0.5) if init else 0.5))

        self.cb_init_cn = QtWidgets.QCheckBox('Enable ControlNet (best-effort)')
        self.cb_init_cn.setChecked(bool(getattr(init, 'enable_controlnet', False) if init else False))
        self.cb_cn_type = QtWidgets.QComboBox(); self.cb_cn_type.addItems(['canny', 'depth', 'pose'])
        self.cb_cn_type.setCurrentText(str(getattr(init, 'controlnet_type', 'canny') if init else 'canny'))
        self.le_cn_model = QtWidgets.QLineEdit(str(getattr(init, 'controlnet_model_id', '') if init else ''))
        self.sp_cn_scale = QtWidgets.QDoubleSpinBox(); self.sp_cn_scale.setRange(0.0, 2.0); self.sp_cn_scale.setDecimals(2); self.sp_cn_scale.setValue(float(getattr(init, 'controlnet_scale', 0.75) if init else 0.75))
        self.sp_refine = QtWidgets.QDoubleSpinBox(); self.sp_refine.setRange(0.0, 1.0); self.sp_refine.setDecimals(2); self.sp_refine.setValue(float(getattr(init, 'refine_strength', 0.35) if init else 0.35))

        # Optional model overrides (avoid hardcoded model IDs)
        self.le_init_flux2_bnb4bit = QtWidgets.QLineEdit(str(getattr(init, 'flux2_bnb4bit_model_id', '') if init else ''))
        self.le_init_flux2_full = QtWidgets.QLineEdit(str(getattr(init, 'flux2_model_id', '') if init else ''))
        self.le_init_zimage_turbo = QtWidgets.QLineEdit(str(getattr(init, 'zimage_turbo_model_id', '') if init else ''))
        self.le_init_zimage = QtWidgets.QLineEdit(str(getattr(init, 'zimage_model_id', '') if init else ''))
        self.le_init_sdxl_base = QtWidgets.QLineEdit(str(getattr(init, 'sdxl_base_model_id', '') if init else ''))
        self.le_init_sdxl_turbo = QtWidgets.QLineEdit(str(getattr(init, 'sdxl_turbo_model_id', '') if init else ''))
        self.le_init_sdxl_lora = QtWidgets.QLineEdit(str(getattr(init, 'sdxl_lightning_lora_id', '') if init else ''))
        self.le_init_sdxl_lora_file = QtWidgets.QLineEdit(str(getattr(init, 'sdxl_lightning_lora_file', '') if init else ''))
        self.le_init_flux1 = QtWidgets.QLineEdit(str(getattr(init, 'flux1_schnell_model_id', '') if init else ''))

        gl_init.addRow(self.cb_init_enabled)
        gl_init.addRow('Preset', self.cb_init_preset)
        gl_init.addRow('Policy', self.cb_init_policy)
        gl_init.addRow('Cache location', self.cb_init_cache_loc)
        gl_init.addRow('Project cache dirname', self.le_init_cache_dirname)
        gl_init.addRow('Init size', wrow_wh)
        gl_init.addRow('Init steps', self.init_steps)
        gl_init.addRow('Init guidance', self.init_gs)
        gl_init.addRow(self.cb_init_use_last_loc)
        gl_init.addRow('Max refs', self.sp_init_max_refs)

        # Advanced toggles
        gl_init.addRow(self.cb_init_ip)
        gl_init.addRow('IP model id', self.le_ip_model)
        gl_init.addRow('IP subfolder', self.le_ip_sub)
        gl_init.addRow('IP weight name', self.le_ip_weight)
        row_ip = QtWidgets.QHBoxLayout(); row_ip.addWidget(QtWidgets.QLabel('char')); row_ip.addWidget(self.sp_ip_char); row_ip.addWidget(QtWidgets.QLabel('loc')); row_ip.addWidget(self.sp_ip_loc); row_ip.addWidget(QtWidgets.QLabel('style')); row_ip.addWidget(self.sp_ip_style)
        wrow_ip = QtWidgets.QWidget(); wrow_ip.setLayout(row_ip)
        gl_init.addRow('IP scales', wrow_ip)

        gl_init.addRow(self.cb_init_cn)
        gl_init.addRow('ControlNet type', self.cb_cn_type)
        gl_init.addRow('ControlNet model id', self.le_cn_model)
        gl_init.addRow('ControlNet scale', self.sp_cn_scale)
        gl_init.addRow('Refine strength', self.sp_refine)

        gl_init.addRow(QtWidgets.QLabel("<b>Model overrides (optional)</b>"))
        gl_init.addRow('Flux2 bnb4bit model id', self.le_init_flux2_bnb4bit)
        gl_init.addRow('Flux2 full model id', self.le_init_flux2_full)
        gl_init.addRow('Z-Image Turbo model id', self.le_init_zimage_turbo)
        gl_init.addRow('Z-Image model id', self.le_init_zimage)
        gl_init.addRow('SDXL base model id', self.le_init_sdxl_base)
        gl_init.addRow('SDXL turbo model id', self.le_init_sdxl_turbo)
        gl_init.addRow('SDXL Lightning LoRA repo', self.le_init_sdxl_lora)
        gl_init.addRow('SDXL Lightning LoRA file', self.le_init_sdxl_lora_file)
        gl_init.addRow('Flux1 schnell model id', self.le_init_flux1)

        lay.addRow(gb_init)

        # --- OOM / VRAM resilience (engine) ---
        gb_oom = QtWidgets.QGroupBox('VRAM / OOM resilience')
        gl_oom = QtWidgets.QFormLayout(gb_oom)
        self.cb_auto_vram = QtWidgets.QCheckBox('Auto VRAM optimizations (reduce frames-per-call on OOM)')
        self.cb_auto_vram.setChecked(bool(getattr(self.cfg, 'auto_vram_optimizations', True)))
        self.sp_max_frames_seg = QtWidgets.QSpinBox(); self.sp_max_frames_seg.setRange(0, 256); self.sp_max_frames_seg.setValue(int(getattr(self.cfg, 'max_frames_per_segment', 0) or 0))
        self.sp_oom_retry = QtWidgets.QSpinBox(); self.sp_oom_retry.setRange(0, 10); self.sp_oom_retry.setValue(int(getattr(self.cfg, 'oom_retry_max_attempts', 2) or 2))
        gl_oom.addRow(self.cb_auto_vram)
        gl_oom.addRow('Max frames/segment (0=auto)', self.sp_max_frames_seg)
        gl_oom.addRow('OOM retry attempts', self.sp_oom_retry)
        lay.addRow(gb_oom)


    def _import_exr_modeltab(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choisir EXR", ".", "EXR (*.exr)")
        if not f:
            return
        d = ExrImportDialog(self)
        if d.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        out_dir = make_inputs_dir(".")
        out_png = unique_path(out_dir, "exr", "png")
        res = exr_to_png(f, out_png, exposure_ev=float(d.exposure.value()), gamma=float(d.gamma.value()), tonemap=str(d.tonemap.currentText()))
        if not res.ok:
            QtWidgets.QMessageBox.critical(self, "EXR import failed", res.message or "Unknown error")
            return
        self.input_img.setText(res.path)
        self.append_log(f"EXR imported: {res.path} ({res.message})")

    def _extract_frame_modeltab(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choisir vidéo", ".", "Vidéos (*.mp4 *.mov *.mkv *.webm *.avi)")
        if not f:
            return
        t, ok = QtWidgets.QInputDialog.getDouble(self, "Timestamp", "Extraire frame à t (secondes):", 0.0, 0.0, 10_000.0, 2)
        if not ok:
            return
        out_dir = make_inputs_dir(".")
        out_png = unique_path(out_dir, "frame", "png")
        res = extract_frame_ffmpeg(f, float(t), out_png)
        if not res.ok:
            QtWidgets.QMessageBox.critical(self, "Frame extract failed", res.message or "Unknown error")
            return
        self.input_img.setText(res.path)
        self.append_log(f"Frame extracted: {res.path}")

    def _open_ai_image_global(self):
        """Generate a global init image (I2V) with a text-to-image model."""
        try:
            from .ai_image_dialog import AiImageDialog
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "AI Image", f"Missing AI image dialog: {e}")
            return

        default_prompt = ""
        try:
            if self.cfg.scenes:
                default_prompt = (self.cfg.scenes[0].prompt or "").strip()
        except Exception:
            pass

        d = AiImageDialog(
            self,
            default_model=os.environ.get("WAN_STUDIO_T2I_MODEL", "black-forest-labs/FLUX.1-schnell"),
            default_prompt=default_prompt,
            default_negative=(self.cfg.negative_prompt or "").strip(),
            default_w=int(self.cfg.width or 1024),
            default_h=int(self.cfg.height or 1024),
            default_steps=int(self.cfg.num_inference_steps or 20),
            default_guidance=float(self.cfg.guidance_scale or 4.0),
        )
        if d.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        req = d.request()
        if not req:
            return

        out_dir = make_inputs_dir(".")
        out_png = unique_path(out_dir, "ai_init", "png")

        def job():
            from .ai_image import generate_image
            return generate_image(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                model_id=req.model_id,
                out_path=out_png,
                width=req.width,
                height=req.height,
                steps=req.steps,
                guidance=req.guidance,
                seed=req.seed,
                dtype=req.dtype,
                enable_cpu_offload=True,
            )

        def on_done(res):
            try:
                if getattr(res, 'ok', False):
                    self.cfg.input_image_path = res.path
                    self.input_img.setText(res.path or "")
                    if hasattr(self, 'q_input'):
                        self.q_input.setText(res.path or "")
                    self.append_log(f"AI init image generated: {res.path}")
                else:
                    QtWidgets.QMessageBox.critical(self, "AI Image", getattr(res, 'message', 'Unknown error'))
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "AI Image", str(e))

        self._run_generic_worker(job, title="AI Image", on_done=on_done)

    def open_ai_image_for_scene(self, scene):
        """Generate an init image and apply it as per-clip override."""
        if scene is None:
            return
        try:
            from .ai_image_dialog import AiImageDialog
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "AI Image", f"Missing AI image dialog: {e}")
            return
        default_prompt = (getattr(scene, 'prompt', '') or '').strip()
        d = AiImageDialog(
            self,
            default_model=os.environ.get("WAN_STUDIO_T2I_MODEL", "black-forest-labs/FLUX.1-schnell"),
            default_prompt=default_prompt,
            default_negative=(self.cfg.negative_prompt or "").strip(),
            default_w=int(self.cfg.width or 1024),
            default_h=int(self.cfg.height or 1024),
            default_steps=int(self.cfg.num_inference_steps or 20),
            default_guidance=float(self.cfg.guidance_scale or 4.0),
        )
        if d.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        req = d.request()
        if not req:
            return

        out_dir = make_inputs_dir(".")
        out_png = unique_path(out_dir, f"ai_{(getattr(scene,'label','shot') or 'shot')}", "png")

        def job():
            from .ai_image import generate_image
            return generate_image(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                model_id=req.model_id,
                out_path=out_png,
                width=req.width,
                height=req.height,
                steps=req.steps,
                guidance=req.guidance,
                seed=req.seed,
                dtype=req.dtype,
                enable_cpu_offload=True,
            )

        def on_done(res):
            if not getattr(res, 'ok', False):
                QtWidgets.QMessageBox.critical(self, "AI Image", getattr(res, 'message', 'Unknown error'))
                return
            try:
                scene.input_image_path_override = res.path
                scene.mode_override = "I2V"
                self.append_log(f"AI init image set for {getattr(scene,'label','shot')}: {res.path}")
                # Refresh timeline + table
                try:
                    self._reload_scene_table()
                except Exception:
                    pass
                try:
                    if hasattr(self, 'premiere') and self.premiere:
                        self.premiere.reload_from_cfg(keep_selection=True)
                except Exception:
                    pass
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "AI Image", str(e))

        self._run_generic_worker(job, title="AI Image", on_done=on_done)

    def _run_generic_worker(self, fn, title: str, on_done=None):
        """Run a function in a QThread; show errors in a dialog."""
        thread = QtCore.QThread(self)
        worker = _FuncWorker(fn)
        worker.moveToThread(thread)

        # Keep strong refs so Python GC can't destroy QThread/worker early.
        # (Fixes: "QThread: Destroyed while thread is still running")
        if not hasattr(self, "_bg_tasks"):
            self._bg_tasks = []
        self._bg_tasks.append((thread, worker))

        box = {"res": None, "tb": None}

        def _cleanup():
            # Ensure Qt objects get released
            try:
                worker.deleteLater()
            except Exception:
                pass
            try:
                thread.deleteLater()
            except Exception:
                pass
            # Best-effort memory cleanup after heavy jobs
            try:
                import gc
                gc.collect()
            except Exception:
                pass
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        def _ok(res):
            # NOTE: Python callables connected to signals may run in the emitter thread.
            # We must never call QThread.wait() from within the thread itself.
            box["res"] = res
            thread.quit()

        def _err(tb):
            box["tb"] = tb
            thread.quit()

        def _after_finished():
            # Called once the thread fully stopped.
            tb = box.get("tb")
            res = box.get("res")
            _cleanup()
            # Remove strong refs
            try:
                self._bg_tasks.remove((thread, worker))
            except Exception:
                pass
            if tb:
                print(tb)
                QtWidgets.QMessageBox.critical(self, title, tb)
            elif callable(on_done):
                on_done(res)

        # Force queued connections so callbacks always run on the GUI thread.
        worker.done.connect(_ok, QtCore.Qt.QueuedConnection)
        worker.error.connect(_err, QtCore.Qt.QueuedConnection)
        thread.started.connect(worker.run)
        thread.finished.connect(_after_finished, QtCore.Qt.QueuedConnection)
        thread.start()

    # ---------------- LoRA tab ----------------

    # ---------------- Model cache / download (HuggingFace) ----------------
    def _download_selected_model(self):
        repo_id = self.model_id.currentText().strip()
        if not repo_id:
            QtWidgets.QMessageBox.warning(self, "Model", "Model ID vide.")
            return
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN") or None
        try:
            from .init_frame_subprocess import resolve_hf_cache_dir
            cache_dir = resolve_hf_cache_dir(self.cfg)
        except Exception:
            cache_dir = os.environ.get("HF_HOME") or None
        repo_id = self._normalize_hf_repo_id(repo_id)

        self.model_dl_prog.setRange(0, 0)  # busy
        self.model_dl_status.setText(f"Downloading: {repo_id} …")
        self.append_log(f"Model download requested: {repo_id}")
        self._start_model_download(repo_id, token=token, cache_dir=cache_dir)

    def _verify_selected_model_cache(self):
        repo_id = self.model_id.currentText().strip()
        if not repo_id:
            QtWidgets.QMessageBox.warning(self, "Model", "Model ID vide.")
            return
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN") or None
        try:
            from .init_frame_subprocess import resolve_hf_cache_dir
            cache_dir = resolve_hf_cache_dir(self.cfg)
        except Exception:
            cache_dir = os.environ.get("HF_HOME") or None
        repo_id = self._normalize_hf_repo_id(repo_id)
        try:
            from huggingface_hub import snapshot_download
            local_dir = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                token=token,
                local_files_only=True,
            )
            self.model_dl_status.setText(f"✅ Cached: {local_dir}")
            self.append_log(f"Model cache OK: {repo_id} -> {local_dir}")
        except Exception as e:
            self.model_dl_status.setText(f"❌ Not cached: {e}")
            self.append_log(f"Model cache missing: {repo_id} ({e})")

    def _start_model_download(self, repo_id: str, revision: str | None = None, token: str | None = None, cache_dir: str | None = None):
        if self.dl_thread and self.dl_thread.isRunning():
            self.append_log("Model download already running.")
            return

        self.dl_thread = QtCore.QThread(self)
        self.dl_worker = ModelDownloadWorker(repo_id=repo_id, revision=revision, cache_dir=cache_dir, token=token)
        self.dl_worker.moveToThread(self.dl_thread)

        self.dl_thread.started.connect(self.dl_worker.run)
        self.dl_worker.log.connect(self.append_log)
        self.dl_worker.finished.connect(self._on_model_download_done)
        self.dl_worker.failed.connect(self._on_model_download_failed)

        self.dl_worker.finished.connect(self.dl_thread.quit)
        self.dl_worker.failed.connect(self.dl_thread.quit)
        self.dl_thread.finished.connect(self._on_model_download_thread_finished)

        self.dl_thread.start()

    def _on_model_download_done(self, local_dir: str):
        self.model_dl_prog.setRange(0, 1)
        self.model_dl_prog.setValue(1)
        self.model_dl_status.setText(f"✅ Downloaded/cached at: {local_dir}")
        self.append_log(f"Model download finished: {local_dir}")

    def _on_model_download_failed(self, msg: str):
        self.model_dl_prog.setRange(0, 1)
        self.model_dl_prog.setValue(0)
        self.model_dl_status.setText(f"❌ Download failed: {msg}")
        self.append_log(f"Model download failed: {msg}")

    def _on_model_download_thread_finished(self):
        self.dl_worker = None
        self.dl_thread = None

    def _normalize_hf_repo_id(self, repo_id: str) -> str:
        """Allow HuggingFace URLs; keep repo_id as org/name."""
        rid = (repo_id or "").strip()
        if rid.startswith("http://") or rid.startswith("https://"):
            try:
                from urllib.parse import urlparse
                parts = urlparse(rid)
                path = (parts.path or "").strip("/")
                if path:
                    rid = path
            except Exception:
                pass
        # Trim trailing '/tree/main' or '/resolve/...'
        for token in ("/tree/", "/resolve/"):
            if token in rid:
                rid = rid.split(token, 1)[0].strip("/")
        return rid

    def _build_lora_tab(self):
        container = self._scrollify_tab(self.tab_lora)
        v = QtWidgets.QVBoxLayout(container)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(10)

        info = QtWidgets.QLabel(
        "<b>LoRA / Styles</b> — ajoute des modules (style, identité, caméra) au modèle.<br>"
        "Ordre recommandé: <i>Style</i> → <i>Camera/Motion</i> → <i>Identity</i>. "
        "Poids typiques: 0.3–1.0 (trop haut = artefacts).<br>"
        "Si un LoRA semble inactif, essaye l'option <i>transformer_2</i>."
        )
        info.setWordWrap(True)
        v.addWidget(info)

        top = QtWidgets.QHBoxLayout()
        v.addLayout(top)

        self.pack = QtWidgets.QComboBox()
        self.pack.addItems(["(aucun)"] + list(default_lora_packs().keys()))
        top.addWidget(QtWidgets.QLabel("Pack:"))
        top.addWidget(self.pack, 1)

        btn_apply = QtWidgets.QPushButton("Appliquer")
        top.addWidget(btn_apply)
        btn_apply.clicked.connect(self.on_apply_pack)

        self.lora_table = QtWidgets.QTableWidget()
        self.lora_table.setColumnCount(6)
        self.lora_table.setHorizontalHeaderLabels(["Name","Weight","Repo ID","Weight name","Local path","Into transformer_2"])
        self.lora_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.lora_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.lora_table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        self.lora_table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch)
        v.addWidget(self.lora_table, 1)

        btns = QtWidgets.QHBoxLayout()
        v.addLayout(btns)
        btn_add = QtWidgets.QPushButton("+ Ajouter")
        btn_del = QtWidgets.QPushButton("− Supprimer")
        btns.addWidget(btn_add)
        btns.addWidget(btn_del)
        btns.addStretch(1)

        btn_add.clicked.connect(self.on_add_lora)
        btn_del.clicked.connect(self.on_del_lora)

    # ---------------- Post tab ----------------
    def _build_post_tab(self):
        container = self._scrollify_tab(self.tab_post)
        lay = QtWidgets.QFormLayout(container)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setVerticalSpacing(10)

        self.post_enabled = QtWidgets.QCheckBox(); self.post_enabled.setChecked(self.cfg.post.enabled)
        lay.addRow("Activer post-prod", self.post_enabled)

        self.tools_dir = QtWidgets.QLineEdit(self.cfg.post.tools_dir)
        btn_tools = QtWidgets.QPushButton("…")
        btn_detect = QtWidgets.QPushButton("Detect tools")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.tools_dir, 1)
        row.addWidget(btn_tools)
        row.addWidget(btn_detect)
        lay.addRow("Tools dir", row)
        btn_tools.clicked.connect(self._choose_tools_dir)
        btn_detect.clicked.connect(self.on_detect_tools)

        self.use_rife = QtWidgets.QCheckBox("Use RIFE if available"); self.use_rife.setChecked(self.cfg.post.use_rife_if_available)
        self.use_re = QtWidgets.QCheckBox("Use Real-ESRGAN if available"); self.use_re.setChecked(self.cfg.post.use_realesrgan_if_available)
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(self.use_rife)
        row2.addWidget(self.use_re)
        lay.addRow("IA tools", row2)

        self.target_fps = QtWidgets.QSpinBox(); self.target_fps.setRange(0, 120); self.target_fps.setValue(self.cfg.post.target_fps)
        lay.addRow("Target FPS (0 off)", self.target_fps)

        self.out_w = QtWidgets.QSpinBox(); self.out_w.setRange(0, 8192); self.out_w.setValue(self.cfg.post.out_width)
        self.out_h = QtWidgets.QSpinBox(); self.out_h.setRange(0, 8192); self.out_h.setValue(self.cfg.post.out_height)
        row3 = QtWidgets.QHBoxLayout()
        row3.addWidget(QtWidgets.QLabel("W")); row3.addWidget(self.out_w)
        row3.addWidget(QtWidgets.QLabel("H")); row3.addWidget(self.out_h)
        lay.addRow("Upscale to", row3)

        # Quick target presets for master resolution (Sprint 3)
        self.master_res = QtWidgets.QComboBox()
        self.master_res.addItem("Off (no upscale)", "0x0")
        self.master_res.addItem("1080p (1920x1080)", "1920x1080")
        self.master_res.addItem("1440p (2560x1440)", "2560x1440")
        self.master_res.addItem("2160p / 4K UHD (3840x2160)", "3840x2160")
        self.master_res.addItem("4K DCI (4096x2160)", "4096x2160")
        # Set current if it matches cfg
        try:
            cur = f"{int(self.cfg.post.out_width)}x{int(self.cfg.post.out_height)}"
            for i in range(self.master_res.count()):
                if self.master_res.itemData(i) == cur:
                    self.master_res.setCurrentIndex(i)
                    break
        except Exception:
            pass
        self.master_res.currentIndexChanged.connect(self._on_master_res_changed)
        lay.addRow("Master target", self.master_res)


        self.deflicker = QtWidgets.QCheckBox("Deflicker"); self.deflicker.setChecked(self.cfg.post.deflicker)
        self.denoise = QtWidgets.QCheckBox("Denoise"); self.denoise.setChecked(self.cfg.post.denoise)
        self.sharpen = QtWidgets.QCheckBox("Sharpen"); self.sharpen.setChecked(self.cfg.post.sharpen)
        row4 = QtWidgets.QHBoxLayout()
        row4.addWidget(self.deflicker)
        row4.addWidget(self.denoise)
        row4.addWidget(self.sharpen)
        lay.addRow("Polish", row4)



        # Advanced polish params (fine tuning)
        gb_adv = QtWidgets.QGroupBox("Advanced polish")
        fl = QtWidgets.QFormLayout(gb_adv)

        self.dn_luma = QtWidgets.QDoubleSpinBox(); self.dn_luma.setRange(0.0, 10.0); self.dn_luma.setDecimals(2); self.dn_luma.setSingleStep(0.10)
        self.dn_luma.setValue(float(getattr(self.cfg.post, "denoise_luma", 1.5) or 1.5))

        self.dn_chroma = QtWidgets.QDoubleSpinBox(); self.dn_chroma.setRange(0.0, 10.0); self.dn_chroma.setDecimals(2); self.dn_chroma.setSingleStep(0.10)
        self.dn_chroma.setValue(float(getattr(self.cfg.post, "denoise_chroma", 1.0) or 1.0))

        self.dn_temp = QtWidgets.QDoubleSpinBox(); self.dn_temp.setRange(0.0, 10.0); self.dn_temp.setDecimals(2); self.dn_temp.setSingleStep(0.10)
        self.dn_temp.setValue(float(getattr(self.cfg.post, "denoise_temporal", 3.0) or 3.0))

        self.dn_temp_chroma = QtWidgets.QDoubleSpinBox(); self.dn_temp_chroma.setRange(0.0, 10.0); self.dn_temp_chroma.setDecimals(2); self.dn_temp_chroma.setSingleStep(0.10)
        self.dn_temp_chroma.setValue(float(getattr(self.cfg.post, "denoise_temporal_chroma", 2.0) or 2.0))

        self.unsharp_mx = QtWidgets.QSpinBox(); self.unsharp_mx.setRange(0, 25); self.unsharp_mx.setValue(int(getattr(self.cfg.post, "unsharp_mx", 5) or 5))
        self.unsharp_my = QtWidgets.QSpinBox(); self.unsharp_my.setRange(0, 25); self.unsharp_my.setValue(int(getattr(self.cfg.post, "unsharp_my", 5) or 5))
        self.unsharp_amt = QtWidgets.QDoubleSpinBox(); self.unsharp_amt.setRange(0.0, 3.0); self.unsharp_amt.setDecimals(2); self.unsharp_amt.setSingleStep(0.05)
        self.unsharp_amt.setValue(float(getattr(self.cfg.post, "unsharp_amount", 0.8) or 0.8))

        fl.addRow("Denoise luma", self.dn_luma)
        fl.addRow("Denoise chroma", self.dn_chroma)
        fl.addRow("Denoise temporal", self.dn_temp)
        fl.addRow("Denoise temporal chroma", self.dn_temp_chroma)

        rowu = QtWidgets.QHBoxLayout()
        rowu.addWidget(QtWidgets.QLabel("mx")); rowu.addWidget(self.unsharp_mx)
        rowu.addWidget(QtWidgets.QLabel("my")); rowu.addWidget(self.unsharp_my)
        rowu.addWidget(QtWidgets.QLabel("amount")); rowu.addWidget(self.unsharp_amt)
        wu = QtWidgets.QWidget(); wu.setLayout(rowu)
        fl.addRow("Unsharp", wu)

        lay.addRow(gb_adv)


        self.export_prores = QtWidgets.QCheckBox("Master ProRes 422 HQ"); self.export_prores.setChecked(self.cfg.post.export_master_prores)
        self.export_h265 = QtWidgets.QCheckBox("Delivery H.265 Main10"); self.export_h265.setChecked(self.cfg.post.export_delivery_h265_main10)
        row5 = QtWidgets.QHBoxLayout()
        row5.addWidget(self.export_prores)
        row5.addWidget(self.export_h265)
        lay.addRow("Exports", row5)

        self.primary_deliver = QtWidgets.QComboBox()
        self.primary_deliver.addItems(["ProRes (Master .mov)", "H.265 Main10 (.mp4)", "Intermediate (.mp4)"])
        prim = str(getattr(self.cfg.post, "primary_deliver", "prores")).lower().strip()
        if prim == "h265":
            self.primary_deliver.setCurrentIndex(1)
        elif prim == "intermediate":
            self.primary_deliver.setCurrentIndex(2)
        else:
            self.primary_deliver.setCurrentIndex(0)
        lay.addRow("Primary output", self.primary_deliver)

        self.prores_profile = QtWidgets.QComboBox()
        self.prores_profile.addItems(["422 HQ", "422", "LT", "Proxy", "4444", "4444 XQ"])
        prof = str(getattr(self.cfg.post, "prores_profile", "422hq")).lower().strip()
        prof_map = {"422hq":0,"422":1,"lt":2,"proxy":3,"4444":4,"4444xq":5}
        self.prores_profile.setCurrentIndex(prof_map.get(prof, 0))
        lay.addRow("ProRes profile", self.prores_profile)

        self.h265_crf = QtWidgets.QSpinBox(); self.h265_crf.setRange(10, 30); self.h265_crf.setValue(self.cfg.post.h265_crf)
        self.h265_preset = QtWidgets.QComboBox(); self.h265_preset.addItems(["ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"])
        self.h265_preset.setCurrentText(self.cfg.post.h265_preset)
        row6 = QtWidgets.QHBoxLayout()
        row6.addWidget(QtWidgets.QLabel("CRF")); row6.addWidget(self.h265_crf)
        row6.addWidget(QtWidgets.QLabel("Preset")); row6.addWidget(self.h265_preset)
        lay.addRow("H.265 settings", row6)

        self.tool_status = QtWidgets.QLabel("Tools: (not checked)")
        lay.addRow("Status", self.tool_status)

    # ---------------- Batch tab ----------------
    def _build_batch_tab(self):
        container = self._scrollify_tab(self.tab_batch)
        lay = QtWidgets.QFormLayout(container)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setVerticalSpacing(10)

        self.batch_enabled = QtWidgets.QCheckBox("Enable batch variants (multiple takes)")
        self.batch_enabled.setChecked(self.cfg.batch.enabled)
        lay.addRow(self.batch_enabled)

        self.batch_variants = QtWidgets.QSpinBox()
        self.batch_variants.setRange(1, 32)
        self.batch_variants.setValue(self.cfg.batch.variants)
        lay.addRow("Variants", self.batch_variants)

        self.batch_seed_step = QtWidgets.QSpinBox()
        self.batch_seed_step.setRange(1, 2_000_000_000)
        self.batch_seed_step.setValue(self.cfg.batch.seed_step)
        lay.addRow("Seed step", self.batch_seed_step)

        self.batch_suffix = QtWidgets.QLineEdit(self.cfg.batch.suffix)
        lay.addRow("Suffix", self.batch_suffix)

    # ---------------- Studio tab ----------------
    def _build_studio_tab(self):
        container = self._scrollify_tab(self.tab_studio)
        lay = QtWidgets.QFormLayout(container)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setVerticalSpacing(10)

        self.st_report = QtWidgets.QCheckBox("Write HTML report + thumbnails")
        self.st_report.setChecked(self.cfg.studio.write_report)
        lay.addRow(self.st_report)

        self.st_shotlist = QtWidgets.QCheckBox("Export shotlist.csv")
        self.st_shotlist.setChecked(self.cfg.studio.export_shotlist_csv)
        lay.addRow(self.st_shotlist)

        self.st_resume = QtWidgets.QCheckBox("Crash-safe resume (state.json)")
        self.st_resume.setChecked(self.cfg.studio.crash_safe_resume)

        try:
            stcfg = getattr(self.cfg, 'studio', None)
            if hasattr(self, 'cb_vram_mgr') and stcfg is not None:
                self.cb_vram_mgr.setChecked(bool(getattr(stcfg, 'vram_manager_enabled', True)))
            if hasattr(self, 'sp_vram_min') and stcfg is not None:
                self.sp_vram_min.setValue(float(getattr(stcfg, 'vram_min_free_gb', 2.0) or 2.0))
            if hasattr(self, 'sp_vram_target') and stcfg is not None:
                self.sp_vram_target.setValue(float(getattr(stcfg, 'vram_target_free_gb', 5.0) or 5.0))
            if hasattr(self, 'cb_unload_init') and stcfg is not None:
                self.cb_unload_init.setChecked(bool(getattr(stcfg, 'unload_init_pipeline_after_use', True)))
        except Exception:
            pass

        lay.addRow(self.st_resume)

        # --- VRAM manager (advanced) ---
        try:
            gb_vram = QtWidgets.QGroupBox('VRAM manager')
            fl_vram = QtWidgets.QFormLayout(gb_vram)
            stcfg = getattr(self.cfg, 'studio', None)

            self.cb_vram_mgr = QtWidgets.QCheckBox('Enable VRAM manager')
            self.cb_vram_mgr.setChecked(bool(getattr(stcfg, 'vram_manager_enabled', True) if stcfg else True))

            self.sp_vram_min = QtWidgets.QDoubleSpinBox(); self.sp_vram_min.setRange(0.0, 64.0); self.sp_vram_min.setDecimals(2)
            self.sp_vram_min.setValue(float(getattr(stcfg, 'vram_min_free_gb', 2.0) if stcfg else 2.0))

            self.sp_vram_target = QtWidgets.QDoubleSpinBox(); self.sp_vram_target.setRange(0.0, 64.0); self.sp_vram_target.setDecimals(2)
            self.sp_vram_target.setValue(float(getattr(stcfg, 'vram_target_free_gb', 5.0) if stcfg else 5.0))

            self.cb_unload_init = QtWidgets.QCheckBox('Unload init-image pipeline after use (free VRAM)')
            self.cb_unload_init.setChecked(bool(getattr(stcfg, 'unload_init_pipeline_after_use', True) if stcfg else True))

            fl_vram.addRow(self.cb_vram_mgr)
            fl_vram.addRow('Min free GB', self.sp_vram_min)
            fl_vram.addRow('Target free GB', self.sp_vram_target)
            fl_vram.addRow(self.cb_unload_init)

            lay.addRow(gb_vram)
        except Exception:
            pass


        note = QtWidgets.QLabel("Crash-safe resume: en cas de crash, utilise Fichier → Resume render from folder.")
        note.setWordWrap(True)
        lay.addRow(note)

    # ---------------- Render tab ----------------
    def _build_render_tab(self):
        container = self._scrollify_tab(self.tab_render)
        v = QtWidgets.QVBoxLayout(container)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(10)

        how = QtWidgets.QLabel(
            "<b>How it works</b>: l'app génère des <i>segments</i> (chunks) pour chaque scène, "
            "puis les <i>assemble</i> avec overlap/blend pour créer une vidéo plus longue."
        )
        how.setWordWrap(True)
        v.addWidget(how)

        row = QtWidgets.QHBoxLayout()
        v.addLayout(row)

        self.btn_start = QtWidgets.QPushButton("▶ Render")
        self.btn_cancel = QtWidgets.QPushButton("⏹ Cancel")
        self.btn_cancel.setEnabled(False)
        try:
            self.btn_cancel_clip.setEnabled(False)
        except Exception:
            pass
        try:
            self.btn_cancel_clip.setEnabled(False)
        except Exception:
            pass
        try:
            self.btn_cancel_clip.setEnabled(False)
        except Exception:
            pass
        try:
            self.btn_cancel_clip.setEnabled(False)
        except Exception:
            pass
        self.btn_cancel_clip = QtWidgets.QPushButton("↷ Cancel clip")
        self.btn_cancel_clip.setEnabled(False)
        self.btn_open_out = QtWidgets.QPushButton("📁 Open output dir")
        self.btn_open_out.clicked.connect(self._open_output_dir)
        self.btn_open_report = QtWidgets.QPushButton("🧾 Open last report")
        self.btn_open_report.clicked.connect(self.open_last_report)

        self.btn_cancel_clip.clicked.connect(self.on_cancel_current_clip)

        row.addWidget(self.btn_start)
        row.addWidget(self.btn_cancel)
        row.addWidget(self.btn_cancel_clip)
        row.addWidget(self.btn_open_out)
        row.addWidget(self.btn_open_report)
        row.addStretch(1)



        # Premiere quick actions (selection / range)
        try:
            gb = QtWidgets.QGroupBox("Premiere: selection / range")
            hb = QtWidgets.QHBoxLayout(gb)
            self.btn_prem_sel_legacy = QtWidgets.QPushButton("🎯 Render selection (legacy)")
            self.btn_prem_rng_legacy = QtWidgets.QPushButton("⟲ Render range (legacy)")
            self.btn_prem_sel_proxy = QtWidgets.QPushButton("⚡ Proxy selection")
            self.btn_prem_rng_proxy = QtWidgets.QPushButton("⚡ Proxy range")
            self.btn_prem_sel_final = QtWidgets.QPushButton("🎬 Final selection")
            self.btn_prem_rng_final = QtWidgets.QPushButton("🎬 Final range")

            hb.addWidget(self.btn_prem_sel_legacy)
            hb.addWidget(self.btn_prem_rng_legacy)
            hb.addSpacing(10)
            hb.addWidget(self.btn_prem_sel_proxy)
            hb.addWidget(self.btn_prem_rng_proxy)
            hb.addSpacing(10)
            hb.addWidget(self.btn_prem_sel_final)
            hb.addWidget(self.btn_prem_rng_final)
            hb.addStretch(1)
            v.addWidget(gb)
        except Exception:
            gb = None


        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        v.addWidget(self.progress)

        self.status = QtWidgets.QLabel("Prêt.")
        v.addWidget(self.status)

        # Optional: mux master audio automatically after render (no video re-encode)
        gb_mux = QtWidgets.QGroupBox("Audio master")
        hb_mux = QtWidgets.QHBoxLayout(gb_mux)
        self.cb_render_mux_audio = QtWidgets.QCheckBox("Mux master audio after render (copy video)")
        self.cb_render_mux_audio.setChecked(True)
        self.cb_render_mux_preset = QtWidgets.QComboBox()
        self.cb_render_mux_preset.addItem("MKV + FLAC (lossless)", "mkv_flac")
        self.cb_render_mux_preset.addItem("MKV + PCM s16 (lossless)", "mkv_pcm")
        self.cb_render_mux_preset.addItem("MP4 + AAC 320k", "mp4_aac")
        hb_mux.addWidget(self.cb_render_mux_audio)
        hb_mux.addWidget(self.cb_render_mux_preset)
        hb_mux.addStretch(1)
        v.addWidget(gb_mux)


        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(320)
        v.addWidget(self.video_widget)

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.audio.setVolume(0.8)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video_widget)

        play_row = QtWidgets.QHBoxLayout()
        v.addLayout(play_row)
        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        play_row.addWidget(self.btn_play)
        play_row.addWidget(self.btn_pause)
        play_row.addWidget(self.btn_stop)
        play_row.addWidget(self.slider, 1)

        self.btn_play.clicked.connect(self.player.play)
        self.btn_pause.clicked.connect(self.player.pause)
        self.btn_stop.clicked.connect(self.player.stop)

        self.player.positionChanged.connect(self._on_pos)
        self.player.durationChanged.connect(self._on_dur)
        self.slider.sliderMoved.connect(self._on_seek)

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        v.addWidget(self.log_box, 1)

        self.btn_start.clicked.connect(self.on_start)
        self.btn_cancel.clicked.connect(self.on_cancel)


        try:
            if hasattr(self, "premiere") and self.premiere is not None:
                self.btn_prem_sel_legacy.clicked.connect(lambda: self.premiere.render_selected())
                self.btn_prem_rng_legacy.clicked.connect(lambda: self.premiere.render_range())
                self.btn_prem_sel_proxy.clicked.connect(lambda: self.premiere.render_proxy_selected())
                self.btn_prem_rng_proxy.clicked.connect(lambda: self.premiere.render_proxy_range())
                self.btn_prem_sel_final.clicked.connect(lambda: self.premiere.render_final_selected())
                self.btn_prem_rng_final.clicked.connect(lambda: self.premiere.render_final_range())
            else:
                # fail-safe: disable if premiere not available
                for b in (getattr(self, 'btn_prem_sel_legacy', None), getattr(self, 'btn_prem_rng_legacy', None),
                          getattr(self, 'btn_prem_sel_proxy', None), getattr(self, 'btn_prem_rng_proxy', None),
                          getattr(self, 'btn_prem_sel_final', None), getattr(self, 'btn_prem_rng_final', None)):
                    try:
                        if b is not None:
                            b.setEnabled(False)
                    except Exception:
                        pass
        except Exception:
            pass


    # ---------- sync ----------



    def _update_cinema_safe_ui(self):
        """Lock risky knobs when Cinema SAFE mode is enabled."""
        try:
            en = bool(getattr(self, 'cb_cinema_safe', None) and self.cb_cinema_safe.isChecked())
        except Exception:
            en = False

        # SAFE implies no-overlap strict and UniPC OFF for Wan.
        try:
            if hasattr(self, 'cb_no_overlap_strict'):
                if en:
                    self.cb_no_overlap_strict.setChecked(True)
                self.cb_no_overlap_strict.setEnabled(not en)
        except Exception:
            pass

        try:
            if hasattr(self, 'cb_unipc'):
                if en:
                    self.cb_unipc.setChecked(False)
                self.cb_unipc.setEnabled(not en)
        except Exception:
            pass

        # Offload toggles (VRAM safety)
        try:
            if hasattr(self, 'offload'):
                if en:
                    self.offload.setChecked(True)
                self.offload.setEnabled(not en)
            if hasattr(self, 'seq_offload'):
                if en:
                    self.seq_offload.setChecked(True)
                self.seq_offload.setEnabled(not en)
        except Exception:
            pass

        # VAE decode fp32 is heavy on VRAM; lock OFF in SAFE mode if UI exists
        try:
            if hasattr(self, 'cb_vae_fp32'):
                if en:
                    self.cb_vae_fp32.setChecked(False)
                self.cb_vae_fp32.setEnabled(not en)
        except Exception:
            pass

        # Apply downstream locks (overlap/blend/transition)
        try:
            # Enforce SAFE overrides in saved config (project.json will match runtime)
            try:
                self.cfg = apply_cinema_safe_overrides(self.cfg)
            except Exception:
                pass
            self._update_no_overlap_ui()
            # Optional: when SAFE mode turns ON, switch to a robust high-realism preset.
            try:
                if en and hasattr(self, 'quality'):
                    if self.quality.findText('Cinema Ultra (photoreal max)') >= 0:
                        self.quality.setCurrentText('Cinema Ultra (photoreal max)')
                        # Apply immediately so W/H/steps etc. match the preset
                        self._apply_quality_preset()
            except Exception:
                pass
        except Exception:
            pass

    def _update_no_overlap_ui(self):
        """Enforce UI constraints for no-overlap strict mode.

        When enabled we force:
          - overlap_frames = 0
          - blend_mode = 'cut'
          - scene transitions = cut/0 (so the UI matches engine behaviour)
        """
        try:
            en = bool(getattr(self, 'cb_no_overlap_strict', None) and self.cb_no_overlap_strict.isChecked())
        except Exception:
            en = False

        # Segment stitch controls
        try:
            if hasattr(self, 'overlap'):
                if en:
                    self.overlap.setValue(0)
                self.overlap.setEnabled(not en)
            if hasattr(self, 'blend'):
                if en:
                    self.blend.setCurrentText('cut')
                self.blend.setEnabled(not en)
        except Exception:
            pass

        # Scene transition controls (to avoid confusing "engine forced cut" vs UI)
        try:
            if hasattr(self, 'trans_mode'):
                if en:
                    self.trans_mode.setCurrentText('cut')
                self.trans_mode.setEnabled(not en)
            if hasattr(self, 'trans_frames'):
                if en:
                    self.trans_frames.setValue(0)
                self.trans_frames.setEnabled(not en)
            if hasattr(self, 'trans_ease'):
                self.trans_ease.setEnabled(not en)
        except Exception:
            pass

    def _sync_cfg_from_ui(self):
        """Read current widget state and write into self.cfg.

        This is the inverse of _sync_ui_from_cfg(): it MUST NOT overwrite user edits
        by pushing cfg values back into the widgets.
        """
        # Basic project
        try:
            if hasattr(self, "out_dir"):
                self.cfg.output_dir = self.out_dir.text().strip() or self.cfg.output_dir
            if hasattr(self, "project_name"):
                self.cfg.project_name = self.project_name.text().strip() or self.cfg.project_name
            if hasattr(self, "negative"):
                self.cfg.negative_prompt = self.negative.toPlainText().strip() or self.cfg.negative_prompt
            if hasattr(self, "base_seed"):
                self.cfg.base_seed = int(self.base_seed.value())
            if hasattr(self, "vary_seed"):
                self.cfg.vary_seed_per_segment = bool(self.vary_seed.isChecked())

            if hasattr(self, "trans_mode"):
                self.cfg.scene_transition_mode = self.trans_mode.currentText() or self.cfg.scene_transition_mode
            if hasattr(self, "trans_frames"):
                self.cfg.scene_transition_frames = int(self.trans_frames.value())
            if hasattr(self, "trans_ease"):
                self.cfg.scene_transition_ease = self.trans_ease.currentText() or self.cfg.scene_transition_ease
        except Exception:
            pass

        # Stability / cache toggles (optional widgets)
        try:
            if hasattr(self, 'cb_cut_strict'):
                self.cfg.cut_strict_between_clips = bool(self.cb_cut_strict.isChecked())
            if hasattr(self, 'cb_reset_cont'):
                self.cfg.reset_continuity_between_clips = bool(self.cb_reset_cont.isChecked())
            if hasattr(self, 'cb_strict_clip'):
                self.cfg.strict_clip_coherence = bool(self.cb_strict_clip.isChecked())
            if hasattr(self, 'cb_force_first'):
                self.cfg.force_first_frame_to_conditioning = bool(self.cb_force_first.isChecked())
            if hasattr(self, 'cb_ultra_iso'):
                self.cfg.ultra_isolated_clips = bool(self.cb_ultra_iso.isChecked())
            if hasattr(self, 'cb_clip_cache'):
                self.cfg.clip_cache_enabled = bool(self.cb_clip_cache.isChecked())
            if hasattr(self, 'cb_prune_cache'):
                self.cfg.prune_cache_on_start = bool(self.cb_prune_cache.isChecked())
            if hasattr(self, 'sp_cache_max'):
                self.cfg.cache_max_gb = float(self.sp_cache_max.value())
            if hasattr(self, 'le_clip_cache_dirname'):
                self.cfg.clip_cache_dirname = self.le_clip_cache_dirname.text().strip() or self.cfg.clip_cache_dirname
        except Exception:
            pass

        # Model / render params
        try:
            if hasattr(self, "model_id"):
                self.cfg.model_id = self.model_id.currentText().strip() or self.cfg.model_id
            if hasattr(self, "mode"):
                self.cfg.mode = self.mode.currentText() or self.cfg.mode
            if hasattr(self, "backend"):
                self.cfg.backend = (self.backend.currentText() or "wan")
            if hasattr(self, "input_img"):
                p = self.input_img.text().strip()
                self.cfg.input_image_path = p or None
            if hasattr(self, "fps"):
                self.cfg.fps = int(self.fps.value())
            if hasattr(self, "w"):
                self.cfg.width = int(self.w.value())
            if hasattr(self, "h"):
                self.cfg.height = int(self.h.value())
            if hasattr(self, "steps"):
                self.cfg.num_inference_steps = int(self.steps.value())
            if hasattr(self, "cfg1"):
                self.cfg.guidance_scale = float(self.cfg1.value())
            if hasattr(self, "cfg2"):
                self.cfg.guidance_scale_2 = float(self.cfg2.value())
            if hasattr(self, "boundary"):
                self.cfg.boundary_ratio = float(self.boundary.value())
            if hasattr(self, "chunk_s"):
                self.cfg.chunk_seconds = float(self.chunk_s.value())
        except Exception:
            pass

        # Cinema safe / overlap controls
        try:
            if hasattr(self, 'cb_cinema_safe'):
                self.cfg.cinema_safe_mode = bool(self.cb_cinema_safe.isChecked())
            if hasattr(self, 'cb_no_overlap_strict'):
                self.cfg.no_overlap_strict = bool(self.cb_no_overlap_strict.isChecked())
        except Exception:
            pass

        try:
            if bool(getattr(self.cfg, 'no_overlap_strict', False)):
                self.cfg.overlap_frames = 0
                self.cfg.blend_mode = 'cut'
                # Also force montage transitions to cut
                self.cfg.scene_transition_mode = 'cut'
                self.cfg.scene_transition_frames = 0
            else:
                if hasattr(self, "overlap"):
                    self.cfg.overlap_frames = int(self.overlap.value())
                if hasattr(self, "blend"):
                    self.cfg.blend_mode = self.blend.currentText() or self.cfg.blend_mode
        except Exception:
            pass

        try:
            if hasattr(self, 'cb_unipc'):
                self.cfg.use_unipc = bool(self.cb_unipc.isChecked())
            if hasattr(self, 'sp_flow_shift'):
                self.cfg.flow_shift = float(self.sp_flow_shift.value())
            if hasattr(self, "offload"):
                self.cfg.enable_model_cpu_offload = bool(self.offload.isChecked())
            if hasattr(self, "seq_offload"):
                self.cfg.enable_sequential_cpu_offload = bool(self.seq_offload.isChecked())
            if hasattr(self, "fp32vae"):
                self.cfg.vae_decode_fp32 = bool(self.fp32vae.isChecked())
            if hasattr(self, "keep_frames"):
                self.cfg.keep_png_frames = bool(self.keep_frames.isChecked())
        except Exception:
            pass

        # Init-frame global settings (optional widgets)
        try:
            if getattr(self.cfg, 'init_image', None) is None:
                self.cfg.init_image = InitImageSpec()
            init = getattr(self.cfg, 'init_image', None)
            if init is not None:
                if hasattr(self, 'cb_init_enabled'):
                    init.enabled = bool(self.cb_init_enabled.isChecked())
                if hasattr(self, 'cb_init_preset'):
                    init.preset = self.cb_init_preset.currentText().strip() or getattr(init, 'preset', 'zimage_turbo')
                if hasattr(self, 'cb_init_policy'):
                    init.policy = str(self.cb_init_policy.currentData() or self.cb_init_policy.currentText()).strip() or 'necessary'
                if hasattr(self, 'cb_init_cache_loc'):
                    init.cache_location = str(self.cb_init_cache_loc.currentData() or self.cb_init_cache_loc.currentText()).strip() or 'assets'
                if hasattr(self, 'le_init_cache_dirname'):
                    init.project_cache_dirname = self.le_init_cache_dirname.text().strip() or getattr(init, 'project_cache_dirname', '.wan_cache')
                if hasattr(self, 'init_w') or hasattr(self, 'init_h'):
                    try:
                        from .init_frame_subprocess import clamp_to_multiple_of_8
                        w_raw = int(self.init_w.value()) if hasattr(self, 'init_w') else int(getattr(init, 'width', 0) or 0)
                        h_raw = int(self.init_h.value()) if hasattr(self, 'init_h') else int(getattr(init, 'height', 0) or 0)
                        if w_raw == 0 or h_raw == 0:
                            init.width = 0
                            init.height = 0
                        else:
                            w_clamp, h_clamp = clamp_to_multiple_of_8(w_raw, h_raw)
                            init.width = int(w_clamp)
                            init.height = int(h_clamp)
                    except Exception:
                        if hasattr(self, 'init_w'):
                            init.width = int(self.init_w.value())
                        if hasattr(self, 'init_h'):
                            init.height = int(self.init_h.value())
                if hasattr(self, 'init_steps'):
                    init.steps = int(self.init_steps.value())
                if hasattr(self, 'init_gs'):
                    init.guidance_scale = float(self.init_gs.value())
                if hasattr(self, 'cb_init_use_last_loc'):
                    init.use_last_location_memory = bool(self.cb_init_use_last_loc.isChecked())
                if hasattr(self, 'sp_init_max_refs'):
                    init.max_refs = int(self.sp_init_max_refs.value())

                if hasattr(self, 'cb_init_ip'):
                    init.enable_ip_adapter = bool(self.cb_init_ip.isChecked())
                if hasattr(self, 'le_ip_model'):
                    init.ip_adapter_model_id = self.le_ip_model.text().strip()
                if hasattr(self, 'le_ip_sub'):
                    init.ip_adapter_subfolder = self.le_ip_sub.text().strip()
                if hasattr(self, 'le_ip_weight'):
                    init.ip_adapter_weight_name = self.le_ip_weight.text().strip()
                if hasattr(self, 'sp_ip_char'):
                    init.ip_adapter_scale_character = float(self.sp_ip_char.value())
                if hasattr(self, 'sp_ip_loc'):
                    init.ip_adapter_scale_location = float(self.sp_ip_loc.value())
                if hasattr(self, 'sp_ip_style'):
                    init.ip_adapter_scale_style = float(self.sp_ip_style.value())

                if hasattr(self, 'cb_init_cn'):
                    init.enable_controlnet = bool(self.cb_init_cn.isChecked())
                if hasattr(self, 'cb_cn_type'):
                    init.controlnet_type = self.cb_cn_type.currentText().strip() or 'canny'
                if hasattr(self, 'le_cn_model'):
                    init.controlnet_model_id = self.le_cn_model.text().strip()
                if hasattr(self, 'sp_cn_scale'):
                    init.controlnet_scale = float(self.sp_cn_scale.value())
                if hasattr(self, 'sp_refine'):
                    init.refine_strength = float(self.sp_refine.value())

                if hasattr(self, 'le_init_flux2_bnb4bit'):
                    init.flux2_bnb4bit_model_id = self.le_init_flux2_bnb4bit.text().strip()
                if hasattr(self, 'le_init_flux2_full'):
                    init.flux2_model_id = self.le_init_flux2_full.text().strip()
                if hasattr(self, 'le_init_zimage_turbo'):
                    init.zimage_turbo_model_id = self.le_init_zimage_turbo.text().strip()
                if hasattr(self, 'le_init_zimage'):
                    init.zimage_model_id = self.le_init_zimage.text().strip()
                if hasattr(self, 'le_init_sdxl_base'):
                    init.sdxl_base_model_id = self.le_init_sdxl_base.text().strip()
                if hasattr(self, 'le_init_sdxl_turbo'):
                    init.sdxl_turbo_model_id = self.le_init_sdxl_turbo.text().strip()
                if hasattr(self, 'le_init_sdxl_lora'):
                    init.sdxl_lightning_lora_id = self.le_init_sdxl_lora.text().strip()
                if hasattr(self, 'le_init_sdxl_lora_file'):
                    init.sdxl_lightning_lora_file = self.le_init_sdxl_lora_file.text().strip()
                if hasattr(self, 'le_init_flux1'):
                    init.flux1_schnell_model_id = self.le_init_flux1.text().strip()
        except Exception:
            pass

        # OOM / VRAM resilience (optional widgets)
        try:
            if hasattr(self, 'cb_auto_vram'):
                self.cfg.auto_vram_optimizations = bool(self.cb_auto_vram.isChecked())
            if hasattr(self, 'sp_max_frames_seg'):
                self.cfg.max_frames_per_segment = int(self.sp_max_frames_seg.value())
            if hasattr(self, 'sp_oom_retry'):
                self.cfg.oom_retry_max_attempts = int(self.sp_oom_retry.value())
        except Exception:
            pass

        # Audio tab
        try:
            if hasattr(self, 'audio_scene_list'):
                self._audio_save_scene_fields()
            if hasattr(self, 'audio_enabled'):
                self._audio_apply_globals()
        except Exception:
            pass

        # Timeline + LoRAs: keep SceneSpec objects stable
        try:
            if hasattr(self, "scene_table"):
                self._apply_scene_table_to_cfg_inplace()
            if hasattr(self, "_read_loras_from_table"):
                self.cfg.loras = self._read_loras_from_table()
        except Exception:
            pass

        # Post tab
        try:
            post = getattr(self.cfg, "post", None)
            if post is None:
                post = PostProcessSpec()
                self.cfg.post = post

            if hasattr(self, "post_enabled"):
                post.enabled = bool(self.post_enabled.isChecked())
            if hasattr(self, "tools_dir"):
                post.tools_dir = self.tools_dir.text().strip() or post.tools_dir
            if hasattr(self, "use_rife"):
                post.use_rife_if_available = bool(self.use_rife.isChecked())
            if hasattr(self, "use_re"):
                post.use_realesrgan_if_available = bool(self.use_re.isChecked())
            if hasattr(self, "target_fps"):
                post.target_fps = int(self.target_fps.value())
            if hasattr(self, "out_w"):
                post.out_width = int(self.out_w.value())
            if hasattr(self, "out_h"):
                post.out_height = int(self.out_h.value())
            if hasattr(self, "deflicker"):
                post.deflicker = bool(self.deflicker.isChecked())
            if hasattr(self, "denoise"):
                post.denoise = bool(self.denoise.isChecked())
            if hasattr(self, "sharpen"):
                post.sharpen = bool(self.sharpen.isChecked())
            if hasattr(self, "export_prores"):
                post.export_master_prores = bool(self.export_prores.isChecked())

            if hasattr(self, "dn_luma"):
                post.denoise_luma = float(self.dn_luma.value())
            if hasattr(self, "dn_chroma"):
                post.denoise_chroma = float(self.dn_chroma.value())
            if hasattr(self, "dn_temp"):
                post.denoise_temporal = float(self.dn_temp.value())
            if hasattr(self, "dn_temp_chroma"):
                post.denoise_temporal_chroma = float(self.dn_temp_chroma.value())
            if hasattr(self, "unsharp_mx"):
                post.unsharp_mx = int(self.unsharp_mx.value())
            if hasattr(self, "unsharp_my"):
                post.unsharp_my = int(self.unsharp_my.value())
            if hasattr(self, "unsharp_amt"):
                post.unsharp_amount = float(self.unsharp_amt.value())

            if hasattr(self, "export_h265"):
                post.export_delivery_h265_main10 = bool(self.export_h265.isChecked())
            if hasattr(self, "h265_crf"):
                post.h265_crf = int(self.h265_crf.value())
            if hasattr(self, "h265_preset"):
                post.h265_preset = self.h265_preset.currentText() or post.h265_preset

            if hasattr(self, "primary_deliver"):
                idx = int(self.primary_deliver.currentIndex())
                post.primary_deliver = "prores" if idx == 0 else ("h265" if idx == 1 else "intermediate")
            if hasattr(self, "prores_profile"):
                idx = int(self.prores_profile.currentIndex())
                post.prores_profile = ["422hq","422","lt","proxy","4444","4444xq"][max(0, min(5, idx))]
        except Exception:
            pass

        # Batch / Studio
        try:
            if hasattr(self.cfg, "batch"):
                b = self.cfg.batch
                if hasattr(self, "batch_enabled"):
                    b.enabled = bool(self.batch_enabled.isChecked())
                if hasattr(self, "batch_variants"):
                    b.variants = int(self.batch_variants.value())
                if hasattr(self, "batch_seed_step"):
                    b.seed_step = int(self.batch_seed_step.value())
                if hasattr(self, "batch_suffix"):
                    b.suffix = self.batch_suffix.text().strip() or "v"
        except Exception:
            pass

        try:
            if hasattr(self.cfg, "studio"):
                s = self.cfg.studio
                if hasattr(self, "st_report"):
                    s.write_report = bool(self.st_report.isChecked())
                if hasattr(self, "st_shotlist"):
                    s.export_shotlist_csv = bool(self.st_shotlist.isChecked())
                if hasattr(self, "st_resume"):
                    s.crash_safe_resume = bool(self.st_resume.isChecked())

                # VRAM manager controls
                if hasattr(self, 'cb_vram_mgr'):
                    s.vram_manager_enabled = bool(self.cb_vram_mgr.isChecked())
                if hasattr(self, 'sp_vram_min'):
                    s.vram_min_free_gb = float(self.sp_vram_min.value())
                if hasattr(self, 'sp_vram_target'):
                    s.vram_target_free_gb = float(self.sp_vram_target.value())
                if hasattr(self, 'cb_unload_init'):
                    s.unload_init_pipeline_after_use = bool(self.cb_unload_init.isChecked())
        except Exception:
            pass

    def _sync_ui_from_cfg(self):
        """Push cfg -> UI.

        This function **must not** mutate cfg. It is called after loading JSON
        projects; changing cfg here can erase loaded scenes and break the
        timeline.
        """
        cfg = self.cfg

        # --- project ---
        try:
            self.out_dir.setText(getattr(cfg, 'output_dir', '') or "./outputs")
            self.project_name.setText(getattr(cfg, 'project_name', '') or "project")
            self.negative.setPlainText(getattr(cfg, 'negative_prompt', '') or "")
            self.base_seed.setValue(int(getattr(cfg, 'base_seed', 0) or 0))
            self.vary_seed.setChecked(bool(getattr(cfg, 'vary_seed_per_segment', False)))
        except Exception:
            pass

        # --- transitions ---
        try:
            self.trans_mode.setCurrentText(getattr(cfg, 'scene_transition_mode', '') or "cut")
            self.trans_frames.setValue(int(getattr(cfg, 'scene_transition_frames', 0) or 0))
            self.trans_ease.setCurrentText(getattr(cfg, 'scene_transition_ease', '') or "linear")
        except Exception:
            pass

        # --- model / render ---
        try:
            self.model_id.setCurrentText(getattr(cfg, 'model_id', '') or "")
            self.mode.setCurrentText(getattr(cfg, 'mode', '') or "")
            if hasattr(self, 'backend'):
                self.backend.setCurrentText(getattr(cfg, 'backend', 'wan') or 'wan')
            self.input_img.setText(getattr(cfg, 'input_image_path', '') or "")
            self.fps.setValue(int(getattr(cfg, 'fps', 24) or 24))
            self.w.setValue(int(getattr(cfg, 'width', 1024) or 1024))
            self.h.setValue(int(getattr(cfg, 'height', 576) or 576))
            self.steps.setValue(int(getattr(cfg, 'num_inference_steps', 30) or 30))
            self.cfg1.setValue(float(getattr(cfg, 'guidance_scale', 5.0) or 0.0))
            self.cfg2.setValue(float(getattr(cfg, 'guidance_scale_2', 0.0) or 0.0))
            self.boundary.setValue(float(getattr(cfg, 'boundary_ratio', 0.0) or 0.0))
            self.chunk_s.setValue(float(getattr(cfg, 'chunk_seconds', 0.0) or 0.0))
        except Exception:
            pass

        # Cinema SAFE / strict modes
        try:
            if hasattr(self, 'cb_cinema_safe'):
                self.cb_cinema_safe.setChecked(bool(getattr(cfg, 'cinema_safe_mode', False)))
            if hasattr(self, 'cb_no_overlap_strict'):
                self.cb_no_overlap_strict.setChecked(bool(getattr(cfg, 'no_overlap_strict', False)))
            if hasattr(self, 'overlap'):
                self.overlap.setValue(int(getattr(cfg, 'overlap_frames', 0) or 0))
            if hasattr(self, 'blend'):
                self.blend.setCurrentText(str(getattr(cfg, 'blend_mode', 'cut') or 'cut'))
            if hasattr(self, 'cb_unipc'):
                self.cb_unipc.setChecked(bool(getattr(cfg, 'use_unipc', False)))
            if hasattr(self, 'sp_flow_shift'):
                self.sp_flow_shift.setValue(float(getattr(cfg, 'flow_shift', 0.0) or 0.0))
        except Exception:
            pass

        try:
            self.offload.setChecked(bool(getattr(cfg, 'enable_model_cpu_offload', False)))
            if hasattr(self, 'seq_offload'):
                self.seq_offload.setChecked(bool(getattr(cfg, 'enable_sequential_cpu_offload', False)))
            self.fp32vae.setChecked(bool(getattr(cfg, 'vae_decode_fp32', False)))
            self.keep_frames.setChecked(bool(getattr(cfg, 'keep_png_frames', False)))
        except Exception:
            pass

        # Init-frame (global)
        try:
            if getattr(cfg, 'init_image', None) is None:
                cfg.init_image = InitImageSpec()
            init = cfg.init_image
            if init is not None:
                if hasattr(self, 'cb_init_enabled'):
                    self.cb_init_enabled.setChecked(bool(getattr(init, 'enabled', False)))
                if hasattr(self, 'cb_init_preset'):
                    self.cb_init_preset.setCurrentText(getattr(init, 'preset', 'zimage_turbo') or 'zimage_turbo')
                if hasattr(self, 'cb_init_policy'):
                    # prefer data if combo stores it
                    pol = getattr(init, 'policy', 'necessary') or 'necessary'
                    try:
                        idx = self.cb_init_policy.findData(pol)
                        if idx >= 0:
                            self.cb_init_policy.setCurrentIndex(idx)
                        else:
                            self.cb_init_policy.setCurrentText(pol)
                    except Exception:
                        self.cb_init_policy.setCurrentText(pol)
                if hasattr(self, 'cb_init_cache_loc'):
                    loc = getattr(init, 'cache_location', 'assets') or 'assets'
                    try:
                        idx = self.cb_init_cache_loc.findData(loc)
                        if idx >= 0:
                            self.cb_init_cache_loc.setCurrentIndex(idx)
                        else:
                            self.cb_init_cache_loc.setCurrentText(loc)
                    except Exception:
                        self.cb_init_cache_loc.setCurrentText(loc)
                if hasattr(self, 'le_init_cache_dirname'):
                    self.le_init_cache_dirname.setText(getattr(init, 'project_cache_dirname', '.wan_cache') or '.wan_cache')
                if hasattr(self, 'init_w'):
                    self.init_w.setValue(int(getattr(init, 'width', 0) or getattr(cfg, 'width', 1024) or 1024))
                if hasattr(self, 'init_h'):
                    self.init_h.setValue(int(getattr(init, 'height', 0) or getattr(cfg, 'height', 576) or 576))
                if hasattr(self, 'init_steps'):
                    self.init_steps.setValue(int(getattr(init, 'steps', 28) or 28))
                if hasattr(self, 'init_gs'):
                    self.init_gs.setValue(float(getattr(init, 'guidance_scale', 4.5) or 4.5))
                if hasattr(self, 'cb_init_use_last_loc'):
                    self.cb_init_use_last_loc.setChecked(bool(getattr(init, 'use_last_location_memory', True)))
                if hasattr(self, 'sp_init_max_refs'):
                    self.sp_init_max_refs.setValue(int(getattr(init, 'max_refs', 4) or 4))

                if hasattr(self, 'cb_init_ip'):
                    self.cb_init_ip.setChecked(bool(getattr(init, 'enable_ip_adapter', False)))
                if hasattr(self, 'le_ip_model'):
                    self.le_ip_model.setText(getattr(init, 'ip_adapter_model_id', '') or '')
                if hasattr(self, 'le_ip_sub'):
                    self.le_ip_sub.setText(getattr(init, 'ip_adapter_subfolder', '') or '')
                if hasattr(self, 'le_ip_weight'):
                    self.le_ip_weight.setText(getattr(init, 'ip_adapter_weight_name', '') or '')
                if hasattr(self, 'sp_ip_char'):
                    self.sp_ip_char.setValue(float(getattr(init, 'ip_adapter_scale_character', 0.0) or 0.0))
                if hasattr(self, 'sp_ip_loc'):
                    self.sp_ip_loc.setValue(float(getattr(init, 'ip_adapter_scale_location', 0.0) or 0.0))
                if hasattr(self, 'sp_ip_style'):
                    self.sp_ip_style.setValue(float(getattr(init, 'ip_adapter_scale_style', 0.0) or 0.0))

                if hasattr(self, 'cb_init_cn'):
                    self.cb_init_cn.setChecked(bool(getattr(init, 'enable_controlnet', False)))
                if hasattr(self, 'cb_cn_type'):
                    self.cb_cn_type.setCurrentText(getattr(init, 'controlnet_type', 'canny') or 'canny')
                if hasattr(self, 'le_cn_model'):
                    self.le_cn_model.setText(getattr(init, 'controlnet_model_id', '') or '')
                if hasattr(self, 'sp_cn_scale'):
                    self.sp_cn_scale.setValue(float(getattr(init, 'controlnet_scale', 1.0) or 1.0))
                if hasattr(self, 'sp_refine'):
                    self.sp_refine.setValue(float(getattr(init, 'refine_strength', 0.3) or 0.3))
                if hasattr(self, 'le_init_flux2_bnb4bit'):
                    self.le_init_flux2_bnb4bit.setText(getattr(init, 'flux2_bnb4bit_model_id', '') or '')
                if hasattr(self, 'le_init_flux2_full'):
                    self.le_init_flux2_full.setText(getattr(init, 'flux2_model_id', '') or '')
                if hasattr(self, 'le_init_zimage_turbo'):
                    self.le_init_zimage_turbo.setText(getattr(init, 'zimage_turbo_model_id', '') or '')
                if hasattr(self, 'le_init_zimage'):
                    self.le_init_zimage.setText(getattr(init, 'zimage_model_id', '') or '')
                if hasattr(self, 'le_init_sdxl_base'):
                    self.le_init_sdxl_base.setText(getattr(init, 'sdxl_base_model_id', '') or '')
                if hasattr(self, 'le_init_sdxl_turbo'):
                    self.le_init_sdxl_turbo.setText(getattr(init, 'sdxl_turbo_model_id', '') or '')
                if hasattr(self, 'le_init_sdxl_lora'):
                    self.le_init_sdxl_lora.setText(getattr(init, 'sdxl_lightning_lora_id', '') or '')
                if hasattr(self, 'le_init_sdxl_lora_file'):
                    self.le_init_sdxl_lora_file.setText(getattr(init, 'sdxl_lightning_lora_file', '') or '')
                if hasattr(self, 'le_init_flux1'):
                    self.le_init_flux1.setText(getattr(init, 'flux1_schnell_model_id', '') or '')
        except Exception:
            pass

        # Optional UI update helpers
        try:
            if hasattr(self, '_update_cinema_safe_ui'):
                self._update_cinema_safe_ui()
        except Exception:
            pass
        try:
            if hasattr(self, '_update_no_overlap_ui'):
                self._update_no_overlap_ui()
        except Exception:
            pass
        try:
            if hasattr(self, '_update_vram'):
                self._update_vram()
        except Exception:
            pass

        # --- timeline & studio ---
        try:
            self._reload_scene_table()
        except Exception:
            pass
        try:
            if hasattr(self, 'premiere') and self.premiere:
                self.premiere.reload_from_cfg(keep_selection=False)
        except Exception:
            pass
        try:
            if hasattr(self, '_refresh_audio_scene_list'):
                self._refresh_audio_scene_list(keep_selection=False)
        except Exception:
            pass


    # ---------- timeline ----------

        # Sync mixer dock (if present)
        try:
            if getattr(self, "dock_mixer", None) is not None:
                self.dock_mixer.sync_from_cfg(self.cfg)
        except Exception:
            pass
    def _reload_scene_table(self):
        # Prevent itemChanged feedback while we rebuild the table.
        self._table_loading = True
        try:
            self.scene_table.setRowCount(len(self.cfg.scenes))
            for r, s in enumerate(self.cfg.scenes):
                def put(col, val):
                    self.scene_table.setItem(r, col, QtWidgets.QTableWidgetItem(val))

                put(0, s.label)
                put(1, s.tags)
                put(2, str(s.seconds))
                put(3, s.prompt)
                put(4, s.negative_prompt or "")
                put(5, "" if s.num_inference_steps is None else str(s.num_inference_steps))
                put(6, "" if s.guidance_scale is None else str(s.guidance_scale))
                put(7, "" if s.guidance_scale_2 is None else str(s.guidance_scale_2))
                put(8, "" if s.boundary_ratio is None else str(s.boundary_ratio))
                put(9, s.blend_mode or "")
                put(10, "" if s.transition_frames is None else str(s.transition_frames))
                put(11, s.transition_ease or "")

                it = self.scene_table.item(r, 0)
                if it is not None:
                    it.setData(QtCore.Qt.ItemDataRole.UserRole, s.notes or "")

            self._apply_filter()

            # Audio tab depends on cfg.scenes (dialogue markers, ordering)
            try:
                if hasattr(self, "audio_scene_list"):
                    self._refresh_audio_scene_list()
            except Exception:
                pass
        finally:
            self._table_loading = False

    def _read_scenes_from_table(self) -> List[SceneSpec]:
        scenes: List[SceneSpec] = []
        for r in range(self.scene_table.rowCount()):
            label = self._cell_text(self.scene_table, r, 0, "").strip()
            tags = self._cell_text(self.scene_table, r, 1, "").strip()
            sec = self._cell_float(self.scene_table, r, 2, 5.0)
            prompt = self._cell_text(self.scene_table, r, 3, "").strip() or "Cinematic shot, gentle camera motion."
            neg = self._cell_text(self.scene_table, r, 4, "").strip() or None
            steps = self._cell_int_opt(self.scene_table, r, 5)
            cfg1 = self._cell_float_opt(self.scene_table, r, 6)
            cfg2 = self._cell_float_opt(self.scene_table, r, 7)
            br = self._cell_float_opt(self.scene_table, r, 8)
            blend = self._cell_text(self.scene_table, r, 9, "").strip() or None
            tf = self._cell_int_opt(self.scene_table, r, 10)
            ease = self._cell_text(self.scene_table, r, 11, "").strip() or None

            notes = ""
            it = self.scene_table.item(r, 0)
            if it is not None:
                notes = it.data(QtCore.Qt.ItemDataRole.UserRole) or ""

            scenes.append(SceneSpec(label=label, tags=tags, notes=str(notes),
                                    seconds=sec, prompt=prompt, negative_prompt=neg,
                                    num_inference_steps=steps, guidance_scale=cfg1, guidance_scale_2=cfg2,
                                    boundary_ratio=br, blend_mode=blend, transition_frames=tf, transition_ease=ease))
        return scenes or [SceneSpec()]

    def _on_scene_table_item_changed(self, item: QtWidgets.QTableWidgetItem):
        """Debounced: keep cfg.scenes in sync when user edits the timeline table."""
        try:
            if getattr(self, "_table_loading", False):
                return
        except Exception:
            return
        self._scene_table_dirty = True
        try:
            # small debounce to avoid committing on every keystroke
            if hasattr(self, "_scene_table_timer"):
                self._scene_table_timer.start(450)
        except Exception:
            pass

    def _commit_scene_table_edits(self):
        if not getattr(self, "_scene_table_dirty", False):
            return
        self._scene_table_dirty = False
        try:
            self._apply_scene_table_to_cfg_inplace()
        except Exception:
            return

        # If Studio (Premiere) is open, refresh clip geometry/text.
        # (This is safe here because it is triggered by edits in the table.)
        try:
            if hasattr(self, "premiere") and self.premiere is not None:
                self.premiere.reload_from_cfg(keep_selection=True)
        except Exception:
            pass

        try:
            if hasattr(self, "audio_scene_list"):
                self._refresh_audio_scene_list()
        except Exception:
            pass

    def _apply_scene_table_to_cfg_inplace(self):
        """Update self.cfg.scenes from the timeline table WITHOUT replacing SceneSpec objects.

        Replacing the list breaks references held by the Studio timeline/inspector
        and makes edits look like they "disappear".
        """
        if not hasattr(self, "scene_table"):
            return

        n = int(self.scene_table.rowCount())
        if n <= 0:
            # Keep at least one scene
            if not self.cfg.scenes:
                self.cfg.scenes = [SceneSpec()]
            return

        # Ensure list size (but keep existing objects when possible)
        while len(self.cfg.scenes) < n:
            self.cfg.scenes.append(SceneSpec())
        if len(self.cfg.scenes) > n:
            self.cfg.scenes = self.cfg.scenes[:n]

        for r in range(n):
            s = self.cfg.scenes[r]

            label = self._cell_text(self.scene_table, r, 0, "").strip()
            tags = self._cell_text(self.scene_table, r, 1, "").strip()
            sec = self._cell_float(self.scene_table, r, 2, float(s.seconds or 5.0))
            prompt_cell = self._cell_text(self.scene_table, r, 3, "").strip()
            neg_cell = self._cell_text(self.scene_table, r, 4, "").strip()

            # Optional per-shot overrides (kept in table)
            steps = self._cell_int_opt(self.scene_table, r, 5)
            cfg1 = self._cell_float_opt(self.scene_table, r, 6)
            cfg2 = self._cell_float_opt(self.scene_table, r, 7)
            br = self._cell_float_opt(self.scene_table, r, 8)
            blend = self._cell_text(self.scene_table, r, 9, "").strip() or None
            tf = self._cell_int_opt(self.scene_table, r, 10)
            ease = self._cell_text(self.scene_table, r, 11, "").strip() or None

            # Notes stored on col 0 UserRole
            notes = ""
            it = self.scene_table.item(r, 0)
            if it is not None:
                try:
                    notes = it.data(QtCore.Qt.ItemDataRole.UserRole) or ""
                except Exception:
                    notes = ""

            # Apply edits
            if label:
                s.label = label
            elif not (s.label or "").strip():
                s.label = f"S01_SH{r+1:02d}"
            s.tags = tags
            s.notes = str(notes)
            s.seconds = float(max(0.1, sec))

            # Prevent accidental prompt wipes when other views are the editor of truth.
            # If the table cell is empty, we keep the current prompt.
            if prompt_cell:
                s.prompt = prompt_cell

            s.negative_prompt = neg_cell or None
            s.num_inference_steps = steps
            s.guidance_scale = cfg1
            s.guidance_scale_2 = cfg2
            s.boundary_ratio = br
            s.blend_mode = blend
            s.transition_frames = tf
            s.transition_ease = ease

    def _refresh_premiere_from_cfg(self, keep_selection: bool = True):
        """Sync the Studio (Premiere) timeline from cfg.scenes (best-effort)."""
        try:
            if hasattr(self, "premiere") and self.premiere is not None:
                self.premiere.reload_from_cfg(keep_selection=bool(keep_selection))
        except Exception:
            pass

    def on_add_scene(self):
        self.cfg.scenes.append(SceneSpec(label=f"S01_SH{len(self.cfg.scenes)+1:02d}", tags="", seconds=5.0))
        self._reload_scene_table()
        self._refresh_premiere_from_cfg(keep_selection=True)

    def on_prompt_to_scene(self):
        r = self.scene_table.currentRow()
        if r < 0:
            r = 0
        d = PromptBuilderDialog(self)
        d.exec()
        prompt = d.prompt()
        if r >= self.scene_table.rowCount():
            self.on_add_scene()
            r = self.scene_table.rowCount() - 1
        self.scene_table.setItem(r, 3, QtWidgets.QTableWidgetItem(prompt))

    def on_dup_scene(self):
        r = self.scene_table.currentRow()
        if r < 0 or r >= len(self.cfg.scenes):
            return
        import copy
        self.cfg.scenes.insert(r + 1, copy.deepcopy(self.cfg.scenes[r]))
        self._reload_scene_table()
        self.scene_table.selectRow(r + 1)
        self._refresh_premiere_from_cfg(keep_selection=True)

    def on_del_scene(self):
        r = self.scene_table.currentRow()
        if r < 0:
            return
        if 0 <= r < len(self.cfg.scenes):
            self.cfg.scenes.pop(r)
            self._reload_scene_table()
            self._refresh_premiere_from_cfg(keep_selection=True)

    def _move_scene(self, delta: int):
        r = self.scene_table.currentRow()
        if r < 0:
            return
        nr = r + delta
        if not (0 <= nr < len(self.cfg.scenes)):
            return
        self.cfg.scenes[r], self.cfg.scenes[nr] = self.cfg.scenes[nr], self.cfg.scenes[r]
        self._reload_scene_table()
        self.scene_table.selectRow(nr)
        self._refresh_premiere_from_cfg(keep_selection=True)

    def _on_scene_select(self):
        r = self.scene_table.currentRow()
        if r < 0:
            return
        self.ins_label.setText(self._cell_text(self.scene_table, r, 0, ""))
        self.ins_tags.setText(self._cell_text(self.scene_table, r, 1, ""))
        it = self.scene_table.item(r, 0)
        notes = ""
        if it is not None:
            notes = it.data(QtCore.Qt.ItemDataRole.UserRole) or ""
        self.ins_notes.setPlainText(str(notes))

    def _apply_inspector(self):
        r = self.scene_table.currentRow()
        if r < 0:
            return
        self.scene_table.setItem(r, 0, QtWidgets.QTableWidgetItem(self.ins_label.text().strip()))
        self.scene_table.setItem(r, 1, QtWidgets.QTableWidgetItem(self.ins_tags.text().strip()))
        it = self.scene_table.item(r, 0)
        if it is not None:
            it.setData(QtCore.Qt.ItemDataRole.UserRole, self.ins_notes.toPlainText().strip())

    def _autofill_labels(self):
        for i in range(self.scene_table.rowCount()):
            lab = self._cell_text(self.scene_table, i, 0, "").strip()
            if not lab:
                self.scene_table.setItem(i, 0, QtWidgets.QTableWidgetItem(f"S01_SH{i+1:02d}"))

    def _apply_filter(self):
        q = self.scene_filter.text().strip().lower()
        for r in range(self.scene_table.rowCount()):
            if not q:
                self.scene_table.setRowHidden(r, False)
                continue
            label = self._cell_text(self.scene_table, r, 0, "").lower()
            tags = self._cell_text(self.scene_table, r, 1, "").lower()
            prompt = self._cell_text(self.scene_table, r, 3, "").lower()
            show = (q in label) or (q in tags) or (q in prompt)
            self.scene_table.setRowHidden(r, not show)

    # ---------- loras ----------
    def _reload_lora_table(self):
        self.lora_table.setRowCount(len(self.cfg.loras))
        for r, l in enumerate(self.cfg.loras):
            def put(col, val): self.lora_table.setItem(r, col, QtWidgets.QTableWidgetItem(val))
            put(0, l.name)
            put(1, str(l.weight))
            put(2, l.repo_id or "")
            put(3, l.weight_name or "")
            put(4, l.local_path or "")
            put(5, "1" if l.load_into_transformer_2 else "0")

    def _read_loras_from_table(self) -> List[LoraSpec]:
        loras: List[LoraSpec] = []
        for r in range(self.lora_table.rowCount()):
            name = self._cell_text(self.lora_table, r, 0, "").strip()
            w = self._cell_float(self.lora_table, r, 1, 1.0)
            repo = self._cell_text(self.lora_table, r, 2, "").strip() or None
            wn = self._cell_text(self.lora_table, r, 3, "").strip() or None
            lp = self._cell_text(self.lora_table, r, 4, "").strip() or None
            t2 = self._cell_text(self.lora_table, r, 5, "0").strip().lower()
            load_t2 = t2 in ("1","true","yes","y")

            # UX guard: allow adding an empty row without crashing renders.
            # If the row has no identifier AND no source, ignore it.
            if not name:
                if lp:
                    name = os.path.splitext(os.path.basename(lp))[0]
                elif repo and wn:
                    name = os.path.splitext(os.path.basename(wn))[0]
                else:
                    continue

            # If user typed a name but didn't provide a source, ignore this row.
            # (Otherwise render will fail with a ValueError.)
            if not lp and not (repo and wn):
                continue

            loras.append(LoraSpec(name=name, weight=w, repo_id=repo, weight_name=wn, local_path=lp, load_into_transformer_2=load_t2))
        return loras

    def on_add_lora(self):
        # UX: don't create placeholder rows that later crash renders.
        # Instead, let the user add a LoRA from a local file or Hugging Face.
        menu = QtWidgets.QMenu(self)
        act_local = menu.addAction("Depuis un fichier local (.safetensors)…")
        act_hf = menu.addAction("Depuis Hugging Face (repo + filename)…")
        act_cancel = menu.addAction("Annuler")
        act_cancel.setEnabled(False)

        chosen = menu.exec_(QtGui.QCursor.pos())
        if chosen is None:
            return

        if chosen == act_local:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Choisir un fichier LoRA",
                "",
                "LoRA Weights (*.safetensors *.pt *.bin);;Tous les fichiers (*)",
            )
            if not path:
                return
            try:
                name = os.path.splitext(os.path.basename(path))[0]
            except Exception:
                name = "lora_local"
            self.cfg.loras.append(LoraSpec(name=name, local_path=path, weight=1.0))
            self._reload_lora_table()
            return

        if chosen == act_hf:
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Ajouter une LoRA (Hugging Face)")
            layout = QtWidgets.QFormLayout(dlg)

            name_edit = QtWidgets.QLineEdit(dlg)
            repo_edit = QtWidgets.QLineEdit(dlg)
            file_edit = QtWidgets.QLineEdit(dlg)
            weight_spin = QtWidgets.QDoubleSpinBox(dlg)
            weight_spin.setRange(0.0, 2.0)
            weight_spin.setSingleStep(0.05)
            weight_spin.setValue(1.0)

            backend_combo = QtWidgets.QComboBox(dlg)
            backend_combo.addItem("")   # no constraint
            backend_combo.addItem("wan")
            backend_combo.addItem("ltx")
            backend_combo.addItem("cog")

            t2_check = QtWidgets.QCheckBox("Charger dans transformer_2 (low-noise)", dlg)

            layout.addRow("Nom (optionnel)", name_edit)
            layout.addRow("Repo ID (ex: lightx2v/Wan2.2-Distill-Loras)", repo_edit)
            layout.addRow("Fichier (ex: loras/wan/xxx.safetensors)", file_edit)
            layout.addRow("Poids", weight_spin)
            layout.addRow("Backend cible (optionnel)", backend_combo)
            layout.addRow("", t2_check)

            btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, dlg)
            layout.addRow(btns)
            btns.accepted.connect(dlg.accept)
            btns.rejected.connect(dlg.reject)

            if dlg.exec_() != QtWidgets.QDialog.Accepted:
                return

            repo_id = repo_edit.text().strip()
            weight_name = file_edit.text().strip()
            if not repo_id or not weight_name:
                QtWidgets.QMessageBox.warning(self, "LoRA", "Il faut renseigner repo_id et fichier (weight_name).")
                return

            nm = name_edit.text().strip()
            if not nm:
                nm = os.path.splitext(os.path.basename(weight_name))[0]

            tb = backend_combo.currentText().strip() or None

            self.cfg.loras.append(
                LoraSpec(
                    name=nm,
                    repo_id=repo_id,
                    weight_name=weight_name,
                    weight=float(weight_spin.value()),
                    load_into_transformer_2=bool(t2_check.isChecked()),
                    target_backend=tb,
                )
            )
            self._reload_lora_table()
            return

    def on_del_lora(self):
        r = self.lora_table.currentRow()
        if r < 0:
            return
        if 0 <= r < len(self.cfg.loras):
            self.cfg.loras.pop(r)
            self._reload_lora_table()

    def on_apply_pack(self):
        name = self.pack.currentText()
        # Guard: some LoRA packs are model-specific (ex: A14B vs 5B)
        cur_model = self.model_id.currentText().strip()
        if "I2V A14B" in name and ("A14B" not in cur_model):
            QtWidgets.QMessageBox.warning(self, "LoRA incompatible",
                "Le pack LightX2V 4-step est entraîné pour les modèles A14B (I2V).\n"
                "Tu es sur un modèle 5B (TI2V).\n\n"
                "→ Choisis un modèle A14B (Wan2.2-I2V-A14B-...) ou utilise un LoRA compatible 5B.")
            return

        if name == "(aucun)":
            self.cfg.loras.clear()
        else:
            self.cfg.loras = default_lora_packs()[name]
        self._reload_lora_table()

        # UX guard: many packs are I2V- or T2V-specific. If the user applies an
        # I2V pack while the app is still in T2V mode (or vice-versa), we end up
        # skipping LoRAs and sometimes loading the wrong pipeline for the selected model.
        # Auto-sync the Mode combobox to avoid confusing behavior.
        try:
            want_mode = None
            up = str(name or "").upper()
            if "I2V" in up:
                want_mode = "I2V"
            elif "T2V" in up:
                want_mode = "T2V"
            else:
                # Infer from LoRA filenames
                hint = " ".join([
                    " ".join([str(l.name or ""), str(l.repo_id or ""), str(l.weight_name or ""), str(l.local_path or "")])
                    for l in (self.cfg.loras or [])
                ]).upper()
                if "I2V" in hint:
                    want_mode = "I2V"
                elif "T2V" in hint:
                    want_mode = "T2V"

            if want_mode and str(getattr(self.cfg, "mode", "") or "").upper() != want_mode:
                self.cfg.mode = want_mode
                try:
                    self.mode.setCurrentText(want_mode)
                except Exception:
                    pass
                try:
                    self.append_log(f"[Mode] Auto-sync: pack LoRA → {want_mode}")
                except Exception:
                    pass
        except Exception:
            pass
        self.append_log(f"Pack LoRA appliqué: {name}")

    # ---------- post ----------
    def on_detect_tools(self):
        d = self.tools_dir.text().strip() or "./tools"
        st = detect_tools(d)
        self.tool_status.setText(f"ffmpeg={bool(st.ffmpeg)} | rife={bool(st.rife)} | realesrgan={bool(st.realesrgan)} (dir={os.path.abspath(d)})")

    # ---------- render ----------
    def _start_worker_cfg(self, cfg_obj, resume_folder: str | None = None):
        """Internal: start render worker from a provided config object."""
        self.progress.setValue(0)
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        try:
            self.btn_cancel_clip.setEnabled(True)
        except Exception:
            pass

        self.thread = QtCore.QThread(self)
        # RenderWorker signature: (cfg, resume_folder=None)
        self.worker = RenderWorker(cfg_obj, resume_folder)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)

        # Live per-clip overlay updates only make sense when rendering the live config (not a copy).
        try:
            if cfg_obj is self.cfg:
                self.worker.clip_state.connect(self.on_clip_overlay_state)
                self.worker.clip_progress.connect(self.on_clip_overlay_progress)
                self.worker.clip_asset.connect(self.on_clip_overlay_asset)
        except Exception:
            pass
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        # Ensure proper QObject cleanup (avoid "QThread destroyed while still running").
        try:
            self.worker.finished.connect(self.worker.deleteLater)
            self.worker.failed.connect(self.worker.deleteLater)
        except Exception:
            pass
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def render_selected_scenes(self, scene_indices: list[int]):
        """Render only selected clips (Premiere-style)."""
        self._sync_cfg_from_ui()
        if not scene_indices:
            return
        # clone cfg but only keep selected scenes
        import copy, time
        cfg2 = copy.deepcopy(self.cfg)
        scene_indices = sorted(set(int(i) for i in scene_indices if 0 <= int(i) < len(self.cfg.scenes)))
        cfg2.scenes = [copy.deepcopy(self.cfg.scenes[i]) for i in scene_indices]
        cfg2.project_name = f"{self.cfg.project_name}_SEL"
        # Ensure output dir exists
        os.makedirs(cfg2.output_dir, exist_ok=True)

        self.append_log(f"▶ Render sélection: {len(cfg2.scenes)} clip(s) — {cfg2.project_name}")
        self._start_worker_cfg(cfg2)


    def _start_clip_batch_worker(self, scene_indices: list[int], kind: str):
        """Start a batch render of independent clips (proxy/final)."""
        self._sync_cfg_from_ui()
        scene_indices = sorted(set(int(i) for i in scene_indices if 0 <= int(i) < len(self.cfg.scenes)))
        if not scene_indices:
            return
        kind = str(kind).lower().strip()
        if kind not in ('proxy', 'final'):
            raise ValueError(kind)

        # Avoid running two jobs at once
        if self.worker or getattr(self, 'clip_worker', None):
            QtWidgets.QMessageBox.warning(self, 'Render', 'Un rendu est déjà en cours.')
            return

        self.progress.setValue(0)
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        try:
            self.btn_cancel_clip.setEnabled(True)
        except Exception:
            pass

        self.clip_thread = QtCore.QThread(self)
        self.clip_worker = ClipBatchRenderWorker(self.cfg, scene_indices, kind)
        self.clip_worker.moveToThread(self.clip_thread)
        self.clip_thread.started.connect(self.clip_worker.run)
        self.clip_worker.progress.connect(self.on_progress)
        self.clip_worker.log.connect(self.append_log)
        self.clip_worker.finished.connect(self.on_clip_batch_finished)
        self.clip_worker.failed.connect(self.on_clip_batch_failed)
        # per-clip overlay
        self.clip_worker.clip_state.connect(self.on_clip_overlay_state)
        self.clip_worker.clip_progress.connect(self.on_clip_overlay_progress)
        self.clip_worker.clip_log.connect(self.on_clip_overlay_log)
        try:
            self.clip_worker.clip_asset.connect(self.on_clip_overlay_asset)
        except Exception:
            pass
        self.clip_worker.finished.connect(self.clip_thread.quit)
        self.clip_worker.failed.connect(self.clip_thread.quit)
        try:
            self.clip_worker.finished.connect(self.clip_worker.deleteLater)
            self.clip_worker.failed.connect(self.clip_worker.deleteLater)
        except Exception:
            pass
        self.clip_thread.finished.connect(self.clip_thread.deleteLater)
        self.clip_thread.start()

        self.append_log(f"▶ Render {kind}: {len(scene_indices)} clip(s)")

    def render_proxy_scenes(self, scene_indices: list[int]):
        """Render low-res proxies for selected clips and store SceneSpec.proxy_video_path."""
        self._start_clip_batch_worker(scene_indices, kind='proxy')

    def render_final_scenes(self, scene_indices: list[int]):
        """Render finals for selected clips and store SceneSpec.final_video_path."""
        self._start_clip_batch_worker(scene_indices, kind='final')

    def build_proxy_preview_for_scene(self, scene_idx: int, *, play_after: bool = True) -> None:
        """Build (or reuse) a per-clip MP4 preview from the clip cache.

        This is meant for the Premiere timeline: it lets you right-click a clip that was already
        rendered (or cache-hit) and instantly play it in the Program Monitor.
        """
        try:
            scene_idx = int(scene_idx)
        except Exception:
            return
        if scene_idx < 0 or scene_idx >= len(self.cfg.scenes):
            return

        sc = self.cfg.scenes[scene_idx]

        def _job():
            import os
            from .clip_cache import clip_cache_root, load_entry, find_latest_entry_for_scene
            from .encode import encode_png_sequence_to_mp4

            cache_root = clip_cache_root(self.cfg)
            key = getattr(sc, '_clip_cache_key', None) or getattr(sc, 'clip_cache_key', None)
            entry = None
            if key:
                try:
                    entry = load_entry(cache_root, str(key))
                except Exception:
                    entry = None
            if entry is None:
                entry = find_latest_entry_for_scene(cache_root, scene_idx, getattr(sc, 'label', None))
            if entry is None:
                raise RuntimeError("Aucun cache clip trouvé pour ce plan. (Rendu nécessaire ou cache désactivé)")

            # Store key for next time
            try:
                setattr(sc, '_clip_cache_key', str(entry.key))
            except Exception:
                pass

            entry_dir = os.path.dirname(str(entry.frames_dir or '').rstrip('/'))
            out_mp4 = os.path.join(entry_dir, 'preview.mp4')
            if os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 1024:
                return out_mp4

            crf = int(getattr(getattr(self.cfg, 'studio', None), 'clip_preview_crf', 23) or 23)
            preset = str(getattr(getattr(self.cfg, 'studio', None), 'clip_preview_preset', 'veryfast') or 'veryfast')
            fps = int(getattr(entry, 'fps', 0) or getattr(self.cfg, 'fps', 24) or 24)
            encode_png_sequence_to_mp4(str(entry.frames_dir), out_mp4, fps=fps, crf=crf, preset=preset)
            return out_mp4

        def _done(pth: str):
            try:
                pth = str(pth)
                if not pth:
                    return
                sc.proxy_video_path = pth
                # Make a stable thumbnail (last.png) if available.
                try:
                    from .clip_cache import clip_cache_root, load_entry
                    key = getattr(sc, '_clip_cache_key', None)
                    if key:
                        ent = load_entry(clip_cache_root(self.cfg), str(key))
                        if ent and ent.last_frame_path and os.path.exists(ent.last_frame_path):
                            sc.preview_frame_path = str(ent.last_frame_path)
                except Exception:
                    pass

                if hasattr(self, 'premiere') and self.premiere:
                    try:
                        self.premiere.reload_from_cfg(keep_selection=True)
                    except Exception:
                        pass
                    if play_after:
                        try:
                            self.premiere.load_preview(pth)
                        except Exception:
                            pass
            except Exception:
                pass

        self._run_generic_worker(_job, title='Clip preview', on_done=_done)

    @QtCore.Slot(object)
    def on_clip_batch_finished(self, payload: object):
        data = payload if isinstance(payload, dict) else {}
        kind = str(data.get('kind', ''))
        results = data.get('results', []) or []
        try:
            self.append_log(f"✅ {kind.upper()} terminé.")

            last_path = None
            for idx, path in results:
                idx = int(idx)
                path = str(path)
                last_path = path or last_path
                if idx < 0 or idx >= len(self.cfg.scenes):
                    continue
                if kind == 'proxy':
                    self.cfg.scenes[idx].proxy_video_path = path
                    try:
                        self._register_take(idx, kind='proxy', video_path=path)
                    except Exception:
                        pass
                elif kind == 'final':
                    self.cfg.scenes[idx].final_video_path = path
                    try:
                        self._register_take(idx, kind='final', video_path=path)
                    except Exception:
                        pass

            # refresh Studio tab UI
            try:
                if hasattr(self, 'premiere') and self.premiere is not None:
                    self.premiere.reload_from_cfg(keep_selection=True)
            except Exception:
                pass
            try:
                self._reload_scene_table()
            except Exception:
                pass

            if last_path:
                self._load_preview(self._select_preview_path(last_path))

            QtWidgets.QMessageBox.information(self, 'Terminé', f"{kind.upper()} terminé pour {len(results)} clip(s).")
        finally:
            # UI cleanup (never block with thread.wait() here)
            self.btn_start.setEnabled(True)
            self.btn_cancel.setEnabled(False)
            try:
                self.btn_cancel_clip.setEnabled(False)
            except Exception:
                pass
            self.status.setText('Terminé.')
            self.progress.setValue(1000)
            self.clip_thread = None
            self.clip_worker = None

    def _register_take(self, idx: int, *, kind: str, video_path: str) -> None:
        """Append a 'take' record on the scene and (optionally) generate a preview PNG.

        This powers the Retakes manager in the Studio inspector.
        """
        import os, time
        from .media import extract_frame_ffmpeg
        from .config import TakeSpec

        idx = int(idx)
        kind = str(kind).lower().strip() if kind else 'final'
        if kind not in ('proxy', 'final'):
            kind = 'custom'

        if not video_path:
            return
        if idx < 0 or idx >= len(self.cfg.scenes):
            return

        sc = self.cfg.scenes[idx]
        # Ensure takes list exists
        takes = list(getattr(sc, 'takes', []) or [])

        # Name: proxy_01 / final_01 ...
        n_same = 0
        for t in takes:
            if getattr(t, 'kind', '') == kind:
                n_same += 1
        name = f"{kind}_{n_same+1:02d}"

        # Preview PNG next to the video
        prev_png = None
        try:
            base, _ = os.path.splitext(video_path)
            prev_png_cand = base + "_preview.png"
            if not os.path.exists(prev_png_cand):
                r = extract_frame_ffmpeg(video_path, 0.05, prev_png_cand)
                if r.ok:
                    prev_png = r.path
            else:
                prev_png = prev_png_cand
        except Exception:
            prev_png = None

        created = time.strftime('%Y-%m-%d %H:%M:%S')
        seed = getattr(sc, 'seed', None)

        takes.append(TakeSpec(
            name=name,
            kind=kind,
            video_path=str(video_path),
            preview_path=str(prev_png) if prev_png else None,
            seed=int(seed) if seed is not None else None,
            created_at=created,
            note="",
        ))
        try:
            sc.takes = takes
        except Exception:
            setattr(sc, 'takes', takes)

        # Keep 'active' pointer for convenience
        try:
            if kind == 'proxy':
                sc.active_proxy_take = name
            elif kind == 'final':
                sc.active_final_take = name
        except Exception:
            pass

    @QtCore.Slot(str)
    def on_clip_batch_failed(self, err: str):
        try:
            self.append_log('❌ Erreur Proxy/Final:')
            self.append_log(err)
            QtWidgets.QMessageBox.critical(self, 'Erreur', err)
        finally:
            self.btn_start.setEnabled(True)
            self.btn_cancel.setEnabled(False)
            try:
                self.btn_cancel_clip.setEnabled(False)
            except Exception:
                pass
            self.status.setText('Erreur.')
            self.clip_thread = None
            self.clip_worker = None


    @QtCore.Slot(int, str, bool)
    def on_clip_overlay_state(self, idx: int, state: str, cache_hit: bool):
        try:
            idx = int(idx)
            if idx < 0 or idx >= len(self.cfg.scenes):
                return
            sc = self.cfg.scenes[idx]
            setattr(sc, '_render_state', str(state))
            setattr(sc, '_clip_cache_hit', bool(cache_hit))
            if hasattr(self, 'premiere') and self.premiere:
                self.premiere.set_clip_render_state(idx, str(state), cache_hit=bool(cache_hit))
        except Exception:
            pass

    @QtCore.Slot(int, float, str)
    def on_clip_overlay_progress(self, idx: int, p: float, msg: str):
        try:
            idx = int(idx)
            if idx < 0 or idx >= len(self.cfg.scenes):
                return
            sc = self.cfg.scenes[idx]
            setattr(sc, '_render_progress', float(p))
            setattr(sc, '_render_msg', str(msg))
            if hasattr(self, 'premiere') and self.premiere:
                self.premiere.set_clip_render_progress(idx, float(p), str(msg))
        except Exception:
            pass

    @QtCore.Slot(int, str, str)
    def on_clip_overlay_asset(self, idx: int, kind: str, path: str):
        """Live asset updates from the engine (init-frame / segment previews)."""
        try:
            idx = int(idx)
            if idx < 0 or idx >= len(self.cfg.scenes):
                return
            sc = self.cfg.scenes[idx]
            k = str(kind or '').strip().lower()
            p = str(path or '').strip()
            if not p:
                return
            if k == 'preview':
                try:
                    sc.preview_frame_path = p
                except Exception:
                    setattr(sc, 'preview_frame_path', p)
            elif k == 'init':
                try:
                    sc.init_frame_path = p
                except Exception:
                    setattr(sc, 'init_frame_path', p)
            elif k in ('proxy_video', 'proxy'):
                try:
                    sc.proxy_video_path = p
                except Exception:
                    setattr(sc, 'proxy_video_path', p)
                # Optional: record as a take (best-effort)
                try:
                    self._register_take(idx, kind='proxy', video_path=p)
                except Exception:
                    pass
            elif k in ('final_video', 'final'):
                try:
                    sc.final_video_path = p
                except Exception:
                    setattr(sc, 'final_video_path', p)
                try:
                    self._register_take(idx, kind='final', video_path=p)
                except Exception:
                    pass

            if hasattr(self, 'premiere') and self.premiere:
                try:
                    self.premiere.set_clip_asset(idx, k, p)
                except Exception:
                    pass
        except Exception:
            pass

    @QtCore.Slot(int, str)
    def on_clip_overlay_log(self, idx: int, msg: str):
        # Keep last message (used by overlay if needed)
        try:
            idx = int(idx)
            if idx < 0 or idx >= len(self.cfg.scenes):
                return
            sc = self.cfg.scenes[idx]
            setattr(sc, '_render_msg', str(msg))
        except Exception:
            pass


    def on_start(self):
        self._sync_cfg_from_ui()

        if self.cfg.mode == "I2V" and not self.cfg.input_image_path:
            # Allow I2V without a global init image when Auto Init-Frame is enabled
            # or when at least one scene provides its own per-clip input image override.
            init_cfg = getattr(self.cfg, "init_image", None)
            auto_ok = bool(init_cfg and getattr(init_cfg, "enabled", False))
            has_scene_override = False
            try:
                for sc in (getattr(self.cfg, "scenes", []) or []):
                    p = getattr(sc, "input_image_path_override", None) or ""
                    if p and os.path.exists(p):
                        has_scene_override = True
                        break
            except Exception:
                has_scene_override = False

            if not auto_ok and not has_scene_override:
                self.append_log("⚠️ I2V: aucune image initiale. Active Auto Init-Frame ou ajoute une image par clip.")
                QtWidgets.QMessageBox.warning(
                    self,
                    "I2V",
                    "Mode I2V: il faut une image initiale (PNG/JPG/WEBP, EXR converti, ou frame vidéo)\n"
                    "— ou activer Auto Init-Frame (Préférences) pour en générer une automatiquement.",
                )
                return

            if auto_ok:
                try:
                    preset = str(getattr(init_cfg, "preset", "") or "")
                except Exception:
                    preset = ""
                self.append_log(f"🧠 I2V: aucune image initiale fournie → Auto Init-Frame activé (preset: {preset}).")
            else:
                self.append_log("🧠 I2V: aucune image globale, utilisation des images override par clip.")

        os.makedirs(self.cfg.output_dir, exist_ok=True)

        # Use unified worker start (handles cleanup + per-clip timeline updates).
        self._start_worker_cfg(self.cfg, resume_folder=None)
        self.append_log("Render démarré...")

    def on_resume_from_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select render folder to resume", self.out_dir.text() or ".")
        if not folder:
            return
        self._start_worker_cfg(self.cfg, resume_folder=folder)
        self.append_log(f"Resume démarré: {folder}")

    def on_cancel_current_clip(self):
        if getattr(self, 'clip_worker', None):
            try:
                self.clip_worker.cancel_current_clip()
            except Exception:
                self.clip_worker.cancel()
            return
        if self.worker:
            self.worker.cancel()

    def on_cancel(self):
        if getattr(self, 'clip_worker', None):
            self.clip_worker.cancel()
            return
        if self.worker:
            self.worker.cancel()



    def on_clear_cache_now(self):
        """Clear init-frame + clip render caches for the current project."""

        def job():
            from .cache_utils import dir_size_bytes, clear_dir
            roots = []
            # Clip cache
            try:
                from .clip_cache import clip_cache_root
                roots.append(clip_cache_root(self.cfg))
            except Exception:
                pass
            # Init cache
            init_dir = None
            cache_json = None
            try:
                from .flux2_init import get_init_cache_dirs
                init_dir, cache_json = get_init_cache_dirs(self.cfg)
                if init_dir:
                    roots.append(init_dir)
            except Exception:
                pass

            # Dedup
            uniq = []
            seen = set()
            for r in roots:
                if not r:
                    continue
                ap = os.path.abspath(r)
                if ap in seen:
                    continue
                seen.add(ap)
                uniq.append(ap)

            before = 0
            for r in uniq:
                try:
                    before += int(dir_size_bytes(r))
                except Exception:
                    pass

            for r in uniq:
                try:
                    clear_dir(r)
                except Exception:
                    pass

            # Delete init cache json if it lives outside init_dir (project cache mode)
            try:
                if cache_json and os.path.exists(cache_json):
                    os.remove(cache_json)
            except Exception:
                pass

            after = 0
            for r in uniq:
                try:
                    after += int(dir_size_bytes(r))
                except Exception:
                    pass

            return (before, after, uniq)

        def on_done(res):
            try:
                before, after, uniq = res
                freed = max(0, before - after)
                self.append_log(f"[Cache] Cleared. Freed {freed/1e9:.2f}GB")
                msg = f"Cache cleared. Freed ~{freed/1e9:.2f}GB\n\nPaths:\n" + "\n".join(uniq)
                QtWidgets.QMessageBox.information(self, 'Cache', msg)
            except Exception:
                QtWidgets.QMessageBox.information(self, 'Cache', 'Cache cleared.')

        self._run_generic_worker(job, title='Clear cache', on_done=on_done)
    def on_clear_vram_now(self):
        """UI panic button: unload all heavy pipelines and clear CUDA cache."""

        def job():
            # Do not call Qt methods from this thread.
            try:
                self.engine.unload_all()
            except Exception:
                # Fallback (shouldn't happen)
                try:
                    from .resource_manager import get_resource_manager, cuda_cleanup
                    rm = get_resource_manager()
                    rm.unload(aggressive=True)
                    cuda_cleanup(aggressive=True)
                except Exception:
                    pass
            try:
                import torch
                if torch.cuda.is_available():
                    free_b, total_b = torch.cuda.mem_get_info()
                    return float(free_b) / (1024**3), float(total_b) / (1024**3)
            except Exception:
                pass
            return None

        def on_done(res):
            if res and isinstance(res, tuple):
                self.append_log(f"[VRAM] Cleared. Free {res[0]:.2f}/{res[1]:.2f}GB")
                try:
                    if getattr(self, "_tb_status", None) is not None:
                        self._tb_status.setText(f"VRAM {res[0]:.1f}/{res[1]:.1f}GB")
                except Exception:
                    pass
            else:
                self.append_log("[VRAM] Cleared.")
            self._update_vram()
            QtWidgets.QMessageBox.information(self, "VRAM", "Pipelines unloaded and VRAM cache cleared.")

        self._run_generic_worker(job, title="Clear VRAM", on_done=on_done)

    @QtCore.Slot(float, str)
    def on_progress(self, p: float, msg: str):
        self.progress.setValue(int(max(0.0, min(1.0, p)) * 1000))
        self.status.setText(msg)

    @QtCore.Slot(list)
    def on_finished(self, outs: list):
        self.append_log("✅ Terminé:")
        for p in outs:
            self.append_log(p)
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        try:
            self.btn_cancel_clip.setEnabled(False)
        except Exception:
            pass
        self.status.setText("Terminé.")
        self.progress.setValue(1000)

        if outs:
            self._load_preview(self._select_preview_path(outs[-1]))

        # Refresh Studio (Premiere) timeline widgets so rendered clips immediately
        # show updated icons/thumbnails (proxy/final/previews) without changing tabs.
        try:
            if hasattr(self, 'premiere') and self.premiere:
                self.premiere.reload_from_cfg(keep_selection=True)
        except Exception:
            pass

        # Optional post-step: mux master audio into the last output (no video re-encode)
        try:
            if outs and hasattr(self, 'cb_render_mux_audio') and self.cb_render_mux_audio.isChecked():
                video_in = outs[-1]
                preset = str(self.cb_render_mux_preset.currentData() or 'mkv_flac')

                base, ext = os.path.splitext(video_in)
                if preset == 'mp4_aac':
                    out_mux = base + '_AUDIO.mp4'
                    mux_kwargs = {"audio_codec": "aac", "aac_bitrate": "320k"}
                elif preset == 'mkv_pcm':
                    out_mux = base + '_AUDIO.mkv'
                    mux_kwargs = {"audio_codec": "pcm_s16le"}
                else:
                    out_mux = base + '_AUDIO.mkv'
                    mux_kwargs = {"audio_codec": "flac"}

                def _job_mux():
                    from .audio_pipeline import mux_master_audio
                    return mux_master_audio(video_in, self.cfg, self.cfg.output_dir, out_mux, **mux_kwargs)

                def _done_mux(pth):
                    try:
                        self.append_log(f"🔊 Audio muxed → {pth}")
                        # Load muxed file in preview
                        self._load_preview(self._select_preview_path(pth))
                    except Exception:
                        pass

                self._run_generic_worker(_job_mux, title='Mux audio', on_done=_done_mux)
        except Exception:
            pass


        # Thread lifecycle is handled by signals; never block the UI with thread.wait().
        try:
            if self.thread:
                self.thread.quit()
        except Exception:
            pass
        self.thread = None
        self.worker = None

        QtWidgets.QMessageBox.information(self, "Terminé", "Exports:\n" + "\n".join(outs))

    @QtCore.Slot(str)
    def on_failed(self, err: str):
        self.append_log("❌ Erreur:")
        self.append_log(err)
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        try:
            self.btn_cancel_clip.setEnabled(False)
        except Exception:
            pass
        self.status.setText("Erreur.")

        try:
            if self.thread:
                self.thread.quit()
        except Exception:
            pass
        self.thread = None
        self.worker = None

        QtWidgets.QMessageBox.critical(self, "Erreur", err)

    # ---------- file ops ----------
    def on_new(self):
        self.cfg = ProjectConfig()
        self._sync_ui_from_cfg()
        self.append_log("Nouveau projet.")

    def on_open(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Ouvrir projet", ".", "Projet Wan (*.json)")
        if not f:
            return
        try:
            self.cfg = ProjectConfig.load(f)
            try:
                from .voice_manager import assign_missing_character_voices
                n = assign_missing_character_voices(self.cfg)
                if n:
                    self.append_log(f"Voices auto-assign: {n} personnage(s).")
            except Exception:
                pass

            self._sync_ui_from_cfg()
            self.append_log(f"Projet chargé: {f}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erreur", str(e))

    def on_save(self):
        f, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Sauver projet", ".", "Projet Wan (*.json)")
        if not f:
            return
        if not f.lower().endswith(".json"):
            f += ".json"
        try:
            self._sync_cfg_from_ui()
            self.cfg.save(f)
            self.append_log(f"Projet sauvé: {f}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erreur", str(e))

    # ---------- studio tools ----------

    def open_character_bible(self):
        try:
            from .library_editors import CharacterBibleDialog
            d = CharacterBibleDialog(self, items=list(getattr(self.cfg, "characters", []) or []))
            if d.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                self.cfg.characters = d.get_items()
                try:
                    from .voice_manager import assign_missing_character_voices
                    n = assign_missing_character_voices(self.cfg)
                    if n:
                        self.append_log(f"Voices auto-assign: {n} personnage(s).")
                except Exception:
                    pass
                try:
                    if hasattr(self, "premiere") and self.premiere:
                        self.premiere._refresh_phasec_lists()
                except Exception:
                    pass
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Character Bible", str(e))

    def open_locations(self):
        try:
            from .library_editors import LocationLibraryDialog
            d = LocationLibraryDialog(self, items=list(getattr(self.cfg, "locations", []) or []))
            if d.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                self.cfg.locations = d.get_items()
                try:
                    if hasattr(self, "premiere") and self.premiere:
                        self.premiere._refresh_phasec_lists()
                except Exception:
                    pass
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Locations", str(e))

    def open_init_image_settings(self):
        """Global init-image settings (Flux2/Flux1/SDXL, policy, adapters)."""
        try:
            from .settings_dialogs import InitImageSettingsDialog
            d = InitImageSettingsDialog(self, cfg=self.cfg)
            if d.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                d.apply_to_cfg()
                try:
                    if hasattr(self, "premiere") and self.premiere:
                        # refresh inspector cache labels, etc.
                        self.premiere.refresh_all()
                except Exception:
                    pass
                self.append_log("Init Image settings updated.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Init Image Settings", str(e))

    def open_cache_manager(self):
        try:
            from .settings_dialogs import CacheManagerDialog
            d = CacheManagerDialog(self, cfg=self.cfg)
            d.exec()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Cache Manager", str(e))

    def open_prompt_builder(self):
        d = PromptBuilderDialog(self)
        d.exec()

    def open_ai_director(self):
        try:
            d = DirectorDialog(self, current_cfg=self.cfg)
            d.applied.connect(self._apply_ai_scenes)
            d.exec()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "AI Director", str(e))




    def _apply_ai_scenes(self, scenes):
        """Apply storyboard scenes while keeping object identity when possible.
    
        - Updates existing SceneSpec objects in-place (timeline/audio references stay valid)
        - Respects per-scene locks: lock_prompt, lock_dialogue, lock_cast, lock_location, lock_shot, lock_music
        - Ensures newly referenced characters/locations are added to the project libraries
        - Auto-assigns missing voices when possible
        """
        try:
            incoming = list(scenes or [])
    
            if not self.cfg.scenes:
                self.cfg.scenes = incoming
            else:
                old = list(self.cfg.scenes)
    
                by_label = {}
                for s in old:
                    lab = getattr(s, "label", None)
                    if lab:
                        by_label.setdefault(str(lab), []).append(s)
    
                new_list = []
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
    
                self.cfg.scenes = new_list
    
            # Ensure Character Bible / Location Library contain any newly referenced names
            try:
                from .config import CharacterSpec, LocationSpec
                added_c = 0
                added_l = 0
                chars = getattr(self.cfg, "characters", None) or []
                locs = getattr(self.cfg, "locations", None) or []
                existing_c = { (getattr(c, "name", "") or "").strip().lower(): c for c in chars if (getattr(c, "name", "") or "").strip() }
                existing_l = { (getattr(l, "name", "") or "").strip().lower(): l for l in locs if (getattr(l, "name", "") or "").strip() }
    
                for s in (self.cfg.scenes or []):
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
    
                self.cfg.characters = chars
                self.cfg.locations = locs
                if added_c or added_l:
                    self.append_log(f"Bible auto: +{added_c} personnage(s), +{added_l} lieu(x).")
            except Exception:
                pass
    
            # Auto-assign missing character voices
            try:
                from .voice_manager import assign_missing_character_voices
                n = assign_missing_character_voices(self.cfg)
                if n:
                    self.append_log(f"Voices auto-assign: {n} personnage(s).")
            except Exception:
                pass
    
            self._sync_ui_from_cfg()
            if hasattr(self, "premiere") and self.premiere:
                try:
                    self.premiere.reload_from_cfg(keep_selection=True)
                except TypeError:
                    self.premiere.reload_from_cfg()
    
            if hasattr(self, "audio_scene_list"):
                try:
                    self._refresh_audio_scene_list(keep_selection=True)
                except Exception:
                    pass
    
            try:
                self._maybe_auto_audio_after_storyboard_apply()
            except Exception:
                pass
    
        except Exception as e:
            try:
                self.append_log(f"Apply storyboard failed: {e}")
            except Exception:
                pass
    
    def _regen_init_frame_for_scene(self, scene, use_cache_only: bool = False):
        """Generate/refresh an init frame for a clip.

        Prefers the automatic init-frame generator (Flux2/Flux1/SDXL presets) if enabled;
        falls back to the manual T2I dialog.

        If use_cache_only=True, it will *only* use the cached init frame (no generation).
        """
        try:
            from .flux2_init import generate_init_image_for_scene, try_use_cached_init_frame
            cfg = self.cfg
            if bool(getattr(cfg, "init_image", None) and getattr(cfg.init_image, "enabled", False)):
                if use_cache_only:
                    res = try_use_cached_init_frame(scene, cfg, log=self.append_log)
                else:
                    res = generate_init_image_for_scene(scene, cfg, force=True, log=self.append_log)
                path = getattr(res, "path", None) if res is not None else None
                if path:
                    try:
                        scene.init_frame_path = path
                        scene.input_image_path_override = path
                    except Exception:
                        pass
                    try:
                        if hasattr(self, "premiere") and self.premiere:
                            self.premiere.reload_from_cfg(keep_selection=True)
                    except Exception:
                        pass
                    return path
        except Exception:
            pass

        # Fallback: manual dialog
        return self._ai_generate_init_image_for_scene(scene)


    def _get_init_cache_status_for_scene(self, scene):
        """Return InitCacheStatus for the clip (used by Premiere inspector)."""
        try:
            from .flux2_init import get_init_cache_status
            if bool(getattr(self.cfg, "init_image", None) and getattr(self.cfg.init_image, "enabled", False)):
                return get_init_cache_status(scene, self.cfg, log=self.append_log)
        except Exception:
            return None
        return None

    def _use_cached_init_frame_for_scene(self, scene):
        """Force cache usage for init frame (no generation)."""
        try:
            return self._regen_init_frame_for_scene(scene, use_cache_only=True)
        except Exception as e:
            try:
                self.append_log(f"[InitFrame] cache miss: {e}")
            except Exception:
                pass
            raise


    def _ai_generate_init_image_for_scene(self, scene):
        """Generate an init image (T2I) and set it as I2V input override for a clip."""
        try:
            # Default to the clip prompt, but user can rewrite.
            default_prompt = getattr(scene, 'prompt', '') or ''
            out_root = os.path.join(self.out_dir.text().strip() or './outputs', 'assets', 'init_images')
            os.makedirs(out_root, exist_ok=True)
            init_cfg = getattr(self.cfg, 'init_image', None)
            # Keep the manual init dialog consistent with global init-frame settings.
            default_preset = getattr(init_cfg, 'preset', 'zimage_turbo') if init_cfg else 'zimage_turbo'
            from .init_frame_subprocess import clamp_to_multiple_of_8
            default_w = int((getattr(init_cfg, 'width', 0) if init_cfg else 0) or getattr(self.cfg, 'width', 1024) or 1024)
            default_h = int((getattr(init_cfg, 'height', 0) if init_cfg else 0) or getattr(self.cfg, 'height', 1024) or 1024)
            default_w, default_h = clamp_to_multiple_of_8(default_w, default_h)
            default_steps = int(getattr(init_cfg, 'steps', 28) if init_cfg else 28)
            default_gs = float(getattr(init_cfg, 'guidance_scale', 4.5) if init_cfg else 4.5)

            from .init_frame_subprocess import resolve_hf_cache_dir
            dlg = ImageInitDialog(
                self,
                default_prompt=default_prompt,
                default_negative=getattr(scene, 'negative_prompt', '') or '',
                default_preset_key=default_preset,
                default_width=default_w,
                default_height=default_h,
                default_steps=default_steps,
                default_guidance=default_gs,
                out_root=out_root,
                cache_dir=resolve_hf_cache_dir(self.cfg),
            )
            if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return None
            path = getattr(getattr(dlg, "result", None), "path", None)
            if not path:
                return None

            scene.input_image_path_override = path
            # Refresh UI surfaces.
            try:
                self._reload_scene_table()
            except Exception:
                pass
            try:
                if hasattr(self, 'premiere') and self.premiere:
                    self.premiere.reload_from_cfg(keep_selection=True)
            except Exception:
                pass
            try:
                if hasattr(self, '_refresh_audio_scene_list'):
                    self._refresh_audio_scene_list(keep_selection=True)
            except Exception:
                pass
            self.append_log(f"Init image générée → {path}")
            return path
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'AI Init Image', str(e))
            return None

    def open_last_report(self):
        out_dir = self.out_dir.text().strip() or "./outputs"
        newest = None
        newest_mtime = -1
        for root, dirs, files in os.walk(out_dir):
            if "report.html" in files:
                p = os.path.join(root, "report.html")
                try:
                    mt = os.path.getmtime(p)
                    if mt > newest_mtime:
                        newest_mtime = mt
                        newest = p
                except Exception:
                    pass
        if newest:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(os.path.abspath(newest)))
        else:
            QtWidgets.QMessageBox.information(self, "Report", "Aucun report.html trouvé (rendu pas encore terminé?)")

    def _autosave(self):
        try:
            self._sync_cfg_from_ui()
            path = os.path.join(os.path.abspath("."), "autosave_wan_studio.json")
            self.cfg.save(path)
        except Exception:
            pass

    # ---------- presets ----------
    def apply_preset_1080p(self):
        self.cfg.width = 960; self.cfg.height = 544; self.cfg.fps = 24
        self.cfg.no_overlap_strict = True
        self.cfg.chunk_seconds = 5.0; self.cfg.overlap_frames = 0
        self.cfg.num_inference_steps = 14
        self.cfg.guidance_scale = 3.5; self.cfg.guidance_scale_2 = 2.5; self.cfg.boundary_ratio = 0.90
        self.cfg.blend_mode = "cut"
        self.cfg.scene_transition_mode = "cut"
        self.cfg.scene_transition_frames = 0
        self.cfg.scene_transition_ease = "smoothstep"
        self.cfg.post.enabled = True
        self.cfg.post.target_fps = 48
        self.cfg.post.out_width = 1920; self.cfg.post.out_height = 1080
        self.cfg.post.export_master_prores = True
        self.cfg.post.export_delivery_h265_main10 = True
        self.cfg.studio.write_report = True
        self.cfg.studio.export_shotlist_csv = True
        self.cfg.studio.crash_safe_resume = True
        self._sync_ui_from_cfg()
        self.append_log("Preset appliqué: 1080p Premium (2x + 48fps)")

    def apply_preset_draft(self):
        # Ultra-quick low-res preview for prompt validation (minimal VRAM)
        self.cfg.width = 512; self.cfg.height = 288; self.cfg.fps = 12
        self.cfg.no_overlap_strict = True
        self.cfg.chunk_seconds = 2.5; self.cfg.overlap_frames = 0
        self.cfg.num_inference_steps = 8
        # Max compatibility: disable boundary/guidance2 if pipeline doesn't support it
        self.cfg.guidance_scale = 4.0
        self.cfg.guidance_scale_2 = 0.0
        self.cfg.boundary_ratio = 0.0
        self.cfg.blend_mode = "cut"
        self.cfg.post.enabled = False
        self._sync_ui_from_cfg()
        self.append_log("Preset appliqué: Draft Preview (Low Res)")

    def apply_preset_fast(self):
        self.cfg.width = 832; self.cfg.height = 480; self.cfg.fps = 16
        self.cfg.no_overlap_strict = True
        self.cfg.chunk_seconds = 4.0; self.cfg.overlap_frames = 0
        self.cfg.num_inference_steps = 10
        self.cfg.guidance_scale = 3.0; self.cfg.guidance_scale_2 = 2.0; self.cfg.boundary_ratio = 0.90
        self.cfg.blend_mode = "cut"
        self.cfg.post.enabled = False
        self._sync_ui_from_cfg()
        self.append_log("Preset appliqué: Fast Preview")

    def apply_preset_svi(self):
        self.cfg.width = 960; self.cfg.height = 544; self.cfg.fps = 24
        self.cfg.no_overlap_strict = True
        self.cfg.chunk_seconds = 5.0; self.cfg.overlap_frames = 0
        self.cfg.num_inference_steps = 12
        self.cfg.guidance_scale = 3.5; self.cfg.guidance_scale_2 = 2.5; self.cfg.boundary_ratio = 0.90
        self.cfg.blend_mode = "cut"
        self.cfg.scene_transition_mode = "cut"
        self.cfg.scene_transition_frames = 0
        self.cfg.scene_transition_ease = "smoothstep"
        self.cfg.vary_seed_per_segment = True
        self._sync_ui_from_cfg()
        self.append_log("Preset appliqué: SVI Longform (stable)")

    # ---------- helpers ----------
    def _choose_out_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choisir dossier", self.out_dir.text() or ".")
        if d:
            self.out_dir.setText(d)

    def _choose_character_image(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose character reference",
            ".",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)",
        )
        if f:
            self.char_img.setText(f)

    def _apply_quality_preset(self):
        preset = self.quality.currentText().strip()
        # --- Agency demo presets (LingBot, 4090/24GB) ---
        if preset == "LingBot Cine A (4090/24GB)":
            # Force LingBot backend + ultra-safe base resolution (Profile A).
            try:
                if hasattr(self, "mode"):
                    self.mode.setCurrentText("T2V")
                if hasattr(self, "backend"):
                    self.backend.setCurrentText("lingbot")
            except Exception:
                pass
            try:
                self.fps.setValue(24)
                self.w.setValue(832)
                self.h.setValue(480)
                # Keep global steps for consistency (used by non-LingBot backends).
                self.steps.setValue(22)
                self.cfg1.setValue(4.8)
                self.chunk_s.setValue(4.0)
            except Exception:
                pass
            try:
                if hasattr(self, 'cb_cinema_safe'):
                    self.cb_cinema_safe.setChecked(True)
                if hasattr(self, 'cb_no_overlap_strict'):
                    self.cb_no_overlap_strict.setChecked(True)
                if hasattr(self, 'overlap'):
                    self.overlap.setValue(0)
                try:
                    self.blend.setCurrentText('cut')
                except Exception:
                    pass
            except Exception:
                pass
            # Style pack (optional)
            try:
                if hasattr(self, 'global_style'):
                    self.global_style.setCurrentText('Cinematic Ultra (Photoreal Max)')
            except Exception:
                pass
            # Configure LingBot sampler (no dedicated UI yet).
            try:
                ling = getattr(self.cfg, "lingbot", None)
                if ling is not None:
                    ling.enabled = True
                    ling.size = "832*480"
                    ling.sample_steps = 24
                    ling.sample_shift = 3.0
                    ling.sample_guide_scale = 5.0
                    ling.offload_model = True
                    ling.t5_cpu = True
                    ling.ulysses_size = 1
            except Exception:
                pass

        elif preset == "Draft super fast":
            # Fastest playable demo preset: fewer LingBot steps + safe resolution.
            try:
                if hasattr(self, "mode"):
                    self.mode.setCurrentText("T2V")
                if hasattr(self, "backend"):
                    self.backend.setCurrentText("lingbot")
            except Exception:
                pass
            try:
                self.fps.setValue(24)
                self.w.setValue(832)
                self.h.setValue(480)
                self.steps.setValue(12)
                self.cfg1.setValue(4.0)
                # Slightly longer chunks => fewer segments for quicker iteration
                self.chunk_s.setValue(6.0)
            except Exception:
                pass
            try:
                if hasattr(self, 'cb_cinema_safe'):
                    self.cb_cinema_safe.setChecked(True)
                if hasattr(self, 'cb_no_overlap_strict'):
                    self.cb_no_overlap_strict.setChecked(True)
                if hasattr(self, 'overlap'):
                    self.overlap.setValue(0)
                try:
                    self.blend.setCurrentText('cut')
                except Exception:
                    pass
            except Exception:
                pass
            try:
                ling = getattr(self.cfg, "lingbot", None)
                if ling is not None:
                    ling.enabled = True
                    ling.size = "832*480"
                    ling.sample_steps = 12
                    ling.sample_shift = 3.0
                    ling.sample_guide_scale = 4.0
                    ling.offload_model = True
                    ling.t5_cpu = True
                    ling.ulysses_size = 1
            except Exception:
                pass

        # Safe defaults for 24GB GPUs + CPU offload
        if preset == "Draft Preview (fast)":
            self.fps.setValue(12)
            self.w.setValue(640)
            self.h.setValue(360)
            self.steps.setValue(10)
            self.cfg1.setValue(3.5)
            self.chunk_s.setValue(2.5)
            if hasattr(self, 'cb_no_overlap_strict'):
                self.cb_no_overlap_strict.setChecked(True)
            self.overlap.setValue(0)
            try:
                self.blend.setCurrentText('cut')
            except Exception:
                pass
        elif preset == "LookDev (balanced)":
            self.fps.setValue(24)
            self.w.setValue(832)
            self.h.setValue(480)
            self.steps.setValue(14)
            self.cfg1.setValue(4.0)
            self.chunk_s.setValue(4.0)
            if hasattr(self, 'cb_no_overlap_strict'):
                self.cb_no_overlap_strict.setChecked(True)
            self.overlap.setValue(0)
            try:
                self.blend.setCurrentText('cut')
            except Exception:
                pass
        elif preset == "Final (base+upscale)":
            self.fps.setValue(24)
            self.w.setValue(960)
            self.h.setValue(544)
            self.steps.setValue(16)
            self.cfg1.setValue(4.5)
            self.chunk_s.setValue(5.0)
            if hasattr(self, 'cb_no_overlap_strict'):
                self.cb_no_overlap_strict.setChecked(True)
            self.overlap.setValue(0)
            try:
                self.blend.setCurrentText('cut')
            except Exception:
                pass
        elif preset == "Cinema Ultra (photoreal max)":
            # High-quality but VRAM-safe settings. Uses the new Photoreal style pack.
            self.fps.setValue(24)
            self.w.setValue(960)
            self.h.setValue(544)
            self.steps.setValue(22)
            self.cfg1.setValue(4.8)
            self.chunk_s.setValue(4.0)
            # Hard-cut stitching (no overlap blending)
            if hasattr(self, 'cb_no_overlap_strict'):
                self.cb_no_overlap_strict.setChecked(True)
            self.overlap.setValue(0)
            try:
                self.blend.setCurrentText('cut')
            except Exception:
                pass
            # Global style pack for better realism
            try:
                if hasattr(self, 'global_style'):
                    self.global_style.setCurrentText('Cinematic Ultra (Photoreal Max)')
            except Exception:
                pass
            # Post polish (light)
            try:
                if hasattr(self, 'post_enabled'):
                    self.post_enabled.setChecked(True)
                if hasattr(self, 'deflicker'):
                    self.deflicker.setChecked(True)
                if hasattr(self, 'denoise'):
                    self.denoise.setChecked(True)
                if hasattr(self, 'sharpen'):
                    self.sharpen.setChecked(False)
                if hasattr(self, 'out_w') and hasattr(self, 'out_h'):
                    self.out_w.setValue(0); self.out_h.setValue(0)
            except Exception:
                pass

        elif preset == "Cinema Final+ (deflicker/denoise)":
            # Final master settings: stronger polish + optional upscale via tools if available.
            self.fps.setValue(24)
            self.w.setValue(960)
            self.h.setValue(544)
            self.steps.setValue(24)
            self.cfg1.setValue(5.0)
            self.chunk_s.setValue(4.5)
            if hasattr(self, 'cb_no_overlap_strict'):
                self.cb_no_overlap_strict.setChecked(True)
            self.overlap.setValue(0)
            try:
                self.blend.setCurrentText('cut')
            except Exception:
                pass
            try:
                if hasattr(self, 'global_style'):
                    self.global_style.setCurrentText('Cinematic Ultra (Photoreal Max)')
            except Exception:
                pass
            try:
                if hasattr(self, 'post_enabled'):
                    self.post_enabled.setChecked(True)
                if hasattr(self, 'deflicker'):
                    self.deflicker.setChecked(True)
                if hasattr(self, 'denoise'):
                    self.denoise.setChecked(True)
                if hasattr(self, 'sharpen'):
                    self.sharpen.setChecked(True)
                # If tools are installed, upscale to 1080p; otherwise leave 0 (no upscale)
                if hasattr(self, 'out_w') and hasattr(self, 'out_h'):
                    self.out_w.setValue(1920)
                    self.out_h.setValue(1080)
                if hasattr(self, 'use_re'):
                    self.use_re.setChecked(True)
            except Exception:
                pass
        elif preset == "Cinema 4K Master (2-stage)":
            # 2-stage workflow: render at a higher-but-safe base resolution, then upscale to 4K in post.
            self.fps.setValue(24)
            # Safe "max native" baseline for 24GB-class GPUs (avoids OOM while staying sharper than 960x544).
            self.w.setValue(1152)
            self.h.setValue(648)
            self.steps.setValue(22)
            self.cfg1.setValue(4.8)
            self.chunk_s.setValue(4.0)
            if hasattr(self, 'cb_no_overlap_strict'):
                self.cb_no_overlap_strict.setChecked(True)
            self.overlap.setValue(0)
            try:
                self.blend.setCurrentText('cut')
            except Exception:
                pass
            # Photoreal style pack
            try:
                if hasattr(self, 'global_style'):
                    self.global_style.setCurrentText('Cinematic Ultra (Photoreal Max)')
            except Exception:
                pass
            # Post: upscale to 4K + mild polish, exports on
            try:
                if hasattr(self, 'post_enabled'):
                    self.post_enabled.setChecked(True)
                if hasattr(self, 'deflicker'):
                    self.deflicker.setChecked(True)
                if hasattr(self, 'denoise'):
                    self.denoise.setChecked(True)
                if hasattr(self, 'sharpen'):
                    self.sharpen.setChecked(False)
                if hasattr(self, 'target_fps'):
                    self.target_fps.setValue(0)  # keep original FPS (avoid heavy interpolation by default)
                if hasattr(self, 'out_w') and hasattr(self, 'out_h'):
                    self.out_w.setValue(3840)
                    self.out_h.setValue(2160)
                if hasattr(self, 'use_re'):
                    self.use_re.setChecked(True)
                if hasattr(self, 'export_prores'):
                    self.export_prores.setChecked(True)
                if hasattr(self, 'export_h265'):
                    self.export_h265.setChecked(True)
                if hasattr(self, 'primary_deliver'):
                    self.primary_deliver.setCurrentText('prores')
                if hasattr(self, 'prores_profile'):
                    self.prores_profile.setCurrentText('422hq')
            except Exception:
                pass

        else:
            self.append_log("Quality preset: Custom (no changes).")
            return
        self.append_log(f"Quality preset applied: {preset}")
        # Persist last used preset (helps demo consistency between runs).
        try:
            if getattr(self, "_settings", None) is not None:
                self._settings.setValue("ui/quality_preset_last", preset)
        except Exception:
            pass

    def _choose_input_image(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choisir image", ".", "Images (*.png *.jpg *.jpeg *.webp)")
        if f:
            self.input_img.setText(f)

    def _choose_tools_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Tools dir", self.tools_dir.text() or ".")
        if d:
            self.tools_dir.setText(d)
    def _on_master_res_changed(self, *_):
        """Quick-set out_w/out_h from the 'Master target' dropdown."""
        try:
            if not hasattr(self, "master_res"):
                return
            val = self.master_res.currentData()
            if not val:
                return
            parts = str(val).lower().split("x")
            if len(parts) != 2:
                return
            w = int(float(parts[0] or 0))
            h = int(float(parts[1] or 0))
            if hasattr(self, "out_w") and hasattr(self, "out_h"):
                self.out_w.setValue(max(0, w))
                self.out_h.setValue(max(0, h))
        except Exception:
            pass


    def append_log(self, s: str):
        try:
            self.log_box.appendPlainText(s)
            self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())
        except Exception:
            pass
        # mirror into dock log if present
        try:
            if hasattr(self, "log_dock_box") and self.log_dock_box is not None:
                self.log_dock_box.appendPlainText(s)
                self.log_dock_box.verticalScrollBar().setValue(self.log_dock_box.verticalScrollBar().maximum())
        except Exception:
            pass

    def _update_vram(self):
        try:
            import torch
            if not torch.cuda.is_available():
                return
            free, total = torch.cuda.mem_get_info()
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            base = self.status.text().split("|")[0].strip()
            self.status.setText(f"{base} | VRAM {free_gb:.1f}/{total_gb:.1f}GB")
            try:
                if getattr(self, "_tb_status", None) is not None:
                    self._tb_status.setText(f"VRAM {free_gb:.1f}/{total_gb:.1f}GB")
            except Exception:
                pass
        except Exception:
            pass

    def _open_output_dir(self):
        d = self.out_dir.text().strip() or "."
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(os.path.abspath(d)))

    def _reload_last_output(self):
        """Reload the last rendered video into the preview widgets (if any)."""
        p = getattr(self, "_last_output_video", None)
        if not p:
            self.append_log("Aucune sortie précédente à recharger.")
            return
        self._load_preview(p)

    def _extract_video_frame_to_png(self, video_path: str, t_seconds: float) -> str:
        """Extract a frame from a video using ffmpeg and return the PNG path."""
        import os, subprocess, time
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)
        out_dir = os.path.join(self.cfg.output_dir, "_extracted_frames")
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_png = os.path.join(out_dir, f"frame_{ts}_{int(t_seconds*1000):07d}ms.png")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(float(t_seconds)),
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            out_png,
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0 or (not os.path.exists(out_png)):
            raise RuntimeError("ffmpeg failed:\n" + (p.stderr[-2000:] if p.stderr else ""))
        return out_png



    def _open_last_output_file(self):
        p = getattr(self, "_last_output_video", None)
        if not p:
            return
        try:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(os.path.abspath(p)))
        except Exception:
            pass




    def _select_preview_path(self, final_path: str) -> str:
        """Choose a preview-friendly media file. Prefer mp4 if final is ProRes .mov."""
        try:
            p = os.path.abspath(final_path)
            if p.lower().endswith(".mov"):
                base_dir = os.path.dirname(p)
                candidates = [
                    # Studio delivery pack (new)
                    os.path.join(base_dir, "delivery_review_h265.mp4"),
                    os.path.join(base_dir, "delivery_review_h265_main10.mp4"),
                    os.path.join(base_dir, "deliver_1080p_h265_main10.mp4"),
                    os.path.join(base_dir, "post", "upscaled.mp4"),
                    os.path.join(base_dir, "post", "interpolated.mp4"),
                    os.path.join(base_dir, "post", "polished.mp4"),
                    os.path.join(base_dir, "raw.mp4"),
                ]
                for c in candidates:
                    if os.path.exists(c):
                        return c
            return p
        except Exception:
            return final_path

    def _load_preview(self, path: str):
        try:
            url = QtCore.QUrl.fromLocalFile(os.path.abspath(path))
            self.player.setSource(url)
            self.player.play()
            # also load into Preview dock
            try:
                if hasattr(self, "preview_player") and self.preview_player is not None:
                    self.preview_player.setSource(url)
                    self.preview_player.play()
                    self._last_output_video = os.path.abspath(path)
                    if hasattr(self, "preview_info"):
                        self.preview_info.setText(f"Last output: {self._last_output_video}")
                    try:
                        if hasattr(self, "tl_last_info"):
                            self.tl_last_info.setText(self._last_output_video)
                        # and into Premiere-style monitor (Studio tab)
                        try:
                            if hasattr(self, "premiere") and self.premiere is not None:
                                self.premiere.load_preview(self._last_output_video)
                        except Exception:
                            pass
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass

    def _set_preview(self, label: QtWidgets.QLabel, image_path: str):
        try:
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((600, 320))
            import io
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            qimg = QtGui.QImage.fromData(buf.getvalue(), "PNG")
            pix = QtGui.QPixmap.fromImage(qimg)
            label.setPixmap(pix)
        except Exception:
            label.setText("Preview: (failed)")

    def _on_dur(self, dur: int):
        self.slider.setRange(0, max(1, dur))

    def _on_pos(self, pos: int):
        self.slider.blockSignals(True)
        self.slider.setValue(pos)
        self.slider.blockSignals(False)

    def _on_seek(self, pos: int):
        self.player.setPosition(pos)

    def _cell_text(self, table: QtWidgets.QTableWidget, r: int, c: int, default: str) -> str:
        it = table.item(r, c)
        return it.text() if it is not None else default

    def _cell_float(self, table: QtWidgets.QTableWidget, r: int, c: int, default: float) -> float:
        try:
            return float(self._cell_text(table, r, c, str(default)).strip())
        except Exception:
            return default

    def _cell_float_opt(self, table: QtWidgets.QTableWidget, r: int, c: int) -> Optional[float]:
        txt = self._cell_text(table, r, c, "").strip()
        if not txt:
            return None
        try:
            return float(txt)
        except Exception:
            return None

    def _cell_int_opt(self, table: QtWidgets.QTableWidget, r: int, c: int) -> Optional[int]:
        txt = self._cell_text(table, r, c, "").strip()
        if not txt:
            return None
        try:
            return int(float(txt))
        except Exception:
            return None


# --- Hotfix: make UI resilient to missing methods (stale __pycache__ / bad indent)

# Some users end up running a stale build where a few MainWindow methods were accidentally
# mis-indented and therefore not attached to the class. Instead of crashing at startup,
# we inject small fallbacks for the UI callbacks.

def _make_missing_ui_cb(name: str):
    def _missing(self, *args, **kwargs):
        msg = f"Fonction manquante: {name}"
        try:
            self.append_log(f"[UI] {msg}")
        except Exception:
            pass
        try:
            QtWidgets.QMessageBox.warning(self, "Fonction manquante", msg)
        except Exception:
            pass
    return _missing

# Prefer real fallbacks for a couple of common actions

def _fallback_reload_last_output(self):
    p = getattr(self, "_last_output_video", None)
    if not p:
        try:
            self.append_log("Aucune sortie précédente à recharger.")
        except Exception:
            pass
        return
    try:
        self._load_preview(p)
    except Exception:
        pass


def _fallback_choose_out_dir(self):
    try:
        cur = "."
        try:
            if hasattr(self, "out_dir") and self.out_dir is not None:
                cur = self.out_dir.text() or "."
        except Exception:
            pass
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choisir dossier", cur)
        if d and hasattr(self, "out_dir"):
            self.out_dir.setText(d)
    except Exception:
        pass


_REQUIRED_UI_METHODS = {
    "_reload_last_output",
    "_choose_out_dir",
    # other callbacks referenced by .clicked.connect(self._xxx)
    "_apply_inspector",
    "_apply_quality_preset",
    "_apply_quick_preset",
    "_autofill_labels",
    "_choose_character_image",
    "_choose_input_image",
    "_choose_input_image_quick",
    "_choose_tools_dir",
    "_copy",
    "_download_selected_model",
    "_extract_frame_modeltab",
    "_extract_from_video_quick",
    "_gen",
    "_import_exr_modeltab",
    "_import_exr_quick",
    "_open_last_output_file",
    "_open_output_dir",
    "_prompt_builder_to_quick",
    "_render_from_quick",
    "_verify_selected_model_cache",
}

try:
    if not hasattr(MainWindow, "_reload_last_output"):
        MainWindow._reload_last_output = _fallback_reload_last_output
    if not hasattr(MainWindow, "_choose_out_dir"):
        MainWindow._choose_out_dir = _fallback_choose_out_dir

    for _name in sorted(_REQUIRED_UI_METHODS):
        if not hasattr(MainWindow, _name):
            setattr(MainWindow, _name, _make_missing_ui_cb(_name))
except Exception:
    pass

def main():
    app = QtWidgets.QApplication([])
    app.setApplicationName("Wan Studio Ultimate+")

    # Theme: modern by default (no heavy deps). Optional override:
    #   WAN_THEME=qdark  -> use qdarkstyle if installed
    #   WAN_THEME=modern -> force built-in modern theme
    theme_mode = (os.environ.get("WAN_THEME", "modern") or "modern").strip().lower()
    if theme_mode in ("qdark", "qdarkstyle") and qdarkstyle is not None:
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyside6())
    else:
        try:
            from .theme import apply_modern_theme
            apply_modern_theme(app)
        except Exception:
            app.setStyle("Fusion")

    win = MainWindow()
    win.show()
    app.exec()
