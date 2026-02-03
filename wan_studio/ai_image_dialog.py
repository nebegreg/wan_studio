from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict

from PySide6 import QtWidgets, QtCore

from .init_frame_subprocess import build_job, run_init_frame_job, resolve_hf_cache_dir, clamp_to_multiple_of_8


# -----------------------------------------------------------------------------
# Backward-compatible dialog expected by gui.py
# -----------------------------------------------------------------------------


@dataclass
class AiImageRequest:
    model_id: str
    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    guidance: float
    seed: Optional[int]
    dtype: str = "auto"  # "auto" | "fp16" | "bf16"


class AiImageDialog(QtWidgets.QDialog):
    """Collect parameters for generating a still image (T2I).

    This dialog does **not** run inference. The caller (gui.py) runs generation
    asynchronously and uses `wan_studio.ai_image.generate_image`.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        default_model: str = "black-forest-labs/FLUX.1-schnell",
        default_prompt: str = "",
        default_negative: str = "",
        default_w: int = 1024,
        default_h: int = 1024,
        default_steps: int = 20,
        default_guidance: float = 4.0,
    ):
        super().__init__(parent)
        self.setWindowTitle("AI Image (T2I)")
        self.setModal(True)
        self._req: Optional[AiImageRequest] = None

        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        lay.addLayout(form)

        self.ed_model = QtWidgets.QLineEdit(default_model or "")
        self.ed_model.setPlaceholderText("HuggingFace model id (ex: black-forest-labs/FLUX.1-schnell)")

        self.te_prompt = QtWidgets.QPlainTextEdit(default_prompt or "")
        self.te_prompt.setMinimumHeight(120)

        self.te_negative = QtWidgets.QPlainTextEdit(default_negative or "")
        self.te_negative.setMinimumHeight(70)

        self.sp_w = QtWidgets.QSpinBox(); self.sp_w.setRange(256, 4096); self.sp_w.setSingleStep(64); self.sp_w.setValue(int(default_w or 1024))
        self.sp_h = QtWidgets.QSpinBox(); self.sp_h.setRange(256, 4096); self.sp_h.setSingleStep(64); self.sp_h.setValue(int(default_h or 1024))

        self.sp_steps = QtWidgets.QSpinBox(); self.sp_steps.setRange(1, 200); self.sp_steps.setValue(int(default_steps or 20))
        self.sp_guidance = QtWidgets.QDoubleSpinBox(); self.sp_guidance.setRange(0.0, 30.0); self.sp_guidance.setDecimals(2); self.sp_guidance.setValue(float(default_guidance or 4.0))

        self.sp_seed = QtWidgets.QSpinBox(); self.sp_seed.setRange(0, 2_147_483_647); self.sp_seed.setValue(0)
        self.sp_seed.setToolTip("0 = seed aléatoire")

        self.cb_dtype = QtWidgets.QComboBox()
        self.cb_dtype.addItems(["auto", "fp16", "bf16"])

        form.addRow("Model", self.ed_model)
        form.addRow("Prompt", self.te_prompt)
        form.addRow("Negative", self.te_negative)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("W")); row.addWidget(self.sp_w)
        row.addSpacing(8)
        row.addWidget(QtWidgets.QLabel("H")); row.addWidget(self.sp_h)
        wrow = QtWidgets.QWidget(); wrow.setLayout(row)
        form.addRow("Size", wrow)

        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Steps")); row2.addWidget(self.sp_steps)
        row2.addSpacing(8)
        row2.addWidget(QtWidgets.QLabel("Guidance")); row2.addWidget(self.sp_guidance)
        row2.addSpacing(8)
        row2.addWidget(QtWidgets.QLabel("Seed")); row2.addWidget(self.sp_seed)
        row2.addSpacing(8)
        row2.addWidget(QtWidgets.QLabel("dtype")); row2.addWidget(self.cb_dtype)
        wrow2 = QtWidgets.QWidget(); wrow2.setLayout(row2)
        form.addRow("Params", wrow2)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("OK")
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        self.resize(720, 520)

    def _on_ok(self):
        model_id = (self.ed_model.text() or "").strip()
        prompt = (self.te_prompt.toPlainText() or "").strip()
        neg = (self.te_negative.toPlainText() or "").strip()
        if not model_id:
            QtWidgets.QMessageBox.warning(self, "AI Image", "Model ID vide")
            return
        if not prompt:
            QtWidgets.QMessageBox.warning(self, "AI Image", "Prompt vide")
            return
        seed_ui = int(self.sp_seed.value())
        seed = None if seed_ui == 0 else seed_ui
        self._req = AiImageRequest(
            model_id=model_id,
            prompt=prompt,
            negative_prompt=neg,
            width=int(self.sp_w.value()),
            height=int(self.sp_h.value()),
            steps=int(self.sp_steps.value()),
            guidance=float(self.sp_guidance.value()),
            seed=seed,
            dtype=str(self.cb_dtype.currentText() or "auto"),
        )
        self.accept()

    def request(self) -> Optional[AiImageRequest]:
        return self._req


@dataclass
class ImageInitResult:
    path: str
    preset_key: str
    prompt: str
    negative_prompt: str
    width: int
    height: int
    seed: int


class _GenWorker(QtCore.QObject):
    finished = QtCore.Signal(object)  # InitFrameSubprocessResult
    error = QtCore.Signal(str)

    def __init__(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        preset: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: Optional[int],
        out_dir: str,
        cache_dir: Optional[str],
    ):
        super().__init__()
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.preset = preset
        self.width = width
        self.height = height
        self.steps = int(steps)
        self.guidance = float(guidance)
        self.seed = seed
        self.out_dir = out_dir
        self.cache_dir = cache_dir

    @QtCore.Slot()
    def run(self):
        try:
            w, h = clamp_to_multiple_of_8(self.width, self.height)
            out_path = os.path.join(self.out_dir, "init_dialog.png")
            job = build_job(
                preset=self.preset,
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                width=w,
                height=h,
                steps=self.steps,
                guidance_scale=self.guidance,
                seed=self.seed,
                out_path=out_path,
                cache_dir=self.cache_dir,
                max_sequence_length=512,
            )
            res = run_init_frame_job(job, target_size=(w, h))
            if not res.ok:
                raise RuntimeError(res.error or "Init image failed")
            self.finished.emit(res)
        except Exception as e:
            self.error.emit(str(e))


class ImageInitDialog(QtWidgets.QDialog):
    """Small dialog to generate an init image for a clip.

    Runs generation in a QThread to keep UI responsive.
    """

    PRESETS: Dict[str, str] = {
        "Z-Image Turbo (default)": "zimage_turbo",
        "Z-Image (full)": "zimage",
        "FLUX.2-dev-bnb-4bit": "flux2_bnb4bit",
        "FLUX.2-dev (full)": "flux2",
        "SDXL-Lightning 4-step (fast)": "sdxl_lightning_4step",
        "SDXL Base (quality)": "sdxl_base",
        "SDXL Turbo (ultra fast)": "sdxl_turbo",
        "FLUX.1 schnell (optional)": "flux1_schnell",
    }

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        default_prompt: str = "",
        default_negative: str = "",
        default_preset_key: str = "zimage_turbo",
        default_width: int = 0,
        default_height: int = 0,
        default_steps: int = 28,
        default_guidance: float = 4.5,
        out_root: str,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Générer une image IA (init I2V)")
        self.setModal(True)

        self._out_root = out_root
        self._cache_dir = cache_dir
        self.result: Optional[ImageInitResult] = None
        self._generated_path: str = ""

        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[_GenWorker] = None

        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        lay.addLayout(form)

        self.cb_preset = QtWidgets.QComboBox()
        for k in self.PRESETS.keys():
            self.cb_preset.addItem(k)
        # Select preset by key (fallback to first).
        want_key = str(default_preset_key or "zimage_turbo")
        selected_index = 0
        for i in range(self.cb_preset.count()):
            k = self.cb_preset.itemText(i)
            if self.PRESETS.get(k) == want_key:
                selected_index = i
                break
        self.cb_preset.setCurrentIndex(selected_index)

        self.te_prompt = QtWidgets.QPlainTextEdit()
        self.te_prompt.setPlainText(default_prompt or "")
        self.te_prompt.setMinimumHeight(120)

        self.te_negative = QtWidgets.QPlainTextEdit()
        self.te_negative.setPlainText(default_negative or "")
        self.te_negative.setMinimumHeight(70)

        self.sp_w = QtWidgets.QSpinBox(); self.sp_w.setRange(256, 4096); self.sp_w.setSingleStep(64); self.sp_w.setValue(int(default_width or 1024))
        self.sp_h = QtWidgets.QSpinBox(); self.sp_h.setRange(256, 4096); self.sp_h.setSingleStep(64); self.sp_h.setValue(int(default_height or 1024))

        self.sp_steps = QtWidgets.QSpinBox(); self.sp_steps.setRange(1, 120); self.sp_steps.setValue(int(default_steps or 28))
        self.sp_gs = QtWidgets.QDoubleSpinBox(); self.sp_gs.setRange(0.0, 20.0); self.sp_gs.setDecimals(2); self.sp_gs.setValue(float(default_guidance or 4.5))

        self.sp_seed = QtWidgets.QSpinBox(); self.sp_seed.setRange(0, 2_147_483_647); self.sp_seed.setValue(0)
        self.sp_seed.setToolTip("0 = seed aléatoire")

        form.addRow("Preset", self.cb_preset)
        form.addRow("Prompt", self.te_prompt)
        form.addRow("Negative prompt", self.te_negative)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("W")); row.addWidget(self.sp_w)
        row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel("H")); row.addWidget(self.sp_h)
        row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel("Steps")); row.addWidget(self.sp_steps)
        row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel("Guidance")); row.addWidget(self.sp_gs)
        row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel("Seed")); row.addWidget(self.sp_seed)
        wrow = QtWidgets.QWidget(); wrow.setLayout(row)
        form.addRow("Image", wrow)

        self.preview = QtWidgets.QLabel()
        self.preview.setMinimumHeight(240)
        self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview.setText("(Aperçu ici)")
        lay.addWidget(self.preview)

        self.status = QtWidgets.QLabel("—")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        self._btn_ok = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self._btn_ok.setText("Valider")
        self._btn_ok.setEnabled(False)
        self._btn_ok.clicked.connect(self._on_accept)
        self._btn_generate = QtWidgets.QPushButton("Générer")
        self._btn_generate.clicked.connect(self._on_generate)
        buttons.addButton(self._btn_generate, QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        self.resize(720, 760)

    @QtCore.Slot()
    def _on_accept(self):
        if self.result and getattr(self.result, 'path', '') and os.path.exists(self.result.path):
            self.accept()
        else:
            QtWidgets.QMessageBox.warning(self, "Image IA", "Aucune image générée à valider.")

    def closeEvent(self, event):
        # Best-effort: stop thread on dialog close
        try:
            if self._thread is not None:
                self._thread.requestInterruption()
                self._thread.quit()
                self._thread.wait(250)
        except Exception:
            pass
        super().closeEvent(event)

    def _cleanup_thread(self):
        try:
            QtWidgets.QApplication.restoreOverrideCursor()
        except Exception:
            pass
        # Best-effort GPU/CPU memory cleanup (helps after big img gen)
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
        try:
            if self._thread is not None:
                self._thread.quit()
        except Exception:
            pass
        self._thread = None
        self._worker = None

    def _on_generate(self):
        if self._thread is not None:
            return

        preset_name = self.cb_preset.currentText()
        preset_key = self.PRESETS.get(preset_name, "sdxl_lightning_4step")
        prompt = (self.te_prompt.toPlainText() or "").strip()
        neg = (self.te_negative.toPlainText() or "").strip()
        w = int(self.sp_w.value()); h = int(self.sp_h.value())
        steps = int(self.sp_steps.value())
        gs = float(self.sp_gs.value())
        seed_ui = int(self.sp_seed.value())
        seed = None if seed_ui == 0 else seed_ui

        # Store current request so finished() can be handled by a QObject slot
        # on the UI thread (avoid lambdas that may run in the worker thread).
        self._pending = {
            "preset_key": preset_key,
            "prompt": prompt,
            "neg": neg,
            "w": w,
            "h": h,
            "steps": steps,
            "gs": gs,
            "seed_ui": seed_ui,
        }

        if not prompt:
            QtWidgets.QMessageBox.warning(self, "Image IA", "Prompt vide.")
            return

        os.makedirs(self._out_root, exist_ok=True)

        self.status.setText("Génération en cours…")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)

        thread = QtCore.QThread(self)
        worker = _GenWorker(
            prompt=prompt,
            negative_prompt=neg,
            preset=preset_key,
            width=w,
            height=h,
            steps=steps,
            guidance=gs,
            seed=seed,
            out_dir=self._out_root,
            cache_dir=self._cache_dir,
        )
        worker.moveToThread(thread)

        # Signals are delivered back on the UI thread.
        worker.finished.connect(self._on_worker_ok)
        worker.error.connect(self._on_worker_err)

        thread.started.connect(worker.run)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._thread = thread
        self._worker = worker
        thread.start()

    @QtCore.Slot(object)
    def _on_worker_ok(self, res: object):
        pend = getattr(self, "_pending", {}) or {}
        preset_key = str(pend.get("preset_key", ""))
        prompt = str(pend.get("prompt", ""))
        neg = str(pend.get("neg", ""))
        w = int(pend.get("w", 1024) or 1024)
        h = int(pend.get("h", 1024) or 1024)
        steps = int(pend.get("steps", 28) or 28)
        gs = float(pend.get("gs", 4.5) or 4.5)
        seed_ui = int(pend.get("seed_ui", 0) or 0)
        self._cleanup_thread()
        try:
            path = str(getattr(res, "path", "") or "")
            used_seed = int(getattr(res, "seed", seed_ui) or seed_ui)
            self.result = ImageInitResult(
                path=path,
                preset_key=preset_key,
                prompt=prompt,
                negative_prompt=neg,
                width=w,
                height=h,
                seed=used_seed,
            )
            self._generated_path = path
            self.status.setText(f"Générée: {path}")
            # Preview (best-effort)
            try:
                from PySide6 import QtGui
                pm = QtGui.QPixmap(path)
                if not pm.isNull():
                    self.preview.setPixmap(pm.scaled(640, 360, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))
            except Exception:
                pass
            try:
                if hasattr(self, '_btn_ok'):
                    self._btn_ok.setEnabled(bool(path and os.path.exists(path)))
            except Exception:
                pass
        except Exception as e:
            self.status.setText("Erreur génération.")
            QtWidgets.QMessageBox.critical(self, "Image IA", str(e))

    @QtCore.Slot(str)
    def _on_worker_err(self, msg: str):
        self._cleanup_thread()
        self.status.setText("Erreur génération.")
        QtWidgets.QMessageBox.critical(self, "Image IA", msg)
