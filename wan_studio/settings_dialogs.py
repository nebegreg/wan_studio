from __future__ import annotations

import os
from typing import List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from .config import InitImageSpec, ProjectConfig
from .cache_utils import clear_dir, dir_size_bytes, prune_to_quota


def _fmt_bytes(n: int) -> str:
    try:
        n = int(n)
    except Exception:
        return "0 B"
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n/1024:.1f} KB"
    if n < 1024**3:
        return f"{n/1024**2:.1f} MB"
    return f"{n/1024**3:.2f} GB"


class InitImageSettingsDialog(QtWidgets.QDialog):
    """Global Init Image settings (Flux2/Flux1/SDXL presets + policy + refs/adapters).

    This edits cfg.init_image (InitImageSpec). Designed to be safe even if
    optional adapter pipelines are not installed.
    """

    def __init__(self, parent=None, *, cfg: ProjectConfig):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle("Init Image Settings")
        self.resize(900, 560)

        self.spec: InitImageSpec = getattr(cfg, "init_image", InitImageSpec())

        tabs = QtWidgets.QTabWidget(self)

        # ---------------- Basic ----------------
        basic = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(basic)

        self.cb_enabled = QtWidgets.QCheckBox("Enable init frame generation")
        self.cb_enabled.setChecked(bool(getattr(self.spec, "enabled", True)))

        self.cb_preset = QtWidgets.QComboBox()
        self._preset_items = [
            ("Z-Image Turbo — default", "zimage_turbo"),
            ("Z-Image (full)", "zimage"),
            ("Flux2 (bnb 4bit)", "flux2_bnb4bit"),
            ("Flux2 (full)", "flux2"),
            ("Flux1 schnell", "flux1_schnell"),
            ("SDXL Lightning (4-step)", "sdxl_lightning_4step"),
            ("SDXL base", "sdxl_base"),
            ("SDXL turbo", "sdxl_turbo"),
        ]
        for label, key in self._preset_items:
            self.cb_preset.addItem(label, key)
        self._set_combo_data(self.cb_preset, str(getattr(self.spec, "preset", "zimage_turbo") or "zimage_turbo"))

        self.cb_policy = QtWidgets.QComboBox()
        self.cb_policy.addItem("necessary (use cache)", "necessary")
        self.cb_policy.addItem("always_refine (regen)", "always_refine")
        self._set_combo_data(self.cb_policy, str(getattr(self.spec, "policy", "always_refine") or "always_refine"))

        self.cb_cache_loc = QtWidgets.QComboBox()
        self.cb_cache_loc.addItem("assets (outputs/.../assets/init_frames)", "assets")
        self.cb_cache_loc.addItem("project cache (.wan_cache/init_frames)", "project")
        self._set_combo_data(self.cb_cache_loc, str(getattr(self.spec, "cache_location", "assets") or "assets"))

        self.sp_w = QtWidgets.QSpinBox(); self.sp_w.setRange(128, 4096); self.sp_w.setValue(int(getattr(self.spec, "width", 1024) or 1024))
        self.sp_h = QtWidgets.QSpinBox(); self.sp_h.setRange(128, 4096); self.sp_h.setValue(int(getattr(self.spec, "height", 1024) or 1024))
        self.sp_steps = QtWidgets.QSpinBox(); self.sp_steps.setRange(1, 200); self.sp_steps.setValue(int(getattr(self.spec, "steps", 28) or 28))
        self.sp_gs = QtWidgets.QDoubleSpinBox(); self.sp_gs.setRange(0.0, 30.0); self.sp_gs.setDecimals(2); self.sp_gs.setSingleStep(0.1)
        self.sp_gs.setValue(float(getattr(self.spec, "guidance_scale", 4.0) or 4.0))
        self.ed_seed = QtWidgets.QLineEdit(str(getattr(self.spec, "seed", "") or ""))
        self.ed_seed.setPlaceholderText("empty = random")

        self.cb_use_existing = QtWidgets.QCheckBox("Use existing init image as reference")
        self.cb_use_existing.setChecked(bool(getattr(self.spec, "use_existing_as_reference", True)))
        self.cb_mem_loc = QtWidgets.QCheckBox("Use last location memory (prev shot refs)")
        self.cb_mem_loc.setChecked(bool(getattr(self.spec, "use_last_location_memory", True)))
        self.sp_max_refs = QtWidgets.QSpinBox(); self.sp_max_refs.setRange(0, 24); self.sp_max_refs.setValue(int(getattr(self.spec, "max_refs", 6) or 6))

        form.addRow(self.cb_enabled)
        form.addRow("Preset", self.cb_preset)
        form.addRow("Policy", self.cb_policy)
        form.addRow("Cache location", self.cb_cache_loc)
        form.addRow("Width", self.sp_w)
        form.addRow("Height", self.sp_h)
        form.addRow("Steps", self.sp_steps)
        form.addRow("Guidance", self.sp_gs)
        form.addRow("Seed", self.ed_seed)
        form.addRow(self.cb_use_existing)
        form.addRow(self.cb_mem_loc)
        form.addRow("Max refs", self.sp_max_refs)

        tabs.addTab(basic, "Basic")

        # ---------------- References ----------------
        refs = QtWidgets.QWidget(); rl = QtWidgets.QVBoxLayout(refs)
        rl.addWidget(QtWidgets.QLabel("<b>Style reference images</b> (optional):"))
        self.list_style = QtWidgets.QListWidget()
        for p in (getattr(self.spec, "style_ref_images", []) or []):
            self.list_style.addItem(str(p))

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add_style = QtWidgets.QPushButton("Add…")
        self.btn_rm_style = QtWidgets.QPushButton("Remove")
        btn_row.addWidget(self.btn_add_style)
        btn_row.addWidget(self.btn_rm_style)
        btn_row.addStretch(1)
        rl.addWidget(self.list_style, 1)
        rl.addLayout(btn_row)
        rl.addWidget(QtWidgets.QLabel("Tip: use 1–2 images for film look / grade. Characters & Locations refs live in their libraries."))
        tabs.addTab(refs, "Style refs")

        self.btn_add_style.clicked.connect(self._add_style_refs)
        self.btn_rm_style.clicked.connect(self._remove_style_refs)

        # ---------------- Adapters ----------------
        ad = QtWidgets.QWidget(); al = QtWidgets.QFormLayout(ad)
        self.cb_ip = QtWidgets.QCheckBox("Enable IP-Adapter (best-effort)")
        self.cb_ip.setChecked(bool(getattr(self.spec, "enable_ip_adapter", False)))
        self.ed_ip_model = QtWidgets.QLineEdit(str(getattr(self.spec, "ip_adapter_model_id", "") or ""))
        self.ed_ip_sub = QtWidgets.QLineEdit(str(getattr(self.spec, "ip_adapter_subfolder", "") or ""))
        self.ed_ip_weight = QtWidgets.QLineEdit(str(getattr(self.spec, "ip_adapter_weight_name", "") or ""))
        self.sp_ip_char = QtWidgets.QDoubleSpinBox(); self.sp_ip_char.setRange(0.0, 2.0); self.sp_ip_char.setSingleStep(0.05); self.sp_ip_char.setDecimals(2)
        self.sp_ip_loc = QtWidgets.QDoubleSpinBox(); self.sp_ip_loc.setRange(0.0, 2.0); self.sp_ip_loc.setSingleStep(0.05); self.sp_ip_loc.setDecimals(2)
        self.sp_ip_style = QtWidgets.QDoubleSpinBox(); self.sp_ip_style.setRange(0.0, 2.0); self.sp_ip_style.setSingleStep(0.05); self.sp_ip_style.setDecimals(2)
        self.sp_ip_char.setValue(float(getattr(self.spec, "ip_adapter_scale_character", 0.8) or 0.8))
        self.sp_ip_loc.setValue(float(getattr(self.spec, "ip_adapter_scale_location", 0.6) or 0.6))
        self.sp_ip_style.setValue(float(getattr(self.spec, "ip_adapter_scale_style", 0.5) or 0.5))

        self.cb_cn = QtWidgets.QCheckBox("Enable ControlNet (best-effort; SDXL init)" )
        self.cb_cn.setChecked(bool(getattr(self.spec, "enable_controlnet", False)))
        self.cb_cn_type = QtWidgets.QComboBox();
        for t in ("canny", "depth", "pose"):
            self.cb_cn_type.addItem(t, t)
        self._set_combo_data(self.cb_cn_type, str(getattr(self.spec, "controlnet_type", "canny") or "canny"))
        self.ed_cn_model = QtWidgets.QLineEdit(str(getattr(self.spec, "controlnet_model_id", "") or ""))
        self.sp_cn_scale = QtWidgets.QDoubleSpinBox(); self.sp_cn_scale.setRange(0.0, 2.0); self.sp_cn_scale.setSingleStep(0.05); self.sp_cn_scale.setDecimals(2)
        self.sp_cn_scale.setValue(float(getattr(self.spec, "controlnet_scale", 0.75) or 0.75))
        self.sp_refine = QtWidgets.QDoubleSpinBox(); self.sp_refine.setRange(0.0, 0.99); self.sp_refine.setSingleStep(0.05); self.sp_refine.setDecimals(2)
        self.sp_refine.setValue(float(getattr(self.spec, "refine_strength", 0.35) or 0.35))

        al.addRow(self.cb_ip)
        al.addRow("IP-Adapter model id", self.ed_ip_model)
        al.addRow("IP-Adapter subfolder", self.ed_ip_sub)
        al.addRow("IP-Adapter weight", self.ed_ip_weight)
        al.addRow("IP scale (character)", self.sp_ip_char)
        al.addRow("IP scale (location)", self.sp_ip_loc)
        al.addRow("IP scale (style)", self.sp_ip_style)
        al.addRow(QtWidgets.QLabel(""))
        al.addRow(self.cb_cn)
        al.addRow("ControlNet type", self.cb_cn_type)
        al.addRow("ControlNet model id", self.ed_cn_model)
        al.addRow("ControlNet scale", self.sp_cn_scale)
        al.addRow("Refine strength", self.sp_refine)
        tabs.addTab(ad, "Adapters")

        # ---------------- Models ----------------
        models = QtWidgets.QWidget()
        ml = QtWidgets.QFormLayout(models)
        ml.addRow(QtWidgets.QLabel("<b>Model overrides (optional)</b> — laisse vide pour utiliser les IDs par défaut/env."))

        self.ed_flux2_bnb4bit = QtWidgets.QLineEdit(str(getattr(self.spec, "flux2_bnb4bit_model_id", "") or ""))
        self.ed_flux2_full = QtWidgets.QLineEdit(str(getattr(self.spec, "flux2_model_id", "") or ""))
        self.ed_zimage_turbo = QtWidgets.QLineEdit(str(getattr(self.spec, "zimage_turbo_model_id", "") or ""))
        self.ed_zimage = QtWidgets.QLineEdit(str(getattr(self.spec, "zimage_model_id", "") or ""))
        self.ed_sdxl_base = QtWidgets.QLineEdit(str(getattr(self.spec, "sdxl_base_model_id", "") or ""))
        self.ed_sdxl_turbo = QtWidgets.QLineEdit(str(getattr(self.spec, "sdxl_turbo_model_id", "") or ""))
        self.ed_sdxl_lora = QtWidgets.QLineEdit(str(getattr(self.spec, "sdxl_lightning_lora_id", "") or ""))
        self.ed_sdxl_lora_file = QtWidgets.QLineEdit(str(getattr(self.spec, "sdxl_lightning_lora_file", "") or ""))
        self.ed_flux1 = QtWidgets.QLineEdit(str(getattr(self.spec, "flux1_schnell_model_id", "") or ""))

        ml.addRow("Flux2 bnb4bit model id", self.ed_flux2_bnb4bit)
        ml.addRow("Flux2 full model id", self.ed_flux2_full)
        ml.addRow("Z-Image Turbo model id", self.ed_zimage_turbo)
        ml.addRow("Z-Image model id", self.ed_zimage)
        ml.addRow("SDXL base model id", self.ed_sdxl_base)
        ml.addRow("SDXL turbo model id", self.ed_sdxl_turbo)
        ml.addRow("SDXL Lightning LoRA repo", self.ed_sdxl_lora)
        ml.addRow("SDXL Lightning LoRA file", self.ed_sdxl_lora_file)
        ml.addRow("Flux1 schnell model id", self.ed_flux1)

        tabs.addTab(models, "Models")

        # ---------------- Buttons ----------------
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(tabs, 1)
        lay.addWidget(btns)

    def _set_combo_data(self, cb: QtWidgets.QComboBox, data: str) -> None:
        for i in range(cb.count()):
            if cb.itemData(i) == data:
                cb.setCurrentIndex(i)
                return

    def _add_style_refs(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Add style reference images", ".", "Images (*.png *.jpg *.jpeg *.webp)")
        if not files:
            return
        for f in files:
            self.list_style.addItem(os.path.abspath(os.path.expanduser(f)))

    def _remove_style_refs(self):
        rows = sorted({i.row() for i in self.list_style.selectedIndexes()}, reverse=True)
        for r in rows:
            self.list_style.takeItem(r)

    def apply_to_cfg(self) -> None:
        s = self.spec
        s.enabled = bool(self.cb_enabled.isChecked())
        s.preset = str(self.cb_preset.currentData() or "zimage_turbo")
        s.policy = str(self.cb_policy.currentData() or "always_refine")
        s.cache_location = str(self.cb_cache_loc.currentData() or "assets")
        s.width = int(self.sp_w.value())
        s.height = int(self.sp_h.value())
        s.steps = int(self.sp_steps.value())
        s.guidance_scale = float(self.sp_gs.value())

        seed_txt = (self.ed_seed.text() or "").strip()
        if seed_txt == "":
            s.seed = None
        else:
            try:
                s.seed = int(seed_txt)
            except Exception:
                s.seed = None

        s.use_existing_as_reference = bool(self.cb_use_existing.isChecked())
        s.use_last_location_memory = bool(self.cb_mem_loc.isChecked())
        s.max_refs = int(self.sp_max_refs.value())

        s.style_ref_images = [self.list_style.item(i).text() for i in range(self.list_style.count())]

        s.enable_ip_adapter = bool(self.cb_ip.isChecked())
        s.ip_adapter_model_id = (self.ed_ip_model.text() or "").strip()
        s.ip_adapter_subfolder = (self.ed_ip_sub.text() or "").strip()
        s.ip_adapter_weight_name = (self.ed_ip_weight.text() or "").strip()
        s.ip_adapter_scale_character = float(self.sp_ip_char.value())
        s.ip_adapter_scale_location = float(self.sp_ip_loc.value())
        s.ip_adapter_scale_style = float(self.sp_ip_style.value())

        s.enable_controlnet = bool(self.cb_cn.isChecked())
        s.controlnet_type = str(self.cb_cn_type.currentData() or "canny")
        s.controlnet_model_id = (self.ed_cn_model.text() or "").strip()
        s.controlnet_scale = float(self.sp_cn_scale.value())
        s.refine_strength = float(self.sp_refine.value())

        s.flux2_bnb4bit_model_id = (self.ed_flux2_bnb4bit.text() or "").strip()
        s.flux2_model_id = (self.ed_flux2_full.text() or "").strip()
        s.zimage_turbo_model_id = (self.ed_zimage_turbo.text() or "").strip()
        s.zimage_model_id = (self.ed_zimage.text() or "").strip()
        s.sdxl_base_model_id = (self.ed_sdxl_base.text() or "").strip()
        s.sdxl_turbo_model_id = (self.ed_sdxl_turbo.text() or "").strip()
        s.sdxl_lightning_lora_id = (self.ed_sdxl_lora.text() or "").strip()
        s.sdxl_lightning_lora_file = (self.ed_sdxl_lora_file.text() or "").strip()
        s.flux1_schnell_model_id = (self.ed_flux1.text() or "").strip()

        self.cfg.init_image = s


class CacheManagerDialog(QtWidgets.QDialog):
    """Simple cache manager: view sizes, clear caches, prune to quota."""

    def __init__(self, parent=None, *, cfg: ProjectConfig):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle("Cache Manager")
        self.resize(860, 460)

        from .clip_cache import clip_cache_root
        from .flux2_init import get_init_cache_dirs

        self.init_dir, self.init_json = get_init_cache_dirs(cfg)
        self.clip_root = clip_cache_root(cfg)

        self.lbl_init = QtWidgets.QLabel()
        self.lbl_clip = QtWidgets.QLabel()
        self.lbl_quota = QtWidgets.QLabel()

        self.sp_quota = QtWidgets.QDoubleSpinBox()
        self.sp_quota.setRange(0.0, 5000.0)
        self.sp_quota.setDecimals(1)
        self.sp_quota.setSingleStep(5.0)
        self.sp_quota.setValue(float(getattr(cfg, "cache_max_gb", 0.0) or 0.0))
        self.sp_quota.setToolTip("0 = unlimited")

        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_clear_init = QtWidgets.QPushButton("Clear init cache")
        self.btn_clear_clip = QtWidgets.QPushButton("Clear clip cache")
        self.btn_clear_all = QtWidgets.QPushButton("Clear ALL")
        self.btn_prune = QtWidgets.QPushButton("Prune to quota")

        self.btn_open_init = QtWidgets.QPushButton("Open init folder")
        self.btn_open_clip = QtWidgets.QPushButton("Open clip folder")

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("<b>Init cache</b>"), 0, 0)
        grid.addWidget(self.lbl_init, 0, 1)
        grid.addWidget(self.btn_open_init, 0, 2)
        grid.addWidget(self.btn_clear_init, 0, 3)

        grid.addWidget(QtWidgets.QLabel("<b>Clip cache</b>"), 1, 0)
        grid.addWidget(self.lbl_clip, 1, 1)
        grid.addWidget(self.btn_open_clip, 1, 2)
        grid.addWidget(self.btn_clear_clip, 1, 3)

        grid.addWidget(QtWidgets.QLabel("<b>Quota (GB)</b>"), 2, 0)
        grid.addWidget(self.sp_quota, 2, 1)
        grid.addWidget(self.btn_prune, 2, 2)
        grid.addWidget(self.btn_refresh, 2, 3)

        grid.addWidget(self.btn_clear_all, 3, 3)

        box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.reject)
        box.accepted.connect(self.accept)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(grid)
        lay.addStretch(1)
        lay.addWidget(box)

        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_clear_init.clicked.connect(self.clear_init)
        self.btn_clear_clip.clicked.connect(self.clear_clip)
        self.btn_clear_all.clicked.connect(self.clear_all)
        self.btn_prune.clicked.connect(self.prune)
        self.btn_open_init.clicked.connect(lambda: self._open_folder(self.init_dir))
        self.btn_open_clip.clicked.connect(lambda: self._open_folder(self.clip_root))

        self.refresh()

    def _open_folder(self, path: str) -> None:
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(path):
            QtWidgets.QMessageBox.information(self, "Open folder", f"Folder does not exist:\n{path}")
            return
        try:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))
        except Exception:
            pass

    def refresh(self) -> None:
        init_size = dir_size_bytes(self.init_dir)
        clip_size = dir_size_bytes(self.clip_root)
        self.lbl_init.setText(f"{self.init_dir}  —  {_fmt_bytes(init_size)}")
        self.lbl_clip.setText(f"{self.clip_root}  —  {_fmt_bytes(clip_size)}")
        q = float(self.sp_quota.value())
        self.cfg.cache_max_gb = float(q)

    def clear_init(self) -> None:
        if QtWidgets.QMessageBox.question(self, "Clear init cache", "Delete all init-frame cache files?") != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        os.makedirs(self.init_dir, exist_ok=True)
        clear_dir(self.init_dir)
        try:
            if self.init_json and os.path.exists(self.init_json):
                os.remove(self.init_json)
        except Exception:
            pass
        self.refresh()

    def clear_clip(self) -> None:
        if QtWidgets.QMessageBox.question(self, "Clear clip cache", "Delete all cached clip renders?") != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        os.makedirs(self.clip_root, exist_ok=True)
        clear_dir(self.clip_root)
        self.refresh()

    def clear_all(self) -> None:
        if QtWidgets.QMessageBox.question(self, "Clear ALL", "Delete init + clip caches?",) != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.clear_init()
        self.clear_clip()

    def prune(self) -> None:
        q = float(self.sp_quota.value())
        self.cfg.cache_max_gb = float(q)
        if q <= 0:
            QtWidgets.QMessageBox.information(self, "Prune", "Quota is 0 (unlimited). Nothing to prune.")
            return
        max_bytes = int(q * 1024**3)
        # Split quota between init and clip caches (simple policy: 25% init, 75% clip)
        init_quota = int(max_bytes * 0.25)
        clip_quota = int(max_bytes * 0.75)
        try:
            prune_to_quota(self.init_dir, init_quota, keep_min=0)
        except Exception:
            pass
        try:
            prune_to_quota(self.clip_root, clip_quota, keep_min=1)
        except Exception:
            pass
        self.refresh()
