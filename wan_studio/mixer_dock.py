from __future__ import annotations

import os
from typing import Optional, Dict

from PySide6 import QtCore, QtWidgets, QtGui

from .waveform_cache import load_peaks


def _clamp(x: float, a: float, b: float) -> float:
    try:
        x = float(x)
    except Exception:
        x = 0.0
    return max(a, min(b, x))


class LevelMeter(QtWidgets.QWidget):
    """Lightweight vertical level meter (0..1)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._level = 0.0
        self.setMinimumWidth(14)
        self.setMinimumHeight(90)

    def setLevel(self, level: float):
        level = _clamp(level, 0.0, 1.0)
        if abs(level - self._level) > 1e-3:
            self._level = level
            self.update()

    def paintEvent(self, e: QtGui.QPaintEvent):  # noqa: N802
        p = QtGui.QPainter(self)
        r = self.rect()
        # background
        p.fillRect(r, QtGui.QColor(25, 25, 25))
        # border
        p.setPen(QtGui.QPen(QtGui.QColor(60, 60, 60), 1))
        p.drawRect(r.adjusted(0, 0, -1, -1))

        # bar
        h = r.height()
        w = r.width()
        lvl = _clamp(self._level, 0.0, 1.0)
        bar_h = int(round(h * lvl))
        if bar_h > 0:
            y0 = r.bottom() - bar_h + 1
            bar = QtCore.QRect(r.left() + 1, y0, w - 2, bar_h - 1)
            # simple "traffic light" gradient-ish
            if lvl < 0.7:
                col = QtGui.QColor(80, 200, 120)
            elif lvl < 0.9:
                col = QtGui.QColor(240, 200, 90)
            else:
                col = QtGui.QColor(255, 90, 90)
            p.fillRect(bar, col)

        p.end()


class _BusStrip(QtWidgets.QWidget):
    """One mixer strip (fader + mute/solo + meter)."""

    changed = QtCore.Signal()  # emitted when user changes something

    def __init__(self, name: str, *, parent=None):
        super().__init__(parent)
        self.bus_name = name
        self.setMinimumWidth(120)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        self.lbl = QtWidgets.QLabel(name)
        self.lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        f = self.lbl.font()
        f.setBold(True)
        self.lbl.setFont(f)
        lay.addWidget(self.lbl)

        self.meter = LevelMeter(self)
        self.meter.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding)
        lay.addWidget(self.meter, 1, QtCore.Qt.AlignmentFlag.AlignHCenter)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self.slider.setRange(-60, 12)
        self.slider.setValue(0)
        self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksRight)
        self.slider.setTickInterval(6)
        lay.addWidget(self.slider, 3)

        self.db = QtWidgets.QLabel("0.0 dB")
        self.db.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(self.db)

        row = QtWidgets.QHBoxLayout()
        self.btn_mute = QtWidgets.QToolButton()
        self.btn_mute.setText("M")
        self.btn_mute.setCheckable(True)
        self.btn_solo = QtWidgets.QToolButton()
        self.btn_solo.setText("S")
        self.btn_solo.setCheckable(True)
        row.addWidget(self.btn_mute, 1)
        row.addWidget(self.btn_solo, 1)
        lay.addLayout(row)

        # signals
        self.slider.valueChanged.connect(self._on_slider)
        self.slider.sliderReleased.connect(self.changed.emit)
        self.btn_mute.toggled.connect(self.changed.emit)
        self.btn_solo.toggled.connect(self.changed.emit)

    def _on_slider(self, v: int):
        self.db.setText(f"{float(v):.1f} dB")

    def setLevel(self, level01: float):
        self.meter.setLevel(level01)

    def setGainDb(self, gain_db: float):
        try:
            v = int(round(float(gain_db)))
        except Exception:
            v = 0
        v = max(-60, min(12, v))
        self.slider.blockSignals(True)
        self.slider.setValue(v)
        self.slider.blockSignals(False)
        self.db.setText(f"{float(v):.1f} dB")

    def gainDb(self) -> float:
        return float(self.slider.value())

    def setMuted(self, muted: bool):
        self.btn_mute.blockSignals(True)
        self.btn_mute.setChecked(bool(muted))
        self.btn_mute.blockSignals(False)

    def muted(self) -> bool:
        return bool(self.btn_mute.isChecked())

    def setSolo(self, solo: bool):
        self.btn_solo.blockSignals(True)
        self.btn_solo.setChecked(bool(solo))
        self.btn_solo.blockSignals(False)

    def solo(self) -> bool:
        return bool(self.btn_solo.isChecked())


class AudioMixerDock(QtWidgets.QDockWidget):
    """Premiere-like audio mixer dock.

    - controls: per-bus gain + mute + solo, ducking, master gain
    - meters: updates from the Audio Preview player position (best-effort)
    """

    def __init__(self, main_window: QtWidgets.QMainWindow):
        super().__init__("Audio Mixer", main_window)
        self._mw = main_window
        self._player = None  # QMediaPlayer

        self._peaks: Dict[str, object] = {}
        self._last_paths: Dict[str, str] = {}
        self._last_dur_ms: int = 0

        w = QtWidgets.QWidget()
        self.setWidget(w)
        root = QtWidgets.QVBoxLayout(w)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        strips = QtWidgets.QHBoxLayout()
        strips.setSpacing(8)

        self.strip_dialogue = _BusStrip("A1 Dialogue")
        self.strip_vfx = _BusStrip("A2 VFX")
        self.strip_music = _BusStrip("A3 Music")
        self.strip_master = _BusStrip("MASTER")
        strips.addWidget(self.strip_dialogue, 1)
        strips.addWidget(self.strip_vfx, 1)
        strips.addWidget(self.strip_music, 1)
        strips.addWidget(self.strip_master, 1)

        root.addLayout(strips, 1)

        # duck + buttons
        row = QtWidgets.QHBoxLayout()
        self.cb_duck = QtWidgets.QCheckBox("Duck music under dialogue")
        self.btn_sync = QtWidgets.QPushButton("Sync from project")
        self.btn_rebuild = QtWidgets.QPushButton("Rebuild master mix")
        row.addWidget(self.cb_duck, 1)
        row.addWidget(self.btn_sync)
        row.addWidget(self.btn_rebuild)
        root.addLayout(row)

        # status
        self.lbl_status = QtWidgets.QLabel("Meters: idle")
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

        # wire changes
        for s in (self.strip_dialogue, self.strip_vfx, self.strip_music, self.strip_master):
            s.changed.connect(self._apply_to_cfg_from_ui)
        self.cb_duck.toggled.connect(self._apply_to_cfg_from_ui)
        self.btn_sync.clicked.connect(self._sync_clicked)
        self.btn_rebuild.clicked.connect(self._rebuild_clicked)

        self._meter_timer = QtCore.QTimer(self)
        self._meter_timer.setInterval(60)  # ~16 fps
        self._meter_timer.timeout.connect(self._update_meters)

        # initial sync
        try:
            self.sync_from_cfg(getattr(self._mw, "cfg", None))
        except Exception:
            pass

    # --- external API ---
    def attach_player(self, player):
        """Attach a QMediaPlayer used for audio preview (to drive meters)."""
        self._player = player
        try:
            self._player.positionChanged.connect(lambda *_: self._meter_timer.start())
            self._player.playbackStateChanged.connect(self._on_playback_state_changed)
        except Exception:
            pass

    def on_preview_started(self, path: str):
        """Called by MainWindow when preview playback is started."""
        self._refresh_peaks()
        self.lbl_status.setText(f"Preview: {os.path.basename(path) if path else 'audio'}")

    def sync_from_cfg(self, cfg):
        """Update UI from cfg.audio."""
        try:
            if cfg is None:
                return
            a = getattr(cfg, "audio", None)
            if a is None:
                return
            self.strip_music.setGainDb(getattr(a, "music_gain_db", -14.0))
            self.strip_dialogue.setGainDb(getattr(a, "dialogue_gain_db", -3.0))
            self.strip_vfx.setGainDb(getattr(a, "vfx_gain_db", -6.0))
            self.strip_master.setGainDb(getattr(a, "master_gain_db", 0.0))

            self.strip_music.setMuted(bool(getattr(a, "music_muted", False)))
            self.strip_dialogue.setMuted(bool(getattr(a, "dialogue_muted", False)))
            self.strip_vfx.setMuted(bool(getattr(a, "vfx_muted", False)))

            self.cb_duck.blockSignals(True)
            self.cb_duck.setChecked(bool(getattr(a, "duck_music_under_dialogue", True)))
            self.cb_duck.blockSignals(False)

            self._refresh_peaks()
        except Exception:
            pass

    # --- UI callbacks ---
    def _sync_clicked(self):
        try:
            self.sync_from_cfg(getattr(self._mw, "cfg", None))
        except Exception:
            pass

    def _rebuild_clicked(self):
        fn = getattr(self._mw, "_audio_rebuild_master_mix", None)
        if callable(fn):
            fn()
        else:
            QtWidgets.QMessageBox.information(self, "Audio", "Rebuild function not available in this build.")

    def _apply_to_cfg_from_ui(self):
        """Apply current dock UI to cfg.audio (and mirror to Audio tab widgets if present)."""
        cfg = getattr(self._mw, "cfg", None)
        if cfg is None:
            return
        a = getattr(cfg, "audio", None)
        if a is None:
            return

        # gain
        a.music_gain_db = float(self.strip_music.gainDb())
        a.dialogue_gain_db = float(self.strip_dialogue.gainDb())
        a.vfx_gain_db = float(self.strip_vfx.gainDb())
        a.master_gain_db = float(self.strip_master.gainDb())

        # solo logic (non-destructive): apply to mutes only while at least one solo is enabled
        solos = {
            "music": self.strip_music.solo(),
            "dialogue": self.strip_dialogue.solo(),
            "vfx": self.strip_vfx.solo(),
        }
        if any(solos.values()):
            # store original mutes if not already stored
            if not hasattr(self._mw, "_mixer_prev_mutes") or self._mw._mixer_prev_mutes is None:
                self._mw._mixer_prev_mutes = {
                    "music_muted": bool(getattr(a, "music_muted", False)),
                    "dialogue_muted": bool(getattr(a, "dialogue_muted", False)),
                    "vfx_muted": bool(getattr(a, "vfx_muted", False)),
                }
            a.music_muted = not solos["music"]
            a.dialogue_muted = not solos["dialogue"]
            a.vfx_muted = not solos["vfx"]
        else:
            prev = getattr(self._mw, "_mixer_prev_mutes", None)
            if isinstance(prev, dict):
                a.music_muted = bool(prev.get("music_muted", False))
                a.dialogue_muted = bool(prev.get("dialogue_muted", False))
                a.vfx_muted = bool(prev.get("vfx_muted", False))
            self._mw._mixer_prev_mutes = None

        # explicit mutes (applied when no solo)
        if not any(solos.values()):
            a.music_muted = bool(self.strip_music.muted())
            a.dialogue_muted = bool(self.strip_dialogue.muted())
            a.vfx_muted = bool(self.strip_vfx.muted())
        else:
            # reflect the effective mute state
            self.strip_music.setMuted(bool(getattr(a, "music_muted", False)))
            self.strip_dialogue.setMuted(bool(getattr(a, "dialogue_muted", False)))
            self.strip_vfx.setMuted(bool(getattr(a, "vfx_muted", False)))

        a.duck_music_under_dialogue = bool(self.cb_duck.isChecked())

        # Mirror into Audio tab widgets (best-effort, no crash if missing)
        try:
            if hasattr(self._mw, "sp_music_gain"):
                self._mw.sp_music_gain.setValue(float(a.music_gain_db))
            if hasattr(self._mw, "sp_dialogue_gain"):
                self._mw.sp_dialogue_gain.setValue(float(a.dialogue_gain_db))
            if hasattr(self._mw, "sp_vfx_gain"):
                self._mw.sp_vfx_gain.setValue(float(a.vfx_gain_db))
            if hasattr(self._mw, "sp_master_gain"):
                self._mw.sp_master_gain.setValue(float(a.master_gain_db))
            if hasattr(self._mw, "cb_mute_music"):
                self._mw.cb_mute_music.setChecked(bool(a.music_muted))
            if hasattr(self._mw, "cb_mute_dialogue"):
                self._mw.cb_mute_dialogue.setChecked(bool(a.dialogue_muted))
            if hasattr(self._mw, "cb_mute_vfx"):
                self._mw.cb_mute_vfx.setChecked(bool(a.vfx_muted))
            if hasattr(self._mw, "cb_duck"):
                self._mw.cb_duck.setChecked(bool(a.duck_music_under_dialogue))
        except Exception:
            pass

    # --- meters ---
    def _resolve_paths(self) -> Dict[str, str]:
        cfg = getattr(self._mw, "cfg", None)
        if cfg is None:
            return {}
        a = getattr(cfg, "audio", None)
        if a is None:
            return {}
        out_dir = getattr(cfg, "output_dir", "") or ""
        adir = os.path.join(out_dir, getattr(a, "audio_dir_name", "audio"))
        return {
            "dialogue": os.path.join(adir, getattr(a, "track_filename", "dialogue_track.wav")),
            "vfx": os.path.join(adir, getattr(a, "vfx_track_filename", "vfx_track.wav")),
            "music": os.path.join(adir, getattr(a, "music_track_filename", "music_track.wav")),
            "master": os.path.join(adir, getattr(a, "master_track_filename", "master_mix.wav")),
        }

    def _refresh_peaks(self):
        paths = self._resolve_paths()
        if not paths:
            return
        # only reload peaks when path changes; mtime caching is handled inside load_peaks
        for key, p in paths.items():
            if not p or not os.path.isfile(p):
                continue
            if self._last_paths.get(key) != p:
                self._last_paths[key] = p
            try:
                self._peaks[key] = load_peaks(p, width=512, sr=8000)
            except Exception:
                self._peaks[key] = None

    def _on_playback_state_changed(self, state):
        try:
            # QMediaPlayer.PlayingState == 1
            if int(state) == 1:
                self._meter_timer.start()
            else:
                self._meter_timer.stop()
                # decay
                self.strip_dialogue.setLevel(0.0)
                self.strip_vfx.setLevel(0.0)
                self.strip_music.setLevel(0.0)
                self.strip_master.setLevel(0.0)
                self.lbl_status.setText("Meters: idle")
        except Exception:
            pass

    def _update_meters(self):
        if self._player is None:
            return
        try:
            dur = int(self._player.duration() or 0)
            pos = int(self._player.position() or 0)
        except Exception:
            return
        if dur <= 0:
            return
        r = _clamp(pos / float(dur), 0.0, 1.0)

        def _sample(name: str) -> float:
            arr = self._peaks.get(name, None)
            if arr is None:
                return 0.0
            try:
                n = int(len(arr))
                if n <= 1:
                    return 0.0
                idx = int(round(r * (n - 1)))
                idx = max(0, min(n - 1, idx))
                v = float(arr[idx])
                # mild perceptual boost
                return float(_clamp(v ** 0.5, 0.0, 1.0))
            except Exception:
                return 0.0

        # Update (best-effort)
        self.strip_dialogue.setLevel(_sample("dialogue"))
        self.strip_vfx.setLevel(_sample("vfx"))
        self.strip_music.setLevel(_sample("music"))
        self.strip_master.setLevel(_sample("master"))
