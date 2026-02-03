from __future__ import annotations

from typing import List, Optional, Callable

import os

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget

from .config import SceneSpec, TakeSpec
from .style_packs import list_packs
from .stability_tools import compute_drift_risk
from .waveform_cache import load_peaks, load_peaks_segment


class ClipItem(QtWidgets.QGraphicsRectItem):
    """A draggable clip rectangle representing one SceneSpec."""

    _THUMB_CACHE = {}
    _THUMB_CACHE_ORDER = []
    _THUMB_CACHE_MAX = 128

    def __init__(self, idx: int, scene: SceneSpec, pps: float, y: float, h: float, on_released: Callable[[], None]):
        super().__init__(0.0, 0.0, max(20.0, float(scene.seconds) * float(pps)), float(h))
        self.idx = idx
        self.scene = scene
        self.music_track_path: Optional[str] = None  # global music bed track (optional)
        self.pps = float(pps)
        self._y = float(y)
        self._h = float(h)
        self._on_released = on_released

        self.setPos(0.0, self._y)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)

        self.setBrush(QtGui.QBrush(QtGui.QColor(60, 70, 85)))
        self.setPen(QtGui.QPen(QtGui.QColor(120, 140, 170), 1))

        self.text = QtWidgets.QGraphicsSimpleTextItem(self)
        self.text.setBrush(QtGui.QBrush(QtGui.QColor(235, 240, 245)))

        # Optional thumbnail (Flux2 init frame)
        self.thumb = QtWidgets.QGraphicsPixmapItem(self)
        self.thumb.setZValue(5)
        self.thumb.setVisible(False)

        # Optional waveforms (dialogue / vfx)
        self._wf_dialogue = QtWidgets.QGraphicsPathItem(self)
        self._wf_vfx = QtWidgets.QGraphicsPathItem(self)
        self._wf_music = QtWidgets.QGraphicsPathItem(self)
        self._wf_dialogue.setZValue(6)
        self._wf_vfx.setZValue(6)
        self._wf_music.setZValue(6)
        self._wf_dialogue.setPen(QtGui.QPen(QtGui.QColor(120, 200, 255, 170), 1))
        self._wf_vfx.setPen(QtGui.QPen(QtGui.QColor(255, 180, 120, 170), 1))
        self._wf_music.setPen(QtGui.QPen(QtGui.QColor(150, 230, 160, 140), 1))
        self._wf_dialogue.setVisible(False)
        self._wf_vfx.setVisible(False)
        self._wf_music.setVisible(False)

        # Audio lanes (A1/A2/A3) mode
        self._audio_lanes_enabled = False
        self._audio_lane_h = 18.0
        self._audio_lane_gap = 6.0
        self._audio_lane_y0 = self._h + 10.0
        self._lane_dialogue: Optional[AudioLaneClip] = None
        self._lane_vfx: Optional[AudioLaneClip] = None
        self._lane_music: Optional[AudioLaneClip] = None

        # Render overlay (progress/state)
        self._render_state = ''
        self._render_progress = 0.0
        self._render_msg = ''
        self._render_cache_hit = False

        self._ov_bg = QtWidgets.QGraphicsRectItem(self)
        self._ov_bg.setZValue(9)
        self._ov_bg.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 120)))
        self._ov_bg.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
        self._ov_bg.setVisible(False)

        self._ov_fg = QtWidgets.QGraphicsRectItem(self._ov_bg)
        self._ov_fg.setZValue(10)
        self._ov_fg.setBrush(QtGui.QBrush(QtGui.QColor(50, 200, 120, 180)))
        self._ov_fg.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))

        self._ov_txt = QtWidgets.QGraphicsSimpleTextItem(self._ov_bg)
        self._ov_txt.setZValue(11)
        self._ov_txt.setBrush(QtGui.QBrush(QtGui.QColor(240, 240, 240)))

        self._update_text()
        try:
            self._update_waveforms()
        except Exception:
            pass

    @classmethod
    def _thumb_pixmap(cls, path: str, size: int) -> QtGui.QPixmap:
        try:
            mtime = QtCore.QFileInfo(path).lastModified().toMSecsSinceEpoch()
        except Exception:
            mtime = 0
        key = (path, int(size), int(mtime))
        pm = cls._THUMB_CACHE.get(key)
        if pm is not None and not pm.isNull():
            return pm
        pm = QtGui.QPixmap(path)
        if pm.isNull():
            return pm
        pm = pm.scaled(size, size, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
        cls._THUMB_CACHE[key] = pm
        cls._THUMB_CACHE_ORDER.append(key)
        if len(cls._THUMB_CACHE_ORDER) > cls._THUMB_CACHE_MAX:
            old = cls._THUMB_CACHE_ORDER.pop(0)
            cls._THUMB_CACHE.pop(old, None)
        return pm

    def _update_thumb(self):
        try:
            # Prefer live segment preview while rendering, else latest take preview, then init-frame.
            path = None

            st = str(getattr(self.scene, '_render_state', '') or getattr(self, '_render_state', '')).strip()
            if st in ('rendering', 'queued'):
                path = getattr(self.scene, 'preview_frame_path', None)

            if not path:
                try:
                    takes = list(getattr(self.scene, 'takes', []) or [])

                    def pick(kind: str) -> Optional[str]:
                        name_key = 'active_final_take' if kind == 'final' else 'active_proxy_take'
                        active = getattr(self.scene, name_key, None)
                        # active first
                        if active:
                            for t in takes:
                                if getattr(t, 'kind', None) == kind and getattr(t, 'name', None) == active:
                                    return getattr(t, 'preview_path', None)
                        # else latest of kind
                        for t in reversed(takes):
                            if getattr(t, 'kind', None) == kind:
                                return getattr(t, 'preview_path', None)
                        return None

                    path = pick('final') or pick('proxy')
                except Exception:
                    path = None

            if not path:
                path = (
                    getattr(self.scene, 'init_frame_path', None)
                    or getattr(self.scene, 'input_image_path_override', None)
                )

            if not path or not QtCore.QFileInfo(str(path)).exists():
                self.thumb.setVisible(False)
                self.thumb.setPixmap(QtGui.QPixmap())
                return

            size = int(min(56, max(32, self._h - 10)))
            pm = self._thumb_pixmap(str(path), size)
            if pm.isNull():
                self.thumb.setVisible(False)
                return
            self.thumb.setPixmap(pm)
            w = pm.width(); h = pm.height()
            x = max(2.0, float(self.rect().width()) - float(w) - 6.0)
            y = max(2.0, float(self._h) - float(h) - 6.0)
            self.thumb.setPos(x, y)
            self.thumb.setVisible(True)
        except Exception:
            try:
                self.thumb.setVisible(False)
            except Exception:
                pass

    def _update_text(self):
        badges = ''
        # stability / drift risk badge (computed by StudioPremiere.reload_from_cfg)
        try:
            ri = str(getattr(self.scene, '_risk_icon', '') or '')
            if ri:
                badges += ri
            tip = str(getattr(self.scene, '_risk_tooltip', '') or '')
            if tip:
                self.setToolTip(tip)
        except Exception:
            pass
        # render state badges
        st = str(getattr(self.scene, '_render_state', '') or getattr(self, '_render_state', '')).strip()
        if st in ('rendering','queued'):
            badges += '⏳'
        elif st == 'done':
            badges += '✅'
        elif st == 'failed':
            badges += '❌'
        if bool(getattr(self.scene, '_clip_cache_hit', False) or getattr(self, '_render_cache_hit', False)):
            badges += '💾'

        if getattr(self.scene, 'characters', None):
            badges += '👤'
        if getattr(self.scene, 'location', None):
            badges += '📍'
        if bool(getattr(self.scene, 'hard_cut', False)):
            badges += '✂'
        if getattr(self.scene, 'init_frame_path', None):
            badges += '🖼'
        try:
            pvp = getattr(self.scene, 'preview_frame_path', None)
            if pvp and QtCore.QFileInfo(str(pvp)).exists():
                badges += '🎞'
        except Exception:
            pass

        label = (self.scene.label or f"Clip {self.idx+1}").strip()
        if badges:
            label = f"{badges} {label}"

        prompt = (self.scene.prompt or '').strip().replace('\n', ' ')
        if len(prompt) > 60:
            prompt = prompt[:57] + '…'
        mid = (self.scene.model_id_override or '').strip()
        mm = (self.scene.mode_override or '').strip()
        extra = ''
        if mid or mm:
            extra = f"  [{mm or 'INH'} | {mid or 'inherit'}]"
        flags = ''
        if getattr(self.scene, 'proxy_video_path', None):
            flags += ' ⚡'
        if getattr(self.scene, 'final_video_path', None):
            flags += ' 🎬'
        if getattr(self.scene, 'style_pack', None):
            flags += ' 🎨'

        self.text.setText(f"{label}{flags}  ({self.scene.seconds:.2f}s){extra}\n{prompt}")
        self._update_thumb()
        self._update_overlay()

    def _update_overlay(self):
        # small progress bar at bottom when rendering
        try:
            st = str(getattr(self.scene, '_render_state', '') or self._render_state or '').strip()
            p = float(getattr(self.scene, '_render_progress', self._render_progress) or 0.0)
            msg = str(getattr(self.scene, '_render_msg', self._render_msg) or '').strip()
            active = st in ('queued','rendering')
            if not active:
                self._ov_bg.setVisible(False)
                return
            w = float(self.rect().width())
            h = float(self.rect().height())
            bh = 12.0
            self._ov_bg.setRect(0.0, h - bh, w, bh)
            self._ov_fg.setRect(0.0, 0.0, max(1.0, w * max(0.0, min(1.0, p))), bh)
            self._ov_txt.setText(f"{int(max(0.0,min(1.0,p))*100)}% {msg}".strip())
            self._ov_txt.setPos(4.0, 1.0)
            self._ov_bg.setVisible(True)
        except Exception:
            try:
                self._ov_bg.setVisible(False)
            except Exception:
                pass

    def set_render_state(self, state: str, *, cache_hit: bool = False):
        self._render_state = str(state or '').strip()
        self._render_cache_hit = bool(cache_hit)
        try:
            setattr(self.scene, '_render_state', self._render_state)
            setattr(self.scene, '_clip_cache_hit', bool(cache_hit))
        except Exception:
            pass
        self._update_text()

    def set_render_progress(self, p: float, msg: str = ''):
        self._render_progress = float(p or 0.0)
        self._render_msg = str(msg or '')
        try:
            setattr(self.scene, '_render_progress', self._render_progress)
            setattr(self.scene, '_render_msg', self._render_msg)
        except Exception:
            pass
        self._update_overlay()
        self.text.setPos(6, 4)

        self._update_thumb()


    def set_audio_lanes(self, enabled: bool, *, lane_h: float = 18.0, lane_gap: float = 6.0, y0: Optional[float] = None) -> None:
        """Enable/disable separate audio lanes below the clip (Dialogue/VFX/Music)."""
        self._audio_lanes_enabled = bool(enabled)
        try:
            self._audio_lane_h = float(lane_h)
        except Exception:
            self._audio_lane_h = 18.0
        try:
            self._audio_lane_gap = float(lane_gap)
        except Exception:
            self._audio_lane_gap = 6.0
        if y0 is not None:
            try:
                self._audio_lane_y0 = float(y0)
            except Exception:
                pass
        else:
            self._audio_lane_y0 = float(self._h) + 10.0

    def _ensure_lane_items(self) -> None:
        if not getattr(self, "_audio_lanes_enabled", False):
            return
        y0 = float(getattr(self, "_audio_lane_y0", float(self._h) + 10.0))
        h = float(getattr(self, "_audio_lane_h", 18.0))
        g = float(getattr(self, "_audio_lane_gap", 6.0))
        if getattr(self, "_lane_dialogue", None) is None:
            self._lane_dialogue = AudioLaneClip(self, "dialogue", y0, h, QtGui.QPen(QtGui.QColor(120, 200, 255, 190), 1))
        if getattr(self, "_lane_vfx", None) is None:
            self._lane_vfx = AudioLaneClip(self, "vfx", y0 + h + g, h, QtGui.QPen(QtGui.QColor(255, 180, 120, 190), 1))
        if getattr(self, "_lane_music", None) is None:
            self._lane_music = AudioLaneClip(self, "music", y0 + (h + g) * 2.0, h, QtGui.QPen(QtGui.QColor(150, 230, 160, 160), 1))

    def _update_lane_items(self) -> None:
        if not getattr(self, "_audio_lanes_enabled", False):
            for it in (getattr(self, "_lane_dialogue", None), getattr(self, "_lane_vfx", None), getattr(self, "_lane_music", None)):
                try:
                    if it is not None:
                        it.setVisible(False)
                except Exception:
                    pass
            return
        self._ensure_lane_items()
        y0 = float(getattr(self, "_audio_lane_y0", float(self._h) + 10.0))
        h = float(getattr(self, "_audio_lane_h", 18.0))
        g = float(getattr(self, "_audio_lane_gap", 6.0))
        for (it, yy) in ((self._lane_dialogue, y0), (self._lane_vfx, y0 + h + g), (self._lane_music, y0 + (h + g) * 2.0)):
            try:
                if it is not None:
                    it.update_layout(yy, h)
                    it.setVisible(True)
                    it.update_waveform()
            except Exception:
                pass

    def _update_waveforms(self) -> None:
        """Update waveform paths based on per-shot audio stems."""

        # Separate audio lanes mode (A1/A2/A3)
        if getattr(self, "_audio_lanes_enabled", False):
            try:
                self._wf_dialogue.setVisible(False)
                self._wf_vfx.setVisible(False)
                self._wf_music.setVisible(False)
            except Exception:
                pass
            self._update_lane_items()
            return
        try:
            r = self.rect()
            w = float(r.width())
            h = float(r.height())
        except Exception:
            return

        # Lightweight: number of points based on pixel width
        pts = int(max(64, min(512, w)))
        lane_h = max(6.0, min(14.0, h * 0.18))
        pad = 3.0
        y_dialogue = h - (lane_h * 3.0) - pad
        y_vfx = h - (lane_h * 2.0) - pad
        y_music = h - lane_h - pad

        def _path_for(audio_path: str, y0: float) -> QtGui.QPainterPath:
            path = QtGui.QPainterPath()
            peaks = None
            try:
                peaks = load_peaks(audio_path, width=pts, sr=8000)
            except Exception:
                peaks = None
            if peaks is None:
                return path
            n = int(len(peaks))
            if n < 2:
                return path
            amp = lane_h * 0.45
            dx = w / float(n - 1)
            mid = y0 + (lane_h * 0.5)
            path.moveTo(0.0, mid)
            for i in range(n):
                x = float(i) * dx
                a = float(peaks[i])
                path.lineTo(x, mid - a * amp)
            for i in range(n - 1, -1, -1):
                x = float(i) * dx
                a = float(peaks[i])
                path.lineTo(x, mid + a * amp)
            path.closeSubpath()
            return path

        # Dialogue
        dp = getattr(self.scene, 'dialogue_audio_path', None)
        if dp and QtCore.QFileInfo(str(dp)).exists():
            try:
                self._wf_dialogue.setPath(_path_for(str(dp), y_dialogue))
                self._wf_dialogue.setVisible(True)
            except Exception:
                self._wf_dialogue.setVisible(False)
        else:
            self._wf_dialogue.setVisible(False)

        # VFX
        vp = getattr(self.scene, 'vfx_audio_path', None)
        if vp and QtCore.QFileInfo(str(vp)).exists():
            try:
                self._wf_vfx.setPath(_path_for(str(vp), y_vfx))
                self._wf_vfx.setVisible(True)
            except Exception:
                self._wf_vfx.setVisible(False)
        else:
            self._wf_vfx.setVisible(False)

        # Music bed (global track sliced to this clip time)
        mp = getattr(self, 'music_track_path', None)
        if mp and QtCore.QFileInfo(str(mp)).exists():
            try:
                # start time is clip x position / pps
                start_s = max(0.0, float(self.pos().x()) / max(1e-6, float(self.pps)))
                dur_s = max(0.0, float(w) / max(1e-6, float(self.pps)))
                peaks = load_peaks_segment(str(mp), start_sec=start_s, dur_sec=dur_s, width=pts, sr=8000)
                if peaks is None or len(peaks) < 2:
                    self._wf_music.setVisible(False)
                else:
                    # build path
                    n2 = int(len(peaks))
                    amp = lane_h * 0.45
                    dx = w / float(n2 - 1)
                    mid = y_music + (lane_h * 0.5)
                    path = QtGui.QPainterPath()
                    path.moveTo(0.0, mid)
                    for i in range(n2):
                        x2 = float(i) * dx
                        a2 = float(peaks[i])
                        path.lineTo(x2, mid - a2 * amp)
                    for i in range(n2 - 1, -1, -1):
                        x2 = float(i) * dx
                        a2 = float(peaks[i])
                        path.lineTo(x2, mid + a2 * amp)
                    path.closeSubpath()
                    self._wf_music.setPath(path)
                    self._wf_music.setVisible(True)
            except Exception:
                self._wf_music.setVisible(False)
        else:
            self._wf_music.setVisible(False)
    def refresh_geometry(self, pps: float):
        self.pps = float(pps)
        w = max(20.0, float(self.scene.seconds) * float(self.pps))
        self.setRect(0.0, 0.0, w, self._h)
        self._update_text()
        try:
            self._update_waveforms()
        except Exception:
            pass

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            p = value
            x = max(0.0, float(p.x()))
            return QtCore.QPointF(x, self._y)
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemSelectedChange:
            if bool(value):
                self.setBrush(QtGui.QBrush(QtGui.QColor(85, 95, 120)))
                self.setPen(QtGui.QPen(QtGui.QColor(220, 200, 120), 2))
            else:
                self.setBrush(QtGui.QBrush(QtGui.QColor(60, 70, 85)))
                self.setPen(QtGui.QPen(QtGui.QColor(120, 140, 170), 1))
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        super().mouseReleaseEvent(event)
        try:
            self._update_waveforms()
        except Exception:
            pass
        try:
            self._on_released()
        except Exception:
            pass


class AudioLaneClip(QtWidgets.QGraphicsRectItem):
    """A lightweight audio lane segment attached to a ClipItem (dialogue / vfx / music)."""

    def __init__(self, parent_clip: "ClipItem", lane: str, y: float, h: float, pen: QtGui.QPen):
        super().__init__(0.0, float(y), float(parent_clip.rect().width()), float(h), parent_clip)
        self.parent_clip = parent_clip
        self.lane = str(lane)
        self.setZValue(6)
        self.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 0)))
        self.setPen(QtGui.QPen(QtGui.QColor(90, 105, 130, 120), 1))

        self._wf = QtWidgets.QGraphicsPathItem(self)
        self._wf.setZValue(7)
        self._wf.setPen(pen)
        self._wf.setVisible(False)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        try:
            if self.parent_clip is not None:
                self.parent_clip.setSelected(True)
        except Exception:
            pass
        event.accept()

    def update_layout(self, y: float, h: float) -> None:
        try:
            w = float(self.parent_clip.rect().width())
        except Exception:
            w = float(self.rect().width())
        self.setRect(0.0, float(y), float(w), float(h))

    def _path_for_peaks(self, peaks, w: float, y0: float, lane_h: float) -> QtGui.QPainterPath:
        path = QtGui.QPainterPath()
        if peaks is None:
            return path
        try:
            n = int(len(peaks))
        except Exception:
            return path
        if n < 2:
            return path
        amp = float(lane_h) * 0.45
        dx = float(w) / float(n - 1)
        mid = float(y0) + (float(lane_h) * 0.5)
        path.moveTo(0.0, mid)
        for i in range(n):
            x = float(i) * dx
            a = float(peaks[i])
            path.lineTo(x, mid - a * amp)
        for i in range(n - 1, -1, -1):
            x = float(i) * dx
            a = float(peaks[i])
            path.lineTo(x, mid + a * amp)
        path.closeSubpath()
        return path

    def update_waveform(self) -> None:
        """Recompute waveform for this lane."""
        try:
            r = self.rect()
            w = float(r.width())
            y0 = float(r.y())
            lane_h = float(r.height())
        except Exception:
            return

        pts = int(max(64, min(768, w)))

        peaks = None
        try:
            if self.lane == "dialogue":
                dp = getattr(self.parent_clip.scene, "dialogue_audio_path", None)
                if dp and QtCore.QFileInfo(str(dp)).exists():
                    peaks = load_peaks(str(dp), width=pts, sr=8000)
            elif self.lane == "vfx":
                vp = getattr(self.parent_clip.scene, "vfx_audio_path", None)
                if vp and QtCore.QFileInfo(str(vp)).exists():
                    peaks = load_peaks(str(vp), width=pts, sr=8000)
            elif self.lane == "music":
                mp = getattr(self.parent_clip, "music_track_path", None)
                if mp and QtCore.QFileInfo(str(mp)).exists():
                    start_s = max(0.0, float(self.parent_clip.pos().x()) / max(1e-6, float(self.parent_clip.pps)))
                    dur_s = max(0.0, float(w) / max(1e-6, float(self.parent_clip.pps)))
                    peaks = load_peaks_segment(str(mp), start_sec=start_s, dur_sec=dur_s, width=pts, sr=8000)
        except Exception:
            peaks = None

        if peaks is None:
            self._wf.setVisible(False)
            return

        try:
            self._wf.setPath(self._path_for_peaks(peaks, w=w, y0=y0, lane_h=lane_h))
            self._wf.setVisible(True)
        except Exception:
            self._wf.setVisible(False)


class TimelineView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None, image_gen_cb=None, regen_init_cb=None, init_cache_status_cb=None, use_cache_cb=None):
        super().__init__(parent)
        self._image_gen_cb = image_gen_cb
        self._regen_init_cb = regen_init_cb
        self._init_cache_status_cb = init_cache_status_cb
        self._use_cache_cb = use_cache_cb
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.RubberBandDrag)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        # Horizontal scroll with wheel (Premiere-like)
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta)
        event.accept()

    def _clip_item_at(self, scene_pos: QtCore.QPointF):
        try:
            scn = self.scene()
            if scn is None:
                return None
            items = scn.items(scene_pos)
            for it in items:
                try:
                    if isinstance(it, ClipItem):
                        return it
                except Exception:
                    continue
        except Exception:
            return None
        return None

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        try:
            it = self._clip_item_at(self.mapToScene(event.pos()))
            if it is not None:
                try:
                    # Delegate to the StudioPremiereTab
                    owner = self.parent()
                    if owner is not None and hasattr(owner, 'play_clip'):
                        owner.play_clip(int(getattr(it, 'idx', 0)), kind='auto')
                        event.accept()
                        return
                except Exception:
                    pass
        except Exception:
            pass
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        it = None
        try:
            it = self._clip_item_at(self.mapToScene(event.pos()))
        except Exception:
            it = None
        if it is None:
            return super().contextMenuEvent(event)

        idx = int(getattr(it, 'idx', 0))
        owner = self.parent()
        menu = QtWidgets.QMenu(self)
        act_play = menu.addAction("▶ Play (auto)")
        act_proxy = menu.addAction("⚡ Play Proxy")
        act_final = menu.addAction("🎬 Play Final")
        menu.addSeparator()
        act_build = menu.addAction("🛠 Build/Refresh preview (cache) + play")
        menu.addSeparator()
        act_render_proxy = menu.addAction("⚡ Render Proxy")
        act_render_final = menu.addAction("🎬 Render Final")
        menu.addSeparator()
        act_in = menu.addAction("⟲ Set In")
        act_out = menu.addAction("⟳ Set Out")
        act_clear = menu.addAction("× Clear Range")
        menu.addSeparator()
        act_reveal = menu.addAction("📁 Reveal file")

        def _call(name: str, *args, **kwargs):
            try:
                if owner is not None and hasattr(owner, name):
                    getattr(owner, name)(*args, **kwargs)
            except Exception:
                pass

        act_play.triggered.connect(lambda: _call('play_clip', idx, 'auto'))
        act_proxy.triggered.connect(lambda: _call('play_clip', idx, 'proxy'))
        act_final.triggered.connect(lambda: _call('play_clip', idx, 'final'))
        act_build.triggered.connect(lambda: _call('build_and_play_preview', idx))
        act_render_proxy.triggered.connect(lambda: _call('render_proxy_indices', [idx]))
        act_render_final.triggered.connect(lambda: _call('render_final_indices', [idx]))
        act_in.triggered.connect(lambda: _call('set_range_in_idx', idx))
        act_out.triggered.connect(lambda: _call('set_range_out_idx', idx))
        act_clear.triggered.connect(lambda: _call('clear_range'))
        act_reveal.triggered.connect(lambda: _call('reveal_clip_file', idx))

        menu.exec(event.globalPos())
        event.accept()


class ClipInspector(QtWidgets.QWidget):
    """Inspector panel for a selected clip (SceneSpec)."""
    def __init__(self, parent=None, image_gen_cb=None, regen_init_cb=None, init_cache_status_cb=None, use_cache_cb=None):
        super().__init__(parent)
        self._image_gen_cb = image_gen_cb
        self._regen_init_cb = regen_init_cb
        self._init_cache_status_cb = init_cache_status_cb
        self._use_cache_cb = use_cache_cb
        self._scene: Optional[SceneSpec] = None
        self._apply_cb: Optional[Callable[[], None]] = None
        self._play_cb: Optional[Callable[[str], None]] = None
        self._loading = False

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("<b>Inspector</b>"))
        header.addStretch(1)
        self.cb_auto = QtWidgets.QCheckBox("Auto-apply")
        self.cb_auto.setChecked(True)
        header.addWidget(self.cb_auto)
        lay.addLayout(header)

        # Scrollable inspector body (so you can reach all controls)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        body = QtWidgets.QWidget()
        scroll.setWidget(body)
        body_lay = QtWidgets.QVBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(10)
        lay.addWidget(scroll, 1)

        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setVerticalSpacing(10)
        body_lay.addLayout(form)

        self.ed_label = QtWidgets.QLineEdit()
        self.ed_tags = QtWidgets.QLineEdit()

        self.sp_seconds = QtWidgets.QDoubleSpinBox()
        self.sp_seconds.setRange(0.1, 120.0)
        self.sp_seconds.setSingleStep(0.25)
        self.sp_seconds.setDecimals(2)

        self.cb_mode = QtWidgets.QComboBox()
        self.cb_mode.addItems(["(inherit)", "I2V", "T2V"])

        self.cb_model = QtWidgets.QComboBox()
        self.cb_model.setEditable(True)
        self.cb_model.addItems(["(inherit)"])

        # Style pack + character reference + backend override (created once; do not recreate on refresh)
        self.cb_style = QtWidgets.QComboBox()
        self.cb_style.addItem("(inherit)")
        for name in list_packs():
            self.cb_style.addItem(name)

        self.cb_use_char = QtWidgets.QCheckBox("Use global Character Ref")
        self.cb_use_char.setToolTip(
            "If checked and no per-shot input override is set, the render will use the project's global character reference image (ProjectConfig.character_image_path)."
        )

        self.cb_backend = QtWidgets.QComboBox()
        self.cb_backend.addItems(["(inherit)", "wan", "cogvideox", "ltx", "ltx2", "lingbot"])

        # --- Phase C (cinema) ---
        self.list_chars = QtWidgets.QListWidget()
        self.list_chars.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        self.list_chars.setFixedHeight(84)

        self.cb_location = QtWidgets.QComboBox()
        self.cb_location.setEditable(False)
        self.cb_location.addItem('(none)')

        self.cb_hard_cut = QtWidgets.QCheckBox('Hard cut (reset continuity)')
        self.cb_hard_cut.setToolTip('If checked, continuity resets before this shot and montage boundary defaults to cut.')

        self.ed_shot_type = QtWidgets.QLineEdit()
        self.ed_camera_move = QtWidgets.QLineEdit()
        self.ed_mood = QtWidgets.QLineEdit()
        self.ed_music_intensity = QtWidgets.QLineEdit()

        # --- LingBot (base-cam) camera controls ---
        self.cb_cam_preset = QtWidgets.QComboBox()
        self.cb_cam_preset.addItems(["static","dolly_in","dolly_out","truck_left","truck_right","pan_left","pan_right","orbit_left","orbit_right"])

        self.sp_cam_fov = QtWidgets.QDoubleSpinBox()
        self.sp_cam_fov.setRange(10.0, 120.0)
        self.sp_cam_fov.setSingleStep(1.0)
        self.sp_cam_fov.setDecimals(1)
        self.sp_cam_fov.setValue(50.0)

        self.sp_cam_amount = QtWidgets.QDoubleSpinBox()
        self.sp_cam_amount.setRange(0.0, 2.0)
        self.sp_cam_amount.setSingleStep(0.05)
        self.sp_cam_amount.setDecimals(2)
        self.sp_cam_amount.setValue(1.0)

        self.sp_cam_strength = QtWidgets.QDoubleSpinBox()
        self.sp_cam_strength.setRange(0.25, 3.0)
        self.sp_cam_strength.setSingleStep(0.05)
        self.sp_cam_strength.setDecimals(2)
        self.sp_cam_strength.setValue(1.0)

        self.init_preview = QtWidgets.QLabel()
        self.init_preview.setFixedSize(192, 108)
        self.init_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.init_preview.setStyleSheet('background:#222;border:1px solid #444;')
        self.lbl_init_path = QtWidgets.QLabel('Init frame: (none)')
        self.lbl_init_path.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.btn_regen_init = QtWidgets.QPushButton('Regenerate init frame')
        self.btn_regen_init.setToolTip('Generate/refresh an init frame for this shot')

        # Init Image Settings (per-clip)
        self.cb_init_preset = QtWidgets.QComboBox()
        self.cb_init_policy = QtWidgets.QComboBox()
        self.lbl_cache = QtWidgets.QLabel('Cache: (n/a)')
        self.lbl_cache.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.btn_use_cache = QtWidgets.QPushButton('Use cache this time')
        self.btn_use_cache.setToolTip('Force using the cached init frame if available (no generation).')

        def _add_cb_item(cb, label, data):
            cb.addItem(label)
            cb.setItemData(cb.count()-1, data)

        _add_cb_item(self.cb_init_preset, '(project default)', '')
        _add_cb_item(self.cb_init_preset, 'Z-Image Turbo — default', 'zimage_turbo')
        _add_cb_item(self.cb_init_preset, 'Z-Image (full)', 'zimage')
        _add_cb_item(self.cb_init_preset, 'Flux2 (bnb 4bit)', 'flux2_bnb4bit')
        _add_cb_item(self.cb_init_preset, 'Flux2 (full)', 'flux2')
        _add_cb_item(self.cb_init_preset, 'Flux1 schnell', 'flux1_schnell')
        _add_cb_item(self.cb_init_preset, 'SDXL Lightning (4-step)', 'sdxl_lightning_4step')
        _add_cb_item(self.cb_init_preset, 'SDXL base', 'sdxl_base')
        _add_cb_item(self.cb_init_preset, 'SDXL turbo', 'sdxl_turbo')

        _add_cb_item(self.cb_init_policy, '(project default)', '')
        _add_cb_item(self.cb_init_policy, 'necessary (use cache)', 'necessary')
        _add_cb_item(self.cb_init_policy, 'always_refine (regen)', 'always_refine')


        # Input override (per clip)
        inp_row = QtWidgets.QHBoxLayout()
        self.ed_input = QtWidgets.QLineEdit()
        self.btn_browse = QtWidgets.QPushButton("…")
        self.btn_clear = QtWidgets.QPushButton("✕")
        self.btn_ai = QtWidgets.QPushButton("AI")
        self.btn_browse.setFixedWidth(28)
        self.btn_clear.setFixedWidth(28)
        self.btn_ai.setFixedWidth(40)
        inp_row.addWidget(self.ed_input, 1)
        inp_row.addWidget(self.btn_browse)
        inp_row.addWidget(self.btn_clear)
        inp_row.addWidget(self.btn_ai)
        inp_wrap = QtWidgets.QWidget()
        inp_wrap.setLayout(inp_row)

        self.txt_prompt = QtWidgets.QPlainTextEdit()
        self.txt_prompt.setPlaceholderText("Prompt du clip…")

        self.txt_neg = QtWidgets.QPlainTextEdit()
        self.txt_neg.setPlaceholderText("Negative prompt (optionnel)…")

        # Dialogue (ties into the Audio tab / Piper TTS)
        self.txt_dialogue = QtWidgets.QPlainTextEdit()
        self.txt_dialogue.setPlaceholderText("Dialogue (optionnel)…")
        self.ed_dlg_voice = QtWidgets.QLineEdit()
        self.ed_dlg_voice.setPlaceholderText("voice override (optionnel) ...")
        self.ed_dlg_lang = QtWidgets.QLineEdit()
        self.ed_dlg_lang.setPlaceholderText("lang override (ex: fr, en)...")

        def opt_int(minv, maxv, default):
            w = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(w); h.setContentsMargins(0,0,0,0)
            cb = QtWidgets.QCheckBox("override")
            sp = QtWidgets.QSpinBox()
            sp.setRange(minv, maxv)
            sp.setValue(default)
            h.addWidget(cb)
            h.addWidget(sp, 1)
            return w, cb, sp

        def opt_float(rmin, rmax, step, default):
            w = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(w); h.setContentsMargins(0,0,0,0)
            cb = QtWidgets.QCheckBox("override")
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(rmin, rmax)
            sp.setSingleStep(step)
            sp.setDecimals(2)
            sp.setValue(default)
            h.addWidget(cb)
            h.addWidget(sp, 1)
            return w, cb, sp

        self.w_seed, self.cb_seed, self.sp_seed = opt_int(0, 2_147_483_647, 1234)
        self.w_steps, self.cb_steps, self.sp_steps = opt_int(1, 200, 30)
        self.w_cfg, self.cb_cfg, self.sp_cfg = opt_float(0.0, 30.0, 0.1, 7.0)
        self.w_cfg2, self.cb_cfg2, self.sp_cfg2 = opt_float(0.0, 30.0, 0.1, 7.0)

        # Scheduler override (per-clip)
        self.w_sched = QtWidgets.QWidget()
        hs = QtWidgets.QHBoxLayout(self.w_sched); hs.setContentsMargins(0,0,0,0)
        self.cb_sched = QtWidgets.QComboBox()
        self.cb_sched.addItems(["(inherit)", "default", "UniPC"])
        self.sp_flow_shift = QtWidgets.QDoubleSpinBox()
        self.sp_flow_shift.setRange(-20.0, 20.0)
        self.sp_flow_shift.setSingleStep(0.5)
        self.sp_flow_shift.setDecimals(2)
        self.sp_flow_shift.setValue(5.0)
        self.sp_flow_shift.setToolTip("Only used for UniPC scheduler. Typical: 5.0")
        hs.addWidget(self.cb_sched, 1)
        hs.addWidget(QtWidgets.QLabel("flow_shift"))
        hs.addWidget(self.sp_flow_shift)
        self.sp_flow_shift.setEnabled(False)

        self.cb_trans_mode = QtWidgets.QComboBox()
        self.cb_trans_mode.addItems(["(inherit)", "cut", "crossfade", "flow"])

        self.sp_trans = QtWidgets.QSpinBox()
        self.sp_trans.setRange(0, 240)
        self.cb_ease = QtWidgets.QComboBox()
        self.cb_ease.addItems(["(inherit)", "linear", "smoothstep", "ease_in_out"])

        form.addRow("Label", self.ed_label)
        form.addRow("Tags", self.ed_tags)
        form.addRow("Duration (s)", self.sp_seconds)
        form.addRow("Mode", self.cb_mode)
        form.addRow("Model override", self.cb_model)
        form.addRow("Input override (I2V)", inp_wrap)
        form.addRow("Style pack", self.cb_style)
        form.addRow("Character", self.cb_use_char)
        form.addRow("Backend", self.cb_backend)
        form.addRow("Characters", self.list_chars)
        form.addRow("Location", self.cb_location)
        form.addRow("Hard cut", self.cb_hard_cut)
        form.addRow("Shot type", self.ed_shot_type)
        form.addRow("Camera move", self.ed_camera_move)

        form.addRow("Cam preset (LingBot)", self.cb_cam_preset)
        form.addRow("Cam FOV° (LingBot)", self.sp_cam_fov)
        form.addRow("Cam amount (LingBot)", self.sp_cam_amount)
        form.addRow("Cam strength (LingBot)", self.sp_cam_strength)
        form.addRow("Mood", self.ed_mood)
        form.addRow("Music intensity", self.ed_music_intensity)

        init_box = QtWidgets.QWidget()
        init_lay = QtWidgets.QVBoxLayout(init_box); init_lay.setContentsMargins(0,0,0,0)
        init_lay.addWidget(self.init_preview)
        init_lay.addWidget(self.lbl_init_path)
        init_lay.addWidget(self.btn_regen_init)
        # settings
        row_set = QtWidgets.QGridLayout()
        row_set.setContentsMargins(0,0,0,0)
        row_set.addWidget(QtWidgets.QLabel('Preset'), 0, 0)
        row_set.addWidget(self.cb_init_preset, 0, 1)
        row_set.addWidget(QtWidgets.QLabel('Policy'), 1, 0)
        row_set.addWidget(self.cb_init_policy, 1, 1)
        row_set.addWidget(self.lbl_cache, 2, 0, 1, 2)
        row_set.addWidget(self.btn_use_cache, 3, 0, 1, 2)
        init_lay.addLayout(row_set)
        form.addRow("Init frame", init_box)

        form.addRow("Prompt", self.txt_prompt)
        form.addRow("Negative", self.txt_neg)
        form.addRow("Dialogue", self.txt_dialogue)
        form.addRow("Dlg voice", self.ed_dlg_voice)
        form.addRow("Dlg lang", self.ed_dlg_lang)
        form.addRow("Seed", self.w_seed)
        form.addRow("Steps", self.w_steps)
        form.addRow("Guidance", self.w_cfg)
        form.addRow("Guidance 2", self.w_cfg2)
        form.addRow("Scheduler", self.w_sched)
        form.addRow("Transition mode", self.cb_trans_mode)
        form.addRow("Transition frames", self.sp_trans)
        form.addRow("Transition ease", self.cb_ease)

        # --- Retakes / Takes manager ---
        self.take_preview = QtWidgets.QLabel()
        self.take_preview.setFixedSize(192, 108)
        self.take_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.take_preview.setStyleSheet('background:#111;border:1px solid #333;')

        self.lbl_take_path = QtWidgets.QLabel('Take: (none)')
        self.lbl_take_path.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)

        self.list_takes = QtWidgets.QListWidget()
        self.list_takes.setFixedHeight(120)
        self.list_takes.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_takes.customContextMenuRequested.connect(self._on_take_context_menu)
        self.list_takes.itemDoubleClicked.connect(lambda _it=None: self._audition_selected_take())

        self.btn_take_use_proxy = QtWidgets.QPushButton('Use as Proxy')
        self.btn_take_use_final = QtWidgets.QPushButton('Use as Final')
        self.btn_take_import = QtWidgets.QPushButton('Import…')
        self.btn_take_remove = QtWidgets.QPushButton('Remove')
        self.btn_take_open = QtWidgets.QPushButton('Open')

        takes_box = QtWidgets.QWidget()
        tb = QtWidgets.QVBoxLayout(takes_box); tb.setContentsMargins(0,0,0,0)
        tb.addWidget(self.take_preview)
        tb.addWidget(self.lbl_take_path)
        tb.addWidget(self.list_takes, 1)
        rowt = QtWidgets.QHBoxLayout()
        rowt.addWidget(self.btn_take_use_proxy)
        rowt.addWidget(self.btn_take_use_final)
        rowt.addWidget(self.btn_take_import)
        rowt.addWidget(self.btn_take_remove)
        rowt.addWidget(self.btn_take_open)
        tb.addLayout(rowt)
        form.addRow('Takes / Retakes', takes_box)

        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_apply.setDefault(True)
        lay.addWidget(self.btn_apply)

        # Auto-apply (debounced)
        self._auto_timer = QtCore.QTimer(self)
        self._auto_timer.setSingleShot(True)
        self._auto_timer.timeout.connect(self.apply)

        # Scheduler UI helper
        try:
            self.cb_sched.currentTextChanged.connect(self._on_sched_changed)
            self.sp_flow_shift.valueChanged.connect(self._on_sched_changed)
        except Exception:
            pass

        def hook(widget, signal_name):
            sig = getattr(widget, signal_name, None)
            if sig is not None:
                try:
                    sig.connect(self._schedule_auto_apply)
                except Exception:
                    pass

        # wire change signals
        hook(self.ed_label, "textChanged")
        hook(self.ed_tags, "textChanged")
        hook(self.sp_seconds, "valueChanged")
        hook(self.cb_mode, "currentTextChanged")
        hook(self.cb_model, "currentTextChanged")
        hook(self.ed_input, "textChanged")
        hook(self.cb_style, "currentTextChanged")
        hook(self.cb_use_char, "toggled")
        hook(self.cb_backend, "currentTextChanged")
        hook(self.cb_location, 'currentTextChanged')
        hook(self.cb_hard_cut, 'toggled')
        hook(self.ed_shot_type, 'textChanged')
        hook(self.ed_camera_move, 'textChanged')
        hook(self.ed_mood, 'textChanged')
        hook(self.ed_music_intensity, 'textChanged')
        try:
            self.list_chars.itemSelectionChanged.connect(self._schedule_auto_apply)
        except Exception:
            pass
        hook(self.txt_prompt, "textChanged")
        hook(self.txt_neg, "textChanged")
        hook(self.txt_dialogue, "textChanged")
        hook(self.ed_dlg_voice, "textChanged")
        hook(self.ed_dlg_lang, "textChanged")
        hook(self.txt_dialogue, "textChanged")
        hook(self.ed_dlg_voice, "textChanged")
        hook(self.ed_dlg_lang, "textChanged")

        hook(self.cb_seed, "toggled"); hook(self.sp_seed, "valueChanged")
        hook(self.cb_steps, "toggled"); hook(self.sp_steps, "valueChanged")
        hook(self.cb_cfg, "toggled"); hook(self.sp_cfg, "valueChanged")
        hook(self.cb_cfg2, "toggled"); hook(self.sp_cfg2, "valueChanged")

        hook(self.cb_sched, "currentTextChanged"); hook(self.sp_flow_shift, "valueChanged")

        hook(self.sp_trans, "valueChanged")
        hook(self.cb_ease, "currentTextChanged")

        # buttons
        self.btn_apply.clicked.connect(self.apply)
        self.btn_browse.clicked.connect(self._browse_input)
        self.btn_clear.clicked.connect(lambda: self.ed_input.setText(""))
        self.btn_ai.clicked.connect(self._ai_gen_input)
        self.btn_regen_init.clicked.connect(self._on_regen_init)
        self.btn_use_cache.clicked.connect(self._on_use_cache)
        hook(self.cb_init_preset, 'currentIndexChanged')
        hook(self.cb_init_policy, 'currentIndexChanged')

        # takes
        try:
            self.list_takes.itemSelectionChanged.connect(self._on_take_selected)
        except Exception:
            pass
        self.btn_take_use_proxy.clicked.connect(self._on_take_use_proxy)
        self.btn_take_use_final.clicked.connect(self._on_take_use_final)
        self.btn_take_import.clicked.connect(self._on_take_import)
        self.btn_take_remove.clicked.connect(self._on_take_remove)
        self.btn_take_open.clicked.connect(self._on_take_open)

        self.setEnabled(False)

    def _update_init_preview(self):
        s = self._scene
        if s is None:
            self.init_preview.setPixmap(QtGui.QPixmap())
            self.lbl_init_path.setText("Init frame: (none)")
            return
        # While rendering, prefer live segment preview if available.
        path = None
        try:
            st = str(getattr(s, '_render_state', '') or '').strip()
            if st in ('rendering', 'queued'):
                path = getattr(s, 'preview_frame_path', None)
        except Exception:
            path = None
        if not path:
            path = getattr(s, "init_frame_path", None) or getattr(s, "input_image_path_override", None)
        if path and isinstance(path, str) and QtCore.QFileInfo(path).exists():
            pm = QtGui.QPixmap(path)
            if not pm.isNull():
                pm = pm.scaled(self.init_preview.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
                self.init_preview.setPixmap(pm)
                label = "Live preview" if (str(getattr(s, '_render_state', '') or '').strip() in ('rendering','queued') and getattr(s, 'preview_frame_path', None) == path) else "Init frame"
                self.lbl_init_path.setText(f"{label}: {path}")
                return
        self.init_preview.setPixmap(QtGui.QPixmap())
        self.lbl_init_path.setText("Init frame: (none)")

    # --- Takes / retakes ---
    def _refresh_takes_ui(self) -> None:
        s = self._scene
        self.list_takes.blockSignals(True)
        try:
            self.list_takes.clear()
            self.take_preview.setPixmap(QtGui.QPixmap())
            self.lbl_take_path.setText('Take: (none)')
            if s is None:
                return
            takes = list(getattr(s, 'takes', []) or [])
            for i, t in enumerate(takes):
                kind = str(getattr(t, 'kind', '')) or 'take'
                name = str(getattr(t, 'name', f'{kind}_{i+1:02d}'))
                created = str(getattr(t, 'created_at', '') or '')
                vid = str(getattr(t, 'video_path', '') or '')
                mark = ''
                try:
                    if kind == 'proxy' and getattr(s, 'active_proxy_take', None) == name:
                        mark = ' ⚡'
                    if kind == 'final' and getattr(s, 'active_final_take', None) == name:
                        mark = ' 🎬'
                except Exception:
                    pass
                rating = 0
                try:
                    rating = int(getattr(t, 'rating', 0) or 0)
                except Exception:
                    rating = 0
                rating = max(0, min(5, rating))
                tags = str(getattr(t, 'tags', '') or '').strip()
                stars = ('★' * rating)
                txt = f"{(stars + ' ') if stars else ''}{kind}:{name}{mark}"
                if tags:
                    txt += f"  [{tags}]"
                if created:
                    txt += f"  ({created})"
                it = QtWidgets.QListWidgetItem(txt)
                it.setData(QtCore.Qt.ItemDataRole.UserRole, int(i))
                if vid:
                    it.setToolTip(vid)
                self.list_takes.addItem(it)
        finally:
            self.list_takes.blockSignals(False)

    def _selected_take_index(self) -> int:
        try:
            it = self.list_takes.currentItem()
            if it is None:
                return -1
            return int(it.data(QtCore.Qt.ItemDataRole.UserRole))
        except Exception:
            return -1

    def _selected_take(self) -> Optional[TakeSpec]:
        s = self._scene
        if s is None:
            return None
        i = self._selected_take_index()
        takes = list(getattr(s, 'takes', []) or [])
        if i < 0 or i >= len(takes):
            return None
        t = takes[i]
        return t if isinstance(t, TakeSpec) else None

    def _on_take_selected(self):
        t = self._selected_take()
        if t is None:
            self.take_preview.setPixmap(QtGui.QPixmap())
            self.lbl_take_path.setText('Take: (none)')
            return
        p = getattr(t, 'preview_path', None)
        if p and isinstance(p, str) and QtCore.QFileInfo(p).exists():
            pm = QtGui.QPixmap(p)
            if not pm.isNull():
                pm = pm.scaled(self.take_preview.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
                self.take_preview.setPixmap(pm)
        else:
            self.take_preview.setPixmap(QtGui.QPixmap())
        vp = str(getattr(t, 'video_path', '') or '')
        self.lbl_take_path.setText(f"Take: {vp if vp else '(missing)'}")

    def _audition_selected_take(self) -> None:
        """Play the selected take in the Program Monitor (if available)."""
        t = self._selected_take()
        if t is None:
            return
        vp = str(getattr(t, "video_path", "") or "").strip()
        if not vp:
            return
        try:
            if callable(self._play_cb):
                self._play_cb(vp)
                return
        except Exception:
            pass
        # Fallback: open externally
        try:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(vp))
        except Exception:
            pass

    def _on_take_context_menu(self, pos: QtCore.QPoint) -> None:
        t = self._selected_take()
        if t is None:
            return
        menu = QtWidgets.QMenu(self)
        act_aud = menu.addAction("▶ Audition in Monitor")
        act_use_proxy = menu.addAction("⚡ Set Active Proxy")
        act_use_final = menu.addAction("🎬 Set Active Final")
        sub_rate = menu.addMenu("Rate ★")
        act_edit_tags = menu.addAction("🏷 Edit tags…")
        act_edit_note = menu.addAction("📝 Edit note…")
        menu.addSeparator()
        act_reveal = menu.addAction("📁 Reveal file")
        act_open = menu.addAction("Open externally")
        menu.addSeparator()
        act_remove = menu.addAction("🗑 Remove take")

        def _reveal(path: str):
            try:
                import os
                p = os.path.abspath(path)
                if os.path.exists(p):
                    QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(os.path.dirname(p)))
            except Exception:
                pass

        act_aud.triggered.connect(lambda: self._audition_selected_take())
        act_use_proxy.triggered.connect(lambda: self._set_active_take(kind="proxy", take=t))
        act_use_final.triggered.connect(lambda: self._set_active_take(kind="final", take=t))
        act_reveal.triggered.connect(lambda: _reveal(str(getattr(t, "video_path", "") or "")))
        act_open.triggered.connect(lambda: self._on_take_open())
        act_remove.triggered.connect(lambda: self._on_take_remove())

        # Rate submenu (0..5)
        try:
            for n in range(0, 6):
                lab = "Clear" if n == 0 else ("★" * n)
                a = sub_rate.addAction(lab)
                a.triggered.connect(lambda _chk=False, n=n: self._set_take_rating(t, n))
        except Exception:
            pass

        act_edit_tags.triggered.connect(lambda: self._edit_take_tags(t))
        act_edit_note.triggered.connect(lambda: self._edit_take_note(t))

        menu.exec(self.list_takes.mapToGlobal(pos))

    def _set_active_take(self, *, kind: str, take: TakeSpec):
        s = self._scene
        if s is None or take is None:
            return
        kind = str(kind).lower().strip()
        name = str(getattr(take, 'name', '') or '')
        vid = str(getattr(take, 'video_path', '') or '')
        if not vid:
            return
        try:
            if kind == 'proxy':
                s.proxy_video_path = vid
                s.active_proxy_take = name
            elif kind == 'final':
                s.final_video_path = vid
                s.active_final_take = name
        except Exception:
            pass
        try:
            if callable(self._apply_cb):
                self._apply_cb()
        except Exception:
            pass
        self._refresh_takes_ui()

    def _on_take_use_proxy(self):
        t = self._selected_take()
        if t is None:
            return
        self._set_active_take(kind='proxy', take=t)

    def _on_take_use_final(self):
        t = self._selected_take()
        if t is None:
            return
        self._set_active_take(kind='final', take=t)

    def _on_take_import(self):
        s = self._scene
        if s is None:
            return
        kind, ok = QtWidgets.QInputDialog.getItem(self, 'Import take', 'Kind', ['final', 'proxy', 'custom'], 0, False)
        if not ok:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select video take', '.', 'Video (*.mp4 *.mov *.mkv *.webm);;All files (*)')
        if not path:
            return
        # Create a new take entry
        takes = list(getattr(s, 'takes', []) or [])
        n_same = 0
        for t in takes:
            if getattr(t, 'kind', '') == str(kind):
                n_same += 1
        name = f"{kind}_{n_same+1:02d}"

        # try generating preview
        prev_png = None
        try:
            import os
            from .media import extract_frame_ffmpeg
            base, _ = os.path.splitext(str(path))
            prev_cand = base + '_preview.png'
            r = extract_frame_ffmpeg(str(path), 0.05, prev_cand)
            if r.ok:
                prev_png = r.path
        except Exception:
            prev_png = None

        import time
        takes.append(TakeSpec(
            name=name,
            kind=str(kind),
            video_path=str(path),
            preview_path=str(prev_png) if prev_png else None,
            seed=getattr(s, 'seed', None),
            created_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            note='import',
        ))
        try:
            s.takes = takes
        except Exception:
            setattr(s, 'takes', takes)

        # Auto-activate imported take for matching kind
        if str(kind) == 'proxy':
            self._set_active_take(kind='proxy', take=takes[-1])
        elif str(kind) == 'final':
            self._set_active_take(kind='final', take=takes[-1])
        else:
            self._refresh_takes_ui()

    def _on_take_remove(self):
        s = self._scene
        if s is None:
            return
        i = self._selected_take_index()
        takes = list(getattr(s, 'takes', []) or [])
        if i < 0 or i >= len(takes):
            return
        t = takes[i]
        name = str(getattr(t, 'name', '') or '')
        kind = str(getattr(t, 'kind', '') or '')
        if QtWidgets.QMessageBox.question(self, 'Remove take', f"Remove {kind}:{name} from the project? (File is not deleted)") != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        takes.pop(i)
        try:
            s.takes = takes
        except Exception:
            setattr(s, 'takes', takes)
        # If it was active, clear active pointer (keep current video_path unchanged)
        try:
            if kind == 'proxy' and getattr(s, 'active_proxy_take', None) == name:
                s.active_proxy_take = None
            if kind == 'final' and getattr(s, 'active_final_take', None) == name:
                s.active_final_take = None
        except Exception:
            pass
        try:
            if callable(self._apply_cb):
                self._apply_cb()
        except Exception:
            pass
        self._refresh_takes_ui()


    def _set_take_rating(self, take: TakeSpec, rating: int) -> None:
        if take is None:
            return
        try:
            take.rating = int(rating)
        except Exception:
            take.rating = 0
        self._refresh_takes_ui()
        try:
            if callable(self._apply_cb):
                self._apply_cb()
        except Exception:
            pass

    def _edit_take_tags(self, take: TakeSpec) -> None:
        if take is None:
            return
        cur = str(getattr(take, "tags", "") or "")
        txt, ok = QtWidgets.QInputDialog.getText(self, "Take tags", "Tags (comma-separated):", text=cur)
        if not ok:
            return
        try:
            take.tags = str(txt or "").strip()
        except Exception:
            pass
        self._refresh_takes_ui()
        try:
            if callable(self._apply_cb):
                self._apply_cb()
        except Exception:
            pass

    def _edit_take_note(self, take: TakeSpec) -> None:
        if take is None:
            return
        cur = str(getattr(take, "note", "") or "")
        txt, ok = QtWidgets.QInputDialog.getMultiLineText(self, "Take note", "Note:", cur)
        if not ok:
            return
        try:
            take.note = str(txt or "")
        except Exception:
            pass
        self._refresh_takes_ui()
        try:
            if callable(self._apply_cb):
                self._apply_cb()
        except Exception:
            pass

    def _on_take_open(self):
        t = self._selected_take()
        if t is None:
            return
        vp = str(getattr(t, 'video_path', '') or '')
        if not vp:
            return
        try:
            url = QtCore.QUrl.fromLocalFile(vp)
            QtGui.QDesktopServices.openUrl(url)
        except Exception:
            pass

    def _refresh_init_cache_status(self):
        s = self._scene
        if s is None:
            self.lbl_cache.setText('Cache: (n/a)')
            return
        cb = getattr(self, '_init_cache_status_cb', None)
        if not callable(cb):
            self.lbl_cache.setText('Cache: (n/a)')
            return
        try:
            st = cb(s)
            # st: InitCacheStatus(cache_key, cached_path, exists)
            key = getattr(st, 'cache_key', '') or ''
            exists = bool(getattr(st, 'exists', False))
            pth = getattr(st, 'cached_path', None)
            if exists and pth:
                self.lbl_cache.setText(f'Cache: HIT ({key[:10]}…)')
            else:
                self.lbl_cache.setText(f'Cache: MISS ({key[:10]}…)')
        except Exception as e:
            self.lbl_cache.setText(f'Cache: error ({e})')

    def _on_use_cache(self):
        if self._scene is None:
            return
        cb = getattr(self, '_use_cache_cb', None) or getattr(self, '_regen_init_cb', None)
        if not callable(cb):
            QtWidgets.QMessageBox.information(self, 'Init cache', 'Cache helper not available.')
            return
        try:
            # try calling cb(scene, use_cache_only=True) if supported
            try:
                out = cb(self._scene, True)
            except TypeError:
                out = cb(self._scene)
            if isinstance(out, str) and out:
                try:
                    self._scene.init_frame_path = out
                except Exception:
                    pass
            self._update_init_preview()
            self._refresh_init_cache_status()
            if self._apply_cb:
                self._apply_cb()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Init cache', str(e))

    def _on_regen_init(self):
        if self._scene is None:
            return
        cb = getattr(self, "_regen_init_cb", None) or getattr(self, "_image_gen_cb", None)
        if not callable(cb):
            QtWidgets.QMessageBox.information(self, "Init frame", "Init generator not available.")
            return
        try:
            out = cb(self._scene)
            # Some callbacks return path, some return result objects
            if isinstance(out, str) and out:
                try:
                    self._scene.init_frame_path = out
                except Exception:
                    pass
            self._update_init_preview()
            if self._apply_cb:
                self._apply_cb()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Init frame", str(e))

    def _browse_input(self):
        """Browse for a per-clip input image override (I2V anchor).

        This sets the inspector field only; the SceneSpec is updated on Apply/Auto-apply.
        """
        try:
            start_dir = ""
            cur = (self.ed_input.text() or "").strip()
            if cur and QtCore.QFileInfo(cur).exists():
                try:
                    start_dir = str(QtCore.QFileInfo(cur).absolutePath())
                except Exception:
                    start_dir = ""
            if not start_dir:
                try:
                    # Prefer project directory if we can infer it from init frame or existing assets.
                    s = self._scene
                    cand = ""
                    for k in ("init_frame_path", "preview_frame_path"):
                        v = str(getattr(s, k, "") or "")
                        if v and QtCore.QFileInfo(v).exists():
                            cand = str(QtCore.QFileInfo(v).absolutePath())
                            break
                    start_dir = cand
                except Exception:
                    start_dir = ""

            filt = "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*)"
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select input image", start_dir, filt)
            if path:
                self.ed_input.setText(path)
                self._update_init_preview()
                self._schedule_auto_apply()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Browse", str(e))

    def _ai_gen_input(self):
        """Generate an input image via the project's init-frame generator, then use it as input override."""
        if self._scene is None:
            return
        cb = getattr(self, "_image_gen_cb", None) or getattr(self, "_regen_init_cb", None)
        if not callable(cb):
            QtWidgets.QMessageBox.information(self, "AI input", "Image generator not available.")
            return
        try:
            out = cb(self._scene)
            # Some callbacks return path, some return result objects
            path = None
            if isinstance(out, str):
                path = out
            else:
                path = getattr(out, "path", None)
            if path:
                self.ed_input.setText(str(path))
                try:
                    # Also set init frame so the inspector thumbnail updates immediately.
                    self._scene.init_frame_path = str(path)
                except Exception:
                    pass
                self._update_init_preview()
                self._schedule_auto_apply()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "AI input", str(e))

    def _schedule_auto_apply(self):
        if self._loading:
            return
        if self.cb_auto.isChecked():
            self._auto_timer.start(350)

    def _on_sched_changed(self, *args, **kwargs):
        try:
            is_uni = (self.cb_sched.currentText().strip().lower().startswith('uni'))
            self.sp_flow_shift.setEnabled(bool(is_uni))
        except Exception:
            pass

    def set_model_items(self, items: List[str]):
        cur = self.cb_model.currentText()
        self.cb_model.clear()
        self.cb_model.addItem("(inherit)")
        # NOTE: style/character/backend widgets are created once in __init__ (do not recreate them here).
        for it in items:
            if it and it not in ("(inherit)",):
                self.cb_model.addItem(it)
        self.cb_model.setCurrentText(cur if cur else "(inherit)")

    def set_character_items(self, names: List[str]):
        cur_sel = {it.text() for it in self.list_chars.selectedItems()}
        self.list_chars.clear()
        for n in (names or []):
            it = QtWidgets.QListWidgetItem(str(n))
            self.list_chars.addItem(it)
            if str(n) in cur_sel:
                it.setSelected(True)

    def set_location_items(self, names: List[str]):
        cur = self.cb_location.currentText()
        self.cb_location.blockSignals(True)
        try:
            self.cb_location.clear()
            self.cb_location.addItem('(none)')
            for n in (names or []):
                if str(n).strip():
                    self.cb_location.addItem(str(n).strip())
            if cur and cur != '(none)':
                self.cb_location.setCurrentText(cur)
        finally:
            self.cb_location.blockSignals(False)

    def bind(self, apply_cb: Callable[[], None], play_cb: Optional[Callable[[str], None]] = None):
        self._apply_cb = apply_cb
        self._play_cb = play_cb

    def set_scene(self, s: Optional[SceneSpec]):
        self._scene = s
        if s is None:
            self.setEnabled(False)
            return

        self._loading = True
        try:
            self.setEnabled(True)
            self.ed_label.setText(s.label or "")
            self.ed_tags.setText(s.tags or "")
            self.sp_seconds.setValue(float(max(0.1, s.seconds)))

            self.cb_mode.setCurrentText(s.mode_override or "(inherit)")
            self.cb_model.setCurrentText(s.model_id_override or "(inherit)")
            self.ed_input.setText(s.input_image_path_override or "")

            st = (getattr(s, "style_pack", None) or "").strip()
            self.cb_style.setCurrentText(st if st else "(inherit)")
            self.cb_use_char.setChecked(bool(getattr(s, "use_character_ref", False)))
            bk = (getattr(s, "backend_override", None) or "").strip()
            self.cb_backend.setCurrentText(bk if bk else "(inherit)")

            # Phase C
            try:
                self.cb_hard_cut.setChecked(bool(getattr(s, "hard_cut", False)))
            except Exception:
                self.cb_hard_cut.setChecked(False)
            try:
                loc = (getattr(s, "location", "") or "").strip()
                self.cb_location.setCurrentText(loc if loc else "(none)")
            except Exception:
                self.cb_location.setCurrentText("(none)")
            try:
                self.ed_shot_type.setText((getattr(s, "shot_type", "") or ""))
                self.ed_camera_move.setText((getattr(s, "camera_move", "") or ""))
                self.ed_mood.setText((getattr(s, "mood", "") or ""))
                self.ed_music_intensity.setText((getattr(s, "music_intensity", "") or ""))

                # LingBot camera controls
                try:
                    self.cb_cam_preset.setCurrentText((getattr(s, "camera_path_preset", "static") or "static"))
                    self.sp_cam_fov.setValue(float(getattr(s, "camera_fov_deg", 50.0) or 50.0))
                    self.sp_cam_amount.setValue(float(getattr(s, "camera_path_amount", 1.0) or 1.0))
                    self.sp_cam_strength.setValue(float(getattr(s, "camera_path_strength", 1.0) or 1.0))
                except Exception:
                    pass
            except Exception:
                pass
            # characters selection
            try:
                wanted = set(getattr(s, "characters", []) or [])
                self.list_chars.blockSignals(True)
                for i in range(self.list_chars.count()):
                    it = self.list_chars.item(i)
                    it.setSelected(it.text() in wanted)
            finally:
                try:
                    self.list_chars.blockSignals(False)
                except Exception:
                    pass

            self.txt_prompt.setPlainText(s.prompt or "")
            self.txt_neg.setPlainText(s.negative_prompt or "")

            self.txt_dialogue.setPlainText(getattr(s, "dialogue_text", "") or "")
            self.ed_dlg_voice.setText(getattr(s, "dialogue_voice", "") or "")
            self.ed_dlg_lang.setText(getattr(s, "dialogue_language", "") or "")

            def set_opt_int(cb, sp, val):
                if val is None:
                    cb.setChecked(False)
                else:
                    cb.setChecked(True); sp.setValue(int(val))

            def set_opt_float(cb, sp, val):
                if val is None:
                    cb.setChecked(False)
                else:
                    cb.setChecked(True); sp.setValue(float(val))

            set_opt_int(self.cb_seed, self.sp_seed, s.seed)
            set_opt_int(self.cb_steps, self.sp_steps, s.num_inference_steps)
            set_opt_float(self.cb_cfg, self.sp_cfg, s.guidance_scale)
            set_opt_float(self.cb_cfg2, self.sp_cfg2, s.guidance_scale_2)

            # Scheduler overrides
            try:
                uo = getattr(s, 'use_unipc_override', None)
                if uo is None:
                    self.cb_sched.setCurrentText('(inherit)')
                else:
                    self.cb_sched.setCurrentText('UniPC' if bool(uo) else 'default')
                fs = getattr(s, 'flow_shift_override', None)
                self.sp_flow_shift.setValue(float(fs) if fs is not None else 5.0)
                self._on_sched_changed()
            except Exception:
                pass

            tm = getattr(s, "transition_mode", None)
            self.cb_trans_mode.setCurrentText(tm if tm else "(inherit)")
            self.sp_trans.setValue(int(s.transition_frames or 0))
            self.cb_ease.setCurrentText(s.transition_ease or "(inherit)")

            self._update_init_preview()
            try:
                self._refresh_takes_ui()
            except Exception:
                pass
        finally:
            self._loading = False

    def apply(self):
        if self._scene is None:
            return
        s = self._scene

        s.label = self.ed_label.text().strip()
        s.tags = self.ed_tags.text().strip()
        s.seconds = float(self.sp_seconds.value())

        mode_txt = self.cb_mode.currentText().strip()
        s.mode_override = None if mode_txt == "(inherit)" else mode_txt

        mid = self.cb_model.currentText().strip()
        s.model_id_override = None if mid in ("(inherit)", "") else mid

        ip = self.ed_input.text().strip()
        s.input_image_path_override = ip if ip else None

        st = self.cb_style.currentText().strip()
        s.style_pack = None if st in ("(inherit)", "", "(none)") else st
        s.use_character_ref = bool(self.cb_use_char.isChecked())

        bk = self.cb_backend.currentText().strip()
        s.backend_override = None if bk in ("(inherit)", "") else bk

        # LingBot camera controls (stored even if another backend is used)
        try:
            s.camera_path_preset = (self.cb_cam_preset.currentText() or "static").strip()
            s.camera_fov_deg = float(self.sp_cam_fov.value())
            s.camera_path_amount = float(self.sp_cam_amount.value())
            s.camera_path_strength = float(self.sp_cam_strength.value())
        except Exception:
            pass

        s.prompt = self.txt_prompt.toPlainText().strip()
        neg = self.txt_neg.toPlainText().strip()
        s.negative_prompt = neg if neg else None

        # Init image settings (per clip overrides)
        try:
            pv = str(self.cb_init_preset.itemData(self.cb_init_preset.currentIndex()) or '').strip()
            s.init_preset_override = pv if pv else None
        except Exception:
            pass
        try:
            pol = str(self.cb_init_policy.itemData(self.cb_init_policy.currentIndex()) or '').strip()
            s.init_policy_override = pol if pol else None
        except Exception:
            pass

        # Phase C fields
        prev_chars = list(getattr(s, "characters", []) or [])
        prev_loc = (getattr(s, "location", "") or "").strip()
        prev_hc = bool(getattr(s, "hard_cut", False))
        prev_prompt = (getattr(s, "prompt", "") or "")
        prev_neg = (getattr(s, "negative_prompt", None) or "")

        try:
            s.characters = [it.text() for it in self.list_chars.selectedItems()]
        except Exception:
            pass
        try:
            loc = self.cb_location.currentText().strip()
            s.location = "" if loc in ("(none)", "") else loc
        except Exception:
            pass
        try:
            s.hard_cut = bool(self.cb_hard_cut.isChecked())
        except Exception:
            pass
        try:
            s.shot_type = self.ed_shot_type.text().strip()
            s.camera_move = self.ed_camera_move.text().strip()
            s.mood = self.ed_mood.text().strip()
            s.music_intensity = self.ed_music_intensity.text().strip()
        except Exception:
            pass

        # Mark init-frame cache as stale if anchors changed
        try:
            if (list(getattr(s, "characters", []) or []) != prev_chars) or ((getattr(s, "location", "") or "").strip() != prev_loc) or (bool(getattr(s, "hard_cut", False)) != prev_hc) or ((getattr(s, "prompt", "") or "") != prev_prompt) or (((getattr(s, "negative_prompt", None) or "")) != prev_neg):
                s.init_frame_hash = None
        except Exception:
            pass

        self._update_init_preview()
        self._refresh_init_cache_status()

        # Dialogue fields (Audio tab / storyboard)
        prev_dlg = (getattr(s, "dialogue_text", "") or "")
        dlg = self.txt_dialogue.toPlainText().strip()
        s.dialogue_text = dlg if dlg else ""
        s.dialogue_voice = self.ed_dlg_voice.text().strip() or ""
        s.dialogue_language = self.ed_dlg_lang.text().strip() or ""
        if (s.dialogue_text or "") != prev_dlg:
            # Force regeneration if user edited dialogue.
            try:
                s.dialogue_audio_path = None
            except Exception:
                pass

        s.seed = int(self.sp_seed.value()) if self.cb_seed.isChecked() else None
        s.num_inference_steps = int(self.sp_steps.value()) if self.cb_steps.isChecked() else None
        s.guidance_scale = float(self.sp_cfg.value()) if self.cb_cfg.isChecked() else None
        s.guidance_scale_2 = float(self.sp_cfg2.value()) if self.cb_cfg2.isChecked() else None

        # Scheduler overrides
        try:
            stxt = self.cb_sched.currentText().strip()
            if stxt == '(inherit)':
                s.use_unipc_override = None
                s.flow_shift_override = None
            elif stxt.lower().startswith('uni'):
                s.use_unipc_override = True
                s.flow_shift_override = float(self.sp_flow_shift.value())
            else:
                s.use_unipc_override = False
                s.flow_shift_override = None
        except Exception:
            pass

        tmode = self.cb_trans_mode.currentText().strip()
        s.transition_mode = None if tmode == "(inherit)" else tmode

        tf = int(self.sp_trans.value())
        s.transition_frames = tf if tf > 0 else None
        te = self.cb_ease.currentText().strip()
        s.transition_ease = None if te == "(inherit)" else te

        if self._apply_cb:
            self._apply_cb()

    def flush_pending(self):
        """Commit any pending edits (used before rendering)."""
        try:
            if hasattr(self, "_auto_timer") and self._auto_timer.isActive():
                self._auto_timer.stop()
        except Exception:
            pass
        self.apply()


class StudioPremiereTab(QtWidgets.QWidget):
    """Premiere-like workspace: Project/Monitor/Inspector + Timeline clips."""
    def __init__(self, main_window, parent=None, image_gen_cb=None, regen_init_cb=None, init_cache_status_cb=None, use_cache_cb=None):
        super().__init__(parent)
        self._image_gen_cb = image_gen_cb
        self._regen_init_cb = regen_init_cb
        self._init_cache_status_cb = init_cache_status_cb
        self._use_cache_cb = use_cache_cb
        self.win = main_window

        self.scene = QtWidgets.QGraphicsScene(self)
        self.timeline = TimelineView(self, image_gen_cb=self._image_gen_cb, regen_init_cb=self._regen_init_cb, init_cache_status_cb=self._init_cache_status_cb, use_cache_cb=self._use_cache_cb)
        self.timeline.setScene(self.scene)

        self.pps = 140.0  # pixels per second
        self.clip_h = 64.0
        self.track_y = 10.0
        # Audio lanes (A1/A2/A3) below V1
        self.audio_lane_h = 18.0
        self.audio_lane_gap = 6.0
        self.audio_lane_pad = 10.0
        self._clip_items: List[ClipItem] = []

        # In/Out range markers (quick render range)
        self._range_in_idx: Optional[int] = None
        self._range_out_idx: Optional[int] = None

        # Monitor
        self.video = QVideoWidget()
        self.video.setMinimumHeight(280)
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.audio.setVolume(0.0)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video)

        self.info = QtWidgets.QLabel("No preview loaded.")
        self.info.setWordWrap(True)

        # Aide / workflow (pour rendre l'UI lisible immédiatement)
        self.help_box = QtWidgets.QGroupBox("Workflow (1→4)")
        self.help_box.setCheckable(True)
        self.help_box.setChecked(False)
        hb = QtWidgets.QVBoxLayout(self.help_box)
        hb.setContentsMargins(10, 8, 10, 8)
        hb.addWidget(QtWidgets.QLabel("1) Importe une image/vidéo dans <b>Project</b> (à gauche)."))
        hb.addWidget(QtWidgets.QLabel("2) Ajoute un clip avec <b>+ Clip</b>, puis sélectionne-le sur la timeline."))
        hb.addWidget(QtWidgets.QLabel("3) À droite, dans <b>Inspector</b> : écris le prompt, durée, et options (seed/steps/guidance)."))
        hb.addWidget(QtWidgets.QLabel("4) Lance <b>Render Selection</b>, <b>Range</b> (In→Out), ou <b>Render All</b>."))
        hb.addWidget(QtWidgets.QLabel("<i>Tip:</i> Pour I2V : bouton <b>Set as Project Input</b> (global) ou <b>Input override</b> (par clip)."))

        # Project bin
        self.bin_list = QtWidgets.QListWidget()
        self.btn_import = QtWidgets.QPushButton("Importer (image/vidéo)")
        self.btn_extract = QtWidgets.QPushButton("Extraire une frame…")
        self.btn_set_input = QtWidgets.QPushButton("Définir comme Input projet")
        self.btn_import.clicked.connect(self._import_media)
        self.btn_extract.clicked.connect(self._extract_frame)
        self.btn_set_input.clicked.connect(self._set_as_input)

        bin_box = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(bin_box); bl.setContentsMargins(0,0,0,0)
        bl.addWidget(QtWidgets.QLabel("<b>Project</b>"))
        bl.addWidget(self.bin_list, 1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.btn_import)
        row.addWidget(self.btn_extract)
        bl.addLayout(row)
        bl.addWidget(self.btn_set_input)

        # Inspector
        self.ins = ClipInspector(image_gen_cb=self._image_gen_cb, regen_init_cb=self._regen_init_cb, init_cache_status_cb=self._init_cache_status_cb, use_cache_cb=self._use_cache_cb)
        self.ins.bind(self._after_inspector_apply, play_cb=self.load_preview)

        self._refresh_phasec_lists()

        # Timeline controls
        ctrl = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("+ Clip")
        self.btn_dup = QtWidgets.QPushButton("⎘ Duplicate")
        self.btn_del = QtWidgets.QPushButton("− Delete")
        self.btn_split = QtWidgets.QPushButton("✂ Split…")
        self.btn_proxy_sel = QtWidgets.QPushButton("⚡ Proxy")
        self.btn_final_sel = QtWidgets.QPushButton("🎬 Final")
        self.btn_render_sel = QtWidgets.QPushButton("▶ Render (legacy)")
        self.btn_render_all = QtWidgets.QPushButton("▶ Render tout")
        self.btn_doctor = QtWidgets.QPushButton("🩺 Doctor")
        self.btn_audit = QtWidgets.QPushButton("🧾 Audit")
        # Range controls (In/Out)
        self.lbl_range = QtWidgets.QLabel("Range: —")
        self.btn_set_in = QtWidgets.QPushButton("⟲ In")
        self.btn_set_out = QtWidgets.QPushButton("⟳ Out")
        self.btn_clear_range = QtWidgets.QPushButton("×")
        self.btn_clear_range.setFixedWidth(34)
        self.btn_proxy_range = QtWidgets.QPushButton("⚡ Range")
        self.btn_final_range = QtWidgets.QPushButton("🎬 Range")
        self.btn_render_range = QtWidgets.QPushButton("▶ Range")
        self.btn_set_in.setToolTip("Définir le point IN depuis le clip sélectionné")
        self.btn_set_out.setToolTip("Définir le point OUT depuis le clip sélectionné")
        self.btn_clear_range.setToolTip("Effacer le Range")
        self.btn_proxy_range.setToolTip("Rendre des proxies pour le Range (In→Out) ou la sélection")
        self.btn_final_range.setToolTip("Rendre les finals pour le Range (In→Out) ou la sélection")
        self.btn_render_range.setToolTip("Rendu legacy (stitch) pour le Range (In→Out) ou la sélection")
        self.zoom = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.zoom.setRange(60, 260)
        self.zoom.setValue(int(self.pps))
        self.zoom.setFixedWidth(220)

        ctrl.addWidget(self.btn_add)
        ctrl.addWidget(self.btn_dup)
        ctrl.addWidget(self.btn_del)
        ctrl.addWidget(self.btn_split)
        ctrl.addWidget(self.btn_set_in)
        ctrl.addWidget(self.btn_set_out)
        ctrl.addWidget(self.btn_clear_range)
        ctrl.addWidget(self.btn_proxy_range)
        ctrl.addWidget(self.btn_final_range)
        ctrl.addWidget(self.btn_render_range)
        ctrl.addWidget(self.lbl_range)
        ctrl.addSpacing(10)
        ctrl.addWidget(self.btn_proxy_sel)
        ctrl.addWidget(self.btn_final_sel)
        ctrl.addWidget(self.btn_render_sel)
        ctrl.addWidget(self.btn_render_all)
        ctrl.addStretch(1)
        ctrl.addWidget(self.btn_doctor)
        ctrl.addWidget(self.btn_audit)
        self.chk_audio_lanes = QtWidgets.QCheckBox("Audio lanes")
        self.chk_audio_lanes.setToolTip("Afficher les pistes audio séparées A1/A2/A3 (Dialogue/VFX/Music).")
        self.chk_audio_lanes.setChecked(True)
        ctrl.addWidget(self.chk_audio_lanes)
        ctrl.addWidget(QtWidgets.QLabel("Zoom"))
        ctrl.addWidget(self.zoom)

        self.btn_add.clicked.connect(self.add_clip)
        self.btn_dup.clicked.connect(self.duplicate_selected)
        self.btn_del.clicked.connect(self.delete_selected)
        self.btn_split.clicked.connect(self.split_selected)
        self.btn_set_in.clicked.connect(self.set_range_in)
        self.btn_set_out.clicked.connect(self.set_range_out)
        self.btn_clear_range.clicked.connect(self.clear_range)
        self.btn_proxy_range.clicked.connect(self.render_proxy_range)
        self.btn_final_range.clicked.connect(self.render_final_range)
        self.btn_render_range.clicked.connect(self.render_range)
        self.btn_proxy_sel.clicked.connect(self.render_proxy_selected)
        self.btn_final_sel.clicked.connect(self.render_final_selected)
        self.btn_render_sel.clicked.connect(self.render_selected)
        self.btn_render_all.clicked.connect(lambda: self.win.on_start())
        self.btn_doctor.clicked.connect(self.show_doctor)
        self.btn_audit.clicked.connect(self.show_audit)
        self.zoom.valueChanged.connect(self._on_zoom)
        try:
            self.chk_audio_lanes.toggled.connect(lambda _=None: self.reload_from_cfg(keep_selection=True))
        except Exception:
            pass

        # Build top layout like Premiere
        top_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        top_split.addWidget(bin_box)

        monitor_box = QtWidgets.QWidget()
        ml = QtWidgets.QVBoxLayout(monitor_box); ml.setContentsMargins(0,0,0,0)
        mon_row = QtWidgets.QHBoxLayout()
        mon_row.addWidget(QtWidgets.QLabel("<b>Program Monitor</b>"))
        mon_row.addStretch(1)
        self.cb_prefer_proxy = QtWidgets.QCheckBox("Prefer proxy")
        try:
            self.cb_prefer_proxy.setChecked(bool(getattr(self.win.cfg.studio, "prefer_proxy_playback", True)))
        except Exception:
            self.cb_prefer_proxy.setChecked(True)
        self.cb_prefer_proxy.toggled.connect(self._on_prefer_proxy)
        mon_row.addWidget(self.cb_prefer_proxy)
        ml.addLayout(mon_row)
        ml.addWidget(self.video, 1)
        tr = QtWidgets.QHBoxLayout()
        btn_play = QtWidgets.QPushButton("Play"); btn_pause = QtWidgets.QPushButton("Pause"); btn_stop = QtWidgets.QPushButton("Stop")
        btn_play.clicked.connect(self.player.play)
        btn_pause.clicked.connect(self.player.pause)
        btn_stop.clicked.connect(self.player.stop)
        tr.addWidget(btn_play); tr.addWidget(btn_pause); tr.addWidget(btn_stop); tr.addStretch(1)
        ml.addLayout(tr)
        ml.addWidget(self.info)
        top_split.addWidget(monitor_box)
        top_split.addWidget(self.ins)
        top_split.setStretchFactor(0, 1)
        top_split.setStretchFactor(1, 3)
        top_split.setStretchFactor(2, 1)

        bottom = QtWidgets.QWidget()
        blay = QtWidgets.QVBoxLayout(bottom); blay.setContentsMargins(0,0,0,0)
        blay.addWidget(self.help_box)
        blay.addLayout(ctrl)
        blay.addWidget(self.timeline, 1)

        main_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        main_split.addWidget(top_split)
        main_split.addWidget(bottom)
        main_split.setStretchFactor(0, 1)
        main_split.setStretchFactor(1, 2)
        try:
            bottom.setMinimumHeight(280)
            main_split.setSizes([520, 520])
        except Exception:
            pass

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(main_split, 1)

        self.scene.selectionChanged.connect(self._on_selection_changed)

        # populate model list from main window
        try:
            items = [self.win.model_id.itemText(i) for i in range(self.win.model_id.count())]
        except Exception:
            items = []
        self.ins.set_model_items(items)

        self.reload_from_cfg()

    # -------- Media bin --------
    def _import_media(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Import media", ".", "Media (*.png *.jpg *.jpeg *.webp *.mp4 *.mov *.mkv *.webm *.avi)"
        )
        for f in files:
            self.bin_list.addItem(QtWidgets.QListWidgetItem(f))

    def _extract_frame(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choisir vidéo", ".", "Vidéos (*.mp4 *.mov *.mkv *.webm *.avi)")
        if not f:
            return
        t, ok = QtWidgets.QInputDialog.getDouble(self, "Timestamp", "Extraire frame à t (secondes):", 0.0, 0.0, 10_000.0, 2)
        if not ok:
            return
        try:
            out = self.win._extract_video_frame_to_png(f, float(t))
            self.bin_list.addItem(QtWidgets.QListWidgetItem(out))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Extract failed", str(e))

    def _set_as_input(self):
        it = self.bin_list.currentItem()
        if not it:
            return
        path = it.text().strip()
        if not path:
            return
        try:
            self.win.cfg.input_image_path = path
            self.win._sync_ui_from_cfg()
        except Exception:
            pass

    # -------- Timeline --------
    def _on_zoom(self, v: int):
        self.pps = float(v)
        self.reload_from_cfg(keep_selection=True)

    def add_clip(self):
        """Add a new clip (SceneSpec) to the end of the timeline."""
        try:
            # Ensure main UI changes are synced (model/mode/fps/etc.)
            self.win._sync_cfg_from_ui()
        except Exception:
            pass

        # Compute next label (follow S01_SHxx when possible)
        import re
        mx = 0
        for sc in self.win.cfg.scenes:
            m = re.search(r"SH(\d+)", sc.label or "")
            if m:
                mx = max(mx, int(m.group(1)))
        n = mx + 1 if mx else (len(self.win.cfg.scenes) + 1)
        label = f"S01_SH{n:02d}"

        from .config import SceneSpec
        sc = SceneSpec()
        sc.label = label
        sc.prompt = ""
        sc.negative_prompt = None

        self.win.cfg.scenes.append(sc)

        # Safe to rebuild here (user isn't typing)
        self.reload_from_cfg(keep_selection=False)

        # Select last clip
        try:
            if self._clip_items:
                self.scene.clearSelection()
                self._clip_items[-1].setSelected(True)
        except Exception:
            pass

        try:
            self.win._reload_scene_table()
            try:
                # Audio tab mirrors per-shot dialogue fields; refresh it when a clip changes.
                if hasattr(self.win, '_refresh_audio_scene_list'):
                    self.win._refresh_audio_scene_list(keep_selection=True)
            except Exception:
                pass
        except Exception:
            pass


    def reload_from_cfg(self, keep_selection: bool=False):
        """Rebuild the timeline view from cfg.scenes (used after add/delete/reorder)."""
        sel_scenes = set()
        if keep_selection:
            for it in self.scene.selectedItems():
                if isinstance(it, ClipItem):
                    sel_scenes.add(id(it.scene))

        self.scene.clear()
        self._clip_items.clear()

        x = 0.0
        lanes_enabled = True
        try:
            lanes_enabled = bool(getattr(self, "chk_audio_lanes", None) and self.chk_audio_lanes.isChecked())
        except Exception:
            lanes_enabled = True

        video_y = float(self.track_y)
        a1_y = float(self.track_y) + float(self.clip_h) + float(self.audio_lane_pad)
        a2_y = a1_y + float(self.audio_lane_h) + float(self.audio_lane_gap)
        a3_y = a2_y + float(self.audio_lane_h) + float(self.audio_lane_gap)

        for idx, sc in enumerate(self.win.cfg.scenes):
            # Precompute drift risk for badges/tooltips
            try:
                rr = compute_drift_risk(cfg=self.win.cfg, scene=sc, scene_index=int(idx))
                setattr(sc, '_risk_icon', rr.icon)
                setattr(sc, '_risk_tooltip', f"{rr.icon} {rr.title} (score {rr.score}/100)\n{rr.details}")
            except Exception:
                setattr(sc, '_risk_icon', '')
                setattr(sc, '_risk_tooltip', '')
            ci = ClipItem(idx, sc, self.pps, video_y, self.clip_h, self._on_clips_released)
            try:
                ci.set_audio_lanes(lanes_enabled, lane_h=self.audio_lane_h, lane_gap=self.audio_lane_gap, y0=(a1_y - video_y))
            except Exception:
                pass
            # Provide global music track path for per-clip music waveforms (optional)
            try:
                adir = os.path.join(self.win.cfg.output_dir, self.win.cfg.audio.audio_dir_name)
                ci.music_track_path = os.path.join(adir, self.win.cfg.audio.music_track_filename)
            except Exception:
                ci.music_track_path = None
            ci.setPos(x, self.track_y)
            try:
                ci._update_waveforms()
            except Exception:
                pass
            self.scene.addItem(ci)
            self._clip_items.append(ci)
            if keep_selection and id(sc) in sel_scenes:
                ci.setSelected(True)
            x += ci.rect().width() + 6.0
        # Track backgrounds / labels
        try:
            width = max(800, x)
            r0 = self.scene.addRect(
                0, video_y, width, float(self.clip_h),
                QtGui.QPen(QtGui.QColor(70, 80, 100, 80), 1),
                QtGui.QBrush(QtGui.QColor(25, 30, 40, 120)),
            )
            r0.setZValue(-10)
            t0 = QtWidgets.QGraphicsSimpleTextItem("V1  Video")
            t0.setBrush(QtGui.QBrush(QtGui.QColor(170, 185, 205)))
            t0.setPos(6, video_y + 4)
            t0.setZValue(-9)
            self.scene.addItem(t0)
            if lanes_enabled:
                for (lab, yy) in (("A1  Dialogue", a1_y), ("A2  VFX", a2_y), ("A3  Music", a3_y)):
                    rr = self.scene.addRect(
                        0, yy, width, float(self.audio_lane_h),
                        QtGui.QPen(QtGui.QColor(70, 80, 100, 70), 1),
                        QtGui.QBrush(QtGui.QColor(18, 22, 30, 110)),
                    )
                    rr.setZValue(-10)
                    tt = QtWidgets.QGraphicsSimpleTextItem(lab)
                    tt.setBrush(QtGui.QBrush(QtGui.QColor(155, 170, 190)))
                    tt.setPos(6, yy + 1)
                    tt.setZValue(-9)
                    self.scene.addItem(tt)
        except Exception:
            pass

        line = self.scene.addLine(
            0,
            self.track_y + self.clip_h + 6,
            max(800, x),
            self.track_y + self.clip_h + 6,
            QtGui.QPen(QtGui.QColor(80, 90, 110), 1),
        )
        line.setZValue(-1)

        try:
            htot = float(self.track_y) + float(self.clip_h) + 40.0
            if lanes_enabled:
                htot = float(a3_y) + float(self.audio_lane_h) + 40.0
        except Exception:
            htot = float(self.track_y) + float(self.clip_h) + 40.0
        self.scene.setSceneRect(0, 0, max(1200, x + 200), htot)

        self._refresh_phasec_lists()

    def refresh_all(self):
        """Refresh timeline + inspector (keep selection) after global setting edits."""
        try:
            self.reload_from_cfg(keep_selection=True)
        except Exception:
            try:
                self.reload_from_cfg(keep_selection=False)
            except Exception:
                pass
        try:
            # If an inspector scene is selected, refresh cache label & preview
            if hasattr(self, 'ins') and self.ins is not None:
                try:
                    self.ins._update_init_preview()
                except Exception:
                    pass
                try:
                    self.ins._refresh_init_cache_status()
                except Exception:
                    pass
        except Exception:
            pass

    def show_doctor(self):
        """Quick stability report + optional one-click global fixes."""
        cfg = self.win.cfg

        def _global_issues() -> List[str]:
            g = []
            try:
                if not bool(getattr(cfg, 'reset_continuity_between_clips', False)):
                    g.append("• Reset continuity désactivé → bleed entre clips")
                if not bool(getattr(cfg, 'strict_clip_coherence', False)):
                    g.append("• Strict clip coherence désactivé → drift intra-clip")
                if not bool(getattr(cfg, 'force_first_frame_to_conditioning', False)):
                    g.append("• First-frame lock désactivé → pops au début de segment")
                cs = float(getattr(cfg, 'chunk_seconds', 0.0) or 0.0)
                if cs >= 6.0:
                    g.append(f"• Chunk_seconds élevé ({cs:.1f}s) → drift plus probable")
            except Exception:
                pass
            return g

        issues = []
        issues.extend(_global_issues())
        per = []
        for i, sc in enumerate(getattr(cfg, 'scenes', []) or []):
            rr = compute_drift_risk(cfg=cfg, scene=sc, scene_index=int(i))
            if rr.score > 20:
                lbl = (getattr(sc, 'label', '') or '').strip() or f"Clip {i+1}"
                det = str(getattr(rr, 'details', '') or '')
                det = det.replace("\n", "\n   ")
                per.append(f"{i+1:02d}. {rr.icon} {lbl} — {rr.title} ({rr.score}/100)\n   {det}")

        if not issues and not per:
            txt = "✅ Tout est vert. Les paramètres actuels sont plutôt safe pour la cohérence."
        else:
            txt = "\n".join([
                "🩺 Stability Doctor — rapport",
                "",
                "Global:",
                *(issues or ["• (OK)"]),
                "",
                "Par clip:",
                *(per or ["• (OK)"]),
            ])

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('Stability Doctor')
        dlg.resize(760, 520)
        vl = QtWidgets.QVBoxLayout(dlg)
        ed = QtWidgets.QPlainTextEdit()
        ed.setReadOnly(True)
        ed.setPlainText(txt)
        vl.addWidget(ed, 1)
        row = QtWidgets.QHBoxLayout()
        btn_fix = QtWidgets.QPushButton('Apply safe stability defaults')
        btn_close = QtWidgets.QPushButton('Close')
        row.addWidget(btn_fix)
        row.addStretch(1)
        row.addWidget(btn_close)
        vl.addLayout(row)

        def _apply_fix():
            try:
                cfg.reset_continuity_between_clips = True
                cfg.strict_clip_coherence = True
                cfg.force_first_frame_to_conditioning = True
                # Ciné stable: no overlap + hard cut stitches (avoids flow/blend artefacts)
                try:
                    cfg.no_overlap_strict = True
                    cfg.overlap_frames = 0
                    cfg.blend_mode = 'cut'
                    cfg.scene_transition_mode = 'cut'
                    cfg.scene_transition_frames = 0
                except Exception:
                    pass
                if float(getattr(cfg, 'chunk_seconds', 0.0) or 0.0) > 4.0:
                    cfg.chunk_seconds = 4.0
            except Exception:
                pass
            try:
                # refresh GUI
                if hasattr(self.win, '_sync_ui_from_cfg'):
                    self.win._sync_ui_from_cfg()
            except Exception:
                pass
            try:
                self.refresh_all()
            except Exception:
                pass
            dlg.accept()

        btn_fix.clicked.connect(_apply_fix)
        btn_close.clicked.connect(dlg.reject)
        dlg.exec()


    def show_audit(self):
        """Open an audit report dialog (env + project snapshot)."""
        try:
            from .audit_tools import run_audit
            txt = run_audit(self.win.cfg)
        except Exception as e:
            txt = f"Audit failed: {e}"

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Audit report")
        dlg.resize(820, 640)
        lay = QtWidgets.QVBoxLayout(dlg)
        te = QtWidgets.QPlainTextEdit()
        te.setReadOnly(True)
        te.setPlainText(txt)
        lay.addWidget(te, 1)

        row = QtWidgets.QHBoxLayout()
        btn_copy = QtWidgets.QPushButton("Copy")
        btn_close = QtWidgets.QPushButton("Close")
        row.addWidget(btn_copy)
        row.addStretch(1)
        row.addWidget(btn_close)
        lay.addLayout(row)

        def _copy():
            try:
                QtWidgets.QApplication.clipboard().setText(te.toPlainText())
            except Exception:
                pass

        btn_copy.clicked.connect(_copy)
        btn_close.clicked.connect(dlg.accept)
        dlg.exec()

    def set_clip_render_state(self, idx: int, state: str, *, cache_hit: bool = False):
        try:
            if 0 <= int(idx) < len(self._clip_items):
                self._clip_items[int(idx)].set_render_state(state, cache_hit=cache_hit)
        except Exception:
            pass

    def set_clip_render_progress(self, idx: int, p: float, msg: str = ''):
        try:
            if 0 <= int(idx) < len(self._clip_items):
                self._clip_items[int(idx)].set_render_progress(float(p or 0.0), msg=msg)
        except Exception:
            pass

    def set_clip_asset(self, idx: int, kind: str, path: str):
        """Update clip UI when the engine produces an asset (init-frame / live preview)."""
        try:
            i = int(idx)
            if i < 0 or i >= len(self._clip_items):
                return
            sc = self.win.cfg.scenes[i]
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
            elif k in ('final_video', 'final'):
                try:
                    sc.final_video_path = p
                except Exception:
                    setattr(sc, 'final_video_path', p)
            # Refresh clip item + inspector thumbnail
            try:
                self._clip_items[i]._update_text()
            except Exception:
                try:
                    self._clip_items[i].update()
                except Exception:
                    pass
            try:
                if getattr(self, 'ins', None) is not None and getattr(self.ins, '_scene', None) is sc:
                    self.ins._update_init_preview()
            except Exception:
                pass
        except Exception:
            pass

    def _refresh_phasec_lists(self):
        """Populate inspector dropdown/list from cfg Character/Location libraries."""
        try:
            chars = [c.name for c in (getattr(self.win.cfg, "characters", []) or []) if getattr(c, "name", None)]
        except Exception:
            chars = []
        try:
            locs = [l.name for l in (getattr(self.win.cfg, "locations", []) or []) if getattr(l, "name", None)]
        except Exception:
            locs = []
        try:
            self.ins.set_character_items(chars)
            self.ins.set_location_items(locs)
        except Exception:
            pass

    def _selected_indices(self) -> List[int]:
        items = [it for it in self.scene.selectedItems() if isinstance(it, ClipItem)]
        if not items:
            return []
        return sorted({it.idx for it in items})

    def _on_selection_changed(self):
        idxs = self._selected_indices()
        if not idxs:
            self.ins.set_scene(None)
            return
        sc = self.win.cfg.scenes[idxs[0]]
        self.ins.set_scene(sc)

        # Keep Audio tab selection in sync with timeline selection.
        try:
            if hasattr(self.win, "_audio_select_scene_index"):
                self.win._audio_select_scene_index(int(idxs[0]))
        except Exception:
            pass

        # Auto-load proxy/final if available
        try:
            prefer_proxy = bool(getattr(self.win.cfg.studio, "prefer_proxy_playback", True))
            cand = None
            if prefer_proxy and getattr(sc, "proxy_video_path", None):
                cand = sc.proxy_video_path
            elif getattr(sc, "final_video_path", None):
                cand = sc.final_video_path
            if cand:
                self.load_preview(cand)
        except Exception:
            pass

    def _layout_clips(self):
        """Update clip positions/sizes without rebuilding (keeps focus while typing)."""
        by_scene = {id(it.scene): it for it in self._clip_items}
        x = 0.0
        for idx, sc in enumerate(self.win.cfg.scenes):
            it = by_scene.get(id(sc))
            if it is None:
                self.reload_from_cfg(keep_selection=True)
                return
            it.idx = idx
            it.setPos(x, self.track_y)
            it.refresh_geometry(self.pps)
            x += it.rect().width() + 6.0
        try:
            htot = float(self.track_y) + float(self.clip_h) + 40.0
            if lanes_enabled:
                htot = float(a3_y) + float(self.audio_lane_h) + 40.0
        except Exception:
            htot = float(self.track_y) + float(self.clip_h) + 40.0
        self.scene.setSceneRect(0, 0, max(1200, x + 200), htot)

    def _after_inspector_apply(self):
        # Do not rebuild during edits: it resets selection and clears fields.
        self._layout_clips()
        try:
            self.win._reload_scene_table()
            try:
                # Audio tab mirrors per-shot dialogue fields; refresh it when a clip changes.
                if hasattr(self.win, '_refresh_audio_scene_list'):
                    self.win._refresh_audio_scene_list(keep_selection=True)
            except Exception:
                pass
        except Exception:
            pass

    def _on_clips_released(self):
        items_sorted = sorted(self._clip_items, key=lambda it: float(it.pos().x()))
        new_scenes = [it.scene for it in items_sorted]
        if new_scenes != self.win.cfg.scenes:
            self.win.cfg.scenes = new_scenes
            for i, it in enumerate(items_sorted):
                it.idx = i
            self.reload_from_cfg(keep_selection=True)
            try:
                self.win._reload_scene_table()
                try:
                    if hasattr(self.win, '_refresh_audio_scene_list'):
                        self.win._refresh_audio_scene_list(keep_selection=True)
                except Exception:
                    pass
            except Exception:
                pass

    def duplicate_selected(self):
        idxs = self._selected_indices()
        if not idxs:
            return
        import copy
        r = idxs[-1]
        self.win.cfg.scenes.insert(r+1, copy.deepcopy(self.win.cfg.scenes[r]))
        self.reload_from_cfg()
        try:
            self.win._reload_scene_table()
            try:
                # Audio tab mirrors per-shot dialogue fields; refresh it when a clip changes.
                if hasattr(self.win, '_refresh_audio_scene_list'):
                    self.win._refresh_audio_scene_list(keep_selection=True)
            except Exception:
                pass
        except Exception:
            pass

    def delete_selected(self):
        idxs = self._selected_indices()
        if not idxs:
            return
        for i in sorted(idxs, reverse=True):
            if 0 <= i < len(self.win.cfg.scenes):
                self.win.cfg.scenes.pop(i)
        self.reload_from_cfg()
        try:
            self.win._reload_scene_table()
            try:
                # Audio tab mirrors per-shot dialogue fields; refresh it when a clip changes.
                if hasattr(self.win, '_refresh_audio_scene_list'):
                    self.win._refresh_audio_scene_list(keep_selection=True)
            except Exception:
                pass
        except Exception:
            pass

    def split_selected(self):
        idxs = self._selected_indices()
        if not idxs:
            return
        i = idxs[0]
        sc = self.win.cfg.scenes[i]
        if sc.seconds <= 0.2:
            return
        t, ok = QtWidgets.QInputDialog.getDouble(self, "Split", "Split at (seconds from start of clip):", sc.seconds/2.0, 0.05, sc.seconds-0.05, 2)
        if not ok:
            return
        import copy
        a = copy.deepcopy(sc); b = copy.deepcopy(sc)
        a.seconds = float(t)
        b.seconds = float(sc.seconds - t)
        b.label = (b.label or f"Clip{i+1}") + "_B"
        self.win.cfg.scenes[i] = a
        self.win.cfg.scenes.insert(i+1, b)
        self.reload_from_cfg()
        try:
            self.win._reload_scene_table()
            try:
                # Audio tab mirrors per-shot dialogue fields; refresh it when a clip changes.
                if hasattr(self.win, '_refresh_audio_scene_list'):
                    self.win._refresh_audio_scene_list(keep_selection=True)
            except Exception:
                pass
        except Exception:
            pass

    def flush_pending(self):
        """Commit pending inspector edits before rendering."""
        try:
            if hasattr(self, "ins") and self.ins is not None:
                self.ins.flush_pending()
        except Exception:
            pass


    def selected_indices(self) -> List[int]:
        """Public accessor used by the main window to query current selection."""
        return self._selected_indices()

    def range_indices(self) -> List[int]:
        """Public accessor: list of clip indices in the In/Out range."""
        return self._range_indices()

    def _range_indices(self) -> List[int]:
        a = self._range_in_idx
        b = self._range_out_idx
        if a is None or b is None:
            return []
        try:
            a = int(a); b = int(b)
        except Exception:
            return []
        lo = max(0, min(a, b))
        hi = max(a, b)
        hi = min(hi, max(0, len(self.win.cfg.scenes) - 1))
        return list(range(lo, hi + 1))

    def _update_range_label(self):
        try:
            idxs = self._range_indices()
            if not idxs:
                self.lbl_range.setText("Range: —")
                return
            self.lbl_range.setText(f"Range: {min(idxs)+1}→{max(idxs)+1} ({len(idxs)} clip(s))")
        except Exception:
            try:
                self.lbl_range.setText("Range: —")
            except Exception:
                pass

    def set_range_in(self):
        idxs = self._selected_indices()
        if not idxs:
            QtWidgets.QMessageBox.information(self, "Range IN", "Sélectionne un clip pour définir IN.")
            return
        self._range_in_idx = int(idxs[0])
        self._update_range_label()

    def set_range_out(self):
        idxs = self._selected_indices()
        if not idxs:
            QtWidgets.QMessageBox.information(self, "Range OUT", "Sélectionne un clip pour définir OUT.")
            return
        self._range_out_idx = int(idxs[-1])
        self._update_range_label()

    def clear_range(self):
        self._range_in_idx = None
        self._range_out_idx = None
        self._update_range_label()

    def _indices_for_range_or_selection(self) -> List[int]:
        idxs = self._range_indices()
        if idxs:
            return idxs
        return self._selected_indices()

    def render_range(self):
        """Legacy render: stitch only the clips in the In/Out range (or current selection)."""
        self.flush_pending()
        idxs = self._indices_for_range_or_selection()
        if not idxs:
            QtWidgets.QMessageBox.information(self, "Render", "Définis un Range (IN/OUT) ou sélectionne des clips.")
            return
        self.win.render_selected_scenes(idxs)

    def render_proxy_range(self):
        """Render proxies for the In/Out range (or current selection)."""
        self.flush_pending()
        idxs = self._indices_for_range_or_selection()
        if not idxs:
            QtWidgets.QMessageBox.information(self, "Proxy", "Définis un Range (IN/OUT) ou sélectionne des clips.")
            return
        self.win.render_proxy_scenes(idxs)

    def render_final_range(self):
        """Render finals for the In/Out range (or current selection)."""
        self.flush_pending()
        idxs = self._indices_for_range_or_selection()
        if not idxs:
            QtWidgets.QMessageBox.information(self, "Final", "Définis un Range (IN/OUT) ou sélectionne des clips.")
            return
        self.win.render_final_scenes(idxs)


    def render_selected(self):
        self.flush_pending()
        idxs = self._selected_indices()
        if not idxs:
            QtWidgets.QMessageBox.information(self, "Render", "Sélectionne un ou plusieurs clips sur la timeline.")
            return
        self.win.render_selected_scenes(idxs)

    def render_proxy_selected(self):
        """Render low-res proxy(s) for the selected clips."""
        self.flush_pending()
        idxs = self._selected_indices()
        if not idxs:
            QtWidgets.QMessageBox.information(self, "Proxy", "Sélectionne un ou plusieurs clips sur la timeline.")
            return
        self.win.render_proxy_scenes(idxs)

    def render_final_selected(self):
        """Render final conform(s) for the selected clips."""
        self.flush_pending()
        idxs = self._selected_indices()
        if not idxs:
            QtWidgets.QMessageBox.information(self, "Final", "Sélectionne un ou plusieurs clips sur la timeline.")
            return
        self.win.render_final_scenes(idxs)

    def render_all(self):
        """Render all clips in timeline order."""
        self.flush_pending()
        try:
            self.win._sync_cfg_from_ui()
        except Exception:
            pass
        try:
            self.win.on_start()
        except Exception:
            self.win.render_selected_scenes(list(range(len(self.win.cfg.scenes))))

    def load_preview(self, path: str):
        """Load a preview-friendly file into the Program Monitor.

        QtMultimedia may not play ProRes .mov on Linux; if a .mov is provided,
        try to locate an mp4 sibling from the delivery pack or raw render.
        """
        import os
        p = os.path.abspath(path)
        if p.lower().endswith(".mov"):
            base_dir = os.path.dirname(p)
            candidates = [
                os.path.join(base_dir, "delivery_review_h265.mp4"),
                os.path.join(base_dir, "delivery_review_h265_main10.mp4"),
                os.path.join(base_dir, "deliver_1080p_h265_main10.mp4"),
                os.path.join(base_dir, "post", "polished.mp4"),
                os.path.join(base_dir, "post", "interpolated.mp4"),
                os.path.join(base_dir, "raw.mp4"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    p = c
                    break

        url = QtCore.QUrl.fromLocalFile(p)
        self.player.setSource(url)
        self.player.play()
        self.info.setText(p)

    def play_clip(self, idx: int, kind: str = 'auto'):
        """Play a clip in the Program Monitor.

        kind:
          - 'auto': follow Prefer proxy setting, else fall back
          - 'proxy': force proxy
          - 'final': force final
        If no playable file is available, we try to build a preview from the clip cache.
        """
        try:
            idx = int(idx)
        except Exception:
            return
        if idx < 0 or idx >= len(self.win.cfg.scenes):
            return

        # Select clip in the timeline (keeps inspector + audio tab in sync)
        try:
            for it in self._clip_items:
                it.setSelected(int(getattr(it, 'idx', -1)) == idx)
        except Exception:
            pass

        sc = self.win.cfg.scenes[idx]
        want = str(kind or 'auto').strip().lower()
        prefer_proxy = bool(getattr(self.win.cfg.studio, 'prefer_proxy_playback', True))

        cand = None
        if want in ('proxy', 'p'):
            cand = getattr(sc, 'proxy_video_path', None)
        elif want in ('final', 'f'):
            cand = getattr(sc, 'final_video_path', None)
        else:
            if prefer_proxy and getattr(sc, 'proxy_video_path', None):
                cand = sc.proxy_video_path
            elif getattr(sc, 'final_video_path', None):
                cand = sc.final_video_path
            elif getattr(sc, 'proxy_video_path', None):
                cand = sc.proxy_video_path

        if cand:
            self.load_preview(str(cand))
            return

        # No direct file: build from cache (best-effort)
        try:
            if hasattr(self.win, 'build_proxy_preview_for_scene'):
                self.win.build_proxy_preview_for_scene(idx, play_after=True)
                return
        except Exception:
            pass

        QtWidgets.QMessageBox.information(self, "Preview", "Aucun proxy/final disponible pour ce clip.\nRends un Proxy/Final, ou active le cache clip.")

    def build_and_play_preview(self, idx: int):
        try:
            if hasattr(self.win, 'build_proxy_preview_for_scene'):
                self.win.build_proxy_preview_for_scene(int(idx), play_after=True)
        except Exception:
            pass

    def render_proxy_indices(self, idxs: List[int]):
        try:
            self.flush_pending()
            self.win.render_proxy_scenes([int(i) for i in (idxs or [])])
        except Exception:
            pass

    def render_final_indices(self, idxs: List[int]):
        try:
            self.flush_pending()
            self.win.render_final_scenes([int(i) for i in (idxs or [])])
        except Exception:
            pass

    def set_range_in_idx(self, idx: int):
        try:
            self._range_in_idx = int(idx)
            self._update_range_label()
        except Exception:
            pass

    def set_range_out_idx(self, idx: int):
        try:
            self._range_out_idx = int(idx)
            self._update_range_label()
        except Exception:
            pass

    def reveal_clip_file(self, idx: int):
        """Reveal the most relevant clip file (proxy/final) in the file manager."""
        try:
            import os
            idx = int(idx)
            sc = self.win.cfg.scenes[idx]
            p = getattr(sc, 'final_video_path', None) or getattr(sc, 'proxy_video_path', None) or ''
            p = str(p or '').strip()
            if p and os.path.exists(p):
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(os.path.dirname(os.path.abspath(p))))
                return
        except Exception:
            pass
        QtWidgets.QMessageBox.information(self, "Reveal", "Aucun fichier proxy/final trouvé pour ce clip.")

    def _on_prefer_proxy(self, checked: bool):
        try:
            self.win.cfg.studio.prefer_proxy_playback = bool(checked)
        except Exception:
            pass
