from __future__ import annotations

import os
from dataclasses import asdict
from typing import Optional, List

from PySide6 import QtCore, QtWidgets
import traceback

from .config import SceneSpec
from .mistral_storyboard import generate_storyboard, Storyboard


def _show_critical(parent, title: str, text: str, details: str = "") -> None:
    """Show an error dialog in a PySide6-compatible way.

    Some PySide6 builds reject QMessageBox.critical(..., detailedText=...).
    To avoid secondary crashes during error handling, we always construct the
    QMessageBox explicitly.
    """
    m = QtWidgets.QMessageBox(parent)
    m.setIcon(QtWidgets.QMessageBox.Critical)
    m.setWindowTitle(title)
    m.setText(text)
    if details:
        m.setDetailedText(details)
        m.setStandardButtons(QtWidgets.QMessageBox.Ok)
    m.exec()


class DirectorDialog(QtWidgets.QDialog):
    """
    AI Director: uses Mistral API to generate story + storyboard + per-shot prompts,
    then can apply them into the project's timeline (cfg.scenes).
    """

    applied = QtCore.Signal(list)  # emits List[SceneSpec]

    def __init__(self, parent=None, *, current_cfg=None):
        super().__init__(parent)
        self.setWindowTitle("AI Director (Mistral) — Storyboard → Timeline")
        self.setModal(True)
        self.resize(1050, 720)

        self._cfg = current_cfg
        self._storyboard: Optional[Storyboard] = None

        root = QtWidgets.QVBoxLayout(self)

        # --- Top controls ---
        form = QtWidgets.QGridLayout()
        root.addLayout(form)

        self.brief = QtWidgets.QPlainTextEdit()
        self.brief.setPlaceholderText("Brief (produit, style, objectifs, contraintes, texte/logo, etc.)…")
        self.brief.setPlainText(
            "Publicité premium, style cinématique, cohérence visuelle, packshot final."
        )

        self.total_seconds = QtWidgets.QDoubleSpinBox()
        self.total_seconds.setRange(2.0, 120.0)
        self.total_seconds.setDecimals(1)
        self.total_seconds.setSingleStep(1.0)
        self.total_seconds.setValue(12.0)

        self.num_shots = QtWidgets.QSpinBox()
        self.num_shots.setRange(1, 40)
        self.num_shots.setValue(6)

        self.fps = QtWidgets.QSpinBox()
        self.fps.setRange(8, 60)
        self.fps.setValue(getattr(current_cfg, "fps", 24) if current_cfg else 24)

        self.style = QtWidgets.QLineEdit()
        self.style.setText("publicité premium cinématique, photoréaliste, caméra stabilisée")

        self.model = QtWidgets.QLineEdit()
        self.model.setText("mistral-large-latest")

        self.temperature = QtWidgets.QDoubleSpinBox()
        self.temperature.setRange(0.0, 1.5)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(0.6)

        self.key = QtWidgets.QLineEdit()
        self.key.setEchoMode(QtWidgets.QLineEdit.Password)
        self.key.setPlaceholderText("Optionnel: sinon utilise MISTRAL_API_KEY (env)")
        # do not prefill for safety

        r = 0
        form.addWidget(QtWidgets.QLabel("Brief"), r, 0)
        form.addWidget(self.brief, r, 1, 1, 5)
        r += 1

        form.addWidget(QtWidgets.QLabel("Durée (s)"), r, 0)
        form.addWidget(self.total_seconds, r, 1)
        form.addWidget(QtWidgets.QLabel("Plans"), r, 2)
        form.addWidget(self.num_shots, r, 3)
        form.addWidget(QtWidgets.QLabel("FPS"), r, 4)
        form.addWidget(self.fps, r, 5)
        r += 1

        form.addWidget(QtWidgets.QLabel("Style"), r, 0)
        form.addWidget(self.style, r, 1, 1, 5)
        r += 1

        form.addWidget(QtWidgets.QLabel("Mistral model"), r, 0)
        form.addWidget(self.model, r, 1, 1, 2)
        form.addWidget(QtWidgets.QLabel("Temp"), r, 3)
        form.addWidget(self.temperature, r, 4)
        form.addWidget(QtWidgets.QLabel("API key"), r, 5)
        # keep key line edit on new row for space
        r += 1
        form.addWidget(self.key, r, 1, 1, 5)
        r += 1

        # --- Buttons ---
        btn_row = QtWidgets.QHBoxLayout()
        root.addLayout(btn_row)

        self.btn_generate = QtWidgets.QPushButton("Générer storyboard")
        self.btn_apply = QtWidgets.QPushButton("Appliquer à la timeline")
        self.btn_apply.setEnabled(False)
        self.btn_close = QtWidgets.QPushButton("Fermer")

        btn_row.addWidget(self.btn_generate)
        btn_row.addWidget(self.btn_apply)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_close)

        # --- Table ---
        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["#", "Label", "s", "Tags", "Prompt", "Dialogue", "Notes"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setWordWrap(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked | QtWidgets.QAbstractItemView.EditKeyPressed)
        root.addWidget(self.table, 1)

        # --- Status ---
        self.status = QtWidgets.QLabel("")
        root.addWidget(self.status)

        self.btn_generate.clicked.connect(self._on_generate)
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_close.clicked.connect(self.reject)

        # Hint auto-fill from current project
        if current_cfg:
            hint = []
            if getattr(current_cfg, "model_id", ""):
                hint.append(f"Model actuel: {current_cfg.model_id}")
            if getattr(current_cfg, "mode", ""):
                hint.append(f"Mode: {current_cfg.mode}")
            if getattr(current_cfg, "input_image_path", None):
                hint.append("Une image de référence est définie (I2V).")
            if hint:
                self.status.setText(" | ".join(hint))

    def _set_busy(self, busy: bool):
        self.btn_generate.setEnabled(not busy)
        self.btn_apply.setEnabled((not busy) and self._storyboard is not None)
        self.btn_close.setEnabled(not busy)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor if busy else QtCore.Qt.ArrowCursor)


    def _project_hint(self) -> dict:
        cfg = self._cfg
        if not cfg:
            return {}
        chars = []
        try:
            chars = [getattr(c, 'name', '') for c in getattr(cfg, 'characters', []) or [] if getattr(c, 'name', '')]
        except Exception:
            pass
        locs = []
        try:
            locs = [getattr(l, 'name', '') for l in getattr(cfg, 'locations', []) or [] if getattr(l, 'name', '')]
        except Exception:
            pass
        return {
            "current_model_id": getattr(cfg, "model_id", ""),
            "current_mode": getattr(cfg, "mode", ""),
            "has_input_image": bool(getattr(cfg, "input_image_path", None)),
            "base_negative_prompt": getattr(cfg, "negative_prompt", ""),
            "characters_library": chars,
            "locations_library": locs,
            "default_language": getattr(getattr(cfg, 'audio', None), 'default_language', 'fr'),
        }
    def _on_generate(self):
        self._set_busy(True)
        self.status.setText("Appel Mistral…")
        QtWidgets.QApplication.processEvents()

        # Fail fast with a clean UI message if the API key is missing.
        api_key = self.key.text().strip() or os.getenv("MISTRAL_API_KEY", "")
        if not api_key:
            self._storyboard = None
            self.btn_apply.setEnabled(False)
            self.status.setText("Erreur: clé API Mistral manquante")
            _show_critical(
                self,
                "Erreur Mistral — génération storyboard",
                "Clé API Mistral manquante.\n\n"
                "Renseigne MISTRAL_API_KEY dans l'environnement ou colle la clé dans le champ 'API Key'.",
                details="",
            )
            self._set_busy(False)
            return

        try:
            sb = generate_storyboard(
                brief=self.brief.toPlainText().strip(),
                total_seconds=float(self.total_seconds.value()),
                fps=int(self.fps.value()),
                num_shots=int(self.num_shots.value()),
                style=self.style.text().strip(),
                model=self.model.text().strip() or "mistral-large-latest",
                temperature=float(self.temperature.value()),
                api_key=api_key,
                project_hint=self._project_hint(),
            )
            self._storyboard = sb
            self._fill_table(sb)
            self.btn_apply.setEnabled(True)
            self.status.setText(f"OK — {sb.title} | {len(sb.shots)} plans")
        except Exception as e:
            self._storyboard = None
            self.btn_apply.setEnabled(False)
            tb = traceback.format_exc()
            # Show a user-friendly message + provide full details for debugging.
            self.status.setText(f"Erreur: {e}")
            _show_critical(self, "Erreur Mistral — génération storyboard", str(e), details=tb)
            print(tb)
        finally:
            self._set_busy(False)

    def _fill_table(self, sb: Storyboard):
        self.table.setRowCount(0)
        for i, sh in enumerate(sb.shots, start=1):
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(i)))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(sh.label or f"S01_SH{i:02d}"))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{sh.seconds:.1f}"))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(sh.tags))
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(sh.prompt))
            self.table.setItem(row, 5, QtWidgets.QTableWidgetItem(getattr(sh, 'dialogue', '') or ''))
            self.table.setItem(row, 6, QtWidgets.QTableWidgetItem(sh.notes))
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(4, 520)
        self.table.setColumnWidth(5, 280)
        self.table.setColumnWidth(6, 220)

    def _on_apply(self):
        if not self._storyboard:
            return

        scenes: List[SceneSpec] = []
        for row in range(self.table.rowCount()):
            label = self.table.item(row, 1).text().strip()
            secs = float(self.table.item(row, 2).text().strip() or "3.0")
            tags = self.table.item(row, 3).text().strip()
            prompt = self.table.item(row, 4).text().strip()
            dialogue = self.table.item(row, 5).text().strip()
            notes = self.table.item(row, 6).text().strip()

            sh = self._storyboard.shots[row] if row < len(self._storyboard.shots) else None
            neg = (sh.negative_prompt if sh else "").strip()

            sc = SceneSpec(
                label=label,
                tags=tags,
                notes=notes,
                seconds=secs,
                prompt=prompt,
                negative_prompt=neg or None,
                dialogue_text=dialogue,
            )

            # Phase C fields (best-effort)
            if sh:
                try:
                    ch = getattr(sh, 'characters', []) or []
                    if isinstance(ch, str):
                        ch = [t.strip() for t in ch.replace(';', ',').split(',') if t.strip()]
                    sc.characters = [str(x).strip() for x in ch if str(x).strip()]
                except Exception:
                    pass
                try:
                    sc.location = str(getattr(sh, 'location', '') or '').strip()
                except Exception:
                    pass
                try:
                    sc.hard_cut = bool(getattr(sh, 'hard_cut', False))
                except Exception:
                    pass
                try:
                    sc.shot_type = str(getattr(sh, 'shot_type', '') or '').strip()
                except Exception:
                    pass
                try:
                    sc.camera_move = str(getattr(sh, 'camera_move', '') or '').strip()
                except Exception:
                    pass
                try:
                    sc.mood = str(getattr(sh, 'mood', '') or '').strip()
                except Exception:
                    pass
                try:
                    mi = getattr(sh, 'music_intensity', '')
                    sc.music_intensity = '' if mi is None else str(mi)
                except Exception:
                    pass

            # Optional per-shot model/mode hints (kept as notes to avoid breaking user config)
            if sh:
                # Dialogue hints from storyboard (optional)
                try:
                    if getattr(sh, 'dialogue_language', '') and not getattr(sc, 'dialogue_language', ''):
                        sc.dialogue_language = getattr(sh, 'dialogue_language', '')
                    if getattr(sh, 'dialogue_voice', '') and not getattr(sc, 'dialogue_voice', ''):
                        sc.dialogue_voice = getattr(sh, 'dialogue_voice', '')
                    spk = (getattr(sh, 'speaker', '') or '').strip()
                    if spk and getattr(sc, 'dialogue_text', '').strip() and (':' not in sc.dialogue_text[:20]):
                        sc.dialogue_text = f"{spk}: {sc.dialogue_text.strip()}"
                except Exception:
                    pass
                if sh.model_id:
                    sc.model_id_override = sh.model_id
                if sh.mode in ("T2V", "I2V"):
                    sc.mode_override = sh.mode

            scenes.append(sc)

        self.applied.emit(scenes)
        self.accept()
