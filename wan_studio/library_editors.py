from __future__ import annotations

from typing import List, Sequence

from PySide6 import QtWidgets

from .config import CharacterSpec, LocationSpec


class _BaseLibraryDialog(QtWidgets.QDialog):
    TITLE = "Library"
    COLUMNS = []  # list[(header, width)]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.TITLE)
        self.resize(980, 520)

        self.table = QtWidgets.QTableWidget(0, len(self.COLUMNS), self)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
            | QtWidgets.QAbstractItemView.EditTrigger.AnyKeyPressed
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        for i, (h, w) in enumerate(self.COLUMNS):
            self.table.setHorizontalHeaderItem(i, QtWidgets.QTableWidgetItem(h))
            if w:
                self.table.setColumnWidth(i, w)

        btn_add = QtWidgets.QPushButton("Add")
        btn_del = QtWidgets.QPushButton("Remove")
        btn_add.clicked.connect(self._add_row)
        btn_del.clicked.connect(self._remove_selected)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(btn_add)
        top.addWidget(btn_del)
        top.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.table, 1)
        layout.addWidget(btn_box)

    def _add_row(self):
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c in range(self.table.columnCount()):
            it = QtWidgets.QTableWidgetItem("")
            self.table.setItem(r, c, it)

    def _remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _get_cell(self, row: int, col: int) -> str:
        it = self.table.item(row, col)
        return (it.text() if it else "").strip()

    def _set_cell(self, row: int, col: int, txt: str):
        it = self.table.item(row, col)
        if it is None:
            it = QtWidgets.QTableWidgetItem("")
            self.table.setItem(row, col, it)
        it.setText(txt or "")



class CharacterBibleDialog(_BaseLibraryDialog):
    TITLE = "Character Bible"
    COLUMNS = [
        ("Name", 160),
        ("Description", 240),
        ("Prompt", 220),
        ("Negative", 200),
        ("Ref images (comma paths)", 260),
        ("Voice gender (female/male/neutral)", 180),
        ("Voice lang", 90),
        ("Voice model (.onnx or id)", 230),
        ("Voice config (.json)", 180),
    ]

    def __init__(self, parent=None, items: Sequence[CharacterSpec] = ()):  # type: ignore
        super().__init__(parent)
        for c in items:
            self._add_row()
            r = self.table.rowCount() - 1
            self._set_cell(r, 0, getattr(c, 'name', ''))
            self._set_cell(r, 1, getattr(c, 'description', ''))
            self._set_cell(r, 2, getattr(c, 'prompt', ''))
            self._set_cell(r, 3, getattr(c, 'negative_prompt', ''))
            self._set_cell(r, 4, ", ".join(getattr(c, 'ref_images', []) or []))
            self._set_cell(r, 5, getattr(c, 'voice_gender', '') or '')
            self._set_cell(r, 6, getattr(c, 'voice_language', '') or '')
            self._set_cell(r, 7, getattr(c, 'voice_model_path', '') or '')
            self._set_cell(r, 8, getattr(c, 'voice_config_path', '') or '')

    def get_items(self) -> List[CharacterSpec]:
        out: List[CharacterSpec] = []
        for r in range(self.table.rowCount()):
            name = self._get_cell(r, 0)
            if not name:
                continue
            desc = self._get_cell(r, 1)
            pr = self._get_cell(r, 2)
            neg = self._get_cell(r, 3)
            refs = [t.strip() for t in self._get_cell(r, 4).split(',') if t.strip()]
            vg = self._get_cell(r, 5)
            vl = self._get_cell(r, 6)
            vm = self._get_cell(r, 7)
            vc = self._get_cell(r, 8)
            out.append(CharacterSpec(
                name=name,
                description=desc,
                prompt=pr,
                negative_prompt=neg,
                ref_images=refs,
                voice_gender=vg,
                voice_language=vl,
                voice_model_path=vm,
                voice_config_path=vc,
            ))
        return out


class LocationLibraryDialog(_BaseLibraryDialog):
    TITLE = "Locations"
    COLUMNS = [
        ("Name", 220),
        ("Description", 320),
        ("Prompt", 260),
        ("Negative", 220),
        ("Ref images (comma paths)", 0),
    ]

    def __init__(self, parent=None, items: Sequence[LocationSpec] = ()):  # type: ignore
        super().__init__(parent)
        for l in items:
            self._add_row()
            r = self.table.rowCount() - 1
            self._set_cell(r, 0, l.name)
            self._set_cell(r, 1, l.description)
            self._set_cell(r, 2, l.prompt)
            self._set_cell(r, 3, l.negative_prompt)
            self._set_cell(r, 4, ", ".join(l.ref_images or []))

    def get_items(self) -> List[LocationSpec]:
        out: List[LocationSpec] = []
        for r in range(self.table.rowCount()):
            name = self._get_cell(r, 0)
            if not name:
                continue
            desc = self._get_cell(r, 1)
            pr = self._get_cell(r, 2)
            neg = self._get_cell(r, 3)
            refs = [t.strip() for t in self._get_cell(r, 4).split(",") if t.strip()]
            out.append(LocationSpec(name=name, description=desc, prompt=pr, negative_prompt=neg, ref_images=refs))
        return out


# Backward-compat alias
LocationsDialog = LocationLibraryDialog
