from __future__ import annotations

from PySide6 import QtGui, QtWidgets


def apply_modern_theme(app: QtWidgets.QApplication) -> None:
    """Apply a lightweight modern dark theme (no external dependencies).

    Goal: product-like UI for demos (Premiere/Resolve vibe) while remaining
    compatible with stock Qt widgets and custom painting.
    """
    try:
        app.setStyle("Fusion")
    except Exception:
        pass

    # --- Palette ---
    pal = QtGui.QPalette()

    # Core surfaces
    base = QtGui.QColor(22, 22, 26)          # main window background
    surface = QtGui.QColor(28, 28, 34)       # panels
    surface2 = QtGui.QColor(36, 36, 44)      # inputs
    text = QtGui.QColor(235, 235, 240)

    # Accent (subtle purple-blue)
    accent = QtGui.QColor(125, 105, 255)
    accent2 = QtGui.QColor(95, 175, 255)

    pal.setColor(QtGui.QPalette.Window, base)
    pal.setColor(QtGui.QPalette.WindowText, text)
    pal.setColor(QtGui.QPalette.Base, surface2)
    pal.setColor(QtGui.QPalette.AlternateBase, surface)
    pal.setColor(QtGui.QPalette.Text, text)
    pal.setColor(QtGui.QPalette.Button, surface)
    pal.setColor(QtGui.QPalette.ButtonText, text)
    pal.setColor(QtGui.QPalette.ToolTipBase, surface)
    pal.setColor(QtGui.QPalette.ToolTipText, text)
    pal.setColor(QtGui.QPalette.Highlight, accent)
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))
    pal.setColor(QtGui.QPalette.Link, accent2)
    pal.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 80, 80))

    # Disabled
    disabled = QtGui.QColor(140, 140, 155)
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, disabled)
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, disabled)
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, disabled)

    try:
        app.setPalette(pal)
    except Exception:
        pass

    # --- Stylesheet (small, robust) ---
    qss = """
    QWidget { font-size: 12px; }
    QMainWindow { background: palette(Window); }

    /* Tabs (left rail) */
    QTabWidget::pane { border: 1px solid rgba(255,255,255,0.06); }
    QTabBar::tab {
        padding: 10px 10px;
        margin: 2px;
        border-radius: 8px;
        background: rgba(255,255,255,0.03);
    }
    QTabBar::tab:selected {
        background: rgba(125,105,255,0.25);
        border: 1px solid rgba(125,105,255,0.45);
    }

    /* Buttons */
    QPushButton {
        padding: 6px 10px;
        border-radius: 8px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
    }
    QPushButton:hover {
        background: rgba(125,105,255,0.18);
        border: 1px solid rgba(125,105,255,0.35);
    }
    QPushButton:pressed {
        background: rgba(125,105,255,0.28);
    }
    QPushButton:disabled {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        color: rgba(235,235,240,0.45);
    }

    /* Inputs */
    QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
        padding: 5px 8px;
        border-radius: 8px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        selection-background-color: rgba(125,105,255,0.45);
    }
    QComboBox::drop-down { border: 0px; width: 18px; }

    /* Group boxes / docks */
    QGroupBox {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        margin-top: 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 6px;
    }
    QDockWidget::title {
        padding: 6px;
        background: rgba(255,255,255,0.04);
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }

    /* Tables */
    QHeaderView::section {
        background: rgba(255,255,255,0.04);
        padding: 6px;
        border: 0px;
        border-right: 1px solid rgba(255,255,255,0.06);
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    QTableView {
        gridline-color: rgba(255,255,255,0.06);
        selection-background-color: rgba(125,105,255,0.25);
        selection-color: palette(Text);
    }

    /* Scrollbars */
    QScrollBar:vertical { width: 12px; margin: 0px; }
    QScrollBar::handle:vertical {
        background: rgba(255,255,255,0.12);
        border-radius: 6px;
        min-height: 24px;
    }
    QScrollBar::handle:vertical:hover { background: rgba(125,105,255,0.35); }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
    """

    try:
        app.setStyleSheet(qss)
    except Exception:
        pass
