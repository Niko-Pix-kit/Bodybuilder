"""Offscreen tests for the actual guided application window."""

from __future__ import annotations

import os

import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")


@pytest.fixture
def gui(tmp_path):
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    return QApplication.instance() or QApplication([])


def test_facial_selection_saved_without_modifying_original(gui, tmp_path):
    from bodybuilder.core.evidence import read_guide
    from bodybuilder.ui.facial_guide import FacialGuideDialog
    from bodybuilder.ui.guided_window import GuidedMainWindow
    path = tmp_path / "source.png"
    Image.new("RGB", (160, 200), (120, 90, 70)).save(path)
    before = path.read_bytes()
    window = GuidedMainWindow()
    window.show()
    gui.processEvents()
    assert window.reference_button.isVisible()
    assert not window.advanced_panel.isVisible()
    window._set_busy(True)
    assert not window.reference_button.isEnabled()
    window._set_busy(False)
    dialog = FacialGuideDialog(path, parent=window)
    dialog.role.setCurrentIndex(1)  # Mouth
    dialog.selector.box = (30, 90, 115, 145)
    dialog._use()
    assert dialog.guide.regions[0].role == "mouth"
    dialog._save()
    assert read_guide(path, (160, 200)).regions[0].box == (30, 90, 115, 145)
    assert path.read_bytes() == before
    window.close()


def test_face_location_selection_uses_original_coordinates(gui, tmp_path):
    from bodybuilder.ui.facial_guide import FacialGuideDialog
    path = tmp_path / "crop.png"
    Image.new("RGB", (160, 200), "blue").save(path)
    dialog = FacialGuideDialog(path)
    dialog.role.setCurrentIndex(3)
    p = dialog.canvas.placement
    dialog.selector.box = (p.x, p.y - 50, p.x + p.source_width, p.y + p.source_height)
    dialog._use()
    assert dialog.guide.face_box[1] < 0
    assert dialog.guide.face_box[0] == 0
    assert dialog.guide.face_box[2:] == (160, 200)
    dialog.reject()
