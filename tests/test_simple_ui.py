"""Offscreen UI tests; no model downloads or photographs leave the machine."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6", reason="PyQt6 is required for desktop smoke tests")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path_factory.mktemp("settings")))
    application = QApplication.instance() or QApplication([])
    yield application


def test_simple_mode_ignores_stale_classical_engine_settings(app, tmp_path):
    from PyQt6.QtCore import QSettings

    from bodybuilder.config import BackendKind, DeviceKind, SubjectKind
    from bodybuilder.ui.main_window import MainWindow

    settings = QSettings("Niko-Pix-kit", "BodyBuilder")
    settings.setValue("backend", "classical")
    settings.setValue("input_dir", str(tmp_path / "source"))
    settings.setValue("output_dir", str(tmp_path / "output"))
    window = MainWindow()
    window.show()
    app.processEvents()
    assert not window.advanced_panel.isVisible()
    assert not window.log_edit.isVisible()
    config = window._collect_config()
    assert config.backend == BackendKind.SDXL
    assert config.subject_kind == SubjectKind.PERSON
    assert config.device == DeviceKind.AUTO
    assert not config.stitch_overlaps
    assert config.synthetic_variants == 0
    window.advanced_toggle.click()
    assert window.advanced_panel.isVisible()
    window.close()


def test_mask_editor_saves_sidecar_not_original(app, tmp_path):
    from PIL import Image

    from bodybuilder.ui.main_window import MainWindow
    from bodybuilder.ui.mask_editor import MaskEditor

    source = tmp_path / "face.png"
    Image.new("RGB", (100, 80), "blue").save(source)
    original = source.read_bytes()
    window = MainWindow()
    editor = MaskEditor(source, window)
    editor.canvas.mask.paste(255, (10, 10, 30, 30))
    editor._save()
    assert source.read_bytes() == original
    assert Path(str(source) + ".mask.png").exists()
    window.close()
