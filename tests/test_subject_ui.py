from PyQt6.QtWidgets import QApplication

from bodybuilder.ui.subject_window import MainWindow


def test_default_ui_requests_whole_subject_without_fragment_layout():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert window.workflow_combo.currentData() is True
    assert not window.margin_spin.isEnabled()
    assert not window.aspect_combo.isEnabled()
    assert not window.advanced_panel.isVisible()
    window.workflow_combo.setCurrentIndex(1)
    assert window.margin_spin.isEnabled()
    assert window.aspect_combo.isEnabled()
    window.close()
    app.processEvents()
