"""Desktop defaults, mode switches and worker wiring without model downloads."""
from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QApplication

from bodybuilder.config import SubjectKind
from bodybuilder.core.run_options import QualityMode, WorkflowMode, quality_profile
from bodybuilder.ui.subject_window import MainWindow


@pytest.fixture
def window(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    widget = MainWindow()
    widget.input_edit.setText(str(tmp_path / "source"))
    widget.output_edit.setText(str(tmp_path / "output"))
    yield widget
    widget.close()
    app.processEvents()


def test_default_ui_restores_sources_and_requests_five_complete_views(window):
    assert window.workflow_combo.currentData() == WorkflowMode.BOTH
    assert window.margin_spin.isEnabled() and window.aspect_combo.isEnabled()
    assert not window.advanced_panel.isVisible()
    assert window.quality_combo.currentData() == QualityMode.BALANCED
    assert window.variants_spin.value() == 5
    assert window.upscale_check.isChecked()
    config = window._collect_config()
    assert config.synthetic_variants == 5 and config.upscale_2x
    assert config.subject_kind == SubjectKind.AUTO
    assert config.target_long_edge == 1024 and config.inference_steps == 40


def test_source_only_and_synthetic_only_are_not_exclusive_side_effects(window):
    window.workflow_combo.setCurrentIndex(1)
    assert not window.variants_spin.isEnabled()
    assert window._collect_config().synthetic_variants == 0
    assert window.margin_spin.isEnabled()
    window.workflow_combo.setCurrentIndex(2)
    assert window.variants_spin.isEnabled()
    assert window.variants_spin.minimum() == 1
    assert window._collect_config().synthetic_variants == 5
    assert not window.margin_spin.isEnabled() and not window.aspect_combo.isEnabled()
    window.workflow_combo.setCurrentIndex(0)
    window.variants_spin.setValue(10)
    assert window._collect_config().synthetic_variants == 10
    assert "10 complete synthetic" in window.plan_summary.text()
    window._set_busy(True)
    assert not window.workflow_panel.isEnabled()
    window._set_busy(False)
    assert window.workflow_panel.isEnabled() and window.margin_spin.isEnabled()


@pytest.mark.parametrize("mode", list(QualityMode))
def test_quality_selection_is_passed_to_real_worker_configuration(window, mode):
    window.quality_combo.setCurrentIndex(window.quality_combo.findData(mode))
    config = window._collect_config()
    expected = quality_profile(mode)
    assert (config.target_long_edge, config.inference_steps) == (expected.working_long_edge, expected.generation_steps)
    assert config.upscale_2x and config.synthetic_variants == 5
    worker = window._create_worker(config)
    assert worker.workflow == WorkflowMode.BOTH
    assert worker.quality == mode and worker.config is config


def test_resource_override_is_not_reset_by_quality_switch(window):
    window.upscale_check.setChecked(False)
    window.quality_combo.setCurrentIndex(2)
    assert not window._collect_config().upscale_2x
    assert "Output: 1x" in window.plan_summary.text()


def test_worker_uses_batch_workflow_and_preset(window, monkeypatch):
    calls = []
    sentinel = object()

    def factory(config, **kwargs):
        calls.append((config, kwargs))
        return SimpleNamespace(run=lambda: sentinel, run_dir=None)

    monkeypatch.setattr("bodybuilder.ui.subject_window.JointReconstructionPipeline", factory)
    worker = window._create_worker(window._collect_config())
    completed = []
    worker.completed.connect(completed.append)
    worker.run()
    assert completed == [sentinel]
    assert calls[0][1]["workflow"] == WorkflowMode.BOTH
    assert calls[0][1]["quality"] == QualityMode.BALANCED
    assert calls[0][0].synthetic_variants == 5


def test_constructor_failure_finishes_worker_and_reports_error(window, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("Test invalid profile")
    monkeypatch.setattr("bodybuilder.ui.subject_window.JointReconstructionPipeline", fail)
    worker = window._create_worker(window._collect_config())
    errors, finished = [], []
    worker.failed.connect(lambda message, details: errors.append((message, details)))
    worker.finished.connect(lambda: finished.append(True))
    worker.run()
    assert finished == [True]
    assert errors[0][0] == "Test invalid profile"
    assert "Traceback" in errors[0][1]
