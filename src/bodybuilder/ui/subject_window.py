"""Source restoration and complete synthetic views in one simple desktop run."""
from __future__ import annotations

import json
import threading
import traceback
from dataclasses import replace
from pathlib import Path

from PyQt6.QtCore import QThread, QUrl, pyqtSlot
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QWidget,
)

from bodybuilder.config import SubjectKind
from bodybuilder.core.joint_reconstruction import JointReconstructionPipeline
from bodybuilder.core.pipeline import PipelineCallbacks
from bodybuilder.core.run_options import (
    MAX_SYNTHETIC_VIEWS,
    QualityMode,
    WorkflowMode,
    desktop_config,
    quality_profile,
)
from bodybuilder.ui.main_window import MainWindow as PhotoWindow
from bodybuilder.ui.main_window import PipelineWorker


class JointWorker(PipelineWorker):
    def __init__(self, config, cancelled, *, subject_hint, complete_subject=True,
                 workflow=None, quality=None):
        super().__init__(config, cancelled)
        self.complete_subject = complete_subject
        self.subject_hint = subject_hint
        self.workflow = workflow
        self.quality = quality

    @pyqtSlot()
    def run(self):
        pipeline = None
        try:
            pipeline = JointReconstructionPipeline(
                self.config, complete_subject=self.complete_subject,
                workflow=self.workflow, quality=self.quality, subject_hint=self.subject_hint,
                cancel_event=self.cancelled,
                callbacks=PipelineCallbacks(log=self.message.emit, progress=self.progress.emit,
                                           preview=lambda path: self.preview.emit(str(path))))
            self.completed.emit(pipeline.run())
        except Exception as exc:
            # The worker boundary reports constructor failures as well as run errors.
            details = traceback.format_exc()
            if pipeline is not None and pipeline.run_dir:
                details += f"\nRun folder: {pipeline.run_dir}"
            self.failed.emit(str(exc), details)
        finally:
            self.finished.emit()


class MainWindow(PhotoWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BodyBuilder - Source restoration and complete views")
        defaults = desktop_config(Path("."), Path("."))
        self.workflow_panel = QWidget()
        layout = QFormLayout(self.workflow_panel)
        self.workflow_combo = QComboBox()
        self.workflow_combo.addItem("Restore source photos + create complete views", WorkflowMode.BOTH)
        self.workflow_combo.addItem("Restore source photos only", WorkflowMode.SOURCES)
        self.workflow_combo.addItem("Create complete views only", WorkflowMode.SYNTHETIC)
        layout.addRow("Result", self.workflow_combo)

        # Reuse the existing count control, but place it in the main workflow.
        # It now counts TOTAL synthetic outputs rather than extra/bonus views.
        removed = self.advanced_panel.layout().takeRow(self.variants_spin)
        if removed.labelItem is not None and removed.labelItem.widget() is not None:
            removed.labelItem.widget().deleteLater()
        self.variants_spin.setRange(0, MAX_SYNTHETIC_VIEWS)
        self.variants_spin.setValue(defaults.synthetic_variants)
        self.variants_spin.setToolTip("Exact number of new complete views, in addition to any restored source photos.")
        layout.addRow("Synthetic views", self.variants_spin)
        self.quality_combo = QComboBox()
        for text, mode in (("Fast", QualityMode.FAST), ("Balanced (recommended)", QualityMode.BALANCED),
                           ("High quality", QualityMode.HIGH)):
            self.quality_combo.addItem(text, mode)
        self.quality_combo.setCurrentIndex(1)
        layout.addRow("Quality", self.quality_combo)
        self.quality_help = QLabel()
        self.quality_help.setWordWrap(True)
        layout.addRow(self.quality_help)
        self.plan_summary = QLabel()
        self.plan_summary.setWordWrap(True)
        layout.addRow(self.plan_summary)
        self.centralWidget().layout().insertWidget(3, self.workflow_panel)

        self.upscale_check.setChecked(defaults.upscale_2x)
        self.upscale_check.setText("2x output (observed areas remain protected)")
        self.upscale_check.setToolTip(
            "Enabled by default in every quality mode. Larger output does not recover unseen details. "
            "Observed areas are restored with deterministic resizing after enhancement. "
            "Uncheck only to reduce output size and processing time.")
        self.subject_combo.insertItem(0, "Detect common subject", SubjectKind.AUTO)
        self.subject_combo.setCurrentIndex(0)
        self.subject_hint = QLineEdit()
        self.subject_hint.setPlaceholderText("Only if ambiguous: person, bicycle, vase...")
        self.advanced_panel.layout().addRow("Subject hint", self.subject_hint)
        self.evidence_button = QPushButton("Open source evidence report")
        self.evidence_button.setEnabled(False)
        self.evidence_button.clicked.connect(self._open_evidence)
        self.centralWidget().layout().addWidget(self.evidence_button)
        self.workflow_combo.currentIndexChanged.connect(self._workflow_changed)
        self.quality_combo.currentIndexChanged.connect(self._quality_changed)
        self.variants_spin.valueChanged.connect(self._update_plan_summary)
        self.upscale_check.toggled.connect(self._update_plan_summary)
        self._workflow_changed()
        self._quality_changed()
        for label in self.findChildren(QLabel):
            if label.text() == "Reconstruct your cropped photographs":
                label.setText("Restore your photos and reconstruct the complete subject")
            elif label.text().startswith("Put fragments of the same person"):
                label.setText("Use original fragments of the same subject. Restore each source and generate "
                              "complete new views in one run. Missing details remain estimates.")
                label.setWordWrap(True)

    def _reload_sources(self):
        super()._reload_sources()
        # PhotoWindow loads the sources before constructing this extra panel.
        if hasattr(self, "plan_summary"):
            self._update_plan_summary()

    def _workflow_changed(self):
        workflow = WorkflowMode(self.workflow_combo.currentData())
        restore = workflow != WorkflowMode.SYNTHETIC
        self.margin_spin.setEnabled(restore)
        self.aspect_combo.setEnabled(restore)
        self.variants_spin.setMinimum(1 if workflow == WorkflowMode.SYNTHETIC else 0)
        self.variants_spin.setEnabled(workflow != WorkflowMode.SOURCES)
        self.prompt_edit.setPlaceholderText("Optional factual details to preserve, not the source crop or camera angle")
        self._update_plan_summary()

    def _quality_changed(self):
        mode = QualityMode(self.quality_combo.currentData())
        descriptions = {
            QualityMode.FAST: "Faster preview: lower working resolution and fewer detail passes.",
            QualityMode.BALANCED: "Balanced detail and processing time. Recommended for a first full run.",
            QualityMode.HIGH: "More generation steps and up to eight detail passes. Slower; fidelity is not guaranteed.",
        }
        self.quality_help.setText(descriptions[mode])
        self.quality_combo.setToolTip(quality_profile(mode).description())
        self._update_plan_summary()

    def _update_plan_summary(self, *_args):
        workflow = WorkflowMode(self.workflow_combo.currentData())
        sources = self.sources.count() if workflow != WorkflowMode.SYNTHETIC else 0
        synthetic = self.variants_spin.value() if workflow != WorkflowMode.SOURCES else 0
        scale = "2x" if self.upscale_check.isChecked() else "1x"
        self.plan_summary.setText(
            f"Planned: {sources} restored source photo(s) + {synthetic} complete synthetic view(s). "
            f"Output: {scale}. Unreadable source files may be excluded during analysis.")

    def _collect_config(self):
        config = super()._collect_config()
        workflow = WorkflowMode(self.workflow_combo.currentData())
        count = 0 if workflow == WorkflowMode.SOURCES else self.variants_spin.value()
        if workflow == WorkflowMode.SYNTHETIC and count < 1:
            raise ValueError("Select at least one synthetic view")
        config = replace(config, synthetic_variants=count)
        return quality_profile(QualityMode(self.quality_combo.currentData())).apply(config)

    def _create_worker(self, config):
        return JointWorker(config, self.cancel_event,
                           workflow=WorkflowMode(self.workflow_combo.currentData()),
                           quality=QualityMode(self.quality_combo.currentData()),
                           subject_hint=self.subject_hint.text().strip())

    def _start_reconstruction(self):
        if self.worker_thread is not None:
            return
        try:
            config = self._collect_config()
        except ValueError as exc:
            QMessageBox.warning(self, "Check settings", str(exc))
            return
        self._reload_sources()
        self.results.clear()
        self.log_edit.clear()
        self.cancel_event = threading.Event()
        self.last_run_dir = None
        self.open_output_button.setEnabled(False)
        self.evidence_button.setEnabled(False)
        self._set_busy(True)
        self._progress(0, 0, "Comparing original fragments once for all requested outputs...")
        thread = QThread(self)
        worker = self._create_worker(config)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.message.connect(self.log_edit.appendPlainText)
        worker.progress.connect(self._progress)
        worker.preview.connect(self._add_result)
        worker.completed.connect(self._completed)
        worker.failed.connect(self._failed)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        thread.finished.connect(self._thread_finished)
        thread.finished.connect(thread.deleteLater)
        self.worker, self.worker_thread = worker, thread
        thread.start()

    def _set_busy(self, busy):
        super()._set_busy(busy)
        self.workflow_panel.setEnabled(not busy)
        if not busy:
            self._workflow_changed()

    @pyqtSlot(str)
    def _add_result(self, path):
        super()._add_result(path)
        item = self.results.item(self.results.count() - 1)
        prefix = "Restored source" if Path(path).name.startswith("source__") else "Synthetic view"
        item.setText(prefix + ": " + Path(path).name)
        if Path(path).stem.endswith("__needs_review"):
            item.setText(item.text() + " [framing needs review]")
            item.setToolTip("The image was kept, but its complete framing was not confirmed. See the processing log.")
            self.status.setText("Image saved with a framing warning. The remaining views will still be processed.")

    @pyqtSlot(object)
    def _completed(self, result):
        super()._completed(result)
        restored = len(getattr(result, "restored_paths", ()))
        synthetic = len(getattr(result, "synthetic_paths", ()))
        if restored + synthetic:
            self.status.setText(self.status.text() + f" {restored} restored sources; {synthetic} synthetic views.")
        review_paths = getattr(result, "review_paths", ())
        if review_paths:
            prefix = "Cancelled; partial results kept." if result.cancelled else "Completed with warnings."
            text = (f"{prefix} {len(result.output_paths)} image(s) saved; "
                    f"{len(review_paths)} need framing review, not confirmed complete.")
            if restored + synthetic:
                text += f" {restored} restored sources; {synthetic} synthetic views."
            if result.errors:
                text += f" {len(result.errors)} other problem(s) recorded."
            self.status.setText(text)
            for message in getattr(result, "review_messages", ()):
                self.log_edit.appendPlainText("REVIEW: " + message)
            self.details_toggle.setChecked(True)
        evidence = result.run_dir / "subject_evidence.json"
        self.evidence_button.setEnabled(evidence.is_file())
        if evidence.is_file():
            try:
                report = json.loads(evidence.read_text(encoding="utf-8"))
                self.status.setText(self.status.text() + " Subject: " + report.get("common_subject_candidate", "unknown") +
                                    ". Review inferred details in the evidence report.")
            except (OSError, ValueError) as exc:
                self.log_edit.appendPlainText(f"Could not read evidence report: {exc}")

    def _open_evidence(self):
        if self.last_run_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_run_dir / "subject_evidence.json")))
