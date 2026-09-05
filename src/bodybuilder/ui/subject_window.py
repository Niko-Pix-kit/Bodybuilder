"""Small workflow choice on top of the existing desktop UI."""
from __future__ import annotations

import json
import threading
import traceback

from PyQt6.QtCore import QThread, QUrl, pyqtSlot
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QWidget

from bodybuilder.config import SubjectKind
from bodybuilder.core.joint_reconstruction import JointReconstructionPipeline
from bodybuilder.core.pipeline import PipelineCallbacks
from bodybuilder.ui.main_window import MainWindow as PhotoWindow
from bodybuilder.ui.main_window import PipelineWorker


class JointWorker(PipelineWorker):
    def __init__(self, config, cancelled, *, complete_subject, subject_hint):
        super().__init__(config, cancelled)
        self.complete_subject = complete_subject
        self.subject_hint = subject_hint

    @pyqtSlot()
    def run(self):
        pipeline = JointReconstructionPipeline(self.config, complete_subject=self.complete_subject,
            subject_hint=self.subject_hint, cancel_event=self.cancelled,
            callbacks=PipelineCallbacks(log=self.message.emit, progress=self.progress.emit,
                                       preview=lambda path: self.preview.emit(str(path))))
        try:
            self.completed.emit(pipeline.run())
        except Exception as exc:
            details = traceback.format_exc()
            if pipeline.run_dir:
                details += f"\nRun folder: {pipeline.run_dir}"
            self.failed.emit(str(exc), details)
        finally:
            self.finished.emit()


class MainWindow(PhotoWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BodyBuilder - Complete subject reconstruction")
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.addWidget(QLabel("Result"))
        self.workflow_combo = QComboBox()
        self.workflow_combo.addItem("Complete subject - new full view", True)
        self.workflow_combo.addItem("Restore each original photo using all fragments", False)
        layout.addWidget(self.workflow_combo, 1)
        self.centralWidget().layout().insertWidget(2, row)
        self.workflow_panel = row
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
        self._workflow_changed()
        for label in self.findChildren(QLabel):
            if label.text() == "Reconstruct your cropped photographs":
                label.setText("Reconstruct the complete subject from its fragments")
            elif label.text().startswith("Put fragments of the same person"):
                label.setText("Use original fragments of the same subject. New views show the complete subject; "
                              "unseen details remain estimates. Generated pictures are not source evidence.")
                label.setWordWrap(True)

    def _workflow_changed(self):
        restore = not self.workflow_combo.currentData()
        self.margin_spin.setEnabled(restore)
        self.aspect_combo.setEnabled(restore)
        self.prompt_edit.setPlaceholderText("Optional factual details to preserve, not the source crop or camera angle")

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
        self._progress(0, 0, "Comparing original fragments and detecting their common subject...")
        thread = QThread(self)
        worker = JointWorker(config, self.cancel_event, complete_subject=self.workflow_combo.currentData(),
                             subject_hint=self.subject_hint.text().strip())
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

    @pyqtSlot(object)
    def _completed(self, result):
        super()._completed(result)
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
