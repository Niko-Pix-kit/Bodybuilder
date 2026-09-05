"""A folder-to-folder workflow; technical controls are hidden by default."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path

from PyQt6.QtCore import QObject, QSettings, Qt, QThread, QTimer, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QCloseEvent, QDesktopServices, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from bodybuilder.config import (
    CompletionAspect,
    DeviceKind,
    PipelineConfig,
    SubjectKind,
    VariantFrame,
)
from bodybuilder.core.image_io import load_rgb, scan_images
from bodybuilder.core.pipeline import PipelineCallbacks, PipelineRunResult, ReconstructionPipeline
from bodybuilder.ui.mask_editor import MaskEditor, qt_image


class PipelineWorker(QObject):
    message = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)
    preview = pyqtSignal(str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str, str)
    finished = pyqtSignal()

    def __init__(self, config: PipelineConfig, cancelled: threading.Event) -> None:
        super().__init__()
        self.config, self.cancelled = config, cancelled

    @pyqtSlot()
    def run(self) -> None:
        pipeline = ReconstructionPipeline(self.config, cancel_event=self.cancelled,
            callbacks=PipelineCallbacks(log=self.message.emit, progress=self.progress.emit,
                                        preview=lambda path: self.preview.emit(str(path))))
        try:
            self.completed.emit(pipeline.run())
        except Exception as exc:
            # Worker boundary: show the complete failure; never turn it into success.
            details = traceback.format_exc()
            if pipeline.run_dir:
                details += f"\nRun folder: {pipeline.run_dir}"
            self.failed.emit(str(exc), details)
        finally:
            self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BodyBuilder - Photo reconstruction")
        self.resize(1050, 820)
        self.settings = QSettings("Niko-Pix-kit", "BodyBuilder")
        self.worker: PipelineWorker | None = None
        self.worker_thread: QThread | None = None
        self.cancel_event = threading.Event()
        self.last_run_dir: Path | None = None
        self.current_preview: QPixmap | None = None
        self.close_when_finished = False
        self._error_dialog: QMessageBox | None = None
        central = QWidget()
        layout = QVBoxLayout(central)
        title = QLabel("Reconstruct your cropped photographs")
        font = title.font()
        font.setPointSize(font.pointSize() + 5)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        notice = QLabel("Put fragments of the same person or object in one folder. "
                        "Visible parts are preserved; missing parts are generated estimates.")
        notice.setWordWrap(True)
        layout.addWidget(notice)
        self.folder_panel = QWidget()
        folders = QFormLayout(self.folder_panel)
        self.input_edit = QLineEdit(self.settings.value("input_dir", "", str))
        self.output_edit = QLineEdit(self.settings.value("output_dir", "", str))
        folders.addRow("Source folder", self._folder_row(self.input_edit, True))
        folders.addRow("Save results to", self._folder_row(self.output_edit, False))
        layout.addWidget(self.folder_panel)
        self.advanced_toggle = QPushButton("Advanced options")
        self.advanced_toggle.setCheckable(True)
        self.advanced_panel = QWidget()
        advanced = QFormLayout(self.advanced_panel)
        self.subject_combo = QComboBox()
        self.subject_combo.addItem("Person", SubjectKind.PERSON)
        self.subject_combo.addItem("Object / scene", SubjectKind.OBJECT)
        advanced.addRow("Subject", self.subject_combo)
        self.device_combo = QComboBox()
        for text, value in (("Automatic", DeviceKind.AUTO), ("CPU (slow)", DeviceKind.CPU),
                            ("CUDA", DeviceKind.CUDA), ("MPS", DeviceKind.MPS)):
            self.device_combo.addItem(text, value)
        advanced.addRow("Processing", self.device_combo)
        self.aspect_combo = QComboBox()
        for text, value in (("Portrait", CompletionAspect.PORTRAIT), ("Square", CompletionAspect.SQUARE),
                            ("Landscape", CompletionAspect.LANDSCAPE), ("Source proportions", CompletionAspect.AUTO)):
            self.aspect_combo.addItem(text, value)
        advanced.addRow("Result frame", self.aspect_combo)
        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(0, 300)
        self.margin_spin.setValue(100)
        self.margin_spin.setSuffix(" % larger")
        advanced.addRow("Extend cropped edges", self.margin_spin)
        self.variants_spin = QSpinBox()
        self.variants_spin.setRange(0, 16)
        advanced.addRow("Extra synthetic views", self.variants_spin)
        self.upscale_check = QCheckBox("Resize results 2x (preserve observed details)")
        advanced.addRow(self.upscale_check)
        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText("Optional: complete the head, keep the red jacket...")
        advanced.addRow("Description", self.prompt_edit)
        self.advanced_panel.hide()
        self.advanced_toggle.toggled.connect(self.advanced_panel.setVisible)
        layout.addWidget(self.advanced_toggle)
        layout.addWidget(self.advanced_panel)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.tabs = QTabWidget()
        source_panel = QWidget()
        source_layout = QVBoxLayout(source_panel)
        self.sources = QListWidget()
        self.sources.currentItemChanged.connect(self._source_selected)
        source_layout.addWidget(self.sources)
        self.mark_button = QPushButton("Mark missing area in selected photo")
        self.mark_button.clicked.connect(self._edit_mask)
        source_layout.addWidget(self.mark_button)
        self.results = QListWidget()
        self.results.currentItemChanged.connect(self._result_selected)
        self.tabs.addTab(source_panel, "Source photos")
        self.tabs.addTab(self.results, "Reconstructed images")
        splitter.addWidget(self.tabs)
        self.preview_label = QLabel("Select a source photo or a reconstructed image")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(280, 220)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        splitter.addWidget(self.preview_label)
        splitter.setSizes([320, 650])
        layout.addWidget(splitter, 1)
        controls = QHBoxLayout()
        self.reconstruct_button = QPushButton("Reconstruct")
        self.reconstruct_button.setDefault(True)
        self.reconstruct_button.clicked.connect(self._start_reconstruction)
        controls.addWidget(self.reconstruct_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        controls.addWidget(self.cancel_button)
        self.open_output_button = QPushButton("Open reconstructed images")
        self.open_output_button.setEnabled(False)
        self.open_output_button.clicked.connect(self._open_output)
        controls.addWidget(self.open_output_button)
        layout.addLayout(controls)
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        self.status = QLabel("Ready. AI models download on first use; photographs stay on this computer.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.details_toggle = QPushButton("Show processing log")
        self.details_toggle.setCheckable(True)
        layout.addWidget(self.details_toggle)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(3000)
        self.log_edit.setMaximumHeight(140)
        self.log_edit.hide()
        self.details_toggle.toggled.connect(self.log_edit.setVisible)
        layout.addWidget(self.log_edit)
        self.setCentralWidget(central)
        self.input_edit.editingFinished.connect(self._reload_sources)
        self._reload_sources()

    def _folder_row(self, edit: QLineEdit, source: bool) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit)
        button = QPushButton("Browse...")
        button.clicked.connect(lambda: self._browse(edit, source))
        layout.addWidget(button)
        return row

    def _browse(self, edit: QLineEdit, source: bool) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder", edit.text() or str(Path.home()))
        if folder:
            edit.setText(folder)
            if source:
                if not self.output_edit.text():
                    path = Path(folder)
                    self.output_edit.setText(str(path.parent / (path.name + "_reconstructed")))
                self._reload_sources()

    def _reload_sources(self) -> None:
        self.sources.clear()
        text = self.input_edit.text().strip()
        if text and Path(text).is_dir():
            for path in scan_images(Path(text)):
                item = QListWidgetItem(str(path.relative_to(Path(text))))
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                self.sources.addItem(item)

    def _collect_config(self) -> PipelineConfig:
        if not self.input_edit.text().strip() or not self.output_edit.text().strip():
            raise ValueError("Select both the source folder and the output folder")
        return PipelineConfig(input_dir=Path(self.input_edit.text().strip()),
            output_dir=Path(self.output_edit.text().strip()),
            subject_kind=SubjectKind(self.subject_combo.currentData()),
            device=DeviceKind(self.device_combo.currentData()),
            completion_aspect=CompletionAspect(self.aspect_combo.currentData()),
            completion_margin_percent=self.margin_spin.value(),
            synthetic_variants=self.variants_spin.value(), variant_frame=VariantFrame.PORTRAIT,
            upscale_2x=self.upscale_check.isChecked(), custom_prompt=self.prompt_edit.text().strip())

    def _start_reconstruction(self) -> None:
        if self.worker_thread is not None:
            return
        try:
            config = self._collect_config()
        except ValueError as exc:
            QMessageBox.warning(self, "Check folders", str(exc))
            return
        self._reload_sources()
        self.results.clear()
        self.log_edit.clear()
        self.cancel_event = threading.Event()
        self.last_run_dir = None
        self.open_output_button.setEnabled(False)
        self._set_busy(True)
        self._progress(0, 0, "Analyzing source photographs...")
        thread = QThread(self)
        worker = PipelineWorker(config, self.cancel_event)
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

    def _set_busy(self, busy: bool) -> None:
        for widget in (self.folder_panel, self.advanced_panel, self.mark_button, self.reconstruct_button):
            widget.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    def _cancel(self) -> None:
        self.cancel_event.set()
        self.cancel_button.setEnabled(False)
        self.status.setText("Cancellation requested. Waiting for the current safe checkpoint; model downloads may finish first.")

    @pyqtSlot(int, int, str)
    def _progress(self, current: int, total: int, text: str) -> None:
        self.progress_bar.setRange(0, max(0, total))
        self.progress_bar.setValue(current)
        self.status.setText(text)

    @pyqtSlot(str)
    def _add_result(self, path: str) -> None:
        item = QListWidgetItem(Path(path).name)
        item.setData(Qt.ItemDataRole.UserRole, path)
        self.results.addItem(item)
        self.last_run_dir = Path(path).parent.parent
        self.open_output_button.setEnabled(True)
        self.results.setCurrentItem(item)
        self.tabs.setCurrentIndex(1)

    @pyqtSlot(object)
    def _completed(self, result: PipelineRunResult) -> None:
        self.last_run_dir = result.run_dir
        text = "Cancelled" if result.cancelled else "Complete"
        if result.errors:
            text += f" with {len(result.errors)} recorded problem(s); see log"
        self.status.setText(f"{text}. {len(result.output_paths)} reconstructed image(s).")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0 if result.cancelled else 1)

    @pyqtSlot(str, str)
    def _failed(self, message: str, details: str) -> None:
        self.log_edit.appendPlainText(details)
        self.details_toggle.setChecked(True)
        self.status.setText("Reconstruction failed. See the error below; diagnostic masks are not results.")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        dialog = QMessageBox(QMessageBox.Icon.Critical, "Reconstruction failed", message, parent=self)
        dialog.setDetailedText(details)
        self._error_dialog = dialog
        dialog.open()

    @pyqtSlot()
    def _thread_finished(self) -> None:
        self.worker = None
        self.worker_thread = None
        self._set_busy(False)
        if self.close_when_finished:
            self.close_when_finished = False
            QTimer.singleShot(0, self.close)

    def _source_selected(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        self._show_item(current)

    def _result_selected(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        self._show_item(current)

    def _show_item(self, item: QListWidgetItem | None) -> None:
        if item is None:
            return
        try:
            image = load_rgb(Path(item.data(Qt.ItemDataRole.UserRole)))
            image.thumbnail((1600, 1600))
            self.current_preview = QPixmap.fromImage(qt_image(image))
            self._refresh_preview()
        except (OSError, ValueError) as exc:
            self.preview_label.setText(str(exc))

    def _refresh_preview(self) -> None:
        if self.current_preview is not None:
            self.preview_label.setPixmap(self.current_preview.scaled(self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        self._refresh_preview()

    def _edit_mask(self) -> None:
        item = self.sources.currentItem()
        if item is None:
            QMessageBox.information(self, "Select a source photo", "Select a photo in the source list first.")
            return
        try:
            MaskEditor(Path(item.data(Qt.ItemDataRole.UserRole)), self).exec()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Cannot mark this photo", str(exc))

    def _open_output(self) -> None:
        if self.last_run_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_run_dir / "images")))

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker_thread is not None:
            answer = QMessageBox.question(self, "Cancel reconstruction?", "Cancel and close after the next safe checkpoint?")
            if answer == QMessageBox.StandardButton.Yes:
                self.close_when_finished = True
                self._cancel()
            event.ignore()
            return
        self.settings.setValue("input_dir", self.input_edit.text())
        self.settings.setValue("output_dir", self.output_edit.text())
        event.accept()
