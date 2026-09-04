"""Main PyQt6 window for BodyBuilder."""

from __future__ import annotations

import importlib.util
import threading
import traceback
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QSettings, QSize, Qt, QThread, QTimer, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QCloseEvent, QDesktopServices, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from bodybuilder import __version__
from bodybuilder.config import (
    BackendKind,
    CompletionAspect,
    DeviceKind,
    PipelineConfig,
    SubjectKind,
    VariantFrame,
)
from bodybuilder.core.pipeline import (
    AnalysisRunResult,
    PipelineCallbacks,
    PipelineCancelled,
    PipelineRunResult,
    ReconstructionPipeline,
    analyze_input_folder,
)
from bodybuilder.core.types import ImageAnalysis


class PipelineWorker(QObject):
    log_message = pyqtSignal(str)
    progress_changed = pyqtSignal(int, int, str)
    analysis_found = pyqtSignal(object)
    preview_ready = pyqtSignal(str)
    completed = pyqtSignal(object)
    cancelled = pyqtSignal(object)
    failed = pyqtSignal(str, str)

    def __init__(self, config: PipelineConfig, mode: str) -> None:
        super().__init__()
        self.config = config
        self.mode = mode
        self.cancel_event = threading.Event()

    @pyqtSlot()
    def run(self) -> None:
        callbacks = PipelineCallbacks(
            log=self.log_message.emit,
            progress=self.progress_changed.emit,
            preview=lambda path: self.preview_ready.emit(str(path)),
            analysis_item=self.analysis_found.emit,
        )
        try:
            if self.mode == "analyze":
                result = analyze_input_folder(
                    self.config,
                    callbacks=callbacks,
                    cancel_event=self.cancel_event,
                )
            else:
                result = ReconstructionPipeline(
                    self.config,
                    callbacks=callbacks,
                    cancel_event=self.cancel_event,
                ).run()
            if getattr(result, "cancelled", False):
                self.cancelled.emit(result)
            else:
                self.completed.emit(result)
        except PipelineCancelled as exc:
            self.cancelled.emit(exc)
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc())

    def request_cancel(self) -> None:
        self.cancel_event.set()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"BodyBuilder {__version__}")
        self.resize(1280, 820)
        self.setMinimumSize(1050, 700)
        self.settings = QSettings("Niko-Pix-kit", "BodyBuilder")
        self.worker: PipelineWorker | None = None
        self.worker_thread: QThread | None = None
        self.last_run_dir: Path | None = None
        self.current_preview_path: Path | None = None
        self.close_after_worker = False

        self._build_ui()
        self._restore_settings()
        self._update_ai_status()
        self._update_backend_controls()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        title = QLabel("BodyBuilder — Fragment Reconstruction Studio")
        title_font = title.font()
        title_font.setPointSize(title_font.pointSize() + 5)
        title_font.setBold(True)
        title.setFont(title_font)
        root.addWidget(title)

        truth_notice = QLabel(
            "Source completions prevent the AI from repainting visible input regions. "
            "Missing regions and all pose variants are generated estimates, not recovered evidence."
        )
        truth_notice.setWordWrap(True)
        truth_notice.setFrameShape(QFrame.Shape.StyledPanel)
        truth_notice.setContentsMargins(9, 7, 9, 7)
        root.addWidget(truth_notice)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_settings_panel())
        splitter.addWidget(self._build_results_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 820])
        root.addWidget(splitter, 1)

        controls = QHBoxLayout()
        self.analyze_button = QPushButton("Analyze source")
        self.analyze_button.clicked.connect(self._start_analysis)
        controls.addWidget(self.analyze_button)

        self.reconstruct_button = QPushButton("Reconstruct")
        self.reconstruct_button.setDefault(True)
        self.reconstruct_button.clicked.connect(self._start_reconstruction)
        controls.addWidget(self.reconstruct_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_work)
        controls.addWidget(self.cancel_button)

        self.open_output_button = QPushButton("Open output")
        self.open_output_button.setEnabled(False)
        self.open_output_button.clicked.connect(self._open_output)
        controls.addWidget(self.open_output_button)

        controls.addStretch(1)
        self.progress = QProgressBar()
        self.progress.setMinimumWidth(330)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("Ready")
        controls.addWidget(self.progress)
        root.addLayout(controls)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def _build_settings_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(390)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 6, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 4, 0)
        content_layout.setSpacing(10)

        folders = QGroupBox("Folders")
        folders_layout = QFormLayout(folders)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Folder containing cropped or fragmented photographs")
        folders_layout.addRow("Source folder", self._folder_row(self.input_edit, self._browse_input))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Target folder for reconstruction runs")
        folders_layout.addRow("Output folder", self._folder_row(self.output_edit, self._browse_output))
        content_layout.addWidget(folders)

        reconstruction = QGroupBox("Reconstruction")
        form = QFormLayout(reconstruction)
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("Local SDXL + IP-Adapter", BackendKind.SDXL.value)
        self.backend_combo.addItem("Classical preview (no generative AI)", BackendKind.CLASSICAL.value)
        self.backend_combo.currentIndexChanged.connect(self._update_backend_controls)
        form.addRow("Engine", self.backend_combo)

        self.ai_status_label = QLabel()
        self.ai_status_label.setWordWrap(True)
        form.addRow("AI status", self.ai_status_label)

        self.subject_combo = QComboBox()
        self.subject_combo.addItem("Auto-detect", SubjectKind.AUTO.value)
        self.subject_combo.addItem("Person", SubjectKind.PERSON.value)
        self.subject_combo.addItem("Object / scene", SubjectKind.OBJECT.value)
        form.addRow("Subject", self.subject_combo)

        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto", DeviceKind.AUTO.value)
        self.device_combo.addItem("CUDA", DeviceKind.CUDA.value)
        self.device_combo.addItem("MPS", DeviceKind.MPS.value)
        self.device_combo.addItem("CPU", DeviceKind.CPU.value)
        form.addRow("Compute device", self.device_combo)

        self.aspect_combo = QComboBox()
        self.aspect_combo.addItem("Keep source aspect", CompletionAspect.AUTO.value)
        self.aspect_combo.addItem("Portrait 4:5", CompletionAspect.PORTRAIT.value)
        self.aspect_combo.addItem("Square 1:1", CompletionAspect.SQUARE.value)
        self.aspect_combo.addItem("Landscape 3:2", CompletionAspect.LANDSCAPE.value)
        form.addRow("Completion frame", self.aspect_combo)

        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(0, 300)
        self.margin_spin.setSuffix(" %")
        self.margin_spin.setValue(60)
        self.margin_spin.setToolTip("How much larger the completion canvas is than the observed source")
        form.addRow("Completion margin", self.margin_spin)

        self.completions_spin = QSpinBox()
        self.completions_spin.setRange(1, 8)
        self.completions_spin.setValue(1)
        form.addRow("Completions / source", self.completions_spin)

        self.variants_spin = QSpinBox()
        self.variants_spin.setRange(0, 16)
        self.variants_spin.setValue(0)
        self.variants_spin.setToolTip("Entirely generated views; no source pixels are preserved")
        form.addRow("Synthetic variants", self.variants_spin)

        self.variant_frame_combo = QComboBox()
        self.variant_frame_combo.addItem("Full body / object 2:3", VariantFrame.FULL_BODY.value)
        self.variant_frame_combo.addItem("Portrait 4:5", VariantFrame.PORTRAIT.value)
        self.variant_frame_combo.addItem("Square 1:1", VariantFrame.SQUARE.value)
        self.variant_frame_combo.addItem("Landscape 3:2", VariantFrame.LANDSCAPE.value)
        form.addRow("Variant frame", self.variant_frame_combo)

        self.target_size_combo = QComboBox()
        for value in (768, 1024, 1280, 1536):
            self.target_size_combo.addItem(str(value), value)
        self.target_size_combo.setCurrentIndex(1)
        form.addRow("AI long edge", self.target_size_combo)
        content_layout.addWidget(reconstruction)

        evidence = QGroupBox("Evidence handling")
        evidence_layout = QVBoxLayout(evidence)
        self.group_all_check = QCheckBox("Treat all images as one subject")
        self.group_all_check.setChecked(True)
        self.group_all_check.setToolTip(
            "Recommended for disconnected body-part fragments; automatic grouping can separate "
            "visually dissimilar parts"
        )
        evidence_layout.addWidget(self.group_all_check)
        self.stitch_check = QCheckBox("Stitch genuinely overlapping fragments before AI completion")
        self.stitch_check.setChecked(True)
        evidence_layout.addWidget(self.stitch_check)
        self.recursive_check = QCheckBox("Include subfolders")
        self.recursive_check.setChecked(True)
        evidence_layout.addWidget(self.recursive_check)
        self.upscale_check = QCheckBox("Enhance final images 2x")
        self.upscale_check.setChecked(True)
        evidence_layout.addWidget(self.upscale_check)
        content_layout.addWidget(evidence)

        advanced = QGroupBox("Generation controls")
        advanced_form = QFormLayout(advanced)
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 150)
        self.steps_spin.setValue(30)
        advanced_form.addRow("Steps", self.steps_spin)

        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(0.0, 30.0)
        self.guidance_spin.setDecimals(2)
        self.guidance_spin.setSingleStep(0.25)
        self.guidance_spin.setValue(6.0)
        advanced_form.addRow("Text guidance", self.guidance_spin)

        self.strength_spin = QDoubleSpinBox()
        self.strength_spin.setRange(0.05, 1.0)
        self.strength_spin.setDecimals(2)
        self.strength_spin.setSingleStep(0.02)
        self.strength_spin.setValue(0.98)
        advanced_form.addRow("Denoising strength", self.strength_spin)

        self.fidelity_spin = QDoubleSpinBox()
        self.fidelity_spin.setRange(0.0, 1.5)
        self.fidelity_spin.setDecimals(2)
        self.fidelity_spin.setSingleStep(0.05)
        self.fidelity_spin.setValue(0.75)
        advanced_form.addRow("Reference fidelity", self.fidelity_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(137)
        advanced_form.addRow("Base seed", self.seed_spin)

        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText(
            "Optional factual evidence, for example: blue jacket, scar on left eyebrow, red metal object."
        )
        self.prompt_edit.setMaximumHeight(76)
        advanced_form.addRow("Additional evidence", self.prompt_edit)
        content_layout.addWidget(advanced)
        content_layout.addStretch(1)

        scroll.setWidget(content)
        layout.addWidget(scroll)
        return panel

    def _build_results_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 0, 0, 0)
        layout.setSpacing(8)

        self.analysis_table = QTableWidget(0, 5)
        self.analysis_table.setHorizontalHeaderLabels(
            ["File", "Dimensions", "Quality", "Faces", "Warnings"]
        )
        self.analysis_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.analysis_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.analysis_table.verticalHeader().setVisible(False)
        header = self.analysis_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.analysis_table.itemSelectionChanged.connect(self._show_selected_source)
        layout.addWidget(self.analysis_table, 2)

        self.preview_label = QLabel("Select an analyzed image or wait for a generated output")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(290)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_label.setFrameShape(QFrame.Shape.StyledPanel)
        layout.addWidget(self.preview_label, 3)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(5000)
        self.log_edit.setPlaceholderText("Run messages and detailed errors appear here.")
        layout.addWidget(self.log_edit, 2)
        return panel

    def _folder_row(self, line_edit: QLineEdit, callback: Any) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit, 1)
        button = QPushButton("Browse…")
        button.clicked.connect(callback)
        layout.addWidget(button)
        return widget

    def _browse_input(self) -> None:
        initial = self.input_edit.text() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select source folder", initial)
        if folder:
            self.input_edit.setText(folder)
            if not self.output_edit.text():
                source = Path(folder)
                self.output_edit.setText(str(source.parent / f"{source.name}_BodyBuilder_output"))

    def _browse_output(self) -> None:
        initial = self.output_edit.text() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select output folder", initial)
        if folder:
            self.output_edit.setText(folder)

    def _collect_config(self) -> PipelineConfig:
        input_text = self.input_edit.text().strip()
        output_text = self.output_edit.text().strip()
        if not input_text:
            raise ValueError("Select a source folder")
        if not output_text:
            raise ValueError("Select an output folder")
        return PipelineConfig(
            input_dir=Path(input_text),
            output_dir=Path(output_text),
            subject_kind=SubjectKind(self.subject_combo.currentData()),
            backend=BackendKind(self.backend_combo.currentData()),
            device=DeviceKind(self.device_combo.currentData()),
            completion_aspect=CompletionAspect(self.aspect_combo.currentData()),
            variant_frame=VariantFrame(self.variant_frame_combo.currentData()),
            completion_margin_percent=self.margin_spin.value(),
            completions_per_source=self.completions_spin.value(),
            synthetic_variants=self.variants_spin.value(),
            target_long_edge=int(self.target_size_combo.currentData()),
            inference_steps=self.steps_spin.value(),
            guidance_scale=self.guidance_spin.value(),
            denoising_strength=self.strength_spin.value(),
            reference_fidelity=self.fidelity_spin.value(),
            seed=self.seed_spin.value(),
            group_all_images=self.group_all_check.isChecked(),
            stitch_overlaps=self.stitch_check.isChecked(),
            upscale_2x=self.upscale_check.isChecked(),
            recursive_scan=self.recursive_check.isChecked(),
            custom_prompt=self.prompt_edit.toPlainText().strip(),
        )

    def _start_analysis(self) -> None:
        try:
            config = self._collect_config()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid settings", str(exc))
            return
        self.analysis_table.setRowCount(0)
        self._start_worker(config, "analyze")

    def _start_reconstruction(self) -> None:
        try:
            config = self._collect_config()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid settings", str(exc))
            return
        if config.backend == BackendKind.SDXL and not self._ai_dependencies_available():
            QMessageBox.critical(
                self,
                "AI dependencies missing",
                'Install the local AI stack with:\n\npython -m pip install -e ".[ai]"',
            )
            return
        self.analysis_table.setRowCount(0)
        self._start_worker(config, "run")

    def _start_worker(self, config: PipelineConfig, mode: str) -> None:
        if self.worker is not None:
            return
        self.log_edit.clear()
        self._set_busy(True)
        self.progress.setRange(0, 0)
        self.progress.setFormat("Starting…")

        thread = QThread(self)
        worker = PipelineWorker(config, mode)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log_message.connect(self._append_log)
        worker.progress_changed.connect(self._update_progress)
        worker.analysis_found.connect(self._add_analysis)
        worker.preview_ready.connect(self._show_preview_path)
        worker.completed.connect(self._worker_completed)
        worker.cancelled.connect(self._worker_cancelled)
        worker.failed.connect(self._worker_failed)
        worker.completed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.completed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._worker_thread_finished)
        thread.finished.connect(thread.deleteLater)

        self.worker = worker
        self.worker_thread = thread
        thread.start()

    def _cancel_work(self) -> None:
        if self.worker is not None:
            self.worker.request_cancel()
            self.cancel_button.setEnabled(False)
            self.progress.setFormat("Cancelling at the next safe checkpoint…")
            self._append_log("Cancellation requested.")

    @pyqtSlot(object)
    def _worker_completed(self, result: AnalysisRunResult | PipelineRunResult) -> None:
        if isinstance(result, AnalysisRunResult):
            self._append_log(
                f"Analysis complete: {len(result.analyses)} usable image(s), "
                f"{len(result.failures)} failure(s)."
            )
            self.statusBar().showMessage("Analysis complete")
        else:
            self.last_run_dir = result.run_dir
            self.open_output_button.setEnabled(True)
            self._append_log(
                f"Run complete: {len(result.output_paths)} output image(s), "
                f"{len(result.errors)} recorded error(s)."
            )
            self.statusBar().showMessage(f"Completed: {result.run_dir}")
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.progress.setFormat("Complete")

    @pyqtSlot(object)
    def _worker_cancelled(self, result: object) -> None:
        if isinstance(result, PipelineRunResult):
            self.last_run_dir = result.run_dir
            self.open_output_button.setEnabled(True)
        self._append_log("Operation cancelled. Completed files and the manifest were kept.")
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("Cancelled")
        self.statusBar().showMessage("Cancelled")

    @pyqtSlot(str, str)
    def _worker_failed(self, message: str, details: str) -> None:
        self._append_log(details)
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle("BodyBuilder operation failed")
        dialog.setText(message or "The operation failed")
        dialog.setInformativeText(
            "The complete traceback is available below and in the run log when a run folder was created."
        )
        dialog.setDetailedText(details)
        dialog.exec()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("Failed")
        self.statusBar().showMessage("Failed")

    @pyqtSlot()
    def _worker_thread_finished(self) -> None:
        self.worker = None
        self.worker_thread = None
        self._set_busy(False)
        if self.close_after_worker:
            self.close_after_worker = False
            QTimer.singleShot(0, self.close)

    @pyqtSlot(int, int, str)
    def _update_progress(self, current: int, total: int, message: str) -> None:
        if total <= 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, total)
            self.progress.setValue(max(0, min(current, total)))
        self.progress.setFormat(message)
        self.statusBar().showMessage(message)

    @pyqtSlot(object)
    def _add_analysis(self, analysis: ImageAnalysis) -> None:
        row = self.analysis_table.rowCount()
        self.analysis_table.insertRow(row)
        filename = QTableWidgetItem(analysis.path.name)
        filename.setData(Qt.ItemDataRole.UserRole, str(analysis.path))
        filename.setToolTip(str(analysis.path))
        self.analysis_table.setItem(row, 0, filename)
        self.analysis_table.setItem(
            row,
            1,
            QTableWidgetItem(f"{analysis.width} × {analysis.height}"),
        )
        quality = QTableWidgetItem(f"{analysis.quality_score * 100:.0f}%")
        quality.setToolTip(
            f"Blur {analysis.blur_score:.2f}; exposure {analysis.exposure_score:.2f}; "
            f"detail {analysis.detail_score:.2f}; resolution {analysis.resolution_score:.2f}"
        )
        self.analysis_table.setItem(row, 2, quality)
        self.analysis_table.setItem(row, 3, QTableWidgetItem(str(len(analysis.faces))))
        self.analysis_table.setItem(row, 4, QTableWidgetItem("; ".join(analysis.warnings)))

    def _show_selected_source(self) -> None:
        selected = self.analysis_table.selectedItems()
        if not selected:
            return
        item = self.analysis_table.item(selected[0].row(), 0)
        if item is not None:
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                self._show_preview_path(str(path))

    @pyqtSlot(str)
    def _show_preview_path(self, path: str) -> None:
        candidate = Path(path)
        if not candidate.exists():
            return
        self.current_preview_path = candidate
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        if self.current_preview_path is None:
            return
        pixmap = QPixmap(str(self.current_preview_path))
        if pixmap.isNull():
            self.preview_label.setText(f"Could not preview {self.current_preview_path.name}")
            return
        available = self.preview_label.size() - QSize(20, 20)
        scaled = pixmap.scaled(
            available,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)
        self.preview_label.setToolTip(str(self.current_preview_path))

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_preview()

    @pyqtSlot(str)
    def _append_log(self, message: str) -> None:
        self.log_edit.appendPlainText(message.rstrip())

    def _set_busy(self, busy: bool) -> None:
        self.analyze_button.setEnabled(not busy)
        self.reconstruct_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        if not busy:
            self._update_backend_controls()

    def _open_output(self) -> None:
        target = self.last_run_dir or Path(self.output_edit.text().strip())
        if target and target.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _update_ai_status(self) -> None:
        if self._ai_dependencies_available():
            self.ai_status_label.setText("Installed. Models download locally on first use.")
        else:
            self.ai_status_label.setText('Not installed. Run: python -m pip install -e ".[ai]"')

    @staticmethod
    def _ai_dependencies_available() -> bool:
        return all(
            importlib.util.find_spec(name) is not None
            for name in ("torch", "diffusers", "transformers", "accelerate", "safetensors")
        )

    def _update_backend_controls(self) -> None:
        is_sdxl = self.backend_combo.currentData() == BackendKind.SDXL.value
        for widget in (
            self.device_combo,
            self.variants_spin,
            self.variant_frame_combo,
            self.steps_spin,
            self.guidance_spin,
            self.strength_spin,
            self.fidelity_spin,
        ):
            widget.setEnabled(is_sdxl and self.worker is None)
        self.upscale_check.setEnabled(self.worker is None)

    def _restore_settings(self) -> None:
        self.input_edit.setText(self.settings.value("input_dir", "", str))
        self.output_edit.setText(self.settings.value("output_dir", "", str))
        self._set_combo_by_data(
            self.backend_combo,
            self.settings.value("backend", BackendKind.SDXL.value, str),
        )
        self._set_combo_by_data(
            self.subject_combo,
            self.settings.value("subject", SubjectKind.AUTO.value, str),
        )
        self._set_combo_by_data(
            self.device_combo,
            self.settings.value("device", DeviceKind.AUTO.value, str),
        )
        self._set_combo_by_data(
            self.aspect_combo,
            self.settings.value("completion_aspect", CompletionAspect.AUTO.value, str),
        )
        self._set_combo_by_data(
            self.variant_frame_combo,
            self.settings.value("variant_frame", VariantFrame.FULL_BODY.value, str),
        )
        self._set_combo_by_data(
            self.target_size_combo,
            self.settings.value("target_long_edge", 1024, int),
        )
        self.margin_spin.setValue(self.settings.value("completion_margin", 60, int))
        self.completions_spin.setValue(self.settings.value("completions_per_source", 1, int))
        self.variants_spin.setValue(self.settings.value("synthetic_variants", 0, int))
        self.steps_spin.setValue(self.settings.value("inference_steps", 30, int))
        self.guidance_spin.setValue(self.settings.value("guidance_scale", 6.0, float))
        self.strength_spin.setValue(self.settings.value("denoising_strength", 0.98, float))
        self.fidelity_spin.setValue(self.settings.value("reference_fidelity", 0.75, float))
        self.seed_spin.setValue(self.settings.value("seed", 137, int))
        self.group_all_check.setChecked(self.settings.value("group_all", True, bool))
        self.stitch_check.setChecked(self.settings.value("stitch", True, bool))
        self.recursive_check.setChecked(self.settings.value("recursive", True, bool))
        self.upscale_check.setChecked(self.settings.value("upscale", True, bool))
        self.prompt_edit.setPlainText(self.settings.value("custom_prompt", "", str))

    def _save_settings(self) -> None:
        self.settings.setValue("input_dir", self.input_edit.text())
        self.settings.setValue("output_dir", self.output_edit.text())
        self.settings.setValue("backend", self.backend_combo.currentData())
        self.settings.setValue("subject", self.subject_combo.currentData())
        self.settings.setValue("device", self.device_combo.currentData())
        self.settings.setValue("completion_aspect", self.aspect_combo.currentData())
        self.settings.setValue("variant_frame", self.variant_frame_combo.currentData())
        self.settings.setValue("target_long_edge", self.target_size_combo.currentData())
        self.settings.setValue("completion_margin", self.margin_spin.value())
        self.settings.setValue("completions_per_source", self.completions_spin.value())
        self.settings.setValue("synthetic_variants", self.variants_spin.value())
        self.settings.setValue("inference_steps", self.steps_spin.value())
        self.settings.setValue("guidance_scale", self.guidance_spin.value())
        self.settings.setValue("denoising_strength", self.strength_spin.value())
        self.settings.setValue("reference_fidelity", self.fidelity_spin.value())
        self.settings.setValue("seed", self.seed_spin.value())
        self.settings.setValue("group_all", self.group_all_check.isChecked())
        self.settings.setValue("stitch", self.stitch_check.isChecked())
        self.settings.setValue("recursive", self.recursive_check.isChecked())
        self.settings.setValue("upscale", self.upscale_check.isChecked())
        self.settings.setValue("custom_prompt", self.prompt_edit.toPlainText())

    @staticmethod
    def _set_combo_by_data(combo: QComboBox, value: Any) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self.worker is not None:
            answer = QMessageBox.question(
                self,
                "Cancel active operation?",
                "Closing BodyBuilder will cancel the active operation at the next safe checkpoint.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.worker.request_cancel()
            self.close_after_worker = True
            self.cancel_button.setEnabled(False)
            self.progress.setFormat("Cancelling before exit…")
            event.ignore()
            return
        self._save_settings()
        event.accept()
