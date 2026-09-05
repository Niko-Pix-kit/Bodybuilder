"""Facial reference controls on top of the folder-to-folder desktop workflow."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox, QPushButton

from bodybuilder.config import CompletionAspect
from bodybuilder.core.pipeline import PipelineRunResult
from bodybuilder.ui.facial_guide import FacialGuideDialog
from bodybuilder.ui.main_window import MainWindow


class GuidedMainWindow(MainWindow):
    def __init__(self) -> None:
        self._facial_review_required = 0
        super().__init__()
        self.reference_button = QPushButton("Facial references in selected photo")
        self.reference_button.setToolTip("Use visible eyes, mouth or hair from different originals. No technical sliders.")
        self.reference_button.clicked.connect(self._edit_references)
        self.mark_button.parentWidget().layout().addWidget(self.reference_button)

    def _edit_references(self) -> None:
        item = self.sources.currentItem()
        if item is None:
            QMessageBox.information(self, "Select a source photo", "Select an original photo in the source list first.")
            return
        try:
            FacialGuideDialog(Path(item.data(Qt.ItemDataRole.UserRole)),
                aspect=CompletionAspect(self.aspect_combo.currentData()),
                margin=self.margin_spin.value(), parent=self).exec()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Cannot read facial guide", str(exc))

    def _set_busy(self, busy: bool) -> None:
        super()._set_busy(busy)
        self.reference_button.setEnabled(not busy)

    def _start_reconstruction(self) -> None:
        if self.worker_thread is None:
            self._facial_review_required = 0
        super()._start_reconstruction()

    def _add_result(self, path: str) -> None:
        super()._add_result(path)
        try:
            with Image.open(path) as image:
                record = json.loads(image.info.get("BodyBuilder", "{}"))
            guide = record.get("generation", {}).get("facial_guidance", {})
            evidence = guide.get("evidence", [])
            if evidence:
                summary = "; ".join(f"{part['role']} from {Path(part['source']).name}" for part in evidence)
                self.log_edit.appendPlainText("Facial references: " + summary)
            if guide.get("status") in {"no_regional_evidence", "face_location_needed"}:
                self._facial_review_required += 1
                self.details_toggle.setChecked(True)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.log_edit.appendPlainText(f"Could not read facial guidance report: {exc}")

    def _completed(self, result: PipelineRunResult) -> None:
        super()._completed(result)
        if self._facial_review_required:
            self.status.setText(self.status.text() +
                " Facial guidance was not applied to some images: use Facial references to confirm details/location.")
