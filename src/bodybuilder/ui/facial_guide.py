"""Small visual editor for source evidence and the optional target face frame."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from bodybuilder.config import CompletionAspect
from bodybuilder.core.canvas import prepare_outpaint_canvas
from bodybuilder.core.evidence import Guide, Region, image_regions, save_guide, valid_box
from bodybuilder.core.image_io import load_fragment
from bodybuilder.ui.mask_editor import qt_image


class RectangleSelector(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(420, 360)
        self.image = None
        self.box: tuple[float, float, float, float] | None = None
        self._start: QPointF | None = None

    def set_image(self, image) -> None:
        self.image = qt_image(image)
        self.box = None
        self._start = None
        self.update()

    def image_rect(self) -> QRectF:
        if self.image is None:
            return QRectF()
        scale = min(self.width() / self.image.width(), self.height() / self.image.height())
        width, height = self.image.width() * scale, self.image.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def image_point(self, point: QPointF) -> QPointF:
        rect = self.image_rect()
        return QPointF(max(0, min(self.image.width(), (point.x() - rect.x()) * self.image.width() / rect.width())),
                       max(0, min(self.image.height(), (point.y() - rect.y()) * self.image.height() / rect.height())))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.image_rect().contains(event.position()):
            self._start = self.image_point(event.position())
            self.box = None

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._start is None:
            return
        point = self.image_point(event.position())
        self.box = (min(self._start.x(), point.x()), min(self._start.y(), point.y()),
                    max(self._start.x(), point.x()), max(self._start.y(), point.y()))
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self.mouseMoveEvent(event)
        self._start = None

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        if self.image is None:
            return
        rect = self.image_rect()
        painter.drawImage(rect, self.image)
        if self.box:
            x0, y0, x1, y1 = self.box
            sx, sy = rect.width() / self.image.width(), rect.height() / self.image.height()
            selection = QRectF(rect.x() + x0 * sx, rect.y() + y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy)
            painter.fillRect(selection, QColor(40, 170, 230, 55))
            painter.setPen(QPen(QColor(40, 170, 230), 2))
            painter.drawRect(selection)


class FacialGuideDialog(QDialog):
    def __init__(self, path: Path, *, aspect: CompletionAspect = CompletionAspect.PORTRAIT,
                 margin: int = 100, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.path = path
        self.image, self.observed = load_fragment(path)
        try:
            self.image, stored, regions = image_regions(path)
        except ValueError as exc:
            stored, regions = Guide(auto_eyes=False), []
            QMessageBox.warning(self, "Facial guide needs updating",
                str(exc) + "\nDraw new selections and Save to replace the guide, or Cancel to keep it unchanged.")
        self.guide = Guide(list(regions), stored.face_box, stored.auto_eyes)
        self.canvas = prepare_outpaint_canvas(self.image, self.observed, aspect=aspect,
                                               margin_percent=margin, target_long_edge=1024)
        self.setWindowTitle("Facial references - " + path.name)
        self.resize(880, 820)
        layout = QVBoxLayout(self)
        help_text = QLabel(
            "Choose a visible detail, draw a rectangle, then click Use selection. "
            "The best eyes, mouth and hair may come from different original photos. "
            "Eye detections are suggestions: correct them when necessary.\n"
            "For a cropped face, Face location lets you draw the whole intended face on the extended canvas. "
            "Paint white/blurred obstructions as missing in the main window first.")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        self.role = QComboBox()
        for label, value in (("Visible eyes and eyebrows (no eyewear)", "eyes"),
                             ("Visible mouth and surrounding skin", "mouth"),
                             ("Visible hair", "hair"),
                             ("Face location, including missing parts", "face")):
            self.role.addItem(label, value)
        self.role.currentIndexChanged.connect(self._mode_changed)
        layout.addWidget(self.role)
        self.selector = RectangleSelector(self)
        layout.addWidget(self.selector, 1)
        row = QHBoxLayout()
        self.use_button = QPushButton("Use selection")
        self.use_button.clicked.connect(self._use)
        row.addWidget(self.use_button)
        self.remove_button = QPushButton("Remove selected entry")
        self.remove_button.clicked.connect(self._remove)
        row.addWidget(self.remove_button)
        layout.addLayout(row)
        self.entries = QListWidget()
        self.entries.setMaximumHeight(130)
        layout.addWidget(self.entries)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._mode_changed()
        self._refresh_entries()

    def _mode_changed(self) -> None:
        self.selector.set_image(self.canvas.image if self.role.currentData() == "face" else self.image)

    def _refresh_entries(self) -> None:
        self.entries.clear()
        for region in self.guide.regions:
            origin = "automatic suggestion" if region.origin != "user" else "selected"
            self.entries.addItem(f"{region.role}: {tuple(round(v) for v in region.box)} ({origin})")
        if self.guide.face_box is not None:
            self.entries.addItem("Face location: " + str(tuple(round(v) for v in self.guide.face_box)))

    def _use(self) -> None:
        if self.selector.box is None:
            return
        box = self.selector.box
        role = self.role.currentData()
        try:
            if role == "face":
                p = self.canvas.placement
                box = ((box[0] - p.x) * self.image.width / p.source_width,
                       (box[1] - p.y) * self.image.height / p.source_height,
                       (box[2] - p.x) * self.image.width / p.source_width,
                       (box[3] - p.y) * self.image.height / p.source_height)
                self.guide.face_box = valid_box(box, self.image.size, outside=True)
            else:
                box = valid_box(box, self.image.size)
                if self.observed.crop(box).getextrema() != (255, 255):
                    raise ValueError("Select only visible pixels, not the marked missing area")
                self.guide.regions = [r for r in self.guide.regions if r.role != role]
                self.guide.regions.append(Region(role, box))
            self._refresh_entries()
        except ValueError as exc:
            QMessageBox.warning(self, "Check selection", str(exc))

    def _remove(self) -> None:
        row = self.entries.currentRow()
        if 0 <= row < len(self.guide.regions):
            self.guide.regions.pop(row)
        elif row == len(self.guide.regions):
            self.guide.face_box = None
        self._refresh_entries()

    def _save(self) -> None:
        try:
            # Save is explicit confirmation of the currently displayed proposals.
            self.guide.regions = [Region(r.role, r.box) for r in self.guide.regions]
            self.guide.auto_eyes = False
            save_guide(self.path, self.guide)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could not save facial references", str(exc))
            return
        self.accept()
