"""Explicit missing-area selection; originals are never overwritten."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QImage, QMouseEvent, QPainter
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from bodybuilder.core.image_io import fragment_mask_path, load_rgb, save_png


def qt_image(image: Image.Image) -> QImage:
    rgb = image.convert("RGB")
    return QImage(rgb.tobytes(), rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888).copy()


class MaskCanvas(QWidget):
    def __init__(self, image: Image.Image, mask: Image.Image, parent: QWidget) -> None:
        super().__init__(parent)
        self.image, self.mask = image, mask.copy()
        self.erase = False
        self.last_point: tuple[float, float] | None = None
        self.setMinimumSize(420, 320)

    def image_rect(self) -> QRectF:
        scale = min(self.width() / self.image.width, self.height() / self.image.height)
        width, height = self.image.width * scale, self.image.height * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def paintEvent(self, event: object) -> None:
        overlay = Image.new("RGB", self.image.size, (255, 80, 80))
        tinted = Image.blend(self.image, overlay, 0.55)
        preview = Image.composite(tinted, self.image, self.mask)
        painter = QPainter(self)
        painter.drawImage(self.image_rect(), qt_image(preview))
        painter.end()

    def _point(self, position: QPointF) -> tuple[float, float] | None:
        rect = self.image_rect()
        if not rect.contains(position):
            return None
        return ((position.x() - rect.x()) * self.image.width / rect.width(),
                (position.y() - rect.y()) * self.image.height / rect.height())

    def _paint(self, position: QPointF) -> None:
        point = self._point(position)
        if point is None:
            self.last_point = None
            return
        radius = max(1, round(14 * self.image.width / self.image_rect().width()))
        value = 0 if self.erase else 255
        draw = ImageDraw.Draw(self.mask)
        if self.last_point is not None:
            draw.line([self.last_point, point], fill=value, width=radius * 2)
        x, y = point
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=value)
        self.last_point = point
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_point = None
            self._paint(event.position())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._paint(event.position())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self.last_point = None


class MaskEditor(QDialog):
    def __init__(self, path: Path, parent: QWidget) -> None:
        super().__init__(parent)
        self.path = path
        self.setWindowTitle("Mark the area to reconstruct")
        self.resize(760, 740)
        layout = QVBoxLayout(self)
        text = QLabel("Paint only missing/hidden areas in red. Leave visible details untouched. "
                      "Cropped edges are extended automatically; they do not need painting.")
        text.setWordWrap(True)
        layout.addWidget(text)
        image = load_rgb(path)
        mask = Image.new("L", image.size, 0)
        sidecar = fragment_mask_path(path)
        if sidecar.exists():
            with Image.open(sidecar) as existing:
                mask = existing.convert("L")
                if mask.size != image.size:
                    raise ValueError("Existing missing-area mask has different dimensions")
        self.canvas = MaskCanvas(image, mask, self)
        layout.addWidget(self.canvas, 1)
        erase = QCheckBox("Erase marks")
        erase.toggled.connect(self._set_erase)
        layout.addWidget(erase)
        clear = QPushButton("Clear marks")
        clear.clicked.connect(self._clear)
        layout.addWidget(clear)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _set_erase(self, checked: bool) -> None:
        self.canvas.erase = checked

    def _clear(self) -> None:
        self.canvas.mask.paste(0, (0, 0, *self.canvas.mask.size))
        self.canvas.update()

    def _save(self) -> None:
        if ImageOps.invert(self.canvas.mask).getbbox() is None:
            QMessageBox.warning(self, "No evidence left", "Keep at least part of the photograph unmarked.")
            return
        try:
            save_png(self.canvas.mask, fragment_mask_path(self.path))
        except OSError as exc:
            QMessageBox.critical(self, "Cannot save missing-area marks", str(exc))
            return
        self.accept()
