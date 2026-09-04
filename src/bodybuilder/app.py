"""Application entry point."""

from __future__ import annotations

import logging
import sys
import traceback

from PyQt6.QtCore import Qt, qInstallMessageHandler
from PyQt6.QtWidgets import QApplication, QMessageBox

from bodybuilder.ui.main_window import MainWindow


def _qt_message_handler(_mode: object, _context: object, message: str) -> None:
    logging.getLogger("qt").warning(message)


def _show_unhandled_exception(exc_type: type[BaseException], exc: BaseException, tb: object) -> None:
    details = "".join(traceback.format_exception(exc_type, exc, tb))
    logging.getLogger(__name__).critical("Unhandled exception\n%s", details)
    app = QApplication.instance()
    if app is None:
        sys.__excepthook__(exc_type, exc, tb)
        return
    dialog = QMessageBox()
    dialog.setIcon(QMessageBox.Icon.Critical)
    dialog.setWindowTitle("BodyBuilder crashed")
    dialog.setText(str(exc) or exc_type.__name__)
    dialog.setInformativeText("The complete traceback is available below.")
    dialog.setDetailedText(details)
    dialog.exec()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.excepthook = _show_unhandled_exception
    qInstallMessageHandler(_qt_message_handler)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("BodyBuilder")
    app.setOrganizationName("Niko-Pix-kit")
    window = MainWindow()
    window.show()
    return app.exec()
