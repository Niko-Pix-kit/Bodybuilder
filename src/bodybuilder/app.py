"""Bootstrap diagnostics before importing optional desktop dependencies."""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path


def main() -> int:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_path = Path.home() / ".local" / "state" / "bodybuilder" / "startup.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    except OSError as exc:
        print(f"Could not open startup log: {exc}", file=sys.stderr)
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(message)s", force=True)
    try:
        from PyQt6.QtCore import Qt, qInstallMessageHandler
        from PyQt6.QtWidgets import QApplication, QMessageBox
    except (ImportError, OSError) as exc:
        logging.exception("Desktop dependencies are missing or incompatible")
        print(f'Install requirements.txt using this interpreter:\n"{sys.executable}" -m pip install -r requirements.txt\n{exc}', file=sys.stderr)
        return 1

    def unhandled(exc_type: type[BaseException], exc: BaseException, tb: object) -> None:
        details = "".join(traceback.format_exception(exc_type, exc, tb))
        logging.critical("Unhandled application error\n%s", details)
        if QApplication.instance() is None:
            return
        dialog = QMessageBox(QMessageBox.Icon.Critical, "BodyBuilder error", str(exc))
        dialog.setInformativeText(f"Details are recorded in {log_path}")
        dialog.setDetailedText(details)
        dialog.exec()

    sys.excepthook = unhandled
    qInstallMessageHandler(lambda _mode, _context, message: logging.getLogger("qt").warning(message))
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName("BodyBuilder")
    app.setOrganizationName("Niko-Pix-kit")
    try:
        from bodybuilder.ui.main_window import MainWindow
        window = MainWindow()
    except Exception:
        # Startup boundary: report and exit nonzero, without swallowing the traceback.
        unhandled(*sys.exc_info())
        return 1
    window.show()
    return app.exec()
