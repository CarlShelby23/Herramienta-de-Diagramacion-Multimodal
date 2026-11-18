from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication, QDialog

from src.ui.theme import APP_STYLESHEET
from src.ui.wf01_setup_dialog import WF01SetupDialog
from src.ui.wf02_main_window import WF02MainWindow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Herramienta de Diagramación Multimodal (Voz + Gestos) WF-01..WF-05"
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Inicia WF-01 con el modo simulación preseleccionado (sin cámara ni micrófono).",
    )
    args = parser.parse_args(argv)

    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)

    setup = WF01SetupDialog(start_hardware=not args.sim)
    if args.sim:
        setup.chk_simulation.setChecked(True)

    if setup.exec() != QDialog.DialogCode.Accepted or setup.session_config is None:
        return 0

    window = WF02MainWindow(
        config=setup.session_config,
        start_workers=not setup.session_config.calibration.simulation_mode,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
