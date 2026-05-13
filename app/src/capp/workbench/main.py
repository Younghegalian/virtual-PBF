from __future__ import annotations

import sys


def main() -> int:
    from capp.workbench.branding import configure_windows_app_id

    configure_windows_app_id()
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("PySide6 is not installed. Install the gui extra to run the workbench.")
        return 1

    from capp.workbench.app import WorkbenchMainWindow
    from capp.workbench.branding import apply_app_branding

    app = QApplication(sys.argv)
    apply_app_branding(app)
    window = WorkbenchMainWindow()
    window.show(maximized=True)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
