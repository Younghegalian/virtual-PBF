from __future__ import annotations

from pathlib import Path
import sys


APP_NAME = "Virtual PBF Workbench"
APP_ORGANIZATION = "Virtual PBF"
APP_ID = "virtual-pbf.workbench"


def app_icon_path() -> Path:
    asset_dir = Path(__file__).resolve().parent / "assets"
    ico_path = asset_dir / "virtual_pbf_icon.ico"
    if ico_path.exists():
        return ico_path
    return asset_dir / "virtual_pbf_icon.svg"


def configure_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:
        return


def apply_app_branding(app) -> None:
    from PySide6.QtGui import QIcon

    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_ORGANIZATION)
    icon = QIcon(str(app_icon_path()))
    if not icon.isNull():
        app.setWindowIcon(icon)
