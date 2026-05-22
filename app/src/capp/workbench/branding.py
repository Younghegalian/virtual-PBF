from __future__ import annotations

import sys
from pathlib import Path

APP_NAME = "Virtual PBF Workbench"
APP_ORGANIZATION = "Virtual PBF"
APP_ID = "VirtualPBF.Workbench"
_WINDOW_ICON_HANDLES = []


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

        set_app_id = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
        set_app_id.argtypes = [ctypes.c_wchar_p]
        set_app_id.restype = ctypes.c_long
        set_app_id(APP_ID)
    except Exception:
        return


def app_icon():
    from PySide6.QtGui import QIcon

    return QIcon(str(app_icon_path()))


def apply_app_branding(app) -> None:
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_ORGANIZATION)
    if hasattr(app, "setDesktopFileName"):
        app.setDesktopFileName(APP_ID)
    icon = app_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)


def apply_window_branding(window) -> None:
    icon = app_icon()
    if not icon.isNull():
        window.setWindowIcon(icon)
    _apply_windows_window_icon(window)


def _apply_windows_window_icon(window) -> None:
    if sys.platform != "win32":
        return
    icon_path = app_icon_path()
    if icon_path.suffix.lower() != ".ico" or not icon_path.exists():
        return
    try:
        import ctypes

        hwnd = int(window.winId())
        image_icon = 1
        lr_load_from_file = 0x00000010
        lr_default_size = 0x00000040
        wm_seticon = 0x0080
        icon_small = 0
        icon_big = 1
        icon_small2 = 2

        load_image = ctypes.windll.user32.LoadImageW
        load_image.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        load_image.restype = ctypes.c_void_p

        send_message = ctypes.windll.user32.SendMessageW
        send_message.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        send_message.restype = ctypes.c_void_p

        hicon_big = load_image(
            None,
            str(icon_path),
            image_icon,
            0,
            0,
            lr_load_from_file | lr_default_size,
        )
        hicon_small = load_image(None, str(icon_path), image_icon, 16, 16, lr_load_from_file)
        for icon_handle, icon_kind in (
            (hicon_big, icon_big),
            (hicon_small, icon_small),
            (hicon_small, icon_small2),
        ):
            if icon_handle:
                send_message(hwnd, wm_seticon, icon_kind, icon_handle)
                _WINDOW_ICON_HANDLES.append(icon_handle)
    except Exception:
        return
