from __future__ import annotations

import os
from contextlib import suppress
from pathlib import Path


def ensure_cupy_cache_dir() -> Path:
    root = Path(__file__).resolve().parents[3]
    cache_dir = root / ".cupy_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUPY_CACHE_DIR", str(cache_dir))

    nvidia_root = root / ".venv" / "Lib" / "site-packages" / "nvidia"
    runtime_root = nvidia_root / "cuda_runtime"
    nvrtc_root = nvidia_root / "cuda_nvrtc"
    if runtime_root.exists():
        os.environ.setdefault("CUDA_PATH", str(runtime_root))
    for dll_dir in (runtime_root / "bin", nvrtc_root / "bin"):
        if dll_dir.exists():
            with suppress(AttributeError, OSError):
                os.add_dll_directory(str(dll_dir))
            path = os.environ.get("PATH", "")
            if str(dll_dir) not in path:
                os.environ["PATH"] = f"{dll_dir}{os.pathsep}{path}"
    return cache_dir
