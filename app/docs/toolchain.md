# Toolchain Plan

The rebuild starts from a clean development environment.

## Recommended Windows Toolchain

Install or expose these on PATH:

- Python 3.11+.
- `uv` for Python environment and dependency management, or standard `venv` + `pip`.
- Git.
- CMake 3.24+.
- Ninja.
- Visual Studio 2022 Build Tools with C++ desktop workload.
- CUDA-capable NVIDIA driver.
- CUDA Toolkit is optional for the current GPU backend because the workbench can use CuPy wheels plus NVIDIA runtime packages. It is still useful later for custom `.cu` kernels.

## Python Environment

Preferred setup:

```powershell
cd "C:\Users\admin\Desktop\workspace\CA_PP\virtual PBF\app"
uv venv --python 3.11
.\.venv\Scripts\Activate.ps1
uv pip install -e ".[dev,gui]"
```

Fallback setup:

```powershell
cd "C:\Users\admin\Desktop\workspace\CA_PP\virtual PBF\app"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,gui]"
```

## Native Environment

Native CPU build:

```powershell
cd "C:\Users\admin\Desktop\workspace\CA_PP\virtual PBF\app\native"
.\build_native.bat
```

The script calls Visual Studio Build Tools, configures CMake/Ninja, and writes `capp_native*.pyd` into `app\src`.

Current GPU setup:

```powershell
cd "C:\Users\admin\Desktop\workspace\CA_PP\virtual PBF\app"
.\.venv\Scripts\python.exe -m pip install cupy-cuda12x nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12
```

The workbench stores CuPy's kernel cache under `app\.cupy_cache` so it does not need write access to the user profile.

## Verification Commands

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m capp.cli inspect-mat "..\data\machine_map\result_007.mat"
.\.venv\Scripts\python.exe -m capp.cli simulate --config examples\minimal_simulation.yaml
.\.venv\Scripts\python.exe launch_workbench.pyw
```

## Current Known Gap

At scaffold creation time, this workspace did not expose `python`, `uv`, `conda`, or `cmake` on PATH. A local `.venv` was later created from `C:\Users\admin\miniconda3\python.exe` using Python 3.13.5. The project venv now carries CMake, Ninja, pybind11, CuPy, and CUDA runtime packages.

