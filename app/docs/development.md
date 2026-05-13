# Development Setup

The rebuild currently expects a Python 3.11+ environment. The first local workspace environment was created with Python 3.13.5 from Miniconda.

See [toolchain.md](toolchain.md) for the full Windows toolchain plan and [native_strategy.md](native_strategy.md) for the C++/CUDA boundary.

## Python Setup

From `virtual PBF/app`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,gui]"
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Run the minimal CLI simulation:

```powershell
.\.venv\Scripts\python.exe -m capp.cli simulate --config examples/minimal_simulation.yaml
```

Launch the desktop shell:

```powershell
.\.venv\Scripts\python.exe launch_workbench.pyw
```

## Native Setup

Native acceleration is scaffolded but not wired into the Python package yet.

Expected future tools:

- CMake 3.24+
- Visual Studio Build Tools with C++20 support
- CUDA Toolkit for GPU kernels
- nanobind for Python bindings

## Current Environment Note

The current `.venv` is local to `virtual PBF/app/` and was created from `C:\Users\admin\miniconda3\python.exe`.

