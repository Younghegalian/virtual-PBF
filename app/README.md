# Virtual PBF Workbench

Desktop workbench for virtual PBF simulation, ROI calibration, and build-plate parameter maps.

The MATLAB prototype is treated as a behavioral reference only. This directory contains the new application and compute stack.

## Current Goal

Build the first vertical slice:

1. Load an STL geometry.
2. Voxelize it.
3. Run a reference layerwise print simulation.
4. Produce probability and binary volume outputs.
5. Inspect slices in a desktop workbench.
6. Export NPZ and VTK-compatible results.

## Planned Stack

- Python core package for orchestration, data, I/O, optimization, and reference algorithms.
- PySide6 desktop workbench for local research workflows.
- C++/CUDA native backend for accelerated solver kernels.
- HDF5/NPZ/YAML data layer for portable project files and results.

## Layout

```text
app/
  .venv/                Local Python runtime and dependencies
  src/capp/             Python package
  native/               C++/CUDA extension scaffold
  tests/                Unit and parity tests
  examples/             Example configs
  docs/                 Rebuild-specific technical docs
```

## Notes

This scaffold intentionally does not preserve MATLAB file names. New module names describe product concepts and implementation responsibilities.

To launch the desktop app without a terminal window on Windows, double-click
`launch_workbench.vbs`.

Command-line launch:

```powershell
cd "C:\Users\admin\Desktop\workspace\CA_PP\virtual PBF\app"
.\.venv\Scripts\python.exe launch_workbench.pyw
```

