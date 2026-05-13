# Rebuild Architecture

This directory implements the architecture governed by `../../docs/PROJECT_TRUTH.md`.

## Layers

```text
Workbench UI
  PySide6 desktop application

Python core
  Configuration, geometry orchestration, solver interfaces, calibration, exports

Native core
  C++/CUDA acceleration for solver and voxel-heavy operations

Data layer
  YAML project config, NPZ/HDF5 result artifacts, VTK/STL exports
```

## First Vertical Slice

The first slice should stay deliberately narrow:

1. STL to voxel grid.
2. Voxel grid to probability/binary simulation result.
3. Result to NPZ and VTK.
4. Basic desktop shell for workflow navigation.

## Parity Strategy

The Python reference solver is the executable specification for the native backend.
MATLAB is the historical reference, but the rebuilt project should create small fixtures with fixed inputs and outputs before CUDA acceleration begins.

