# Virtual PBF

Virtual PBF is the rebuilt desktop workbench for powder bed fusion simulation,
ROI-based model calibration, and machine parameter map generation. The old MATLAB
prototype is treated as reference material; this repository contains the Python/PySide6
application, solver code, tests, and the portable non-MATLAB data needed to run it.

## Repository Layout

```text
virtual PBF/
  app/                     Python package, PySide6 workbench, tests, docs
  data/
    geometry_examples/      STL fixtures and calibration artifact geometry
    calibration_samples/    ROI target masks for Model Calibration
    machine_map/            sample coordinate workbook and legacy weight source data
    legacy_models/          MATLAB-era model artifacts kept for migration/reference
  DATA_MANIFEST.md          Description of the included data files
```

## Application Entry Point

The desktop application lives in `app/`.

On Windows, launch without a terminal by double-clicking:

```text
app/launch_workbench.vbs
```

Command-line launch:

```powershell
cd "C:\Users\admin\Desktop\workspace\virtual PBF\app"
.\.venv\Scripts\python.exe launch_workbench.pyw
```

## Preset Library

Reusable calibration and machine-map artifacts are not stored under `examples/outputs`.
They live under one preset library:

```text
app/workbench_library/machine_presets/<preset_name>/
  calibration/
    model_calibration_weights.csv
    run_configuration.json
    research_artifacts/
  map/
    machine_parameter_map.npz
    machine_parameter_map.json
    machine_parameter_grid.csv
    machine_parameter_samples.csv
    run_configuration.json
    inputs/
```

`<preset_name>` is the sanitized folder form of the name entered in the Workbench
`Preset name` field, for example `Machine Map` becomes `Machine_Map`. The simulation
page discovers saved machine presets from:

```text
app/workbench_library/machine_presets/*/map/machine_parameter_map.npz
```

Generated library contents are ignored by git.

## Current Status

- STL loading, voxelization, preview, simulation execution, and NPZ/VTK export are present.
- Basic, C++ vectorized, and GPU CUDA backend selection/validation are exposed in the workbench.
- Model Calibration runs in memory first, then saves requested outputs under the active preset.
- Machine parameter maps can be generated, saved, previewed as contours, and listed by preset name.
- Applying saved machine maps as spatial solver bias is still a pending hook; generated presets are
  available for inspection and future solver integration.

## Development

Run the tests from `app/`:

```powershell
cd "C:\Users\admin\Desktop\workspace\virtual PBF\app"
.\.venv\Scripts\python.exe -m pytest
```

Additional technical notes are in `app/docs/`.
