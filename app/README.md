# Virtual PBF Workbench

Desktop workbench for virtual PBF simulation, ROI model calibration, and build-plate
machine parameter maps.

The app is a Python/PySide6 rebuild. MATLAB-era files are reference inputs only; the
runtime path is Python package code under `src/capp`.

## What Works Now

- Load STL part/support geometry and estimate voxel spacing.
- Voxelize geometry and preview STL, voxel, and result slices.
- Run the layerwise virtual printing simulation with CPU reference, native CPU, or CUDA
  backend selection when the local device stack validates.
- Save simulation outputs as `simulation_result.npz`, `probability.vtk`, and `binary.vtk`.
- Run Model Calibration from ROI sample masks and calibration geometry.
- Save Model Calibration weights, run configuration, sample artifacts, and optional research
  artifacts into the active preset folder.
- Generate and save machine parameter maps from calibration weights plus SP coordinates.
- Preview saved/generated machine maps as contour images.
- List saved machine presets by configured preset name in the simulation page.
- Render a geometry deviation heatmap by comparing an original STL against the result
  iso-surface extracted from a virtual printing volume.

## Important Limitation

Machine preset library management is implemented, but applying a saved machine map as a
spatial solver bias is not complete yet. The simulation pipeline keeps the hook in place, but
`apply_machine_parameter_map` currently returns the incoming solver parameters unchanged and
the solvers still reject `MachineBiasMode.PRESET`. Treat saved machine maps as generated
artifacts ready for inspection and future solver integration, not as fully active simulation
bias input yet.

## Layout

```text
app/
  launch_workbench.vbs       Windows no-console launcher
  launch_workbench.pyw       Python GUI launcher
  pyproject.toml             Package metadata and test configuration
  src/capp/
    workbench/               PySide6 UI and app orchestration
    geometry/                STL stats, mesh loading, voxelization
    solver/                  CPU reference, native, and CUDA solver backends
    simulation/              Simulation pipeline and output saving
    calibration/             ROI losses and Model Calibration
    machine_map/             Machine map generation, metadata, and exports
    io/                      NPZ/VTK output helpers
  tests/                     Unit, workflow, and backend tests
  examples/                  Example configs only; generated example outputs are ignored
  docs/                      Architecture, development, native, and toolchain notes
```

## Launch

Preferred Windows launch:

```text
launch_workbench.vbs
```

Command-line launch:

```powershell
cd "C:\Users\admin\Desktop\workspace\virtual PBF\app"
.\.venv\Scripts\python.exe launch_workbench.pyw
```

CLI help:

```powershell
.\.venv\Scripts\python.exe -m capp.cli --help
```

## Preset Library

The Model Calibration page has a `Preset library` field. Its default is:

```text
workbench_library/machine_presets
```

Every reusable calibration or machine-map save is grouped by the configured preset name:

```text
workbench_library/machine_presets/<preset_name>/
  calibration/
    model_calibration_weights.csv
    run_configuration.json
    <sample>_model_calibration_artifacts.npz
    research_artifacts/
  map/
    machine_parameter_map.npz
    machine_parameter_map.json
    machine_parameter_grid.csv
    machine_parameter_samples.csv
    run_configuration.json
    inputs/
```

The simulation page discovers presets by scanning:

```text
workbench_library/machine_presets/*/map/machine_parameter_map.npz
```

The visible combo box label comes from the saved map metadata `preset_name`. The path field is
only the backing `.npz` location.

## Output Paths

- Reusable calibration and machine-map artifacts: `workbench_library/machine_presets/<preset>/`
- Simulation run outputs: the simulation page `Output dir`, defaulting to
  `examples/outputs/gui_simulation`
- Saved simulation NPZ files include the source STL path when the result was produced inside
  the Workbench, so Result Display can auto-fill the geometry deviation input.
- Generated folders are ignored by git: `workbench_library/`, `outputs/`, and
  `examples/outputs/`

## Data Inputs

Default research inputs are stored outside the app package:

```text
../data/geometry_examples/KITECH_ARTIFACTS/Test_artifact_ver.4.stl
../data/calibration_samples/
../data/machine_map/sp_coordinates.xlsx
```

Machine-map generation expects Model Calibration weights with `Sample`, `param1` through
`param6`, and optional `Loss`, plus the SP coordinate workbook/CSV.

## Tests

Run all tests from `app/`:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The current suite covers CLI behavior, voxelization, ROI/model calibration, machine-map
generation/export, backend validation, solver parity, and workbench rendering helpers.

## Notes For Future Work

- Complete `apply_machine_parameter_map` so saved map presets become active simulation bias.
- Add explicit preset rename/delete/import actions in the Workbench UI.
- Keep CUDA/native parity tests green while moving more solver work out of Python.
- Preserve the preset library shape: one preset folder containing `calibration/` and `map/`.
