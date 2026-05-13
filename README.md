# Virtual PBF

This directory collects the current greenfield desktop rebuild and the non-MATLAB-source data needed for simulation, Model Calibration, and machine-map work.

## Layout

```text
virtual PBF/
  app/                     Python/PySide6 workbench, solver code, tests, docs
  data/
    geometry_examples/      STL fixtures and artifact geometry
    calibration_samples/    ROI target images
    machine_map/            sample coordinates and Model Calibration weight source data
    legacy_models/          MATLAB-era model artifacts kept as migration inputs
```

The original MATLAB prototype under `CA-PP/` is left in place and was not moved.

## Model Inputs

Machine-map generation needs:

- `data/machine_map/sp_coordinates.csv` or `.xlsx`: sample ID to build-plate X/Y coordinates.
- Model Calibration result weights with columns `Sample`, `param1` through `param6`, and optionally `Loss`.
- `data/calibration_samples/*_xSliceROI.tif` and `*_ySliceROI.tif`: ROI targets for fitting those weights.
- `data/geometry_examples/KITECH_ARTIFACTS/Test_artifact_ver.4.stl`: current reference geometry.

`data/machine_map/result_007.mat` is included as the current MATLAB-table source artifact. The portable target format for future runs is CSV or Parquet.
