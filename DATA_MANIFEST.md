# Data Manifest

## Included

| Path | Role |
| --- | --- |
| `data/geometry_examples/` | STL geometry fixtures and calibration artifacts. |
| `data/calibration_samples/` | ROI target masks copied from the MATLAB-era `samples/` folder. |
| `data/machine_map/sp_coordinates.xlsx` | Original sample coordinate workbook. |
| `data/machine_map/sp_coordinates.csv` | Portable CSV copy of the coordinate table. |
| `data/machine_map/result_007.mat` | Existing weight source artifact from MATLAB. |

## Not Included

The MATLAB source code (`EXECUTION.m`, `functions/`, `OpenLibrary/`) and CT preprocessing scripts were intentionally left in the original folders. The app now treats them as reference material, not runtime code.

Legacy MATLAB machine-map presets (`Huneed_TI64.mat`, `Daegun_Cu_Models.mat`, and
`CentralBias.mat`) are intentionally omitted because the current workbench does not
load them at runtime. Reproducible machine-map inputs are kept under
`data/machine_map/` as the portable coordinate CSV/workbook and the compact
`result_007.mat` source artifact.

Python virtual environments, caches, generated example outputs, Workbench preset libraries,
and native build folders are also omitted.
