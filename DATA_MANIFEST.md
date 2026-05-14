# Data Manifest

## Included

| Path | Role |
| --- | --- |
| `data/geometry_examples/` | STL geometry fixtures and calibration artifacts. |
| `data/calibration_samples/` | ROI target masks copied from the MATLAB-era `samples/` folder. |
| `data/machine_map/sp_coordinates.xlsx` | Original sample coordinate workbook. |
| `data/machine_map/sp_coordinates.csv` | Portable CSV copy of the coordinate table. |
| `data/machine_map/result_007.mat` | Existing weight source artifact from MATLAB. |
| `data/legacy_models/Huneed_TI64.mat` | Legacy machine-map preset used by the MATLAB solver. |
| `data/legacy_models/Daegun_Cu_Models.mat` | Legacy machine-map preset candidate. |
| `data/legacy_models/CentralBias.mat` | Legacy bias data kept for inspection. |

## Not Included

The MATLAB source code (`EXECUTION.m`, `functions/`, `OpenLibrary/`) and CT preprocessing scripts were intentionally left in the original folders. The app now treats them as reference material, not runtime code.

Python virtual environments, caches, generated example outputs, Workbench preset libraries,
and native build folders are also omitted.
