from __future__ import annotations

import csv
from html import unescape
import json
import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from time import perf_counter
from xml.etree import ElementTree

import numpy as np
from scipy.interpolate import RBFInterpolator

from capp.calibration.rmc import RmcParameterSet
from capp.domain import SolverParameters, VoxelGrid

PARAMETER_NAMES = ("NX", "PX", "NY", "PY", "EPS", "IDP")
_XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
ProgressCallback = Callable[[int, str], None]


def apply_machine_parameter_map(
    parameters: SolverParameters,
    _grid: VoxelGrid,
) -> SolverParameters:
    """Return mapped solver parameters when a preset is available.

    The restored strict snapshot keeps this hook so virtual printing can run
    even when no machine-map preset is selected. Full spatial map application
    can be reattached on top of this stable launch path.
    """

    return parameters


@dataclass(frozen=True)
class SampleCoordinate:
    sample: str
    x: float
    y: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample", self.sample.strip().strip('"'))


@dataclass(frozen=True)
class MachineParameterRow:
    sample: str
    parameters: RmcParameterSet
    loss: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample", self.sample.strip().strip('"'))


@dataclass(frozen=True)
class MachineMapExportResult:
    output_dir: Path
    map_npz: Path
    grid_csv: Path
    sample_csv: Path
    metadata_json: Path
    configuration_json: Path | None
    resolution: int
    sample_count: int
    elapsed_seconds: float
    preset_name: str = "Machine Map"
    voxel_spacing: float | None = None


@dataclass(frozen=True)
class MachineMapGenerationResult:
    model: MachineParameterMap
    grid: dict[str, np.ndarray]
    parameters: list[MachineParameterRow]
    coordinates: list[SampleCoordinate]
    resolution: int
    sample_count: int
    elapsed_seconds: float
    preset_name: str = "Machine Map"
    voxel_spacing: float | None = None
    weights_csv: Path | None = None
    coordinates_xlsx: Path | None = None


@dataclass(frozen=True)
class MachineParameterMapMetadata:
    path: Path
    preset_name: str
    voxel_spacing: float | None
    resolution: int
    sample_count: int
    parameter_order: tuple[str, ...]


@dataclass(frozen=True)
class CoordinateNormalizer:
    x_offset: float = 125.0
    y_offset: float = 125.0
    x_range: float = 250.0
    y_range: float = 250.0

    def normalize(self, x: float, y: float) -> tuple[float, float]:
        if self.x_range == 0.0 or self.y_range == 0.0:
            raise ValueError("Coordinate ranges must be non-zero.")
        return (
            (float(x) + self.x_offset) / self.x_range,
            (float(y) + self.y_offset) / self.y_range,
        )


class MachineParameterMap:
    def __init__(
        self,
        models: dict[str, RBFInterpolator],
        normalizer: CoordinateNormalizer,
    ) -> None:
        self._models = dict(models)
        self.normalizer = normalizer

    @classmethod
    def fit(
        cls,
        parameters: list[MachineParameterRow],
        coordinates: list[SampleCoordinate],
        *,
        normalizer: CoordinateNormalizer | None = None,
    ) -> MachineParameterMap:
        normalizer = normalizer or CoordinateNormalizer()
        coordinate_by_sample = {coord.sample: coord for coord in coordinates}

        points: list[tuple[float, float]] = []
        values: list[tuple[float, float, float, float, float, float]] = []
        missing: list[str] = []
        for row in parameters:
            coordinate = coordinate_by_sample.get(row.sample)
            if coordinate is None:
                missing.append(row.sample)
                continue
            points.append(normalizer.normalize(coordinate.x, coordinate.y))
            values.append(row.parameters.as_tuple())

        if missing:
            raise ValueError(f"Missing coordinates for samples: {', '.join(sorted(missing))}")
        if len(points) < 3:
            raise ValueError("At least three calibrated samples are required to fit a map.")

        point_array = np.asarray(points, dtype=np.float64)
        value_array = np.asarray(values, dtype=np.float64)
        models = _fit_rbf_models(point_array, value_array)
        return cls(models=models, normalizer=normalizer)

    def evaluate(self, x: float, y: float) -> RmcParameterSet:
        return self.evaluate_normalized(*self.normalizer.normalize(x, y))

    def evaluate_normalized(self, x: float, y: float) -> RmcParameterSet:
        point = np.asarray([[float(x), float(y)]], dtype=np.float64)
        values = tuple(float(self._models[name](point)[0]) for name in PARAMETER_NAMES)
        return RmcParameterSet.from_sequence(values)

    def to_grid(self, resolution: int = 200) -> dict[str, np.ndarray]:
        axis = np.linspace(0.0, 1.0, int(resolution))
        x_grid, y_grid = np.meshgrid(axis, axis)
        points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
        grid = {"X": x_grid, "Y": y_grid}
        for name in PARAMETER_NAMES:
            grid[name] = self._models[name](points).reshape(x_grid.shape)
        return grid


def generate_machine_parameter_map_from_files(
    *,
    weights_csv: str | Path,
    coordinates_xlsx: str | Path,
    output_dir: str | Path,
    resolution: int = 200,
    preset_name: str = "Machine Map",
    voxel_spacing: float | None = None,
    normalizer: CoordinateNormalizer | None = None,
    progress_callback: ProgressCallback | None = None,
) -> MachineMapExportResult:
    result = build_machine_parameter_map_from_files(
        weights_csv=weights_csv,
        coordinates_xlsx=coordinates_xlsx,
        resolution=resolution,
        preset_name=preset_name,
        voxel_spacing=voxel_spacing,
        normalizer=normalizer,
        progress_callback=progress_callback,
        emit_complete=False,
    )
    if progress_callback is not None:
        progress_callback(75, "Writing machine parameter map")
    saved = save_machine_parameter_map_outputs(
        output_dir=output_dir,
        model=result.model,
        grid=result.grid,
        parameters=result.parameters,
        coordinates=result.coordinates,
        resolution=result.resolution,
        preset_name=result.preset_name,
        weights_csv=result.weights_csv,
        coordinates_xlsx=result.coordinates_xlsx,
        elapsed_seconds=result.elapsed_seconds,
        voxel_spacing=result.voxel_spacing,
    )
    if progress_callback is not None:
        progress_callback(100, f"Machine parameter map saved: {saved.map_npz.name}")
    return saved


def build_machine_parameter_map_from_files(
    *,
    weights_csv: str | Path,
    coordinates_xlsx: str | Path,
    resolution: int = 200,
    preset_name: str = "Machine Map",
    voxel_spacing: float | None = None,
    normalizer: CoordinateNormalizer | None = None,
    progress_callback: ProgressCallback | None = None,
    emit_complete: bool = True,
) -> MachineMapGenerationResult:
    started = perf_counter()
    if progress_callback is not None:
        progress_callback(0, "Reading calibration weights")
    rows = read_model_calibration_weights_csv(weights_csv)

    if progress_callback is not None:
        progress_callback(15, "Reading sample coordinates")
    coordinates = read_sample_coordinates_xlsx(coordinates_xlsx)

    if progress_callback is not None:
        progress_callback(30, "Fitting thin-plate parameter map")
    return build_machine_parameter_map(
        parameters=rows,
        coordinates=coordinates,
        resolution=resolution,
        preset_name=preset_name,
        voxel_spacing=voxel_spacing,
        normalizer=normalizer,
        weights_csv=weights_csv,
        coordinates_xlsx=coordinates_xlsx,
        started_at=started,
        progress_callback=progress_callback,
        emit_complete=emit_complete,
    )


def build_machine_parameter_map(
    *,
    parameters: list[MachineParameterRow],
    coordinates: list[SampleCoordinate],
    resolution: int = 200,
    preset_name: str = "Machine Map",
    voxel_spacing: float | None = None,
    normalizer: CoordinateNormalizer | None = None,
    weights_csv: str | Path | None = None,
    coordinates_xlsx: str | Path | None = None,
    started_at: float | None = None,
    progress_callback: ProgressCallback | None = None,
    emit_complete: bool = True,
) -> MachineMapGenerationResult:
    started = perf_counter() if started_at is None else started_at
    if progress_callback is not None and started_at is None:
        progress_callback(30, "Fitting thin-plate parameter map")
    model = MachineParameterMap.fit(parameters, coordinates, normalizer=normalizer)

    if progress_callback is not None:
        progress_callback(55, "Evaluating machine parameter grid")
    grid = model.to_grid(resolution=resolution)

    elapsed = perf_counter() - started
    if emit_complete and progress_callback is not None:
        progress_callback(100, "Machine parameter map ready in memory")
    return MachineMapGenerationResult(
        model=model,
        grid=grid,
        parameters=list(parameters),
        coordinates=coordinates,
        resolution=int(resolution),
        sample_count=len(parameters),
        elapsed_seconds=elapsed,
        preset_name=preset_name.strip() or "Machine Map",
        voxel_spacing=voxel_spacing,
        weights_csv=Path(weights_csv) if weights_csv is not None else None,
        coordinates_xlsx=Path(coordinates_xlsx) if coordinates_xlsx is not None else None,
    )


def machine_parameter_rows_from_calibration_result(result) -> list[MachineParameterRow]:
    return [
        MachineParameterRow(
            sample.sample,
            RmcParameterSet.from_sequence(sample.best.parameters.as_tuple()),
            loss=sample.best.loss.total,
        )
        for sample in result.samples
    ]


def read_model_calibration_weights_csv(path: str | Path) -> list[MachineParameterRow]:
    weights_path = Path(path)
    rows: list[MachineParameterRow] = []
    with weights_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            sample = _csv_value(raw, "Sample")
            if not sample:
                continue
            values = [_csv_float(raw, f"param{index}") for index in range(1, 7)]
            loss_text = _csv_value(raw, "Loss")
            rows.append(
                MachineParameterRow(
                    sample=sample,
                    parameters=RmcParameterSet.from_sequence(values),
                    loss=float(loss_text) if loss_text else None,
                )
            )
    if not rows:
        raise ValueError(f"No model calibration rows found in {weights_path}")
    return rows


def _fit_rbf_models(
    points: np.ndarray,
    values: np.ndarray,
) -> dict[str, RBFInterpolator]:
    return {
        name: RBFInterpolator(
            points,
            values[:, index],
            kernel="thin_plate_spline",
            degree=1,
            smoothing=0.0,
        )
        for index, name in enumerate(PARAMETER_NAMES)
    }


def save_machine_parameter_map_outputs(
    *,
    output_dir: str | Path,
    model: MachineParameterMap,
    grid: dict[str, np.ndarray],
    parameters: list[MachineParameterRow],
    coordinates: list[SampleCoordinate],
    resolution: int,
    preset_name: str = "Machine Map",
    weights_csv: str | Path | None = None,
    coordinates_xlsx: str | Path | None = None,
    elapsed_seconds: float = 0.0,
    voxel_spacing: float | None = None,
    run_configuration: dict[str, object] | None = None,
) -> MachineMapExportResult:
    effective_preset_name, folder = _unique_preset_folder(Path(output_dir), preset_name)
    folder.mkdir(parents=True, exist_ok=True)
    map_npz = folder / "machine_parameter_map.npz"
    grid_csv = folder / "machine_parameter_grid.csv"
    sample_csv = folder / "machine_parameter_samples.csv"
    metadata_json = folder / "machine_parameter_map.json"
    configuration_json = folder / "run_configuration.json"
    copied_inputs = _copy_machine_map_inputs(
        folder,
        weights_csv=weights_csv,
        coordinates_xlsx=coordinates_xlsx,
    )

    coordinate_by_sample = {coord.sample: coord for coord in coordinates}
    sample_rows = _joined_sample_rows(parameters, coordinate_by_sample, model.normalizer)
    np.savez_compressed(
        map_npz,
        preset_name=np.asarray([effective_preset_name]),
        parameter_names=np.asarray(PARAMETER_NAMES),
        x_normalized=grid["X"],
        y_normalized=grid["Y"],
        NX=grid["NX"],
        PX=grid["PX"],
        NY=grid["NY"],
        PY=grid["PY"],
        EPS=grid["EPS"],
        IDP=grid["IDP"],
        sample_names=np.asarray([row["Sample"] for row in sample_rows]),
        sample_x=np.asarray([row["X"] for row in sample_rows], dtype=np.float64),
        sample_y=np.asarray([row["Y"] for row in sample_rows], dtype=np.float64),
        sample_x_normalized=np.asarray([row["Xn"] for row in sample_rows], dtype=np.float64),
        sample_y_normalized=np.asarray([row["Yn"] for row in sample_rows], dtype=np.float64),
        sample_parameters=np.asarray(
            [[row[name] for name in PARAMETER_NAMES] for row in sample_rows],
            dtype=np.float64,
        ),
        sample_loss=np.asarray([row["Loss"] for row in sample_rows], dtype=np.float64),
        voxel_spacing=np.asarray(
            [np.nan if voxel_spacing is None else float(voxel_spacing)],
            dtype=np.float64,
        ),
    )
    _write_grid_csv(grid_csv, grid)
    _write_sample_csv(sample_csv, sample_rows)
    _write_metadata_json(
        metadata_json,
        grid=grid,
        sample_rows=sample_rows,
        normalizer=model.normalizer,
        resolution=resolution,
        preset_name=effective_preset_name,
        weights_csv=weights_csv,
        coordinates_xlsx=coordinates_xlsx,
        copied_inputs=copied_inputs,
        elapsed_seconds=elapsed_seconds,
        voxel_spacing=voxel_spacing,
    )
    if run_configuration is not None:
        payload = dict(run_configuration)
        payload["saved_preset_name"] = effective_preset_name
        payload["preset_folder"] = str(folder)
        configuration_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        configuration_json = None
    return MachineMapExportResult(
        output_dir=folder,
        map_npz=map_npz,
        grid_csv=grid_csv,
        sample_csv=sample_csv,
        metadata_json=metadata_json,
        configuration_json=configuration_json,
        resolution=int(resolution),
        sample_count=len(sample_rows),
        elapsed_seconds=float(elapsed_seconds),
        preset_name=effective_preset_name,
        voxel_spacing=voxel_spacing,
    )


def read_machine_parameter_map_metadata(path: str | Path) -> MachineParameterMapMetadata:
    map_path = Path(path)
    metadata_path = map_path.with_suffix(".json")
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        return MachineParameterMapMetadata(
            path=map_path,
            preset_name=str(payload.get("preset_name") or "Machine Map"),
            voxel_spacing=payload.get("voxel_spacing"),
            resolution=int(payload.get("resolution") or 0),
            sample_count=int(payload.get("sample_count") or 0),
            parameter_order=tuple(payload.get("parameter_order") or PARAMETER_NAMES),
        )

    with np.load(map_path, allow_pickle=False) as data:
        preset = data["preset_name"][0] if "preset_name" in data.files else "Machine Map"
        spacing = None
        if "voxel_spacing" in data.files:
            raw_spacing = float(data["voxel_spacing"][0])
            if np.isfinite(raw_spacing):
                spacing = raw_spacing
        sample_count = int(data["sample_names"].shape[0]) if "sample_names" in data.files else 0
        resolution = int(data["NX"].shape[0]) if "NX" in data.files else 0
        order = tuple(str(name) for name in data["parameter_names"]) if "parameter_names" in data.files else PARAMETER_NAMES
    return MachineParameterMapMetadata(
        path=map_path,
        preset_name=str(preset),
        voxel_spacing=spacing,
        resolution=resolution,
        sample_count=sample_count,
        parameter_order=order,
    )


def read_sample_coordinates_xlsx(path: str | Path) -> list[SampleCoordinate]:
    workbook = Path(path)
    with zipfile.ZipFile(workbook) as archive:
        shared_strings = _read_shared_strings(archive)
        sheet_xml = archive.read("xl/worksheets/sheet1.xml")
        try:
            root = ElementTree.fromstring(sheet_xml)
        except ImportError:
            return _read_sample_coordinates_xlsx_text(sheet_xml, shared_strings)

    rows: list[dict[str, str]] = []
    for row in root.findall(".//a:sheetData/a:row", _XLSX_NS):
        cells: dict[str, str] = {}
        for cell in row.findall("a:c", _XLSX_NS):
            reference = cell.attrib.get("r", "")
            column = "".join(ch for ch in reference if ch.isalpha())
            value_node = cell.find("a:v", _XLSX_NS)
            if value_node is None or value_node.text is None:
                continue
            value = value_node.text
            if cell.attrib.get("t") == "s":
                value = shared_strings[int(value)]
            cells[column] = value
        rows.append(cells)

    coordinates: list[SampleCoordinate] = []
    for cells in rows[1:]:
        sample = cells.get("A")
        x_value = cells.get("B")
        y_value = cells.get("C")
        if sample is None or x_value is None or y_value is None:
            continue
        coordinates.append(SampleCoordinate(sample=sample, x=float(x_value), y=float(y_value)))
    return coordinates


def _joined_sample_rows(
    parameters: list[MachineParameterRow],
    coordinate_by_sample: dict[str, SampleCoordinate],
    normalizer: CoordinateNormalizer,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for row in parameters:
        coordinate = coordinate_by_sample[row.sample]
        x_normalized, y_normalized = normalizer.normalize(coordinate.x, coordinate.y)
        values = row.parameters.as_tuple()
        rows.append(
            {
                "Sample": row.sample,
                "X": coordinate.x,
                "Y": coordinate.y,
                "Xn": x_normalized,
                "Yn": y_normalized,
                **{name: values[index] for index, name in enumerate(PARAMETER_NAMES)},
                "Loss": float("nan") if row.loss is None else row.loss,
            }
        )
    return rows


def _write_grid_csv(path: Path, grid: dict[str, np.ndarray]) -> None:
    x_values = grid["X"].ravel()
    y_values = grid["Y"].ravel()
    parameter_values = [grid[name].ravel() for name in PARAMETER_NAMES]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_normalized", "y_normalized", *PARAMETER_NAMES])
        for index in range(x_values.size):
            writer.writerow(
                [
                    float(x_values[index]),
                    float(y_values[index]),
                    *[float(values[index]) for values in parameter_values],
                ]
            )


def _write_sample_csv(path: Path, sample_rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Sample", "X", "Y", "Xn", "Yn", *PARAMETER_NAMES, "Loss"],
        )
        writer.writeheader()
        writer.writerows(sample_rows)


def _write_metadata_json(
    path: Path,
    *,
    grid: dict[str, np.ndarray],
    sample_rows: list[dict[str, float | str]],
    normalizer: CoordinateNormalizer,
    resolution: int,
    preset_name: str,
    weights_csv: str | Path | None,
    coordinates_xlsx: str | Path | None,
    copied_inputs: dict[str, str],
    elapsed_seconds: float,
    voxel_spacing: float | None,
) -> None:
    value_ranges = {
        name: {
            "min": float(np.nanmin(grid[name])),
            "max": float(np.nanmax(grid[name])),
        }
        for name in PARAMETER_NAMES
    }
    payload = {
        "format": "virtual_pbf.machine_parameter_map.v1",
        "preset_name": preset_name.strip() or "Machine Map",
        "parameter_order": list(PARAMETER_NAMES),
        "interpolator": {
            "type": "scipy.interpolate.RBFInterpolator",
            "kernel": "thin_plate_spline",
            "degree": 1,
            "smoothing": 0.0,
        },
        "normalizer": {
            "x_offset": normalizer.x_offset,
            "y_offset": normalizer.y_offset,
            "x_range": normalizer.x_range,
            "y_range": normalizer.y_range,
        },
        "voxel_spacing": None if voxel_spacing is None else float(voxel_spacing),
        "resolution": int(resolution),
        "sample_count": len(sample_rows),
        "value_ranges": value_ranges,
        "sources": {
            "weights_csv": str(weights_csv) if weights_csv is not None else None,
            "coordinates_xlsx": str(coordinates_xlsx) if coordinates_xlsx is not None else None,
            "copied_inputs": copied_inputs,
        },
        "outputs": {
            "grid_npz": "machine_parameter_map.npz",
            "grid_csv": "machine_parameter_grid.csv",
            "samples_csv": "machine_parameter_samples.csv",
            "preset_folder": str(path.parent),
        },
        "elapsed_seconds": float(elapsed_seconds),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _preset_directory_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", (name or "Machine Map").strip())
    safe = safe.strip("._-")
    return safe or "Machine_Map"


def _unique_preset_folder(output_dir: Path, preset_name: str) -> tuple[str, Path]:
    base_name = (preset_name or "Machine Map").strip() or "Machine Map"
    base_folder_name = _preset_directory_name(base_name)
    parent = output_dir / "machine_presets"
    folder = parent / base_folder_name
    if not folder.exists() or (folder.is_dir() and not any(folder.iterdir())):
        return base_name, folder

    for index in range(2, 1000):
        suffix = f"{index:03d}"
        candidate = parent / f"{base_folder_name}_{suffix}"
        if not candidate.exists() or (candidate.is_dir() and not any(candidate.iterdir())):
            return f"{base_name} {suffix}", candidate
    raise FileExistsError(f"Too many machine preset folders named like '{base_folder_name}'.")


def _copy_machine_map_inputs(
    folder: Path,
    *,
    weights_csv: str | Path | None,
    coordinates_xlsx: str | Path | None,
) -> dict[str, str]:
    inputs_dir = folder / "inputs"
    copied: dict[str, str] = {}
    for key, source in (
        ("weights_csv", weights_csv),
        ("coordinates_xlsx", coordinates_xlsx),
    ):
        if source is None:
            continue
        source_path = Path(source)
        if not source_path.exists() or not source_path.is_file():
            continue
        inputs_dir.mkdir(parents=True, exist_ok=True)
        target = inputs_dir / source_path.name
        if source_path.resolve() != target.resolve():
            copy2(source_path, target)
        copied[key] = str(target)
    return copied


def _csv_value(row: dict[str, str], key: str) -> str:
    value = row.get(key) or row.get(key.lower()) or row.get(key.upper()) or ""
    return value.strip().strip('"')


def _csv_float(row: dict[str, str], key: str) -> float:
    value = _csv_value(row, key)
    if not value:
        raise ValueError(f"Missing required column '{key}' in model calibration weights.")
    return float(value)


def _read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        shared_xml = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    try:
        root = ElementTree.fromstring(shared_xml)
    except ImportError:
        text = shared_xml.decode("utf-8", errors="replace")
        return [
            unescape("".join(re.findall(r"<a?:?t[^>]*>(.*?)</a?:?t>", item, flags=re.S)))
            for item in re.findall(r"<a?:?si[^>]*>(.*?)</a?:?si>", text, flags=re.S)
        ]
    return [
        "".join(text.text or "" for text in item.findall(".//a:t", _XLSX_NS))
        for item in root.findall("a:si", _XLSX_NS)
    ]


def _read_sample_coordinates_xlsx_text(
    sheet_xml: bytes,
    shared_strings: list[str],
) -> list[SampleCoordinate]:
    text = sheet_xml.decode("utf-8", errors="replace")
    rows: list[dict[str, str]] = []
    for row_text in re.findall(r"<a?:?row\b[^>]*>(.*?)</a?:?row>", text, flags=re.S):
        cells: dict[str, str] = {}
        for cell_match in re.finditer(r"<a?:?c\b([^>]*)>(.*?)</a?:?c>", row_text, flags=re.S):
            attrs, cell_text = cell_match.groups()
            reference_match = re.search(r'\br="([A-Z]+)\d+"', attrs)
            value_match = re.search(r"<a?:?v[^>]*>(.*?)</a?:?v>", cell_text, flags=re.S)
            if reference_match is None or value_match is None:
                continue
            column = reference_match.group(1)
            value = unescape(value_match.group(1).strip())
            if re.search(r'\bt="s"', attrs):
                value = shared_strings[int(value)]
            cells[column] = value
        rows.append(cells)

    coordinates: list[SampleCoordinate] = []
    for cells in rows[1:]:
        sample = cells.get("A")
        x_value = cells.get("B")
        y_value = cells.get("C")
        if sample is None or x_value is None or y_value is None:
            continue
        coordinates.append(SampleCoordinate(sample=sample, x=float(x_value), y=float(y_value)))
    return coordinates
