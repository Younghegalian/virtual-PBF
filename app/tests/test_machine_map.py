import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from capp.calibration.rmc import RmcParameterSet
from capp.domain import (
    MachineBiasMode,
    MachineMapCoordinateMode,
    NeighborhoodModel,
    SolverParameters,
    VoxelGrid,
)
from capp.machine_map import (
    CoordinateNormalizer,
    MachineParameterMap,
    MachineParameterRow,
    SampleCoordinate,
    apply_machine_parameter_map,
    build_machine_parameter_map_from_files,
    generate_machine_parameter_map_from_files,
    read_model_calibration_weights_csv,
    read_sample_coordinates_xlsx,
    save_machine_parameter_map_outputs,
)


def test_coordinate_normalizer_matches_matlab_generator():
    normalizer = CoordinateNormalizer()

    assert normalizer.normalize(-125, -125) == (0.0, 0.0)
    assert normalizer.normalize(0, 0) == (0.5, 0.5)
    assert normalizer.normalize(125, 125) == (1.0, 1.0)


def test_read_sample_coordinates_xlsx_reads_sample_column(tmp_path: Path):
    workbook = tmp_path / "coords.xlsx"
    _write_minimal_coordinates_xlsx(workbook)

    coordinates = read_sample_coordinates_xlsx(workbook)

    assert coordinates == [
        SampleCoordinate("A1", -50.0, 100.0),
        SampleCoordinate("A2", 0.0, 100.0),
        SampleCoordinate("B1", -100.0, 50.0),
    ]


def test_read_sample_coordinates_xlsx_without_expat(tmp_path: Path, monkeypatch):
    workbook = tmp_path / "coords.xlsx"
    _write_minimal_coordinates_xlsx(workbook)

    import capp.machine_map.models as machine_map_models

    def missing_expat(_content):
        raise ImportError("No module named expat; use SimpleXMLTreeBuilder instead")

    monkeypatch.setattr(machine_map_models.ElementTree, "fromstring", missing_expat)

    coordinates = read_sample_coordinates_xlsx(workbook)

    assert coordinates == [
        SampleCoordinate("A1", -50.0, 100.0),
        SampleCoordinate("A2", 0.0, 100.0),
        SampleCoordinate("B1", -100.0, 50.0),
    ]


def test_machine_parameter_map_reproduces_training_points():
    coordinates = [
        SampleCoordinate("A", -125.0, -125.0),
        SampleCoordinate("B", 125.0, -125.0),
        SampleCoordinate("C", -125.0, 125.0),
        SampleCoordinate("D", 125.0, 125.0),
    ]
    rows = [
        MachineParameterRow("A", RmcParameterSet(0.1, 0.2, 0.3, 0.4, 0.01, 0.2)),
        MachineParameterRow("B", RmcParameterSet(0.2, 0.3, 0.4, 0.5, 0.02, 0.3)),
        MachineParameterRow("C", RmcParameterSet(0.3, 0.4, 0.5, 0.6, 0.03, 0.4)),
        MachineParameterRow("D", RmcParameterSet(0.4, 0.5, 0.6, 0.7, 0.04, 0.5)),
    ]

    model = MachineParameterMap.fit(rows, coordinates)

    assert np.allclose(model.evaluate(-125.0, -125.0).as_tuple(), rows[0].parameters.as_tuple())
    assert np.allclose(model.evaluate(125.0, 125.0).as_tuple(), rows[3].parameters.as_tuple())


def test_machine_parameter_map_requires_coordinate_for_each_sample():
    rows = [MachineParameterRow("A", RmcParameterSet(0.1, 0.2, 0.3, 0.4, 0.01, 0.2))]

    with pytest.raises(ValueError, match="Missing coordinates"):
        MachineParameterMap.fit(rows, [])


def test_read_model_calibration_weights_csv_reads_parameter_rows(tmp_path: Path):
    weights = tmp_path / "model_calibration_weights.csv"
    _write_minimal_weights_csv(weights)

    rows = read_model_calibration_weights_csv(weights)

    assert [row.sample for row in rows] == ["A1", "A2", "B1"]
    assert rows[0].parameters.as_tuple() == (0.1, 0.2, 0.3, 0.4, 0.01, 0.2)
    assert rows[0].loss == 1.5


def test_build_machine_parameter_map_from_files_keeps_result_in_memory(tmp_path: Path):
    weights = tmp_path / "model_calibration_weights.csv"
    coordinates = tmp_path / "sp_coordinates.xlsx"
    output_dir = tmp_path / "map"
    _write_minimal_weights_csv(weights)
    _write_minimal_coordinates_xlsx(coordinates)

    result = build_machine_parameter_map_from_files(
        weights_csv=weights,
        coordinates_xlsx=coordinates,
        resolution=9,
    )

    assert result.sample_count == 3
    assert result.grid["NX"].shape == (9, 9)
    assert not output_dir.exists()


def test_generate_machine_parameter_map_from_files_writes_solver_ready_outputs(tmp_path: Path):
    weights = tmp_path / "model_calibration_weights.csv"
    coordinates = tmp_path / "sp_coordinates.xlsx"
    output_dir = tmp_path / "map"
    _write_minimal_weights_csv(weights)
    _write_minimal_coordinates_xlsx(coordinates)

    result = generate_machine_parameter_map_from_files(
        weights_csv=weights,
        coordinates_xlsx=coordinates,
        output_dir=output_dir,
        resolution=9,
    )

    assert result.sample_count == 3
    assert result.map_npz.exists()
    assert result.grid_csv.exists()
    assert result.sample_csv.exists()
    assert result.metadata_json.exists()
    payload = np.load(result.map_npz)
    assert payload["NX"].shape == (9, 9)
    assert payload["sample_parameters"].shape == (3, 6)
    assert "virtual_pbf.machine_parameter_map.v1" in result.metadata_json.read_text(
        encoding="utf-8"
    )


def test_machine_parameter_map_duplicate_preset_names_use_new_folder(tmp_path: Path):
    weights = tmp_path / "model_calibration_weights.csv"
    coordinates = tmp_path / "sp_coordinates.xlsx"
    output_dir = tmp_path / "map"
    _write_minimal_weights_csv(weights)
    _write_minimal_coordinates_xlsx(coordinates)

    first = generate_machine_parameter_map_from_files(
        weights_csv=weights,
        coordinates_xlsx=coordinates,
        output_dir=output_dir,
        resolution=9,
        preset_name="Machine Map",
    )
    second = generate_machine_parameter_map_from_files(
        weights_csv=weights,
        coordinates_xlsx=coordinates,
        output_dir=output_dir,
        resolution=9,
        preset_name="Machine Map",
    )

    assert first.output_dir.name == "Machine_Map"
    assert second.output_dir.name == "Machine_Map_002"
    assert first.preset_name == "Machine Map"
    assert second.preset_name == "Machine Map 002"
    assert second.map_npz.exists()


def test_save_machine_parameter_map_outputs_keeps_configuration_and_inputs(tmp_path: Path):
    weights = tmp_path / "model_calibration_weights.csv"
    coordinates_path = tmp_path / "sp_coordinates.xlsx"
    _write_minimal_weights_csv(weights)
    _write_minimal_coordinates_xlsx(coordinates_path)
    built = build_machine_parameter_map_from_files(
        weights_csv=weights,
        coordinates_xlsx=coordinates_path,
        resolution=9,
        preset_name="Preset A",
    )

    saved = save_machine_parameter_map_outputs(
        output_dir=tmp_path / "map",
        model=built.model,
        grid=built.grid,
        parameters=built.parameters,
        coordinates=built.coordinates,
        resolution=built.resolution,
        preset_name=built.preset_name,
        weights_csv=weights,
        coordinates_xlsx=coordinates_path,
        run_configuration={"format": "test.config", "preset": "Preset A"},
    )

    assert saved.configuration_json is not None
    assert saved.configuration_json.exists()
    assert '"format": "test.config"' in saved.configuration_json.read_text(encoding="utf-8")
    assert (saved.output_dir / "inputs" / weights.name).exists()
    assert (saved.output_dir / "inputs" / coordinates_path.name).exists()


def test_save_machine_parameter_map_outputs_can_use_exact_preset_folder(tmp_path: Path):
    weights = tmp_path / "model_calibration_weights.csv"
    coordinates_path = tmp_path / "sp_coordinates.xlsx"
    _write_minimal_weights_csv(weights)
    _write_minimal_coordinates_xlsx(coordinates_path)
    built = build_machine_parameter_map_from_files(
        weights_csv=weights,
        coordinates_xlsx=coordinates_path,
        resolution=9,
        preset_name="Preset A",
    )
    library_root = tmp_path / "library"
    preset_folder = library_root / "Preset_A" / "map"

    saved = save_machine_parameter_map_outputs(
        output_dir=library_root,
        preset_folder=preset_folder,
        model=built.model,
        grid=built.grid,
        parameters=built.parameters,
        coordinates=built.coordinates,
        resolution=built.resolution,
        preset_name=built.preset_name,
        weights_csv=weights,
        coordinates_xlsx=coordinates_path,
    )

    assert saved.output_dir == preset_folder
    assert saved.map_npz == preset_folder / "machine_parameter_map.npz"
    assert not (library_root / "machine_presets").exists()


def test_apply_machine_parameter_map_samples_grid_to_solver_parameters(tmp_path: Path):
    map_path = tmp_path / "machine_parameter_map.npz"
    base = np.asarray([[0.0, 1.0], [10.0, 11.0]], dtype=np.float32)
    np.savez_compressed(
        map_path,
        parameter_names=np.asarray(["NX", "PX", "NY", "PY", "EPS", "IDP"]),
        NX=base,
        PX=base + 100.0,
        NY=base + 200.0,
        PY=base + 300.0,
        EPS=base + 400.0,
        IDP=base + 500.0,
    )
    map_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "normalizer": {
                    "x_offset": 0.0,
                    "y_offset": 0.0,
                    "x_range": 1.0,
                    "y_range": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )
    grid = VoxelGrid(
        np.ones((2, 2, 1), dtype=bool),
        spacing=1.0,
        origin=(0.0, 0.0, 0.0),
    )
    parameters = SolverParameters(
        neighborhood=NeighborhoodModel.DIRECTIONAL_VON_NEUMANN,
        machine_bias=MachineBiasMode.PRESET,
        machine_map_path=map_path,
        machine_map_coordinate_mode=MachineMapCoordinateMode.EXPLICIT_BOUNDS,
        machine_map_bounds=(0.0, 1.0, 0.0, 1.0),
    )

    mapped = apply_machine_parameter_map(parameters, grid)

    expected = np.asarray([[0.0, 10.0], [1.0, 11.0]], dtype=np.float32)
    assert mapped.machine_bias is MachineBiasMode.NONE
    assert mapped.spatial_current_coefficients is not None
    assert np.allclose(mapped.spatial_current_coefficients[0], expected)
    assert np.allclose(mapped.spatial_current_coefficients[1], expected + 100.0)
    assert np.allclose(mapped.spatial_min_bias, expected + 400.0)
    assert np.allclose(mapped.spatial_initial_deviation, expected + 500.0)


def test_apply_machine_parameter_map_centers_on_part_not_support(tmp_path: Path):
    map_path = tmp_path / "machine_parameter_map.npz"
    x_values = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    np.savez_compressed(
        map_path,
        parameter_names=np.asarray(["NX", "PX", "NY", "PY", "EPS", "IDP"]),
        NX=x_values,
        PX=x_values,
        NY=x_values,
        PY=x_values,
        EPS=x_values,
        IDP=x_values,
    )
    map_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "normalizer": {
                    "x_offset": 1.0,
                    "y_offset": 1.0,
                    "x_range": 2.0,
                    "y_range": 2.0,
                }
            }
        ),
        encoding="utf-8",
    )
    data = np.zeros((5, 1, 1), dtype=bool)
    data[[0, 3, 4], 0, 0] = True
    support_mask = np.zeros_like(data)
    support_mask[0, 0, 0] = True
    grid = VoxelGrid(data, spacing=1.0, origin=(0.0, 0.0, 0.0), support_mask=support_mask)
    parameters = SolverParameters(
        neighborhood=NeighborhoodModel.DIRECTIONAL_VON_NEUMANN,
        machine_bias=MachineBiasMode.PRESET,
        machine_map_path=map_path,
        machine_map_coordinate_mode=MachineMapCoordinateMode.PART_CENTER,
        machine_map_position=(0.0, 0.0),
    )

    mapped = apply_machine_parameter_map(parameters, grid)

    assert mapped.spatial_current_coefficients is not None
    assert np.isclose(mapped.spatial_current_coefficients[0][3, 0], 0.25)
    assert np.isclose(mapped.spatial_current_coefficients[0][4, 0], 0.75)


def _write_minimal_weights_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "Sample,param1,param2,param3,param4,param5,param6,Loss",
                "A1,0.1,0.2,0.3,0.4,0.01,0.2,1.5",
                "A2,0.2,0.3,0.4,0.5,0.02,0.3,1.2",
                "B1,0.3,0.4,0.5,0.6,0.03,0.4,0.9",
            ]
        ),
        encoding="utf-8",
    )


def _write_minimal_coordinates_xlsx(path: Path) -> None:
    shared_strings = ["A1", "X", "Y", "A2", "B1"]
    shared_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="5" '
        'uniqueCount="5">'
        + "".join(f"<si><t>{value}</t></si>" for value in shared_strings)
        + "</sst>"
    )
    sheet_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1"><c r="B1" t="s"><v>1</v></c><c r="C1" t="s"><v>2</v></c></row>
    <row r="2"><c r="A2" t="s"><v>0</v></c><c r="B2"><v>-50</v></c><c r="C2"><v>100</v></c></row>
    <row r="3"><c r="A3" t="s"><v>3</v></c><c r="B3"><v>0</v></c><c r="C3"><v>100</v></c></row>
    <row r="4"><c r="A4" t="s"><v>4</v></c><c r="B4"><v>-100</v></c><c r="C4"><v>50</v></c></row>
  </sheetData>
</worksheet>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/sharedStrings.xml", shared_xml)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)
