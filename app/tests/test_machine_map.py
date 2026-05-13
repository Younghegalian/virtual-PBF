import zipfile
from pathlib import Path

import numpy as np
import pytest

from capp.calibration.rmc import RmcParameterSet
from capp.machine_map import (
    build_machine_parameter_map_from_files,
    CoordinateNormalizer,
    MachineParameterMap,
    MachineParameterRow,
    SampleCoordinate,
    generate_machine_parameter_map_from_files,
    read_model_calibration_weights_csv,
    read_sample_coordinates_xlsx,
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
