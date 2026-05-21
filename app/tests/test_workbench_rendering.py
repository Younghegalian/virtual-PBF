from pathlib import Path

import numpy as np

from capp.workbench.app import (
    WorkbenchMainWindow,
    _default_machine_preset_library_root,
    _legacy_model_calibration_root,
    _machine_map_preset_output_dir,
    _machine_preset_folder,
    _model_calibration_preset_output_dir,
    _roi_outline_rgb,
    _roi_overlay_rgb,
    _workbench_colormap,
)
from capp.workbench.preview import PreviewPane


class _ButtonProbe:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)


class _LineEditProbe:
    def __init__(self):
        self.text = ""

    def setText(self, text):
        self.text = str(text)


def test_roi_overlay_rgb_preserves_array_orientation():
    target = np.zeros((4, 4), dtype=bool)
    simulated = np.zeros((4, 4), dtype=bool)
    target[0, 0] = True

    rgb = _roi_overlay_rgb(target, simulated)

    assert tuple(rgb[0, 0]) == (224, 132, 58)
    assert tuple(rgb[-1, 0]) == (248, 248, 248)


def test_roi_outline_rgb_marks_target_and_simulated_edges():
    target = np.zeros((6, 6), dtype=bool)
    simulated = np.zeros((6, 6), dtype=bool)
    target[1:4, 1:4] = True
    simulated[2:5, 2:5] = True

    rgb = _roi_outline_rgb(target, simulated)

    assert tuple(rgb[1, 1]) == (224, 132, 58)
    assert tuple(rgb[4, 4]) == (56, 116, 196)
    assert tuple(rgb[2, 2]) == (62, 153, 101)


def test_roi_outline_rgb_can_use_original_background():
    target = np.zeros((4, 4), dtype=bool)
    simulated = np.zeros((4, 4), dtype=bool)
    original = np.arange(16, dtype=np.uint8).reshape(4, 4)

    rgb = _roi_outline_rgb(target, simulated, original)

    assert rgb.shape == (4, 4, 3)
    assert tuple(rgb[0, 0]) != tuple(rgb[-1, -1])


def test_workbench_colormap_has_continuous_parameter_colors():
    values = np.asarray([[0.0, 0.5, 1.0]], dtype=np.float64)

    rgb = _workbench_colormap(values)

    assert rgb.shape == (1, 3, 3)
    assert tuple(rgb[0, 0]) != tuple(rgb[0, 1])
    assert tuple(rgb[0, 1]) != tuple(rgb[0, 2])


def test_model_calibration_output_dir_uses_preset_folder(tmp_path):
    output_dir = _model_calibration_preset_output_dir(tmp_path, "Machine Map")

    assert _default_machine_preset_library_root() == (
        Path("workbench_library") / "machine_presets"
    )
    assert _legacy_model_calibration_root() == (
        Path("examples") / "outputs" / "model_calibration"
    )
    assert _machine_preset_folder(tmp_path, "Machine Map") == tmp_path / "Machine_Map"
    assert output_dir == tmp_path / "Machine_Map" / "calibration"
    assert _machine_map_preset_output_dir(tmp_path, "Machine Map") == (
        tmp_path / "Machine_Map" / "map"
    )
    assert _model_calibration_preset_output_dir(tmp_path, "Preset A/Trial") == (
        tmp_path / "Preset_A_Trial" / "calibration"
    )


def test_geometry_deviation_button_updates_after_result_sync(tmp_path):
    source_geometry = tmp_path / "part.stl"
    view = WorkbenchMainWindow.__new__(WorkbenchMainWindow)
    view._busy = False
    view._loaded_result = {"source_geometry": source_geometry}
    view._deviation_button = _ButtonProbe()
    view._result_display_preview_button = _ButtonProbe()
    view._deviation_stl_path = _LineEditProbe()

    view._sync_deviation_stl_from_result()

    assert view._deviation_stl_path.text == str(source_geometry)
    assert view._deviation_button.enabled is True
    assert view._result_display_preview_button.enabled is True

    view._busy = True
    view._update_result_action_state()

    assert view._deviation_button.enabled is False
    assert view._result_display_preview_button.enabled is False


def test_loaded_result_change_advances_deviation_revision():
    view = WorkbenchMainWindow.__new__(WorkbenchMainWindow)
    view._result_revision = 4
    view._deviation_summary = _LineEditProbe()

    view._mark_loaded_result_changed()

    assert view._result_revision == 5
    assert view._deviation_summary.text == "Ready"


def test_stale_geometry_deviation_finish_is_discarded():
    view = WorkbenchMainWindow.__new__(WorkbenchMainWindow)
    view._result_revision = 2
    view._geometry_deviation_worker = object()
    logs = []
    busy_states = []
    view._append_log = logs.append
    view._set_busy = lambda busy, message: busy_states.append((busy, message))

    view._geometry_deviation_finished(object(), 1)

    assert view._geometry_deviation_worker is None
    assert logs == ["Discarded stale geometry deviation for a previous result."]
    assert busy_states == [(False, "Stale geometry deviation discarded")]


def test_voxel_preview_reports_failures_and_can_raise():
    pane = PreviewPane.__new__(PreviewPane)
    pane._status = _LineEditProbe()
    pane._load_pyvista = lambda: (_ for _ in ()).throw(RuntimeError("no renderer"))
    volume = np.ones((2, 2, 2), dtype=bool)

    assert pane._render_voxels(volume, 1.0, (0.0, 0.0, 0.0), "Binary") is False
    assert pane._status.text == "Voxel preview failed: no renderer"

    try:
        pane._render_voxels(
            volume,
            1.0,
            (0.0, 0.0, 0.0),
            "Binary",
            raise_errors=True,
        )
    except RuntimeError as exc:
        assert str(exc) == "no renderer"
    else:
        raise AssertionError("Expected preview failure to be raised.")
