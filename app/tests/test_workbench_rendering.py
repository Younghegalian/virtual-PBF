from pathlib import Path

import numpy as np
import pytest

from capp.domain import VoxelGrid
from capp.workbench.app import (
    WorkbenchMainWindow,
    _default_machine_preset_library_root,
    _legacy_model_calibration_root,
    _load_generated_support_grid_cache,
    _machine_map_preset_output_dir,
    _machine_preset_folder,
    _model_calibration_preset_output_dir,
    _orientation_label,
    _oriented_geometry_path,
    _roi_outline_rgb,
    _roi_overlay_rgb,
    _workbench_colormap,
    _write_generated_support_grid_cache,
)
from capp.workbench.preview import PreviewPane


class _ButtonProbe:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)


class _WidgetProbe(_ButtonProbe):
    def __init__(self):
        super().__init__()
        self.visible = None

    def setVisible(self, visible):
        self.visible = bool(visible)


class _LineEditProbe:
    def __init__(self):
        self.text = ""

    def setText(self, text):
        self.text = str(text)


class _TextFieldProbe:
    def __init__(self, text):
        self._text = str(text)
        self.enabled = None

    def text(self):
        return self._text

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)


class _ComboProbe:
    def __init__(self, text):
        self.text = text
        self.visible = None

    def currentText(self):
        return self.text

    def setVisible(self, visible):
        self.visible = bool(visible)


class _ToggleProbe(_WidgetProbe):
    def __init__(self, checked=False):
        super().__init__()
        self.checked = bool(checked)
        self.text = ""
        self.arrow = None

    def isChecked(self):
        return self.checked

    def setText(self, text):
        self.text = text

    def setArrowType(self, arrow):
        self.arrow = arrow


class _CheckProbe(_WidgetProbe):
    def __init__(self, checked=False):
        super().__init__()
        self.checked = bool(checked)

    def isChecked(self):
        return self.checked


class _QtProbe:
    class ArrowType:
        DownArrow = "down"
        RightArrow = "right"


class _PreviewProbe:
    def __init__(self, mode="Shaded", overhang_limit="60"):
        self.stl_display_mode = _ComboProbe(mode)
        self.overhang_limit = _TextFieldProbe(overhang_limit)
        self.overlay_kwargs = None
        self.voxel_kwargs = None

    def show_stl_overlay_mesh(self, *args, **kwargs):
        self.overlay_kwargs = kwargs

    def show_voxels(self, *args, **kwargs):
        self.voxel_kwargs = kwargs


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


def test_oriented_geometry_path_exports_rotated_stl(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    source = tmp_path / "part.stl"
    trimesh.creation.box(extents=(2.0, 4.0, 6.0)).export(source)

    oriented = _oriented_geometry_path(source, tmp_path / "out", (90.0, 0.0, 0.0))
    loaded = trimesh.load_mesh(oriented, process=False)

    assert oriented != source.resolve()
    assert oriented.parent.name == "oriented_geometry"
    assert _orientation_label((90.0, 0.0, 0.0)) in oriented.name
    assert np.allclose(loaded.extents, (2.0, 6.0, 4.0))


def test_oriented_geometry_path_returns_source_for_identity(tmp_path):
    source = tmp_path / "part.stl"
    source.write_text("solid part\nendsolid part\n", encoding="utf-8")

    assert (
        _oriented_geometry_path(source, tmp_path / "out", (360.0, 0.0, 0.0))
        == source.resolve()
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


def test_selected_result_volume_can_hide_support_mask():
    view = WorkbenchMainWindow.__new__(WorkbenchMainWindow)
    binary = np.ones((2, 2, 2), dtype=bool)
    probability = np.full((2, 2, 2), 100, dtype=np.uint8)
    support_mask = np.zeros((2, 2, 2), dtype=bool)
    support_mask[0, 0, 0] = True
    view._loaded_result = {
        "binary": binary,
        "probability": probability,
        "voxel": binary,
        "support_mask": support_mask,
    }
    view._result_hide_support = _CheckProbe(checked=True)
    view._result_volume_choice = _ComboProbe("Binary")

    selected = view._selected_result_volume()

    assert not bool(selected[0, 0, 0])
    assert bool(selected[1, 1, 1])

    view._result_volume_choice.text = "Probability"
    selected_probability = view._selected_result_volume()

    assert selected_probability[0, 0, 0] == 0
    assert selected_probability[1, 1, 1] == 100


def test_geometry_deviation_volume_always_removes_support_mask():
    view = WorkbenchMainWindow.__new__(WorkbenchMainWindow)
    binary = np.ones((2, 2, 2), dtype=bool)
    probability = np.full((2, 2, 2), 100, dtype=np.uint8)
    support_mask = np.zeros((2, 2, 2), dtype=bool)
    support_mask[0, 0, 0] = True
    view._loaded_result = {
        "binary": binary,
        "probability": probability,
        "voxel": binary,
        "support_mask": support_mask,
    }
    view._result_hide_support = _CheckProbe(checked=False)
    view._result_volume_choice = _ComboProbe("Binary")

    selected = view._selected_result_volume_without_support()

    assert not bool(selected[0, 0, 0])
    assert bool(selected[1, 1, 1])

    view._result_volume_choice.text = "Probability"
    selected_probability = view._selected_result_volume_without_support()

    assert selected_probability[0, 0, 0] == 0
    assert selected_probability[1, 1, 1] == 100


def test_current_result_for_save_can_remove_support_mask():
    support_mask = np.zeros((2, 2, 2), dtype=bool)
    support_mask[0, 0, 0] = True
    view = WorkbenchMainWindow.__new__(WorkbenchMainWindow)
    view._last_result = None
    view._loaded_result = {
        "probability": np.full((2, 2, 2), 100, dtype=np.uint8),
        "binary": np.ones((2, 2, 2), dtype=bool),
        "voxel": np.ones((2, 2, 2), dtype=bool),
        "support_mask": support_mask,
        "spacing": 1.0,
        "origin": (0.0, 0.0, 0.0),
        "rest_volume": 100.0,
        "probability_density": 100.0,
        "elapsed_seconds": 1.0,
        "source_geometry": None,
    }
    view._result_hide_support = _CheckProbe(checked=True)

    result = view._current_result_for_save()

    assert not result.voxel[0, 0, 0]
    assert not result.binary[0, 0, 0]
    assert result.probability[0, 0, 0] == 0
    assert result.support_mask.sum() == 0


def test_support_overlay_uses_current_stl_display_settings():
    view = WorkbenchMainWindow.__new__(WorkbenchMainWindow)
    view._preview = _PreviewProbe(mode="Overhang angle", overhang_limit="45")
    view._last_support_overlay_preview = (
        "part.stl",
        object(),
        10,
        "support.stl",
        object(),
        5,
    )
    view._last_stl_preview = None

    view._show_stl_preview_from_cache()

    assert view._preview.overlay_kwargs == {
        "display_mode": "Overhang angle",
        "overhang_limit": 45.0,
    }


def test_voxelization_preview_passes_support_mask():
    data = np.ones((2, 2, 2), dtype=bool)
    support_mask = np.zeros_like(data)
    support_mask[0, 0, 0] = True
    view = WorkbenchMainWindow.__new__(WorkbenchMainWindow)
    view._preview = _PreviewProbe()
    view._last_voxel_grid = VoxelGrid(
        data=data,
        spacing=1.0,
        origin=(0.0, 0.0, 0.0),
        support_mask=support_mask,
    )
    view._last_voxel_preview_data = None

    view._show_voxelization_preview_from_cache()

    assert np.array_equal(view._preview.voxel_kwargs["support_mask"], support_mask)


def test_backend_status_line_is_compact():
    view = WorkbenchMainWindow.__new__(WorkbenchMainWindow)
    status = type("StatusProbe", (), {"label": "GPU CUDA", "available": False})()

    assert view._backend_status_line(status) == "GPU CUDA: unavailable"
    assert "\n" not in view._backend_status_line(status)


def test_support_options_expand_only_for_part_and_support():
    view = WorkbenchMainWindow.__new__(WorkbenchMainWindow)
    view._Qt = _QtProbe()
    view._part_type = _ComboProbe("Part only")
    view._support_options_toggle = _ToggleProbe(checked=True)
    view._support_options_panel = _WidgetProbe()
    view._support_geometry = _WidgetProbe()
    view._support_type = _WidgetProbe()

    view._update_support_controls()

    assert view._support_options_toggle.enabled is False
    assert view._support_geometry.enabled is False
    assert view._support_type.enabled is False
    assert view._support_options_panel.visible is False
    assert view._support_options_toggle.text == "Show support options"
    assert view._support_options_toggle.arrow == "right"

    view._part_type.text = "Part & Support"
    view._support_options_toggle.checked = False
    view._update_support_controls()

    assert view._support_options_toggle.enabled is True
    assert view._support_geometry.enabled is True
    assert view._support_type.enabled is True
    assert view._support_options_panel.visible is False

    view._toggle_support_options(True)

    assert view._support_options_panel.visible is True
    assert view._support_options_toggle.text == "Hide support options"
    assert view._support_options_toggle.arrow == "down"


def test_generated_support_is_inactive_until_button_result_matches(tmp_path):
    part = tmp_path / "part.stl"
    support = tmp_path / "support.stl"
    part.write_text("solid part\nendsolid part\n", encoding="utf-8")
    support.write_text("solid support\nendsolid support\n", encoding="utf-8")
    view = WorkbenchMainWindow.__new__(WorkbenchMainWindow)
    view._part_type = _ComboProbe("Part & Support")
    view._support_source = _ComboProbe("Generate from overhang")
    view._part_geometry = _TextFieldProbe(part)
    view._grid_spacing = _TextFieldProbe("1.0")
    view._support_type = _ComboProbe("X surface support")
    view._support_overhang_angle = _TextFieldProbe("60")
    view._support_pitch = _TextFieldProbe("2.0")
    view._support_thickness = _TextFieldProbe("1.0")
    view._support_footprint_offset = _TextFieldProbe("0.0")
    view._support_build_plate_z = _TextFieldProbe("0")
    view._last_generated_support_path = support
    view._last_generated_support_signature = None

    assert view._active_generated_support_options() is None

    options = view._support_generation_from_form()
    view._last_generated_support_signature = view._support_generation_signature(
        part,
        1.0,
        options,
    )

    assert view._active_generated_support_options() == options


def test_generated_support_path_is_ignored_when_options_are_stale(tmp_path):
    support = tmp_path / "support.stl"
    support.write_text("solid support\nendsolid support\n", encoding="utf-8")
    view = WorkbenchMainWindow.__new__(WorkbenchMainWindow)
    view._part_type = _ComboProbe("Part & Support")
    view._support_source = _ComboProbe("Generate from overhang")
    view._support_geometry = _TextFieldProbe("")
    view._support_type = _ComboProbe("Volume support")
    view._last_generated_support_path = support
    view._last_generated_support_type = "X surface support"
    view._last_generated_support_signature = None

    assert view._generated_support_path_and_voxel_type() is None


def test_generated_support_path_uses_active_generation_options(tmp_path):
    part = tmp_path / "part.stl"
    support = tmp_path / "support.stl"
    part.write_text("solid part\nendsolid part\n", encoding="utf-8")
    support.write_text("solid support\nendsolid support\n", encoding="utf-8")
    view = WorkbenchMainWindow.__new__(WorkbenchMainWindow)
    view._part_type = _ComboProbe("Part & Support")
    view._support_source = _ComboProbe("Generate from overhang")
    view._part_geometry = _TextFieldProbe(part)
    view._grid_spacing = _TextFieldProbe("1.0")
    view._support_geometry = _TextFieldProbe("")
    view._support_type = _ComboProbe("X surface support")
    view._support_overhang_angle = _TextFieldProbe("60")
    view._support_pitch = _TextFieldProbe("2.0")
    view._support_thickness = _TextFieldProbe("1.0")
    view._support_footprint_offset = _TextFieldProbe("0.0")
    view._support_build_plate_z = _TextFieldProbe("0")
    view._last_generated_support_path = support
    view._last_generated_support_type = None
    options = view._support_generation_from_form()
    view._last_generated_support_signature = view._support_generation_signature(
        part,
        1.0,
        options,
    )

    assert view._generated_support_path_and_voxel_type() == (support, "Line support")


def test_cached_generated_support_grid_is_reused_for_matching_config(tmp_path):
    support = tmp_path / "support.stl"
    support.write_text("solid support\nendsolid support\n", encoding="utf-8")
    grid = VoxelGrid(np.ones((1, 1, 1), dtype=bool), spacing=0.5)
    config = type(
        "ConfigProbe",
        (),
        {"support_geometry_path": support, "voxel_spacing": 0.5},
    )()
    view = WorkbenchMainWindow.__new__(WorkbenchMainWindow)
    view._last_generated_support_path = support
    view._last_generated_support_grid = grid

    assert view._cached_generated_support_grid_for_config(config) is grid


def test_cached_generated_support_grid_can_load_sidecar_cache(tmp_path):
    support = tmp_path / "support.stl"
    support.write_text("solid support\nendsolid support\n", encoding="utf-8")
    data = np.zeros((2, 2, 2), dtype=bool)
    data[0, 1, 1] = True
    grid = VoxelGrid(data, spacing=0.25, origin=(1.0, 2.0, 3.0), support_mask=data)
    _write_generated_support_grid_cache(support, grid, "X surface support")
    config = type(
        "ConfigProbe",
        (),
        {"support_geometry_path": support, "voxel_spacing": 0.25},
    )()
    view = WorkbenchMainWindow.__new__(WorkbenchMainWindow)
    view._last_generated_support_path = None
    view._last_generated_support_grid = None
    view._last_generated_support_type = None

    loaded = view._cached_generated_support_grid_for_config(config)

    assert loaded is not None
    assert loaded.filled_count == 1
    assert loaded.origin == (1.0, 2.0, 3.0)
    assert view._last_generated_support_type == "X surface support"


def test_generated_support_cache_round_trips_type_and_grid(tmp_path):
    support = tmp_path / "support.stl"
    data = np.ones((1, 2, 1), dtype=bool)
    grid = VoxelGrid(data, spacing=0.5, origin=(0.5, 0.0, -1.0), support_mask=data)

    _write_generated_support_grid_cache(support, grid, "Column support")
    loaded, support_type = _load_generated_support_grid_cache(support)

    assert support_type == "Column support"
    assert loaded is not None
    assert np.array_equal(loaded.data, data)
    assert loaded.origin == (0.5, 0.0, -1.0)


def test_support_overlay_downsampling_preserves_thin_features():
    pane = PreviewPane.__new__(PreviewPane)
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1, 1, 1] = True

    prepared = pane._prepare_support_overlay_data(mask, stride=2, shape=(2, 2, 2))

    assert prepared is not None
    assert prepared[0, 0, 0]


def test_overhang_limit_control_is_hidden_until_overhang_mode():
    pane = PreviewPane.__new__(PreviewPane)
    pane._stl_controls_visible = True
    pane.stl_display_mode = _ComboProbe("Shaded")
    pane.overhang_limit = _WidgetProbe()

    pane._sync_stl_control_visibility()

    assert pane.overhang_limit.visible is False

    pane.stl_display_mode.text = "Overhang angle"
    pane._sync_stl_control_visibility()

    assert pane.overhang_limit.visible is True

    pane.set_stl_controls_visible(False)

    assert pane.overhang_limit.visible is False


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
