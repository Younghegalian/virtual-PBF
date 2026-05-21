import numpy as np
import pytest

from capp.workbench.preview import (
    PreviewPane,
    _align_surface_to_source_origin,
    _deviation_color_limits,
    _deviation_color_values_mm,
    _deviation_jet_colormap,
    build_geometry_deviation_preview,
    pack_geometry_deviation_preview,
    unpack_geometry_deviation_preview,
)


def test_paraview_volume_keeps_large_volume_full_resolution():
    pane = object.__new__(PreviewPane)
    volume = np.ones((360, 360, 360), dtype=bool)

    data, stride = PreviewPane._prepare_volume_data(pane, volume, "ParaView Volume")

    assert stride == 1
    assert data.shape == volume.shape


def test_isosurface_uses_surface_extraction_safety_stride():
    pane = object.__new__(PreviewPane)
    volume = np.ones((720, 720, 720), dtype=bool)

    data, stride = PreviewPane._prepare_volume_data(pane, volume, "Isosurface")

    assert stride > 1
    assert data.shape[0] < volume.shape[0]


def test_block_preview_uses_geometry_safety_pooling():
    pane = object.__new__(PreviewPane)
    volume = np.ones((180, 180, 180), dtype=bool)

    data, stride = PreviewPane._prepare_volume_data(pane, volume, "Voxel Blocks")
    block_data, block_stride = PreviewPane._prepare_binary_block_data(pane, data)

    assert stride == 1
    assert block_stride > 1
    assert block_data.size < data.size


def test_smooth_volume_preserves_shape_and_normalizes():
    pane = object.__new__(PreviewPane)
    volume = np.zeros((9, 9, 9), dtype=np.float32)
    volume[4, 4, 4] = 1.0

    smooth = PreviewPane._prepare_smooth_volume_data(pane, volume, binary_like=True)

    assert smooth.shape == volume.shape
    assert np.isclose(smooth.max(), 1.0)
    assert smooth[4, 4, 4] == smooth.max()
    assert np.count_nonzero(smooth > 0.0) > 1


def test_isosurface_mesh_builds_from_binary_volume():
    pv = pytest.importorskip("pyvista")
    pane = object.__new__(PreviewPane)
    volume = np.zeros((8, 8, 8), dtype=np.float32)
    volume[2:6, 2:6, 2:6] = 1.0

    surface = PreviewPane._make_isosurface_mesh(
        pane,
        pv,
        volume,
        spacing=0.5,
        origin=(1.0, 2.0, 3.0),
    )

    assert surface.n_points > 0
    assert surface.n_cells > 0
    assert surface.bounds[0] >= 1.0
    assert surface.bounds[2] >= 2.0
    assert surface.bounds[4] >= 3.0


def test_isosurface_closes_boundary_touching_volume():
    pv = pytest.importorskip("pyvista")
    pane = object.__new__(PreviewPane)
    volume = np.ones((5, 5, 3), dtype=np.float32)

    surface = PreviewPane._make_isosurface_mesh(
        pane,
        pv,
        volume,
        spacing=1.0,
        origin=(0.0, 0.0, 0.0),
    )
    boundary_edges = surface.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )

    assert surface.bounds == (0.0, 5.0, 0.0, 5.0, 0.0, 3.0)
    assert boundary_edges.n_cells == 0


def test_geometry_deviation_surface_reports_near_zero_for_matching_box():
    pv = pytest.importorskip("pyvista")
    pane = object.__new__(PreviewPane)
    volume = np.zeros((8, 8, 8), dtype=np.float32)
    volume[2:6, 2:6, 2:6] = 1.0
    printed = PreviewPane._make_isosurface_mesh(
        pane,
        pv,
        volume,
        spacing=1.0,
        origin=(0.0, 0.0, 0.0),
    )
    original = pv.Box(bounds=(2.0, 6.0, 2.0, 6.0, 2.0, 6.0)).triangulate()

    surface, metrics = PreviewPane._geometry_deviation_surface(pane, printed, original)

    assert metrics["sample_count"] > 0
    assert metrics["max_abs_mm"] < 1e-5
    assert "Deviation color (mm)" in surface.point_data


def test_deviation_color_limits_use_mm_ranges_independently():
    metrics = {"negative_scale_mm": 2.0, "positive_scale_mm": 4.0}

    assert _deviation_color_limits(metrics, 1.0) == (-2.0, 4.0)
    assert _deviation_color_limits({"negative_scale_mm": 2.0}, 1.0) == (-2.0, 0.0)
    assert _deviation_color_limits({"positive_scale_mm": 4.0}, 1.0) == (0.0, 4.0)


def test_deviation_jet_colormap_keeps_zero_color_constant():
    both = _deviation_jet_colormap(2.0, 4.0)
    positive = _deviation_jet_colormap(0.0, 4.0)
    negative = _deviation_jet_colormap(2.0, 0.0)

    assert both(2.0 / 6.0) == pytest.approx(positive(0.0))
    assert both(2.0 / 6.0) == pytest.approx(negative(1.0))


def test_deviation_color_values_keep_mm_scale_and_zero_band():
    values = _deviation_color_values_mm(
        np.asarray([-2.0, -0.01, 0.0, 0.01, 4.0]),
        zero_tolerance=0.02,
    )

    assert values.tolist() == pytest.approx([-2.0, 0.0, 0.0, 0.0, 4.0])


def test_geometry_deviation_alignment_restores_legacy_normalized_origin():
    pv = pytest.importorskip("pyvista")
    pane = object.__new__(PreviewPane)
    volume = np.zeros((6, 6, 4), dtype=np.float32)
    volume[2:4, 2:4, 0:2] = 1.0
    printed = PreviewPane._make_isosurface_mesh(
        pane,
        pv,
        volume,
        spacing=1.0,
        origin=(0.0, 0.0, 0.0),
    )
    original = pv.Box(bounds=(10.0, 12.0, 20.0, 22.0, 30.0, 32.0)).triangulate()

    aligned, offset = _align_surface_to_source_origin(
        printed,
        original,
        voxel_spacing=1.0,
        origin=(0.0, 0.0, 0.0),
    )

    assert offset == (8.0, 18.0, 30.0)
    assert aligned.bounds == pytest.approx(original.bounds)


def test_geometry_deviation_alignment_keeps_source_origin_in_place():
    pv = pytest.importorskip("pyvista")
    pane = object.__new__(PreviewPane)
    volume = np.zeros((6, 6, 4), dtype=np.float32)
    volume[2:4, 2:4, 0:2] = 1.0
    original = pv.Box(bounds=(10.0, 12.0, 20.0, 22.0, 30.0, 32.0)).triangulate()
    printed = PreviewPane._make_isosurface_mesh(
        pane,
        pv,
        volume,
        spacing=1.0,
        origin=(8.0, 18.0, 30.0),
    )

    aligned, offset = _align_surface_to_source_origin(
        printed,
        original,
        voxel_spacing=1.0,
        origin=(8.0, 18.0, 30.0),
    )

    assert offset == (0.0, 0.0, 0.0)
    assert aligned is printed
    assert aligned.bounds == pytest.approx(original.bounds)


def test_geometry_deviation_build_aligns_legacy_result_to_source_stl(tmp_path):
    pytest.importorskip("pyvista")
    trimesh = pytest.importorskip("trimesh")
    stl_path = tmp_path / "box.stl"
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    mesh.apply_translation((11.0, 21.0, 31.0))
    mesh.export(stl_path)
    volume = np.zeros((6, 6, 4), dtype=np.float32)
    volume[2:4, 2:4, 0:2] = 1.0

    preview = build_geometry_deviation_preview(
        stl_path,
        volume,
        spacing=1.0,
        origin=(0.0, 0.0, 0.0),
    )

    assert preview.alignment_offset == (8.0, 18.0, 30.0)
    assert preview.metrics["sample_count"] > 0
    assert preview.metrics["max_abs_mm"] < 1e-5


def test_geometry_deviation_preview_payload_round_trip(tmp_path):
    pytest.importorskip("pyvista")
    trimesh = pytest.importorskip("trimesh")
    stl_path = tmp_path / "box.stl"
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    mesh.apply_translation((1.0, 1.0, 1.0))
    mesh.export(stl_path)
    volume = np.zeros((6, 6, 4), dtype=np.float32)
    volume[2:4, 2:4, 0:2] = 1.0
    preview = build_geometry_deviation_preview(
        stl_path,
        volume,
        spacing=1.0,
        origin=(-2.0, -2.0, 0.0),
    )

    restored = unpack_geometry_deviation_preview(pack_geometry_deviation_preview(preview))

    assert restored.original_mesh.n_points == preview.original_mesh.n_points
    assert restored.deviation_surface.n_points == preview.deviation_surface.n_points
    assert restored.metrics == preview.metrics
    assert "Deviation color (mm)" in restored.deviation_surface.point_data
