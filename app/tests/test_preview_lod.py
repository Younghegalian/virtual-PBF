import numpy as np
import pytest

from capp.workbench.preview import PreviewPane


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

    _surface, metrics = PreviewPane._geometry_deviation_surface(pane, printed, original)

    assert metrics["sample_count"] > 0
    assert metrics["max_abs_mm"] < 1e-5
