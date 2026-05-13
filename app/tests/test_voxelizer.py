import trimesh

from capp.geometry.voxelizer import voxelize_mesh


def test_voxelizer_uses_matlab_virtual_printing_padding(tmp_path):
    path = tmp_path / "box.stl"
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    mesh.apply_translation((1.0, 1.0, 1.0))
    mesh.export(path)

    grid = voxelize_mesh(path, spacing=1.0)

    assert grid.shape == (6, 6, 4)
    assert grid.data[:2, :, :].sum() == 0
    assert grid.data[-2:, :, :].sum() == 0
    assert grid.data[:, :2, :].sum() == 0
    assert grid.data[:, -2:, :].sum() == 0
    assert grid.data[:, :, -2:].sum() == 0
    assert grid.filled_count > 0


def test_voxelizer_reports_progress(tmp_path):
    path = tmp_path / "box.stl"
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    mesh.apply_translation((1.0, 1.0, 1.0))
    mesh.export(path)
    events = []

    voxelize_mesh(path, spacing=1.0, progress_callback=lambda p, m: events.append((p, m)))

    assert events[0][0] == 0
    assert events[-1][0] == 100
    assert all(0 <= percent <= 100 for percent, _message in events)
