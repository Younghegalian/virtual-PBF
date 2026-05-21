import trimesh

from capp.geometry.voxelizer import (
    union_voxel_grids,
    voxelize_mesh,
    voxelize_part_and_support,
    voxelize_surface_shell,
)


def test_voxelizer_uses_matlab_virtual_printing_padding(tmp_path):
    path = tmp_path / "box.stl"
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    mesh.apply_translation((1.0, 1.0, 1.0))
    mesh.export(path)

    grid = voxelize_mesh(path, spacing=1.0)

    assert grid.shape == (6, 6, 4)
    assert grid.origin == (-2.0, -2.0, 0.0)
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


def test_union_voxel_grids_expands_to_cover_offsets(tmp_path):
    part = voxelize_mesh(
        _box_stl(tmp_path, "part", (2.0, 2.0, 2.0), (1.0, 1.0, 1.0)),
        spacing=1.0,
    )
    support = voxelize_mesh(
        _box_stl(tmp_path, "support", (1.0, 1.0, 1.0), (5.0, 1.0, 0.5)),
        spacing=1.0,
    )

    combined = union_voxel_grids([part, support])

    assert combined.origin[0] == part.origin[0]
    assert combined.shape[0] > part.shape[0]
    assert combined.filled_count >= part.filled_count + support.filled_count


def test_part_and_volume_support_voxelization_unions_inputs(tmp_path):
    part_path = _box_stl(tmp_path, "part", (2.0, 2.0, 2.0), (1.0, 1.0, 1.0))
    support_path = _box_stl(tmp_path, "support", (1.0, 1.0, 2.0), (4.0, 1.0, 1.0))

    part = voxelize_mesh(part_path, spacing=1.0)
    combined = voxelize_part_and_support(
        part_path,
        support_path,
        spacing=1.0,
        support_type="Volume support",
    )

    assert combined.filled_count > part.filled_count
    assert combined.shape[0] > part.shape[0]


def test_line_support_voxelization_accepts_open_surface(tmp_path):
    path = tmp_path / "line_support.stl"
    mesh = trimesh.Trimesh(
        vertices=[
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 2.0, 0.0),
            (0.0, 2.0, 0.0),
        ],
        faces=[(0, 1, 2), (0, 2, 3)],
        process=False,
    )
    mesh.export(path)

    grid = voxelize_surface_shell(path, spacing=1.0)

    assert grid.filled_count > 0
    assert grid.shape[0] >= 4


def _box_stl(tmp_path, name, extents, translation):
    path = tmp_path / f"{name}.stl"
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(translation)
    mesh.export(path)
    return path
