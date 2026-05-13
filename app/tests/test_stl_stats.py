import trimesh

from capp.geometry.stl_stats import estimate_spacing_from_bounds, read_stl_stats


def test_binary_stl_stats_reads_bounds_without_full_mesh_loader(tmp_path):
    path = tmp_path / "box.stl"
    mesh = trimesh.creation.box(extents=(2.0, 4.0, 6.0))
    mesh.export(path)

    stats = read_stl_stats(path)

    assert stats.triangle_count == 12
    assert stats.bounds.shape == (2, 3)
    assert estimate_spacing_from_bounds(stats.bounds) > 0.0
