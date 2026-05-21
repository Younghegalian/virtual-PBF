import numpy as np
import trimesh

from capp.domain import SupportGenerationParameters
from capp.geometry.voxelizer import (
    generate_overhang_support_grid,
    union_voxel_grids,
    voxelize_mesh,
    voxelize_part_and_support,
    voxelize_surface_shell,
)
from capp.io.exports import write_binary_stl, write_surface_stl


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
    assert combined.support_mask.sum() > 0


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


def test_overhang_support_generation_extrudes_to_build_plate(tmp_path):
    part_path = _box_stl(tmp_path, "floating_plate", (4.0, 4.0, 1.0), (2.0, 2.0, 2.0))
    part = voxelize_mesh(part_path, spacing=1.0)
    support = generate_overhang_support_grid(
        part_path,
        part,
        SupportGenerationParameters(
            support_type="Volume support",
            overhang_angle=60.0,
            footprint_offset=0.0,
            build_plate_z=0.0,
        ),
    )

    assert support.filled_count > 0
    assert np.array_equal(support.support_mask, support.data)
    assert support.origin[2] <= 0.0
    assert not np.any(support.data[:, :, -part.shape[2] :] & part.data)
    assert np.any(support.data[:, :, :2])


def test_overhang_support_skips_faces_already_on_build_plate(tmp_path):
    part_path = _box_stl(tmp_path, "grounded_plate", (4.0, 4.0, 1.0), (2.0, 2.0, 0.5))
    part = voxelize_mesh(part_path, spacing=1.0)

    support = generate_overhang_support_grid(
        part_path,
        part,
        SupportGenerationParameters(
            support_type="Volume support",
            overhang_angle=60.0,
            footprint_offset=0.0,
            build_plate_z=0.0,
        ),
    )

    assert support.filled_count == 0
    assert support.support_mask.sum() == 0


def test_overhang_support_skips_near_bed_bottom_faces(tmp_path):
    part_path = _box_stl(tmp_path, "near_bed_plate", (4.0, 4.0, 1.0), (2.0, 2.0, 0.85))
    part = voxelize_mesh(part_path, spacing=0.5)

    support = generate_overhang_support_grid(
        part_path,
        part,
        SupportGenerationParameters(
            support_type="Volume support",
            overhang_angle=60.0,
            footprint_offset=0.0,
            build_plate_z=0.0,
        ),
    )

    assert support.filled_count == 0
    assert support.support_mask.sum() == 0


def test_generated_support_modes_shape_lattice_density(tmp_path):
    part_path = _box_stl(tmp_path, "wide_overhang", (6.0, 6.0, 1.0), (3.0, 3.0, 2.0))
    part = voxelize_mesh(part_path, spacing=1.0)

    volume = generate_overhang_support_grid(
        part_path,
        part,
        SupportGenerationParameters(
            support_type="Volume support",
            overhang_angle=60.0,
            footprint_offset=0.0,
            build_plate_z=0.0,
        ),
    )
    x_surface = generate_overhang_support_grid(
        part_path,
        part,
        SupportGenerationParameters(
            support_type="X surface support",
            overhang_angle=60.0,
            pitch=3.0,
            thickness=1.0,
            footprint_offset=0.0,
            build_plate_z=0.0,
        ),
    )
    column = generate_overhang_support_grid(
        part_path,
        part,
        SupportGenerationParameters(
            support_type="Column support",
            overhang_angle=60.0,
            pitch=3.0,
            thickness=1.0,
            footprint_offset=0.0,
            build_plate_z=0.0,
        ),
    )

    assert volume.filled_count >= x_surface.filled_count >= column.filled_count > 0


def test_part_and_generated_support_voxelization_unions_inputs(tmp_path):
    part_path = _box_stl(tmp_path, "generated_support_part", (4.0, 4.0, 1.0), (2.0, 2.0, 2.0))
    part = voxelize_mesh(part_path, spacing=1.0)

    combined = voxelize_part_and_support(
        part_path,
        None,
        spacing=1.0,
        support_generation=SupportGenerationParameters(
            support_type="X surface support",
            overhang_angle=60.0,
            pitch=2.0,
            thickness=1.0,
            build_plate_z=0.0,
        ),
    )

    assert combined.filled_count > part.filled_count
    assert combined.support_mask.sum() > 0
    assert combined.origin[2] <= 0.0


def test_part_and_support_without_generation_request_keeps_part_only(tmp_path):
    part_path = _box_stl(tmp_path, "part_only_when_not_generated", (4.0, 4.0, 1.0), (2.0, 2.0, 2.0))
    part = voxelize_mesh(part_path, spacing=1.0)

    combined = voxelize_part_and_support(part_path, None, spacing=1.0)

    assert combined.filled_count == part.filled_count
    assert combined.support_mask.sum() == 0


def test_x_surface_support_ignores_thickness_parameter(tmp_path):
    part_path = _box_stl(tmp_path, "x_surface_thickness", (6.0, 6.0, 1.0), (3.0, 3.0, 2.0))
    part = voxelize_mesh(part_path, spacing=1.0)

    thin = generate_overhang_support_grid(
        part_path,
        part,
        SupportGenerationParameters(
            support_type="X surface support",
            overhang_angle=60.0,
            pitch=3.0,
            thickness=1.0,
            footprint_offset=0.0,
            build_plate_z=0.0,
        ),
    )
    thick = generate_overhang_support_grid(
        part_path,
        part,
        SupportGenerationParameters(
            support_type="X surface support",
            overhang_angle=60.0,
            pitch=3.0,
            thickness=4.0,
            footprint_offset=0.0,
            build_plate_z=0.0,
        ),
    )

    assert np.array_equal(thin.data, thick.data)


def test_footprint_offset_does_not_leave_floating_support_segments(tmp_path):
    base = trimesh.creation.box(extents=(2.0, 2.0, 1.5))
    base.apply_translation((0.0, 0.0, 0.75))
    plate = trimesh.creation.box(extents=(6.0, 6.0, 0.5))
    plate.apply_translation((0.0, 0.0, 2.25))
    part_path = tmp_path / "offset_floating_segment.stl"
    trimesh.util.concatenate([base, plate]).export(part_path)
    part = voxelize_mesh(part_path, spacing=0.5)

    support = generate_overhang_support_grid(
        part_path,
        part,
        SupportGenerationParameters(
            support_type="Volume support",
            overhang_angle=60.0,
            footprint_offset=1.0,
            build_plate_z=0.0,
        ),
    )

    occupied_xy = np.argwhere(support.data.any(axis=2))
    assert occupied_xy.size > 0
    column_bottoms = [
        int(np.where(support.data[x_index, y_index, :])[0].min())
        for x_index, y_index in occupied_xy
    ]
    assert set(column_bottoms) == {min(column_bottoms)}


def test_generated_support_grid_can_be_exported_as_stl(tmp_path):
    part_path = _box_stl(tmp_path, "export_support_part", (4.0, 4.0, 1.0), (2.0, 2.0, 2.0))
    part = voxelize_mesh(part_path, spacing=1.0)
    support = generate_overhang_support_grid(
        part_path,
        part,
        SupportGenerationParameters(
            support_type="Volume support",
            overhang_angle=60.0,
            footprint_offset=0.0,
            build_plate_z=0.0,
        ),
    )
    output_path = tmp_path / "generated_support.stl"

    write_binary_stl(
        output_path,
        support.data,
        support.spacing,
        support.origin,
        clip_min_z=0.0,
        voxel_bounds=True,
    )

    loaded = trimesh.load_mesh(output_path, process=False)
    assert output_path.exists()
    assert output_path.stat().st_size > 84
    assert np.isclose(float(loaded.vertices[:, 2].min()), 0.0)
    assert np.isclose(float(loaded.vertices[:, 2].max()), 1.5)


def test_x_surface_support_can_be_exported_as_open_surface_stl(tmp_path):
    part_path = _box_stl(tmp_path, "surface_export_part", (4.0, 4.0, 1.0), (2.0, 2.0, 2.0))
    part = voxelize_mesh(part_path, spacing=1.0)
    support = generate_overhang_support_grid(
        part_path,
        part,
        SupportGenerationParameters(
            support_type="X surface support",
            overhang_angle=60.0,
            pitch=2.0,
            thickness=1.0,
            build_plate_z=0.0,
        ),
    )
    output_path = tmp_path / "generated_surface_support.stl"

    write_surface_stl(output_path, support.data, support.spacing, support.origin, bottom_z=0.0)

    loaded = trimesh.load_mesh(output_path, process=False)
    assert output_path.exists()
    assert len(loaded.faces) > 0
    assert np.isclose(float(loaded.vertices[:, 2].min()), 0.0)


def _box_stl(tmp_path, name, extents, translation):
    path = tmp_path / f"{name}.stl"
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(translation)
    mesh.export(path)
    return path
