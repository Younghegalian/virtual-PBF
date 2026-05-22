import numpy as np
import trimesh

from capp.domain import SupportGenerationParameters
from capp.geometry.voxelizer import (
    _column_lattice_pattern,
    _grid_data_in_frame,
    _mark_projected_triangle,
    _x_surface_lattice_pattern,
    generate_overhang_support_grid,
    union_voxel_grids,
    voxelize_mesh,
    voxelize_part_and_support,
    voxelize_part_with_support_grid,
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


def test_line_support_voxelization_handles_thin_vertical_surface(tmp_path):
    path = tmp_path / "thin_vertical_support.stl"
    mesh = trimesh.Trimesh(
        vertices=[
            (0.0, 0.0, 0.0),
            (0.05, 0.0, 0.0),
            (0.05, 0.0, 5.0),
            (0.0, 0.0, 5.0),
        ],
        faces=[(0, 1, 2), (0, 2, 3)],
        process=False,
    )
    mesh.export(path)

    grid = voxelize_surface_shell(path, spacing=0.2)
    occupied = np.argwhere(grid.data)

    assert grid.filled_count > 20
    assert np.unique(occupied[:, 1]).size == 1
    assert np.ptp(occupied[:, 2]) >= 20


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


def test_support_contact_overlap_can_penetrate_part_for_preview_connection(tmp_path):
    part_path = _box_stl(tmp_path, "contact_overlap_plate", (4.0, 4.0, 1.0), (2.0, 2.0, 2.0))
    part = voxelize_mesh(part_path, spacing=1.0)

    support = generate_overhang_support_grid(
        part_path,
        part,
        SupportGenerationParameters(
            support_type="Volume support",
            overhang_angle=60.0,
            footprint_offset=0.0,
            contact_depth=1.0,
            build_plate_z=0.0,
        ),
    )

    offset = np.rint(
        (np.asarray(part.origin) - np.asarray(support.origin)) / support.spacing
    ).astype(int)
    part_in_support = np.zeros_like(support.data, dtype=bool)
    part_in_support[
        offset[0] : offset[0] + part.shape[0],
        offset[1] : offset[1] + part.shape[1],
        offset[2] : offset[2] + part.shape[2],
    ] = part.data

    assert np.any(support.data & part_in_support)


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


def test_column_support_lattice_is_centered_and_uniform():
    footprint = np.ones((12, 12), dtype=bool)

    anchors = _column_lattice_pattern(footprint, pitch_cells=4)

    occupied = np.argwhere(anchors)
    assert occupied.size > 0
    assert np.array_equal(np.unique(occupied[:, 0]), np.asarray([2, 6, 10]))
    assert np.array_equal(np.unique(occupied[:, 1]), np.asarray([2, 6, 10]))


def test_x_surface_lattice_uses_global_phase_across_components():
    footprint = np.ones((9, 9), dtype=bool)
    footprint[4, :] = False

    pattern = _x_surface_lattice_pattern(footprint, pitch_cells=3)

    assert pattern.sum() < footprint.sum()
    assert pattern[0, 0]
    assert pattern[8, 8]
    assert not pattern[4, 4]
    assert np.array_equal(pattern, footprint & np.flipud(np.fliplr(pattern)))


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


def test_materialized_support_stl_takes_precedence_over_generation_options(tmp_path):
    part_path = _box_stl(tmp_path, "materialized_support_part", (4.0, 4.0, 1.0), (2.0, 2.0, 0.5))
    support_path = _box_stl(
        tmp_path,
        "materialized_support",
        (1.0, 1.0, 1.0),
        (2.0, 2.0, 2.0),
    )

    combined = voxelize_part_and_support(
        part_path,
        support_path,
        spacing=1.0,
        support_type="Volume support",
        support_generation=SupportGenerationParameters(
            support_type="invalid generated support mode",
            overhang_angle=60.0,
        ),
    )

    assert combined.support_mask.sum() > 0


def test_generated_support_label_preserves_contact_overlap(tmp_path):
    part_path = _box_stl(tmp_path, "generated_support_overlap", (4.0, 4.0, 1.0), (2.0, 2.0, 2.0))

    combined = voxelize_part_and_support(
        part_path,
        None,
        spacing=1.0,
        support_generation=SupportGenerationParameters(
            support_type="X surface support",
            overhang_angle=60.0,
            pitch=2.0,
            thickness=1.0,
            contact_depth=1.0,
            build_plate_z=0.0,
        ),
    )
    part = voxelize_mesh(part_path, spacing=1.0)
    part_in_combined = _grid_data_in_frame(
        part,
        combined.origin,
        combined.shape,
        combined.spacing,
    )

    assert np.any(combined.support_mask & part_in_combined)


def test_cached_generated_support_grid_can_be_combined_without_rasterizing_stl(tmp_path):
    part_path = _box_stl(tmp_path, "cached_support_part", (4.0, 4.0, 1.0), (2.0, 2.0, 2.0))
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

    combined = voxelize_part_with_support_grid(part_path, support, spacing=1.0)

    assert combined.filled_count > part.filled_count
    assert combined.support_mask.sum() > 0


def test_multi_axis_center_projection_keeps_surface_support_thin():
    triangle = np.asarray(
        [
            [0.1, 0.1, 0.1],
            [4.8, 0.4, 2.2],
            [0.4, 4.8, 2.4],
        ],
        dtype=np.float64,
    )
    origin = np.zeros(3, dtype=np.float64)
    single = np.zeros((6, 6, 6), dtype=bool)
    multi = np.zeros_like(single)

    _mark_projected_triangle(
        single,
        triangle,
        origin,
        spacing=1.0,
        step=0.5,
        sample_offsets=((0.5, 0.5),),
    )
    _mark_projected_triangle(
        multi,
        triangle,
        origin,
        spacing=1.0,
        step=0.5,
        multi_axis=True,
        sample_offsets=((0.5, 0.5),),
    )

    assert multi.sum() >= single.sum()
    assert multi.sum() <= single.sum() * 3


def test_corner_projection_is_only_used_for_solid_overhang_rasterization():
    triangle = np.asarray(
        [
            [0.1, 0.1, 0.1],
            [4.8, 0.4, 2.2],
            [0.4, 4.8, 2.4],
        ],
        dtype=np.float64,
    )
    origin = np.zeros(3, dtype=np.float64)
    single = np.zeros((6, 6, 6), dtype=bool)
    multi = np.zeros_like(single)

    _mark_projected_triangle(single, triangle, origin, spacing=1.0, step=0.5)
    _mark_projected_triangle(
        multi,
        triangle,
        origin,
        spacing=1.0,
        step=0.5,
        multi_axis=True,
    )

    assert multi.sum() >= single.sum()
    assert np.any(multi & ~single)


def test_cached_generated_support_label_preserves_contact_overlap(tmp_path):
    part_path = _box_stl(tmp_path, "cached_support_overlap", (4.0, 4.0, 1.0), (2.0, 2.0, 2.0))
    part = voxelize_mesh(part_path, spacing=1.0)
    support = generate_overhang_support_grid(
        part_path,
        part,
        SupportGenerationParameters(
            support_type="X surface support",
            overhang_angle=60.0,
            pitch=2.0,
            thickness=1.0,
            contact_depth=1.0,
            build_plate_z=0.0,
        ),
    )

    combined = voxelize_part_with_support_grid(part_path, support, spacing=1.0)
    part_in_combined = _grid_data_in_frame(
        part,
        combined.origin,
        combined.shape,
        combined.spacing,
    )

    assert np.any(combined.support_mask & part_in_combined)


def test_cached_support_label_is_not_silently_dropped_when_it_overlaps_part(tmp_path):
    part_path = _box_stl(tmp_path, "overlapped_support_part", (2.0, 2.0, 2.0), (1.0, 1.0, 1.0))
    part = voxelize_mesh(part_path, spacing=1.0)

    combined = voxelize_part_with_support_grid(part_path, part, spacing=1.0)

    assert combined.support_mask.sum() == part.filled_count


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
    base_z = min(column_bottoms)
    for (x_index, y_index), bottom_z in zip(occupied_xy, column_bottoms, strict=True):
        assert bottom_z == base_z or part.data[x_index, y_index, bottom_z - 1]


def test_overhang_support_can_anchor_to_lower_part_surface(tmp_path):
    lower_plate = trimesh.creation.box(extents=(4.0, 4.0, 0.5))
    lower_plate.apply_translation((0.0, 0.0, 0.25))
    upper_plate = trimesh.creation.box(extents=(6.0, 6.0, 0.5))
    upper_plate.apply_translation((0.0, 0.0, 3.25))
    part_path = tmp_path / "part_anchored_support.stl"
    trimesh.util.concatenate([lower_plate, upper_plate]).export(part_path)
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

    occupied_xy = np.argwhere(support.data.any(axis=2))
    column_bottoms = np.array(
        [
            int(np.where(support.data[x_index, y_index, :])[0].min())
            for x_index, y_index in occupied_xy
        ],
        dtype=np.int32,
    )
    anchored = occupied_xy[column_bottoms > 0]

    assert anchored.size > 0
    for (x_index, y_index), bottom_z in zip(
        anchored,
        column_bottoms[column_bottoms > 0],
        strict=True,
    ):
        assert part.data[x_index, y_index, bottom_z - 1]


def test_auto_build_plate_support_does_not_extend_below_part_bottom(tmp_path):
    base = trimesh.creation.box(extents=(2.0, 2.0, 1.0))
    base.apply_translation((0.0, 0.0, 0.7))
    plate = trimesh.creation.box(extents=(6.0, 6.0, 0.5))
    plate.apply_translation((0.0, 0.0, 2.2))
    part_path = tmp_path / "near_bed_overhang.stl"
    part_mesh = trimesh.util.concatenate([base, plate])
    part_mesh.export(part_path)
    part = voxelize_mesh(part_path, spacing=0.5)

    support = generate_overhang_support_grid(
        part_path,
        part,
        SupportGenerationParameters(
            support_type="Volume support",
            overhang_angle=60.0,
            footprint_offset=0.0,
            build_plate_z=None,
        ),
    )
    occupied = np.argwhere(support.data)

    assert support.filled_count > 0
    support_min_z = support.origin[2] + int(occupied[:, 2].min()) * support.spacing
    assert support_min_z >= float(part_mesh.bounds[0, 2]) - 1e-9


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


def test_x_surface_export_keeps_both_diagonals_at_intersection(tmp_path):
    volume = np.zeros((3, 3, 2), dtype=bool)
    for x_index, y_index in [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0)]:
        volume[x_index, y_index, :] = True
    output_path = tmp_path / "x_intersection_surface.stl"

    write_surface_stl(output_path, volume, spacing=1.0, origin=(0.0, 0.0, 0.0))

    loaded = trimesh.load_mesh(output_path, process=False)
    assert len(loaded.faces) == 12


def test_x_surface_export_spans_full_diagonal_cells(tmp_path):
    volume = np.zeros((2, 2, 2), dtype=bool)
    volume[0, 0, :] = True
    volume[1, 1, :] = True
    output_path = tmp_path / "x_connected_surface.stl"

    write_surface_stl(output_path, volume, spacing=1.0, origin=(0.0, 0.0, 0.0))

    loaded = trimesh.load_mesh(output_path, process=False)
    vertices = np.asarray(loaded.vertices)
    assert np.isclose(float(vertices[:, 0].min()), 0.0)
    assert np.isclose(float(vertices[:, 0].max()), 2.0)
    assert np.isclose(float(vertices[:, 1].min()), 0.0)
    assert np.isclose(float(vertices[:, 1].max()), 2.0)


def _box_stl(tmp_path, name, extents, translation):
    path = tmp_path / f"{name}.stl"
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(translation)
    mesh.export(path)
    return path
