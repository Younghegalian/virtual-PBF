from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy import ndimage

from capp.domain import SupportGenerationParameters, VoxelGrid

ProgressCallback = Callable[[int, str], None]
VOXEL_LOWER_PADDING_CELLS = (2, 2, 0)
VOXEL_UPPER_PADDING_CELLS = (2, 2, 2)
BUILD_PLATE_CONTACT_TOLERANCE_MM = 1.0


def voxelize_mesh(
    path: str | Path,
    spacing: float,
    progress_callback: ProgressCallback | None = None,
) -> VoxelGrid:
    """Voxelize an STL using a VTK C++ image-stencil backend."""
    if spacing <= 0:
        raise ValueError("Voxel spacing must be positive.")

    _report_progress(progress_callback, 0, "Preparing voxelization")
    data, source_min = _voxelize_stl_with_vtk(Path(path), spacing, progress_callback)
    _report_progress(progress_callback, 96, "Applying virtual printing padding")
    data = _matlab_virtual_printing_padding(data)
    origin_array = np.asarray(source_min, dtype=np.float64) - (
        np.asarray(VOXEL_LOWER_PADDING_CELLS, dtype=np.float64) * float(spacing)
    )
    _report_progress(progress_callback, 100, "Voxelization complete")
    return VoxelGrid(
        data=data,
        spacing=float(spacing),
        origin=tuple(float(value) for value in origin_array),
    )


def voxelize_support_mesh(
    path: str | Path,
    spacing: float,
    support_type: str = "Volume support",
    progress_callback: ProgressCallback | None = None,
) -> VoxelGrid:
    mode = support_type.strip().lower()
    if mode == "volume support":
        return voxelize_mesh(path, spacing, progress_callback=progress_callback)
    if mode == "line support":
        return voxelize_surface_shell(path, spacing, progress_callback=progress_callback)
    raise ValueError(f"Unsupported support type: {support_type}")


def voxelize_part_and_support(
    part_path: str | Path,
    support_path: str | Path | None,
    spacing: float,
    support_type: str = "Volume support",
    support_generation: SupportGenerationParameters | None = None,
    progress_callback: ProgressCallback | None = None,
) -> VoxelGrid:
    if support_path is None and support_generation is None:
        return voxelize_mesh(part_path, spacing, progress_callback=progress_callback)

    def part_progress(percent: int, message: str) -> None:
        _report_progress(progress_callback, int(percent * 0.42), f"Part: {message}")

    def support_progress(percent: int, message: str) -> None:
        _report_progress(progress_callback, 72 + int(percent * 0.23), f"Support: {message}")

    def generated_progress(percent: int, message: str) -> None:
        _report_progress(
            progress_callback,
            44 + int(percent * 0.25),
            f"Generated support: {message}",
        )

    part_grid = voxelize_mesh(part_path, spacing, progress_callback=part_progress)
    grids = [part_grid]
    if support_generation is not None:
        generated_grid = generate_overhang_support_grid(
            part_path,
            part_grid,
            support_generation,
            progress_callback=generated_progress,
        )
        if generated_grid.filled_count > 0:
            grids.append(generated_grid)
    if support_path is not None:
        support_grid = _support_labeled_grid(
            voxelize_support_mesh(
                support_path,
                spacing,
                support_type,
                progress_callback=support_progress,
            )
        )
        grids.append(support_grid)
    _report_progress(progress_callback, 96, "Combining part and support voxels")
    combined = union_voxel_grids(grids)
    part_mask = _grid_data_in_frame(part_grid, combined.origin, combined.shape, combined.spacing)
    support_mask = combined.support_mask & ~part_mask
    return VoxelGrid(
        data=combined.data,
        spacing=combined.spacing,
        origin=combined.origin,
        support_mask=support_mask,
    )


def voxelize_part_with_support_grid(
    part_path: str | Path,
    support_grid: VoxelGrid,
    spacing: float,
    progress_callback: ProgressCallback | None = None,
) -> VoxelGrid:
    def part_progress(percent: int, message: str) -> None:
        _report_progress(progress_callback, int(percent * 0.82), f"Part: {message}")

    part_grid = voxelize_mesh(part_path, spacing, progress_callback=part_progress)
    if not np.isclose(float(support_grid.spacing), float(spacing)):
        raise ValueError("Cached support grid spacing does not match voxel spacing.")

    _report_progress(progress_callback, 88, "Using cached generated support grid")
    combined = union_voxel_grids([part_grid, _support_labeled_grid(support_grid)])
    part_mask = _grid_data_in_frame(part_grid, combined.origin, combined.shape, combined.spacing)
    support_mask = combined.support_mask & ~part_mask
    _report_progress(progress_callback, 96, "Combining part and cached support voxels")
    return VoxelGrid(
        data=combined.data,
        spacing=combined.spacing,
        origin=combined.origin,
        support_mask=support_mask,
    )


def generate_overhang_support_grid(
    part_path: str | Path,
    part_grid: VoxelGrid,
    options: SupportGenerationParameters | None = None,
    progress_callback: ProgressCallback | None = None,
) -> VoxelGrid:
    params = options or SupportGenerationParameters()
    mode = params.support_type.strip().lower()
    if mode not in {"volume support", "column support", "x surface support"}:
        raise ValueError(f"Unsupported generated support type: {params.support_type}")

    _report_progress(progress_callback, 0, "Reading source mesh")
    mesh = _load_trimesh(Path(part_path))
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) == 0:
        return VoxelGrid(
            data=np.zeros_like(part_grid.data, dtype=bool),
            spacing=part_grid.spacing,
            origin=part_grid.origin,
        )

    support_shape, support_origin, part_z_offset = _support_grid_frame(part_grid, params)
    base_z = _build_plate_index(support_origin, float(part_grid.spacing), params)
    spacing = float(part_grid.spacing)
    requested_build_plate_z = (
        float(part_grid.origin[2]) if params.build_plate_z is None else float(params.build_plate_z)
    )
    bed_contact_z = _bed_contact_z(mesh, requested_build_plate_z, spacing)
    part_contacts_bed = bed_contact_z > requested_build_plate_z + 1e-9
    if part_contacts_bed:
        base_z = max(base_z, _z_index_at_or_above(support_origin, spacing, bed_contact_z))
    _report_progress(progress_callback, 12, "Finding overhang faces")
    triangles = _overhang_triangles(
        mesh,
        params.overhang_angle,
        build_plate_z=bed_contact_z,
        spacing=spacing,
    )
    support_data = np.zeros(support_shape, dtype=bool)
    part_mask = np.zeros(support_shape, dtype=bool)
    part_mask[:, :, part_z_offset : part_z_offset + part_grid.shape[2]] = part_grid.data
    if triangles.size == 0:
        _report_progress(progress_callback, 100, "No overhang faces found")
        return VoxelGrid(
            data=support_data,
            spacing=part_grid.spacing,
            origin=tuple(float(value) for value in support_origin),
        )

    _report_progress(progress_callback, 25, "Rasterizing overhang footprint")
    contact = np.zeros(support_shape, dtype=bool)
    step = max(float(part_grid.spacing) * 0.5, 1e-9)
    _mark_triangles_in_grid(
        contact,
        triangles,
        support_origin,
        spacing,
        step,
        progress_callback=progress_callback,
        progress_start=25,
        progress_end=60,
        message="Rasterizing overhang footprint",
    )

    contact &= part_mask
    contact &= _unsupported_part_voxels(
        part_mask,
        base_z,
        support_origin,
        spacing,
        params.overhang_angle,
        bed_contact_z,
        part_contacts_bed=part_contacts_bed,
    )
    top_z = _overhang_top_indices(contact, params.footprint_offset, spacing)
    top_z = _clear_bed_contact_top_indices(
        top_z,
        support_origin,
        spacing,
        bed_contact_z,
        part_contacts_bed=part_contacts_bed,
    )
    if not np.any(top_z >= 0):
        _report_progress(progress_callback, 100, "No supportable overhang footprint found")
        return VoxelGrid(
            data=support_data,
            spacing=part_grid.spacing,
            origin=tuple(float(value) for value in support_origin),
        )

    _report_progress(progress_callback, 66, "Building support lattice")
    xy_top = _filter_support_footprint(top_z, params, spacing)
    xy_start = _support_start_indices(xy_top, part_mask, base_z)
    xy_top_with_contact = _apply_support_contact_depth(
        xy_top,
        params.contact_depth,
        spacing,
        support_shape[2],
    )
    support_data = _extrude_support_spans(xy_start, xy_top_with_contact, support_shape)
    if params.contact_depth <= 0:
        support_data &= ~part_mask
    _report_progress(progress_callback, 100, "Generated support voxelization complete")
    return VoxelGrid(
        data=support_data,
        spacing=part_grid.spacing,
        origin=tuple(float(value) for value in support_origin),
        support_mask=support_data,
    )


def union_voxel_grids(grids: list[VoxelGrid] | tuple[VoxelGrid, ...]) -> VoxelGrid:
    if not grids:
        raise ValueError("At least one voxel grid is required.")
    spacing = float(grids[0].spacing)
    for grid in grids:
        if not np.isclose(float(grid.spacing), spacing):
            raise ValueError("Voxel grids must use the same spacing to be combined.")

    origins = np.asarray([grid.origin for grid in grids], dtype=np.float64)
    bounds_hi = np.asarray(
        [
            np.asarray(grid.origin, dtype=np.float64)
            + np.asarray(grid.shape, dtype=np.float64) * spacing
            for grid in grids
        ],
        dtype=np.float64,
    )
    origin = origins.min(axis=0)
    max_bound = bounds_hi.max(axis=0)
    shape = np.maximum(np.ceil((max_bound - origin) / spacing).astype(int), 1)
    data = np.zeros(tuple(int(value) for value in shape), dtype=bool)
    support_mask = np.zeros_like(data, dtype=bool)
    for grid in grids:
        offset = np.rint((np.asarray(grid.origin, dtype=np.float64) - origin) / spacing).astype(int)
        slices = tuple(slice(offset[axis], offset[axis] + grid.shape[axis]) for axis in range(3))
        data[slices] |= grid.data
        support_mask[slices] |= grid.support_mask
    support_mask &= data
    return VoxelGrid(
        data=data,
        spacing=spacing,
        origin=tuple(float(v) for v in origin),
        support_mask=support_mask,
    )


def _grid_data_in_frame(
    grid: VoxelGrid,
    origin: tuple[float, float, float],
    shape: tuple[int, int, int],
    spacing: float,
) -> np.ndarray:
    data = np.zeros(shape, dtype=bool)
    offset = np.rint(
        (np.asarray(grid.origin, dtype=np.float64) - np.asarray(origin, dtype=np.float64))
        / float(spacing)
    ).astype(int)
    slices = tuple(slice(offset[axis], offset[axis] + grid.shape[axis]) for axis in range(3))
    data[slices] = grid.data
    return data


def voxelize_surface_shell(
    path: str | Path,
    spacing: float,
    thickness: float | None = None,
    progress_callback: ProgressCallback | None = None,
) -> VoxelGrid:
    if spacing <= 0:
        raise ValueError("Voxel spacing must be positive.")
    shell_thickness = max(float(spacing if thickness is None else thickness), float(spacing))
    _report_progress(progress_callback, 0, "Reading line support mesh")
    mesh = _load_trimesh(Path(path))
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    if not np.all(np.isfinite(bounds)):
        raise ValueError(f"Invalid mesh bounds for {path}.")

    lower_padding = np.asarray(VOXEL_LOWER_PADDING_CELLS, dtype=np.float64)
    upper_padding = np.asarray(VOXEL_UPPER_PADDING_CELLS, dtype=np.float64)
    origin = bounds[0] - lower_padding * float(spacing)
    max_bound = bounds[1] + upper_padding * float(spacing)
    shape = np.maximum(np.ceil((max_bound - origin) / float(spacing)).astype(int), 1)
    data = np.zeros(tuple(int(value) for value in shape), dtype=bool)
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    if triangles.size == 0:
        raise ValueError(f"No support triangles found in {path}.")

    step = max(float(spacing) * 0.5, 1e-9)
    _mark_triangles_in_grid(
        data,
        triangles,
        origin,
        float(spacing),
        step,
        progress_callback=progress_callback,
        progress_start=5,
        progress_end=85,
        message="Rasterizing support",
    )

    iterations = max(0, int(np.ceil(shell_thickness / float(spacing))) - 1)
    if iterations > 0:
        _report_progress(progress_callback, 88, "Thickening line support")
        data = ndimage.binary_dilation(
            data,
            structure=np.ones((3, 3, 3), dtype=bool),
            iterations=iterations,
        )
    else:
        _report_progress(progress_callback, 88, "Keeping line support as a thin surface")
    _report_progress(progress_callback, 100, "Line support voxelization complete")
    return VoxelGrid(
        data=data,
        spacing=float(spacing),
        origin=tuple(float(value) for value in origin),
    )


def _support_labeled_grid(grid: VoxelGrid) -> VoxelGrid:
    return VoxelGrid(
        data=grid.data,
        spacing=grid.spacing,
        origin=grid.origin,
        support_mask=grid.data,
    )


def _overhang_triangles(
    mesh,
    overhang_angle: float,
    *,
    build_plate_z: float,
    spacing: float,
) -> np.ndarray:
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    normal_norm = np.linalg.norm(normals, axis=1)
    normal_norm[normal_norm == 0.0] = 1.0
    cosine = np.clip(normals[:, 2] / normal_norm, -1.0, 1.0)
    angles = np.abs(np.degrees(np.arccos(cosine)) - 180.0)
    overhang = angles <= float(overhang_angle)
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    above_build_plate = (
        triangles[:, :, 2].min(axis=1) > float(build_plate_z) + float(spacing) * 0.5
    )
    return triangles[overhang & above_build_plate]


def _support_grid_frame(
    part_grid: VoxelGrid,
    options: SupportGenerationParameters,
) -> tuple[tuple[int, int, int], np.ndarray, int]:
    spacing = float(part_grid.spacing)
    part_origin = np.asarray(part_grid.origin, dtype=np.float64)
    build_plate_z = (
        part_origin[2]
        if options.build_plate_z is None
        else float(options.build_plate_z)
    )
    support_origin = part_origin.copy()
    if build_plate_z < part_origin[2]:
        lower_cells = int(np.ceil((part_origin[2] - build_plate_z) / spacing))
        support_origin[2] = part_origin[2] - lower_cells * spacing
    part_z_offset = int(round((part_origin[2] - support_origin[2]) / spacing))
    shape = (
        int(part_grid.shape[0]),
        int(part_grid.shape[1]),
        int(part_grid.shape[2] + part_z_offset),
    )
    return shape, support_origin, part_z_offset


def _build_plate_index(
    support_origin: np.ndarray,
    spacing: float,
    options: SupportGenerationParameters,
) -> int:
    if options.build_plate_z is None:
        return 0
    return max(0, int(np.floor((float(options.build_plate_z) - support_origin[2]) / spacing)))


def _z_index_at_or_above(
    origin: np.ndarray,
    spacing: float,
    z_value: float,
) -> int:
    return max(0, int(np.ceil((float(z_value) - float(origin[2])) / float(spacing) - 1e-9)))


def _bed_contact_z(mesh, build_plate_z: float, spacing: float) -> float:
    mesh_min_z = float(np.asarray(mesh.bounds, dtype=np.float64)[0, 2])
    tolerance = _bed_contact_tolerance(spacing)
    if mesh_min_z >= float(build_plate_z) and mesh_min_z - float(build_plate_z) <= tolerance:
        return mesh_min_z
    return float(build_plate_z)


def _bed_contact_tolerance(spacing: float) -> float:
    return max(float(spacing), BUILD_PLATE_CONTACT_TOLERANCE_MM)


def _overhang_top_indices(
    contact: np.ndarray,
    footprint_offset: float,
    spacing: float,
) -> np.ndarray:
    contact_positions = np.where(contact)
    top_z = np.full(contact.shape[:2], -1, dtype=np.int32)
    if len(contact_positions[0]) == 0:
        return top_z
    np.maximum.at(top_z, (contact_positions[0], contact_positions[1]), contact_positions[2])
    iterations = int(np.ceil(float(footprint_offset) / spacing))
    if iterations > 0:
        top_z = ndimage.grey_dilation(
            top_z,
            size=(iterations * 2 + 1, iterations * 2 + 1),
            mode="constant",
            cval=-1,
        )
    return top_z


def _clear_bed_contact_top_indices(
    top_z: np.ndarray,
    support_origin: np.ndarray,
    spacing: float,
    bed_contact_z: float,
    *,
    part_contacts_bed: bool,
) -> np.ndarray:
    filtered = top_z.copy()
    valid = filtered >= 0
    if not np.any(valid):
        return filtered
    top_world_z = support_origin[2] + filtered.astype(np.float64) * float(spacing)
    minimum_support_height = _bed_contact_exclusion_height(spacing, part_contacts_bed)
    filtered[valid & (top_world_z <= float(bed_contact_z) + minimum_support_height)] = -1
    return filtered


def _unsupported_part_voxels(
    part_mask: np.ndarray,
    base_z: int,
    support_origin: np.ndarray,
    spacing: float,
    overhang_angle: float,
    bed_contact_z: float,
    *,
    part_contacts_bed: bool,
) -> np.ndarray:
    unsupported = np.zeros_like(part_mask, dtype=bool)
    radius = _self_support_radius_cells(overhang_angle)
    if radius <= 0:
        footprint = np.ones((1, 1), dtype=bool)
    else:
        footprint = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=bool)
    bed_contact_layer = int(
        np.floor((float(bed_contact_z) - float(support_origin[2])) / float(spacing))
    )
    supported_until = max(
        int(base_z),
        bed_contact_layer
        + int(np.ceil(_bed_contact_exclusion_height(spacing, part_contacts_bed) / spacing)),
    )
    for z_index in range(max(1, supported_until + 1), part_mask.shape[2]):
        lower = part_mask[:, :, z_index - 1]
        lower_support = ndimage.binary_dilation(lower, structure=footprint)
        unsupported[:, :, z_index] = part_mask[:, :, z_index] & ~lower_support
    return unsupported


def _self_support_radius_cells(overhang_angle: float) -> int:
    angle = float(np.clip(overhang_angle, 1.0, 89.0))
    return max(0, int(np.ceil(1.0 / np.tan(np.radians(angle)))))


def _bed_contact_exclusion_height(spacing: float, part_contacts_bed: bool) -> float:
    if part_contacts_bed:
        return max(float(spacing) * 8.0, BUILD_PLATE_CONTACT_TOLERANCE_MM * 5.0)
    return _bed_contact_tolerance(spacing)


def _filter_support_footprint(
    top_z: np.ndarray,
    options: SupportGenerationParameters,
    spacing: float,
) -> np.ndarray:
    mode = options.support_type.strip().lower()
    if mode == "volume support":
        return top_z

    pitch_cells = max(1, int(round(float(options.pitch) / spacing)))
    footprint = top_z >= 0

    if mode == "x surface support":
        pattern = _x_surface_lattice_pattern(footprint, pitch_cells)
        filtered = np.where(footprint & pattern, top_z, -1)
        if np.any(filtered >= 0):
            return filtered
        return top_z

    if mode == "column support":
        anchor = _column_lattice_pattern(footprint, pitch_cells)
        filtered = np.where(anchor, top_z, -1)
        radius = max(0, int(np.floor((float(options.thickness) * 0.5) / spacing)))
        if radius > 0:
            filtered = ndimage.grey_dilation(
                filtered,
                footprint=_disk_footprint(radius),
                mode="constant",
                cval=-1,
            )
            filtered = np.where(footprint & (filtered >= 0), filtered, -1)
        return filtered

    return top_z


def _apply_support_contact_depth(
    top_z: np.ndarray,
    contact_depth: float,
    spacing: float,
    z_size: int,
) -> np.ndarray:
    if contact_depth <= 0:
        return top_z
    contact_cells = max(1, int(np.ceil(float(contact_depth) / float(spacing))))
    extended = top_z.copy()
    valid = extended >= 0
    extended[valid] = np.minimum(extended[valid] + contact_cells, int(z_size))
    return extended


def _x_surface_lattice_pattern(footprint: np.ndarray, pitch_cells: int) -> np.ndarray:
    footprint = np.asarray(footprint, dtype=bool)
    pattern = np.zeros_like(footprint, dtype=bool)
    labels, count = ndimage.label(footprint)
    if count == 0:
        return pattern

    period = max(1, int(round(float(pitch_cells) * np.sqrt(2.0))))
    x_index, y_index = np.indices(footprint.shape)
    for label in range(1, count + 1):
        component = labels == label
        coords = np.argwhere(component)
        if coords.size == 0:
            continue
        center = np.rint((coords.min(axis=0) + coords.max(axis=0)) * 0.5).astype(np.int32)
        positive = (x_index + y_index) - int(center[0] + center[1])
        negative = (x_index - y_index) - int(center[0] - center[1])
        component_pattern = _on_centered_lattice(positive, period) | _on_centered_lattice(
            negative,
            period,
        )
        pattern |= component & component_pattern
    return pattern


def _column_lattice_pattern(footprint: np.ndarray, pitch_cells: int) -> np.ndarray:
    footprint = np.asarray(footprint, dtype=bool)
    anchors = np.zeros_like(footprint, dtype=bool)
    labels, count = ndimage.label(footprint)
    if count == 0:
        return anchors

    for label in range(1, count + 1):
        component = labels == label
        coords = np.argwhere(component)
        if coords.size == 0:
            continue
        lower = coords.min(axis=0)
        upper = coords.max(axis=0)
        x_positions = _centered_lattice_positions(int(lower[0]), int(upper[0]), pitch_cells)
        y_positions = _centered_lattice_positions(int(lower[1]), int(upper[1]), pitch_cells)
        local = np.zeros_like(footprint, dtype=bool)
        local[np.ix_(x_positions, y_positions)] = True
        local &= component
        if not np.any(local):
            center = (lower + upper) * 0.5
            nearest = coords[np.argmin(np.sum((coords - center) ** 2, axis=1))]
            local[int(nearest[0]), int(nearest[1])] = True
        anchors |= local
    return anchors


def _centered_lattice_positions(lower: int, upper: int, pitch_cells: int) -> np.ndarray:
    if upper < lower:
        return np.asarray([], dtype=np.int32)
    if pitch_cells <= 1:
        return np.arange(lower, upper + 1, dtype=np.int32)

    span = int(upper - lower)
    count = max(1, int(np.floor(span / pitch_cells)) + 1)
    total = (count - 1) * int(pitch_cells)
    start = int(round((lower + upper - total) * 0.5))
    positions = start + np.arange(count, dtype=np.int32) * int(pitch_cells)
    positions = positions[(positions >= lower) & (positions <= upper)]
    if positions.size == 0:
        positions = np.asarray([int(round((lower + upper) * 0.5))], dtype=np.int32)
    return positions


def _on_centered_lattice(values: np.ndarray, period: int) -> np.ndarray:
    if period <= 1:
        return np.ones_like(values, dtype=bool)
    remainder = np.mod(values, int(period))
    distance = np.minimum(remainder, int(period) - remainder)
    return distance == 0


def _disk_footprint(radius: int) -> np.ndarray:
    axis = np.arange(-int(radius), int(radius) + 1)
    x_grid, y_grid = np.meshgrid(axis, axis, indexing="ij")
    return (x_grid**2 + y_grid**2) <= int(radius) ** 2


def _support_start_indices(
    top_z: np.ndarray,
    part_mask: np.ndarray,
    base_z: int,
) -> np.ndarray:
    start_z = np.full(top_z.shape, int(base_z), dtype=np.int32)
    valid = top_z > int(base_z)
    if not np.any(valid):
        start_z[:] = -1
        return start_z

    z_index = np.arange(part_mask.shape[2], dtype=np.int32)
    below_top = z_index[None, None, :] < top_z[:, :, None]
    part_below = part_mask & below_top
    nearest_part_below = np.where(part_below, z_index[None, None, :], -1).max(axis=2)
    anchored_to_part = valid & (nearest_part_below >= int(base_z))
    start_z[anchored_to_part] = nearest_part_below[anchored_to_part] + 1
    start_z[~valid | (start_z >= top_z)] = -1
    return start_z


def _extrude_support_spans(
    start_z: np.ndarray,
    top_z: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    z_index = np.arange(shape[2], dtype=np.int32)
    return (
        (top_z[:, :, None] >= 0)
        & (start_z[:, :, None] >= 0)
        & (z_index[None, None, :] >= start_z[:, :, None])
        & (z_index[None, None, :] < top_z[:, :, None])
    )


def _load_trimesh(path: Path):
    import trimesh

    loaded = trimesh.load_mesh(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geom
            for geom in loaded.geometry.values()
            if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0
        ]
        if not meshes:
            raise ValueError(f"No mesh geometry found in {path}.")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported support mesh type from {path}: {type(loaded)!r}")
    return loaded


def _mark_triangles_in_grid(
    data: np.ndarray,
    triangles: np.ndarray,
    origin: np.ndarray,
    spacing: float,
    step: float,
    *,
    progress_callback: ProgressCallback | None = None,
    progress_start: int = 0,
    progress_end: int = 100,
    message: str = "Rasterizing support",
) -> None:
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.size == 0:
        return

    total = int(len(triangles))
    _report_progress(progress_callback, progress_start, message)
    progress_interval = max(1, total // 25)
    for index, triangle in enumerate(triangles):
        if index % progress_interval == 0:
            progress = progress_start + int((index / total) * (progress_end - progress_start))
            _report_progress(progress_callback, progress, message)
        try:
            _mark_projected_triangle(data, triangle, origin, float(spacing), float(step))
        except Exception:
            _mark_triangle_samples(data, triangle, origin, float(spacing), float(step))
    _report_progress(progress_callback, progress_end, message)


def _mark_projected_triangle(
    data: np.ndarray,
    triangle: np.ndarray,
    origin: np.ndarray,
    spacing: float,
    step: float,
) -> None:
    triangle = np.asarray(triangle, dtype=np.float64)
    _mark_points_in_grid(data, triangle, origin, spacing)

    grid_triangle = (triangle - origin) / float(spacing)
    normal = np.cross(grid_triangle[1] - grid_triangle[0], grid_triangle[2] - grid_triangle[0])
    if float(np.linalg.norm(normal)) <= 1e-12:
        _mark_triangle_edges(data, triangle, origin, spacing, step)
        return

    drop_axis = int(np.argmax(np.abs(normal)))
    axes = [axis for axis in range(3) if axis != drop_axis]
    projected = grid_triangle[:, axes]
    lower = np.floor(projected.min(axis=0)).astype(np.int64) - 1
    upper = np.floor(projected.max(axis=0)).astype(np.int64) + 1
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, np.asarray(data.shape, dtype=np.int64)[axes] - 1)
    if np.any(upper < lower):
        return

    u_index = np.arange(lower[0], upper[0] + 1, dtype=np.int64)
    v_index = np.arange(lower[1], upper[1] + 1, dtype=np.int64)
    if u_index.size == 0 or v_index.size == 0:
        return

    u_grid, v_grid = np.meshgrid(u_index, v_index, indexing="ij")
    p0, p1, p2 = projected
    denominator = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
    if abs(float(denominator)) <= 1e-12:
        _mark_triangle_edges(data, triangle, origin, spacing, step)
        return

    index_chunks = []
    for u_offset, v_offset in (
        (0.5, 0.5),
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
    ):
        u = u_grid.astype(np.float64) + u_offset
        v = v_grid.astype(np.float64) + v_offset
        alpha = (
            (p1[1] - p2[1]) * (u - p2[0]) + (p2[0] - p1[0]) * (v - p2[1])
        ) / denominator
        beta = (
            (p2[1] - p0[1]) * (u - p2[0]) + (p0[0] - p2[0]) * (v - p2[1])
        ) / denominator
        gamma = 1.0 - alpha - beta
        tolerance = 1e-9
        inside = (alpha >= -tolerance) & (beta >= -tolerance) & (gamma >= -tolerance)
        if not np.any(inside):
            continue
        dropped = (
            alpha[inside] * grid_triangle[0, drop_axis]
            + beta[inside] * grid_triangle[1, drop_axis]
            + gamma[inside] * grid_triangle[2, drop_axis]
        )
        chunk = np.zeros((int(np.count_nonzero(inside)), 3), dtype=np.int64)
        chunk[:, axes[0]] = u_grid[inside]
        chunk[:, axes[1]] = v_grid[inside]
        chunk[:, drop_axis] = np.floor(dropped).astype(np.int64)
        index_chunks.append(chunk)
    if not index_chunks:
        _mark_triangle_edges(data, triangle, origin, spacing, step)
        return

    indices = np.vstack(index_chunks)
    valid = np.all((indices >= 0) & (indices < np.asarray(data.shape)), axis=1)
    if np.any(valid):
        valid_indices = indices[valid]
        data[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = True


def _mark_triangle_edges(
    data: np.ndarray,
    triangle: np.ndarray,
    origin: np.ndarray,
    spacing: float,
    step: float,
) -> None:
    points = []
    for start, end in ((0, 1), (1, 2), (2, 0)):
        edge = triangle[end] - triangle[start]
        divisions = max(1, int(np.ceil(np.linalg.norm(edge) / max(float(step), 1e-9))))
        t = np.linspace(0.0, 1.0, divisions + 1, dtype=np.float64)
        points.append(triangle[start] + t[:, None] * edge)
    _mark_points_in_grid(data, np.vstack(points), origin, spacing)


def _mark_points_in_grid(
    data: np.ndarray,
    points: np.ndarray,
    origin: np.ndarray,
    spacing: float,
) -> None:
    indices = np.floor((np.asarray(points, dtype=np.float64) - origin) / spacing).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < np.asarray(data.shape)), axis=1)
    if np.any(valid):
        valid_indices = indices[valid]
        data[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = True


def _mark_triangle_samples(
    data: np.ndarray,
    triangle: np.ndarray,
    origin: np.ndarray,
    spacing: float,
    step: float,
) -> None:
    edges = (
        np.linalg.norm(triangle[1] - triangle[0]),
        np.linalg.norm(triangle[2] - triangle[0]),
        np.linalg.norm(triangle[2] - triangle[1]),
    )
    divisions = max(1, int(np.ceil(max(edges) / step)))
    points = []
    for i in range(divisions + 1):
        for j in range(divisions + 1 - i):
            a = i / divisions
            b = j / divisions
            points.append(
                triangle[0]
                + a * (triangle[1] - triangle[0])
                + b * (triangle[2] - triangle[0])
            )
    _mark_points_in_grid(data, np.asarray(points, dtype=np.float64), origin, spacing)


def _voxelize_stl_with_vtk(
    path: Path,
    spacing: float,
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    last_progress = -1

    def report(percent: int, message: str) -> None:
        nonlocal last_progress
        percent = max(0, min(100, int(percent)))
        if percent == last_progress:
            return
        last_progress = percent
        _report_progress(progress_callback, percent, message)

    observers = []

    def observe(algorithm, start: int, end: int, message: str) -> None:
        def on_progress(caller, _event) -> None:
            value = start + int((end - start) * float(caller.GetProgress()))
            report(value, message)

        observers.append(on_progress)
        algorithm.AddObserver(vtk.vtkCommand.ProgressEvent, on_progress)

    report(2, "Reading STL")
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(path))
    observe(reader, 2, 15, "Reading STL")
    reader.Update()
    poly = reader.GetOutput()
    if poly.GetNumberOfPoints() == 0:
        raise ValueError(f"No mesh geometry found in {path}.")

    report(18, "Normalizing mesh bounds")
    bounds = poly.GetBounds()
    source_min = (float(bounds[0]), float(bounds[2]), float(bounds[4]))
    transform = vtk.vtkTransform()
    transform.Translate(-bounds[0], -bounds[2], -bounds[4])
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(poly)
    transform_filter.SetTransform(transform)
    observe(transform_filter, 18, 26, "Normalizing mesh bounds")
    transform_filter.Update()
    poly = transform_filter.GetOutput()

    report(30, "Allocating voxel image")
    bounds = poly.GetBounds()
    resolution = np.ceil(
        [
            (bounds[1] - bounds[0]) / spacing,
            (bounds[3] - bounds[2]) / spacing,
            (bounds[5] - bounds[4]) / spacing,
        ]
    ).astype(int)
    if np.any(resolution <= 0):
        raise ValueError(f"Invalid voxel resolution for {path}.")

    image = vtk.vtkImageData()
    image.SetSpacing(float(spacing), float(spacing), float(spacing))
    image.SetOrigin(float(spacing) / 2.0, float(spacing) / 2.0, float(spacing) / 2.0)
    image.SetDimensions(int(resolution[0]), int(resolution[1]), int(resolution[2]))
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    image.GetPointData().GetScalars().Fill(1)

    report(38, "Building mesh stencil")
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(poly)
    stencil.SetOutputOrigin(image.GetOrigin())
    stencil.SetOutputSpacing(image.GetSpacing())
    stencil.SetOutputWholeExtent(image.GetExtent())
    observe(stencil, 38, 72, "Building mesh stencil")
    stencil.Update()

    report(74, "Applying mesh stencil")
    image_stencil = vtk.vtkImageStencil()
    image_stencil.SetInputData(image)
    image_stencil.SetStencilConnection(stencil.GetOutputPort())
    image_stencil.ReverseStencilOff()
    image_stencil.SetBackgroundValue(0)
    observe(image_stencil, 74, 92, "Applying mesh stencil")
    image_stencil.Update()

    report(94, "Transferring voxel image")
    scalars = image_stencil.GetOutput().GetPointData().GetScalars()
    flat = vtk_to_numpy(scalars).astype(bool, copy=False)
    data = flat.reshape((resolution[2], resolution[1], resolution[0])).transpose(2, 1, 0)
    return data, source_min


def _matlab_virtual_printing_padding(data: np.ndarray) -> np.ndarray:
    pad_width = tuple(
        (lower, upper)
        for lower, upper in zip(
            VOXEL_LOWER_PADDING_CELLS,
            VOXEL_UPPER_PADDING_CELLS,
            strict=True,
        )
    )
    return np.pad(data, pad_width, mode="constant", constant_values=False)


def _report_progress(
    progress_callback: ProgressCallback | None,
    percent: int,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(max(0, min(100, int(percent))), message)
