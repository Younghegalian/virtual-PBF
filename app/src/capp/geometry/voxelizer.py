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
    build_plate_z = support_origin[2] + base_z * spacing
    bed_contact_z = _bed_contact_z(mesh, build_plate_z, spacing)
    part_contacts_bed = bed_contact_z > build_plate_z + 1e-9
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
    total = len(triangles)
    for index, triangle in enumerate(triangles):
        if index % max(1, total // 20) == 0:
            _report_progress(
                progress_callback,
                25 + int(index * 35 / total),
                "Rasterizing overhang footprint",
            )
        _mark_triangle_samples(contact, triangle, support_origin, spacing, step)

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
    xy_top = _clear_columns_with_part_below(xy_top, part_mask, base_z)
    support_data = _extrude_support_columns(xy_top, base_z, support_shape)
    support_data &= ~part_mask
    support_data = _keep_build_plate_connected_support(support_data, base_z)
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
    total = len(triangles)
    for index, triangle in enumerate(triangles):
        if index % max(1, total // 25) == 0:
            _report_progress(progress_callback, 5 + int(index * 80 / total), "Rasterizing support")
        _mark_triangle_samples(data, triangle, origin, float(spacing), step)

    iterations = max(1, int(np.ceil(shell_thickness / float(spacing))) - 1)
    _report_progress(progress_callback, 88, "Thickening line support")
    if iterations > 0:
        data = ndimage.binary_dilation(data, structure=np.ones((3, 3, 3), dtype=bool), iterations=iterations)
    _report_progress(progress_callback, 100, "Line support voxelization complete")
    return VoxelGrid(data=data, spacing=float(spacing), origin=tuple(float(value) for value in origin))


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
    build_plate_z = part_origin[2] if options.build_plate_z is None else float(options.build_plate_z)
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
    x_index, y_index = np.indices(top_z.shape)
    footprint = top_z >= 0

    if mode == "x surface support":
        thickness_cells = 1
        pattern = (
            ((x_index + y_index) % pitch_cells) < thickness_cells
        ) | (((x_index - y_index) % pitch_cells) < thickness_cells)
        filtered = np.where(footprint & pattern, top_z, -1)
        if np.any(filtered >= 0):
            return filtered
        return top_z

    if mode == "column support":
        anchor = footprint & ((x_index % pitch_cells) == 0) & ((y_index % pitch_cells) == 0)
        if not np.any(anchor):
            anchor = footprint
        filtered = np.where(anchor, top_z, -1)
        radius = max(0, int(np.ceil((float(options.thickness) * 0.5) / spacing)) - 1)
        if radius > 0:
            filtered = ndimage.grey_dilation(
                filtered,
                size=(radius * 2 + 1, radius * 2 + 1),
                mode="constant",
                cval=-1,
            )
            filtered = np.where(footprint & (filtered >= 0), filtered, -1)
        return filtered

    return top_z


def _clear_columns_with_part_below(
    top_z: np.ndarray,
    part_mask: np.ndarray,
    base_z: int,
) -> np.ndarray:
    filtered = top_z.copy()
    for x_index, y_index in np.argwhere(filtered >= 0):
        upper = int(filtered[x_index, y_index])
        if upper <= base_z:
            filtered[x_index, y_index] = -1
            continue
        if np.any(part_mask[x_index, y_index, base_z:upper]):
            filtered[x_index, y_index] = -1
    return filtered


def _extrude_support_columns(
    top_z: np.ndarray,
    base_z: int,
    shape: tuple[int, int, int],
) -> np.ndarray:
    support = np.zeros(shape, dtype=bool)
    filled_positions = np.argwhere(top_z >= 0)
    for x_index, y_index in filled_positions:
        upper = int(top_z[x_index, y_index])
        if upper > base_z:
            support[x_index, y_index, base_z:upper] = True
    return support


def _keep_build_plate_connected_support(support: np.ndarray, base_z: int) -> np.ndarray:
    connected = np.zeros_like(support, dtype=bool)
    if base_z < 0 or base_z >= support.shape[2]:
        return connected

    for x_index, y_index in np.argwhere(support[:, :, base_z]):
        column = support[x_index, y_index, base_z:]
        gaps = np.flatnonzero(~column)
        end = base_z + int(gaps[0]) if gaps.size else support.shape[2]
        connected[x_index, y_index, base_z:end] = True
    return connected


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
            points.append(triangle[0] + a * (triangle[1] - triangle[0]) + b * (triangle[2] - triangle[0]))
    indices = np.floor((np.asarray(points, dtype=np.float64) - origin) / spacing).astype(int)
    valid = np.all((indices >= 0) & (indices < np.asarray(data.shape)), axis=1)
    if np.any(valid):
        valid_indices = indices[valid]
        data[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = True


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
