from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy import ndimage

from capp.domain import VoxelGrid

ProgressCallback = Callable[[int, str], None]
VOXEL_LOWER_PADDING_CELLS = (2, 2, 0)
VOXEL_UPPER_PADDING_CELLS = (2, 2, 2)


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
    progress_callback: ProgressCallback | None = None,
) -> VoxelGrid:
    if support_path is None:
        return voxelize_mesh(part_path, spacing, progress_callback=progress_callback)

    def part_progress(percent: int, message: str) -> None:
        _report_progress(progress_callback, int(percent * 0.48), f"Part: {message}")

    def support_progress(percent: int, message: str) -> None:
        _report_progress(progress_callback, 50 + int(percent * 0.45), f"Support: {message}")

    part_grid = voxelize_mesh(part_path, spacing, progress_callback=part_progress)
    support_grid = voxelize_support_mesh(
        support_path,
        spacing,
        support_type,
        progress_callback=support_progress,
    )
    _report_progress(progress_callback, 96, "Combining part and support voxels")
    return union_voxel_grids([part_grid, support_grid])


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
    for grid in grids:
        offset = np.rint((np.asarray(grid.origin, dtype=np.float64) - origin) / spacing).astype(int)
        slices = tuple(slice(offset[axis], offset[axis] + grid.shape[axis]) for axis in range(3))
        data[slices] |= grid.data
    return VoxelGrid(data=data, spacing=spacing, origin=tuple(float(v) for v in origin))


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
