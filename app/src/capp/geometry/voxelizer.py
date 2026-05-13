from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from capp.domain import VoxelGrid

ProgressCallback = Callable[[int, str], None]


def voxelize_mesh(
    path: str | Path,
    spacing: float,
    progress_callback: ProgressCallback | None = None,
) -> VoxelGrid:
    """Voxelize an STL using a VTK C++ image-stencil backend."""
    if spacing <= 0:
        raise ValueError("Voxel spacing must be positive.")

    _report_progress(progress_callback, 0, "Preparing voxelization")
    data = _voxelize_stl_with_vtk(Path(path), spacing, progress_callback)
    _report_progress(progress_callback, 96, "Applying virtual printing padding")
    data = _matlab_virtual_printing_padding(data)
    _report_progress(progress_callback, 100, "Voxelization complete")
    return VoxelGrid(data=data, spacing=float(spacing), origin=(0.0, 0.0, 0.0))


def _voxelize_stl_with_vtk(
    path: Path,
    spacing: float,
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
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
    return flat.reshape((resolution[2], resolution[1], resolution[0])).transpose(2, 1, 0)


def _matlab_virtual_printing_padding(data: np.ndarray) -> np.ndarray:
    return np.pad(data, ((2, 2), (2, 2), (0, 2)), mode="constant", constant_values=False)


def _report_progress(
    progress_callback: ProgressCallback | None,
    percent: int,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(max(0, min(100, int(percent))), message)
