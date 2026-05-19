from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from skimage import measure

from capp.domain import SimulationResult


def save_npz(path: str | Path, result: SimulationResult) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_geometry = "" if result.source_geometry is None else str(result.source_geometry)
    np.savez_compressed(
        output_path,
        probability=result.probability,
        binary=result.binary,
        voxel=result.voxel,
        spacing=np.array([result.spacing], dtype=np.float64),
        origin=np.array(result.origin, dtype=np.float64),
        rest_volume=np.array([result.rest_volume], dtype=np.float64),
        probability_density=np.array([result.probability_density], dtype=np.float64),
        source_geometry=np.array([source_geometry]),
    )


def write_vtk_volume(
    path: str | Path,
    volume: NDArray,
    spacing: float,
    origin: tuple[float, float, float],
    scalar_name: str,
    *,
    binary: bool = True,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError("VTK volume export requires a 3D array.")

    vtk_type = "unsigned_char" if array.dtype == np.uint8 or array.dtype == bool else "float"
    flat = array.astype(np.uint8 if vtk_type == "unsigned_char" else np.float32).ravel(order="F")
    mode = "BINARY" if binary else "ASCII"

    header = (
        "# vtk DataFile Version 3.0\n"
        "virtual PBF volume\n"
        f"{mode}\n"
        "DATASET STRUCTURED_POINTS\n"
        f"DIMENSIONS {array.shape[0]} {array.shape[1]} {array.shape[2]}\n"
        f"ORIGIN {origin[0]} {origin[1]} {origin[2]}\n"
        f"SPACING {spacing} {spacing} {spacing}\n"
        f"POINT_DATA {array.size}\n"
        f"SCALARS {scalar_name} {vtk_type} 1\n"
        "LOOKUP_TABLE default\n"
    )
    if binary:
        data = flat if vtk_type == "unsigned_char" else flat.astype(">f4", copy=False)
        with output_path.open("wb") as handle:
            handle.write(header.encode("ascii"))
            handle.write(np.ascontiguousarray(data).tobytes())
            handle.write(b"\n")
        return

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        for index, value in enumerate(flat, start=1):
            handle.write(f"{value.item()} ")
            if index % 12 == 0:
                handle.write("\n")
        handle.write("\n")


def write_binary_stl(
    path: str | Path,
    volume: NDArray,
    spacing: float,
    origin: tuple[float, float, float],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(volume, dtype=bool)
    if array.ndim != 3:
        raise ValueError("STL volume export requires a 3D array.")
    padded = np.pad(array.astype(np.float32), 1, mode="constant", constant_values=0.0)
    if float(padded.max()) <= 0.0:
        _write_empty_stl(output_path)
        return
    vertices, faces, _normals, _values = measure.marching_cubes(
        padded,
        level=0.5,
        spacing=(float(spacing), float(spacing), float(spacing)),
    )
    vertices = vertices - float(spacing) + np.asarray(origin, dtype=np.float32)
    _write_binary_stl(output_path, vertices, faces)


def _write_empty_stl(path: Path) -> None:
    with path.open("wb") as handle:
        handle.write(b"virtual PBF empty STL".ljust(80, b"\0"))
        handle.write(struct.pack("<I", 0))


def _write_binary_stl(path: Path, vertices: NDArray, faces: NDArray) -> None:
    triangles = np.asarray(vertices, dtype=np.float32)[np.asarray(faces, dtype=np.int64)]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    valid = norms > 0.0
    normals[valid] /= norms[valid, None]
    normals[~valid] = 0.0

    records = np.empty(
        len(triangles),
        dtype=[
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute_byte_count", "<u2"),
        ],
    )
    records["normal"] = normals.astype(np.float32, copy=False)
    records["vertices"] = triangles.astype(np.float32, copy=False)
    records["attribute_byte_count"] = 0

    with path.open("wb") as handle:
        handle.write(b"virtual PBF binary STL".ljust(80, b"\0"))
        handle.write(struct.pack("<I", len(triangles)))
        handle.write(records.tobytes())


def write_binary_stl(
    path: str | Path,
    volume: NDArray,
    spacing: float,
    origin: tuple[float, float, float],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(volume, dtype=bool)
    if array.ndim != 3:
        raise ValueError("STL volume export requires a 3D array.")
    padded = np.pad(array.astype(np.float32), 1, mode="constant", constant_values=0.0)
    if float(padded.max()) <= 0.0:
        _write_empty_stl(output_path)
        return
    vertices, faces, _normals, _values = measure.marching_cubes(
        padded,
        level=0.5,
        spacing=(float(spacing), float(spacing), float(spacing)),
    )
    vertices = vertices - float(spacing) + np.asarray(origin, dtype=np.float32)
    _write_binary_stl(output_path, vertices, faces)


def _write_empty_stl(path: Path) -> None:
    with path.open("wb") as handle:
        handle.write(b"virtual PBF empty STL".ljust(80, b"\0"))
        handle.write(struct.pack("<I", 0))


def _write_binary_stl(path: Path, vertices: NDArray, faces: NDArray) -> None:
    triangles = np.asarray(vertices, dtype=np.float32)[np.asarray(faces, dtype=np.int64)]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    valid = norms > 0.0
    normals[valid] /= norms[valid, None]
    normals[~valid] = 0.0

    records = np.empty(
        len(triangles),
        dtype=[
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute_byte_count", "<u2"),
        ],
    )
    records["normal"] = normals.astype(np.float32, copy=False)
    records["vertices"] = triangles.astype(np.float32, copy=False)
    records["attribute_byte_count"] = 0

    with path.open("wb") as handle:
        handle.write(b"virtual PBF binary STL".ljust(80, b"\0"))
        handle.write(struct.pack("<I", len(triangles)))
        handle.write(records.tobytes())
