from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class StlStats:
    bounds: NDArray[np.float64]
    triangle_count: int


def read_stl_stats(path: str | Path) -> StlStats:
    source = Path(path)
    binary = _read_binary_stl_stats(source)
    if binary is not None:
        return binary

    from capp.geometry.mesh import load_mesh

    mesh = load_mesh(source)
    return StlStats(bounds=mesh.bounds, triangle_count=int(mesh.faces.shape[0]))


def estimate_spacing_from_bounds(
    bounds: NDArray[np.float64],
    target_voxels: int = 4_000_000,
) -> float:
    intervals = np.maximum(bounds[1] - bounds[0], 0.0)
    volume = float(intervals[0] * intervals[1] * intervals[2])
    if volume <= 0.0:
        return 0.5
    return max(round((volume / float(target_voxels)) ** (1.0 / 3.0), 2), 0.01)


def _read_binary_stl_stats(path: Path) -> StlStats | None:
    size = path.stat().st_size
    if size < 84:
        return None

    with path.open("rb") as handle:
        handle.seek(80)
        triangle_count = int(np.frombuffer(handle.read(4), dtype="<u4")[0])

    expected_size = 84 + triangle_count * 50
    if expected_size != size:
        return None

    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=84, shape=(triangle_count, 50))
    floats = raw[:, :48].reshape(-1).view("<f4").reshape(triangle_count, 12)
    vertices = np.asarray(floats[:, 3:12].reshape(-1, 3), dtype=np.float64)
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)], dtype=np.float64)
    return StlStats(bounds=bounds, triangle_count=triangle_count)
