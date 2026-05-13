from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MeshGeometry:
    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]
    source_path: Path | None = None

    @property
    def bounds(self) -> NDArray[np.float64]:
        return np.array([self.vertices.min(axis=0), self.vertices.max(axis=0)])


def load_mesh(path: str | Path) -> MeshGeometry:
    import trimesh

    source_path = Path(path)
    loaded = trimesh.load_mesh(source_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No mesh geometry found in {source_path}.")
        loaded = trimesh.util.concatenate(meshes)

    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type from {source_path}: {type(loaded)!r}")

    return MeshGeometry(
        vertices=np.asarray(loaded.vertices, dtype=np.float64),
        faces=np.asarray(loaded.faces, dtype=np.int64),
        source_path=source_path,
    )

