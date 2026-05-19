from pathlib import Path

import numpy as np

from capp.domain import SimulationResult
from capp.io.exports import save_npz


def test_save_npz_preserves_source_geometry_path(tmp_path: Path):
    geometry = tmp_path / "part.stl"
    result = SimulationResult(
        probability=np.ones((2, 2, 2), dtype=np.uint8),
        binary=np.ones((2, 2, 2), dtype=bool),
        voxel=np.ones((2, 2, 2), dtype=bool),
        spacing=0.5,
        origin=(1.0, 2.0, 3.0),
        rest_volume=0.0,
        probability_density=100.0,
        elapsed_seconds=0.01,
        source_geometry=geometry,
    )

    output = tmp_path / "simulation_result.npz"
    save_npz(output, result)

    with np.load(output, allow_pickle=False) as payload:
        assert str(payload["source_geometry"][0]) == str(geometry)
