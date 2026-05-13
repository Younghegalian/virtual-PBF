from pathlib import Path

import numpy as np
import pytest

from capp.compute.devices import solver_backend_statuses
from capp.config import SimulationConfig
from capp.domain import SolverBackend, SolverParameters, VoxelGrid
from capp.simulation.runner import run_simulation_grid


def _backend_available(backend: SolverBackend) -> bool:
    return any(
        status.backend is backend and status.available for status in solver_backend_statuses()
    )


def _tiny_grid() -> VoxelGrid:
    voxel = np.zeros((8, 8, 6), dtype=bool)
    voxel[2:6, 2:6, 1:5] = True
    return VoxelGrid(voxel, spacing=0.5)


@pytest.mark.skipif(
    not _backend_available(SolverBackend.CPU_NATIVE),
    reason="Native C++ solver module is not available.",
)
def test_cpu_native_solver_runs():
    config = SimulationConfig(
        geometry_path=Path("dummy.stl"),
        output_dir=Path("outputs"),
        voxel_spacing=0.5,
        solver=SolverParameters(backend=SolverBackend.CPU_NATIVE, iteration_bound=4),
    )

    result = run_simulation_grid(_tiny_grid(), config)

    assert result.probability.shape == (8, 8, 6)
    assert result.binary.shape == (8, 8, 6)


@pytest.mark.skipif(
    not _backend_available(SolverBackend.CUDA),
    reason="CUDA solver backend is not available.",
)
def test_cuda_solver_runs():
    config = SimulationConfig(
        geometry_path=Path("dummy.stl"),
        output_dir=Path("outputs"),
        voxel_spacing=0.5,
        solver=SolverParameters(backend=SolverBackend.CUDA, iteration_bound=4),
    )

    result = run_simulation_grid(_tiny_grid(), config)

    assert result.probability.shape == (8, 8, 6)
    assert result.binary.shape == (8, 8, 6)
