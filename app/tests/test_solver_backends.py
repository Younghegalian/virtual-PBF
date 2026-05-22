from pathlib import Path

import numpy as np
import pytest

from capp.compute.devices import solver_backend_statuses
from capp.config import SimulationConfig
from capp.domain import SolverBackend, SolverParameters, StochasticMode, VoxelGrid
from capp.simulation.runner import run_simulation_grid
from capp.solver.native import NativeLayerwiseMarkovSolver
from capp.solver.reference import ReferenceLayerwiseMarkovSolver


def _backend_available(backend: SolverBackend) -> bool:
    return any(
        status.backend is backend and status.available for status in solver_backend_statuses()
    )


def _tiny_grid() -> VoxelGrid:
    voxel = np.zeros((8, 8, 6), dtype=bool)
    voxel[2:6, 2:6, 1:5] = True
    return VoxelGrid(voxel, spacing=0.5)


def _idp_halo_grid(support_only: bool) -> VoxelGrid:
    voxel = np.zeros((5, 5, 5), dtype=bool)
    support_mask = np.zeros_like(voxel)
    voxel[2, 1, 3] = True
    if support_only:
        support_mask[2, 1, 3] = True
    return VoxelGrid(voxel, spacing=1.0, support_mask=support_mask)


def _idp_halo_parameters(backend: SolverBackend) -> SolverParameters:
    return SolverParameters(
        backend=backend,
        iteration_bound=10,
        min_bias=0.5,
        initial_deviation=0.8,
        stochastic_mode=StochasticMode.IN_VOLUME,
        residual_criteria=(0.0, 0.0, 0.0, 0.0),
    )


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


def test_run_simulation_grid_preserves_support_geometry_path():
    support_path = Path("support.stl")
    config = SimulationConfig(
        geometry_path=Path("dummy.stl"),
        output_dir=Path("outputs"),
        voxel_spacing=0.5,
        support_geometry_path=support_path,
        solver=SolverParameters(iteration_bound=4),
    )

    result = run_simulation_grid(_tiny_grid(), config)

    assert result.support_geometry == support_path


@pytest.mark.skipif(
    not _backend_available(SolverBackend.CPU_NATIVE),
    reason="Native C++ solver module is not available.",
)
def test_cpu_native_suppresses_support_only_idp_halo():
    reference = ReferenceLayerwiseMarkovSolver()
    native = NativeLayerwiseMarkovSolver()

    support_reference = reference.solve(
        _idp_halo_grid(support_only=True),
        _idp_halo_parameters(SolverBackend.CPU_REFERENCE),
    )
    support_native = native.solve(
        _idp_halo_grid(support_only=True),
        _idp_halo_parameters(SolverBackend.CPU_NATIVE),
    )

    assert np.array_equal(support_native.probability, support_reference.probability)
    assert support_native.probability[1, 1, 3] == 0

    part_reference = reference.solve(
        _idp_halo_grid(support_only=False),
        _idp_halo_parameters(SolverBackend.CPU_REFERENCE),
    )
    part_native = native.solve(
        _idp_halo_grid(support_only=False),
        _idp_halo_parameters(SolverBackend.CPU_NATIVE),
    )

    assert np.array_equal(part_native.probability, part_reference.probability)
    assert part_native.probability[1, 1, 3] > 0


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
