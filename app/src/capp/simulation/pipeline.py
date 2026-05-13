from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from capp.domain import SimulationResult, SolverParameters, VoxelGrid
from capp.geometry.voxelizer import voxelize_mesh
from capp.solver.base import PrintSolver
from capp.solver.reference import ReferenceLayerwiseMarkovSolver

ProgressCallback = Callable[[int, str], None]


class SimulationPipeline:
    def __init__(self, solver: PrintSolver | None = None) -> None:
        self._solver = solver or ReferenceLayerwiseMarkovSolver()

    def run_from_stl(
        self,
        geometry_path: str | Path,
        voxel_spacing: float,
        parameters: SolverParameters,
        progress_callback: ProgressCallback | None = None,
    ) -> SimulationResult:
        from capp.machine_map import apply_machine_parameter_map
        from capp.solver.factory import create_solver

        grid = voxelize_mesh(
            geometry_path,
            spacing=voxel_spacing,
            progress_callback=(
                _scale_progress(progress_callback, 0, 35, "Voxelizing geometry")
                if progress_callback is not None
                else None
            ),
        )
        parameters = apply_machine_parameter_map(parameters, grid)
        solver = create_solver(parameters)
        result = self.run_voxel_grid(
            grid,
            parameters,
            solver=solver,
            progress_callback=(
                _scale_progress(progress_callback, 35, 100, "Solving virtual printing")
                if progress_callback is not None
                else None
            ),
        )
        return SimulationResult(
            probability=result.probability,
            binary=result.binary,
            voxel=result.voxel,
            spacing=result.spacing,
            origin=result.origin,
            rest_volume=result.rest_volume,
            probability_density=result.probability_density,
            elapsed_seconds=result.elapsed_seconds,
            source_geometry=Path(geometry_path),
        )

    def run_voxel_grid(
        self,
        grid: VoxelGrid,
        parameters: SolverParameters,
        solver: PrintSolver | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> SimulationResult:
        active_solver = solver or self._solver
        return active_solver.solve(grid, parameters, progress_callback=progress_callback)


def _scale_progress(
    callback: ProgressCallback | None,
    start: int,
    end: int,
    fallback_message: str,
) -> ProgressCallback | None:
    if callback is None:
        return None

    def scaled(percent: int, message: str) -> None:
        value = start + int((end - start) * max(0, min(100, percent)) / 100)
        callback(value, message or fallback_message)

    return scaled

