from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from capp.config import SimulationConfig
from capp.domain import SimulationResult, VoxelGrid
from capp.io.exports import save_npz, write_vtk_volume
from capp.simulation.pipeline import SimulationPipeline
from capp.solver.factory import create_solver

ProgressCallback = Callable[[int, str], None]


def run_simulation_config(
    config: SimulationConfig,
    progress_callback: ProgressCallback | None = None,
) -> SimulationResult:
    pipeline = SimulationPipeline(solver=None)
    return pipeline.run_from_stl(
        geometry_path=config.geometry_path,
        voxel_spacing=config.voxel_spacing,
        parameters=config.solver,
        progress_callback=progress_callback,
    )


def run_simulation_grid(
    grid: VoxelGrid,
    config: SimulationConfig,
    progress_callback: ProgressCallback | None = None,
) -> SimulationResult:
    from capp.machine_map import apply_machine_parameter_map

    solver_parameters = apply_machine_parameter_map(config.solver, grid)
    pipeline = SimulationPipeline(solver=create_solver(solver_parameters))
    result = pipeline.run_voxel_grid(grid, solver_parameters, progress_callback=progress_callback)
    return SimulationResult(
        probability=result.probability,
        binary=result.binary,
        voxel=result.voxel,
        spacing=result.spacing,
        origin=result.origin,
        rest_volume=result.rest_volume,
        probability_density=result.probability_density,
        elapsed_seconds=result.elapsed_seconds,
        source_geometry=config.geometry_path,
    )


def save_default_outputs(
    output_dir: str | Path,
    result: SimulationResult,
    progress_callback: ProgressCallback | None = None,
) -> None:
    output_path = Path(output_dir)
    if progress_callback is not None:
        progress_callback(5, "Preparing output folder")
    output_path.mkdir(parents=True, exist_ok=True)
    if progress_callback is not None:
        progress_callback(15, "Saving simulation_result.npz")
    save_npz(output_path / "simulation_result.npz", result)
    if progress_callback is not None:
        progress_callback(40, "Saving probability.vtk")
    write_vtk_volume(
        output_path / "probability.vtk",
        result.probability,
        spacing=result.spacing,
        origin=result.origin,
        scalar_name="Probability",
    )
    if progress_callback is not None:
        progress_callback(70, "Saving binary.vtk")
    write_vtk_volume(
        output_path / "binary.vtk",
        result.binary.astype("uint8"),
        spacing=result.spacing,
        origin=result.origin,
        scalar_name="Binary",
    )
    if progress_callback is not None:
        progress_callback(100, "Output save complete")
