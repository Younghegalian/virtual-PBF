from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import numpy as np

from capp.compute.devices import validate_solver_backend
from capp.domain import SimulationResult, SolverBackend, SolverParameters, VoxelGrid

ProgressCallback = Callable[[int, str], None]


class NativeLayerwiseMarkovSolver:
    """Adapter for the native C++ solver module."""

    def __init__(self) -> None:
        validate_solver_backend(SolverBackend.CPU_NATIVE)
        import capp_native  # type: ignore[import-not-found]

        self._module = capp_native

    def solve(
        self,
        grid: VoxelGrid,
        parameters: SolverParameters,
        progress_callback: ProgressCallback | None = None,
    ) -> SimulationResult:
        started = perf_counter()
        payload = self._module.solve_von_neumann(
            grid.data,
            float(grid.spacing),
            _parameters_payload(parameters),
            progress_callback,
        )
        return _result_from_payload(payload, grid, perf_counter() - started)


def _parameters_payload(parameters: SolverParameters) -> dict[str, object]:
    payload: dict[str, object] = {
        "neighborhood": parameters.neighborhood.value,
        "current_coefficients": parameters.current_coefficients,
        "lower_coefficients": parameters.lower_coefficients,
        "residual_criteria": parameters.residual_criteria,
        "overwrap_criterion": parameters.overwrap_criterion,
        "iteration_bound": parameters.iteration_bound,
        "min_bias": parameters.min_bias,
        "stochastic_mode": parameters.stochastic_mode.value,
        "machine_bias": parameters.machine_bias.value,
        "initial_deviation": parameters.initial_deviation,
        "rng_seed": parameters.rng_seed,
        "spatial_current_coefficients": None,
        "spatial_min_bias": None,
        "spatial_initial_deviation": None,
    }
    if parameters.spatial_current_coefficients is not None:
        payload["spatial_current_coefficients"] = parameters.spatial_current_coefficients
    if parameters.spatial_min_bias is not None:
        payload["spatial_min_bias"] = parameters.spatial_min_bias
    if parameters.spatial_initial_deviation is not None:
        payload["spatial_initial_deviation"] = parameters.spatial_initial_deviation
    return payload


def _result_from_payload(
    payload: dict[str, object],
    grid: VoxelGrid,
    elapsed_seconds: float,
) -> SimulationResult:
    probability = np.asarray(payload["probability"], dtype=np.uint8)
    binary = np.asarray(payload["binary"], dtype=bool)
    return SimulationResult(
        probability=probability,
        binary=binary,
        voxel=grid.data,
        spacing=grid.spacing,
        origin=grid.origin,
        rest_volume=float(payload["rest_volume"]),
        probability_density=float(payload["probability_density"]),
        elapsed_seconds=float(payload.get("elapsed_seconds", elapsed_seconds)),
    )
