from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from capp.domain import (
    MachineBiasMode,
    MachineMapCoordinateMode,
    NeighborhoodModel,
    SolverBackend,
    SolverParameters,
    StochasticMode,
)


@dataclass(frozen=True)
class SimulationConfig:
    geometry_path: Path
    output_dir: Path
    voxel_spacing: float
    solver: SolverParameters


def load_simulation_config(path: str | Path) -> SimulationConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    base_dir = config_path.parent
    geometry_path = _resolve_path(base_dir, raw["geometry_path"])
    output_dir = _resolve_path(base_dir, raw.get("output_dir", "outputs"))

    solver_raw: dict[str, Any] = raw.get("solver", {})
    machine_map_path = solver_raw.get("machine_map_path")
    solver = SolverParameters(
        neighborhood=NeighborhoodModel(
            solver_raw.get("neighborhood", NeighborhoodModel.DIRECTIONAL_VON_NEUMANN.value)
        ),
        current_coefficients=_tuple_or_scalar(
            solver_raw.get("current_coefficients", [0.2, 0.2, 0.2, 0.2])
        ),
        lower_coefficients=_tuple_or_scalar(solver_raw.get("lower_coefficients", 1.0)),
        residual_criteria=tuple(
            float(v) for v in solver_raw.get("residual_criteria", [1e-5, 1e-4, 1e-4, 1e-3])
        ),
        overwrap_criterion=float(solver_raw.get("overwrap_criterion", 0.1)),
        iteration_bound=int(solver_raw.get("iteration_bound", 100)),
        min_bias=float(solver_raw.get("min_bias", 0.05)),
        stochastic_mode=StochasticMode(
            solver_raw.get("stochastic_mode", StochasticMode.IN_LAYER.value)
        ),
        machine_bias=MachineBiasMode(solver_raw.get("machine_bias", MachineBiasMode.NONE.value)),
        machine_map_path=(
            _resolve_path(base_dir, machine_map_path) if machine_map_path is not None else None
        ),
        machine_map_coordinate_mode=MachineMapCoordinateMode(
            solver_raw.get(
                "machine_map_coordinate_mode",
                MachineMapCoordinateMode.PART_CENTER.value,
            )
        ),
        machine_map_position=tuple(
            float(v) for v in solver_raw.get("machine_map_position", [0.0, 0.0])
        ),
        machine_map_bounds=(
            tuple(float(v) for v in solver_raw["machine_map_bounds"])
            if solver_raw.get("machine_map_bounds") is not None
            else None
        ),
        initial_deviation=float(solver_raw.get("initial_deviation", 0.0)),
        backend=SolverBackend(
            solver_raw.get(
                "backend",
                SolverBackend.CUDA.value
                if bool(solver_raw.get("use_gpu", False))
                else SolverBackend.CPU_REFERENCE.value,
            )
        ),
        use_gpu=bool(solver_raw.get("use_gpu", False)),
        rng_seed=solver_raw.get("rng_seed", 1000),
    )

    return SimulationConfig(
        geometry_path=geometry_path,
        output_dir=output_dir,
        voxel_spacing=float(raw["voxel_spacing"]),
        solver=solver,
    )


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _tuple_or_scalar(value: Any) -> float | tuple[float, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(float(v) for v in value)
    return float(value)
