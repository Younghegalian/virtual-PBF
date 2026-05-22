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
    SupportGenerationParameters,
)


@dataclass(frozen=True)
class SimulationConfig:
    geometry_path: Path
    output_dir: Path
    voxel_spacing: float
    solver: SolverParameters
    support_geometry_path: Path | None = None
    support_type: str = "Volume support"
    support_generation: SupportGenerationParameters | None = None


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

    support_geometry_path = raw.get("support_geometry_path")

    return SimulationConfig(
        geometry_path=geometry_path,
        output_dir=output_dir,
        voxel_spacing=float(raw["voxel_spacing"]),
        solver=solver,
        support_geometry_path=(
            _resolve_path(base_dir, support_geometry_path)
            if support_geometry_path is not None
            else None
        ),
        support_type=str(raw.get("support_type", "Volume support")),
        support_generation=_support_generation_from_raw(raw.get("support_generation")),
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


def _support_generation_from_raw(value: Any) -> SupportGenerationParameters | None:
    if value is None or value is False:
        return None
    if value is True:
        return SupportGenerationParameters()
    if not isinstance(value, dict):
        raise TypeError("support_generation must be a mapping when provided.")
    return SupportGenerationParameters(
        support_type=str(value.get("support_type", "X surface support")),
        overhang_angle=float(value.get("overhang_angle", 60.0)),
        pitch=float(value.get("pitch", 2.0)),
        thickness=float(value.get("thickness", 0.5)),
        footprint_offset=float(value.get("footprint_offset", 0.5)),
        contact_depth=float(value.get("contact_depth", 0.0)),
        build_plate_z=(
            None
            if value.get("build_plate_z") is None
            else float(value.get("build_plate_z", 0.0))
        ),
    )
