from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class NeighborhoodModel(StrEnum):
    SIMPLE_VON_NEUMANN = "simple_von_neumann"
    DIRECTIONAL_VON_NEUMANN = "directional_von_neumann"
    SIMPLE_MOORE = "simple_moore"


class StochasticMode(StrEnum):
    IN_LAYER = "in_layer"
    IN_VOLUME = "in_volume"


class MachineBiasMode(StrEnum):
    NONE = "none"
    PRESET = "preset"


class MachineMapCoordinateMode(StrEnum):
    FULL_BASE_PLATE = "full_base_plate"
    PART_CENTER = "part_center"
    EXPLICIT_BOUNDS = "explicit_bounds"


class SolverBackend(StrEnum):
    CPU_REFERENCE = "cpu_reference"
    CPU_NATIVE = "cpu_native"
    CUDA = "cuda"


@dataclass(frozen=True)
class SupportGenerationParameters:
    support_type: str = "X surface support"
    overhang_angle: float = 60.0
    pitch: float = 2.0
    thickness: float = 0.5
    footprint_offset: float = 0.5
    contact_depth: float = 0.0
    build_plate_z: float | None = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "support_type", str(self.support_type))
        object.__setattr__(self, "overhang_angle", float(self.overhang_angle))
        object.__setattr__(self, "pitch", float(self.pitch))
        object.__setattr__(self, "thickness", float(self.thickness))
        object.__setattr__(self, "footprint_offset", float(self.footprint_offset))
        object.__setattr__(self, "contact_depth", float(self.contact_depth))
        if self.build_plate_z is not None:
            object.__setattr__(self, "build_plate_z", float(self.build_plate_z))
        if not 0.0 <= self.overhang_angle <= 90.0:
            raise ValueError("Support overhang angle must be between 0 and 90 degrees.")
        if self.pitch <= 0:
            raise ValueError("Support pitch must be positive.")
        if self.thickness <= 0:
            raise ValueError("Support thickness must be positive.")
        if self.footprint_offset < 0:
            raise ValueError("Support footprint offset cannot be negative.")
        if self.contact_depth < 0:
            raise ValueError("Support contact overlap cannot be negative.")


@dataclass(frozen=True)
class VoxelGrid:
    data: NDArray[np.bool_]
    spacing: float
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    support_mask: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError("VoxelGrid.data must be a 3D array.")
        if self.spacing <= 0:
            raise ValueError("VoxelGrid.spacing must be positive.")
        data = self.data.astype(bool, copy=False)
        if self.support_mask is None:
            support_mask = np.zeros(data.shape, dtype=bool)
        else:
            support_mask = np.asarray(self.support_mask, dtype=bool)
            if support_mask.shape != data.shape:
                raise ValueError("VoxelGrid.support_mask must match VoxelGrid.data shape.")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "support_mask", support_mask)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.data.shape)

    @property
    def filled_count(self) -> int:
        return int(np.count_nonzero(self.data))


@dataclass(frozen=True)
class SolverParameters:
    neighborhood: NeighborhoodModel = NeighborhoodModel.DIRECTIONAL_VON_NEUMANN
    current_coefficients: float | tuple[float, float, float, float] = (
        0.2,
        0.2,
        0.2,
        0.2,
    )
    lower_coefficients: float | tuple[float, float] = 1.0
    residual_criteria: tuple[float, float, float, float] = (1e-5, 1e-4, 1e-4, 1e-3)
    overwrap_criterion: float = 0.1
    iteration_bound: int = 100
    min_bias: float = 0.05
    stochastic_mode: StochasticMode = StochasticMode.IN_LAYER
    machine_bias: MachineBiasMode = MachineBiasMode.NONE
    machine_map_path: Path | None = None
    machine_map_coordinate_mode: MachineMapCoordinateMode = MachineMapCoordinateMode.PART_CENTER
    machine_map_position: tuple[float, float] = (0.0, 0.0)
    machine_map_bounds: tuple[float, float, float, float] | None = None
    spatial_current_coefficients: tuple[NDArray[np.float32], ...] | None = None
    spatial_min_bias: NDArray[np.float32] | None = None
    spatial_initial_deviation: NDArray[np.float32] | None = None
    initial_deviation: float = 0.0
    backend: SolverBackend = SolverBackend.CPU_REFERENCE
    use_gpu: bool = False
    rng_seed: int | None = 1000

    def __post_init__(self) -> None:
        backend = SolverBackend(self.backend)
        if self.use_gpu and backend is SolverBackend.CPU_REFERENCE:
            backend = SolverBackend.CUDA
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "use_gpu", backend is SolverBackend.CUDA)
        if self.machine_map_path is not None:
            object.__setattr__(self, "machine_map_path", Path(self.machine_map_path))
        object.__setattr__(
            self,
            "machine_map_coordinate_mode",
            MachineMapCoordinateMode(self.machine_map_coordinate_mode),
        )
        if len(self.machine_map_position) != 2:
            raise ValueError("machine_map_position requires X and Y coordinates.")
        object.__setattr__(
            self,
            "machine_map_position",
            (float(self.machine_map_position[0]), float(self.machine_map_position[1])),
        )
        if self.machine_map_bounds is not None:
            if len(self.machine_map_bounds) != 4:
                raise ValueError("machine_map_bounds requires X min, X max, Y min, Y max.")
            bounds = tuple(float(value) for value in self.machine_map_bounds)
            if bounds[0] == bounds[1] or bounds[2] == bounds[3]:
                raise ValueError("machine_map_bounds must span non-zero X and Y ranges.")
            object.__setattr__(self, "machine_map_bounds", bounds)
        if self.spatial_current_coefficients is not None:
            if len(self.spatial_current_coefficients) != 4:
                raise ValueError("spatial_current_coefficients requires four parameter grids.")
            object.__setattr__(
                self,
                "spatial_current_coefficients",
                tuple(
                    np.asarray(value, dtype=np.float32)
                    for value in self.spatial_current_coefficients
                ),
            )
        if self.spatial_min_bias is not None:
            object.__setattr__(
                self,
                "spatial_min_bias",
                np.asarray(self.spatial_min_bias, dtype=np.float32),
            )
        if self.spatial_initial_deviation is not None:
            object.__setattr__(
                self,
                "spatial_initial_deviation",
                np.asarray(self.spatial_initial_deviation, dtype=np.float32),
            )

    def current_as_directional(self) -> tuple[float, float, float, float]:
        values = self.current_coefficients
        if isinstance(values, Sequence):
            if len(values) != 4:
                raise ValueError("Directional current coefficients require four values.")
            return tuple(float(v) for v in values)
        v = float(values)
        return (v, v, v, v)

    def lower_center(self) -> float:
        values = self.lower_coefficients
        if isinstance(values, Sequence):
            if len(values) == 0:
                raise ValueError("lower_coefficients cannot be empty.")
            return float(values[-1])
        return float(values)

    @property
    def has_spatial_parameters(self) -> bool:
        return (
            self.spatial_current_coefficients is not None
            or self.spatial_min_bias is not None
            or self.spatial_initial_deviation is not None
        )


@dataclass(frozen=True)
class SimulationResult:
    probability: NDArray[np.uint8]
    binary: NDArray[np.bool_]
    voxel: NDArray[np.bool_]
    spacing: float
    origin: tuple[float, float, float]
    rest_volume: float
    probability_density: float
    elapsed_seconds: float
    source_geometry: Path | None = None
    support_geometry: Path | None = None
    support_mask: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        if self.probability.shape != self.binary.shape:
            raise ValueError("probability and binary arrays must have the same shape.")
        if self.probability.shape != self.voxel.shape:
            raise ValueError("probability and voxel arrays must have the same shape.")
        if self.support_mask is None:
            support_mask = np.zeros(self.voxel.shape, dtype=bool)
        else:
            support_mask = np.asarray(self.support_mask, dtype=bool)
            if support_mask.shape != self.voxel.shape:
                raise ValueError("support_mask must match simulation volume shape.")
        object.__setattr__(self, "probability", self.probability.astype(np.uint8, copy=False))
        object.__setattr__(self, "binary", self.binary.astype(bool, copy=False))
        object.__setattr__(self, "voxel", self.voxel.astype(bool, copy=False))
        object.__setattr__(self, "support_mask", support_mask)
