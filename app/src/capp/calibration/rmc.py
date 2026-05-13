from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from capp.calibration.roi import central_slices, extract_rmc_roi
from capp.domain import (
    MachineBiasMode,
    NeighborhoodModel,
    SolverBackend,
    SolverParameters,
    StochasticMode,
)


@dataclass(frozen=True)
class RmcParameterSet:
    nx: float
    px: float
    ny: float
    py: float
    eps: float
    idp: float

    @classmethod
    def from_sequence(cls, values: tuple[float, ...] | list[float] | NDArray[np.floating]):
        if len(values) != 6:
            raise ValueError("RMC parameter set requires six values.")
        return cls(*(float(value) for value in values))

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (self.nx, self.px, self.ny, self.py, self.eps, self.idp)

    def to_solver_parameters(
        self,
        *,
        backend: SolverBackend = SolverBackend.CPU_NATIVE,
        rng_seed: int | None = 1000,
    ) -> SolverParameters:
        return SolverParameters(
            neighborhood=NeighborhoodModel.DIRECTIONAL_VON_NEUMANN,
            current_coefficients=(self.nx, self.px, self.ny, self.py),
            lower_coefficients=1.0,
            residual_criteria=(1e-5, 1e-4, 1e-4, 1e-3),
            overwrap_criterion=0.1,
            iteration_bound=100,
            min_bias=self.eps,
            stochastic_mode=StochasticMode.IN_LAYER,
            machine_bias=MachineBiasMode.NONE,
            initial_deviation=self.idp,
            backend=backend,
            rng_seed=rng_seed,
        )


@dataclass(frozen=True)
class RmcBounds:
    nx: tuple[float, float] = (0.05, 0.4)
    px: tuple[float, float] = (0.05, 0.4)
    ny: tuple[float, float] = (0.05, 0.4)
    py: tuple[float, float] = (0.05, 0.4)
    eps: tuple[float, float] = (0.005, 0.25)
    idp: tuple[float, float] = (0.1, 0.6)

    def as_pairs(self) -> tuple[tuple[float, float], ...]:
        return (self.nx, self.px, self.ny, self.py, self.eps, self.idp)


@dataclass(frozen=True)
class RmcTarget:
    sample: str
    roi_x: NDArray[np.bool_]
    roi_y: NDArray[np.bool_]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample", self.sample.strip())
        object.__setattr__(self, "roi_x", np.asarray(self.roi_x, dtype=bool))
        object.__setattr__(self, "roi_y", np.asarray(self.roi_y, dtype=bool))
        if self.roi_x.ndim != 2 or self.roi_y.ndim != 2:
            raise ValueError("RMC target ROIs must be 2D masks.")


def simulation_rois(
    binary_volume: NDArray[np.bool_],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    x_slice, y_slice = central_slices(np.asarray(binary_volume, dtype=bool))
    return extract_rmc_roi(x_slice), extract_rmc_roi(y_slice)
