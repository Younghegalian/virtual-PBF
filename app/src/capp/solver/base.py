from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from capp.domain import SimulationResult, SolverParameters, VoxelGrid

ProgressCallback = Callable[[int, str], None]


class PrintSolver(Protocol):
    def solve(
        self,
        grid: VoxelGrid,
        parameters: SolverParameters,
        progress_callback: ProgressCallback | None = None,
    ) -> SimulationResult:
        """Run a print simulation against a voxel grid."""

