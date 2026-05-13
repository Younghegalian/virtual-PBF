from __future__ import annotations

from capp.compute.devices import validate_solver_backend
from capp.domain import SolverBackend, SolverParameters
from capp.solver.base import PrintSolver
from capp.solver.cuda import CudaLayerwiseMarkovSolver
from capp.solver.native import NativeLayerwiseMarkovSolver
from capp.solver.reference import ReferenceLayerwiseMarkovSolver


def create_solver(parameters: SolverParameters) -> PrintSolver:
    backend = SolverBackend(parameters.backend)
    validate_solver_backend(backend)

    if backend is SolverBackend.CPU_REFERENCE:
        return ReferenceLayerwiseMarkovSolver()
    if backend is SolverBackend.CPU_NATIVE:
        return NativeLayerwiseMarkovSolver()
    if backend is SolverBackend.CUDA:
        return CudaLayerwiseMarkovSolver()
    raise ValueError(f"Unsupported solver backend: {backend}")
