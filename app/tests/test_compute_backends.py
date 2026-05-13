from capp.compute.devices import solver_backend_statuses, validate_solver_backend
from capp.domain import SolverBackend, SolverParameters


def test_solver_parameters_preserve_legacy_use_gpu_flag():
    parameters = SolverParameters(use_gpu=True)

    assert parameters.backend is SolverBackend.CUDA
    assert parameters.use_gpu is True


def test_cpu_reference_backend_is_always_available():
    statuses = {status.backend: status for status in solver_backend_statuses()}

    assert statuses[SolverBackend.CPU_REFERENCE].available is True
    assert validate_solver_backend(SolverBackend.CPU_REFERENCE).available is True
