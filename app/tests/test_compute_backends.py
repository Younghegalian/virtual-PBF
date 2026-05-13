import os
from pathlib import Path

from capp.compute import cuda_env
from capp.compute.cuda_env import ensure_cupy_cache_dir
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


def test_cupy_environment_prefers_bundled_runtime(monkeypatch, tmp_path):
    fake_module = tmp_path / "project" / "src" / "capp" / "compute" / "cuda_env.py"
    runtime_root = (
        tmp_path
        / "project"
        / ".venv"
        / "Lib"
        / "site-packages"
        / "nvidia"
        / "cuda_runtime"
    )
    runtime_root.mkdir(parents=True)
    monkeypatch.setenv("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2")
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setattr(cuda_env, "__file__", str(fake_module))

    ensure_cupy_cache_dir()

    assert Path(os.environ["CUDA_PATH"]) == runtime_root
    assert Path(os.environ["CUDA_HOME"]) == runtime_root
