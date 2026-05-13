from __future__ import annotations

import subprocess
from dataclasses import dataclass
from importlib.util import find_spec
from shutil import which
from subprocess import DEVNULL, PIPE

from capp.compute.cuda_env import ensure_cupy_cache_dir
from capp.domain import SolverBackend


@dataclass(frozen=True)
class BackendStatus:
    backend: SolverBackend
    label: str
    available: bool
    detail: str


def solver_backend_statuses() -> list[BackendStatus]:
    gpu_detail = _gpu_detail()
    return [
        BackendStatus(
            backend=SolverBackend.CPU_REFERENCE,
            label="PBF Standard",
            available=True,
            detail="Baseline analysis engine for verification and debugging.",
        ),
        _native_cpu_status(),
        _cuda_status(gpu_detail),
    ]


def validate_solver_backend(backend: SolverBackend) -> BackendStatus:
    backend = SolverBackend(backend)
    if backend is SolverBackend.CPU_REFERENCE:
        status = BackendStatus(
            backend=SolverBackend.CPU_REFERENCE,
            label="PBF Standard",
            available=True,
            detail="Baseline analysis engine for verification and debugging.",
        )
    elif backend is SolverBackend.CPU_NATIVE:
        status = _native_cpu_status()
    elif backend is SolverBackend.CUDA:
        status = _cuda_status(_gpu_detail())
    else:
        raise RuntimeError(f"Unknown solver backend: {backend}")

    if status.available:
        return status
    raise RuntimeError(status.detail)


def _native_cpu_status() -> BackendStatus:
    module_ready = _has_module("capp_native")
    detail_parts = []
    if module_ready:
        try:
            import capp_native  # type: ignore[import-not-found]

            module_ready = hasattr(capp_native, "solve_von_neumann")
        except Exception:
            module_ready = False

    if module_ready:
        return BackendStatus(
            backend=SolverBackend.CPU_NATIVE,
            label="PBF Direct",
            available=True,
            detail="Primary production solver for calibration and simulation runs.",
        )

    detail_parts.append("Native layerwise solver module is not built or lacks solve_von_neumann().")
    detail_parts.append(f"CMake: {_tool_state('cmake')}.")
    detail_parts.append(f"C++ compiler: {_cxx_tool_state()}.")
    return BackendStatus(
        backend=SolverBackend.CPU_NATIVE,
        label="PBF Direct",
        available=False,
        detail=" ".join(detail_parts),
    )


def _cuda_status(gpu_detail: str) -> BackendStatus:
    try:
        ensure_cupy_cache_dir()
        import cupy as cp

        count = int(cp.cuda.runtime.getDeviceCount())
        if count <= 0:
            raise RuntimeError("CuPy reported zero CUDA devices.")
        test = cp.arange(8, dtype=cp.float32)
        float(cp.sum(test).get())
        return BackendStatus(
            backend=SolverBackend.CUDA,
            label="PBF X",
            available=True,
            detail=f"High-performance analysis engine is available. {gpu_detail}",
        )
    except Exception as exc:
        cupy_detail = f"CUDA layerwise solver unavailable: {exc}."

    detail_parts = [cupy_detail]
    detail_parts.append(gpu_detail)
    detail_parts.append(f"nvcc: {_tool_state('nvcc')}.")
    return BackendStatus(
        backend=SolverBackend.CUDA,
        label="PBF X",
        available=False,
        detail=" ".join(detail_parts),
    )


def _gpu_detail() -> str:
    nvidia_smi = which("nvidia-smi")
    if not nvidia_smi:
        return "No nvidia-smi command found."

    try:
        completed = _run_hidden(
            [nvidia_smi, "-L"],
            stdout=PIPE,
            stderr=DEVNULL,
            text=True,
            timeout=4,
            check=False,
        )
    except Exception as exc:
        return f"GPU query failed: {exc}."

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or not lines:
        return "No NVIDIA GPU reported by nvidia-smi."
    return "Detected " + "; ".join(lines) + "."


def _run_hidden(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)
    return subprocess.run(args, **kwargs)


def _has_module(name: str) -> bool:
    return find_spec(name) is not None


def _tool_state(command: str) -> str:
    path = which(command)
    return path if path else "not found on PATH"


def _cxx_tool_state() -> str:
    for command in ("cl", "clang++", "g++"):
        path = which(command)
        if path:
            return f"{command} at {path}"
    return "not found on PATH"
