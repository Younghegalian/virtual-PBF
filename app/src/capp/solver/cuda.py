from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import numpy as np

from capp.compute.cuda_env import ensure_cupy_cache_dir
from capp.compute.devices import validate_solver_backend
from capp.domain import (
    MachineBiasMode,
    NeighborhoodModel,
    SimulationResult,
    SolverBackend,
    SolverParameters,
    StochasticMode,
    VoxelGrid,
)
from capp.solver.reference import _has_initial_deviation, _kernel_coefficients, _postprocess_binary

ProgressCallback = Callable[[int, str], None]


class CudaLayerwiseMarkovSolver:
    """CuPy-backed CUDA solver for the layer-wise Markov update."""

    def __init__(self) -> None:
        validate_solver_backend(SolverBackend.CUDA)
        ensure_cupy_cache_dir()
        import cupy as cp

        self._cp = cp

    def solve(
        self,
        grid: VoxelGrid,
        parameters: SolverParameters,
        progress_callback: ProgressCallback | None = None,
    ) -> SimulationResult:
        if parameters.machine_bias is not MachineBiasMode.NONE:
            raise NotImplementedError("Machine bias presets are not migrated yet.")
        if parameters.neighborhood is NeighborhoodModel.SIMPLE_MOORE:
            raise NotImplementedError("Moore neighborhood parity is pending.")
        if grid.filled_count == 0:
            raise ValueError("Cannot simulate an empty voxel grid.")

        cp = self._cp
        started = perf_counter()
        progress = _ProgressReporter(progress_callback)
        _report(progress.report, 0, "Preparing CUDA solver")

        voxel = cp.asarray(grid.data, dtype=cp.float32)
        x_size, y_size, z_size = grid.shape

        voxel_calc = cp.pad(voxel, ((1, 1), (1, 1), (1, 1)), mode="constant")
        voxel_calc[:, :, 0] = 1.0

        probability = cp.zeros_like(voxel_calc, dtype=cp.float32)
        probability[:, :, 0] = 1.0
        probability_export = cp.zeros_like(voxel_calc, dtype=cp.float32)
        binary = cp.zeros(voxel_calc.shape, dtype=cp.bool_)
        rng = cp.random.default_rng(parameters.rng_seed)
        random_field = rng.random(voxel_calc.shape, dtype=cp.float32)

        spacing_scale = (
            grid.spacing
            if parameters.stochastic_mode is StochasticMode.IN_LAYER
            else grid.spacing**12
        )
        min_val = (
            cp.asarray(parameters.spatial_min_bias, dtype=cp.float32) * spacing_scale
            if parameters.spatial_min_bias is not None
            else parameters.min_bias * spacing_scale
        )
        beta = int(np.ceil(0.1 / grid.spacing) + 1)
        coeffs = tuple(
            cp.asarray(value, dtype=cp.float32) if np.ndim(value) else float(value)
            for value in _kernel_coefficients(parameters)
        )
        idp_model = (
            cp.asarray(parameters.spatial_initial_deviation, dtype=cp.float32) / grid.spacing
            if parameters.spatial_initial_deviation is not None
            else parameters.initial_deviation / grid.spacing
        )

        for layer in range(1, z_size + 1):
            center_voxel = voxel_calc[1 : x_size + 1, 1 : y_size + 1, layer]
            layer_view = probability[1 : x_size + 1, 1 : y_size + 1, layer]
            _layer_progress(progress.report, layer, z_size, 0, parameters.iteration_bound)

            if layer <= beta:
                layer_view[...] = center_voxel
            else:
                ma_residual = 1.0
                max_residual = 1.0
                iteration = 0
                while (
                    ma_residual > parameters.residual_criteria[0]
                    or max_residual > parameters.residual_criteria[1]
                ) and iteration < parameters.iteration_bound:
                    previous_center = layer_view.copy()
                    updated = _update_von_neumann_layer_gpu(
                        cp=cp,
                        probability=probability,
                        voxel_calc=voxel_calc,
                        layer=layer,
                        coeffs=coeffs,
                        min_val=min_val,
                        idp_model=idp_model,
                    )
                    delta = cp.abs(previous_center - updated)
                    changed = int(cp.count_nonzero(delta).get())
                    ma_residual = float((delta.sum() / (changed + 1)).get())
                    max_residual = float(delta.max().get()) if delta.size else 0.0
                    layer_view[...] = updated
                    iteration += 1
                    _layer_progress(
                        progress.report,
                        layer,
                        z_size,
                        iteration,
                        parameters.iteration_bound,
                    )

            probability_export[1 : x_size + 1, 1 : y_size + 1, layer] = probability[
                1 : x_size + 1, 1 : y_size + 1, layer
            ]

            if parameters.stochastic_mode is StochasticMode.IN_LAYER:
                binary[:, :, layer] = random_field[:, :, layer] <= probability[:, :, layer]
                probability[:, :, layer] = binary[:, :, layer].astype(cp.float32)

        if parameters.stochastic_mode is StochasticMode.IN_VOLUME:
            _report(progress.report, 92, "Sampling CUDA in-volume stochastic field")
            binary = (random_field <= probability) & voxel_calc.astype(cp.bool_)
            _report(progress.report, 94, "Smoothing CUDA sampled volume")
            binary = _smooth_binary_gpu(cp, binary, max_iterations=20)

        _report(progress.report, 96, "Transferring CUDA result to CPU")
        cropped_probability = cp.asnumpy(
            probability_export[1 : x_size + 1, 1 : y_size + 1, 1 : z_size + 1]
        )
        binary_cpu = cp.asnumpy(binary)

        _report(progress.report, 98, "Post-processing connected components")
        cropped_binary = _postprocess_binary(
            binary_cpu,
            x_size,
            y_size,
            z_size,
            apply_area_open=(
                parameters.stochastic_mode is StochasticMode.IN_LAYER
                and _has_initial_deviation(parameters)
            ),
        )
        probability_uint8 = np.uint8(np.floor(np.round(cropped_probability * 100)))
        solid_count = float(grid.filled_count)
        rest_volume = 100.0 * float(np.count_nonzero(cropped_binary)) / solid_count
        probability_density = float(
            (probability_uint8 * grid.data.astype(np.uint8)).sum() / solid_count
        )
        _report(progress.report, 100, "CUDA solver complete")
        return SimulationResult(
            probability=probability_uint8,
            binary=cropped_binary,
            voxel=grid.data,
            spacing=grid.spacing,
            origin=grid.origin,
            rest_volume=rest_volume,
            probability_density=probability_density,
            elapsed_seconds=perf_counter() - started,
            support_mask=grid.support_mask,
        )


def _update_von_neumann_layer_gpu(
    cp,
    probability,
    voxel_calc,
    layer: int,
    coeffs: tuple[float, float, float, float, float],
    min_val: float,
    idp_model: float,
):
    x_size = probability.shape[0] - 2
    y_size = probability.shape[1] - 2
    neg_y, pos_x, pos_y, neg_x, lower = coeffs

    left = probability[1 : x_size + 1, 0:y_size, layer] * neg_y
    upper = probability[0:x_size, 1 : y_size + 1, layer] * pos_x
    lower_x = probability[2 : x_size + 2, 1 : y_size + 1, layer] * pos_y
    right = probability[1 : x_size + 1, 2 : y_size + 2, layer] * neg_x
    center_voxel = voxel_calc[1 : x_size + 1, 1 : y_size + 1, layer]
    below = probability[1 : x_size + 1, 1 : y_size + 1, layer - 1] * center_voxel * lower

    no_growth = (1.0 - left) * (1.0 - upper) * (1.0 - lower_x) * (1.0 - right) * (
        1.0 - below
    )
    previous_center = probability[1 : x_size + 1, 1 : y_size + 1, layer]
    epsilon = (previous_center == 0.0).astype(cp.float32) * min_val * center_voxel
    updated = ((center_voxel + idp_model) / (1.0 + idp_model)) * (
        1.0 - no_growth + epsilon
    )
    return cp.clip(updated, 0.0, 1.0).astype(cp.float32, copy=False)


def _smooth_binary_gpu(cp, binary, max_iterations: int):
    smoothed = binary.copy()
    threshold = 5.0
    for _ in range(max_iterations):
        previous = smoothed.copy()
        neighbors = (
            smoothed[:-2, :-2, 1:-1].astype(cp.int16)
            + 2 * smoothed[1:-1, :-2, 1:-1].astype(cp.int16)
            + smoothed[2:, :-2, 1:-1].astype(cp.int16)
            + 2 * smoothed[:-2, 1:-1, 1:-1].astype(cp.int16)
            + 2 * smoothed[2:, 1:-1, 1:-1].astype(cp.int16)
            + smoothed[:-2, 2:, 1:-1].astype(cp.int16)
            + 2 * smoothed[1:-1, 2:, 1:-1].astype(cp.int16)
            + smoothed[2:, 2:, 1:-1].astype(cp.int16)
            + 3 * smoothed[1:-1, 1:-1, :-2].astype(cp.int16)
            + 3 * smoothed[1:-1, 1:-1, 2:].astype(cp.int16)
        )
        smoothed[1:-1, 1:-1, 1:-1] &= neighbors >= threshold
        threshold -= 0.2
        if bool(cp.array_equal(previous, smoothed).get()):
            break
    return smoothed


def _report(progress_callback: ProgressCallback | None, percent: int, message: str) -> None:
    if progress_callback is not None:
        progress_callback(max(0, min(100, int(percent))), message)


class _ProgressReporter:
    def __init__(self, callback: ProgressCallback | None) -> None:
        self._callback = callback
        self._last_percent = -1

    def report(self, percent: int, message: str) -> None:
        if self._callback is None:
            return
        percent = max(0, min(100, int(percent)))
        if percent == self._last_percent:
            return
        self._last_percent = percent
        self._callback(percent, message)


def _layer_progress(
    progress_callback: ProgressCallback | None,
    layer: int,
    layer_count: int,
    iteration: int,
    iteration_bound: int,
) -> None:
    safe_iteration_bound = max(1, iteration_bound)
    layer_fraction = (layer - 1 + min(iteration, safe_iteration_bound) / safe_iteration_bound)
    percent = int((layer_fraction / max(1, layer_count)) * 92)
    _report(progress_callback, percent, "Running GPU CUDA solver")
