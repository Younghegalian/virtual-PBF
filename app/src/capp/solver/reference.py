from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from capp.domain import (
    MachineBiasMode,
    NeighborhoodModel,
    SimulationResult,
    SolverBackend,
    SolverParameters,
    StochasticMode,
    VoxelGrid,
)
from capp.geometry.components import remove_small_components

ProgressCallback = Callable[[int, str], None]


class ReferenceLayerwiseMarkovSolver:
    """CPU reference implementation for the layer-wise Markov solver.

    This is intentionally written for clarity and parity testing before CUDA acceleration.
    It is not expected to be the final high-performance implementation.
    """

    def solve(
        self,
        grid: VoxelGrid,
        parameters: SolverParameters,
        progress_callback: ProgressCallback | None = None,
    ) -> SimulationResult:
        if parameters.backend is not SolverBackend.CPU_REFERENCE:
            raise NotImplementedError(
                f"The Python reference solver cannot run backend {parameters.backend.value}."
            )
        if parameters.machine_bias is not MachineBiasMode.NONE:
            raise NotImplementedError("Machine bias presets are not migrated yet.")
        if parameters.neighborhood is NeighborhoodModel.SIMPLE_MOORE:
            raise NotImplementedError("Moore neighborhood parity is pending.")
        if grid.filled_count == 0:
            raise ValueError("Cannot simulate an empty voxel grid.")

        started = perf_counter()
        progress = _ProgressReporter(progress_callback)
        progress.report(0, "Preparing virtual printing solver")
        rng = np.random.default_rng(parameters.rng_seed)
        voxel = grid.data.astype(np.float32, copy=False)
        x_size, y_size, z_size = grid.shape

        voxel_calc = np.pad(voxel, ((1, 1), (1, 1), (1, 1)), mode="constant")
        voxel_calc[:, :, 0] = 1.0

        probability = np.zeros_like(voxel_calc, dtype=np.float32)
        probability[:, :, 0] = 1.0
        probability_export = np.zeros_like(voxel_calc, dtype=np.float32)
        binary = np.zeros_like(voxel_calc, dtype=bool)
        random_field = rng.random(voxel_calc.shape, dtype=np.float32)

        spacing_scale = (
            grid.spacing
            if parameters.stochastic_mode is StochasticMode.IN_LAYER
            else grid.spacing**12
        )
        min_val = (
            parameters.spatial_min_bias * spacing_scale
            if parameters.spatial_min_bias is not None
            else parameters.min_bias * spacing_scale
        )
        beta = int(np.ceil(0.1 / grid.spacing) + 1)

        coeffs = _kernel_coefficients(parameters)
        idp_model = (
            parameters.spatial_initial_deviation / grid.spacing
            if parameters.spatial_initial_deviation is not None
            else parameters.initial_deviation / grid.spacing
        )

        for layer in range(1, z_size + 1):
            center_voxel = voxel_calc[1 : x_size + 1, 1 : y_size + 1, layer]
            layer_view = probability[1 : x_size + 1, 1 : y_size + 1, layer]
            progress.layer(
                layer=layer,
                layer_count=z_size,
                iteration=0,
                iteration_bound=parameters.iteration_bound,
            )

            if layer <= beta:
                layer_view[...] = center_voxel
                ma_residual = 0.0
                max_residual = 0.0
            else:
                ma_residual = 1.0
                max_residual = 1.0
                iteration = 0
                while (
                    ma_residual > parameters.residual_criteria[0]
                    or max_residual > parameters.residual_criteria[1]
                ) and iteration < parameters.iteration_bound:
                    previous_center = layer_view.copy()
                    updated = _update_von_neumann_layer(
                        probability=probability,
                        voxel_calc=voxel_calc,
                        layer=layer,
                        coeffs=coeffs,
                        min_val=min_val,
                        idp_model=idp_model,
                    )

                    delta = np.abs(previous_center - updated)
                    changed = int(np.count_nonzero(delta))
                    ma_residual = float(delta.sum() / (changed + 1))
                    max_residual = float(delta.max(initial=0.0))
                    layer_view[...] = updated
                    iteration += 1
                    progress.layer(
                        layer=layer,
                        layer_count=z_size,
                        iteration=iteration,
                        iteration_bound=parameters.iteration_bound,
                    )

            probability_export[1 : x_size + 1, 1 : y_size + 1, layer] = probability[
                1 : x_size + 1, 1 : y_size + 1, layer
            ]

            if parameters.stochastic_mode is StochasticMode.IN_LAYER:
                binary[:, :, layer] = random_field[:, :, layer] <= probability[:, :, layer]
                probability[:, :, layer] = binary[:, :, layer].astype(np.float32)

        if parameters.stochastic_mode is StochasticMode.IN_VOLUME:
            progress.report(92, "Sampling in-volume stochastic field")
            binary = (random_field <= probability) & voxel_calc.astype(bool)
            progress.report(94, "Smoothing sampled volume")
            binary = _smooth_binary(binary, max_iterations=20)

        progress.report(96, "Post-processing connected components")
        cropped_probability = probability_export[1 : x_size + 1, 1 : y_size + 1, 1 : z_size + 1]
        cropped_binary = _postprocess_binary(
            binary,
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

        progress.report(100, "Virtual printing solver complete")
        return SimulationResult(
            probability=probability_uint8,
            binary=cropped_binary,
            voxel=grid.data,
            spacing=grid.spacing,
            origin=grid.origin,
            rest_volume=rest_volume,
            probability_density=probability_density,
            elapsed_seconds=perf_counter() - started,
        )


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

    def layer(
        self,
        layer: int,
        layer_count: int,
        iteration: int,
        iteration_bound: int,
    ) -> None:
        safe_iteration_bound = max(1, iteration_bound)
        layer_fraction = (layer - 1 + min(iteration, safe_iteration_bound) / safe_iteration_bound)
        percent = int((layer_fraction / max(1, layer_count)) * 92)
        self.report(percent, f"Solving layer {layer}/{layer_count}")


def _kernel_coefficients(parameters: SolverParameters):
    if parameters.spatial_current_coefficients is not None:
        lower = parameters.lower_center()
        if parameters.neighborhood is not NeighborhoodModel.DIRECTIONAL_VON_NEUMANN:
            raise NotImplementedError("Spatial parameter maps require DirectionalVN.")
        return (*parameters.spatial_current_coefficients, lower)

    current = parameters.current_as_directional()
    lower = parameters.lower_center()
    if parameters.neighborhood is NeighborhoodModel.SIMPLE_VON_NEUMANN:
        c = float(current[0])
        return (c, c, c, c, lower)
    if parameters.neighborhood is NeighborhoodModel.DIRECTIONAL_VON_NEUMANN:
        # Order is negative-y, positive-x, positive-y, negative-x, lower-layer center.
        return (float(current[0]), float(current[1]), float(current[2]), float(current[3]), lower)
    raise NotImplementedError(f"Unsupported neighborhood: {parameters.neighborhood}")


def _has_initial_deviation(parameters: SolverParameters) -> bool:
    if parameters.spatial_initial_deviation is not None:
        return bool(np.any(parameters.spatial_initial_deviation > 0.0))
    return parameters.initial_deviation > 0.0


def _has_initial_deviation(parameters: SolverParameters) -> bool:
    if parameters.spatial_initial_deviation is not None:
        return bool(np.any(parameters.spatial_initial_deviation > 0.0))
    return parameters.initial_deviation > 0.0


def _update_von_neumann_layer(
    probability: NDArray[np.float32],
    voxel_calc: NDArray[np.float32],
    layer: int,
    coeffs: tuple[float, float, float, float, float],
    min_val: float,
    idp_model: float,
) -> NDArray[np.float32]:
    x_size = probability.shape[0] - 2
    y_size = probability.shape[1] - 2
    neg_y, pos_x, pos_y, neg_x, lower = coeffs

    left = probability[1 : x_size + 1, 0:y_size, layer] * neg_y
    upper = probability[0:x_size, 1 : y_size + 1, layer] * pos_x
    lower_x = probability[2 : x_size + 2, 1 : y_size + 1, layer] * pos_y
    right = probability[1 : x_size + 1, 2 : y_size + 2, layer] * neg_x
    center_voxel = voxel_calc[1 : x_size + 1, 1 : y_size + 1, layer]
    below = probability[1 : x_size + 1, 1 : y_size + 1, layer - 1] * center_voxel * lower

    no_growth = (1.0 - left) * (1.0 - upper) * (1.0 - lower_x) * (1.0 - right) * (1.0 - below)
    previous_center = probability[1 : x_size + 1, 1 : y_size + 1, layer]
    epsilon = (previous_center == 0.0).astype(np.float32) * min_val * center_voxel

    updated = ((center_voxel + idp_model) / (1.0 + idp_model)) * (1.0 - no_growth + epsilon)
    return np.clip(updated, 0.0, 1.0).astype(np.float32, copy=False)


def _smooth_binary(binary: NDArray[np.bool_], max_iterations: int) -> NDArray[np.bool_]:
    smoothed = binary.copy()
    threshold = 5.0
    for _ in range(max_iterations):
        previous = smoothed.copy()
        neighbors = (
            smoothed[:-2, :-2, 1:-1].astype(np.int16)
            + 2 * smoothed[1:-1, :-2, 1:-1].astype(np.int16)
            + smoothed[2:, :-2, 1:-1].astype(np.int16)
            + 2 * smoothed[:-2, 1:-1, 1:-1].astype(np.int16)
            + 2 * smoothed[2:, 1:-1, 1:-1].astype(np.int16)
            + smoothed[:-2, 2:, 1:-1].astype(np.int16)
            + 2 * smoothed[1:-1, 2:, 1:-1].astype(np.int16)
            + smoothed[2:, 2:, 1:-1].astype(np.int16)
            + 3 * smoothed[1:-1, 1:-1, :-2].astype(np.int16)
            + 3 * smoothed[1:-1, 1:-1, 2:].astype(np.int16)
        )
        smoothed[1:-1, 1:-1, 1:-1] &= neighbors >= threshold
        threshold -= 0.2
        if np.array_equal(previous, smoothed):
            break
    return smoothed


def _postprocess_binary(
    binary: NDArray[np.bool_],
    x_size: int,
    y_size: int,
    z_size: int,
    apply_area_open: bool = False,
) -> NDArray[np.bool_]:
    if apply_area_open:
        binary = binary.copy()
        cropped = binary[1 : x_size + 1, 1 : y_size + 1, 1 : z_size + 1]
        binary[1 : x_size + 1, 1 : y_size + 1, 1 : z_size + 1] = remove_small_components(
            cropped,
            min_size=8,
            connectivity=3,
        )
    filtered = remove_small_components(binary, min_size=50, connectivity=1)
    cropped = filtered[1 : x_size + 1, 1 : y_size + 1, 1 : z_size + 1]
    return remove_small_components(cropped, min_size=50, connectivity=1)
