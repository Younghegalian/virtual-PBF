from __future__ import annotations

import csv
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from threading import Lock
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.stats import qmc

from capp.calibration.losses import RoiBoundaryLoss, RoiLossResult
from capp.calibration.roi import central_slices, extract_model_calibration_roi
from capp.domain import (
    MachineBiasMode,
    NeighborhoodModel,
    SimulationResult,
    SolverBackend,
    SolverParameters,
    StochasticMode,
    VoxelGrid,
)
from capp.geometry.voxelizer import voxelize_mesh
from capp.io.exports import write_binary_stl, write_vtk_volume
from capp.simulation.pipeline import SimulationPipeline
from capp.solver.factory import create_solver

ProgressCallback = Callable[[int, str], None]
MODEL_CALIBRATION_OPTIMIZERS = ("adaptive_sobol", "sobol", "latin_hypercube")
_SOLVER_LABELS = {
    SolverBackend.CPU_REFERENCE: "PBF Standard",
    SolverBackend.CPU_NATIVE: "PBF Direct",
    SolverBackend.CUDA: "PBF X",
}


@dataclass(frozen=True)
class ModelCalibrationParameterSet:
    nx: float
    px: float
    ny: float
    py: float
    eps: float
    idp: float

    @classmethod
    def from_sequence(cls, values: tuple[float, ...] | list[float] | NDArray[np.floating]):
        if len(values) != 6:
            raise ValueError("Model calibration parameter set requires six values.")
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
class ModelCalibrationBounds:
    nx: tuple[float, float] = (0.05, 0.4)
    px: tuple[float, float] = (0.05, 0.4)
    ny: tuple[float, float] = (0.05, 0.4)
    py: tuple[float, float] = (0.05, 0.4)
    eps: tuple[float, float] = (0.005, 0.25)
    idp: tuple[float, float] = (0.1, 0.6)

    def as_pairs(self) -> tuple[tuple[float, float], ...]:
        return (self.nx, self.px, self.ny, self.py, self.eps, self.idp)


@dataclass(frozen=True)
class ModelCalibrationTarget:
    sample: str
    roi_x: NDArray[np.bool_]
    roi_y: NDArray[np.bool_]
    roi_x_inverted: bool = False
    roi_y_inverted: bool = False
    roi_x_path: Path | None = None
    roi_y_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample", self.sample.strip())
        object.__setattr__(self, "roi_x", np.asarray(self.roi_x, dtype=bool))
        object.__setattr__(self, "roi_y", np.asarray(self.roi_y, dtype=bool))
        object.__setattr__(self, "roi_x_inverted", bool(self.roi_x_inverted))
        object.__setattr__(self, "roi_y_inverted", bool(self.roi_y_inverted))
        if self.roi_x_path is not None:
            object.__setattr__(self, "roi_x_path", Path(self.roi_x_path))
        if self.roi_y_path is not None:
            object.__setattr__(self, "roi_y_path", Path(self.roi_y_path))
        if self.roi_x.ndim != 2 or self.roi_y.ndim != 2:
            raise ValueError("Model calibration target ROIs must be 2D masks.")


@dataclass(frozen=True)
class ModelCalibrationOptions:
    max_evaluations: int = 24
    backend: SolverBackend = SolverBackend.CPU_NATIVE
    rng_seed: int | None = 1000
    bounds: ModelCalibrationBounds = ModelCalibrationBounds()
    boundary_radius: int = 15
    max_workers: int = 1
    optimizer: str = "adaptive_sobol"
    save_research_artifacts: bool = False

    def __post_init__(self) -> None:
        if self.max_evaluations < 1:
            raise ValueError("max_evaluations must be at least 1.")
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1.")
        object.__setattr__(self, "backend", SolverBackend(self.backend))
        optimizer = self.optimizer.strip().lower().replace("-", "_")
        if optimizer not in MODEL_CALIBRATION_OPTIMIZERS:
            choices = ", ".join(MODEL_CALIBRATION_OPTIMIZERS)
            raise ValueError(f"optimizer must be one of: {choices}.")
        object.__setattr__(self, "optimizer", optimizer)
        object.__setattr__(self, "save_research_artifacts", bool(self.save_research_artifacts))


@dataclass(frozen=True)
class ModelCalibrationEvaluation:
    parameters: ModelCalibrationParameterSet
    loss: RoiLossResult
    simulated_x: NDArray[np.bool_]
    simulated_y: NDArray[np.bool_]
    result: SimulationResult
    elapsed_seconds: float = 0.0
    solver_seconds: float = 0.0
    roi_seconds: float = 0.0
    loss_seconds: float = 0.0
    target_x: NDArray[np.bool_] | None = None
    target_y: NDArray[np.bool_] | None = None
    target_x_path: Path | None = None
    target_y_path: Path | None = None


@dataclass(frozen=True)
class ModelCalibrationSampleResult:
    sample: str
    best: ModelCalibrationEvaluation
    evaluations: int
    elapsed_seconds: float
    solver_seconds: float = 0.0
    roi_seconds: float = 0.0
    loss_seconds: float = 0.0


@dataclass(frozen=True)
class ModelCalibrationRunResult:
    samples: tuple[ModelCalibrationSampleResult, ...]
    output_dir: Path | None
    elapsed_seconds: float
    target_load_seconds: float = 0.0
    voxelization_seconds: float = 0.0
    save_seconds: float = 0.0

    @property
    def average_loss(self) -> float:
        if not self.samples:
            return float("nan")
        return float(np.mean([sample.best.loss.total for sample in self.samples]))

    @property
    def solver_seconds(self) -> float:
        return float(sum(sample.solver_seconds for sample in self.samples))

    @property
    def roi_seconds(self) -> float:
        return float(sum(sample.roi_seconds for sample in self.samples))

    @property
    def loss_seconds(self) -> float:
        return float(sum(sample.loss_seconds for sample in self.samples))


def simulation_rois(
    binary_volume: NDArray[np.bool_],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    x_slice, y_slice = central_slices(np.asarray(binary_volume, dtype=bool))
    return extract_model_calibration_roi(x_slice), extract_model_calibration_roi(y_slice)


def validate_model_calibration_grid_resolution(grid: VoxelGrid) -> None:
    roi_x, roi_y = simulation_rois(grid.data)
    empty_axes = []
    if roi_x.size == 0:
        empty_axes.append("X")
    if roi_y.size == 0:
        empty_axes.append("Y")
    if not empty_axes:
        return

    axes = "/".join(empty_axes)
    raise ValueError(
        f"Model Calibration {axes} ROI window is empty for voxel grid {grid.shape} "
        f"at {grid.spacing:g} mm spacing. Reduce Grid spacing and run again."
    )


def discover_model_calibration_targets(
    sample_dir: str | Path,
    *,
    sample_names: set[str] | None = None,
) -> list[ModelCalibrationTarget]:
    folder = Path(sample_dir)
    if not folder.exists():
        raise FileNotFoundError(f"Sample ROI folder does not exist: {folder}")

    targets: list[ModelCalibrationTarget] = []
    x_paths = sorted(folder.glob("*_xSliceROI.tif"))
    for x_path in x_paths:
        sample = x_path.name[: -len("_xSliceROI.tif")]
        if sample_names is not None and sample not in sample_names:
            continue
        y_path = folder / f"{sample}_ySliceROI.tif"
        if not y_path.exists():
            y_path = folder / f"{sample}_zSliceROI.tif"
        if not y_path.exists():
            raise FileNotFoundError(f"Missing Y ROI pair for sample '{sample}' in {folder}")
        roi_x, roi_x_inverted = _read_roi_image_with_polarity(x_path)
        roi_y, roi_y_inverted = _read_roi_image_with_polarity(y_path)
        targets.append(
            ModelCalibrationTarget(
                sample=sample,
                roi_x=roi_x,
                roi_y=roi_y,
                roi_x_inverted=roi_x_inverted,
                roi_y_inverted=roi_y_inverted,
                roi_x_path=x_path,
                roi_y_path=y_path,
            )
        )

    if not targets:
        raise ValueError(f"No '*_xSliceROI.tif' model calibration targets found in {folder}")
    return targets


def evaluate_model_calibration_candidate(
    grid: VoxelGrid,
    target: ModelCalibrationTarget,
    parameters: ModelCalibrationParameterSet,
    *,
    pipeline: SimulationPipeline | None = None,
    options: ModelCalibrationOptions | None = None,
) -> ModelCalibrationEvaluation:
    started = perf_counter()
    options = options or ModelCalibrationOptions()
    solver_parameters = parameters.to_solver_parameters(
        backend=options.backend,
        rng_seed=options.rng_seed,
    )
    pipeline = pipeline or SimulationPipeline(solver=create_solver(solver_parameters))
    solver_started = perf_counter()
    result = pipeline.run_voxel_grid(grid, solver_parameters)
    solver_seconds = perf_counter() - solver_started

    roi_started = perf_counter()
    simulated_x, simulated_y = simulation_rois(result.binary)
    roi_seconds = perf_counter() - roi_started

    loss_started = perf_counter()
    loss = RoiBoundaryLoss(boundary_radius=options.boundary_radius).evaluate(
        target.roi_x,
        target.roi_y,
        simulated_x,
        simulated_y,
    )
    loss_seconds = perf_counter() - loss_started
    return ModelCalibrationEvaluation(
        parameters=parameters,
        loss=loss,
        simulated_x=simulated_x,
        simulated_y=simulated_y,
        result=result,
        elapsed_seconds=perf_counter() - started,
        solver_seconds=solver_seconds,
        roi_seconds=roi_seconds,
        loss_seconds=loss_seconds,
        target_x=target.roi_x,
        target_y=target.roi_y,
        target_x_path=target.roi_x_path,
        target_y_path=target.roi_y_path,
    )


def optimize_model_calibration_target(
    grid: VoxelGrid,
    target: ModelCalibrationTarget,
    *,
    options: ModelCalibrationOptions | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ModelCalibrationSampleResult:
    options = options or ModelCalibrationOptions()
    started = perf_counter()
    candidates = _candidate_sequence(options)
    seed_offset = _stable_sample_seed(target.sample)
    pipeline = SimulationPipeline(
        solver=create_solver(
            candidates[0].to_solver_parameters(
                backend=options.backend,
                rng_seed=_offset_seed(options.rng_seed, seed_offset),
            )
        )
    )

    best: ModelCalibrationEvaluation | None = None
    solver_seconds = 0.0
    roi_seconds = 0.0
    loss_seconds = 0.0
    for index, candidate in enumerate(candidates, start=1):
        candidate_options = ModelCalibrationOptions(
            max_evaluations=options.max_evaluations,
            backend=options.backend,
            rng_seed=_offset_seed(options.rng_seed, seed_offset + index),
            bounds=options.bounds,
            boundary_radius=options.boundary_radius,
            max_workers=1,
            optimizer=options.optimizer,
            save_research_artifacts=options.save_research_artifacts,
        )
        evaluation = evaluate_model_calibration_candidate(
            grid,
            target,
            candidate,
            pipeline=pipeline,
            options=candidate_options,
        )
        solver_seconds += evaluation.solver_seconds
        roi_seconds += evaluation.roi_seconds
        loss_seconds += evaluation.loss_seconds
        if best is None or evaluation.loss.total < best.loss.total:
            best = evaluation
        if progress_callback is not None:
            progress_callback(
                int(index * 100 / len(candidates)),
                (
                    f"{target.sample}: evaluation {index}/{len(candidates)}, "
                    f"last {evaluation.elapsed_seconds:.2f}s "
                    f"(solver {evaluation.solver_seconds:.2f}s, "
                    f"ROI/loss {evaluation.roi_seconds + evaluation.loss_seconds:.2f}s), "
                    f"best loss {best.loss.total:.4g}"
                ),
            )

    if best is None:
        raise RuntimeError("Model calibration finished without evaluating any candidate.")

    return ModelCalibrationSampleResult(
        sample=target.sample,
        best=best,
        evaluations=len(candidates),
        elapsed_seconds=perf_counter() - started,
        solver_seconds=solver_seconds,
        roi_seconds=roi_seconds,
        loss_seconds=loss_seconds,
    )


def run_model_calibration(
    grid: VoxelGrid,
    targets: list[ModelCalibrationTarget],
    *,
    output_dir: str | Path | None = None,
    options: ModelCalibrationOptions | None = None,
    calibration_geometry_path: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ModelCalibrationRunResult:
    if not targets:
        raise ValueError("At least one model calibration target is required.")
    options = options or ModelCalibrationOptions()
    validate_model_calibration_grid_resolution(grid)
    started = perf_counter()
    worker_count = min(options.max_workers, options.max_evaluations)
    if options.backend is SolverBackend.CUDA:
        worker_count = 1
    samples = _run_model_calibration_shared_candidates(
        grid,
        targets,
        options=options,
        worker_count=worker_count,
        progress_callback=progress_callback,
    )

    output_path = Path(output_dir) if output_dir is not None else None
    save_seconds = 0.0
    run_result = ModelCalibrationRunResult(
        samples=tuple(samples),
        output_dir=output_path,
        elapsed_seconds=perf_counter() - started,
    )
    if output_path is not None:
        if progress_callback is not None:
            progress_callback(98, "Saving Model Calibration outputs")
        save_started = perf_counter()
        save_model_calibration_outputs(
            output_path,
            run_result,
            save_research_artifacts=options.save_research_artifacts,
            calibration_geometry_path=calibration_geometry_path,
            progress_callback=progress_callback,
        )
        save_seconds = perf_counter() - save_started
        run_result = ModelCalibrationRunResult(
            samples=tuple(samples),
            output_dir=output_path,
            elapsed_seconds=perf_counter() - started,
            save_seconds=save_seconds,
        )
    if progress_callback is not None:
        progress_callback(
            100,
            (
                "Model calibration complete "
                f"(solver CPU time {run_result.solver_seconds:.2f}s, "
                f"ROI/loss {run_result.roi_seconds + run_result.loss_seconds:.2f}s, "
                f"save {run_result.save_seconds:.2f}s)"
            ),
        )
    return run_result


def run_model_calibration_from_paths(
    *,
    geometry_path: str | Path,
    sample_dir: str | Path,
    voxel_spacing: float,
    output_dir: str | Path,
    options: ModelCalibrationOptions | None = None,
    sample_names: set[str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ModelCalibrationRunResult:
    started = perf_counter()
    if progress_callback is not None:
        progress_callback(0, "Loading model calibration targets")
    target_started = perf_counter()
    targets = discover_model_calibration_targets(sample_dir, sample_names=sample_names)
    target_load_seconds = perf_counter() - target_started
    if progress_callback is not None:
        inverted = [
            f"{target.sample}:{axis}"
            for target in targets
            for axis, was_inverted in (("x", target.roi_x_inverted), ("y", target.roi_y_inverted))
            if was_inverted
        ]
        if inverted:
            preview = ", ".join(inverted[:6])
            suffix = f" (+{len(inverted) - 6} more)" if len(inverted) > 6 else ""
            progress_callback(
                3,
                f"Applied ROI TIFF WhiteIsZero interpretation to "
                f"{len(inverted)} slice(s): {preview}{suffix}",
            )
        else:
            progress_callback(3, "ROI TIFF polarity checked")

    if progress_callback is not None:
        progress_callback(5, "Voxelizing calibration geometry")
    voxel_started = perf_counter()
    grid = voxelize_mesh(
        geometry_path,
        spacing=voxel_spacing,
        progress_callback=(
            _scale_progress(progress_callback, 5, 25, "Voxelizing calibration geometry")
            if progress_callback is not None
            else None
        ),
    )
    voxelization_seconds = perf_counter() - voxel_started

    def calibration_progress(percent: int, message: str) -> None:
        if progress_callback is None:
            return
        progress_callback(25 + int(max(0, min(100, percent)) * 0.70), message)

    result = run_model_calibration(
        grid,
        targets,
        output_dir=output_dir,
        options=options,
        calibration_geometry_path=geometry_path,
        progress_callback=calibration_progress,
    )
    result = ModelCalibrationRunResult(
        samples=result.samples,
        output_dir=result.output_dir,
        elapsed_seconds=perf_counter() - started,
        target_load_seconds=target_load_seconds,
        voxelization_seconds=voxelization_seconds,
        save_seconds=result.save_seconds,
    )
    if progress_callback is not None:
        progress_callback(
            100,
            (
                f"Model calibration outputs saved: {output_dir} "
                f"(load {result.target_load_seconds:.2f}s, "
                f"voxelize {result.voxelization_seconds:.2f}s, "
                f"solver CPU time {result.solver_seconds:.2f}s, "
                f"ROI/loss {result.roi_seconds + result.loss_seconds:.2f}s, "
                f"save {result.save_seconds:.2f}s)"
            ),
        )
    return result


def save_model_calibration_outputs(
    output_dir: str | Path,
    result: ModelCalibrationRunResult,
    *,
    save_research_artifacts: bool = False,
    include_volume_arrays: bool = False,
    calibration_geometry_path: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> None:
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / "model_calibration_weights.csv"
    if progress_callback is not None:
        progress_callback(98, f"Writing {csv_path.name}")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Sample",
                "param1",
                "param2",
                "param3",
                "param4",
                "param5",
                "param6",
                "Loss",
                "DiceX",
                "DiceY",
                "IoUX",
                "IoUY",
                "OverlapLossX",
                "OverlapLossY",
                "BoundaryLossX",
                "BoundaryLossY",
                "AreaLossX",
                "AreaLossY",
            ]
        )
        for sample in result.samples:
            writer.writerow(
                [
                    sample.sample,
                    *sample.best.parameters.as_tuple(),
                    sample.best.loss.total,
                    sample.best.loss.x_dice,
                    sample.best.loss.y_dice,
                    sample.best.loss.x_iou,
                    sample.best.loss.y_iou,
                    sample.best.loss.x_overlap_loss,
                    sample.best.loss.y_overlap_loss,
                    sample.best.loss.x_boundary_loss,
                    sample.best.loss.y_boundary_loss,
                    sample.best.loss.x_area_loss,
                    sample.best.loss.y_area_loss,
                ]
            )

    total_samples = max(1, len(result.samples))
    for index, sample in enumerate(result.samples, start=1):
        if progress_callback is not None:
            progress_callback(
                98 + int(index * 2 / total_samples),
                f"Writing artifacts for {sample.sample}",
            )
        simulation = sample.best.result
        artifact_payload = {
            "simulated_x": sample.best.simulated_x.astype(np.uint8),
            "simulated_y": sample.best.simulated_y.astype(np.uint8),
            "loss_x": sample.best.loss.x_map,
            "loss_y": sample.best.loss.y_map,
            "parameters": np.asarray(sample.best.parameters.as_tuple(), dtype=np.float64),
            "loss": np.asarray([sample.best.loss.total], dtype=np.float64),
            "loss_components": np.asarray(
                [
                    sample.best.loss.x_overlap_loss,
                    sample.best.loss.y_overlap_loss,
                    sample.best.loss.x_boundary_loss,
                    sample.best.loss.y_boundary_loss,
                    sample.best.loss.x_area_loss,
                    sample.best.loss.y_area_loss,
                    sample.best.loss.x_dice,
                    sample.best.loss.y_dice,
                    sample.best.loss.x_iou,
                    sample.best.loss.y_iou,
                ],
                dtype=np.float64,
            ),
            "volume_shape": np.asarray(simulation.probability.shape, dtype=np.int64),
            "spacing": np.asarray([simulation.spacing], dtype=np.float64),
            "origin": np.asarray(simulation.origin, dtype=np.float64),
            "rest_volume": np.asarray([simulation.rest_volume], dtype=np.float64),
            "probability_density": np.asarray([simulation.probability_density], dtype=np.float64),
            "binary_voxels": np.asarray([np.count_nonzero(simulation.binary)], dtype=np.int64),
            "input_voxels": np.asarray([np.count_nonzero(simulation.voxel)], dtype=np.int64),
        }
        if include_volume_arrays:
            artifact_payload.update(
                {
                    "probability": simulation.probability,
                    "binary": simulation.binary.astype(np.uint8),
                    "voxel": simulation.voxel.astype(np.uint8),
                }
            )
        if sample.best.target_x is not None:
            artifact_payload["target_x"] = sample.best.target_x.astype(np.uint8)
        if sample.best.target_y is not None:
            artifact_payload["target_y"] = sample.best.target_y.astype(np.uint8)
        np.savez_compressed(
            folder / f"{sample.sample}_model_calibration_artifacts.npz",
            **artifact_payload,
        )
    if save_research_artifacts:
        export_model_calibration_research_artifacts(
            folder,
            result,
            calibration_geometry_path=calibration_geometry_path,
        )


def export_model_calibration_research_artifacts(
    output_dir: str | Path,
    result: ModelCalibrationRunResult,
    *,
    calibration_geometry_path: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    folder = Path(output_dir)
    total_samples = max(1, len(result.samples))
    for index, sample in enumerate(result.samples, start=1):
        base = int((index - 1) * 100 / total_samples)
        limit = int(index * 100 / total_samples)

        def sample_progress(
            fraction: float,
            message: str,
            *,
            progress_base: int = base,
            progress_limit: int = limit,
            sample_name: str = sample.sample,
        ) -> None:
            if progress_callback is None:
                return
            span = progress_limit - progress_base
            percent = progress_base + int(span * max(0.0, min(1.0, fraction)))
            progress_callback(percent, f"{sample_name}: {message}")

        if progress_callback is not None:
            progress_callback(
                base,
                f"{sample.sample}: exporting research artifacts",
            )
        _save_model_calibration_research_artifacts(
            folder,
            sample,
            calibration_geometry_path=calibration_geometry_path,
            progress_callback=sample_progress,
        )
    return folder / "research_artifacts"


def _save_model_calibration_research_artifacts(
    output_dir: Path,
    sample: ModelCalibrationSampleResult,
    *,
    calibration_geometry_path: str | Path | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> None:
    sample_dir = output_dir / "research_artifacts" / _safe_sample_name(sample.sample)
    roi_dir = sample_dir / "roi"
    geometry_dir = sample_dir / "geometry"
    roi_dir.mkdir(parents=True, exist_ok=True)
    geometry_dir.mkdir(parents=True, exist_ok=True)

    best = sample.best
    _emit_export_progress(progress_callback, 0.05, "writing ROI masks")
    _copy_if_available(best.target_x_path, roi_dir / "target_x_original.tif")
    _copy_if_available(best.target_y_path, roi_dir / "target_y_original.tif")
    if best.target_x is not None:
        _write_mask_tiff(roi_dir / "target_x_mask.tif", best.target_x)
    if best.target_y is not None:
        _write_mask_tiff(roi_dir / "target_y_mask.tif", best.target_y)
    _write_mask_tiff(roi_dir / "simulated_x_mask.tif", best.simulated_x)
    _write_mask_tiff(roi_dir / "simulated_y_mask.tif", best.simulated_y)

    if calibration_geometry_path is not None:
        source_geometry = Path(calibration_geometry_path)
        if source_geometry.exists():
            _emit_export_progress(progress_callback, 0.12, "copying input geometry")
            _copy_if_available(source_geometry, geometry_dir / f"input{source_geometry.suffix}")

    result = best.result
    _emit_export_progress(progress_callback, 0.2, "writing probability volume")
    write_vtk_volume(
        geometry_dir / "best_probability.vtk",
        result.probability,
        spacing=result.spacing,
        origin=result.origin,
        scalar_name="Probability",
    )
    _emit_export_progress(progress_callback, 0.5, "writing binary volume")
    write_vtk_volume(
        geometry_dir / "best_binary.vtk",
        result.binary.astype(np.uint8),
        spacing=result.spacing,
        origin=result.origin,
        scalar_name="Binary",
    )
    _emit_export_progress(progress_callback, 0.75, "writing STL surface")
    write_binary_stl(
        geometry_dir / "best_binary.stl",
        result.binary,
        spacing=result.spacing,
        origin=result.origin,
    )
    _emit_export_progress(progress_callback, 1.0, "research artifacts exported")


def _emit_export_progress(
    progress_callback: Callable[[float, str], None] | None,
    fraction: float,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(fraction, message)


def _copy_if_available(source: str | Path | None, destination: Path) -> None:
    if source is None:
        return
    source_path = Path(source)
    if source_path.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(source_path, destination)


def _write_mask_tiff(path: Path, mask: NDArray[np.bool_]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path)


def _safe_sample_name(sample: str) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", sample.strip())
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._ ")
    return value or "sample"


def _candidate_sequence(options: ModelCalibrationOptions) -> list[ModelCalibrationParameterSet]:
    count = (
        _adaptive_initial_count(options)
        if options.optimizer == "adaptive_sobol"
        else options.max_evaluations
    )
    return _global_candidate_sequence(options, count)


def _adaptive_initial_count(options: ModelCalibrationOptions) -> int:
    if options.max_evaluations <= 8:
        return options.max_evaluations
    return min(options.max_evaluations, max(8, int(np.ceil(options.max_evaluations * 0.5))))


def _global_candidate_sequence(
    options: ModelCalibrationOptions,
    count: int,
) -> list[ModelCalibrationParameterSet]:
    bounds = np.asarray(options.bounds.as_pairs(), dtype=np.float64)
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    candidates = [ModelCalibrationParameterSet(0.2, 0.2, 0.2, 0.2, 0.05, 0.3)]
    remaining = count - len(candidates)
    if remaining > 0:
        if options.optimizer == "latin_hypercube":
            sampler = qmc.LatinHypercube(d=6, seed=options.rng_seed)
            unit_samples = sampler.random(remaining)
        else:
            sampler = qmc.Sobol(d=6, scramble=True, seed=options.rng_seed)
            power = int(np.ceil(np.log2(max(1, remaining))))
            unit_samples = sampler.random_base2(power)[:remaining]
        samples = qmc.scale(unit_samples, lower, upper)
        candidates.extend(ModelCalibrationParameterSet.from_sequence(row) for row in samples)
    return candidates[:count]


def _candidate_key(candidate: ModelCalibrationParameterSet) -> tuple[float, ...]:
    return tuple(round(value, 8) for value in candidate.as_tuple())


def _adaptive_candidate_batch(
    options: ModelCalibrationOptions,
    best_by_sample: list[ModelCalibrationEvaluation | None],
    *,
    seen: set[tuple[float, ...]],
    remaining: int,
    round_index: int,
) -> list[ModelCalibrationParameterSet]:
    bounds = np.asarray(options.bounds.as_pairs(), dtype=np.float64)
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    span = upper - lower
    centers = _adaptive_centers(best_by_sample)
    if not centers:
        centers = [ModelCalibrationParameterSet(0.2, 0.2, 0.2, 0.2, 0.05, 0.3)]

    scales = (0.25, 0.14, 0.08, 0.045, 0.025)
    scale = scales[min(round_index, len(scales) - 1)]
    rng = np.random.default_rng(_offset_seed(options.rng_seed, 10_000 + round_index))
    candidates: list[ModelCalibrationParameterSet] = []

    def add_candidate(values: NDArray[np.float64]) -> None:
        if len(candidates) >= remaining:
            return
        values = np.clip(values, lower, upper)
        candidate = ModelCalibrationParameterSet.from_sequence(values)
        key = _candidate_key(candidate)
        if key in seen:
            return
        seen.add(key)
        candidates.append(candidate)

    for center in centers:
        center_values = np.asarray(center.as_tuple(), dtype=np.float64)
        for dimension in range(center_values.size):
            offset = np.zeros_like(center_values)
            offset[dimension] = span[dimension] * scale
            add_candidate(center_values + offset)
            add_candidate(center_values - offset)
            if len(candidates) >= remaining:
                return candidates
        for _ in range(4):
            noise = rng.normal(loc=0.0, scale=scale, size=center_values.size) * span
            add_candidate(center_values + noise)
            if len(candidates) >= remaining:
                return candidates

    fallback_round = 0
    while len(candidates) < remaining and fallback_round < 8:
        fallback_round += 1
        fallback_options = ModelCalibrationOptions(
            max_evaluations=remaining + 1,
            backend=options.backend,
            rng_seed=_offset_seed(options.rng_seed, 20_000 + round_index * 100 + fallback_round),
            bounds=options.bounds,
            boundary_radius=options.boundary_radius,
            max_workers=options.max_workers,
            optimizer="sobol",
        )
        for candidate in _global_candidate_sequence(fallback_options, remaining + 1):
            add_candidate(np.asarray(candidate.as_tuple(), dtype=np.float64))
            if len(candidates) >= remaining:
                break
    return candidates


def _adaptive_centers(
    best_by_sample: list[ModelCalibrationEvaluation | None],
) -> list[ModelCalibrationParameterSet]:
    ranked: list[tuple[float, ModelCalibrationParameterSet]] = []
    seen: set[tuple[float, ...]] = set()
    for evaluation in best_by_sample:
        if evaluation is None:
            continue
        key = _candidate_key(evaluation.parameters)
        if key in seen:
            continue
        seen.add(key)
        ranked.append((evaluation.loss.total, evaluation.parameters))
    ranked.sort(key=lambda item: item[0])
    return [parameters for _, parameters in ranked]


def _run_model_calibration_sequential(
    grid: VoxelGrid,
    targets: list[ModelCalibrationTarget],
    *,
    options: ModelCalibrationOptions,
    progress_callback: ProgressCallback | None,
) -> list[ModelCalibrationSampleResult]:
    samples: list[ModelCalibrationSampleResult] = []
    for sample_index, target in enumerate(targets):
        if progress_callback is not None:
            progress_callback(
                int(sample_index * 100 / len(targets)),
                f"Starting model calibration sample {target.sample}",
            )

        def sample_progress(
            percent: int,
            message: str,
            sample_index: int = sample_index,
        ) -> None:
            if progress_callback is None:
                return
            overall = int((sample_index + max(0, min(100, percent)) / 100) * 100 / len(targets))
            progress_callback(overall, message)

        samples.append(
            optimize_model_calibration_target(
                grid,
                target,
                options=options,
                progress_callback=sample_progress,
            )
        )
    return samples


def _run_model_calibration_parallel(
    grid: VoxelGrid,
    targets: list[ModelCalibrationTarget],
    *,
    options: ModelCalibrationOptions,
    worker_count: int,
    progress_callback: ProgressCallback | None,
) -> list[ModelCalibrationSampleResult]:
    progress_by_sample = [0] * len(targets)
    results: list[ModelCalibrationSampleResult | None] = [None] * len(targets)
    progress_lock = Lock()

    def update_progress(sample_index: int, percent: int, message: str) -> None:
        with progress_lock:
            progress_by_sample[sample_index] = max(0, min(100, int(percent)))
            overall = int(sum(progress_by_sample) / len(progress_by_sample))
        if progress_callback is not None:
            progress_callback(overall, f"[{targets[sample_index].sample}] {message}")

    def run_one(sample_index: int, target: ModelCalibrationTarget) -> ModelCalibrationSampleResult:
        update_progress(sample_index, 0, "starting")

        def sample_progress(percent: int, message: str) -> None:
            update_progress(sample_index, percent, message)

        sample_result = optimize_model_calibration_target(
            grid,
            target,
            options=options,
            progress_callback=sample_progress,
        )
        update_progress(sample_index, 100, "complete")
        return sample_result

    if progress_callback is not None:
        progress_callback(0, f"Running {len(targets)} samples on {worker_count} workers")

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(run_one, sample_index, target): sample_index
            for sample_index, target in enumerate(targets)
        }
        for future in as_completed(future_to_index):
            sample_index = future_to_index[future]
            results[sample_index] = future.result()

    return [sample for sample in results if sample is not None]


def _run_model_calibration_shared_candidates(
    grid: VoxelGrid,
    targets: list[ModelCalibrationTarget],
    *,
    options: ModelCalibrationOptions,
    worker_count: int,
    progress_callback: ProgressCallback | None,
) -> list[ModelCalibrationSampleResult]:
    started = perf_counter()
    candidates = _candidate_sequence(options)
    seen_candidates = {_candidate_key(candidate) for candidate in candidates}
    best_by_sample: list[ModelCalibrationEvaluation | None] = [None] * len(targets)
    loss_seconds_by_sample = [0.0] * len(targets)
    completed = 0
    completed_lock = Lock()

    def report_completion(message: str) -> None:
        nonlocal completed
        with completed_lock:
            completed += 1
            percent = int(completed * 100 / options.max_evaluations)
        if progress_callback is not None:
            progress_callback(percent, message)

    def run_candidate(
        candidate_index: int,
        candidate: ModelCalibrationParameterSet,
    ) -> tuple[int, list[ModelCalibrationEvaluation], float, float, float]:
        with completed_lock:
            running_percent = int(completed * 100 / options.max_evaluations)
        if progress_callback is not None:
            progress_callback(
                running_percent,
                (
                    f"Running candidate {candidate_index}/{options.max_evaluations} "
                    f"with {_SOLVER_LABELS.get(options.backend, options.backend.value)}"
                ),
            )
        candidate_options = ModelCalibrationOptions(
            max_evaluations=options.max_evaluations,
            backend=options.backend,
            rng_seed=_offset_seed(options.rng_seed, candidate_index),
            bounds=options.bounds,
            boundary_radius=options.boundary_radius,
            max_workers=1,
            optimizer=options.optimizer,
        )
        solver_parameters = candidate.to_solver_parameters(
            backend=candidate_options.backend,
            rng_seed=candidate_options.rng_seed,
        )
        pipeline = SimulationPipeline(solver=create_solver(solver_parameters))

        solver_progress_callback = None
        if progress_callback is not None:

            def solver_progress(percent: int, message: str) -> None:
                with completed_lock:
                    running_fraction = completed + min(0.95, max(0, min(100, percent)) / 100)
                progress_callback(
                    int(running_fraction * 100 / options.max_evaluations),
                    f"Candidate {candidate_index}/{options.max_evaluations}: {message}",
                )

            solver_progress_callback = solver_progress

        candidate_started = perf_counter()
        solver_started = perf_counter()
        result = pipeline.run_voxel_grid(
            grid,
            solver_parameters,
            progress_callback=solver_progress_callback,
        )
        solver_seconds = perf_counter() - solver_started

        roi_started = perf_counter()
        simulated_x, simulated_y = simulation_rois(result.binary)
        roi_seconds = perf_counter() - roi_started

        evaluations: list[ModelCalibrationEvaluation] = []
        loss_seconds = 0.0
        loss_evaluator = RoiBoundaryLoss(boundary_radius=candidate_options.boundary_radius)
        for target in targets:
            loss_started = perf_counter()
            loss = loss_evaluator.evaluate(
                target.roi_x,
                target.roi_y,
                simulated_x,
                simulated_y,
            )
            target_loss_seconds = perf_counter() - loss_started
            loss_seconds += target_loss_seconds
            evaluations.append(
                ModelCalibrationEvaluation(
                    parameters=candidate,
                    loss=loss,
                    simulated_x=simulated_x,
                    simulated_y=simulated_y,
                    result=result,
                    elapsed_seconds=perf_counter() - candidate_started,
                    solver_seconds=solver_seconds,
                    roi_seconds=roi_seconds,
                    loss_seconds=target_loss_seconds,
                    target_x=target.roi_x,
                    target_y=target.roi_y,
                )
            )
        return candidate_index, evaluations, solver_seconds, roi_seconds, loss_seconds

    if progress_callback is not None:
        progress_callback(
            0,
            (
                f"Running {options.optimizer.replace('_', ' ')} search: "
                f"{options.max_evaluations} shared candidate simulations "
                f"for {len(targets)} samples on {worker_count} workers"
            ),
        )

    total_solver_seconds = 0.0
    total_roi_seconds = 0.0

    def consume_candidate_result(
        candidate_index: int,
        evaluations: list[ModelCalibrationEvaluation],
        solver_seconds: float,
        roi_seconds: float,
        loss_seconds: float,
        phase: str,
    ) -> None:
        nonlocal total_solver_seconds, total_roi_seconds
        total_solver_seconds += solver_seconds
        total_roi_seconds += roi_seconds
        _merge_candidate_evaluations(evaluations, best_by_sample, loss_seconds_by_sample)
        best_loss = min(
            evaluation.loss.total for evaluation in best_by_sample if evaluation is not None
        )
        report_completion(
            f"{phase} candidate {candidate_index}/{options.max_evaluations} shared across "
            f"{len(targets)} samples: solver {solver_seconds:.2f}s, "
            f"ROI/loss {roi_seconds + loss_seconds:.2f}s, best loss {best_loss:.4g}"
        )

    def evaluate_batch(
        candidate_batch: list[tuple[int, ModelCalibrationParameterSet]],
        phase: str,
    ) -> None:
        if worker_count == 1 or len(candidate_batch) == 1:
            for candidate_index, candidate in candidate_batch:
                _, evaluations, solver_seconds, roi_seconds, loss_seconds = run_candidate(
                    candidate_index,
                    candidate,
                )
                consume_candidate_result(
                    candidate_index,
                    evaluations,
                    solver_seconds,
                    roi_seconds,
                    loss_seconds,
                    phase,
                )
            return

        with ThreadPoolExecutor(max_workers=min(worker_count, len(candidate_batch))) as executor:
            future_to_candidate = {
                executor.submit(run_candidate, candidate_index, candidate): candidate_index
                for candidate_index, candidate in candidate_batch
            }
            for future in as_completed(future_to_candidate):
                candidate_index = future_to_candidate[future]
                _, evaluations, solver_seconds, roi_seconds, loss_seconds = future.result()
                consume_candidate_result(
                    candidate_index,
                    evaluations,
                    solver_seconds,
                    roi_seconds,
                    loss_seconds,
                    phase,
                )

    evaluate_batch(
        [(index, candidate) for index, candidate in enumerate(candidates, start=1)],
        "global",
    )
    next_candidate_index = len(candidates) + 1
    round_index = 0
    while options.optimizer == "adaptive_sobol" and completed < options.max_evaluations:
        remaining = options.max_evaluations - completed
        batch_size = min(remaining, max(2, worker_count * 2))
        local_candidates = _adaptive_candidate_batch(
            options,
            best_by_sample,
            seen=seen_candidates,
            remaining=batch_size,
            round_index=round_index,
        )
        if not local_candidates:
            break
        evaluate_batch(
            [
                (next_candidate_index + offset, candidate)
                for offset, candidate in enumerate(local_candidates)
            ],
            "adaptive",
        )
        next_candidate_index += len(local_candidates)
        round_index += 1

    elapsed_seconds = perf_counter() - started
    solver_share = total_solver_seconds / len(targets)
    roi_share = total_roi_seconds / len(targets)
    completed_count = max(1, completed)
    samples: list[ModelCalibrationSampleResult] = []
    for sample_index, target in enumerate(targets):
        best = best_by_sample[sample_index]
        if best is None:
            raise RuntimeError(f"No candidate evaluation completed for sample {target.sample}.")
        samples.append(
            ModelCalibrationSampleResult(
                sample=target.sample,
                best=best,
                evaluations=completed_count,
                elapsed_seconds=elapsed_seconds,
                solver_seconds=solver_share,
                roi_seconds=roi_share,
                loss_seconds=loss_seconds_by_sample[sample_index],
            )
        )
    return samples


def _merge_candidate_evaluations(
    evaluations: list[ModelCalibrationEvaluation],
    best_by_sample: list[ModelCalibrationEvaluation | None],
    loss_seconds_by_sample: list[float],
) -> None:
    for sample_index, evaluation in enumerate(evaluations):
        loss_seconds_by_sample[sample_index] += evaluation.loss_seconds
        best = best_by_sample[sample_index]
        if best is None or evaluation.loss.total < best.loss.total:
            best_by_sample[sample_index] = evaluation


def _read_roi_image(path: Path) -> NDArray[np.bool_]:
    return _read_roi_image_with_polarity(path)[0]


def _read_roi_image_with_polarity(path: Path) -> tuple[NDArray[np.bool_], bool]:
    # MATLAB writes logical ROI TIFFs as 1-bit images, and some exports use
    # PhotometricInterpretation=WhiteIsZero. Pillow applies that photometric
    # interpretation during conversion; tifffile/skimage expose the stored bits
    # more directly, which makes those targets appear inverted in the workbench.
    with Image.open(path) as image:
        photometric = image.tag_v2.get(262)
        array = np.asarray(image.convert("L"))
    if array.ndim > 2:
        array = array[..., 0]
    return np.asarray(array != 0, dtype=bool), photometric == 0


def _stable_sample_seed(sample: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(sample))


def _offset_seed(seed: int | None, offset: int) -> int | None:
    if seed is None:
        return None
    return int(seed) + int(offset)


def _scale_progress(
    callback: ProgressCallback,
    start: int,
    end: int,
    fallback_message: str,
) -> ProgressCallback:
    def scaled(percent: int, message: str) -> None:
        value = start + int((end - start) * max(0, min(100, percent)) / 100)
        callback(value, message or fallback_message)

    return scaled
