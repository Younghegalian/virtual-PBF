import numpy as np
import tifffile
from skimage.io import imsave

from capp.calibration.losses import RoiLossResult
from capp.calibration.model_calibration import (
    ModelCalibrationBounds,
    ModelCalibrationEvaluation,
    ModelCalibrationOptions,
    ModelCalibrationParameterSet,
    ModelCalibrationRunResult,
    ModelCalibrationSampleResult,
    discover_model_calibration_targets,
    run_model_calibration,
    save_model_calibration_outputs,
    simulation_rois,
    validate_model_calibration_grid_resolution,
)
from capp.calibration.roi import extract_model_calibration_roi
from capp.domain import NeighborhoodModel, SimulationResult, SolverBackend, VoxelGrid


def test_extract_model_calibration_roi_matches_matlab_fixed_window():
    image = np.zeros((100, 120), dtype=bool)
    image[30:40, 45:55] = True

    roi = extract_model_calibration_roi(image)

    assert roi.shape == (23, 57)
    assert roi[1:11, 13:23].all()


def test_model_calibration_parameter_set_builds_solver_parameters():
    params = ModelCalibrationParameterSet(0.11, 0.12, 0.13, 0.14, 0.02, 0.33)

    solver = params.to_solver_parameters(backend=SolverBackend.CPU_REFERENCE, rng_seed=7)

    assert solver.neighborhood is NeighborhoodModel.DIRECTIONAL_VON_NEUMANN
    assert solver.current_coefficients == (0.11, 0.12, 0.13, 0.14)
    assert solver.min_bias == 0.02
    assert solver.initial_deviation == 0.33
    assert solver.backend is SolverBackend.CPU_REFERENCE
    assert solver.rng_seed == 7


def test_model_calibration_bounds_match_current_matlab_search_space():
    bounds = ModelCalibrationBounds().as_pairs()

    assert bounds == (
        (0.05, 0.4),
        (0.05, 0.4),
        (0.05, 0.4),
        (0.05, 0.4),
        (0.005, 0.25),
        (0.1, 0.6),
    )


def test_simulation_rois_extracts_x_and_y_windows():
    volume = np.zeros((12, 14, 100), dtype=bool)
    volume[:, 7, 40:60] = True
    volume[6, :, 45:65] = True

    roi_x, roi_y = simulation_rois(volume)

    assert roi_x.ndim == 2
    assert roi_y.ndim == 2
    assert roi_x.shape[0] > 0
    assert roi_y.shape[0] > 0


def test_model_calibration_rejects_grid_too_coarse_for_roi_window():
    coarse_grid = VoxelGrid(data=np.ones((84, 24, 22), dtype=bool), spacing=0.5)

    try:
        validate_model_calibration_grid_resolution(coarse_grid)
    except ValueError as exc:
        assert "ROI window is empty" in str(exc)
        assert "Reduce Grid spacing" in str(exc)
    else:
        raise AssertionError("Expected coarse model calibration grid to be rejected.")


def test_discover_model_calibration_targets_reads_y_or_z_roi_pairs(tmp_path):
    x_roi = np.zeros((6, 8), dtype=np.uint8)
    z_roi = np.zeros((6, 8), dtype=np.uint8)
    x_roi[2:4, 3:5] = 255
    z_roi[1:3, 2:6] = 255
    imsave(tmp_path / "A1_xSliceROI.tif", x_roi)
    imsave(tmp_path / "A1_zSliceROI.tif", z_roi)

    targets = discover_model_calibration_targets(tmp_path)

    assert len(targets) == 1
    assert targets[0].sample == "A1"
    assert targets[0].roi_x.dtype == bool
    assert targets[0].roi_y.sum() == 8
    assert targets[0].roi_x_path == tmp_path / "A1_xSliceROI.tif"
    assert targets[0].roi_y_path == tmp_path / "A1_zSliceROI.tif"


def test_discover_model_calibration_targets_respects_tiff_photometric(tmp_path):
    x_roi = np.zeros((8, 10), dtype=bool)
    z_roi = np.zeros((8, 10), dtype=bool)
    x_roi[2:6, 3:7] = True
    z_roi[1:4, 2:5] = True
    tifffile.imwrite(tmp_path / "A1_xSliceROI.tif", x_roi, photometric="miniswhite")
    tifffile.imwrite(tmp_path / "A1_zSliceROI.tif", z_roi, photometric="miniswhite")

    targets = discover_model_calibration_targets(tmp_path)

    assert targets[0].roi_x.sum() == x_roi.size - 16
    assert targets[0].roi_y.sum() == z_roi.size - 9
    assert targets[0].roi_x[0, 0]
    assert targets[0].roi_x_inverted
    assert targets[0].roi_y_inverted


def test_save_model_calibration_outputs_writes_weights_and_artifacts(tmp_path):
    raw_x = tmp_path / "raw_x.tif"
    raw_y = tmp_path / "raw_y.tif"
    imsave(raw_x, np.eye(2, dtype=np.uint8) * 255)
    imsave(raw_y, np.fliplr(np.eye(2, dtype=np.uint8)) * 255)
    volume = np.ones((3, 3, 3), dtype=bool)
    simulation = SimulationResult(
        probability=np.full((3, 3, 3), 100, dtype=np.uint8),
        binary=volume,
        voxel=volume,
        spacing=0.1,
        origin=(0.0, 0.0, 0.0),
        rest_volume=100.0,
        probability_density=100.0,
        elapsed_seconds=0.01,
    )
    loss = RoiLossResult(
        total=1.5,
        x_loss=1.0,
        y_loss=0.5,
        x_map=np.zeros((2, 2), dtype=np.float64),
        y_map=np.zeros((2, 2), dtype=np.float64),
        valid=True,
    )
    sample = ModelCalibrationSampleResult(
        sample="A1",
        best=ModelCalibrationEvaluation(
            parameters=ModelCalibrationParameterSet(0.1, 0.2, 0.3, 0.4, 0.05, 0.6),
            loss=loss,
            simulated_x=np.ones((2, 2), dtype=bool),
            simulated_y=np.ones((2, 2), dtype=bool),
            result=simulation,
            target_x=np.eye(2, dtype=bool),
            target_y=np.fliplr(np.eye(2, dtype=bool)),
            target_x_path=raw_x,
            target_y_path=raw_y,
        ),
        evaluations=1,
        elapsed_seconds=0.01,
    )
    result = ModelCalibrationRunResult(samples=(sample,), output_dir=tmp_path, elapsed_seconds=0.02)

    save_model_calibration_outputs(tmp_path, result, save_research_artifacts=True)

    csv_text = (tmp_path / "model_calibration_weights.csv").read_text(encoding="utf-8")
    assert "Sample,param1,param2,param3,param4,param5,param6,Loss" in csv_text
    assert "A1,0.1,0.2,0.3,0.4,0.05,0.6,1.5" in csv_text
    artifact_path = tmp_path / "A1_model_calibration_artifacts.npz"
    assert artifact_path.exists()
    artifact = np.load(artifact_path)
    assert artifact["target_x"].sum() == 2
    assert artifact["target_y"].sum() == 2
    research_dir = tmp_path / "research_artifacts" / "A1"
    assert (research_dir / "roi" / "target_x_original.tif").exists()
    assert (research_dir / "roi" / "target_y_original.tif").exists()
    assert (research_dir / "roi" / "target_x_mask.tif").exists()
    assert (research_dir / "roi" / "simulated_y_mask.tif").exists()
    assert (research_dir / "geometry" / "best_binary.vtk").exists()
    assert (research_dir / "geometry" / "best_probability.vtk").exists()
    assert (research_dir / "geometry" / "best_binary.stl").exists()


def test_run_model_calibration_reuses_candidate_simulations_across_samples(monkeypatch):
    from capp.calibration import model_calibration as module

    calls = []
    roi = np.zeros((23, 57), dtype=bool)
    roi[5:15, 20:30] = True

    class FakePipeline:
        def __init__(self, solver):
            self.solver = solver

        def run_voxel_grid(self, grid, parameters, progress_callback=None):
            calls.append(parameters.current_coefficients)
            volume = np.ones(grid.shape, dtype=bool)
            return SimulationResult(
                probability=np.full(grid.shape, 100, dtype=np.uint8),
                binary=volume,
                voxel=grid.data,
                spacing=grid.spacing,
                origin=grid.origin,
                rest_volume=100.0,
                probability_density=100.0,
                elapsed_seconds=0.01,
            )

    monkeypatch.setattr(module, "create_solver", lambda _parameters: object())
    monkeypatch.setattr(module, "SimulationPipeline", FakePipeline)
    monkeypatch.setattr(module, "simulation_rois", lambda _binary: (roi, roi))

    grid = VoxelGrid(data=np.ones((4, 4, 4), dtype=bool), spacing=0.1)
    targets = [
        module.ModelCalibrationTarget(sample="A1", roi_x=roi, roi_y=roi),
        module.ModelCalibrationTarget(sample="A2", roi_x=roi, roi_y=roi),
    ]

    result = run_model_calibration(
        grid,
        targets,
        options=ModelCalibrationOptions(
            max_evaluations=3,
            backend=SolverBackend.CPU_REFERENCE,
            max_workers=1,
        ),
    )

    assert len(calls) == 3
    assert len(result.samples) == 2
    assert all(sample.evaluations == 3 for sample in result.samples)


def test_model_calibration_reports_candidate_start_progress(monkeypatch):
    from capp.calibration import model_calibration as module

    roi = np.zeros((23, 57), dtype=bool)
    roi[5:15, 20:30] = True

    class FakePipeline:
        def __init__(self, solver):
            self.solver = solver

        def run_voxel_grid(self, grid, parameters, progress_callback=None):
            if progress_callback is not None:
                progress_callback(50, "Running PBF X solver")
            volume = np.ones(grid.shape, dtype=bool)
            return SimulationResult(
                probability=np.full(grid.shape, 100, dtype=np.uint8),
                binary=volume,
                voxel=grid.data,
                spacing=grid.spacing,
                origin=grid.origin,
                rest_volume=100.0,
                probability_density=100.0,
                elapsed_seconds=0.01,
            )

    monkeypatch.setattr(module, "create_solver", lambda _parameters: object())
    monkeypatch.setattr(module, "SimulationPipeline", FakePipeline)
    monkeypatch.setattr(module, "simulation_rois", lambda _binary: (roi, roi))

    progress_messages = []
    run_model_calibration(
        VoxelGrid(data=np.ones((4, 4, 4), dtype=bool), spacing=0.1),
        [module.ModelCalibrationTarget(sample="A1", roi_x=roi, roi_y=roi)],
        options=ModelCalibrationOptions(
            max_evaluations=2,
            backend=SolverBackend.CPU_REFERENCE,
            max_workers=1,
            optimizer="sobol",
        ),
        progress_callback=lambda percent, message: progress_messages.append((percent, message)),
    )

    assert any("Running candidate 1/2 with PBF Standard" in msg for _, msg in progress_messages)
    assert any("Candidate 1/2: Running PBF X solver" in msg for _, msg in progress_messages)


def test_cuda_model_calibration_keeps_requested_parallel_workers(monkeypatch):
    from capp.calibration import model_calibration as module

    roi = np.zeros((23, 57), dtype=bool)
    roi[5:15, 20:30] = True

    class FakePipeline:
        def __init__(self, solver):
            self.solver = solver

        def run_voxel_grid(self, grid, parameters, progress_callback=None):
            volume = np.ones(grid.shape, dtype=bool)
            return SimulationResult(
                probability=np.full(grid.shape, 100, dtype=np.uint8),
                binary=volume,
                voxel=grid.data,
                spacing=grid.spacing,
                origin=grid.origin,
                rest_volume=100.0,
                probability_density=100.0,
                elapsed_seconds=0.01,
            )

    monkeypatch.setattr(module, "create_solver", lambda _parameters: object())
    monkeypatch.setattr(module, "SimulationPipeline", FakePipeline)
    monkeypatch.setattr(module, "simulation_rois", lambda _binary: (roi, roi))

    progress_messages = []
    run_model_calibration(
        VoxelGrid(data=np.ones((4, 4, 4), dtype=bool), spacing=0.1),
        [module.ModelCalibrationTarget(sample="A1", roi_x=roi, roi_y=roi)],
        options=ModelCalibrationOptions(
            max_evaluations=4,
            backend=SolverBackend.CUDA,
            max_workers=4,
            optimizer="sobol",
        ),
        progress_callback=lambda percent, message: progress_messages.append((percent, message)),
    )

    assert any("for 1 samples on 4 workers" in msg for _, msg in progress_messages)
    completion_percents = [
        percent for percent, message in progress_messages if message.startswith("global candidate")
    ]
    assert completion_percents == [25, 50, 75, 100]


def test_parallel_candidate_solver_progress_is_aggregated(monkeypatch):
    from threading import Barrier

    from capp.calibration import model_calibration as module

    roi = np.zeros((23, 57), dtype=bool)
    roi[5:15, 20:30] = True
    barrier = Barrier(4)

    class FakePipeline:
        def __init__(self, solver):
            self.solver = solver

        def run_voxel_grid(self, grid, parameters, progress_callback=None):
            barrier.wait(timeout=5)
            if progress_callback is not None:
                progress_callback(50, "Running PBF X solver")
            volume = np.ones(grid.shape, dtype=bool)
            return SimulationResult(
                probability=np.full(grid.shape, 100, dtype=np.uint8),
                binary=volume,
                voxel=grid.data,
                spacing=grid.spacing,
                origin=grid.origin,
                rest_volume=100.0,
                probability_density=100.0,
                elapsed_seconds=0.01,
            )

    monkeypatch.setattr(module, "create_solver", lambda _parameters: object())
    monkeypatch.setattr(module, "SimulationPipeline", FakePipeline)
    monkeypatch.setattr(module, "simulation_rois", lambda _binary: (roi, roi))

    progress_messages = []
    run_model_calibration(
        VoxelGrid(data=np.ones((4, 4, 4), dtype=bool), spacing=0.1),
        [module.ModelCalibrationTarget(sample="A1", roi_x=roi, roi_y=roi)],
        options=ModelCalibrationOptions(
            max_evaluations=4,
            backend=SolverBackend.CUDA,
            max_workers=4,
            optimizer="sobol",
        ),
        progress_callback=lambda percent, message: progress_messages.append((percent, message)),
    )

    running_progress = [
        percent
        for percent, message in progress_messages
        if message.endswith("Running PBF X solver")
    ]
    assert running_progress
    assert max(running_progress) > 25


def test_adaptive_sobol_uses_full_evaluation_budget(monkeypatch):
    from capp.calibration import model_calibration as module

    calls = []
    roi = np.zeros((23, 57), dtype=bool)
    roi[5:15, 20:30] = True

    class FakePipeline:
        def __init__(self, solver):
            self.solver = solver

        def run_voxel_grid(self, grid, parameters, progress_callback=None):
            calls.append(
                (
                    *parameters.current_as_directional(),
                    parameters.min_bias,
                    parameters.initial_deviation,
                )
            )
            volume = np.ones(grid.shape, dtype=bool)
            return SimulationResult(
                probability=np.full(grid.shape, 100, dtype=np.uint8),
                binary=volume,
                voxel=grid.data,
                spacing=grid.spacing,
                origin=grid.origin,
                rest_volume=100.0,
                probability_density=100.0,
                elapsed_seconds=0.01,
            )

    monkeypatch.setattr(module, "create_solver", lambda _parameters: object())
    monkeypatch.setattr(module, "SimulationPipeline", FakePipeline)
    monkeypatch.setattr(module, "simulation_rois", lambda _binary: (roi, roi))

    result = run_model_calibration(
        VoxelGrid(data=np.ones((4, 4, 4), dtype=bool), spacing=0.1),
        [module.ModelCalibrationTarget(sample="A1", roi_x=roi, roi_y=roi)],
        options=ModelCalibrationOptions(
            max_evaluations=10,
            backend=SolverBackend.CPU_REFERENCE,
            max_workers=1,
            optimizer="adaptive_sobol",
        ),
    )

    assert len(calls) == 10
    assert len(set(calls)) == 10
    assert result.samples[0].evaluations == 10
