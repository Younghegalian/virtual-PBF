import numpy as np

from capp.calibration.rmc import RmcBounds, RmcParameterSet, simulation_rois
from capp.calibration.roi import extract_rmc_roi
from capp.domain import NeighborhoodModel, SolverBackend


def test_extract_rmc_roi_matches_matlab_fixed_window():
    image = np.zeros((100, 120), dtype=bool)
    image[30:40, 45:55] = True

    roi = extract_rmc_roi(image)

    assert roi.shape == (23, 57)
    assert roi[1:11, 13:23].all()


def test_rmc_parameter_set_builds_solver_parameters():
    params = RmcParameterSet(0.11, 0.12, 0.13, 0.14, 0.02, 0.33)

    solver = params.to_solver_parameters(backend=SolverBackend.CPU_REFERENCE, rng_seed=7)

    assert solver.neighborhood is NeighborhoodModel.DIRECTIONAL_VON_NEUMANN
    assert solver.current_coefficients == (0.11, 0.12, 0.13, 0.14)
    assert solver.min_bias == 0.02
    assert solver.initial_deviation == 0.33
    assert solver.backend is SolverBackend.CPU_REFERENCE
    assert solver.rng_seed == 7


def test_rmc_bounds_match_current_matlab_search_space():
    bounds = RmcBounds().as_pairs()

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
