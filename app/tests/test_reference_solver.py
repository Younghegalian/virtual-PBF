import numpy as np

from capp.domain import SolverParameters, VoxelGrid
from capp.solver.reference import (
    ReferenceLayerwiseMarkovSolver,
    _postprocess_binary,
    _update_von_neumann_layer,
)


def test_reference_solver_returns_volume_shapes():
    voxel = np.zeros((6, 6, 5), dtype=bool)
    voxel[2:4, 2:4, 1:4] = True
    grid = VoxelGrid(voxel, spacing=0.5)

    result = ReferenceLayerwiseMarkovSolver().solve(grid, SolverParameters(iteration_bound=4))

    assert result.probability.shape == voxel.shape
    assert result.binary.shape == voxel.shape
    assert result.voxel.shape == voxel.shape
    assert result.probability.dtype == np.uint8
    assert result.binary.dtype == bool
    assert result.rest_volume >= 0.0


def test_reference_solver_reports_monotonic_progress():
    voxel = np.zeros((6, 6, 5), dtype=bool)
    voxel[2:4, 2:4, 1:4] = True
    grid = VoxelGrid(voxel, spacing=0.5)
    events = []

    ReferenceLayerwiseMarkovSolver().solve(
        grid,
        SolverParameters(iteration_bound=4),
        progress_callback=lambda percent, message: events.append((percent, message)),
    )

    assert events[0][0] == 0
    assert events[-1][0] == 100
    assert [percent for percent, _message in events] == sorted(
        percent for percent, _message in events
    )


def test_binary_postprocess_removes_padding_connected_floating_voxels():
    binary = np.zeros((8, 8, 6), dtype=bool)
    binary[:, :, 0] = True
    binary[1:5, 1:5, 1:5] = True
    binary[6, 6, 1] = True

    cropped = _postprocess_binary(binary, x_size=6, y_size=6, z_size=4)

    assert cropped[0:4, 0:4, 0:4].all()
    assert not cropped[5, 5, 0]


def test_binary_postprocess_applies_matlab_area_open_when_requested():
    binary = np.zeros((12, 12, 8), dtype=bool)
    binary[1:6, 1:6, 1:4] = True
    binary[8, 8, 2] = True
    binary[8, 8, 3] = True
    binary[8, 9, 3] = True
    binary[9, 8, 3] = True

    cropped = _postprocess_binary(binary, x_size=10, y_size=10, z_size=6, apply_area_open=True)

    assert cropped[0:5, 0:5, 0:3].all()
    assert not cropped[7:9, 7:9, 1:3].any()


def test_lower_layer_growth_is_gated_by_current_cad_voxel():
    probability = np.zeros((3, 3, 3), dtype=np.float32)
    voxel_calc = np.zeros((3, 3, 3), dtype=np.float32)
    probability[1, 1, 0] = 1.0

    updated = _update_von_neumann_layer(
        probability=probability,
        voxel_calc=voxel_calc,
        layer=1,
        coeffs=(0.0, 0.0, 0.0, 0.0, 1.0),
        min_val=0.0,
        idp_model=0.5,
    )

    assert updated[0, 0] == 0.0

