import numpy as np

from capp.calibration.losses import RoiBoundaryLoss


def test_roi_boundary_loss_is_zero_for_identical_masks():
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 5:11] = True

    result = RoiBoundaryLoss().evaluate(mask, mask, mask, mask)

    assert result.valid
    assert result.total == 0.0
    assert result.x_dice == 1.0
    assert result.x_iou == 1.0


def test_roi_boundary_loss_penalizes_extra_simulated_region():
    target = np.zeros((32, 32), dtype=bool)
    simulated = np.zeros((32, 32), dtype=bool)
    target[6:14, 6:14] = True
    simulated[6:14, 6:14] = True
    simulated[22:28, 22:28] = True

    result = RoiBoundaryLoss().evaluate(target, target, simulated, simulated)

    assert result.valid
    assert result.total > 0.0
    assert result.x_dice < 1.0
    assert result.x_area_loss > 0.0


def test_roi_boundary_loss_increases_with_shift_distance():
    target = np.zeros((40, 40), dtype=bool)
    target[14:24, 14:24] = True
    small_shift = np.zeros_like(target)
    large_shift = np.zeros_like(target)
    small_shift[16:26, 14:24] = True
    large_shift[24:34, 14:24] = True

    small = RoiBoundaryLoss().evaluate(target, target, small_shift, small_shift)
    large = RoiBoundaryLoss().evaluate(target, target, large_shift, large_shift)

    assert small.valid
    assert large.valid
    assert large.total > small.total
