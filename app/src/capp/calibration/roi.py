from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def central_slices(volume: NDArray[np.bool_]) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    if volume.ndim != 3:
        raise ValueError("volume must be 3D.")

    x_slice = np.squeeze(np.transpose(volume[:, volume.shape[1] // 2, :], (1, 0)))
    y_slice = np.squeeze(np.transpose(volume[volume.shape[0] // 2, :, :], (1, 0)))
    return np.flipud(x_slice), np.flipud(y_slice)


def extract_roi(binary_image: NDArray[np.bool_]) -> NDArray[np.bool_]:
    image = np.asarray(binary_image, dtype=bool)
    coords = np.argwhere(image)
    if coords.size == 0:
        return image.copy()

    (row_min, col_min), (row_max, col_max) = coords.min(axis=0), coords.max(axis=0)
    return image[row_min : row_max + 1, col_min : col_max + 1]


def extract_model_calibration_roi(
    binary_image: NDArray[np.bool_],
    *,
    top_margin: int = 7,
    lateral_margin: int = 10,
) -> NDArray[np.bool_]:
    """Extract the fixed model-calibration comparison window used by MATLAB."""

    image = np.asarray(binary_image, dtype=bool)
    if image.ndim != 2:
        raise ValueError("binary_image must be a 2D array.")
    if image.size == 0:
        return image.copy()

    # MATLAB code used 1-based coordinates and a fixed window tied to sample width.
    top_y = 1
    min_x = 1
    max_x = image.shape[1]
    base_length = max_x - min_x - int(lateral_margin)
    if base_length <= 0:
        return image.copy()

    roi_top_y = _matlab_round(top_y + 0.2 * base_length) + int(top_margin)
    roi_width = _matlab_round(base_length * 0.5)
    roi_center_x = _matlab_round((min_x + max_x) / 2)
    half_width = _matlab_round(roi_width / 2)
    roi_left_x = max(1, roi_center_x - half_width)
    roi_right_x = min(image.shape[1], roi_center_x + half_width)

    roi_height = _matlab_round(base_length * 0.2)
    roi_bottom_y = min(image.shape[0], roi_top_y + roi_height)

    if roi_top_y > image.shape[0] or roi_left_x > roi_right_x:
        return np.zeros((0, 0), dtype=bool)

    return image[roi_top_y - 1 : roi_bottom_y, roi_left_x - 1 : roi_right_x]


def extract_rmc_roi(
    binary_image: NDArray[np.bool_],
    *,
    top_margin: int = 7,
    lateral_margin: int = 10,
) -> NDArray[np.bool_]:
    return extract_model_calibration_roi(
        binary_image,
        top_margin=top_margin,
        lateral_margin=lateral_margin,
    )


def model_calibration_slices(
    binary_volume: NDArray[np.bool_],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    x_slice, y_slice = central_slices(np.asarray(binary_volume, dtype=bool))
    return extract_model_calibration_roi(x_slice), extract_model_calibration_roi(y_slice)


def _matlab_round(value: float) -> int:
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))
