from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage


@dataclass(frozen=True)
class RoiLossResult:
    total: float
    x_loss: float
    y_loss: float
    x_map: NDArray[np.float64]
    y_map: NDArray[np.float64]
    valid: bool
    x_dice: float = float("nan")
    y_dice: float = float("nan")
    x_iou: float = float("nan")
    y_iou: float = float("nan")
    x_overlap_loss: float = float("nan")
    y_overlap_loss: float = float("nan")
    x_boundary_loss: float = float("nan")
    y_boundary_loss: float = float("nan")
    x_area_loss: float = float("nan")
    y_area_loss: float = float("nan")


@dataclass(frozen=True)
class _AxisLoss:
    loss: float
    loss_map: NDArray[np.float64]
    valid: bool
    dice: float
    iou: float
    overlap_loss: float
    boundary_loss: float
    area_loss: float


class RoiBoundaryLoss:
    def __init__(
        self,
        boundary_radius: int = 15,
        empty_penalty: float = 1e6,
        overlap_weight: float = 0.55,
        boundary_weight: float = 0.35,
        area_weight: float = 0.10,
    ) -> None:
        self.boundary_radius = int(boundary_radius)
        self.empty_penalty = float(empty_penalty)
        self.overlap_weight = float(overlap_weight)
        self.boundary_weight = float(boundary_weight)
        self.area_weight = float(area_weight)

    def evaluate(
        self,
        target_x: NDArray[np.bool_],
        target_y: NDArray[np.bool_],
        simulated_x: NDArray[np.bool_],
        simulated_y: NDArray[np.bool_],
    ) -> RoiLossResult:
        target_x = _resize_nearest(np.asarray(target_x, dtype=bool), simulated_x.shape)
        target_y = _resize_nearest(np.asarray(target_y, dtype=bool), simulated_y.shape)
        simulated_x = np.asarray(simulated_x, dtype=bool)
        simulated_y = np.asarray(simulated_y, dtype=bool)

        result_x = self._single_axis_loss(target_x, simulated_x)
        result_y = self._single_axis_loss(target_y, simulated_y)
        valid = result_x.valid and result_y.valid

        if not valid:
            return RoiLossResult(
                total=self.empty_penalty,
                x_loss=self.empty_penalty,
                y_loss=self.empty_penalty,
                x_map=np.zeros_like(simulated_x, dtype=np.float64),
                y_map=np.zeros_like(simulated_y, dtype=np.float64),
                valid=False,
                x_dice=result_x.dice,
                y_dice=result_y.dice,
                x_iou=result_x.iou,
                y_iou=result_y.iou,
                x_overlap_loss=result_x.overlap_loss,
                y_overlap_loss=result_y.overlap_loss,
                x_boundary_loss=result_x.boundary_loss,
                y_boundary_loss=result_y.boundary_loss,
                x_area_loss=result_x.area_loss,
                y_area_loss=result_y.area_loss,
            )

        return RoiLossResult(
            total=(result_x.loss + result_y.loss) / 2.0,
            x_loss=result_x.loss,
            y_loss=result_y.loss,
            x_map=result_x.loss_map,
            y_map=result_y.loss_map,
            valid=True,
            x_dice=result_x.dice,
            y_dice=result_y.dice,
            x_iou=result_x.iou,
            y_iou=result_y.iou,
            x_overlap_loss=result_x.overlap_loss,
            y_overlap_loss=result_y.overlap_loss,
            x_boundary_loss=result_x.boundary_loss,
            y_boundary_loss=result_y.boundary_loss,
            x_area_loss=result_x.area_loss,
            y_area_loss=result_y.area_loss,
        )

    def _single_axis_loss(
        self,
        target: NDArray[np.bool_],
        simulated: NDArray[np.bool_],
    ) -> _AxisLoss:
        target_perimeter = _perimeter(target)
        simulated_perimeter = _perimeter(simulated)
        _clear_border(target_perimeter)
        _clear_border(simulated_perimeter)

        target_count = int(np.count_nonzero(target))
        simulated_count = int(np.count_nonzero(simulated))
        if (
            target_count == 0
            or simulated_count == 0
            or not target_perimeter.any()
            or not simulated_perimeter.any()
        ):
            return _AxisLoss(
                loss=self.empty_penalty,
                loss_map=np.zeros_like(target, dtype=np.float64),
                valid=False,
                dice=0.0,
                iou=0.0,
                overlap_loss=self.empty_penalty,
                boundary_loss=self.empty_penalty,
                area_loss=self.empty_penalty,
            )

        overlap = target & simulated
        union = target | simulated
        overlap_count = int(np.count_nonzero(overlap))
        union_count = int(np.count_nonzero(union))
        dice = 2.0 * overlap_count / max(1, target_count + simulated_count)
        iou = overlap_count / max(1, union_count)
        overlap_loss = 1.0 - dice
        area_loss = abs(simulated_count - target_count) / max(1, target_count + simulated_count)

        distance_to_simulated = ndimage.distance_transform_edt(~simulated_perimeter)
        distance_to_target = ndimage.distance_transform_edt(~target_perimeter)
        target_boundary_error = float(distance_to_simulated[target_perimeter].mean())
        simulated_boundary_error = float(distance_to_target[simulated_perimeter].mean())
        radius = max(1, self.boundary_radius)
        boundary_loss = min(
            2.0,
            (target_boundary_error + simulated_boundary_error) / (2.0 * radius),
        )

        loss = (
            self.overlap_weight * overlap_loss
            + self.boundary_weight * boundary_loss
            + self.area_weight * area_loss
        )
        loss_map = _diagnostic_loss_map(
            target=target,
            simulated=simulated,
            target_perimeter=target_perimeter,
            simulated_perimeter=simulated_perimeter,
            distance_to_target=distance_to_target,
            distance_to_simulated=distance_to_simulated,
            radius=radius,
        )

        return _AxisLoss(
            loss=float(loss),
            loss_map=loss_map,
            valid=True,
            dice=float(dice),
            iou=float(iou),
            overlap_loss=float(overlap_loss),
            boundary_loss=float(boundary_loss),
            area_loss=float(area_loss),
        )


def _perimeter(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return mask & ~eroded


def _clear_border(mask: NDArray[np.bool_]) -> None:
    if mask.size == 0:
        return
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False


def _resize_nearest(image: NDArray[np.bool_], shape: tuple[int, int]) -> NDArray[np.bool_]:
    if image.shape == shape:
        return image
    factors = (shape[0] / image.shape[0], shape[1] / image.shape[1])
    return ndimage.zoom(image.astype(np.uint8), zoom=factors, order=0).astype(bool)


def _diagnostic_loss_map(
    *,
    target: NDArray[np.bool_],
    simulated: NDArray[np.bool_],
    target_perimeter: NDArray[np.bool_],
    simulated_perimeter: NDArray[np.bool_],
    distance_to_target: NDArray[np.float64],
    distance_to_simulated: NDArray[np.float64],
    radius: int,
) -> NDArray[np.float64]:
    loss_map = np.zeros_like(target, dtype=np.float64)
    loss_map[target ^ simulated] = 1.0
    loss_map[target_perimeter] = np.maximum(
        loss_map[target_perimeter],
        np.minimum(1.0, distance_to_simulated[target_perimeter] / radius),
    )
    loss_map[simulated_perimeter] = np.maximum(
        loss_map[simulated_perimeter],
        np.minimum(1.0, distance_to_target[simulated_perimeter] / radius),
    )
    return loss_map


def _diagnostic_loss_map(
    *,
    target: NDArray[np.bool_],
    simulated: NDArray[np.bool_],
    target_perimeter: NDArray[np.bool_],
    simulated_perimeter: NDArray[np.bool_],
    distance_to_target: NDArray[np.float64],
    distance_to_simulated: NDArray[np.float64],
    radius: int,
) -> NDArray[np.float64]:
    loss_map = np.zeros_like(target, dtype=np.float64)
    loss_map[target ^ simulated] = 1.0
    loss_map[target_perimeter] = np.maximum(
        loss_map[target_perimeter],
        np.minimum(1.0, distance_to_simulated[target_perimeter] / radius),
    )
    loss_map[simulated_perimeter] = np.maximum(
        loss_map[simulated_perimeter],
        np.minimum(1.0, distance_to_target[simulated_perimeter] / radius),
    )
    return loss_map
