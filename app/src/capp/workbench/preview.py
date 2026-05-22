from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

DEVIATION_HEATMAP_SCALE_MM = 1.0


def compute_overhang_angles(normals: NDArray) -> NDArray[np.float32]:
    normal_array = np.asarray(normals, dtype=np.float32)
    if normal_array.ndim != 2 or normal_array.shape[1] != 3:
        raise ValueError("Overhang angle calculation requires Nx3 normals.")
    normal_norm = np.linalg.norm(normal_array, axis=1)
    normal_norm[normal_norm == 0.0] = 1.0
    cosine = np.clip(normal_array[:, 2] / normal_norm, -1.0, 1.0)
    return np.abs(np.degrees(np.arccos(cosine)) - 180.0).astype(np.float32)


def prepare_stl_preview_mesh(path: str | Path):
    import pyvista as pv

    mesh = pv.read(str(path)).extract_surface(algorithm="dataset_surface")
    original_cells = int(mesh.n_cells)

    with suppress(Exception):
        mesh = mesh.compute_normals(
            point_normals=True,
            cell_normals=False,
            auto_orient_normals=True,
            consistent_normals=True,
            split_vertices=False,
        )
    return mesh, original_cells


@dataclass(frozen=True)
class GeometryDeviationPreview:
    original_mesh: object
    original_cells: int
    deviation_surface: object
    metrics: dict[str, float]
    stride: int
    render_spacing: float
    alignment_offset: tuple[float, float, float]


@dataclass(frozen=True)
class PolyDataPayload:
    points: NDArray[np.float32]
    faces: NDArray[np.int64]
    point_data: dict[str, NDArray]


@dataclass(frozen=True)
class GeometryDeviationPreviewPayload:
    original_mesh: PolyDataPayload
    original_cells: int
    deviation_surface: PolyDataPayload
    metrics: dict[str, float]
    stride: int
    render_spacing: float
    alignment_offset: tuple[float, float, float]


def build_geometry_deviation_preview(
    stl_path: str | Path,
    volume: NDArray,
    spacing: float,
    origin: tuple[float, float, float],
    *,
    label: str = "Binary",
    threshold: float = 0.5,
    progress_callback=None,
) -> GeometryDeviationPreview:
    def progress(percent: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0, min(100, int(percent))), message)

    pane = object.__new__(PreviewPane)
    pane._pyvista = None
    pv = pane._load_pyvista()

    progress(5, "Preparing result volume")
    data, stride = pane._prepare_volume_data(volume)
    if not pane._is_binary_volume(data):
        threshold = 0.5
    render_spacing = float(spacing) * stride

    progress(18, "Loading original STL")
    original_mesh, original_cells = prepare_stl_preview_mesh(stl_path)

    progress(36, "Extracting printed iso-surface")
    printed_surface = pane._make_isosurface_mesh(
        pv,
        data,
        spacing=render_spacing,
        origin=origin,
        level=float(threshold),
    )
    printed_surface, alignment_offset = _align_surface_to_source_origin(
        printed_surface,
        original_mesh,
        voxel_spacing=spacing,
        tolerance_spacing=render_spacing,
        origin=origin,
    )
    with suppress(Exception):
        printed_surface = printed_surface.compute_normals(
            point_normals=True,
            cell_normals=False,
            auto_orient_normals=True,
            consistent_normals=True,
            split_vertices=False,
        )

    progress(68, "Computing signed distance field")
    color_zero_tolerance = max(float(spacing) * 1e-3, 1e-9)
    deviation_surface, metrics = pane._geometry_deviation_surface(
        printed_surface,
        original_mesh,
        color_zero_tolerance=color_zero_tolerance,
    )

    progress(92, "Preparing heatmap render")
    return GeometryDeviationPreview(
        original_mesh=original_mesh,
        original_cells=original_cells,
        deviation_surface=deviation_surface,
        metrics=metrics,
        stride=stride,
        render_spacing=render_spacing,
        alignment_offset=alignment_offset,
    )


def pack_geometry_deviation_preview(
    preview: GeometryDeviationPreview,
) -> GeometryDeviationPreviewPayload:
    return GeometryDeviationPreviewPayload(
        original_mesh=_polydata_payload(preview.original_mesh),
        original_cells=preview.original_cells,
        deviation_surface=_polydata_payload(preview.deviation_surface),
        metrics=preview.metrics,
        stride=preview.stride,
        render_spacing=preview.render_spacing,
        alignment_offset=preview.alignment_offset,
    )


def unpack_geometry_deviation_preview(
    preview: GeometryDeviationPreview | GeometryDeviationPreviewPayload,
) -> GeometryDeviationPreview:
    if isinstance(preview, GeometryDeviationPreview):
        return preview

    import pyvista as pv

    return GeometryDeviationPreview(
        original_mesh=_polydata_from_payload(pv, preview.original_mesh),
        original_cells=preview.original_cells,
        deviation_surface=_polydata_from_payload(pv, preview.deviation_surface),
        metrics=preview.metrics,
        stride=preview.stride,
        render_spacing=preview.render_spacing,
        alignment_offset=preview.alignment_offset,
    )


def _polydata_payload(mesh) -> PolyDataPayload:
    return PolyDataPayload(
        points=np.asarray(mesh.points, dtype=np.float32).copy(),
        faces=np.asarray(mesh.faces, dtype=np.int64).copy(),
        point_data={
            name: np.asarray(mesh.point_data[name]).copy()
            for name in mesh.point_data
        },
    )


def _polydata_from_payload(pv, payload: PolyDataPayload):
    mesh = pv.PolyData(payload.points, payload.faces)
    for name, values in payload.point_data.items():
        mesh.point_data[name] = values
    return mesh


def _source_aligned_volume_origin(original_mesh, voxel_spacing: float) -> np.ndarray:
    from capp.geometry.voxelizer import VOXEL_LOWER_PADDING_CELLS

    bounds = original_mesh.bounds
    source_min = np.asarray([bounds[0], bounds[2], bounds[4]], dtype=np.float64)
    lower_padding = np.asarray(VOXEL_LOWER_PADDING_CELLS, dtype=np.float64)
    return source_min - lower_padding * float(voxel_spacing)


def _align_surface_to_source_origin(
    printed_surface,
    original_mesh,
    *,
    voxel_spacing: float,
    origin,
    tolerance_spacing: float | None = None,
):
    expected_origin = _source_aligned_volume_origin(original_mesh, voxel_spacing)
    current_origin = np.asarray(origin, dtype=np.float64)
    offset = expected_origin - current_origin
    tolerance = float(tolerance_spacing if tolerance_spacing is not None else voxel_spacing)
    if np.linalg.norm(offset) <= max(tolerance * 1e-5, 1e-9):
        return printed_surface, (0.0, 0.0, 0.0)
    aligned = printed_surface.copy(deep=True)
    aligned.translate(tuple(float(value) for value in offset), inplace=True)
    return aligned, tuple(float(value) for value in offset)


def _combined_mesh_bounds(*meshes):
    bounds = np.asarray([mesh.bounds for mesh in meshes], dtype=np.float64)
    return (
        float(bounds[:, 0].min()),
        float(bounds[:, 1].max()),
        float(bounds[:, 2].min()),
        float(bounds[:, 3].max()),
        float(bounds[:, 4].min()),
        float(bounds[:, 5].max()),
    )


def _deviation_color_values_mm(
    distances: NDArray,
    *,
    zero_tolerance: float = 0.0,
) -> NDArray[np.float32]:
    values = np.nan_to_num(
        np.asarray(distances, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if zero_tolerance > 0.0:
        values = values.copy()
        values[np.abs(values) <= float(zero_tolerance)] = 0.0
    return values.astype(np.float32)


def _deviation_color_limits(
    metrics: dict[str, float],
    fallback_scale: float,
) -> tuple[float, float]:
    scale = DEVIATION_HEATMAP_SCALE_MM
    return -scale, scale


def _deviation_jet_colormap(negative_scale: float, positive_scale: float):
    from matplotlib import colormaps
    from matplotlib.colors import LinearSegmentedColormap

    jet = colormaps["jet"]
    neutral = jet(0.5)
    negative_scale = max(float(negative_scale), 0.0)
    positive_scale = max(float(positive_scale), 0.0)
    if negative_scale > 0.0 and positive_scale > 0.0:
        zero_position = negative_scale / (negative_scale + positive_scale)
        colors = [
            (0.0, jet(0.0)),
            (zero_position * 0.5, jet(0.25)),
            (zero_position, neutral),
            (zero_position + (1.0 - zero_position) * 0.5, jet(0.75)),
            (1.0, jet(1.0)),
        ]
    elif negative_scale > 0.0:
        colors = [(0.0, jet(0.0)), (0.5, jet(0.25)), (1.0, neutral)]
    elif positive_scale > 0.0:
        colors = [(0.0, neutral), (0.5, jet(0.75)), (1.0, jet(1.0))]
    else:
        colors = [(0.0, jet(0.0)), (0.5, neutral), (1.0, jet(1.0))]
    return LinearSegmentedColormap.from_list("deviation_jet_mm", colors, N=257)


class PreviewPane:
    def __init__(
        self,
        *,
        show_source_selector: bool = False,
        show_stl_controls: bool = False,
    ) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QComboBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QVBoxLayout,
            QWidget,
        )

        self.widget = QWidget()
        self.widget.setObjectName("PreviewPane")
        self._layout = QVBoxLayout(self.widget)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._title = QLabel("3D Preview")
        self._title.setObjectName("PreviewTitle")
        self._title.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.source_selector = QComboBox()
        self.source_selector.setObjectName("PreviewSource")
        self.source_selector.addItems(["STL", "Voxelization", "Result"])
        self.source_selector.setVisible(show_source_selector)
        self._stl_controls_visible = show_stl_controls
        self.stl_display_mode = QComboBox()
        self.stl_display_mode.setObjectName("PreviewStlMode")
        self.stl_display_mode.addItems(["Shaded", "Overhang angle"])
        self.stl_display_mode.setVisible(show_stl_controls)
        self.stl_display_mode.currentTextChanged.connect(self._sync_stl_control_visibility)
        self.overhang_limit = QLineEdit("60")
        self.overhang_limit.setObjectName("PreviewOverhangLimit")
        self.overhang_limit.setFixedWidth(44)
        self.overhang_limit.setVisible(False)
        self._status = QLabel("Open an STL or run a simulation to preview geometry.")
        self._status.setObjectName("PreviewStatus")

        header = QWidget()
        header.setObjectName("PreviewHeader")
        header.setFixedHeight(20)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)
        header_layout.addWidget(self._title, 1)
        header_layout.addWidget(self.source_selector)
        header_layout.addWidget(self.stl_display_mode)
        header_layout.addWidget(self.overhang_limit)
        self._layout.addWidget(header)
        self._layout.addWidget(self._status)
        self._plotter = None
        self._pyvista = None
        self._last_volume_request = None
        self._sync_stl_control_visibility()

    def show_stl(
        self,
        path: str | Path,
        *,
        display_mode: str = "Shaded",
        overhang_limit: float = 60.0,
    ) -> None:
        try:
            self._last_volume_request = None
            mesh, original_cells = prepare_stl_preview_mesh(path)
            self.show_stl_mesh(
                path,
                mesh,
                original_cells,
                display_mode=display_mode,
                overhang_limit=overhang_limit,
            )
        except Exception as exc:
            self.show_message(f"STL preview failed: {exc}")

    def show_stl_mesh(
        self,
        path: str | Path,
        mesh,
        original_cells: int | None = None,
        *,
        display_mode: str = "Shaded",
        overhang_limit: float = 60.0,
    ) -> None:
        try:
            self._last_volume_request = None
            plotter = self._ensure_plotter()
            plotter.clear()
            self._prepare_scene(plotter)
            with suppress(Exception):
                plotter.enable_eye_dome_lighting()

            if display_mode == "Overhang angle":
                self._add_overhang_mesh(plotter, mesh, overhang_limit)
            else:
                self._add_shaded_cad_mesh(plotter, mesh)
            if mesh.n_cells <= 180_000:
                self._add_feature_edges(plotter, mesh)
            plotter.add_mesh(mesh.outline(), color="#334155", line_width=1.4)
            plotter.add_axes()
            self._set_cad_camera(plotter, mesh.bounds)
            self._render_plotter(plotter)
            title = "CAD Overhang" if display_mode == "Overhang angle" else "CAD Preview"
            self._title.setText(title)
        except Exception as exc:
            self.show_message(f"STL preview failed: {exc}")

    def show_stl_overlay_mesh(
        self,
        part_path: str | Path,
        part_mesh,
        part_cells: int,
        support_path: str | Path,
        support_mesh,
        support_cells: int,
        *,
        display_mode: str = "Shaded",
        overhang_limit: float = 60.0,
    ) -> None:
        try:
            self._last_volume_request = None
            plotter = self._ensure_plotter()
            plotter.clear()
            self._prepare_scene(plotter)
            with suppress(Exception):
                plotter.enable_eye_dome_lighting()

            if display_mode == "Overhang angle":
                self._add_overhang_mesh(plotter, part_mesh, overhang_limit)
            else:
                self._add_shaded_cad_mesh(plotter, part_mesh)
            if part_mesh.n_cells <= 180_000:
                self._add_feature_edges(plotter, part_mesh)
            plotter.add_mesh(
                support_mesh,
                color="#e85d04",
                edge_color="#7c2d12",
                show_edges=support_mesh.n_cells <= 180_000,
                smooth_shading=False,
                lighting=True,
                ambient=0.42,
                diffuse=0.76,
                specular=0.18,
                silhouette={"color": "#7c2d12", "line_width": 1.7},
            )
            plotter.add_mesh(part_mesh.outline(), color="#64748b", line_width=1.1)
            plotter.add_mesh(support_mesh.outline(), color="#9a3412", line_width=1.3)
            plotter.add_axes()
            self._set_cad_camera(plotter, _combined_mesh_bounds(part_mesh, support_mesh))
            self._render_plotter(plotter)
            title = (
                "CAD + Support Overhang"
                if display_mode == "Overhang angle"
                else "CAD + Support"
            )
            self._title.setText(title)
            self._status.setText("")
        except Exception as exc:
            self.show_message(f"Support overlay preview failed: {exc}")

    def _add_shaded_cad_mesh(self, plotter, mesh) -> None:
        plotter.add_mesh(
            mesh,
            color="#cbd5e1",
            edge_color="#475569",
            show_edges=mesh.n_cells <= 120_000,
            smooth_shading=False,
            lighting=True,
            ambient=0.32,
            diffuse=0.82,
            specular=0.22,
            specular_power=20,
            silhouette={"color": "#1f2937", "line_width": 2.0},
        )

    def _add_support_mesh_overlay(self, plotter, support_mesh) -> None:
        plotter.add_mesh(
            support_mesh,
            color="#e85d04",
            edge_color="#7c2d12",
            show_edges=support_mesh.n_cells <= 180_000,
            smooth_shading=False,
            lighting=True,
            ambient=0.42,
            diffuse=0.76,
            specular=0.18,
            silhouette={"color": "#7c2d12", "line_width": 1.7},
        )

    def set_stl_controls_visible(self, visible: bool) -> None:
        self._stl_controls_visible = bool(visible)
        self.stl_display_mode.setVisible(visible)
        self._sync_stl_control_visibility()

    def _sync_stl_control_visibility(self, *_args) -> None:
        show_overhang_limit = (
            self._stl_controls_visible
            and self.stl_display_mode.currentText() == "Overhang angle"
        )
        self.overhang_limit.setVisible(show_overhang_limit)

    def set_volume_controls_visible(self, visible: bool) -> None:
        return

    def show_voxels(
        self,
        volume: NDArray,
        spacing: float,
        origin: tuple[float, float, float],
        label: str = "Binary",
        *,
        support_mask: NDArray | None = None,
        support_path: str | Path | None = None,
        raise_errors: bool = False,
    ) -> bool:
        self._last_volume_request = (
            volume,
            spacing,
            origin,
            label,
            support_mask,
            support_path,
        )
        return self._render_voxels(
            volume,
            spacing,
            origin,
            label,
            support_mask=support_mask,
            support_path=support_path,
            raise_errors=raise_errors,
        )

    def show_geometry_deviation(
        self,
        stl_path: str | Path,
        volume: NDArray,
        spacing: float,
        origin: tuple[float, float, float],
        *,
        label: str = "Binary",
        threshold: float = 0.5,
    ) -> dict[str, float]:
        try:
            preview = build_geometry_deviation_preview(
                stl_path,
                volume,
                spacing=spacing,
                origin=origin,
                label=label,
                threshold=threshold,
            )
            self.show_geometry_deviation_preview(preview)
            return preview.metrics
        except Exception as exc:
            self.show_message(f"Geometry deviation preview failed: {exc}")
            raise

    def show_geometry_deviation_preview(
        self,
        preview: GeometryDeviationPreview | GeometryDeviationPreviewPayload,
    ) -> None:
        try:
            preview = unpack_geometry_deviation_preview(preview)
            self._last_volume_request = None
            plotter = self._ensure_plotter()
            plotter.clear()
            self._prepare_scene(plotter)
            with suppress(Exception):
                plotter.enable_eye_dome_lighting()

            plotter.add_mesh(
                preview.original_mesh,
                color="#cbd5e1",
                opacity=0.22,
                smooth_shading=True,
                show_edges=False,
                lighting=True,
                ambient=0.5,
                diffuse=0.45,
                specular=0.08,
            )
            metrics = preview.metrics
            color_limits = _deviation_color_limits(metrics, preview.render_spacing)
            plotter.add_mesh(
                preview.deviation_surface,
                scalars="Deviation color (mm)",
                cmap=_deviation_jet_colormap(
                    DEVIATION_HEATMAP_SCALE_MM,
                    DEVIATION_HEATMAP_SCALE_MM,
                ),
                clim=color_limits,
                smooth_shading=True,
                show_edges=False,
                lighting=True,
                ambient=0.42,
                diffuse=0.66,
                specular=0.12,
                specular_power=12.0,
                show_scalar_bar=True,
                scalar_bar_args={
                    "title": "Deviation (mm)",
                    "vertical": True,
                    "height": 0.58,
                    "width": 0.08,
                    "position_x": 0.9,
                    "position_y": 0.2,
                    "n_labels": 5,
                    "fmt": "%.3g",
                },
            )
            plotter.add_mesh(preview.original_mesh.outline(), color="#334155", line_width=1.3)
            plotter.add_axes()
            self._set_cad_camera(
                plotter,
                _combined_mesh_bounds(preview.original_mesh, preview.deviation_surface),
            )
            self._render_plotter(plotter)
            self._title.setText("Geometry Deviation")
            self._status.setText(
                f"Mean |d| {metrics['mean_abs_mm']:.4g} mm, "
                f"p95 {metrics['p95_abs_mm']:.4g} mm"
            )
        except Exception as exc:
            self.show_message(f"Geometry deviation preview failed: {exc}")
            raise

    def _render_voxels(
        self,
        volume: NDArray,
        spacing: float,
        origin: tuple[float, float, float],
        label: str,
        *,
        support_mask: NDArray | None = None,
        support_path: str | Path | None = None,
        raise_errors: bool = False,
    ) -> bool:
        try:
            pv = self._load_pyvista()
            plotter = self._ensure_plotter()
            data, stride = self._prepare_volume_data(volume)
            binary_like = self._is_binary_volume(data)
            raw_support_data = self._prepare_support_overlay_data(
                support_mask,
                stride,
                data.shape,
                bridge_diagonals=False,
            )
            support_data = (
                self._bridge_diagonal_support_overlay(raw_support_data)
                if raw_support_data is not None
                else None
            )
            support_mesh = self._support_preview_mesh_for_path(support_path)
            has_support_overlay = (
                support_mesh is not None
                or (
                    support_data is not None
                    and binary_like
                    and bool(np.any(support_data))
                )
            )
            if has_support_overlay and raw_support_data is not None:
                data = data.copy()
                data[raw_support_data] = 0.0
            render_spacing = spacing * stride
            grid = self._make_volume_cell_grid(pv, data, render_spacing, origin, label)

            plotter.clear()
            self._prepare_scene(plotter)
            with suppress(Exception):
                plotter.enable_eye_dome_lighting()

            self._add_isosurface(plotter, data, spacing=render_spacing, origin=origin)
            if support_mesh is not None:
                self._add_support_mesh_overlay(plotter, support_mesh)
            elif has_support_overlay and support_data is not None:
                self._add_isosurface(
                    plotter,
                    support_data.astype(np.float32),
                    spacing=render_spacing,
                    origin=origin,
                    color="#e85d04",
                    edge_color="#7c2d12",
                )
            plotter.add_mesh(grid.outline(), color="#1f2937", line_width=1.8)
            plotter.add_axes()
            self._set_cad_camera(plotter, grid.bounds)
            self._render_plotter(plotter)

            mode = "Isosurface"
            if has_support_overlay:
                mode = f"{mode} + Support"
            self._title.setText(mode)
            self._status.setText("")
            return True
        except Exception as exc:
            self.show_message(f"Voxel preview failed: {exc}")
            if raise_errors:
                raise
            return False

    def _support_preview_mesh_for_path(self, support_path: str | Path | None):
        if support_path is None:
            return None
        path = Path(support_path)
        if not path.exists():
            return None
        with suppress(Exception):
            resolved = path.resolve()
            stat = resolved.stat()
            cache_key = (str(resolved), int(stat.st_mtime_ns), int(stat.st_size))
            cached_key = getattr(self, "_support_preview_mesh_cache_key", None)
            cached_mesh = getattr(self, "_support_preview_mesh_cache", None)
            if cached_key == cache_key and cached_mesh is not None:
                return cached_mesh
            mesh, _original_cells = prepare_stl_preview_mesh(resolved)
            if int(getattr(mesh, "n_cells", 0) or 0) <= 0:
                return None
            self._support_preview_mesh_cache_key = cache_key
            self._support_preview_mesh_cache = mesh
            return mesh
        return None

    def show_message(self, message: str) -> None:
        self._status.setText(message)

    def _load_pyvista(self):
        if self._pyvista is None:
            import pyvista as pv

            self._pyvista = pv
        return self._pyvista

    def _ensure_plotter(self):
        if self._plotter is None:
            from pyvistaqt import QtInteractor

            self._plotter = QtInteractor(self.widget)
            self._layout.addWidget(self._plotter.interactor)
            self._status.setText("")
        return self._plotter

    def _prepare_volume_data(
        self,
        volume: NDArray,
    ) -> tuple[NDArray[np.float32], int]:
        data = np.asarray(volume)
        if data.ndim != 3:
            raise ValueError("Volume preview requires a 3D array.")
        if 0 in data.shape:
            raise ValueError("Volume preview cannot render an empty array.")

        max_dimension, target_voxels = self._volume_lod_budget()
        stride_by_dimension = int(np.ceil(max(data.shape) / max_dimension))
        stride_by_budget = int(np.ceil((data.size / target_voxels) ** (1.0 / 3.0)))
        stride = max(1, stride_by_dimension, stride_by_budget)
        if stride > 1:
            data = data[::stride, ::stride, ::stride]

        if data.dtype == bool:
            normalized = data.astype(np.float32)
        else:
            normalized = data.astype(np.float32, copy=False)
            if normalized.size and normalized.max() > 1.0:
                scale = 100.0 if normalized.max() <= 100.0 else float(normalized.max())
                normalized = normalized / scale
            normalized = np.clip(normalized, 0.0, 1.0)

        if not np.any(normalized > 0.0):
            raise ValueError("Volume preview has no visible voxels.")
        return np.ascontiguousarray(normalized), stride

    def _prepare_support_overlay_data(
        self,
        support_mask: NDArray | None,
        stride: int,
        shape: tuple[int, int, int],
        *,
        bridge_diagonals: bool = True,
    ) -> NDArray[np.bool_] | None:
        if support_mask is None:
            return None
        mask = np.asarray(support_mask, dtype=bool)
        if mask.ndim != 3:
            return None
        if stride > 1:
            mask = self._max_pool_binary(mask.astype(np.float32), stride) > 0.0
        slices = tuple(slice(0, min(mask.shape[axis], shape[axis])) for axis in range(3))
        prepared = np.zeros(shape, dtype=bool)
        prepared[slices] = mask[slices]
        if bridge_diagonals:
            prepared = self._bridge_diagonal_support_overlay(prepared)
        return np.ascontiguousarray(prepared)

    def _bridge_diagonal_support_overlay(
        self,
        mask: NDArray[np.bool_],
    ) -> NDArray[np.bool_]:
        prepared = np.asarray(mask, dtype=bool).copy()
        if prepared.ndim != 3 or prepared.shape[0] < 2 or prepared.shape[1] < 2:
            return prepared

        down_right_pairs = mask[:-1, :-1, :] & mask[1:, 1:, :]
        prepared[:-1, 1:, :] |= down_right_pairs
        prepared[1:, :-1, :] |= down_right_pairs

        down_left_pairs = mask[:-1, 1:, :] & mask[1:, :-1, :]
        prepared[:-1, :-1, :] |= down_left_pairs
        prepared[1:, 1:, :] |= down_left_pairs
        return prepared

    def _volume_lod_budget(self) -> tuple[int, int]:
        return 768, 160_000_000

    def _is_binary_volume(self, data: NDArray[np.float32]) -> bool:
        occupied = data[data > 0.0]
        return bool(occupied.size and np.allclose(occupied, 1.0))

    def _max_pool_binary(self, data: NDArray[np.float32], stride: int) -> NDArray[np.float32]:
        pad_width = []
        for size in data.shape:
            remainder = size % stride
            pad_width.append((0, 0 if remainder == 0 else stride - remainder))

        padded = np.pad(data > 0.0, pad_width, mode="constant", constant_values=False)
        pooled = padded.reshape(
            padded.shape[0] // stride,
            stride,
            padded.shape[1] // stride,
            stride,
            padded.shape[2] // stride,
            stride,
        ).max(axis=(1, 3, 5))
        return pooled.astype(np.float32)

    def _make_volume_cell_grid(
        self,
        pv,
        data: NDArray[np.float32],
        spacing: float,
        origin: tuple[float, float, float],
        label: str,
    ):
        grid = pv.ImageData(
            dimensions=(data.shape[0] + 1, data.shape[1] + 1, data.shape[2] + 1),
            spacing=(spacing, spacing, spacing),
            origin=origin,
        )
        grid.cell_data[label] = data.astype(np.float32, copy=False).ravel(order="F")
        grid.set_active_scalars(label, preference="cell")
        return grid

    def _make_isosurface_mesh(
        self,
        pv,
        data: NDArray[np.float32],
        *,
        spacing: float,
        origin: tuple[float, float, float],
        level: float = 0.5,
    ):
        padded = np.pad(
            data.astype(np.float32, copy=False),
            ((1, 1), (1, 1), (1, 1)),
            mode="constant",
            constant_values=0.0,
        )
        if float(padded.min()) >= level or float(padded.max()) <= level:
            raise ValueError("Isosurface requires values on both sides of the threshold.")

        from skimage.measure import marching_cubes

        verts, faces, _normals, _values = marching_cubes(
            padded,
            level=level,
            spacing=(float(spacing), float(spacing), float(spacing)),
        )
        verts = verts - (0.5 * float(spacing)) + np.asarray(origin, dtype=np.float32)
        vtk_faces = np.empty((faces.shape[0], 4), dtype=np.int64)
        vtk_faces[:, 0] = 3
        vtk_faces[:, 1:] = faces
        return pv.PolyData(verts, vtk_faces.ravel())

    def _add_isosurface(
        self,
        plotter,
        data: NDArray[np.float32],
        *,
        spacing: float,
        origin: tuple[float, float, float],
        color: str | None = None,
        edge_color: str = "#1f2937",
    ) -> None:
        pv = self._load_pyvista()
        surface = self._make_isosurface_mesh(
            pv,
            data,
            spacing=spacing,
            origin=origin,
            level=0.5,
        )
        with suppress(Exception):
            surface = surface.compute_normals(
                point_normals=True,
                cell_normals=False,
                auto_orient_normals=True,
                consistent_normals=True,
                split_vertices=False,
            )
        plotter.add_mesh(
            surface,
            color=color or "#2d6f99",
            smooth_shading=False,
            show_edges=False,
            lighting=True,
            ambient=0.56,
            diffuse=0.5,
            specular=0.0,
            specular_power=10.0,
            silhouette={"color": edge_color, "line_width": 0.8},
        )

    def _geometry_deviation_surface(
        self,
        printed_surface,
        original_mesh,
        *,
        color_zero_tolerance: float = 0.0,
    ):
        with suppress(Exception):
            original_mesh = original_mesh.compute_normals(
                point_normals=True,
                cell_normals=False,
                auto_orient_normals=True,
                consistent_normals=True,
                split_vertices=False,
            )
        deviation_surface = printed_surface.compute_implicit_distance(
            original_mesh,
            inplace=False,
        )
        distances = np.asarray(deviation_surface.point_data["implicit_distance"], dtype=np.float64)
        distances = np.nan_to_num(distances, nan=0.0, posinf=0.0, neginf=0.0)
        deviation_surface.point_data["Deviation (mm)"] = distances.astype(np.float32)
        deviation_surface.point_data["Deviation color (mm)"] = _deviation_color_values_mm(
            distances,
            zero_tolerance=color_zero_tolerance,
        )
        abs_distances = np.abs(distances)
        negative = distances[distances < 0.0]
        positive = distances[distances > 0.0]
        metrics = {
            "mean_abs_mm": float(abs_distances.mean()) if abs_distances.size else 0.0,
            "p95_abs_mm": float(np.percentile(abs_distances, 95)) if abs_distances.size else 0.0,
            "max_abs_mm": float(abs_distances.max()) if abs_distances.size else 0.0,
            "min_signed_mm": float(distances.min()) if distances.size else 0.0,
            "max_signed_mm": float(distances.max()) if distances.size else 0.0,
            "negative_scale_mm": abs(float(negative.min())) if negative.size else 0.0,
            "positive_scale_mm": float(positive.max()) if positive.size else 0.0,
            "color_zero_tolerance_mm": float(max(color_zero_tolerance, 0.0)),
            "sample_count": float(distances.size),
        }
        return deviation_surface, metrics

    def _add_overhang_mesh(self, plotter, mesh, overhang_limit: float) -> None:
        overhang_mesh = mesh.copy(deep=False)
        if "Normals" not in overhang_mesh.cell_data:
            with suppress(Exception):
                overhang_mesh = overhang_mesh.compute_normals(
                    point_normals=False,
                    cell_normals=True,
                    auto_orient_normals=True,
                    consistent_normals=True,
                    split_vertices=False,
                )
        normals = np.asarray(overhang_mesh.cell_data.get("Normals"))
        if normals.shape != (overhang_mesh.n_cells, 3):
            centers = overhang_mesh.cell_centers().points
            normals = centers - np.asarray(overhang_mesh.center)
        angles = compute_overhang_angles(normals)
        overhang_mesh.cell_data["Overhang angle"] = angles
        overhang_limit = max(1.0, float(overhang_limit))
        plotter.add_mesh(
            overhang_mesh,
            scalars="Overhang angle",
            preference="cell",
            cmap="jet_r",
            clim=(0.0, overhang_limit),
            show_edges=False,
            lighting=True,
            ambient=0.32,
            diffuse=0.82,
            specular=0.12,
            smooth_shading=False,
            show_scalar_bar=True,
            scalar_bar_args={
                "title": "Overhang angle (deg)",
                "vertical": True,
                "height": 0.55,
                "width": 0.08,
                "position_x": 0.9,
                "position_y": 0.22,
            },
            silhouette={"color": "#1f2937", "line_width": 1.8},
        )

    def _prepare_scene(self, plotter) -> None:
        pv = self._load_pyvista()
        plotter.set_background("#f8fafc", top="#dbe7f3")
        plotter.enable_anti_aliasing()

        with suppress(Exception):
            plotter.remove_all_lights()

        with suppress(Exception):
            plotter.enable_parallel_projection()

        plotter.add_light(
            pv.Light(
                position=(2.5, -3.0, 4.0),
                focal_point=(0.0, 0.0, 0.0),
                color="#ffffff",
                intensity=0.95,
            )
        )
        plotter.add_light(
            pv.Light(
                position=(-3.0, 2.0, 2.0),
                focal_point=(0.0, 0.0, 0.0),
                color="#dbeafe",
                intensity=0.55,
            )
        )
        plotter.add_light(
            pv.Light(
                position=(0.0, 0.0, -5.0),
                focal_point=(0.0, 0.0, 0.0),
                color="#ffffff",
                intensity=0.25,
            )
        )

    def _render_plotter(self, plotter) -> None:
        with suppress(Exception):
            plotter.reset_camera_clipping_range()
        plotter.render()

    def _set_cad_camera(self, plotter, bounds) -> None:
        bounds_array = np.asarray(bounds, dtype=float)
        center = np.array(
            [
                0.5 * (bounds_array[0] + bounds_array[1]),
                0.5 * (bounds_array[2] + bounds_array[3]),
                0.5 * (bounds_array[4] + bounds_array[5]),
            ]
        )
        span = np.array(
            [
                max(bounds_array[1] - bounds_array[0], 1.0),
                max(bounds_array[3] - bounds_array[2], 1.0),
                max(bounds_array[5] - bounds_array[4], 1.0),
            ]
        )
        radius = float(np.linalg.norm(span))
        position = center + np.array([1.35 * radius, -1.65 * radius, 1.15 * radius])

        plotter.camera_position = (
            tuple(position),
            tuple(center),
            (0.0, 0.0, 1.0),
        )
        with suppress(Exception):
            plotter.camera.SetViewUp(0.0, 0.0, 1.0)
            plotter.camera.SetFocalPoint(*center)
            plotter.camera.SetPosition(*position)
            plotter.camera.OrthogonalizeViewUp()
        plotter.reset_camera_clipping_range()

    def _add_feature_edges(self, plotter, mesh) -> None:
        try:
            edges = mesh.extract_feature_edges(
                boundary_edges=True,
                feature_edges=True,
                manifold_edges=False,
                non_manifold_edges=True,
                feature_angle=28,
            )
        except Exception:
            return

        if edges.n_cells > 0:
            plotter.add_mesh(edges, color="#0f172a", line_width=1.9)
