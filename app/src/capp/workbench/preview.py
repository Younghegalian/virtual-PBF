from __future__ import annotations

from contextlib import suppress
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


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
    if mesh.n_cells > 450_000:
        reduction = min(0.92, 1.0 - (220_000.0 / float(mesh.n_cells)))
        mesh = mesh.decimate_pro(reduction, preserve_topology=True)

    with suppress(Exception):
        mesh = mesh.compute_normals(
            point_normals=True,
            cell_normals=False,
            auto_orient_normals=True,
            consistent_normals=True,
            split_vertices=False,
        )
    return mesh, original_cells


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
        self.stl_display_mode = QComboBox()
        self.stl_display_mode.setObjectName("PreviewStlMode")
        self.stl_display_mode.addItems(["Shaded", "Overhang angle"])
        self.stl_display_mode.setVisible(show_stl_controls)
        self.overhang_limit = QLineEdit("60")
        self.overhang_limit.setObjectName("PreviewOverhangLimit")
        self.overhang_limit.setFixedWidth(44)
        self.overhang_limit.setVisible(show_stl_controls)
        self._render_mode = QComboBox()
        self._render_mode.setObjectName("PreviewMode")
        self._render_mode.addItems(
            [
                "Isosurface",
                "Smooth Surface",
                "ParaView Volume",
                "Smooth Volume",
                "Voxel Blocks",
                "Slices",
                "Points",
            ]
        )
        self._render_mode.currentTextChanged.connect(self._rerender_last_volume)
        self._render_mode.setVisible(False)
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
        header_layout.addWidget(self._render_mode)
        self._layout.addWidget(header)
        self._layout.addWidget(self._status)
        self._plotter = None
        self._pyvista = None
        self._last_volume_request = None

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
            self._render_mode.setVisible(False)
            plotter = self._ensure_plotter()
            plotter.clear()
            self._prepare_scene(plotter)
            with suppress(Exception):
                plotter.enable_eye_dome_lighting()

            if display_mode == "Overhang angle":
                self._add_overhang_mesh(plotter, mesh, overhang_limit)
            else:
                show_edges = mesh.n_cells <= 120_000
                plotter.add_mesh(
                    mesh,
                    color="#cbd5e1",
                    edge_color="#475569",
                    show_edges=show_edges,
                    smooth_shading=False,
                    lighting=True,
                    ambient=0.32,
                    diffuse=0.82,
                    specular=0.22,
                    specular_power=20,
                    silhouette={"color": "#1f2937", "line_width": 2.0},
                )
            if mesh.n_cells <= 180_000:
                self._add_feature_edges(plotter, mesh)
            plotter.add_mesh(mesh.outline(), color="#334155", line_width=1.4)
            plotter.add_axes()
            self._set_cad_camera(plotter, mesh.bounds)
            suffix = ""
            if original_cells is not None and original_cells > mesh.n_cells:
                suffix = f" ({mesh.n_cells:,}/{original_cells:,} cells)"
            if display_mode == "Overhang angle":
                suffix = f"{suffix} overhang 0-{overhang_limit:g} deg"
            self._title.setText(f"STL Preview: {Path(path).name}{suffix}")
        except Exception as exc:
            self.show_message(f"STL preview failed: {exc}")

    def set_stl_controls_visible(self, visible: bool) -> None:
        self.stl_display_mode.setVisible(visible)
        self.overhang_limit.setVisible(visible)

    def show_voxels(
        self,
        volume: NDArray,
        spacing: float,
        origin: tuple[float, float, float],
        label: str = "Binary",
    ) -> None:
        self._last_volume_request = (volume, spacing, origin, label)
        self._render_voxels(volume, spacing, origin, label)

    def _render_voxels(
        self,
        volume: NDArray,
        spacing: float,
        origin: tuple[float, float, float],
        label: str,
    ) -> None:
        try:
            pv = self._load_pyvista()
            plotter = self._ensure_plotter()
            self._render_mode.setVisible(True)
            render_mode = self._render_mode.currentText()
            data, stride = self._prepare_volume_data(volume, render_mode)
            binary_like = self._is_binary_volume(data)
            smooth_volume = render_mode == "Smooth Volume"
            surface_mode = render_mode in {"Isosurface", "Smooth Surface"}
            smooth_point_grid = False
            if smooth_volume:
                data = self._prepare_smooth_volume_data(data, binary_like=binary_like)
                smooth_point_grid = data.size <= 128_000_000
            elif render_mode == "Smooth Surface":
                data = self._prepare_smooth_volume_data(data, binary_like=binary_like)
            block_stride = 1
            if surface_mode:
                render_spacing = spacing * stride
                grid = self._make_volume_cell_grid(pv, data, render_spacing, origin, label)
            elif smooth_point_grid:
                render_spacing = spacing * stride
                grid = self._make_volume_point_grid(pv, data, render_spacing, origin, label)
            elif render_mode == "Voxel Blocks" and binary_like:
                data, block_stride = self._prepare_binary_block_data(data)
                render_spacing = spacing * stride * block_stride
                grid = self._make_binary_cell_grid(pv, data, render_spacing, origin, label)
            elif render_mode == "Points" and binary_like:
                render_spacing = spacing * stride
                grid = self._make_point_grid(pv, data, render_spacing, origin, label)
            else:
                render_spacing = spacing * stride
                grid = self._make_volume_cell_grid(pv, data, render_spacing, origin, label)

            plotter.clear()
            self._prepare_scene(plotter)
            with suppress(Exception):
                plotter.enable_eye_dome_lighting()

            fallback_stride = 0
            volume_mapper = ""
            if surface_mode:
                self._add_isosurface(
                    plotter,
                    data,
                    spacing=render_spacing,
                    origin=origin,
                    smooth=render_mode == "Smooth Surface",
                )
            elif render_mode == "Points" and binary_like:
                fallback_stride = self._add_binary_points(
                    plotter,
                    data,
                    spacing=render_spacing,
                    origin=origin,
                )
            elif render_mode == "Voxel Blocks" and binary_like:
                fallback_stride = self._add_binary_blocks(plotter, grid, label, data)
            elif render_mode == "Slices":
                self._add_volume_slices(plotter, grid, label)
            else:
                try:
                    volume_mapper = self._add_paraview_volume(
                        plotter,
                        grid,
                        label,
                        data,
                        binary_like=binary_like,
                        smooth=smooth_point_grid,
                        preference="point" if smooth_point_grid else "cell",
                    )
                except Exception:
                    if smooth_volume:
                        self._add_volume(plotter, grid, label, data)
                    elif binary_like:
                        grid = self._make_binary_cell_grid(pv, data, render_spacing, origin, label)
                        fallback_stride = self._add_binary_blocks(plotter, grid, label, data)
                    else:
                        self._add_volume(plotter, grid, label, data)
            plotter.add_mesh(grid.outline(), color="#1f2937", line_width=1.8)
            plotter.add_axes()
            self._set_cad_camera(plotter, grid.bounds)

            total_stride = stride * block_stride
            suffix = f" (safety stride x{total_stride})" if total_stride > 1 else ""
            if fallback_stride > 1:
                suffix = f"{suffix} point fallback 1/{fallback_stride}"
            if volume_mapper:
                suffix = f"{suffix} {volume_mapper}"
            mode = render_mode if render_mode != "Points" else "Point Preview"
            self._title.setText(f"{mode}: {label}{suffix}")
        except Exception as exc:
            self.show_message(f"Voxel preview failed: {exc}")

    def _rerender_last_volume(self) -> None:
        if self._last_volume_request is None:
            return
        volume, spacing, origin, label = self._last_volume_request
        self._render_voxels(volume, spacing, origin, label)

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
        render_mode: str = "ParaView Volume",
    ) -> tuple[NDArray[np.float32], int]:
        data = np.asarray(volume)
        if data.ndim != 3:
            raise ValueError("Volume preview requires a 3D array.")
        if 0 in data.shape:
            raise ValueError("Volume preview cannot render an empty array.")

        max_dimension, target_voxels = self._volume_lod_budget(render_mode)
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

    def _volume_lod_budget(self, render_mode: str) -> tuple[int, int]:
        if render_mode == "Isosurface":
            return 768, 160_000_000
        if render_mode == "Smooth Surface":
            return 512, 96_000_000
        if render_mode in {"ParaView Volume", "Slices"}:
            return 1536, 512_000_000
        if render_mode == "Points":
            return 1536, 512_000_000
        return 1536, 512_000_000

    def _volume_opacity(self, data: NDArray[np.float32]) -> list[float]:
        if np.count_nonzero((data > 0.0) & (data < 1.0)):
            return [0.0, 0.02, 0.08, 0.22, 0.42, 0.68]
        return [0.0, 0.0, 0.72, 0.9]

    def _prepare_smooth_volume_data(
        self,
        data: NDArray[np.float32],
        *,
        binary_like: bool,
    ) -> NDArray[np.float32]:
        if data.size > 96_000_000:
            return np.ascontiguousarray(data)

        from scipy.ndimage import gaussian_filter

        smooth = data.astype(np.float32, copy=False)
        sigma = 0.72 if binary_like else 0.48
        smooth = gaussian_filter(smooth, sigma=sigma, mode="nearest")
        max_value = float(smooth.max()) if smooth.size else 0.0
        if max_value > 0.0:
            smooth = smooth / max_value
        return np.ascontiguousarray(np.clip(smooth, 0.0, 1.0))

    def _is_binary_volume(self, data: NDArray[np.float32]) -> bool:
        occupied = data[data > 0.0]
        return bool(occupied.size and np.allclose(occupied, 1.0))

    def _prepare_binary_block_data(
        self,
        data: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], int]:
        occupied = int(np.count_nonzero(data))
        max_blocks = 1_000_000
        stride = max(1, int(np.ceil((occupied / max_blocks) ** (1.0 / 3.0))))
        if stride > 1:
            data = self._max_pool_binary(data, stride)
        return np.ascontiguousarray(data), stride

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

    def _make_binary_cell_grid(
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
        grid.cell_data[label] = data.astype(np.uint8, copy=False).ravel(order="F")
        return grid

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

    def _make_volume_point_grid(
        self,
        pv,
        data: NDArray[np.float32],
        spacing: float,
        origin: tuple[float, float, float],
        label: str,
    ):
        point_data = np.pad(data, ((0, 1), (0, 1), (0, 1)), mode="edge")
        grid = pv.ImageData(
            dimensions=point_data.shape,
            spacing=(spacing, spacing, spacing),
            origin=origin,
        )
        grid.point_data[label] = point_data.astype(np.float32, copy=False).ravel(order="F")
        grid.set_active_scalars(label, preference="point")
        return grid

    def _make_point_grid(
        self,
        pv,
        data: NDArray[np.float32],
        spacing: float,
        origin: tuple[float, float, float],
        label: str,
    ):
        grid = pv.ImageData(
            dimensions=data.shape,
            spacing=(spacing, spacing, spacing),
            origin=origin,
        )
        grid.point_data[label] = data.astype(np.float32, copy=False).ravel(order="F")
        grid.set_active_scalars(label)
        return grid

    def _add_binary_points(
        self,
        plotter,
        data: NDArray[np.float32],
        *,
        spacing: float,
        origin: tuple[float, float, float],
    ) -> int:
        pv = self._load_pyvista()
        coords = np.argwhere(data > 0.0)
        if coords.size == 0:
            raise ValueError("Voxel preview has no visible voxels.")

        max_points = 2_000_000
        point_stride = max(1, int(np.ceil(coords.shape[0] / max_points)))
        if point_stride > 1:
            coords = coords[::point_stride]

        points = coords.astype(np.float32)
        points = points * float(spacing) + np.asarray(origin, dtype=np.float32)
        points += 0.5 * float(spacing)
        cloud = pv.PolyData(points)
        plotter.add_points(
            cloud,
            color="#2563eb",
            point_size=2.6 if point_stride == 1 else 2.2,
            opacity=0.72,
            render_points_as_spheres=False,
        )
        return point_stride

    def _add_binary_blocks(
        self,
        plotter,
        grid,
        label: str,
        data: NDArray[np.float32],
    ) -> int:
        try:
            blocks = grid.threshold(0.5, scalars=label, preference="cell")
            if blocks.n_cells == 0:
                raise ValueError("No occupied voxel cells to render.")
            surface = blocks.extract_surface()
            if surface.n_cells == 0:
                raise ValueError("No occupied voxel surface to render.")
            plotter.add_mesh(
                surface,
                color="#2563eb",
                edge_color="#1e3a8a",
                show_edges=surface.n_cells <= 180_000,
                line_width=0.08,
                opacity=1.0,
                lighting=True,
                ambient=0.28,
                diffuse=0.84,
                specular=0.08,
                specular_power=10.0,
                smooth_shading=False,
                culling=False,
            )
        except Exception:
            return self._add_binary_points(
                plotter,
                data,
                spacing=float(grid.spacing[0]),
                origin=tuple(float(v) for v in grid.origin),
            )
        return 0

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
        smooth: bool,
    ) -> None:
        pv = self._load_pyvista()
        try:
            surface = self._make_isosurface_mesh(
                pv,
                data,
                spacing=spacing,
                origin=origin,
                level=0.45 if smooth else 0.5,
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
                color="#2f6f9f" if smooth else "#2d6f99",
                smooth_shading=smooth,
                show_edges=False,
                lighting=True,
                ambient=0.46 if smooth else 0.56,
                diffuse=0.62 if smooth else 0.5,
                specular=0.08 if smooth else 0.0,
                specular_power=10.0,
                silhouette={"color": "#1f2937", "line_width": 0.8} if smooth else False,
            )
        except Exception:
            grid = self._make_binary_cell_grid(pv, data > 0.0, spacing, origin, "Binary")
            self._add_binary_blocks(plotter, grid, "Binary", data)

    def _add_paraview_volume(
        self,
        plotter,
        grid,
        label: str,
        data: NDArray[np.float32],
        *,
        binary_like: bool,
        smooth: bool = False,
        preference: str = "cell",
    ) -> str:
        if smooth:
            opacity = [0.0, 0.0, 0.025, 0.14, 0.38, 0.78]
            cmap = "Blues"
            opacity_unit_distance = max(float(grid.spacing[0]) * 0.58, 1e-4)
            ambient = 0.14
            diffuse = 0.9
            specular = 0.2
            specular_power = 18.0
        elif binary_like:
            opacity = [0.0, 0.0, 0.08, 0.68, 0.92]
            cmap = "Blues"
            opacity_unit_distance = max(float(grid.spacing[0]) * 0.45, 1e-4)
            ambient = 0.22
            diffuse = 0.82
            specular = 0.12
            specular_power = 12.0
        else:
            opacity = [0.0, 0.015, 0.06, 0.18, 0.34, 0.58]
            cmap = "viridis"
            opacity_unit_distance = max(float(grid.spacing[0]) * 0.75, 1e-4)
            ambient = 0.22
            diffuse = 0.82
            specular = 0.12
            specular_power = 12.0

        options = {
            "scalars": label,
            "clim": (0.0, 1.0),
            "opacity": opacity,
            "cmap": cmap,
            "blending": "composite",
            "shade": True,
            "ambient": ambient,
            "diffuse": diffuse,
            "specular": specular,
            "specular_power": specular_power,
            "show_scalar_bar": False,
            "preference": preference,
            "opacity_unit_distance": opacity_unit_distance,
        }
        last_error = None
        mappers = ("gpu", "smart", "fixed_point") if smooth else ("fixed_point", "smart", "gpu")
        for mapper in mappers:
            try:
                actor = plotter.add_volume(grid, mapper=mapper, **options)
                if smooth:
                    self._set_volume_interpolation(actor, "linear")
                else:
                    self._set_volume_interpolation(actor, "nearest")
                return f"[{mapper}]"
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Volume mapper failed: {last_error}")

    def _add_volume_slices(self, plotter, grid, label: str) -> None:
        bounds = grid.bounds
        slices = grid.slice_orthogonal(
            x=0.5 * (bounds[0] + bounds[1]),
            y=0.5 * (bounds[2] + bounds[3]),
            z=0.5 * (bounds[4] + bounds[5]),
        )
        plotter.add_mesh(
            slices,
            scalars=label,
            cmap="viridis",
            clim=(0.0, 1.0),
            opacity=0.9,
            lighting=False,
            show_scalar_bar=False,
        )

    def _add_volume(self, plotter, grid, label: str, data: NDArray[np.float32]) -> None:
        options = {
            "scalars": label,
            "clim": (0.0, 1.0),
            "opacity": self._volume_opacity(data),
            "cmap": "Blues",
            "blending": "composite",
            "shade": True,
            "diffuse": 0.8,
            "specular": 0.15,
            "show_scalar_bar": False,
        }
        try:
            actor = plotter.add_volume(grid, mapper="gpu", **options)
        except Exception:
            actor = plotter.add_volume(grid, mapper="smart", **options)
        self._set_volume_interpolation(actor, "nearest")

    def _set_volume_interpolation(self, actor, mode: str) -> None:
        if isinstance(actor, list):
            for item in actor:
                self._set_volume_interpolation(item, mode)
            return
        setter = (
            "SetInterpolationTypeToLinear"
            if mode == "linear"
            else "SetInterpolationTypeToNearest"
        )
        with suppress(Exception):
            getattr(actor.GetProperty(), setter)()
        with suppress(Exception):
            getattr(actor.prop, setter)()

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

