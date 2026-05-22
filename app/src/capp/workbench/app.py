from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from time import perf_counter

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


def _preset_directory_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", (name or "Machine Map").strip())
    safe = safe.strip("._-")
    return safe or "Machine_Map"


def _default_machine_preset_library_root() -> Path:
    return Path("workbench_library") / "machine_presets"


def _legacy_model_calibration_root() -> Path:
    return Path("examples") / "outputs" / "model_calibration"


def _intermediate_model_calibration_root() -> Path:
    return Path("workbench_library") / "model_calibration"


def _machine_preset_folder(library_root: str | Path, preset_name: str) -> Path:
    return Path(library_root) / _preset_directory_name(preset_name)


def _model_calibration_preset_output_dir(library_root: str | Path, preset_name: str) -> Path:
    return _machine_preset_folder(library_root, preset_name) / "calibration"


def _machine_map_preset_output_dir(library_root: str | Path, preset_name: str) -> Path:
    return _machine_preset_folder(library_root, preset_name) / "map"


def _normalize_orientation_angles(angles: tuple[float, float, float]) -> tuple[float, float, float]:
    normalized = []
    for angle in angles:
        value = ((float(angle) + 180.0) % 360.0) - 180.0
        if abs(value) < 1e-9:
            value = 0.0
        normalized.append(value)
    return tuple(normalized)


def _orientation_is_identity(angles: tuple[float, float, float]) -> bool:
    return all(abs(value) < 1e-9 for value in _normalize_orientation_angles(angles))


def _orientation_label(angles: tuple[float, float, float]) -> str:
    normalized = _normalize_orientation_angles(angles)
    if _orientation_is_identity(normalized):
        return "as_loaded"

    tokens = []
    for axis, value in zip(("rx", "ry", "rz"), normalized, strict=True):
        if abs(value) < 1e-9:
            continue
        sign = "p" if value >= 0 else "m"
        text = f"{abs(value):g}".replace(".", "p")
        tokens.append(f"{axis}{sign}{text}")
    return "_".join(tokens)


def _oriented_geometry_path(
    source_path: str | Path,
    output_dir: str | Path,
    angles: tuple[float, float, float],
) -> Path:
    source = Path(source_path).resolve()
    normalized = _normalize_orientation_angles(angles)
    if _orientation_is_identity(normalized):
        return source

    target = Path(output_dir) / "intermediate" / "active_oriented_geometry.stl"

    import numpy as np
    import trimesh

    loaded = trimesh.load_mesh(source, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geom
            for geom in loaded.geometry.values()
            if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0
        ]
        if not meshes:
            raise ValueError(f"No mesh geometry found in {source}.")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type from {source}: {type(loaded)!r}")

    mesh = loaded.copy()
    center = np.asarray(mesh.bounds, dtype=np.float64).mean(axis=0)
    transform = np.eye(4, dtype=np.float64)
    axes = (
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    )
    for angle, axis in zip(normalized, axes, strict=True):
        if abs(angle) < 1e-9:
            continue
        transform = (
            trimesh.transformations.rotation_matrix(
                np.deg2rad(angle),
                axis,
                point=center,
            )
            @ transform
        )
    mesh.apply_transform(transform)
    target.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(target)
    return target


def _workbench_colormap(values):
    import numpy as np

    palette = np.asarray(
        [
            (22, 34, 68),
            (27, 87, 145),
            (31, 154, 177),
            (113, 204, 159),
            (246, 211, 101),
            (231, 111, 81),
        ],
        dtype=np.float64,
    )
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    scaled = clipped * (len(palette) - 1)
    lower = np.floor(scaled).astype(np.int32)
    upper = np.clip(lower + 1, 0, len(palette) - 1)
    fraction = (scaled - lower)[..., None]
    rgb = palette[lower] * (1.0 - fraction) + palette[upper] * fraction
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _roi_overlay_rgb(target_array, simulated_array):
    import numpy as np

    target = np.asarray(target_array, dtype=bool)
    simulated = np.asarray(simulated_array, dtype=bool)
    rgb = np.full((*target.shape, 3), 248, dtype=np.uint8)
    target_only = target & ~simulated
    simulated_only = simulated & ~target
    overlap = target & simulated
    rgb[target_only] = (224, 132, 58)
    rgb[simulated_only] = (56, 116, 196)
    rgb[overlap] = (62, 153, 101)
    return np.ascontiguousarray(rgb)


def _roi_mask_rgb(mask_array, foreground=(35, 101, 158)):
    import numpy as np

    mask = np.asarray(mask_array, dtype=bool)
    rgb = np.full((*mask.shape, 3), 248, dtype=np.uint8)
    rgb[mask] = foreground
    return np.ascontiguousarray(rgb)


def _roi_grayscale_rgb(gray_array):
    import numpy as np

    gray = np.asarray(gray_array)
    if gray.ndim > 2:
        gray = gray[..., 0]
    if gray.size == 0:
        gray = np.zeros((1, 1), dtype=np.uint8)
    gray = gray.astype(np.float64)
    finite = np.isfinite(gray)
    if finite.any():
        lower = float(np.nanmin(gray[finite]))
        upper = float(np.nanmax(gray[finite]))
        if upper > lower:
            gray = (gray - lower) * (255.0 / (upper - lower))
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(np.repeat(gray[..., None], 3, axis=2))


def _resize_array_nearest(array, shape: tuple[int, int]):
    import numpy as np

    values = np.asarray(array)
    if values.ndim > 2:
        values = values[..., 0]
    if values.shape == shape:
        return values
    if values.size == 0 or shape[0] <= 0 or shape[1] <= 0:
        return np.zeros(shape, dtype=values.dtype)
    y_index = np.linspace(0, values.shape[0] - 1, shape[0]).round().astype(int)
    x_index = np.linspace(0, values.shape[1] - 1, shape[1]).round().astype(int)
    return values[np.ix_(y_index, x_index)]


def _mask_outline(mask_array):
    import numpy as np

    mask = np.asarray(mask_array, dtype=bool)
    if mask.size == 0:
        return mask
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return mask & ~eroded


def _dilate_mask(mask_array, iterations: int = 1):
    import numpy as np

    mask = np.asarray(mask_array, dtype=bool)
    for _ in range(max(0, int(iterations))):
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        mask = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )
    return mask


def _roi_outline_rgb(target_array, simulated_array, background_array=None):
    import numpy as np

    target = np.asarray(target_array, dtype=bool)
    simulated = np.asarray(simulated_array, dtype=bool)
    if background_array is None:
        rgb = np.full((*target.shape, 3), 248, dtype=np.uint8)
    else:
        rgb = _roi_grayscale_rgb(_resize_array_nearest(background_array, target.shape))
        rgb = np.clip(rgb.astype(np.float64) * 0.76 + 38.0, 0, 255).astype(np.uint8)

    target_outline = _dilate_mask(_mask_outline(target), iterations=1)
    simulated_outline = _dilate_mask(_mask_outline(simulated), iterations=1)
    both = target_outline & simulated_outline
    rgb[target_outline] = (224, 132, 58)
    rgb[simulated_outline] = (56, 116, 196)
    rgb[both] = (62, 153, 101)
    return np.ascontiguousarray(rgb)


class _StlPreviewWorkerSignals(QObject):
    finished = Signal(str, object, int, float)
    failed = Signal(str, str)


class _StlPreviewWorker(QRunnable):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        self.signals = _StlPreviewWorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            from capp.workbench.preview import prepare_stl_preview_mesh

            started = perf_counter()
            mesh, original_cells = prepare_stl_preview_mesh(self.path)
            elapsed = perf_counter() - started
        except Exception as exc:
            self.signals.failed.emit(self.path, str(exc))
            return

        self.signals.finished.emit(self.path, mesh, original_cells, elapsed)


class _SupportOverlayPreviewWorkerSignals(QObject):
    finished = Signal(str, object, int, str, object, int, float)
    failed = Signal(str, str, str)


class _SupportOverlayPreviewWorker(QRunnable):
    def __init__(self, part_path: str, support_path: str) -> None:
        super().__init__()
        self.part_path = part_path
        self.support_path = support_path
        self.signals = _SupportOverlayPreviewWorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            from capp.workbench.preview import prepare_stl_preview_mesh

            started = perf_counter()
            part_mesh, part_cells = prepare_stl_preview_mesh(self.part_path)
            support_mesh, support_cells = prepare_stl_preview_mesh(self.support_path)
            elapsed = perf_counter() - started
        except Exception as exc:
            self.signals.failed.emit(self.part_path, self.support_path, str(exc))
            return

        self.signals.finished.emit(
            self.part_path,
            part_mesh,
            part_cells,
            self.support_path,
            support_mesh,
            support_cells,
            elapsed,
        )


class _SimulationWorkerSignals(QObject):
    finished = Signal(object, object)
    failed = Signal(object, str)
    progress = Signal(int, str)


class _SimulationWorker(QRunnable):
    def __init__(self, config, grid) -> None:
        super().__init__()
        self.config = config
        self.grid = grid
        self.signals = _SimulationWorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            from capp.simulation.runner import run_simulation_grid

            self.signals.progress.emit(0, "Preparing virtual printing")

            def progress(percent: int, message: str) -> None:
                self.signals.progress.emit(1 + int(percent * 0.94), message)

            result = run_simulation_grid(self.grid, self.config, progress_callback=progress)
        except Exception as exc:
            self.signals.failed.emit(self.config, str(exc))
            return

        self.signals.progress.emit(100, "Virtual printing complete")
        self.signals.finished.emit(self.config, result)


class _SaveOutputsWorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(object, str)
    progress = Signal(int, str)


class _SaveOutputsWorker(QRunnable):
    def __init__(self, output_dir, result) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.result = result
        self.signals = _SaveOutputsWorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            from capp.simulation.runner import save_default_outputs

            def progress(percent: int, message: str) -> None:
                self.signals.progress.emit(percent, message)

            save_default_outputs(
                self.output_dir,
                self.result,
                progress_callback=progress,
            )
        except Exception as exc:
            self.signals.failed.emit(self.output_dir, str(exc))
            return

        self.signals.finished.emit(self.output_dir)


class _VoxelizationWorkerSignals(QObject):
    finished = Signal(object, object, object)
    failed = Signal(object, str)
    progress = Signal(int, str)


class _VoxelizationWorker(QRunnable):
    def __init__(self, config, generated_support_grid=None) -> None:
        super().__init__()
        self.config = config
        self.generated_support_grid = generated_support_grid
        self.signals = _VoxelizationWorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            from capp.geometry.voxelizer import (
                voxelize_part_and_support,
                voxelize_part_with_support_grid,
            )

            def progress(percent: int, message: str) -> None:
                self.signals.progress.emit(percent, message)

            if self.generated_support_grid is not None:
                grid = voxelize_part_with_support_grid(
                    self.config.geometry_path,
                    self.generated_support_grid,
                    self.config.voxel_spacing,
                    progress_callback=progress,
                )
            else:
                grid = voxelize_part_and_support(
                    self.config.geometry_path,
                    getattr(self.config, "support_geometry_path", None),
                    self.config.voxel_spacing,
                    getattr(self.config, "support_type", "Volume support"),
                    support_generation=getattr(self.config, "support_generation", None),
                    progress_callback=progress,
                )
        except Exception as exc:
            self.signals.failed.emit(self.config, str(exc))
            return

        self.signals.progress.emit(100, "Voxelization complete")
        self.signals.finished.emit(self.config, grid, grid.data)


class _GeneratedSupportWorkerSignals(QObject):
    finished = Signal(object, str, object)
    failed = Signal(object, str)
    progress = Signal(int, str)


class _GeneratedSupportWorker(QRunnable):
    def __init__(
        self,
        request_signature: tuple[object, ...],
        geometry_path: str,
        spacing: float,
        support_generation,
        output_path: str,
    ) -> None:
        super().__init__()
        self.request_signature = request_signature
        self.geometry_path = geometry_path
        self.spacing = spacing
        self.support_generation = support_generation
        self.output_path = output_path
        self.signals = _GeneratedSupportWorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            from capp.geometry.voxelizer import generate_overhang_support_grid, voxelize_mesh
            from capp.io.exports import write_binary_stl, write_surface_stl

            def part_progress(percent: int, message: str) -> None:
                self.signals.progress.emit(
                    int(percent * 0.42),
                    f"Part: {message}",
                )

            def support_progress(percent: int, message: str) -> None:
                self.signals.progress.emit(
                    44 + int(percent * 0.42),
                    f"Support: {message}",
                )

            part_grid = voxelize_mesh(
                self.geometry_path,
                self.spacing,
                progress_callback=part_progress,
            )
            support_grid = generate_overhang_support_grid(
                self.geometry_path,
                part_grid,
                self.support_generation,
                progress_callback=support_progress,
            )
            self.signals.progress.emit(90, "Preparing support preview mesh")
            build_plate_z = self.support_generation.build_plate_z
            if self.support_generation.support_type == "X surface support":
                write_surface_stl(
                    self.output_path,
                    support_grid.data,
                    support_grid.spacing,
                    support_grid.origin,
                    bottom_z=build_plate_z,
                )
            else:
                write_binary_stl(
                    self.output_path,
                    support_grid.data,
                    support_grid.spacing,
                    support_grid.origin,
                    clip_min_z=build_plate_z,
                    voxel_bounds=True,
                )
        except Exception as exc:
            self.signals.failed.emit(self.request_signature, str(exc))
            return

        self.signals.progress.emit(100, "Support preview ready")
        self.signals.finished.emit(self.request_signature, self.output_path, support_grid)


class _GeometryDeviationWorkerSignals(QObject):
    finished = Signal(object, int)
    failed = Signal(str, int)
    progress = Signal(int, str)


class _GeometryDeviationWorker(QRunnable):
    def __init__(
        self,
        stl_path: str,
        volume,
        spacing: float,
        origin: tuple[float, float, float],
        label: str,
        result_revision: int,
    ) -> None:
        super().__init__()
        self.stl_path = stl_path
        self.volume = volume
        self.spacing = spacing
        self.origin = origin
        self.label = label
        self.result_revision = result_revision
        self.signals = _GeometryDeviationWorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            from capp.workbench.preview import (
                build_geometry_deviation_preview,
                pack_geometry_deviation_preview,
            )

            self.signals.progress.emit(0, "Preparing geometry deviation")

            def progress(percent: int, message: str) -> None:
                self.signals.progress.emit(percent, message)

            preview = pack_geometry_deviation_preview(
                build_geometry_deviation_preview(
                    self.stl_path,
                    self.volume,
                    spacing=self.spacing,
                    origin=self.origin,
                    label=self.label,
                    progress_callback=progress,
                )
            )
        except Exception as exc:
            self.signals.failed.emit(str(exc), self.result_revision)
            return

        self.signals.progress.emit(100, "Geometry deviation ready")
        self.signals.finished.emit(preview, self.result_revision)


class _ModelCalibrationWorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, str)


class _ModelCalibrationWorker(QRunnable):
    def __init__(
        self,
        geometry_path: str,
        sample_dir: str,
        voxel_spacing: float,
        output_dir: str,
        max_evaluations: int,
        backend: str,
        optimizer: str,
        parallel_samples: int,
        sample_names: set[str] | None,
    ) -> None:
        super().__init__()
        self.geometry_path = geometry_path
        self.sample_dir = sample_dir
        self.voxel_spacing = voxel_spacing
        self.output_dir = output_dir
        self.max_evaluations = max_evaluations
        self.backend = backend
        self.optimizer = optimizer
        self.parallel_samples = parallel_samples
        self.sample_names = sample_names
        self.signals = _ModelCalibrationWorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.signals.progress.emit(0, "Starting Model Calibration worker")
            from capp.calibration.model_calibration import (
                ModelCalibrationOptions,
                run_model_calibration_from_paths,
            )
            from capp.domain import SolverBackend

            def progress(percent: int, message: str) -> None:
                self.signals.progress.emit(percent, message)

            result = run_model_calibration_from_paths(
                geometry_path=self.geometry_path,
                sample_dir=self.sample_dir,
                voxel_spacing=self.voxel_spacing,
                output_dir=None,
                options=ModelCalibrationOptions(
                    max_evaluations=self.max_evaluations,
                    backend=SolverBackend(self.backend),
                    max_workers=self.parallel_samples,
                    optimizer=self.optimizer,
                ),
                sample_names=self.sample_names,
                progress_callback=progress,
            )
        except Exception as exc:
            self.signals.failed.emit(str(exc))
            return

        self.signals.finished.emit(result)


class _SaveModelCalibrationWorkerSignals(QObject):
    finished = Signal(str, float)
    failed = Signal(str)
    progress = Signal(int, str)


class _SaveModelCalibrationWorker(QRunnable):
    def __init__(self, output_dir: str, result, geometry_path: str, run_configuration) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.result = result
        self.geometry_path = geometry_path
        self.run_configuration = run_configuration
        self.signals = _SaveModelCalibrationWorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            from time import perf_counter

            from capp.calibration.model_calibration import save_model_calibration_outputs

            def progress(percent: int, message: str) -> None:
                self.signals.progress.emit(percent, message)

            started = perf_counter()
            save_model_calibration_outputs(
                self.output_dir,
                self.result,
                calibration_geometry_path=self.geometry_path,
                run_configuration=self.run_configuration,
                progress_callback=progress,
            )
            elapsed = perf_counter() - started
        except Exception as exc:
            self.signals.failed.emit(str(exc))
            return

        self.signals.finished.emit(self.output_dir, elapsed)


class _ResearchArtifactExportWorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, str)


class _ResearchArtifactExportWorker(QRunnable):
    def __init__(self, output_dir: str, result, geometry_path: str) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.result = result
        self.geometry_path = geometry_path
        self.signals = _ResearchArtifactExportWorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            from capp.calibration.model_calibration import (
                export_model_calibration_research_artifacts,
            )

            def progress(percent: int, message: str) -> None:
                self.signals.progress.emit(percent, message)

            output_path = export_model_calibration_research_artifacts(
                self.output_dir,
                self.result,
                calibration_geometry_path=self.geometry_path,
                progress_callback=progress,
            )
        except Exception as exc:
            self.signals.failed.emit(str(exc))
            return

        self.signals.finished.emit(output_path)


class _MachineMapWorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, str)


class _MachineMapWorker(QRunnable):
    def __init__(
        self,
        weights_csv: str,
        coordinates_xlsx: str,
        resolution: int,
        preset_name: str,
        voxel_spacing: float,
        calibration_result=None,
    ) -> None:
        super().__init__()
        self.weights_csv = weights_csv
        self.coordinates_xlsx = coordinates_xlsx
        self.resolution = resolution
        self.preset_name = preset_name
        self.voxel_spacing = voxel_spacing
        self.calibration_result = calibration_result
        self.signals = _MachineMapWorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            from capp.machine_map import (
                build_machine_parameter_map,
                build_machine_parameter_map_from_files,
                machine_parameter_rows_from_calibration_result,
                read_sample_coordinates_xlsx,
            )

            def progress(percent: int, message: str) -> None:
                self.signals.progress.emit(percent, message)

            if self.calibration_result is not None:
                progress(0, "Reading in-memory calibration weights")
                rows = machine_parameter_rows_from_calibration_result(self.calibration_result)
                progress(15, "Reading sample coordinates")
                coordinates = read_sample_coordinates_xlsx(self.coordinates_xlsx)
                result = build_machine_parameter_map(
                    parameters=rows,
                    coordinates=coordinates,
                    resolution=self.resolution,
                    preset_name=self.preset_name,
                    voxel_spacing=self.voxel_spacing,
                    coordinates_xlsx=self.coordinates_xlsx,
                    progress_callback=progress,
                )
            else:
                result = build_machine_parameter_map_from_files(
                    weights_csv=self.weights_csv,
                    coordinates_xlsx=self.coordinates_xlsx,
                    resolution=self.resolution,
                    preset_name=self.preset_name,
                    voxel_spacing=self.voxel_spacing,
                    progress_callback=progress,
                )
        except Exception as exc:
            self.signals.failed.emit(str(exc))
            return

        self.signals.finished.emit(result)


class _SaveMachineMapWorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, str)


class _SaveMachineMapWorker(QRunnable):
    def __init__(
        self,
        output_dir: str,
        preset_folder: str,
        result,
        run_configuration,
    ) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.preset_folder = preset_folder
        self.result = result
        self.run_configuration = run_configuration
        self.signals = _SaveMachineMapWorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            from capp.machine_map import save_machine_parameter_map_outputs

            self.signals.progress.emit(0, "Writing machine parameter map")
            saved = save_machine_parameter_map_outputs(
                output_dir=self.output_dir,
                preset_folder=self.preset_folder,
                model=self.result.model,
                grid=self.result.grid,
                parameters=self.result.parameters,
                coordinates=self.result.coordinates,
                resolution=self.result.resolution,
                preset_name=self.result.preset_name,
                weights_csv=self.result.weights_csv,
                coordinates_xlsx=self.result.coordinates_xlsx,
                elapsed_seconds=self.result.elapsed_seconds,
                voxel_spacing=self.result.voxel_spacing,
                run_configuration=self.run_configuration,
            )
            self.signals.progress.emit(100, f"Machine parameter map saved: {saved.map_npz.name}")
        except Exception as exc:
            self.signals.failed.emit(str(exc))
            return

        self.signals.finished.emit(saved)


class WorkbenchMainWindow:
    def __init__(self) -> None:
        from PySide6.QtCore import QSettings, QSize, Qt, QThreadPool
        from PySide6.QtWidgets import (
            QFileDialog,
            QLabel,
            QListWidget,
            QMainWindow,
            QMessageBox,
            QProgressBar,
            QSplitter,
            QStackedWidget,
            QStyle,
            QVBoxLayout,
            QWidget,
        )

        from capp.workbench.branding import APP_NAME, APP_ORGANIZATION, apply_window_branding

        self._window = QMainWindow()
        self._window.setWindowTitle(APP_NAME)
        apply_window_branding(self._window)
        self._window.resize(1600, 960)
        self._current_theme = "workbench_light"
        self._settings = QSettings(APP_ORGANIZATION, APP_NAME)

        self._QFileDialog = QFileDialog
        self._QMessageBox = QMessageBox
        self._Qt = Qt
        style = self._window.style()
        self._icons = {
            "home": style.standardIcon(QStyle.StandardPixmap.SP_DesktopIcon),
            "simulation": style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay),
            "calibration": style.standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton),
            "results": style.standardIcon(QStyle.StandardPixmap.SP_FileIcon),
            "data": style.standardIcon(QStyle.StandardPixmap.SP_DriveHDIcon),
            "settings": style.standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView),
            "open": style.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon),
        }

        self._navigation = QListWidget()
        self._navigation.setObjectName("Navigation")
        self._navigation.setFixedWidth(190)
        self._navigation.setSpacing(2)
        self._navigation.setIconSize(QSize(0, 0))
        self._navigation.setUniformItemSizes(True)
        self._navigation.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._navigation.currentRowChanged.connect(self._set_page)
        self._last_result = None
        self._last_result_config = None
        self._last_voxel_grid = None
        self._last_voxel_preview_data = None
        self._last_voxel_signature = None
        self._loaded_result = None
        self._result_revision = 0
        self._busy = False
        self._cursor_busy = False
        self._simulation_worker = None
        self._voxelization_worker = None
        self._generated_support_worker = None
        self._geometry_deviation_worker = None
        self._stl_preview_worker = None
        self._support_overlay_preview_worker = None
        self._stl_preview_expected_path = None
        self._last_stl_preview = None
        self._last_support_overlay_preview = None
        self._last_generated_support_grid = None
        self._last_generated_support_path = None
        self._last_generated_support_type = None
        self._last_generated_support_signature = None
        self._last_generated_support_options = None
        self._save_outputs_worker = None
        self._model_calibration_worker = None
        self._save_model_calibration_worker = None
        self._research_artifact_worker = None
        self._machine_map_worker = None
        self._save_machine_map_worker = None
        self._last_calibration_result = None
        self._last_calibration_geometry_path = None
        self._last_machine_map_result = None
        self._last_machine_map_export_result = None
        self._calibration_comparison_data = {}
        self._calibration_overlay_source_pixmap = None
        self._machine_map_contour_data = None
        self._machine_map_contour_source_pixmap = None
        self._result_slice_source_image = None
        self._preference_fields = {}
        self._thread_pool = QThreadPool.globalInstance()
        self._log = self._make_log()

        self._stack = QStackedWidget()
        self._stack.setObjectName("FeatureStack")
        self._add_feature("simulation", "Virtual Printing", self._build_simulation_page())
        self._add_feature("results", "Result Display", self._build_results_page())
        self._add_feature("calibration", "Model Calibration", self._build_lab_page())
        self._add_feature("settings", "Preferences", self._build_preferences_page())
        self._load_preferences()

        splitter = QSplitter(self._Qt.Orientation.Horizontal)
        splitter.setObjectName("MainSplitter")
        splitter.setHandleWidth(3)
        splitter.addWidget(self._navigation)
        splitter.addWidget(self._stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        self._main_splitter = splitter

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(splitter, 1)
        log_header = QLabel("Log")
        log_header.setObjectName("LogHeader")
        central_layout.addWidget(log_header)
        central_layout.addWidget(self._log)
        self._window.setCentralWidget(central)

        self._status_progress = QProgressBar()
        self._status_progress.setFixedWidth(220)
        self._status_progress.setTextVisible(True)
        self._status_progress.setFormat("%p%")
        self._status_progress.setRange(0, 100)
        self._status_progress.setValue(0)
        self._status_progress.setVisible(False)
        self._window.statusBar().addPermanentWidget(self._status_progress)
        self._window.statusBar().showMessage("Ready")
        self._apply_style()
        self._navigation.setCurrentRow(0)
        self._sync_main_splitter_sizes()

    def show(self, maximized: bool = False) -> None:
        from PySide6.QtCore import QTimer

        from capp.workbench.branding import apply_window_branding

        if maximized:
            self._window.showMaximized()
        else:
            self._window.show()
        apply_window_branding(self._window)
        QTimer.singleShot(0, lambda: apply_window_branding(self._window))
        QTimer.singleShot(250, lambda: apply_window_branding(self._window))
        self._sync_main_splitter_sizes()

    def _sync_main_splitter_sizes(self) -> None:
        if not hasattr(self, "_main_splitter"):
            return
        nav_width = max(1, self._navigation.width() or self._navigation.minimumWidth() or 204)
        total_width = max(nav_width + 1, self._main_splitter.width() or self._window.width())
        content_width = max(1, total_width - nav_width - self._main_splitter.handleWidth())
        self._main_splitter.setSizes([nav_width, content_width])

    def _add_feature(self, _icon_key: str, title: str, widget: object) -> None:
        from PySide6.QtWidgets import QListWidgetItem

        item = QListWidgetItem(f"{self._navigation.count() + 1:02d}  {title}")
        item.setToolTip(title)
        item.setTextAlignment(
            self._Qt.AlignmentFlag.AlignVCenter | self._Qt.AlignmentFlag.AlignLeft
        )
        self._navigation.addItem(item)
        self._stack.addWidget(widget)

    def _apply_style(self, theme: str | None = None) -> None:
        if theme is not None:
            self._current_theme = theme
        theme = getattr(self, "_current_theme", "workbench_light")
        base_style = (
            """
            QMainWindow {
                background: #e6ebf1;
            }
            QWidget {
                color: #111827;
                font-size: 10px;
            }
            QLabel {
                background: transparent;
            }
            #Navigation {
                background: #182231;
                border-right: 1px solid #101722;
                color: #dbe4ee;
                padding: 8px 4px;
                outline: 0;
            }
            #Navigation::item {
                min-height: 34px;
                padding: 0 11px;
                border-radius: 3px;
                margin: 1px 2px;
                color: #cbd5e1;
                font-weight: 600;
            }
            #Navigation::item:hover {
                background: #223047;
                color: #ffffff;
            }
            #Navigation::item:selected {
                background: #f7f9fb;
                border-left: 3px solid #2f8acb;
                color: #0f172a;
            }
            #FeatureStack {
                background: #e6ebf1;
                border-left: 1px solid #c2cad4;
            }
            #Page {
                background: #e6ebf1;
            }
            QLabel#PageTitle {
                background: #dce3eb;
                border-bottom: 1px solid #aeb8c4;
                min-height: 19px;
                max-height: 19px;
                padding: 0 6px;
                font-size: 10px;
                font-weight: 700;
                color: #0f172a;
            }
            QWidget#ParameterPane, QWidget#ViewerPane, QWidget#ResultPane {
                background: #f7f9fb;
                border: 1px solid #aeb8c4;
            }
            QWidget#PreviewPane {
                background: #ffffff;
                border: 1px solid #aeb7c2;
            }
            QWidget#PreviewHeader {
                background: #dce3eb;
                border-bottom: 1px solid #aeb7c2;
                min-height: 20px;
                max-height: 20px;
            }
            QLabel#PreviewTitle {
                background: transparent;
                padding: 0 6px;
                font-weight: 600;
            }
            QComboBox#PreviewMode {
                min-width: 150px;
                margin: 1px 4px 1px 0;
                background: #ffffff;
            }
            QLabel#PreviewStatus {
                background: #ffffff;
                color: #475569;
                padding: 8px;
            }
            QLabel#PanelSubTitle {
                color: #1f2937;
                font-weight: 600;
                padding: 2px 0 0 0;
            }
            QLabel#BackendStatus {
                color: #475569;
                background: #f1f5f9;
                border: 1px solid #d1d6de;
                padding: 3px 4px;
            }
            QLabel#SliceView {
                background: #ffffff;
                border: 1px solid #aeb7c2;
            }
            QScrollArea#ParameterScroll {
                background: #f7f9fb;
                border: 0;
            }
            QWidget#ParameterViewport, QWidget#ParameterContent, QWidget#ResultContent {
                background: #f7f9fb;
                background-color: #f7f9fb;
            }
            QGroupBox {
                background: #f7f9fb;
                border: 1px solid #aeb8c4;
                border-radius: 1px;
                margin-top: 7px;
                padding: 6px 5px 5px 5px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 7px;
                padding: 0 2px;
                color: #1f2937;
                background: #f7f9fb;
                font-weight: 600;
            }
            QLineEdit, QComboBox {
                min-height: 18px;
                min-width: 0;
                background: #ffffff;
                border: 1px solid #b7c0ca;
                border-radius: 1px;
                padding: 0 4px;
            }
            QLineEdit:disabled, QComboBox:disabled {
                color: #7c8794;
                background: #edf0f3;
                border-color: #d1d6de;
            }
            QToolButton#SupportOptionsToggle {
                min-height: 19px;
                min-width: 0;
                background: #f6f8fb;
                border: 1px solid #aeb7c2;
                border-radius: 1px;
                padding: 0 7px;
                text-align: left;
            }
            QToolButton#SupportOptionsToggle:hover {
                background: #e9f1fb;
                border-color: #6b93c4;
            }
            QToolButton#SupportOptionsToggle:disabled {
                color: #9aa3ad;
                background: #edf0f3;
                border-color: #d1d6de;
            }
            QPlainTextEdit#DetailsText {
                background: #ffffff;
                background-color: #ffffff;
                color: #111827;
                border: 1px solid #b7c0ca;
                border-radius: 1px;
                padding: 3px;
                selection-background-color: #bfdbfe;
                font-family: Consolas, "Courier New", monospace;
                font-size: 9px;
            }
            QPushButton {
                min-height: 19px;
                min-width: 0;
                background: #f6f8fb;
                border: 1px solid #aeb7c2;
                border-radius: 1px;
                padding: 0 7px;
            }
            QPushButton:hover {
                background: #e9f1fb;
                border-color: #6b93c4;
            }
            QPushButton:disabled {
                color: #9aa3ad;
                background: #edf0f3;
                border-color: #d1d6de;
            }
            QPushButton#PrimaryButton {
                background: #dbeafe;
                border-color: #6b93c4;
                font-weight: 600;
            }
            QMessageBox {
                background: #f7f9fb;
            }
            QMessageBox QLabel {
                color: #111827;
                background: transparent;
            }
            QMessageBox QPushButton {
                color: #111827;
                background: #f6f8fb;
                border: 1px solid #8fa1b5;
                min-width: 64px;
                min-height: 22px;
            }
            QMessageBox QPushButton:hover {
                background: #e9f1fb;
                border-color: #5d84b8;
            }
            QLabel#LogHeader {
                background: #dce3eb;
                border-top: 1px solid #aeb7c2;
                border-bottom: 1px solid #b8c0cc;
                min-height: 16px;
                max-height: 16px;
                padding: 0 6px;
                font-weight: 600;
            }
            QPlainTextEdit#Log {
                background: #ffffff;
                background-color: #ffffff;
                border: 0;
                border-bottom: 1px solid #aeb7c2;
                font-family: Consolas, "Courier New", monospace;
                font-size: 9px;
            }
            QSplitter::handle {
                background: #c4ccd6;
            }
            QSplitter::handle:horizontal {
                width: 3px;
            }
            QSplitter::handle:vertical {
                height: 3px;
            }
            QScrollBar:vertical {
                width: 12px;
                background: #edf1f5;
                border-left: 1px solid #c1cad5;
            }
            QScrollBar:horizontal {
                height: 12px;
                background: #edf1f5;
                border-top: 1px solid #c1cad5;
            }
            QScrollBar::handle {
                background: #c8d1dc;
                border: 1px solid #aeb8c4;
                min-height: 18px;
                min-width: 18px;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                width: 0;
                height: 0;
            }
            QProgressBar {
                min-height: 14px;
                max-height: 14px;
                border: 1px solid #9faab7;
                background: #f7f9fb;
                text-align: center;
            }
            QStatusBar {
                background: #e0e5ec;
                border-top: 1px solid #b8c0cc;
                min-height: 20px;
            }
            """
        )
        theme_override = ""
        if theme == "graphite":
            theme_override = """
            QMainWindow, #Page, #FeatureStack {
                background: #c9d1dc;
            }
            #Navigation {
                background: #253140;
                border-right-color: #111827;
                color: #e5edf6;
            }
            #Navigation::item {
                color: #e5edf6;
            }
            #Navigation::item:hover {
                background: #344255;
            }
            #Navigation::item:selected {
                background: #3f5875;
                border-left-color: #62b8ff;
                color: #ffffff;
            }
            QLabel#PageTitle {
                background: #2f3d50;
                border-bottom-color: #16202c;
                color: #ffffff;
            }
            QWidget#PreviewHeader, QLabel#LogHeader {
                background: #344255;
                border-color: #16202c;
                color: #ffffff;
            }
            QWidget#ParameterPane, QWidget#ViewerPane, QWidget#ResultPane,
            QWidget#ParameterViewport, QWidget#ParameterContent, QWidget#ResultContent {
                background: #e4e9ef;
                background-color: #e4e9ef;
                border-color: #7b8797;
            }
            QGroupBox {
                background: #edf1f5;
                background-color: #edf1f5;
                border-color: #7b8797;
            }
            QGroupBox::title {
                background: #edf1f5;
                color: #0f172a;
            }
            QScrollArea#ParameterScroll {
                background: #e4e9ef;
            }
            QLineEdit, QComboBox {
                background: #f9fbfd;
                border-color: #8794a5;
            }
            QLabel#BackendStatus {
                background: #d6dde7;
                border-color: #96a1af;
                color: #253140;
            }
            QLabel#SliceView, QWidget#PreviewPane, QLabel#PreviewStatus,
            QPlainTextEdit#Log, QPlainTextEdit#DetailsText {
                background: #f8fafc;
                background-color: #f8fafc;
                border-color: #7b8797;
                color: #111827;
            }
            QPushButton {
                background: #d8e0ea;
                border-color: #728197;
            }
            QToolButton#SupportOptionsToggle {
                background: #d8e0ea;
                border-color: #728197;
            }
            QPushButton:hover {
                background: #c8d7e8;
                border-color: #4c6b91;
            }
            QToolButton#SupportOptionsToggle:hover {
                background: #c8d7e8;
                border-color: #4c6b91;
            }
            QPushButton#PrimaryButton {
                background: #9ecdf7;
                border-color: #2563a7;
                color: #0b1724;
            }
            QProgressBar {
                background: #e7ebf0;
                border-color: #7b8797;
            }
            QProgressBar::chunk {
                background: #2f8de4;
            }
            QStatusBar {
                background: #b7c2cf;
                border-top-color: #7b8797;
            }
            QSplitter::handle {
                background: #6e7b8c;
            }
            """
        elif theme == "clean_light":
            theme_override = """
            QMainWindow, #Page, #FeatureStack {
                background: #f8fafc;
            }
            #Navigation {
                background: #ffffff;
                border-right-color: #d8e1eb;
            }
            #Navigation::item {
                color: #172033;
            }
            #Navigation::item:selected {
                background: #eaf5ff;
                border-left-color: #0891d6;
                color: #0f172a;
            }
            QLabel#PageTitle, QWidget#PreviewHeader, QLabel#LogHeader {
                background: #ffffff;
                border-color: #d8e1eb;
                color: #0f172a;
            }
            QWidget#ParameterPane, QWidget#ViewerPane, QWidget#ResultPane,
            QWidget#ParameterViewport, QWidget#ParameterContent, QWidget#ResultContent,
            QGroupBox, QGroupBox::title, QScrollArea#ParameterScroll {
                background: #ffffff;
                background-color: #ffffff;
                border-color: #d8e1eb;
            }
            QLabel#BackendStatus {
                background: #f8fbff;
                border-color: #d8e1eb;
            }
            QLabel#SliceView, QWidget#PreviewPane, QLabel#PreviewStatus,
            QPlainTextEdit#Log, QPlainTextEdit#DetailsText {
                background: #ffffff;
                background-color: #ffffff;
                border-color: #d8e1eb;
            }
            QLineEdit, QComboBox {
                background: #ffffff;
                border-color: #cbd6e2;
            }
            QPushButton {
                background: #ffffff;
                border-color: #b8c7d8;
            }
            QToolButton#SupportOptionsToggle {
                background: #ffffff;
                border-color: #b8c7d8;
            }
            QPushButton:hover {
                background: #f0f8ff;
                border-color: #6ba7d8;
            }
            QToolButton#SupportOptionsToggle:hover {
                background: #f0f8ff;
                border-color: #6ba7d8;
            }
            QPushButton#PrimaryButton {
                background: #d9f0ff;
                border-color: #0f87c8;
                color: #0f172a;
            }
            QProgressBar::chunk {
                background: #0ea5e9;
            }
            QStatusBar {
                background: #ffffff;
                border-top-color: #d8e1eb;
            }
            QSplitter::handle {
                background: #dde6ef;
            }
            """
        self._window.setStyleSheet(base_style + theme_override)

    def _page_title(self, text: str):
        from PySide6.QtWidgets import QLabel

        label = QLabel(text)
        label.setObjectName("PageTitle")
        label.setAlignment(self._Qt.AlignmentFlag.AlignVCenter | self._Qt.AlignmentFlag.AlignLeft)
        return label

    def _configure_form(self, form) -> None:
        from PySide6.QtWidgets import QFormLayout

        form.setContentsMargins(4, 4, 4, 4)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(3)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setLabelAlignment(
            self._Qt.AlignmentFlag.AlignLeft | self._Qt.AlignmentFlag.AlignVCenter
        )
        form.setFormAlignment(self._Qt.AlignmentFlag.AlignTop)

    def _build_start_page(self):
        from PySide6.QtWidgets import QGridLayout, QPushButton, QVBoxLayout, QWidget

        panel = QWidget()
        panel.setObjectName("Page")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self._page_title("Workbench"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)

        entries = [
            ("simulation", "Virtual Printing", 0),
            ("results", "Result Display", 1),
            ("calibration", "Model Calibration", 2),
            ("settings", "Preferences", 3),
        ]
        for position, (icon_key, label, page_index) in enumerate(entries):
            button = QPushButton(label)
            button.setIcon(self._icons[icon_key])
            button.setMinimumHeight(44)
            button.clicked.connect(lambda _=False, index=page_index: self._select_feature(index))
            grid.addWidget(button, position // 2, position % 2)

        layout.addLayout(grid)
        layout.addStretch(1)
        return panel

    def _build_simulation_page(self):
        from PySide6.QtWidgets import (
            QComboBox,
            QFormLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPushButton,
            QScrollArea,
            QSplitter,
            QToolButton,
            QVBoxLayout,
            QWidget,
        )

        panel = QWidget()
        panel.setObjectName("Page")
        outer = QHBoxLayout(panel)
        outer.setContentsMargins(3, 3, 3, 3)
        outer.setSpacing(3)

        left_shell = QWidget()
        left_shell.setObjectName("ParameterPane")
        left_shell.setMinimumWidth(500)
        left_shell.setMaximumWidth(650)
        left_shell_layout = QVBoxLayout(left_shell)
        left_shell_layout.setContentsMargins(0, 0, 0, 0)
        left_shell_layout.setSpacing(0)
        left_shell_layout.addWidget(self._page_title("Virtual Printing"))

        left_content = QWidget()
        left_content.setObjectName("ParameterContent")
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(4)
        left_layout.setAlignment(self._Qt.AlignmentFlag.AlignTop)

        geometry_box = QGroupBox("Geometry In")
        geometry_form = QFormLayout(geometry_box)
        self._configure_form(geometry_form)
        self._part_type = QComboBox()
        self._part_type.addItems(["Part only", "Part & Support"])
        self._part_type.currentTextChanged.connect(self._on_part_type_changed)
        geometry_form.addRow("Part type", self._part_type)

        self._part_geometry = QLineEdit()
        self._part_geometry.textChanged.connect(self._invalidate_voxelization)
        self._part_geometry.editingFinished.connect(self._preview_part_geometry)
        geometry_form.addRow(
            "Part geometry",
            self._file_row(self._part_geometry, self._browse_part),
        )

        orientation_row = QHBoxLayout()
        orientation_row.setContentsMargins(0, 0, 0, 0)
        orientation_row.setSpacing(4)
        self._orientation_x = QLineEdit("0")
        self._orientation_y = QLineEdit("0")
        self._orientation_z = QLineEdit("0")
        self._orientation_fields = [
            self._orientation_x,
            self._orientation_y,
            self._orientation_z,
        ]
        for label_text, field in zip(("X", "Y", "Z"), self._orientation_fields, strict=True):
            label = QLabel(label_text)
            field.setMaximumWidth(64)
            field.textChanged.connect(self._geometry_orientation_changed)
            field.editingFinished.connect(self._preview_part_geometry)
            orientation_row.addWidget(label)
            orientation_row.addWidget(field)
        geometry_form.addRow("Orientation (deg)", orientation_row)

        self._support_options_toggle = QToolButton()
        self._support_options_toggle.setObjectName("SupportOptionsToggle")
        self._support_options_toggle.setCheckable(True)
        self._support_options_toggle.setChecked(False)
        self._support_options_toggle.setToolButtonStyle(
            self._Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._support_options_toggle.clicked.connect(self._toggle_support_options)
        geometry_form.addRow("Support setup", self._support_options_toggle)

        self._support_options_panel = QWidget()
        self._support_options_panel.setObjectName("SupportOptionsPanel")
        support_form = QFormLayout(self._support_options_panel)
        self._configure_form(support_form)

        self._support_source = QComboBox()
        self._support_source.addItems(["Generate from overhang", "External STL"])
        self._support_source.currentTextChanged.connect(self._on_support_source_changed)
        support_form.addRow("Support source", self._support_source)

        self._support_geometry = QLineEdit()
        self._support_geometry.textChanged.connect(self._invalidate_voxelization)
        support_form.addRow(
            "Support geometry",
            self._file_row(self._support_geometry, self._browse_support),
        )

        self._support_type = QComboBox()
        self._support_type.currentTextChanged.connect(self._on_support_type_changed)
        support_form.addRow("Support type", self._support_type)

        self._support_overhang_angle = QLineEdit("60")
        self._support_pitch = QLineEdit("2.0")
        self._support_thickness = QLineEdit("0.5")
        self._support_footprint_offset = QLineEdit("0.5")
        self._support_contact_depth = QLineEdit("0.5")
        self._support_build_plate_z = QLineEdit("auto")
        self._support_generation_fields = [
            self._support_overhang_angle,
            self._support_pitch,
            self._support_thickness,
            self._support_footprint_offset,
            self._support_contact_depth,
            self._support_build_plate_z,
        ]
        for field in self._support_generation_fields:
            field.textChanged.connect(self._invalidate_voxelization)
        support_form.addRow("Overhang angle (deg)", self._support_overhang_angle)
        support_form.addRow("Pattern pitch (mm)", self._support_pitch)
        support_form.addRow("Thickness (mm)", self._support_thickness)
        support_form.addRow("Footprint offset XY (mm)", self._support_footprint_offset)
        support_form.addRow("Contact overlap (mm)", self._support_contact_depth)
        support_form.addRow("Build plate Z (mm)", self._support_build_plate_z)
        support_action_row = QHBoxLayout()
        support_action_row.setContentsMargins(0, 0, 0, 0)
        support_action_row.setSpacing(4)
        generate_support = QPushButton("Generate")
        generate_support.setToolTip("Generate support preview")
        generate_support.clicked.connect(self._generate_support_preview)
        self._generate_support_button = generate_support
        clear_support = QPushButton("Clear")
        clear_support.setToolTip("Clear generated or selected support")
        clear_support.clicked.connect(self._clear_support_selection)
        self._clear_support_button = clear_support
        save_support = QPushButton("Save STL")
        save_support.setToolTip("Save generated support as STL")
        save_support.setEnabled(False)
        save_support.clicked.connect(self._save_generated_support_stl)
        self._save_support_button = save_support
        support_action_row.addWidget(generate_support)
        support_action_row.addWidget(clear_support)
        support_action_row.addWidget(save_support)
        support_form.addRow("", support_action_row)
        self._refresh_support_type_options()
        geometry_form.addRow("", self._support_options_panel)

        self._grid_spacing = QLineEdit("0.5")
        self._grid_spacing.textChanged.connect(self._invalidate_voxelization)
        estimate = QPushButton("Estimate")
        self._estimate_spacing_button = estimate
        estimate.clicked.connect(self._estimate_grid_spacing)
        spacing_row = QHBoxLayout()
        spacing_row.setContentsMargins(0, 0, 0, 0)
        spacing_row.setSpacing(4)
        spacing_row.addWidget(self._grid_spacing)
        spacing_row.addWidget(estimate)
        geometry_form.addRow("Grid spacing (mm)", spacing_row)

        left_layout.addWidget(geometry_box)

        computation_box = QGroupBox("Computation Preset")
        computation_form = QFormLayout(computation_box)
        self._configure_form(computation_form)
        self._neighborhood = QComboBox()
        self._neighborhood.addItems(["SimpleVN", "DirectionalVN", "SimpleM"])
        self._neighborhood.setCurrentText("DirectionalVN")
        self._neighborhood.currentTextChanged.connect(self._set_parameter_defaults)
        computation_form.addRow("Rule / neighborhood", self._neighborhood)

        self._coeff_x_neg = QLineEdit("0.2")
        self._coeff_x_pos = QLineEdit("0.2")
        self._coeff_y_neg = QLineEdit("0.2")
        self._coeff_y_pos = QLineEdit("0.2")
        self._coeff_current = QLineEdit("0.2")
        self._coeff_lower = QLineEdit("1")
        self._coeff_moore_l = QLineEdit("0.125")
        self._coeff_moore_cl = QLineEdit("1")

        computation_form.addRow("-x Neumann", self._coeff_x_neg)
        computation_form.addRow("+x Neumann", self._coeff_x_pos)
        computation_form.addRow("-y Neumann", self._coeff_y_neg)
        computation_form.addRow("+y Neumann", self._coeff_y_pos)
        computation_form.addRow("Equivalent coefficient", self._coeff_current)
        computation_form.addRow("-z / lower coefficient", self._coeff_lower)
        computation_form.addRow("Moore L coefficient", self._coeff_moore_l)
        computation_form.addRow("Moore CL coefficient", self._coeff_moore_cl)

        self._residual_avg = QLineEdit("1E-5")
        self._residual_max = QLineEdit("1E-4")
        self._iteration_bound = QLineEdit("400")
        self._overwrap = QLineEdit("0.1")
        self._min_bias = QLineEdit("0.05")
        self._idp = QLineEdit("0.3")
        computation_form.addRow("MAE residual", self._residual_avg)
        computation_form.addRow("MaxE residual", self._residual_max)
        computation_form.addRow("Max iteration", self._iteration_bound)
        computation_form.addRow("Overwrap criterion", self._overwrap)
        computation_form.addRow("Minimum bias", self._min_bias)
        computation_form.addRow("IDP value", self._idp)
        left_layout.addWidget(computation_box)

        process_box = QGroupBox("Process Options")
        process_form = QFormLayout(process_box)
        self._configure_form(process_form)
        self._stochastic_mode = QComboBox()
        self._stochastic_mode.addItems(["In-layer", "In-volume"])
        process_form.addRow("Stochastic process", self._stochastic_mode)

        self._machine_preset = QComboBox()
        self._machine_preset.addItem("None", None)
        self._machine_preset.currentIndexChanged.connect(self._update_machine_preset_controls)
        process_form.addRow("Machine preset", self._machine_preset)

        self._machine_map_path = QLineEdit(str(self._default_machine_map_path()))
        process_form.addRow(
            "Machine map",
            self._file_row(self._machine_map_path, self._browse_machine_map_path),
        )

        self._machine_map_coordinate_mode = QComboBox()
        self._machine_map_coordinate_mode.addItems(
            ["Full base plate", "Part center", "Explicit bounds"]
        )
        self._machine_map_coordinate_mode.currentIndexChanged.connect(
            self._update_machine_map_coordinate_fields
        )
        process_form.addRow("Map coordinates", self._machine_map_coordinate_mode)

        self._machine_map_preset_name = QLabel("-")
        self._machine_map_preset_name.setWordWrap(True)
        process_form.addRow("Map name", self._machine_map_preset_name)

        self._machine_map_x = QLineEdit("0")
        self._machine_map_y = QLineEdit("0")
        map_position_row = QHBoxLayout()
        map_position_row.setContentsMargins(0, 0, 0, 0)
        map_position_row.setSpacing(4)
        self._machine_map_center_widgets = [
            QLabel("X"),
            self._machine_map_x,
            QLabel("Y"),
            self._machine_map_y,
        ]
        map_position_row.addWidget(self._machine_map_center_widgets[0])
        map_position_row.addWidget(self._machine_map_x)
        map_position_row.addWidget(self._machine_map_center_widgets[2])
        map_position_row.addWidget(self._machine_map_y)
        process_form.addRow("Part center (mm)", map_position_row)

        self._machine_map_x_min = QLineEdit("-125")
        self._machine_map_x_max = QLineEdit("125")
        self._machine_map_y_min = QLineEdit("-125")
        self._machine_map_y_max = QLineEdit("125")
        map_bounds_row = QHBoxLayout()
        map_bounds_row.setContentsMargins(0, 0, 0, 0)
        map_bounds_row.setSpacing(4)
        self._machine_map_bounds_widgets = [
            QLabel("X min"),
            self._machine_map_x_min,
            QLabel("X max"),
            self._machine_map_x_max,
            QLabel("Y min"),
            self._machine_map_y_min,
            QLabel("Y max"),
            self._machine_map_y_max,
        ]
        for widget in self._machine_map_bounds_widgets:
            map_bounds_row.addWidget(widget)
        process_form.addRow("Map bounds (mm)", map_bounds_row)
        self._update_machine_map_coordinate_fields()
        self._machine_map_path.editingFinished.connect(self._machine_map_path_edited)
        self._machine_preset_locked_widgets = [
            self._grid_spacing,
            self._estimate_spacing_button,
            self._neighborhood,
            self._coeff_x_neg,
            self._coeff_x_pos,
            self._coeff_y_neg,
            self._coeff_y_pos,
            self._coeff_current,
            self._coeff_lower,
            self._coeff_moore_l,
            self._coeff_moore_cl,
            self._residual_avg,
            self._residual_max,
            self._iteration_bound,
            self._overwrap,
            self._min_bias,
            self._idp,
            self._stochastic_mode,
        ]
        self._refresh_machine_preset_list(select_path=self._default_machine_map_path())
        self._refresh_machine_map_name()
        self._update_machine_preset_controls()

        self._processor = QComboBox()
        self._processor.currentIndexChanged.connect(self._update_backend_status_label)
        self._processor_status = QLabel()
        self._processor_status.setWordWrap(False)
        self._processor_status.setObjectName("BackendStatus")
        refresh_devices = QPushButton("Validate")
        refresh_devices.clicked.connect(lambda: self._refresh_compute_backends(log=True))
        processor_row = QHBoxLayout()
        processor_row.setContentsMargins(0, 0, 0, 0)
        processor_row.setSpacing(4)
        processor_row.addWidget(self._processor, 1)
        processor_row.addWidget(refresh_devices)
        process_form.addRow("Processor", processor_row)
        process_form.addRow("Device status", self._processor_status)
        self._refresh_compute_backends(log=False)

        self._output_dir = QLineEdit("examples/outputs/gui_simulation")
        process_form.addRow("Output dir", self._file_row(self._output_dir, self._browse_output_dir))
        left_layout.addWidget(process_box)
        left_layout.addStretch(1)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(4, 4, 4, 4)
        action_row.setSpacing(4)
        voxelize = QPushButton("Voxelize Geometry")
        self._voxelize_button = voxelize
        voxelize.clicked.connect(self._voxelize_geometry)
        action_row.addWidget(voxelize)

        save_voxel_grid = QPushButton("Save Voxel Grid")
        save_voxel_grid.setEnabled(False)
        save_voxel_grid.clicked.connect(self._save_voxel_grid)
        self._save_voxel_grid_button = save_voxel_grid
        action_row.addWidget(save_voxel_grid)

        self._run_button = QPushButton("Run Virtual Printing")
        self._run_button.setObjectName("PrimaryButton")
        self._run_button.setIcon(self._icons["simulation"])
        self._run_button.setEnabled(False)
        self._run_button.clicked.connect(self._run_simulation)
        action_row.addWidget(self._run_button)

        self._preview_result_button = QPushButton("Preview Result")
        self._preview_result_button.setEnabled(False)
        self._preview_result_button.clicked.connect(self._preview_result)
        action_row.addWidget(self._preview_result_button)

        self._save_result_button = QPushButton("Save Outputs")
        self._save_result_button.setEnabled(False)
        self._save_result_button.clicked.connect(self._save_outputs)
        action_row.addWidget(self._save_result_button)

        scroll = QScrollArea()
        scroll.setObjectName("ParameterScroll")
        scroll.viewport().setObjectName("ParameterViewport")
        scroll.setWidgetResizable(True)
        scroll.setAlignment(self._Qt.AlignmentFlag.AlignTop)
        scroll.setHorizontalScrollBarPolicy(self._Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(left_content)
        left_shell_layout.addWidget(scroll, 1)
        left_shell_layout.addLayout(action_row)

        right = QWidget()
        right.setObjectName("ViewerPane")
        right.setMinimumWidth(640)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(3, 3, 3, 3)
        right_layout.setSpacing(3)
        from capp.workbench.preview import PreviewPane

        self._preview = PreviewPane(show_source_selector=True, show_stl_controls=True)
        self._preview.source_selector.currentTextChanged.connect(self._preview_source_changed)
        self._preview.stl_display_mode.currentTextChanged.connect(self._refresh_stl_preview_style)
        self._preview.overhang_limit.editingFinished.connect(self._refresh_stl_preview_style)
        right_layout.addWidget(self._preview.widget, 1)

        summary_box = QGroupBox("Run Summary")
        summary = QFormLayout(summary_box)
        self._configure_form(summary)
        self._shape_label = QLabel("-")
        self._spacing_label = QLabel("-")
        self._rest_label = QLabel("-")
        self._density_label = QLabel("-")
        self._outside_label = QLabel("-")
        self._elapsed_label = QLabel("-")
        self._voxel_status_label = QLabel("Required")
        summary.addRow("Shape", self._shape_label)
        summary.addRow("Spacing", self._spacing_label)
        summary.addRow("Voxel Grid", self._voxel_status_label)
        summary.addRow("Rest Volume", self._rest_label)
        summary.addRow("Probability Density", self._density_label)
        summary.addRow("Out-of-CAD Voxels", self._outside_label)
        summary.addRow("Elapsed", self._elapsed_label)
        right_layout.addWidget(summary_box)

        splitter = QSplitter(self._Qt.Orientation.Horizontal)
        splitter.addWidget(left_shell)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([540, 1210])
        outer.addWidget(splitter)

        self._on_part_type_changed(self._part_type.currentText())
        self._set_parameter_defaults(self._neighborhood.currentText())
        self._update_preview_source_controls("STL")
        self._update_preview_source_availability()
        return panel

    def _build_results_page(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QCheckBox,
            QComboBox,
            QFormLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPushButton,
            QSlider,
            QSplitter,
            QVBoxLayout,
            QWidget,
        )

        panel = QWidget()
        panel.setObjectName("Page")
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(3)

        left = QWidget()
        left.setObjectName("ResultPane")
        left.setMinimumWidth(360)
        left.setMaximumWidth(460)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(self._page_title("Result Display"))

        result_content = QWidget()
        result_content.setObjectName("ResultContent")
        result_layout = QVBoxLayout(result_content)
        result_layout.setContentsMargins(4, 4, 4, 4)
        result_layout.setSpacing(4)

        file_box = QGroupBox("Result File")
        file_layout = QVBoxLayout(file_box)
        file_layout.setContentsMargins(4, 4, 4, 4)
        self._result_npz_path = QLineEdit()
        load_row = QHBoxLayout()
        load_row.setContentsMargins(0, 0, 0, 0)
        load_row.setSpacing(4)
        load_row.addWidget(self._result_npz_path)
        browse = QPushButton("Open NPZ")
        browse.setIcon(self._icons["open"])
        browse.clicked.connect(self._browse_result_npz)
        load_row.addWidget(browse)
        file_layout.addLayout(load_row)
        result_layout.addWidget(file_box)

        controls_box = QGroupBox("View Controls")
        form = QFormLayout(controls_box)
        self._configure_form(form)
        self._result_volume_choice = QComboBox()
        self._result_volume_choice.addItems(["Binary", "Probability"])
        self._result_volume_choice.currentTextChanged.connect(self._refresh_result_views)
        form.addRow("Volume", self._result_volume_choice)

        self._result_hide_support = QCheckBox("Remove support")
        self._result_hide_support.setEnabled(False)
        self._result_hide_support.stateChanged.connect(self._result_support_visibility_changed)
        form.addRow("Support", self._result_hide_support)

        self._slice_axis = QComboBox()
        self._slice_axis.addItems(["Z", "X", "Y"])
        self._slice_axis.currentTextChanged.connect(self._update_result_slice)
        form.addRow("Slice axis", self._slice_axis)

        self._slice_slider = QSlider(Qt.Orientation.Horizontal)
        self._slice_slider.valueChanged.connect(self._update_result_slice)
        form.addRow("Slice", self._slice_slider)
        result_layout.addWidget(controls_box)

        deviation_box = QGroupBox("Geometry Deviation")
        deviation_form = QFormLayout(deviation_box)
        self._configure_form(deviation_form)
        self._deviation_stl_path = QLineEdit()
        deviation_form.addRow(
            "Original STL",
            self._file_row(self._deviation_stl_path, self._browse_deviation_stl),
        )
        deviation_button = QPushButton("Show Deviation Heatmap")
        deviation_button.setEnabled(False)
        deviation_button.clicked.connect(self._preview_geometry_deviation)
        self._deviation_button = deviation_button
        deviation_form.addRow("", deviation_button)
        self._deviation_summary = QLabel("-")
        self._deviation_summary.setWordWrap(True)
        deviation_form.addRow("Summary", self._deviation_summary)
        result_layout.addWidget(deviation_box)

        preview_button = QPushButton("Preview 3D")
        preview_button.setEnabled(False)
        preview_button.clicked.connect(self._preview_loaded_result)
        self._result_display_preview_button = preview_button
        result_layout.addWidget(preview_button)

        self._slice_label = QLabel("-")
        self._slice_label.setObjectName("SliceView")
        self._slice_label.setMinimumSize(320, 280)
        self._slice_label.setMargin(6)
        self._slice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        slice_title = QLabel("Slice View")
        slice_title.setObjectName("PanelSubTitle")
        result_layout.addWidget(slice_title)
        result_layout.addWidget(self._slice_label, 1)

        output_box = QGroupBox("Output")
        output_layout = QVBoxLayout(output_box)
        output_layout.setContentsMargins(4, 4, 4, 4)
        output_layout.setSpacing(3)
        self._output_label = QLabel("-")
        self._files_label = QLabel("-")
        output_layout.addWidget(self._output_label)
        output_layout.addWidget(self._files_label)
        save_result = QPushButton("Save Current Result")
        save_result.setEnabled(False)
        save_result.clicked.connect(self._save_outputs)
        output_layout.addWidget(save_result)
        self._save_loaded_result_button = save_result
        result_layout.addWidget(output_box)
        left_layout.addWidget(result_content, 1)

        from capp.workbench.preview import PreviewPane

        right = QWidget()
        right.setObjectName("ViewerPane")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(3, 3, 3, 3)
        right_layout.setSpacing(3)
        self._result_preview = PreviewPane()
        right_layout.addWidget(self._result_preview.widget, 1)

        splitter = QSplitter(self._Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setSizes([400, 1220])
        layout.addWidget(splitter)
        return panel

    def _build_lab_page(self):
        from PySide6.QtWidgets import (
            QComboBox,
            QFormLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPlainTextEdit,
            QProgressBar,
            QPushButton,
            QScrollArea,
            QSizePolicy,
            QSpinBox,
            QSplitter,
            QVBoxLayout,
            QWidget,
        )

        panel = QWidget()
        panel.setObjectName("Page")
        outer = QHBoxLayout(panel)
        outer.setContentsMargins(3, 3, 3, 3)
        outer.setSpacing(3)

        left_shell = QWidget()
        left_shell.setObjectName("ParameterPane")
        left_shell.setMinimumWidth(520)
        left_shell.setMaximumWidth(700)
        left_shell_layout = QVBoxLayout(left_shell)
        left_shell_layout.setContentsMargins(0, 0, 0, 0)
        left_shell_layout.setSpacing(0)
        left_shell_layout.addWidget(self._page_title("Model Calibration"))

        left_content = QWidget()
        left_content.setObjectName("ParameterContent")
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(4)
        left_layout.setAlignment(self._Qt.AlignmentFlag.AlignTop)

        input_box = QGroupBox("Data In")
        input_form = QFormLayout(input_box)
        self._configure_form(input_form)
        self._calibration_geometry = QLineEdit(str(self._default_calibration_geometry_path()))
        input_form.addRow(
            "Calibration STL",
            self._file_row(self._calibration_geometry, self._browse_calibration_geometry),
        )
        self._calibration_sample_dir = QLineEdit(str(self._default_calibration_sample_dir()))
        input_form.addRow(
            "ROI sample folder",
            self._file_row(self._calibration_sample_dir, self._browse_calibration_sample_dir),
        )
        self._calibration_spacing = QLineEdit("0.07")
        input_form.addRow("Grid spacing (mm)", self._calibration_spacing)
        self._calibration_sample_filter = QLineEdit()
        input_form.addRow("Sample filter", self._calibration_sample_filter)
        self._calibration_output_dir = QLineEdit(str(_default_machine_preset_library_root()))
        self._calibration_output_dir.editingFinished.connect(
            lambda: self._refresh_machine_preset_list(preserve_current=False)
        )
        input_form.addRow(
            "Preset library",
            self._file_row(self._calibration_output_dir, self._browse_calibration_output_dir),
        )
        left_layout.addWidget(input_box)

        solver_box = QGroupBox("Calibration Solver")
        solver_form = QFormLayout(solver_box)
        self._configure_form(solver_form)
        self._calibration_optimizer = QComboBox()
        self._calibration_optimizer.addItem("Global Evolution", "global_evolution")
        self._calibration_optimizer.addItem("Adaptive Sobol", "adaptive_sobol")
        self._calibration_optimizer.addItem("Sobol", "sobol")
        self._calibration_optimizer.addItem("Latin Hypercube", "latin_hypercube")
        solver_form.addRow("Search method", self._calibration_optimizer)
        self._calibration_max_evaluations = QSpinBox()
        self._calibration_max_evaluations.setRange(1, 10000)
        self._calibration_max_evaluations.setValue(40)
        solver_form.addRow("Evaluations", self._calibration_max_evaluations)
        self._calibration_parallel_samples = QSpinBox()
        self._calibration_parallel_samples.setRange(1, 128)
        self._calibration_parallel_samples.setValue(self._recommended_parallel_samples())
        solver_form.addRow("Parallel samples", self._calibration_parallel_samples)
        self._calibration_processor = QComboBox()
        self._calibration_processor.currentIndexChanged.connect(
            self._update_calibration_backend_status_label
        )
        validate = QPushButton("Validate")
        validate.clicked.connect(lambda: self._refresh_calibration_backends(log=True))
        processor_row = QHBoxLayout()
        processor_row.setContentsMargins(0, 0, 0, 0)
        processor_row.setSpacing(4)
        processor_row.addWidget(self._calibration_processor, 1)
        processor_row.addWidget(validate)
        solver_form.addRow("Solver", processor_row)
        self._calibration_processor_status = QLabel()
        self._calibration_processor_status.setObjectName("BackendStatus")
        self._calibration_processor_status.setWordWrap(False)
        solver_form.addRow("Device status", self._calibration_processor_status)
        left_layout.addWidget(solver_box)

        progress_box = QGroupBox("Run Summary")
        progress_form = QFormLayout(progress_box)
        self._configure_form(progress_form)
        self._calibration_progress_bar = QProgressBar()
        self._calibration_progress_bar.setRange(0, 100)
        self._calibration_progress_bar.setValue(0)
        self._calibration_progress_bar.setFormat("0%")
        progress_form.addRow("Progress", self._calibration_progress_bar)
        self._calibration_progress_message = QLabel("Ready")
        self._calibration_progress_message.setWordWrap(True)
        progress_form.addRow("Status", self._calibration_progress_message)
        self._calibration_samples_label = QLabel("-")
        self._calibration_loss_label = QLabel("-")
        self._calibration_elapsed_label = QLabel("-")
        self._calibration_csv_label = QLabel("-")
        self._calibration_csv_label.setWordWrap(True)
        progress_form.addRow("Samples", self._calibration_samples_label)
        progress_form.addRow("Average loss", self._calibration_loss_label)
        progress_form.addRow("Elapsed", self._calibration_elapsed_label)
        progress_form.addRow("Weights CSV", self._calibration_csv_label)
        self._calibration_details = QPlainTextEdit()
        self._calibration_details.setObjectName("DetailsText")
        self._calibration_details.setReadOnly(True)
        self._calibration_details.setMaximumHeight(130)
        self._calibration_details.setStyleSheet(
            """
            QPlainTextEdit#DetailsText {
                background: #ffffff;
                background-color: #ffffff;
                color: #111827;
                border: 1px solid #b7c0ca;
                border-radius: 1px;
                padding: 3px;
                selection-background-color: #bfdbfe;
                font-family: Consolas, "Courier New", monospace;
                font-size: 9px;
            }
            QPlainTextEdit#DetailsText QWidget {
                background: #ffffff;
                background-color: #ffffff;
            }
            """
        )
        self._calibration_details.setPlainText("-")
        progress_form.addRow("Details", self._calibration_details)
        left_layout.addWidget(progress_box)

        map_box = QGroupBox("Machine Parameter Map")
        map_form = QFormLayout(map_box)
        self._configure_form(map_form)
        self._machine_map_name = QLineEdit("Machine Map")
        map_form.addRow("Preset name", self._machine_map_name)
        self._machine_map_coordinates = QLineEdit(str(self._default_machine_coordinate_path()))
        map_form.addRow(
            "SP coordinates",
            self._file_row(self._machine_map_coordinates, self._browse_machine_map_coordinates),
        )
        self._machine_map_resolution = QSpinBox()
        self._machine_map_resolution.setRange(8, 2000)
        self._machine_map_resolution.setValue(200)
        map_form.addRow("Contour grid", self._machine_map_resolution)
        self._machine_map_status = QLabel("Waiting for Model Calibration result.")
        self._machine_map_status.setWordWrap(True)
        map_form.addRow("Status", self._machine_map_status)
        self._machine_map_outputs_label = QLabel("-")
        self._machine_map_outputs_label.setWordWrap(True)
        map_form.addRow("Preset/Folder", self._machine_map_outputs_label)
        left_layout.addWidget(map_box)
        left_layout.addStretch(1)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(4, 4, 4, 4)
        action_row.setSpacing(4)
        self._run_calibration_button = QPushButton("Run Model Calibration")
        self._run_calibration_button.setObjectName("PrimaryButton")
        self._run_calibration_button.clicked.connect(self._run_model_calibration)
        action_row.addWidget(self._run_calibration_button)
        self._save_calibration_button = QPushButton("Save Calibration")
        self._save_calibration_button.setEnabled(False)
        self._save_calibration_button.clicked.connect(self._save_model_calibration)
        action_row.addWidget(self._save_calibration_button)
        self._generate_machine_map_button = QPushButton("Generate Machine Map")
        self._generate_machine_map_button.setEnabled(False)
        self._generate_machine_map_button.clicked.connect(self._generate_machine_map)
        action_row.addWidget(self._generate_machine_map_button)
        self._save_machine_map_button = QPushButton("Save Machine Map")
        self._save_machine_map_button.setEnabled(False)
        self._save_machine_map_button.clicked.connect(self._save_machine_map)
        action_row.addWidget(self._save_machine_map_button)
        self._export_research_artifacts_button = QPushButton("Export Research Artifacts")
        self._export_research_artifacts_button.setEnabled(False)
        self._export_research_artifacts_button.clicked.connect(
            self._export_calibration_research_artifacts
        )
        action_row.addWidget(self._export_research_artifacts_button)

        scroll = QScrollArea()
        scroll.setObjectName("ParameterScroll")
        scroll.viewport().setObjectName("ParameterViewport")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(self._Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(left_content)
        left_shell_layout.addWidget(scroll, 1)
        left_shell_layout.addLayout(action_row)

        right = QWidget()
        right.setObjectName("ViewerPane")
        right.setMinimumWidth(0)
        right.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(3, 3, 3, 3)
        right_layout.setSpacing(3)
        right_layout.addWidget(self._page_title("Calibration Review"))

        comparison_box = QGroupBox("ROI Comparison")
        comparison_box.setMinimumWidth(0)
        comparison_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        comparison_layout = QVBoxLayout(comparison_box)
        comparison_layout.setContentsMargins(4, 4, 4, 4)
        comparison_layout.setSpacing(4)
        comparison_controls = QHBoxLayout()
        comparison_controls.setContentsMargins(0, 0, 0, 0)
        comparison_controls.setSpacing(4)
        self._calibration_comparison_sample = QComboBox()
        self._calibration_comparison_sample.setMinimumContentsLength(12)
        self._calibration_comparison_sample.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._calibration_comparison_sample.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self._calibration_comparison_sample.currentTextChanged.connect(
            self._refresh_calibration_overlay
        )
        self._calibration_comparison_axis = QComboBox()
        self._calibration_comparison_axis.addItems(["X ROI", "Y ROI"])
        self._calibration_comparison_axis.setMinimumContentsLength(5)
        self._calibration_comparison_axis.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._calibration_comparison_axis.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        self._calibration_comparison_axis.currentTextChanged.connect(
            self._refresh_calibration_overlay
        )
        self._calibration_comparison_mode = QComboBox()
        self._calibration_comparison_mode.addItems(
            ["Overlay", "Outline", "Original + Outline", "Target Mask", "Simulated Mask"]
        )
        self._calibration_comparison_mode.setMinimumContentsLength(14)
        self._calibration_comparison_mode.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._calibration_comparison_mode.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        self._calibration_comparison_mode.currentTextChanged.connect(
            self._refresh_calibration_overlay
        )
        comparison_controls.addWidget(self._calibration_comparison_sample, 2)
        comparison_controls.addWidget(self._calibration_comparison_axis)
        comparison_controls.addWidget(self._calibration_comparison_mode)
        comparison_layout.addLayout(comparison_controls)
        self._calibration_overlay_label = QLabel("Run Model Calibration to compare ROI images.")
        self._calibration_overlay_label.setObjectName("SliceView")
        self._calibration_overlay_label.setAlignment(self._Qt.AlignmentFlag.AlignCenter)
        self._calibration_overlay_label.setMinimumSize(1, 280)
        self._calibration_overlay_label.setMinimumHeight(280)
        self._calibration_overlay_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Expanding,
        )
        comparison_layout.addWidget(self._calibration_overlay_label, 1)
        right_layout.addWidget(comparison_box, 1)

        contour_box = QGroupBox("Machine Map Contour")
        contour_box.setMinimumWidth(0)
        contour_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        contour_layout = QVBoxLayout(contour_box)
        contour_layout.setContentsMargins(4, 4, 4, 4)
        contour_layout.setSpacing(4)
        self._machine_map_contour_variable = QComboBox()
        self._machine_map_contour_variable.addItems(["NX", "PX", "NY", "PY", "EPS", "IDP"])
        self._machine_map_contour_variable.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._machine_map_contour_variable.currentTextChanged.connect(
            self._refresh_machine_map_contour
        )
        contour_layout.addWidget(self._machine_map_contour_variable)
        self._machine_map_contour_label = QLabel("Generate or load a machine parameter map.")
        self._machine_map_contour_label.setObjectName("SliceView")
        self._machine_map_contour_label.setAlignment(self._Qt.AlignmentFlag.AlignCenter)
        self._machine_map_contour_label.setMinimumSize(1, 260)
        self._machine_map_contour_label.setMinimumHeight(260)
        self._machine_map_contour_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Expanding,
        )
        contour_layout.addWidget(self._machine_map_contour_label, 1)
        right_layout.addWidget(contour_box, 1)

        splitter = QSplitter(self._Qt.Orientation.Horizontal)
        splitter.addWidget(left_shell)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([620, 980])
        outer.addWidget(splitter, 1)

        self._refresh_calibration_backends(log=False)
        return panel

    def _build_placeholder_page(self, name: str):
        from PySide6.QtWidgets import QVBoxLayout, QWidget

        panel = QWidget()
        panel.setObjectName("Page")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        label = self._page_title(name)
        label.setAlignment(self._Qt.AlignmentFlag.AlignTop | self._Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(label)
        layout.addStretch(1)
        return panel

    def _build_preferences_page(self):
        from PySide6.QtWidgets import (
            QComboBox,
            QFormLayout,
            QGroupBox,
            QHBoxLayout,
            QLineEdit,
            QPushButton,
            QScrollArea,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )

        panel = QWidget()
        panel.setObjectName("Page")
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(3, 3, 3, 3)
        outer.setSpacing(3)

        shell = QWidget()
        shell.setObjectName("ParameterPane")
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        shell_layout.addWidget(self._page_title("Preferences"))

        content = QWidget()
        content.setObjectName("ParameterContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(4)
        content_layout.setAlignment(self._Qt.AlignmentFlag.AlignTop)

        defaults = self._preference_defaults()

        interface_box = QGroupBox("Interface")
        interface_form = QFormLayout(interface_box)
        self._configure_form(interface_form)
        theme = QComboBox()
        theme.addItem("Workbench Light", "workbench_light")
        theme.addItem("Graphite Workbench", "graphite")
        theme.addItem("Clean Light", "clean_light")
        self._preference_fields["interface_theme"] = theme
        interface_form.addRow("Theme", theme)
        content_layout.addWidget(interface_box)

        path_box = QGroupBox("Default Paths")
        path_form = QFormLayout(path_box)
        self._configure_form(path_form)

        simulation_output = QLineEdit(defaults["simulation_output_dir"])
        self._preference_fields["simulation_output_dir"] = simulation_output
        path_form.addRow(
            "Simulation output dir",
            self._file_row(
                simulation_output,
                lambda: self._browse_preference_dir(
                    simulation_output, "Select default simulation output directory"
                ),
            ),
        )

        preset_library = QLineEdit(defaults["calibration_output_dir"])
        self._preference_fields["calibration_output_dir"] = preset_library
        path_form.addRow(
            "Preset library",
            self._file_row(
                preset_library,
                lambda: self._browse_preference_dir(
                    preset_library, "Select default machine preset library"
                ),
            ),
        )

        calibration_geometry = QLineEdit(defaults["calibration_geometry"])
        self._preference_fields["calibration_geometry"] = calibration_geometry
        path_form.addRow(
            "Calibration STL",
            self._file_row(
                calibration_geometry,
                lambda: self._browse_preference_file(
                    calibration_geometry,
                    "Select default calibration STL",
                    "STL files (*.stl);;All files (*.*)",
                ),
            ),
        )

        calibration_sample_dir = QLineEdit(defaults["calibration_sample_dir"])
        self._preference_fields["calibration_sample_dir"] = calibration_sample_dir
        path_form.addRow(
            "ROI sample folder",
            self._file_row(
                calibration_sample_dir,
                lambda: self._browse_preference_dir(
                    calibration_sample_dir, "Select default ROI sample folder"
                ),
            ),
        )

        machine_coordinate_workbook = QLineEdit(defaults["machine_coordinate_workbook"])
        self._preference_fields["machine_coordinate_workbook"] = machine_coordinate_workbook
        path_form.addRow(
            "SP coordinates",
            self._file_row(
                machine_coordinate_workbook,
                lambda: self._browse_preference_file(
                    machine_coordinate_workbook,
                    "Select default SP coordinate workbook",
                    "Excel workbooks (*.xlsx);;All files (*.*)",
                ),
            ),
        )

        machine_map_path = QLineEdit(defaults["machine_map_path"])
        self._preference_fields["machine_map_path"] = machine_map_path
        path_form.addRow(
            "Machine map",
            self._file_row(
                machine_map_path,
                lambda: self._browse_preference_file(
                    machine_map_path,
                    "Select default machine parameter map",
                    "Machine parameter maps (*.npz);;All files (*.*)",
                ),
            ),
        )
        content_layout.addWidget(path_box)

        calibration_box = QGroupBox("Model Calibration Defaults")
        calibration_form = QFormLayout(calibration_box)
        self._configure_form(calibration_form)

        optimizer = QComboBox()
        optimizer.addItem("Global Evolution", "global_evolution")
        optimizer.addItem("Adaptive Sobol", "adaptive_sobol")
        optimizer.addItem("Sobol", "sobol")
        optimizer.addItem("Latin Hypercube", "latin_hypercube")
        self._preference_fields["calibration_optimizer"] = optimizer
        calibration_form.addRow("Search method", optimizer)

        spacing = QLineEdit(defaults["calibration_spacing"])
        self._preference_fields["calibration_spacing"] = spacing
        calibration_form.addRow("Grid spacing (mm)", spacing)

        evaluations = QSpinBox()
        evaluations.setRange(1, 10000)
        evaluations.setValue(int(defaults["calibration_evaluations"]))
        self._preference_fields["calibration_evaluations"] = evaluations
        calibration_form.addRow("Evaluations", evaluations)

        parallel = QSpinBox()
        parallel.setRange(1, 128)
        parallel.setValue(int(defaults["calibration_parallel_samples"]))
        self._preference_fields["calibration_parallel_samples"] = parallel
        calibration_form.addRow("Parallel samples", parallel)
        content_layout.addWidget(calibration_box)

        map_box = QGroupBox("Machine Map Defaults")
        map_form = QFormLayout(map_box)
        self._configure_form(map_form)
        machine_map_name = QLineEdit(defaults["machine_map_name"])
        self._preference_fields["machine_map_name"] = machine_map_name
        map_form.addRow("Preset name", machine_map_name)
        content_layout.addWidget(map_box)
        content_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setObjectName("ParameterScroll")
        scroll.viewport().setObjectName("ParameterViewport")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(self._Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        shell_layout.addWidget(scroll, 1)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(4, 4, 4, 4)
        action_row.setSpacing(4)
        apply_button = QPushButton("Apply to Current Forms")
        apply_button.clicked.connect(self._apply_preferences_from_page)
        action_row.addWidget(apply_button)
        save_button = QPushButton("Save Preferences")
        save_button.setObjectName("PrimaryButton")
        save_button.clicked.connect(self._save_preferences_from_page)
        action_row.addWidget(save_button)
        reset_button = QPushButton("Reset Defaults")
        reset_button.clicked.connect(self._reset_preferences)
        action_row.addWidget(reset_button)
        shell_layout.addLayout(action_row)

        outer.addWidget(shell, 1)
        return panel

    def _file_row(self, line_edit, callback):
        from PySide6.QtWidgets import QHBoxLayout, QPushButton

        button = QPushButton("Open")
        button.setIcon(self._icons["open"])
        button.clicked.connect(callback)
        line_edit._browse_button = button

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(line_edit)
        row.addWidget(button)
        return row

    def _preference_defaults(self) -> dict[str, str]:
        return {
            "interface_theme": "workbench_light",
            "simulation_output_dir": "examples/outputs/gui_simulation",
            "calibration_output_dir": str(_default_machine_preset_library_root()),
            "calibration_geometry": str(self._default_calibration_geometry_path()),
            "calibration_sample_dir": str(self._default_calibration_sample_dir()),
            "calibration_spacing": "0.07",
            "calibration_evaluations": "40",
            "calibration_parallel_samples": str(self._recommended_parallel_samples()),
            "calibration_optimizer": "global_evolution",
            "machine_coordinate_workbook": str(self._default_machine_coordinate_path()),
            "machine_map_path": str(self._default_machine_map_path()),
            "machine_map_name": "Machine Map",
        }

    def _preference_key(self, key: str) -> str:
        return f"preferences/{key}"

    def _legacy_default_machine_map_path(self) -> Path:
        return (
            _legacy_model_calibration_root()
            / "machine_presets"
            / "Machine_Map"
            / "machine_parameter_map.npz"
        )

    def _intermediate_default_machine_map_path(self) -> Path:
        return (
            _intermediate_model_calibration_root()
            / "machine_presets"
            / "Machine_Map"
            / "machine_parameter_map.npz"
        )

    def _migrate_preference_value(self, key: str, value: str, default: str) -> str:
        normalized = Path(value).as_posix().rstrip("/")

        def matches_path(path: Path) -> bool:
            legacy = path.as_posix().rstrip("/")
            return normalized == legacy or normalized.endswith(f"/{legacy}")

        if key == "calibration_output_dir" and (
            matches_path(_legacy_model_calibration_root())
            or matches_path(_intermediate_model_calibration_root())
        ):
            return default
        if key == "machine_map_path" and (
            matches_path(self._legacy_default_machine_map_path())
            or matches_path(self._intermediate_default_machine_map_path())
        ):
            return default
        return value

    def _preference_values_from_settings(self) -> dict[str, str]:
        defaults = self._preference_defaults()
        values = {}
        for key, default in defaults.items():
            value = self._settings.value(self._preference_key(key), default)
            values[key] = (
                default
                if value is None
                else self._migrate_preference_value(key, str(value), default)
            )
        return values

    def _preference_widget_value(self, widget) -> str:
        if hasattr(widget, "currentData"):
            value = widget.currentData()
            return "" if value is None else str(value)
        if hasattr(widget, "value"):
            return str(widget.value())
        if hasattr(widget, "text"):
            return widget.text().strip()
        return ""

    def _preferences_from_widgets(self) -> dict[str, str]:
        return {
            key: self._preference_widget_value(widget)
            for key, widget in self._preference_fields.items()
        }

    def _set_combo_by_data(self, combo, value: str) -> bool:
        index = combo.findData(value)
        if index < 0:
            return False
        combo.setCurrentIndex(index)
        return True

    def _set_preference_widget_value(self, widget, value: str) -> None:
        if hasattr(widget, "findData") and hasattr(widget, "setCurrentIndex"):
            if not self._set_combo_by_data(widget, value):
                index = widget.findText(value)
                if index >= 0:
                    widget.setCurrentIndex(index)
            return
        if hasattr(widget, "setValue"):
            try:
                widget.setValue(int(float(value)))
            except ValueError:
                return
            return
        if hasattr(widget, "setText"):
            widget.setText(value)

    def _set_preference_widgets(self, values: dict[str, str]) -> None:
        for key, value in values.items():
            widget = self._preference_fields.get(key)
            if widget is not None:
                self._set_preference_widget_value(widget, value)

    def _load_preferences(self) -> None:
        values = self._preference_values_from_settings()
        self._set_preference_widgets(values)
        self._apply_preferences_to_forms(values)

    def _apply_preferences_from_page(self) -> None:
        values = self._preferences_from_widgets()
        self._apply_preferences_to_forms(values)
        self._append_log("Preferences applied to current forms.")

    def _save_preferences_from_page(self) -> None:
        values = self._preferences_from_widgets()
        for key, value in values.items():
            self._settings.setValue(self._preference_key(key), value)
        self._settings.sync()
        self._apply_preferences_to_forms(values)
        self._append_log("Preferences saved.")

    def _reset_preferences(self) -> None:
        values = self._preference_defaults()
        for key in values:
            self._settings.remove(self._preference_key(key))
        self._settings.sync()
        self._set_preference_widgets(values)
        self._apply_preferences_to_forms(values)
        self._append_log("Preferences reset to defaults.")

    def _set_line_edit_if_present(self, attribute: str, value: str) -> None:
        widget = getattr(self, attribute, None)
        if widget is not None:
            widget.setText(value)

    def _set_spin_box_if_present(self, attribute: str, value: str) -> None:
        widget = getattr(self, attribute, None)
        if widget is None:
            return
        try:
            widget.setValue(int(float(value)))
        except ValueError:
            return

    def _apply_preferences_to_forms(self, values: dict[str, str]) -> None:
        self._apply_style(values.get("interface_theme", "workbench_light"))
        self._set_line_edit_if_present("_output_dir", values.get("simulation_output_dir", ""))
        self._set_line_edit_if_present(
            "_calibration_output_dir", values.get("calibration_output_dir", "")
        )
        self._set_line_edit_if_present(
            "_calibration_geometry", values.get("calibration_geometry", "")
        )
        self._set_line_edit_if_present(
            "_calibration_sample_dir", values.get("calibration_sample_dir", "")
        )
        self._set_line_edit_if_present(
            "_calibration_spacing", values.get("calibration_spacing", "")
        )
        self._set_line_edit_if_present(
            "_machine_map_coordinates", values.get("machine_coordinate_workbook", "")
        )
        machine_map_path = values.get("machine_map_path", "")
        self._set_line_edit_if_present("_machine_map_path", machine_map_path)
        self._set_line_edit_if_present("_machine_map_name", values.get("machine_map_name", ""))

        if hasattr(self, "_calibration_optimizer"):
            self._set_combo_by_data(
                self._calibration_optimizer,
                values.get("calibration_optimizer", "global_evolution"),
            )
        self._set_spin_box_if_present(
            "_calibration_max_evaluations", values.get("calibration_evaluations", "40")
        )
        self._set_spin_box_if_present(
            "_calibration_parallel_samples",
            values.get("calibration_parallel_samples", str(self._recommended_parallel_samples())),
        )
        if hasattr(self, "_refresh_machine_preset_list"):
            self._refresh_machine_preset_list(
                select_path=Path(machine_map_path) if machine_map_path else None,
                preserve_current=False,
            )
        elif hasattr(self, "_refresh_machine_map_name"):
            self._refresh_machine_map_name()
        if hasattr(self, "_update_machine_preset_controls"):
            self._update_machine_preset_controls()

    def _browse_preference_dir(self, line_edit, title: str) -> None:
        path = self._QFileDialog.getExistingDirectory(self._window, title, line_edit.text())
        if path:
            line_edit.setText(path)

    def _browse_preference_file(self, line_edit, title: str, filter_text: str) -> None:
        path, _ = self._QFileDialog.getOpenFileName(
            self._window,
            title,
            line_edit.text() or str(Path.cwd()),
            filter_text,
        )
        if path:
            line_edit.setText(path)

    def _make_log(self):
        from PySide6.QtWidgets import QPlainTextEdit

        log = QPlainTextEdit()
        log.setObjectName("Log")
        log.setReadOnly(True)
        log.setMinimumHeight(64)
        log.setMaximumHeight(88)
        log.setPlainText("Ready.\n")
        return log

    def _select_feature(self, index: int) -> None:
        self._navigation.setCurrentRow(index)

    def _set_page(self, index: int) -> None:
        if index >= 0:
            self._stack.setCurrentIndex(index)
        if index == 1:
            self._sync_main_splitter_sizes()
            self._fit_result_slice_pixmap()

    def _browse_part(self) -> None:
        path = self._open_file("Select part geometry", "STL files (*.stl);;All files (*.*)")
        if path:
            self._part_geometry.setText(path)
            self._append_log(f"Part geometry selected: {path}")
            self._estimate_grid_spacing_for_path(path)
            self._preview_part_geometry()

    def _geometry_orientation_angles(self) -> tuple[float, float, float]:
        fields = getattr(self, "_orientation_fields", None)
        if not fields:
            return (0.0, 0.0, 0.0)
        values = []
        for label, field in zip(("X", "Y", "Z"), fields, strict=True):
            text = field.text().strip()
            if not text:
                values.append(0.0)
                continue
            values.append(self._float(field, f"Orientation {label}"))
        return _normalize_orientation_angles(tuple(values))

    def _geometry_processing_output_dir(self) -> Path:
        output_widget = getattr(self, "_output_dir", None)
        text = output_widget.text().strip() if output_widget is not None else ""
        return Path(text or "examples/outputs/gui_simulation")

    def _part_geometry_processing_path(self) -> Path:
        source_path = Path(self._part_geometry.text().strip())
        if not source_path.exists():
            raise ValueError("Select a valid part geometry STL file.")
        return _oriented_geometry_path(
            source_path,
            self._geometry_processing_output_dir(),
            self._geometry_orientation_angles(),
        )

    def _geometry_orientation_changed(self, *_args) -> None:
        self._invalidate_voxelization()
        support_source = getattr(self, "_support_source", None)
        if support_source is not None and support_source.currentText() == "Generate from overhang":
            self._last_generated_support_grid = None
            self._last_generated_support_path = None
            self._last_generated_support_type = None
            self._last_generated_support_signature = None
            self._last_generated_support_options = None
            self._last_support_overlay_preview = None
            if hasattr(self, "_save_support_button"):
                self._save_support_button.setEnabled(False)
            if hasattr(self, "_support_geometry"):
                if hasattr(self._support_geometry, "blockSignals"):
                    self._support_geometry.blockSignals(True)
                self._support_geometry.setText("")
                if hasattr(self._support_geometry, "blockSignals"):
                    self._support_geometry.blockSignals(False)

    def _browse_support(self) -> None:
        path = self._open_file("Select support geometry", "STL files (*.stl);;All files (*.*)")
        if path:
            self._support_geometry.setText(path)
            self._append_log(f"Support geometry selected: {path}")
            self._invalidate_voxelization()
            try:
                part_path = self._part_geometry_processing_path()
            except Exception:
                part_path = None
            if part_path is not None:
                self._preview_support_overlay(part_path, path)

    def _browse_output_dir(self) -> None:
        path = self._QFileDialog.getExistingDirectory(
            self._window,
            "Select output directory",
            str(Path.cwd()),
        )
        if path:
            self._output_dir.setText(path)
            self._append_log(f"Output directory selected: {path}")

    def _browse_calibration_geometry(self) -> None:
        path = self._open_file("Select calibration STL", "STL files (*.stl);;All files (*.*)")
        if path:
            self._calibration_geometry.setText(path)
            self._append_log(f"Model Calibration STL selected: {path}")

    def _browse_calibration_sample_dir(self) -> None:
        path = self._QFileDialog.getExistingDirectory(
            self._window,
            "Select ROI sample folder",
            str(Path.cwd()),
        )
        if path:
            self._calibration_sample_dir.setText(path)
            self._append_log(f"Model Calibration sample folder selected: {path}")

    def _browse_calibration_output_dir(self) -> None:
        path = self._QFileDialog.getExistingDirectory(
            self._window,
            "Select machine preset library",
            str(Path.cwd()),
        )
        if path:
            self._calibration_output_dir.setText(path)
            self._refresh_machine_preset_list(preserve_current=False)
            self._append_log(f"Machine preset library selected: {path}")

    def _calibration_output_root(self) -> Path:
        widget = getattr(self, "_calibration_output_dir", None)
        text = widget.text().strip() if widget is not None else ""
        return Path(text or _default_machine_preset_library_root())

    def _machine_preset_root(self) -> Path:
        return self._calibration_output_root()

    def _path_match_key(self, path: str | Path) -> str:
        try:
            return str(Path(path).resolve()).casefold()
        except Exception:
            return str(Path(path)).casefold()

    def _read_machine_map_metadata(self, path: str | Path):
        from capp.machine_map import read_machine_parameter_map_metadata

        return read_machine_parameter_map_metadata(path)

    def _machine_map_detail_label(self, metadata) -> str:
        label = metadata.preset_name
        if metadata.voxel_spacing is not None:
            label = f"{label} ({metadata.voxel_spacing:g} mm)"
        return label

    def _discover_machine_map_presets(self) -> list[tuple[str, Path]]:
        root = self._machine_preset_root()
        if not root.exists():
            return []
        presets: list[tuple[str, Path]] = []
        paths = list(root.glob("*/map/machine_parameter_map.npz"))
        paths.extend(root.glob("*/machine_parameter_map.npz"))
        for path in sorted(set(paths)):
            try:
                metadata = self._read_machine_map_metadata(path)
            except Exception:
                continue
            presets.append((metadata.preset_name, path))
        return presets

    def _selected_machine_map_path(self) -> Path | None:
        if not hasattr(self, "_machine_preset"):
            return None
        data = self._machine_preset.currentData()
        if not data:
            return None
        return Path(str(data))

    def _sync_machine_map_path_from_preset(self) -> None:
        if not hasattr(self, "_machine_map_path"):
            return
        path = self._selected_machine_map_path()
        self._machine_map_path.setText("" if path is None else str(path))

    def _refresh_machine_preset_list(
        self,
        select_path: str | Path | None = None,
        *,
        preserve_current: bool = True,
        load_contour: bool = False,
    ) -> None:
        if not hasattr(self, "_machine_preset"):
            return
        target_path = Path(select_path) if select_path else None
        if target_path is None and preserve_current:
            target_path = self._selected_machine_map_path()
            if target_path is None and hasattr(self, "_machine_map_path"):
                text = self._machine_map_path.text().strip()
                target_path = Path(text) if text else None

        target_key = self._path_match_key(target_path) if target_path is not None else ""
        self._machine_preset.blockSignals(True)
        self._machine_preset.clear()
        self._machine_preset.addItem("None", None)
        selected_index = 0
        known_paths: set[str] = set()
        for label, path in self._discover_machine_map_presets():
            key = self._path_match_key(path)
            known_paths.add(key)
            self._machine_preset.addItem(label, str(path))
            if target_key and key == target_key:
                selected_index = self._machine_preset.count() - 1

        if (
            target_path is not None
            and target_path.exists()
            and target_key
            and target_key not in known_paths
        ):
            try:
                metadata = self._read_machine_map_metadata(target_path)
                label = metadata.preset_name
            except Exception:
                label = target_path.stem
            self._machine_preset.addItem(label, str(target_path))
            selected_index = self._machine_preset.count() - 1

        self._machine_preset.setCurrentIndex(selected_index)
        self._machine_preset.blockSignals(False)
        self._sync_machine_map_path_from_preset()
        self._refresh_machine_map_name(load_contour=load_contour)
        if hasattr(self, "_machine_preset_locked_widgets"):
            self._update_machine_preset_controls()

    def _machine_map_path_edited(self) -> None:
        text = self._machine_map_path.text().strip()
        self._refresh_machine_preset_list(select_path=Path(text) if text else None)

    def _browse_machine_map_coordinates(self) -> None:
        path = self._open_file(
            "Select sample coordinate workbook",
            "Excel workbooks (*.xlsx);;All files (*.*)",
        )
        if path:
            self._machine_map_coordinates.setText(path)
            self._append_log(f"Machine map coordinate workbook selected: {path}")

    def _browse_machine_map_path(self) -> None:
        path = self._open_file(
            "Select machine parameter map",
            "Machine parameter maps (*.npz);;All files (*.*)",
        )
        if path:
            self._machine_map_path.setText(path)
            self._refresh_machine_preset_list(select_path=Path(path), load_contour=True)
            self._append_log(f"Machine parameter map selected: {path}")

    def _update_machine_map_coordinate_fields(self) -> None:
        machine_map_active = self._selected_machine_map_path() is not None
        if hasattr(self, "_machine_map_coordinate_mode"):
            self._machine_map_coordinate_mode.setEnabled(machine_map_active)
        coordinate_mode = (
            self._machine_map_coordinate_mode.currentText()
            if machine_map_active
            else ""
        )
        center_mode = machine_map_active and coordinate_mode == "Part center"
        bounds_mode = machine_map_active and coordinate_mode == "Explicit bounds"
        for widget in self._machine_map_center_widgets:
            widget.setEnabled(center_mode)
        for widget in self._machine_map_bounds_widgets:
            widget.setEnabled(bounds_mode)

    def _refresh_machine_map_name(self, *, load_contour: bool = False) -> None:
        path = self._selected_machine_map_path()
        if path is None:
            text = self._machine_map_path.text().strip()
            path = Path(text) if text else None
        if path is None:
            self._machine_map_preset_name.setText("-")
            return
        if not path.exists():
            self._machine_map_preset_name.setText("-")
            return
        try:
            metadata = self._read_machine_map_metadata(path)
            label = self._machine_map_detail_label(metadata)
            self._machine_map_preset_name.setText(label)
            if self._selected_machine_map_path() is not None:
                self._apply_machine_map_spacing(metadata.voxel_spacing)
            if load_contour and hasattr(self, "_machine_map_contour_label"):
                self._load_machine_map_contour(path, silent=True)
        except Exception as exc:
            self._machine_map_preset_name.setText(f"Unreadable map: {exc}")

    def _update_machine_preset_controls(self, *_args) -> None:
        machine_map_active = self._selected_machine_map_path() is not None
        self._sync_machine_map_path_from_preset()
        if machine_map_active:
            self._neighborhood.blockSignals(True)
            self._neighborhood.setCurrentText("DirectionalVN")
            self._neighborhood.blockSignals(False)
            self._stochastic_mode.setCurrentText("In-layer")
            self._coeff_lower.setText("1")
            self._residual_avg.setText("1E-5")
            self._residual_max.setText("1E-4")
            self._iteration_bound.setText("400")
            self._overwrap.setText("0.1")
            self._refresh_machine_map_name()
        else:
            self._machine_map_preset_name.setText("-")

        self._apply_machine_preset_lock_state(machine_map_active)
        self._apply_machine_map_input_state(machine_map_active)
        self._update_machine_map_coordinate_fields()

    def _apply_machine_map_spacing(self, voxel_spacing: float | None) -> None:
        if voxel_spacing is None:
            return
        text = f"{voxel_spacing:g}"
        if self._grid_spacing.text().strip() == text:
            return
        self._grid_spacing.setText(text)
        self._append_log(f"Grid spacing locked from Machine Map: {text} mm")

    def _apply_machine_preset_lock_state(self, machine_map_active: bool) -> None:
        if not hasattr(self, "_machine_preset_locked_widgets"):
            return
        for widget in self._machine_preset_locked_widgets:
            widget.setEnabled(not machine_map_active)
        if not machine_map_active:
            self._set_parameter_defaults(self._neighborhood.currentText())
            return
        for widget in self._machine_preset_locked_widgets:
            widget.setEnabled(False)

    def _apply_machine_map_input_state(self, machine_map_active: bool) -> None:
        if not hasattr(self, "_machine_map_path"):
            return
        self._machine_map_path.setEnabled(machine_map_active)
        self._machine_map_path.setReadOnly(True)
        browse_button = getattr(self._machine_map_path, "_browse_button", None)
        if browse_button is not None:
            browse_button.setEnabled(True)
        self._machine_map_preset_name.setEnabled(machine_map_active)

    def _default_machine_coordinate_path(self) -> Path:
        project_root = Path(__file__).resolve().parents[4]
        path = project_root / "data" / "machine_map" / "sp_coordinates.xlsx"
        if path.exists():
            return path
        return Path("..") / "data" / "machine_map" / "sp_coordinates.xlsx"

    def _default_calibration_geometry_path(self) -> Path:
        project_root = Path(__file__).resolve().parents[4]
        path = (
            project_root
            / "data"
            / "geometry_examples"
            / "KITECH_ARTIFACTS"
            / "Test_artifact_ver.4.stl"
        )
        if path.exists():
            return path
        return (
            Path("..")
            / "data"
            / "geometry_examples"
            / "KITECH_ARTIFACTS"
            / "Test_artifact_ver.4.stl"
        )

    def _default_calibration_sample_dir(self) -> Path:
        project_root = Path(__file__).resolve().parents[4]
        path = project_root / "data" / "calibration_samples"
        if path.exists():
            return path
        return Path("..") / "data" / "calibration_samples"

    def _default_machine_map_path(self) -> Path:
        path = (
            _machine_map_preset_output_dir(
                _default_machine_preset_library_root(),
                "Machine Map",
            )
            / "machine_parameter_map.npz"
        )
        if path.exists():
            return path
        return path

    def _preview_source_changed(self, source: str) -> None:
        self._render_preview_source(source)

    def _set_preview_source(self, source: str) -> None:
        if not hasattr(self, "_preview"):
            return
        selector = self._preview.source_selector
        selector.blockSignals(True)
        selector.setCurrentText(source)
        selector.blockSignals(False)
        self._update_preview_source_controls(source)
        self._update_preview_source_availability()

    def _update_preview_source_controls(self, source: str | None = None) -> None:
        if not hasattr(self, "_preview"):
            return
        source = source or self._preview.source_selector.currentText()
        self._preview.set_stl_controls_visible(source == "STL")
        self._preview.set_volume_controls_visible(source in {"Voxelization", "Result"})

    def _update_preview_source_availability(self) -> None:
        if not hasattr(self, "_preview"):
            return
        selector = self._preview.source_selector
        has_stl = self._last_stl_preview is not None or bool(self._part_geometry.text().strip())
        available = {
            "STL": has_stl,
            "Voxelization": self._last_voxel_grid is not None,
            "Result": self._last_result is not None,
        }
        model = selector.model()
        for index in range(selector.count()):
            item = model.item(index)
            if item is not None:
                item.setEnabled(available.get(selector.itemText(index), False))

    def _render_preview_source(self, source: str | None = None) -> None:
        if not hasattr(self, "_preview"):
            return
        source = source or self._preview.source_selector.currentText()
        self._update_preview_source_controls(source)
        self._update_preview_source_availability()
        if source == "STL":
            self._show_stl_preview_from_cache()
        elif source == "Voxelization":
            self._show_voxelization_preview_from_cache()
        elif source == "Result":
            self._show_result_preview_from_cache()

    def _show_stl_preview_from_cache(self) -> None:
        if self._last_support_overlay_preview is not None:
            mode, overhang_limit = self._stl_preview_display_settings()
            (
                part_path,
                part_mesh,
                part_cells,
                support_path,
                support_mesh,
                support_cells,
            ) = self._last_support_overlay_preview
            self._preview.show_stl_overlay_mesh(
                part_path,
                part_mesh,
                part_cells,
                support_path,
                support_mesh,
                support_cells,
                display_mode=mode,
                overhang_limit=overhang_limit,
            )
            return

        if self._last_stl_preview is not None:
            path, mesh, original_cells = self._last_stl_preview
            mode, overhang_limit = self._stl_preview_display_settings()
            self._preview.show_stl_mesh(
                path,
                mesh,
                original_cells,
                display_mode=mode,
                overhang_limit=overhang_limit,
            )
            return

        path = self._part_geometry.text().strip()
        if path and Path(path).exists():
            self._preview_part_geometry()
        else:
            self._preview.show_message("Open an STL to preview geometry.")

    def _show_voxelization_preview_from_cache(self) -> None:
        if self._last_voxel_grid is None:
            self._preview.show_message("Voxelize geometry to preview voxelization.")
            return
        volume = self._last_voxel_preview_data
        if volume is None:
            volume = self._last_voxel_grid.data
        self._preview.show_voxels(
            volume,
            spacing=self._last_voxel_grid.spacing,
            origin=self._last_voxel_grid.origin,
            label="Voxelization",
            support_mask=self._last_voxel_grid.support_mask,
        )

    def _show_result_preview_from_cache(self) -> None:
        if self._last_result is None:
            self._preview.show_message("Run virtual printing to preview result.")
            return
        self._preview.show_voxels(
            self._last_result.binary,
            spacing=self._last_result.spacing,
            origin=self._last_result.origin,
            label="Result",
        )

    def _preview_part_geometry(self) -> None:
        path = self._part_geometry.text().strip()
        if not path:
            return
        if not Path(path).exists():
            self._append_log(f"STL preview skipped: file not found ({path})")
            return
        try:
            preview_path = self._part_geometry_processing_path()
        except Exception as exc:
            self._append_log(f"STL orientation skipped: {exc}")
            return
        message = (
            "Preparing oriented STL preview"
            if Path(preview_path) != Path(path).resolve()
            else "Preparing STL preview"
        )
        self._preview_stl_path(preview_path, message)

    def _preview_stl_path(self, path: str | Path, message: str = "Preparing STL preview") -> None:
        preview_path = Path(path)
        self._last_support_overlay_preview = None
        self._last_stl_preview = None
        self._stl_preview_expected_path = str(preview_path.resolve())
        self._set_preview_source("STL")
        self._append_log(f"{message}: {preview_path}")
        self._preview.show_message("Loading STL preview...")
        worker = _StlPreviewWorker(str(preview_path))
        worker.signals.finished.connect(self._stl_preview_finished)
        worker.signals.failed.connect(self._stl_preview_failed)
        self._stl_preview_worker = worker
        self._thread_pool.start(worker)

    def _stl_preview_display_settings(self) -> tuple[str, float]:
        mode = self._preview.stl_display_mode.currentText()
        try:
            overhang_limit = float(self._preview.overhang_limit.text())
        except ValueError:
            overhang_limit = 60.0
        return mode, max(1.0, overhang_limit)

    def _refresh_stl_preview_style(self) -> None:
        if (
            self._last_stl_preview is None
            and self._last_support_overlay_preview is None
        ) or not hasattr(self, "_preview"):
            return
        self._set_preview_source("STL")
        mode = self._preview.stl_display_mode.currentText()
        self._show_stl_preview_from_cache()
        self._append_log(f"STL display mode: {mode}")

    def _stl_preview_finished(
        self,
        path: str,
        mesh,
        original_cells: int,
        elapsed_seconds: float,
    ) -> None:
        self._stl_preview_worker = None
        expected_path = getattr(self, "_stl_preview_expected_path", None)
        if expected_path and str(Path(path).resolve()) != expected_path:
            self._append_log("STL preview discarded because the preview target changed.")
            return
        self._last_stl_preview = (path, mesh, original_cells)
        self._set_preview_source("STL")
        self._show_stl_preview_from_cache()
        self._append_log(
            f"STL preview ready: {Path(path).name}, cells={mesh.n_cells}/{original_cells}, "
            f"{elapsed_seconds:.2f} s"
        )

    def _stl_preview_failed(self, path: str, message: str) -> None:
        self._stl_preview_worker = None
        expected_path = getattr(self, "_stl_preview_expected_path", None)
        if expected_path and str(Path(path).resolve()) != expected_path:
            return
        self._append_log(f"STL preview failed: {message}")
        self._preview.show_message(f"STL preview failed: {message}")
        self._update_preview_source_availability()

    def _preview_support_overlay(self, part_path: str | Path, support_path: str | Path) -> None:
        self._last_stl_preview = None
        self._last_support_overlay_preview = None
        self._set_preview_source("STL")
        part = str(Path(part_path).resolve())
        support = str(Path(support_path).resolve())
        self._append_log(f"Preparing CAD + support preview: {part} + {support}")
        self._preview.show_message("Loading CAD + support preview...")
        worker = _SupportOverlayPreviewWorker(part, support)
        worker.signals.finished.connect(self._support_overlay_preview_finished)
        worker.signals.failed.connect(self._support_overlay_preview_failed)
        self._support_overlay_preview_worker = worker
        self._thread_pool.start(worker)

    def _support_overlay_preview_finished(
        self,
        part_path: str,
        part_mesh,
        part_cells: int,
        support_path: str,
        support_mesh,
        support_cells: int,
        elapsed_seconds: float,
    ) -> None:
        self._support_overlay_preview_worker = None
        expected_support = getattr(self, "_last_generated_support_path", None)
        if expected_support is not None and str(Path(support_path).resolve()) != str(
            expected_support.resolve()
        ):
            self._append_log("CAD + support preview discarded because the support target changed.")
            return
        self._last_support_overlay_preview = (
            part_path,
            part_mesh,
            part_cells,
            support_path,
            support_mesh,
            support_cells,
        )
        self._set_preview_source("STL")
        self._show_stl_preview_from_cache()
        self._append_log(
            "CAD + support preview ready: "
            f"{Path(part_path).name} + {Path(support_path).name}, "
            f"{elapsed_seconds:.2f} s"
        )

    def _support_overlay_preview_failed(
        self,
        part_path: str,
        support_path: str,
        message: str,
    ) -> None:
        self._support_overlay_preview_worker = None
        self._append_log(
            f"CAD + support preview failed for {Path(part_path).name} + "
            f"{Path(support_path).name}: {message}"
        )
        self._preview.show_message(f"CAD + support preview failed: {message}")
        self._update_preview_source_availability()

    def _preview_result(self) -> None:
        if self._last_result is None:
            self._QMessageBox.warning(self._window, "Missing result", "Run a simulation first.")
            return
        self._set_preview_source("Result")
        self._append_log("Previewing virtual printing result.")
        self._show_result_preview_from_cache()

    def _voxelize_geometry(self) -> None:
        if self._busy:
            self._append_log("A task is already in progress.")
            return
        try:
            config = self._simulation_config_from_form()
        except Exception as exc:
            self._QMessageBox.warning(self._window, "Invalid input", str(exc))
            return

        self._append_log(
            f"Voxelizing selected geometry: {config.geometry_path} at {config.voxel_spacing:g} mm"
        )
        self._set_busy(True, "Voxelizing geometry...", task="voxelization")
        generated_support_grid = self._cached_generated_support_grid_for_config(config)
        if generated_support_grid is not None:
            self._append_log("Using cached generated support grid for voxelization.")
        worker = _VoxelizationWorker(config, generated_support_grid=generated_support_grid)
        worker.signals.progress.connect(self._set_task_progress)
        worker.signals.finished.connect(self._voxelization_finished)
        worker.signals.failed.connect(self._voxelization_failed)
        self._voxelization_worker = worker
        self._thread_pool.start(worker)

    def _voxelization_finished(self, config, grid, display_data) -> None:
        self._voxelization_worker = None
        try:
            current_config = self._simulation_config_from_form()
            is_stale = self._voxel_signature(config) != self._voxel_signature(current_config)
        except Exception:
            is_stale = True
        if is_stale:
            self._last_voxel_grid = None
            self._last_voxel_preview_data = None
            self._last_voxel_signature = None
            self._voxel_status_label.setText("Required")
            self._set_busy(False, "Voxelization discarded")
            self._update_preview_source_availability()
            self._append_log(
                "Voxelization finished, but geometry or spacing changed. Run voxelization again."
            )
            return

        self._last_voxel_grid = grid
        self._last_voxel_preview_data = display_data
        self._last_voxel_signature = self._voxel_signature(config)
        self._set_preview_source("Voxelization")
        self._show_voxelization_preview_from_cache()
        self._voxel_status_label.setText(
            f"{grid.shape[0]} x {grid.shape[1]} x {grid.shape[2]}, {grid.filled_count} voxels"
        )
        self._shape_label.setText(" x ".join(str(v) for v in grid.shape))
        self._spacing_label.setText(f"{grid.spacing:g}")
        self._run_button.setEnabled(True)
        self._set_busy(False, "Voxelization complete")
        support_voxels = int(grid.support_mask.sum())
        self._append_log(
            f"Voxel grid ready: shape={grid.shape}, filled={grid.filled_count}, "
            f"support={support_voxels}"
        )

    def _voxelization_failed(self, _config, message: str) -> None:
        self._voxelization_worker = None
        self._last_voxel_grid = None
        self._last_voxel_preview_data = None
        self._last_voxel_signature = None
        self._voxel_status_label.setText("Required")
        self._set_busy(False, "Voxelization failed")
        self._update_preview_source_availability()
        self._append_log(f"Voxelization failed: {message}")
        self._QMessageBox.critical(self._window, "Voxelization failed", message)

    def _browse_result_npz(self) -> None:
        path = self._open_file("Open simulation result", "NPZ files (*.npz);;All files (*.*)")
        if path:
            self._result_npz_path.setText(path)
            self._load_result_npz(path)

    def _browse_deviation_stl(self) -> None:
        path = self._open_file("Select original STL", "STL files (*.stl);;All files (*.*)")
        if path:
            self._deviation_stl_path.setText(path)
            self._append_log(f"Geometry deviation original STL selected: {path}")

    def _load_result_npz(self, path: str | Path) -> None:
        try:
            import numpy as np

            data = np.load(path)
            source_geometry = self._result_source_geometry_from_npz(data)
            voxel = data["voxel"].astype(bool)
            support_mask = (
                data["support_mask"].astype(bool)
                if "support_mask" in data.files
                else np.zeros_like(voxel, dtype=bool)
            )
            self._loaded_result = {
                "probability": data["probability"],
                "binary": data["binary"].astype(bool),
                "voxel": voxel,
                "support_mask": support_mask,
                "spacing": float(data["spacing"][0]),
                "origin": tuple(float(v) for v in data["origin"]),
                "rest_volume": (
                    float(data["rest_volume"][0])
                    if "rest_volume" in data.files
                    else 0.0
                ),
                "probability_density": (
                    float(data["probability_density"][0])
                    if "probability_density" in data.files
                    else 0.0
                ),
                "elapsed_seconds": (
                    float(data["elapsed_seconds"][0])
                    if "elapsed_seconds" in data.files
                    else 0.0
                ),
                "path": Path(path),
                "source_geometry": source_geometry,
            }
            self._mark_loaded_result_changed()
            self._sync_deviation_stl_from_result()
            self._output_label.setText(str(Path(path).parent))
            self._files_label.setText("\n".join(data.files))
            self._append_log(f"Loaded result NPZ: {path}")
            self._refresh_result_views()
        except Exception as exc:
            self._append_log(f"Result load failed: {exc}")
            self._QMessageBox.critical(self._window, "Result load failed", str(exc))

    def _result_source_geometry_from_npz(self, data) -> Path | None:
        if "source_geometry" not in data.files:
            return None
        raw = data["source_geometry"]
        if raw.size == 0:
            return None
        text = str(raw[0]).strip()
        return Path(text) if text else None

    def _set_loaded_result_from_simulation(self, result, output_dir: Path) -> None:
        self._loaded_result = {
            "probability": result.probability,
            "binary": result.binary,
            "voxel": result.voxel,
            "support_mask": result.support_mask,
            "spacing": result.spacing,
            "origin": result.origin,
            "rest_volume": result.rest_volume,
            "probability_density": result.probability_density,
            "elapsed_seconds": result.elapsed_seconds,
            "path": None,
            "source_geometry": result.source_geometry,
        }
        self._mark_loaded_result_changed()
        self._sync_deviation_stl_from_result()
        self._result_npz_path.setText("")
        self._output_label.setText(f"In memory; target: {output_dir}")
        self._files_label.setText("Not saved")
        self._refresh_result_views()

    def _sync_deviation_stl_from_result(self) -> None:
        if not hasattr(self, "_deviation_stl_path"):
            return
        source = None if self._loaded_result is None else self._loaded_result.get("source_geometry")
        self._deviation_stl_path.setText("" if source is None else str(source))
        self._update_result_action_state()

    def _mark_loaded_result_changed(self) -> None:
        self._result_revision = int(getattr(self, "_result_revision", 0)) + 1
        if hasattr(self, "_deviation_summary"):
            self._deviation_summary.setText("Ready")

    def _update_result_action_state(self) -> None:
        enabled = (not getattr(self, "_busy", False)) and self._loaded_result is not None
        if hasattr(self, "_deviation_button"):
            self._deviation_button.setEnabled(enabled)
        if hasattr(self, "_result_display_preview_button"):
            self._result_display_preview_button.setEnabled(enabled)
        if hasattr(self, "_result_hide_support"):
            support_mask = self._loaded_support_mask()
            self._result_hide_support.setEnabled(enabled and bool(support_mask.any()))

    def _loaded_support_mask(self):
        import numpy as np

        if self._loaded_result is None:
            return np.zeros((1, 1, 1), dtype=bool)
        voxel = self._loaded_result["voxel"]
        support_mask = self._loaded_result.get("support_mask")
        if support_mask is None:
            return np.zeros_like(voxel, dtype=bool)
        return np.asarray(support_mask, dtype=bool)

    def _result_support_hidden(self) -> bool:
        return (
            hasattr(self, "_result_hide_support")
            and self._result_hide_support.isChecked()
            and bool(self._loaded_support_mask().any())
        )

    def _result_support_removed(self) -> bool:
        return self._result_support_hidden()

    def _result_support_visibility_changed(self, *_args) -> None:
        if self._loaded_result is None:
            return
        self._mark_loaded_result_changed()
        self._refresh_result_views()

    def _refresh_result_views(self, *_args) -> None:
        if self._loaded_result is None:
            return
        self._update_result_action_state()
        volume = self._selected_result_volume()
        axis = self._slice_axis.currentText()
        axis_index = {"X": 0, "Y": 1, "Z": 2}[axis]
        self._slice_slider.blockSignals(True)
        self._slice_slider.setMinimum(0)
        self._slice_slider.setMaximum(max(0, volume.shape[axis_index] - 1))
        self._slice_slider.setValue(volume.shape[axis_index] // 2)
        self._slice_slider.blockSignals(False)
        self._update_result_slice()

    def _preview_loaded_result(self, *_args, show_error: bool = True) -> None:
        if self._loaded_result is None:
            if show_error:
                self._QMessageBox.warning(
                    self._window,
                    "Missing result",
                    "Load or run a result first.",
                )
            return
        volume = self._selected_result_volume()
        name = self._result_volume_choice.currentText()
        if name == "Probability":
            preview_volume = volume
            label = "Probability"
        else:
            preview_volume = volume.astype(bool)
            label = name
        if self._result_support_removed():
            label = f"{label} (support removed)"
        self._append_log(f"Previewing result volume: {label}")
        try:
            self._result_preview.show_voxels(
                preview_volume,
                spacing=self._loaded_result["spacing"],
                origin=self._loaded_result["origin"],
                label=label,
                raise_errors=True,
            )
        except Exception as exc:
            self._append_log(f"Result 3D preview failed: {exc}")
            if show_error:
                self._QMessageBox.critical(
                    self._window,
                    "Result 3D preview failed",
                    str(exc),
                )

    def _preview_geometry_deviation(self) -> None:
        if self._loaded_result is None:
            self._QMessageBox.warning(self._window, "Missing result", "Load or run a result first.")
            return
        if self._busy:
            self._append_log("A task is already in progress.")
            return
        stl_text = self._deviation_stl_path.text().strip()
        if not stl_text:
            self._QMessageBox.warning(
                self._window,
                "Missing STL",
                "Select the original STL before generating a deviation heatmap.",
            )
            return
        stl_path = Path(stl_text)
        if not stl_path.exists():
            self._QMessageBox.warning(
                self._window,
                "Invalid STL",
                "Select a valid original STL file.",
            )
            return
        volume = self._selected_result_volume_without_support()
        label = self._result_volume_choice.currentText()
        if bool(self._loaded_support_mask().any()):
            label = f"{label} (support removed)"
        self._append_log(f"Rendering geometry deviation heatmap: {stl_path}")
        self._deviation_summary.setText("Preparing geometry deviation...")
        self._set_busy(True, "Preparing geometry deviation...", task="geometry_deviation")
        worker = _GeometryDeviationWorker(
            str(stl_path),
            volume,
            float(self._loaded_result["spacing"]),
            tuple(float(v) for v in self._loaded_result["origin"]),
            label,
            int(self._result_revision),
        )
        worker.signals.progress.connect(self._set_geometry_deviation_progress)
        worker.signals.finished.connect(self._geometry_deviation_finished)
        worker.signals.failed.connect(self._geometry_deviation_failed)
        self._geometry_deviation_worker = worker
        self._thread_pool.start(worker)

    def _set_geometry_deviation_progress(self, percent: int, message: str) -> None:
        self._set_task_progress(percent, message)
        if hasattr(self, "_deviation_summary"):
            self._deviation_summary.setText(message)

    def _geometry_deviation_finished(self, preview, result_revision: int) -> None:
        self._geometry_deviation_worker = None
        if result_revision != self._result_revision:
            self._append_log("Discarded stale geometry deviation for a previous result.")
            self._set_busy(False, "Stale geometry deviation discarded")
            return
        try:
            self._result_preview.show_geometry_deviation_preview(preview)
        except Exception as exc:
            self._geometry_deviation_failed(str(exc))
            return

        metrics = preview.metrics
        alignment = preview.alignment_offset
        alignment_text = ""
        if max(abs(value) for value in alignment) > 1e-9:
            alignment_text = (
                f"; aligned origin by {alignment[0]:.4g}, "
                f"{alignment[1]:.4g}, {alignment[2]:.4g} mm"
            )
        self._deviation_summary.setText(
            "mean |d| "
            f"{metrics['mean_abs_mm']:.4g} mm, p95 {metrics['p95_abs_mm']:.4g} mm, "
            f"max |d| {metrics['max_abs_mm']:.4g} mm, signed "
            f"{metrics['min_signed_mm']:.4g} to {metrics['max_signed_mm']:.4g} mm"
            ", color scale -1/+1 mm"
            f"{alignment_text}"
        )
        self._append_log(
            "Geometry deviation ready: "
            f"mean |d|={metrics['mean_abs_mm']:.4g} mm, "
            f"p95={metrics['p95_abs_mm']:.4g} mm, "
            f"max |d|={metrics['max_abs_mm']:.4g} mm"
        )
        self._set_busy(False, "Geometry deviation ready")

    def _geometry_deviation_failed(self, message: str, result_revision: int | None = None) -> None:
        self._geometry_deviation_worker = None
        if result_revision is not None and result_revision != self._result_revision:
            self._append_log(f"Discarded stale geometry deviation failure: {message}")
            self._set_busy(False, "Stale geometry deviation discarded")
            return
        self._deviation_summary.setText(f"Failed: {message}")
        self._append_log(f"Geometry deviation heatmap failed: {message}")
        self._set_busy(False, "Geometry deviation failed")
        self._QMessageBox.critical(self._window, "Geometry deviation failed", message)

    def _update_result_slice(self, *_args) -> None:
        if self._loaded_result is None:
            return
        volume = self._selected_result_volume()
        axis = self._slice_axis.currentText()
        index = self._slice_slider.value()
        if axis == "Z":
            image = volume[:, :, index].T
        elif axis == "X":
            image = volume[index, :, :].T
        else:
            image = volume[:, index, :].T
        self._result_slice_source_image = image
        self._fit_result_slice_pixmap()

    def _fit_result_slice_pixmap(self) -> None:
        if self._result_slice_source_image is None or not hasattr(self, "_slice_label"):
            return
        colorize = (
            hasattr(self, "_result_volume_choice")
            and self._result_volume_choice.currentText() == "Probability"
        )
        self._slice_label.setPixmap(
            self._array_to_pixmap(self._result_slice_source_image, colorize=colorize)
        )

    def _selected_result_volume(self):
        if self._loaded_result is None:
            raise ValueError("No result loaded.")
        choice = self._result_volume_choice.currentText()
        support_mask = self._loaded_support_mask()
        hide_support = self._result_support_removed()
        if choice.startswith("Binary"):
            volume = self._loaded_result["binary"]
            return volume & ~support_mask if hide_support else volume
        volume = self._loaded_result["probability"]
        if not hide_support:
            return volume
        masked = volume.copy()
        masked[support_mask] = 0
        return masked

    def _selected_result_volume_without_support(self):
        if self._loaded_result is None:
            raise ValueError("No result loaded.")
        choice = self._result_volume_choice.currentText()
        support_mask = self._loaded_support_mask()
        if choice.startswith("Binary"):
            return self._loaded_result["binary"] & ~support_mask
        volume = self._loaded_result["probability"].copy()
        volume[support_mask] = 0
        return volume

    def _array_to_pixmap(self, image, *, colorize: bool = False):
        import numpy as np
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QImage, QPixmap

        array = np.asarray(image)
        if array.ndim != 2:
            raise ValueError("Slice preview requires a 2D image.")

        pad_y = max(4, int(round(array.shape[0] * 0.04)))
        pad_x = max(4, int(round(array.shape[1] * 0.04)))

        if colorize:
            values = array.astype(np.float32, copy=False)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            max_value = float(values.max()) if values.size else 0.0
            if max_value <= 1.0:
                normalized = values
            elif max_value <= 100.0:
                normalized = values / 100.0
            else:
                normalized = values / 255.0
            normalized = np.pad(
                normalized,
                ((pad_y, pad_y), (pad_x, pad_x)),
                mode="constant",
                constant_values=0.0,
            )
            rgb = np.ascontiguousarray(np.flipud(_workbench_colormap(normalized)))
            height, width, _channels = rgb.shape
            qimage = QImage(
                rgb.data,
                width,
                height,
                width * 3,
                QImage.Format.Format_RGB888,
            ).copy()
        elif array.dtype == bool:
            array = array.astype(np.uint8) * 255
            array = np.pad(array, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant")
            array = np.ascontiguousarray(np.flipud(array))
            height, width = array.shape
            qimage = QImage(
                array.data,
                width,
                height,
                width,
                QImage.Format.Format_Grayscale8,
            ).copy()
        else:
            array = array.astype(np.float32)
            if array.size and array.max() > array.min():
                array = 255.0 * (array - array.min()) / (array.max() - array.min())
            array = array.astype(np.uint8)
            array = np.pad(array, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant")
            array = np.ascontiguousarray(np.flipud(array))
            height, width = array.shape
            qimage = QImage(
                array.data,
                width,
                height,
                width,
                QImage.Format.Format_Grayscale8,
            ).copy()
        pixmap = QPixmap.fromImage(qimage)
        target_size = self._slice_label.contentsRect().size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            target_size = self._slice_label.size()
        return pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _refresh_compute_backends(self, log: bool = False) -> None:
        from capp.compute.devices import solver_backend_statuses

        statuses = solver_backend_statuses()
        self._backend_statuses = {status.backend.value: status for status in statuses}
        current_backend = self._processor.currentData()

        self._processor.blockSignals(True)
        self._processor.clear()
        first_available_index = 0
        preferred_index = None
        for index, status in enumerate(statuses):
            self._processor.addItem(status.label, status.backend.value)
            item = self._processor.model().item(index)
            if item is not None:
                item.setEnabled(status.available)
                item.setToolTip(status.detail)
            if status.available and first_available_index == 0:
                first_available_index = index
            if status.available and status.backend.value == "cpu_native":
                preferred_index = index

        restored = False
        if current_backend is not None:
            for index in range(self._processor.count()):
                if self._processor.itemData(index) == current_backend:
                    item = self._processor.model().item(index)
                    if item is None or item.isEnabled():
                        self._processor.setCurrentIndex(index)
                        restored = True
                    break
        if not restored:
            self._processor.setCurrentIndex(
                preferred_index if preferred_index is not None else first_available_index
            )
        self._processor.blockSignals(False)
        self._update_backend_status_label()

        if log:
            for status in statuses:
                state = "available" if status.available else "unavailable"
                self._append_log(f"{status.label}: {state}. {status.detail}")

    def _update_backend_status_label(self, *_args) -> None:
        status = getattr(self, "_backend_statuses", {}).get(self._processor.currentData())
        if status is None:
            self._processor_status.setText("Backend status not validated.")
            self._processor_status.setToolTip("")
            return
        self._processor_status.setText(self._backend_status_line(status))
        self._processor_status.setToolTip(status.detail)

    def _selected_solver_backend(self):
        from capp.domain import SolverBackend

        return SolverBackend(self._processor.currentData() or SolverBackend.CPU_REFERENCE.value)

    def _refresh_calibration_backends(self, log: bool = False) -> None:
        from capp.compute.devices import solver_backend_statuses

        if not hasattr(self, "_calibration_processor"):
            return
        statuses = solver_backend_statuses()
        self._calibration_backend_statuses = {status.backend.value: status for status in statuses}
        current_backend = self._calibration_processor.currentData()

        self._calibration_processor.blockSignals(True)
        self._calibration_processor.clear()
        first_available_index = 0
        preferred_index = None
        for index, status in enumerate(statuses):
            self._calibration_processor.addItem(status.label, status.backend.value)
            item = self._calibration_processor.model().item(index)
            if item is not None:
                item.setEnabled(status.available)
                item.setToolTip(status.detail)
            if status.available and first_available_index == 0:
                first_available_index = index
            if status.available and status.backend.value == "cpu_native":
                preferred_index = index

        restored = False
        if current_backend is not None:
            for index in range(self._calibration_processor.count()):
                if self._calibration_processor.itemData(index) == current_backend:
                    item = self._calibration_processor.model().item(index)
                    if item is None or item.isEnabled():
                        self._calibration_processor.setCurrentIndex(index)
                        restored = True
                    break
        if not restored:
            self._calibration_processor.setCurrentIndex(
                preferred_index if preferred_index is not None else first_available_index
            )
        self._calibration_processor.blockSignals(False)
        self._update_calibration_backend_status_label()

        if log:
            for status in statuses:
                state = "available" if status.available else "unavailable"
                self._append_log(f"{status.label}: {state}. {status.detail}")

    def _update_calibration_backend_status_label(self, *_args) -> None:
        if not hasattr(self, "_calibration_processor_status"):
            return
        status = getattr(self, "_calibration_backend_statuses", {}).get(
            self._calibration_processor.currentData()
        )
        if status is None:
            self._calibration_processor_status.setText("Solver status not validated.")
            self._calibration_processor_status.setToolTip("")
            return
        if hasattr(self, "_calibration_parallel_samples"):
            self._calibration_parallel_samples.setEnabled(True)
            self._calibration_parallel_samples.setToolTip("")
        self._calibration_processor_status.setText(self._backend_status_line(status))
        self._calibration_processor_status.setToolTip(status.detail)

    def _selected_calibration_backend(self):
        from capp.domain import SolverBackend

        return SolverBackend(
            self._calibration_processor.currentData() or SolverBackend.CPU_REFERENCE.value
        )

    def _backend_status_line(self, status) -> str:
        state = "ready" if status.available else "unavailable"
        return f"{status.label}: {state}"

    def _solver_label(self, backend_value: str) -> str:
        status = getattr(self, "_backend_statuses", {}).get(backend_value)
        if status is not None:
            return status.label
        status = getattr(self, "_calibration_backend_statuses", {}).get(backend_value)
        if status is not None:
            return status.label
        return backend_value

    def _open_file(self, title: str, filter_text: str) -> str:
        path, _ = self._QFileDialog.getOpenFileName(
            self._window,
            title,
            str(Path.cwd()),
            filter_text,
        )
        return path

    def _on_part_type_changed(self, value: str) -> None:
        self._update_support_controls(value)
        self._invalidate_voxelization()

    def _on_support_source_changed(self, *_args) -> None:
        self._refresh_support_type_options()
        self._update_support_controls(self._part_type.currentText())
        self._invalidate_voxelization()

    def _on_support_type_changed(self, *_args) -> None:
        self._update_support_controls(self._part_type.currentText())
        self._invalidate_voxelization()

    def _refresh_support_type_options(self) -> None:
        if not hasattr(self, "_support_type"):
            return
        source = (
            self._support_source.currentText()
            if hasattr(self, "_support_source")
            else "External STL"
        )
        options = (
            ["X surface support", "Column support", "Volume support"]
            if source == "Generate from overhang"
            else ["Volume support", "Line support"]
        )
        current = self._support_type.currentText()
        self._support_type.blockSignals(True)
        self._support_type.clear()
        self._support_type.addItems(options)
        self._support_type.setCurrentText(current if current in options else options[0])
        self._support_type.blockSignals(False)

    def _toggle_support_options(self, checked: bool) -> None:
        self._update_support_controls(self._part_type.currentText(), expanded=checked)

    def _update_support_controls(
        self, value: str | None = None, expanded: bool | None = None
    ) -> None:
        part_type = value if value is not None else self._part_type.currentText()
        support_enabled = part_type == "Part & Support"
        if expanded is None:
            expanded = bool(self._support_options_toggle.isChecked())
        panel_visible = support_enabled and bool(expanded)
        source = (
            self._support_source.currentText()
            if hasattr(self, "_support_source")
            else "External STL"
        )
        generated_enabled = support_enabled and source == "Generate from overhang"
        external_enabled = support_enabled and source == "External STL"

        self._support_options_toggle.setEnabled(support_enabled)
        if hasattr(self, "_support_source"):
            self._support_source.setEnabled(support_enabled)
        self._support_geometry.setEnabled(external_enabled)
        browse_button = getattr(self._support_geometry, "_browse_button", None)
        if browse_button is not None:
            browse_button.setEnabled(external_enabled)
        self._support_type.setEnabled(support_enabled)
        for field in getattr(self, "_support_generation_fields", []):
            field.setEnabled(generated_enabled)
        if hasattr(self, "_support_thickness"):
            support_type = (
                self._support_type.currentText()
                if hasattr(self._support_type, "currentText")
                else ""
            )
            self._support_thickness.setEnabled(
                generated_enabled and support_type != "X surface support"
            )
        if hasattr(self, "_generate_support_button"):
            self._generate_support_button.setEnabled(generated_enabled and not self._busy)
        if hasattr(self, "_clear_support_button"):
            self._clear_support_button.setEnabled(support_enabled and not self._busy)
        if hasattr(self, "_save_support_button"):
            self._save_support_button.setEnabled(
                generated_enabled
                and not self._busy
                and self._last_generated_support_grid is not None
            )
        self._support_options_panel.setVisible(panel_visible)

        text = "Hide support options" if panel_visible else "Show support options"
        self._support_options_toggle.setText(text)
        if hasattr(self._support_options_toggle, "setArrowType"):
            arrow = (
                self._Qt.ArrowType.DownArrow
                if panel_visible
                else self._Qt.ArrowType.RightArrow
            )
            self._support_options_toggle.setArrowType(arrow)

    def _support_generation_from_form(self):
        from capp.domain import SupportGenerationParameters

        build_plate_text = self._support_build_plate_z.text().strip()
        support_type = self._support_type.currentText()
        contact_depth_field = getattr(self, "_support_contact_depth", None)
        contact_depth = (
            self._float(contact_depth_field, "Support contact overlap")
            if contact_depth_field is not None
            else 0.0
        )
        thickness = (
            1.0
            if support_type == "X surface support"
            else self._float(self._support_thickness, "Support thickness")
        )
        return SupportGenerationParameters(
            support_type=support_type,
            overhang_angle=self._float(
                self._support_overhang_angle,
                "Support overhang angle",
            ),
            pitch=self._float(self._support_pitch, "Support pitch"),
            thickness=thickness,
            footprint_offset=self._float(
                self._support_footprint_offset,
                "Support footprint offset",
            ),
            contact_depth=contact_depth,
            build_plate_z=(
                None
                if build_plate_text.lower() in {"", "auto"}
                else self._float(self._support_build_plate_z, "Support build plate Z")
            ),
        )

    def _support_generation_signature(
        self,
        geometry_path: Path,
        spacing: float,
        options,
    ) -> tuple[object, ...]:
        return (
            str(geometry_path.resolve()),
            _orientation_label(self._geometry_orientation_angles()),
            float(spacing),
            options.support_type,
            float(options.overhang_angle),
            float(options.pitch),
            float(options.thickness),
            float(options.footprint_offset),
            float(options.contact_depth),
            options.build_plate_z,
        )

    def _generated_support_stl_path(self, geometry_path: Path) -> Path:
        output_dir = Path(self._output_dir.text().strip() or "examples/outputs/gui_simulation")
        orientation = _orientation_label(self._geometry_orientation_angles())
        suffix = "" if orientation == "as_loaded" else f"_{orientation}"
        return output_dir / "generated_support" / f"{geometry_path.stem}{suffix}_support.stl"

    def _generated_support_preview_stl_path(self) -> Path:
        import tempfile

        return (
            Path(tempfile.gettempdir())
            / "virtual_pbf_workbench"
            / "active_generated_support.stl"
        )

    def _current_generated_support_signature_and_options(self):
        if self._part_type.currentText() != "Part & Support":
            return None
        if self._support_source.currentText() != "Generate from overhang":
            return None
        try:
            geometry_path = Path(self._part_geometry.text().strip())
            spacing = self._float(self._grid_spacing, "Grid spacing")
            options = self._support_generation_from_form()
            signature = self._support_generation_signature(geometry_path, spacing, options)
        except Exception:
            return None
        return signature, options

    def _active_generated_support_options(self):
        current = self._current_generated_support_signature_and_options()
        if current is None:
            return None
        signature, options = current
        if signature != getattr(self, "_last_generated_support_signature", None):
            return None
        support_path = getattr(self, "_last_generated_support_path", None)
        if support_path is None or not Path(support_path).exists():
            return None
        return options

    def _active_generated_support_path_and_options(self):
        options = self._active_generated_support_options()
        if options is None:
            return None
        return Path(self._last_generated_support_path), options

    def _generated_support_path_and_voxel_type(self):
        if self._part_type.currentText() != "Part & Support":
            return None
        if self._support_source.currentText() != "Generate from overhang":
            return None
        active = self._active_generated_support_path_and_options()
        if active is not None:
            support_path, options = active
            generated_type = (
                getattr(self, "_last_generated_support_type", None)
                or options.support_type
            )
        else:
            current = self._current_generated_support_signature_and_options()
            if current is None:
                return None
            signature, options = current
            support_text = self._support_geometry.text().strip()
            support_path = Path(support_text) if support_text else None
            if support_path is None or not support_path.exists():
                return None
            if signature != getattr(self, "_last_generated_support_signature", None):
                return None
            generated_type = options.support_type

        if generated_type is None:
            signature = getattr(self, "_last_generated_support_signature", None)
            if isinstance(signature, tuple) and len(signature) >= 3:
                generated_type = signature[3]
        if generated_type is None:
            try:
                generated_type = self._support_generation_from_form().support_type
            except Exception:
                generated_type = self._support_type.currentText()

        voxel_type = "Line support" if generated_type == "X surface support" else "Volume support"
        return Path(support_path), voxel_type

    def _cached_generated_support_grid_for_config(self, config):
        import numpy as np

        support_path = getattr(config, "support_geometry_path", None)
        if support_path is None:
            return None
        support_path = Path(support_path)
        if not support_path.exists():
            return None
        cached_path = getattr(self, "_last_generated_support_path", None)
        cached_grid = getattr(self, "_last_generated_support_grid", None)
        if cached_path is not None:
            if str(support_path.resolve()) != str(Path(cached_path).resolve()):
                return None
            try:
                if cached_grid is not None and np.isclose(
                    float(cached_grid.spacing),
                    float(config.voxel_spacing),
                ):
                    return cached_grid
            except Exception:
                pass
        else:
            support_source = getattr(self, "_support_source", None)
            if (
                support_source is not None
                and support_source.currentText() != "Generate from overhang"
            ):
                return None
        return None

    def _clear_support_selection(self) -> None:
        self._last_generated_support_grid = None
        self._last_generated_support_path = None
        self._last_generated_support_type = None
        self._last_generated_support_signature = None
        self._last_generated_support_options = None
        self._last_support_overlay_preview = None
        if hasattr(self, "_save_support_button"):
            self._save_support_button.setEnabled(False)
        self._support_geometry.blockSignals(True)
        self._support_geometry.setText("")
        self._support_geometry.blockSignals(False)
        if hasattr(self._part_type, "setCurrentText"):
            self._part_type.setCurrentText("Part only")
        self._invalidate_voxelization()
        self._append_log(
            "Support cleared. Part type is now Part only; switch back to Part & Support "
            "to generate or select support."
        )
        part_path = self._part_geometry.text().strip()
        if part_path and Path(part_path).exists():
            self._preview_part_geometry()

    def _generate_support_preview(self) -> None:
        if self._busy:
            self._append_log("A task is already in progress.")
            return
        if self._part_type.currentText() != "Part & Support":
            self._QMessageBox.warning(
                self._window,
                "Support disabled",
                "Select Part & Support before generating support.",
            )
            return
        if self._support_source.currentText() != "Generate from overhang":
            self._QMessageBox.warning(
                self._window,
                "External support selected",
                "Select Generate from overhang before generating support.",
            )
            return
        try:
            geometry_path = Path(self._part_geometry.text().strip())
            if not geometry_path.exists():
                raise ValueError("Select a valid part geometry STL file.")
            processing_geometry_path = self._part_geometry_processing_path()
            spacing = self._float(self._grid_spacing, "Grid spacing")
            options = self._support_generation_from_form()
            output_path = self._generated_support_preview_stl_path()
            request_signature = self._support_generation_signature(
                geometry_path,
                spacing,
                options,
            )
        except Exception as exc:
            self._QMessageBox.warning(self._window, "Invalid input", str(exc))
            return

        self._append_log(
            "Generating support preview "
            f"({options.support_type}, overhang <= {options.overhang_angle:g} deg, "
            f"contact overlap {options.contact_depth:g} mm)"
        )
        self._set_busy(True, "Generating support...", task="support_generation")
        worker = _GeneratedSupportWorker(
            request_signature,
            str(processing_geometry_path),
            spacing,
            options,
            str(output_path),
        )
        worker.signals.progress.connect(self._set_task_progress)
        worker.signals.finished.connect(self._generated_support_finished)
        worker.signals.failed.connect(self._generated_support_failed)
        self._generated_support_worker = worker
        self._thread_pool.start(worker)

    def _generated_support_finished(self, request_signature, path: str, support_grid) -> None:
        self._generated_support_worker = None
        try:
            geometry_path = Path(self._part_geometry.text().strip())
            processing_geometry_path = self._part_geometry_processing_path()
            spacing = self._float(self._grid_spacing, "Grid spacing")
            options = self._support_generation_from_form()
            is_stale = request_signature != self._support_generation_signature(
                geometry_path,
                spacing,
                options,
            )
        except Exception:
            is_stale = True
        if is_stale:
            self._set_busy(False, "Generated support discarded")
            self._append_log("Generated support discarded because the settings changed.")
            return

        self._last_generated_support_grid = support_grid
        self._last_generated_support_path = Path(path)
        self._last_generated_support_type = options.support_type
        self._last_generated_support_signature = request_signature
        self._last_generated_support_options = options
        self._support_geometry.blockSignals(True)
        self._support_geometry.setText(path)
        self._support_geometry.blockSignals(False)
        self._set_busy(False, "Support generated")
        if support_grid.filled_count <= 0:
            self._append_log("Generated support is empty.")
            self._preview.show_message(
                "No support voxels were generated for the current overhang settings."
            )
            return
        self._append_log(
            f"Generated support ready for preview: filled voxels={support_grid.filled_count}"
        )
        self._preview_support_overlay(processing_geometry_path, path)

    def _generated_support_failed(self, _request_signature, message: str) -> None:
        self._generated_support_worker = None
        self._last_generated_support_grid = None
        self._last_generated_support_type = None
        self._last_generated_support_signature = None
        self._last_generated_support_options = None
        if hasattr(self, "_save_support_button"):
            self._save_support_button.setEnabled(False)
        self._set_busy(False, "Support generation failed")
        self._append_log(f"Support generation failed: {message}")
        self._QMessageBox.critical(self._window, "Support generation failed", message)

    def _save_generated_support_stl(self) -> None:
        support_grid = getattr(self, "_last_generated_support_grid", None)
        if support_grid is None:
            self._QMessageBox.warning(
                self._window,
                "No generated support",
                "Generate support before saving a support STL.",
            )
            return
        try:
            from capp.io.exports import write_binary_stl, write_surface_stl

            geometry_path = Path(self._part_geometry.text().strip())
            output_path = self._generated_support_stl_path(geometry_path)
            options = (
                getattr(self, "_last_generated_support_options", None)
                or self._support_generation_from_form()
            )
            support_type = (
                getattr(self, "_last_generated_support_type", None)
                or options.support_type
            )
            if support_type == "X surface support":
                write_surface_stl(
                    output_path,
                    support_grid.data,
                    support_grid.spacing,
                    support_grid.origin,
                    bottom_z=options.build_plate_z,
                )
            else:
                write_binary_stl(
                    output_path,
                    support_grid.data,
                    support_grid.spacing,
                    support_grid.origin,
                    clip_min_z=options.build_plate_z,
                    voxel_bounds=True,
                )
        except Exception as exc:
            self._QMessageBox.critical(self._window, "Save support failed", str(exc))
            return

        self._append_log(f"Saved support STL: {output_path}")
        self._QMessageBox.information(
            self._window,
            "Support saved",
            f"Saved support STL:\n{output_path}",
        )

    def _save_voxel_grid(self) -> None:
        voxel_grid = getattr(self, "_last_voxel_grid", None)
        if voxel_grid is None:
            self._QMessageBox.warning(
                self._window,
                "No voxel grid",
                "Voxelize geometry before saving the current voxel grid.",
            )
            return
        try:
            import numpy as np

            from capp.io.exports import write_vtk_volume

            output_dir = (
                Path(self._output_dir.text().strip() or "examples/outputs/gui_simulation")
                / "voxelization"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            npz_path = output_dir / "voxel_grid.npz"
            voxel_path = output_dir / "voxel.vtk"
            support_path = output_dir / "support_mask.vtk"
            np.savez_compressed(
                npz_path,
                data=voxel_grid.data.astype(bool, copy=False),
                support_mask=voxel_grid.support_mask.astype(bool, copy=False),
                spacing=np.array([voxel_grid.spacing], dtype=np.float64),
                origin=np.array(voxel_grid.origin, dtype=np.float64),
            )
            write_vtk_volume(
                voxel_path,
                voxel_grid.data.astype(np.uint8, copy=False),
                voxel_grid.spacing,
                voxel_grid.origin,
                "Voxel",
            )
            write_vtk_volume(
                support_path,
                voxel_grid.support_mask.astype(np.uint8, copy=False),
                voxel_grid.spacing,
                voxel_grid.origin,
                "SupportMask",
            )
        except Exception as exc:
            self._QMessageBox.critical(self._window, "Save voxel grid failed", str(exc))
            return

        self._append_log(f"Saved voxel grid: {npz_path}")
        self._QMessageBox.information(
            self._window,
            "Voxel grid saved",
            f"Saved voxel grid:\n{npz_path}",
        )

    def _set_parameter_defaults(self, value: str) -> None:
        is_simple = value == "SimpleVN"
        is_directional = value == "DirectionalVN"
        is_moore = value == "SimpleM"

        self._coeff_x_neg.setEnabled(is_directional)
        self._coeff_x_pos.setEnabled(is_directional)
        self._coeff_y_neg.setEnabled(is_directional)
        self._coeff_y_pos.setEnabled(is_directional)
        self._coeff_current.setEnabled(is_simple or is_moore)
        self._coeff_moore_l.setEnabled(is_moore)
        self._coeff_moore_cl.setEnabled(is_moore)

        if is_simple:
            self._coeff_current.setText("0.2")
            self._coeff_lower.setText("1")
        elif is_directional:
            for field in [
                self._coeff_x_neg,
                self._coeff_x_pos,
                self._coeff_y_neg,
                self._coeff_y_pos,
            ]:
                field.setText("0.2")
            self._coeff_lower.setText("1")
        elif is_moore:
            self._coeff_current.setText("0.07")
            self._coeff_moore_l.setText("0.125")
            self._coeff_moore_cl.setText("1")

        self._residual_avg.setText("1E-5")
        self._residual_max.setText("1E-4")
        self._iteration_bound.setText("400")
        self._overwrap.setText("0.1")
        self._min_bias.setText("0.05")
        self._idp.setText("0.3")

    def _estimate_grid_spacing(self) -> None:
        path = self._part_geometry.text().strip()
        if not path:
            return
        processing_path = None
        try:
            processing_path = self._part_geometry_processing_path()
        except Exception:
            processing_path = None
        if processing_path is not None:
            path = str(processing_path)
        self._estimate_grid_spacing_for_path(path)

    def _estimate_grid_spacing_for_path(self, path: str) -> None:
        try:
            from capp.geometry.stl_stats import estimate_spacing_from_bounds, read_stl_stats

            stats = read_stl_stats(path)
            spacing = estimate_spacing_from_bounds(stats.bounds)
            self._grid_spacing.setText(str(spacing))
            self._append_log(
                f"Estimated grid spacing: {self._grid_spacing.text()} mm "
                f"({stats.triangle_count:,} triangles)"
            )
        except Exception as exc:
            self._append_log(f"Grid spacing estimate failed: {exc}")

    def _run_model_calibration(self) -> None:
        if self._busy:
            self._append_log("A task is already in progress.")
            return
        try:
            geometry_path = Path(self._calibration_geometry.text().strip())
            sample_dir = Path(self._calibration_sample_dir.text().strip())
            output_dir = Path(self._calibration_output_dir.text().strip())
            if not geometry_path.exists():
                raise ValueError("Select a valid calibration STL file.")
            if not sample_dir.exists():
                raise ValueError("Select a valid ROI sample folder.")
            if not str(output_dir):
                raise ValueError("Select a preset library.")
            spacing = self._float(self._calibration_spacing, "Grid spacing")
            sample_names = self._model_calibration_sample_filter()
            backend = self._selected_calibration_backend()
        except Exception as exc:
            self._QMessageBox.warning(self._window, "Invalid input", str(exc))
            return

        evaluations = int(self._calibration_max_evaluations.value())
        optimizer = self._calibration_optimizer.currentData() or "adaptive_sobol"
        parallel_samples = int(self._calibration_parallel_samples.value())
        self._append_log(
            "Running Model Calibration: "
            f"geometry={geometry_path}, samples={sample_dir}, evaluations={evaluations}, "
            f"optimizer={optimizer}, parallel_samples={parallel_samples}"
        )
        self._last_calibration_geometry_path = geometry_path
        self._last_calibration_log_progress = -10
        self._calibration_samples_label.setText("-")
        self._calibration_loss_label.setText("-")
        self._calibration_elapsed_label.setText("-")
        self._calibration_csv_label.setText("-")
        self._machine_map_outputs_label.setText("-")
        self._machine_map_status.setText("Waiting for Model Calibration result.")
        self._save_calibration_button.setEnabled(False)
        self._generate_machine_map_button.setEnabled(False)
        self._save_machine_map_button.setEnabled(False)
        self._export_research_artifacts_button.setEnabled(False)
        self._calibration_details.setPlainText("-")
        self._clear_calibration_comparison()
        self._calibration_progress_bar.setValue(0)
        self._calibration_progress_bar.setFormat("0%")
        self._calibration_progress_message.setText("Starting Model Calibration")
        self._set_model_calibration_progress(0, "Starting Model Calibration worker")
        self._set_busy(True, "Running Model Calibration...", task="model_calibration")
        from PySide6.QtWidgets import QApplication

        QApplication.processEvents()
        worker = _ModelCalibrationWorker(
            str(geometry_path),
            str(sample_dir),
            spacing,
            str(output_dir),
            evaluations,
            backend.value,
            optimizer,
            parallel_samples,
            sample_names,
        )
        worker.signals.progress.connect(self._set_model_calibration_progress)
        worker.signals.finished.connect(self._model_calibration_finished)
        worker.signals.failed.connect(self._model_calibration_failed)
        self._model_calibration_worker = worker
        self._thread_pool.start(worker)
        self._append_log(
            "Model Calibration worker queued "
            f"(active {self._thread_pool.activeThreadCount()}/"
            f"{self._thread_pool.maxThreadCount()})"
        )

    def _model_calibration_sample_filter(self) -> set[str] | None:
        text = self._calibration_sample_filter.text().strip()
        if not text:
            return None
        return {part.strip() for part in text.split(",") if part.strip()}

    def _model_calibration_run_configuration(self, output_dir: Path) -> dict[str, object]:
        backend_value = (
            self._calibration_processor.currentData()
            if hasattr(self, "_calibration_processor")
            else None
        )
        return {
            "format": "virtual_pbf.model_calibration_run.v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "geometry_path": str(
                self._last_calibration_geometry_path or self._calibration_geometry.text().strip()
            ),
            "roi_sample_folder": self._calibration_sample_dir.text().strip(),
            "sample_filter": self._calibration_sample_filter.text().strip(),
            "voxel_spacing_mm": self._calibration_spacing.text().strip(),
            "search_method": self._calibration_optimizer.currentData(),
            "evaluations": int(self._calibration_max_evaluations.value()),
            "parallel_samples": int(self._calibration_parallel_samples.value()),
            "solver": {
                "backend": backend_value,
                "label": self._solver_label(str(backend_value)) if backend_value else None,
            },
            "output_dir": str(output_dir),
            "machine_map_defaults": {
                "preset_name": self._machine_map_name.text().strip() or "Machine Map",
                "sp_coordinates": self._machine_map_coordinates.text().strip(),
                "contour_grid": int(self._machine_map_resolution.value()),
            },
        }

    def _machine_map_run_configuration(self, output_dir: Path, result) -> dict[str, object]:
        requested_preset_name = self._machine_map_name.text().strip() or "Machine Map"
        calibration_output_dir = _model_calibration_preset_output_dir(
            output_dir,
            requested_preset_name,
        )
        calibration_config = self._model_calibration_run_configuration(calibration_output_dir)
        return {
            "format": "virtual_pbf.machine_parameter_map_run.v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "requested_preset_name": requested_preset_name,
            "generated_preset_name": result.preset_name,
            "output_root": str(output_dir),
            "weights_csv": str(result.weights_csv) if result.weights_csv is not None else None,
            "sp_coordinates": str(result.coordinates_xlsx)
            if result.coordinates_xlsx is not None
            else self._machine_map_coordinates.text().strip(),
            "contour_grid": int(result.resolution),
            "voxel_spacing_mm": result.voxel_spacing,
            "source": (
                "in_memory_model_calibration"
                if self._last_calibration_result is not None
                else "saved_model_calibration_weights"
            ),
            "model_calibration": calibration_config,
        }

    def _recommended_parallel_samples(self) -> int:
        import os

        return max(1, min(4, os.cpu_count() or 1))

    def _set_model_calibration_progress(self, percent: int, message: str) -> None:
        value = max(0, min(100, int(percent)))
        self._set_task_progress(value, message)
        self._calibration_progress_bar.setValue(value)
        self._calibration_progress_bar.setFormat(f"{value}%")
        self._calibration_progress_message.setText(message)

        last_logged = getattr(self, "_last_calibration_log_progress", -10)
        if value >= last_logged + 10 or value in (0, 100):
            self._append_log(f"Model Calibration progress {value}%: {message}")
            self._last_calibration_log_progress = value

    def _model_calibration_finished(self, result) -> None:
        self._model_calibration_worker = None
        self._last_calibration_result = result
        self._last_machine_map_result = None
        self._last_machine_map_export_result = None
        self._calibration_samples_label.setText(str(len(result.samples)))
        self._calibration_progress_bar.setValue(100)
        self._calibration_progress_bar.setFormat("100%")
        self._calibration_progress_message.setText(
            f"Complete: load {result.target_load_seconds:.2f}s, "
            f"voxelize {result.voxelization_seconds:.2f}s, "
            f"solver CPU time {result.solver_seconds:.2f}s, "
            f"ROI/loss {result.roi_seconds + result.loss_seconds:.2f}s"
        )
        self._calibration_loss_label.setText(f"{result.average_loss:.6g}")
        self._calibration_elapsed_label.setText(f"{result.elapsed_seconds:.3f} s")
        self._calibration_csv_label.setText("Not saved")
        details = []
        for sample in result.samples:
            params = ", ".join(f"{value:.5g}" for value in sample.best.parameters.as_tuple())
            details.append(
                f"{sample.sample}: loss={sample.best.loss.total:.6g}, "
                f"dice=({sample.best.loss.x_dice:.3f}, {sample.best.loss.y_dice:.3f}), "
                f"wall={sample.elapsed_seconds:.2f}s, "
                f"solver={sample.solver_seconds:.2f}s, "
                f"ROI/loss={sample.roi_seconds + sample.loss_seconds:.2f}s, "
                f"params=[{params}], evals={sample.evaluations}"
            )
        self._calibration_details.setPlainText("\n".join(details) if details else "-")
        self._populate_calibration_comparison(result)
        self._append_log(f"Model Calibration complete: {len(result.samples)} sample(s)")
        self._append_log("Model Calibration result is ready in memory. Save outputs on request.")
        self._machine_map_status.setText(
            "Calibration result ready in memory. "
            "Generate the machine parameter map or save outputs."
        )
        self._save_calibration_button.setEnabled(True)
        self._generate_machine_map_button.setEnabled(True)
        self._save_machine_map_button.setEnabled(False)
        self._export_research_artifacts_button.setEnabled(True)
        self._set_busy(False, "Model Calibration complete")

    def _model_calibration_failed(self, message: str) -> None:
        self._model_calibration_worker = None
        self._calibration_progress_message.setText(message)
        self._append_log(f"Model Calibration failed: {message}")
        self._set_busy(False, "Model Calibration failed")
        self._QMessageBox.critical(self._window, "Model Calibration failed", message)

    def _save_model_calibration(self) -> None:
        if self._busy:
            self._append_log("A task is already in progress.")
            return
        result = self._last_calibration_result
        if result is None or not result.samples:
            self._QMessageBox.warning(
                self._window,
                "No calibration result",
                "Run Model Calibration before saving outputs.",
            )
            return
        try:
            output_root = Path(self._calibration_output_dir.text().strip())
            if not str(output_root):
                raise ValueError("Select a preset library.")
            preset_name = self._machine_map_name.text().strip() or "Machine Map"
            output_dir = _model_calibration_preset_output_dir(output_root, preset_name)
        except Exception as exc:
            self._QMessageBox.warning(self._window, "Invalid output", str(exc))
            return

        geometry_path = self._last_calibration_geometry_path or Path(
            self._calibration_geometry.text().strip()
        )
        run_configuration = self._model_calibration_run_configuration(output_dir)
        self._append_log(f"Saving Model Calibration outputs to: {output_dir}")
        self._set_busy(True, "Saving Model Calibration outputs...", task="calibration_save")
        worker = _SaveModelCalibrationWorker(
            str(output_dir),
            result,
            str(geometry_path),
            run_configuration,
        )
        worker.signals.progress.connect(self._set_model_calibration_progress)
        worker.signals.finished.connect(self._model_calibration_save_finished)
        worker.signals.failed.connect(self._model_calibration_save_failed)
        self._save_model_calibration_worker = worker
        self._thread_pool.start(worker)

    def _model_calibration_save_finished(self, output_dir: str, elapsed: float) -> None:
        from dataclasses import replace

        self._save_model_calibration_worker = None
        output_path = Path(output_dir)
        if self._last_calibration_result is not None:
            self._last_calibration_result = replace(
                self._last_calibration_result,
                output_dir=output_path,
                save_seconds=elapsed,
            )
        csv_path = output_path / "model_calibration_weights.csv"
        self._calibration_csv_label.setText(str(csv_path))
        self._calibration_progress_bar.setValue(100)
        self._calibration_progress_bar.setFormat("100%")
        self._calibration_progress_message.setText(
            f"Model Calibration outputs saved in {elapsed:.2f}s"
        )
        self._append_log(
            f"Model Calibration outputs saved: {csv_path}, "
            f"{output_path / 'run_configuration.json'}"
        )
        self._set_busy(False, "Model Calibration outputs saved")

    def _model_calibration_save_failed(self, message: str) -> None:
        self._save_model_calibration_worker = None
        self._append_log(f"Model Calibration save failed: {message}")
        self._set_busy(False, "Model Calibration save failed")
        self._QMessageBox.critical(self._window, "Model Calibration save failed", message)

    def _clear_calibration_comparison(self) -> None:
        self._calibration_comparison_data = {}
        self._calibration_overlay_source_pixmap = None
        if hasattr(self, "_calibration_comparison_sample"):
            self._calibration_comparison_sample.blockSignals(True)
            self._calibration_comparison_sample.clear()
            self._calibration_comparison_sample.blockSignals(False)
        if hasattr(self, "_calibration_overlay_label"):
            self._calibration_overlay_label.clear()
            self._calibration_overlay_label.setText("Run Model Calibration to compare ROI images.")

    def _populate_calibration_comparison(self, result) -> None:
        import numpy as np

        target_by_sample = {}
        try:
            from capp.calibration.model_calibration import discover_model_calibration_targets

            sample_dir = Path(self._calibration_sample_dir.text().strip())
            targets = discover_model_calibration_targets(sample_dir)
            target_by_sample = {target.sample: target for target in targets}
        except Exception as exc:
            self._append_log(f"ROI target reload skipped: {exc}")

        comparison = {}
        for sample in result.samples:
            target = target_by_sample.get(sample.sample)
            simulated_x = np.asarray(sample.best.simulated_x, dtype=bool)
            simulated_y = np.asarray(sample.best.simulated_y, dtype=bool)
            target_x_source = target.roi_x if target is not None else sample.best.target_x
            target_y_source = target.roi_y if target is not None else sample.best.target_y
            target_x_path = target.roi_x_path if target is not None else sample.best.target_x_path
            target_y_path = target.roi_y_path if target is not None else sample.best.target_y_path
            target_x = (
                self._resize_mask_to_shape(target_x_source, simulated_x.shape)
                if target_x_source is not None
                else np.zeros(simulated_x.shape, dtype=bool)
            )
            target_y = (
                self._resize_mask_to_shape(target_y_source, simulated_y.shape)
                if target_y_source is not None
                else np.zeros(simulated_y.shape, dtype=bool)
            )
            comparison[sample.sample] = {
                "X ROI": {
                    "target": target_x,
                    "simulated": self._resize_mask_to_shape(simulated_x, target_x.shape),
                    "original": self._read_roi_original_image(target_x_path, target_x.shape),
                },
                "Y ROI": {
                    "target": target_y,
                    "simulated": self._resize_mask_to_shape(simulated_y, target_y.shape),
                    "original": self._read_roi_original_image(target_y_path, target_y.shape),
                },
            }
        self._calibration_comparison_data = comparison

        self._calibration_comparison_sample.blockSignals(True)
        self._calibration_comparison_sample.clear()
        self._calibration_comparison_sample.addItems(list(comparison.keys()))
        self._calibration_comparison_sample.blockSignals(False)
        self._refresh_calibration_overlay()

    def _resize_mask_to_shape(self, mask, shape: tuple[int, int]):
        import numpy as np

        array = np.asarray(mask, dtype=bool)
        if array.shape == shape:
            return array
        if array.size == 0 or shape[0] <= 0 or shape[1] <= 0:
            return np.zeros(shape, dtype=bool)
        y_index = np.linspace(0, array.shape[0] - 1, shape[0]).round().astype(int)
        x_index = np.linspace(0, array.shape[1] - 1, shape[1]).round().astype(int)
        return array[np.ix_(y_index, x_index)]

    def _read_roi_original_image(self, path, shape: tuple[int, int]):
        if path is None:
            return None
        try:
            import numpy as np
            from PIL import Image

            with Image.open(path) as image:
                array = np.asarray(image.convert("L"))
            return _resize_array_nearest(array, shape)
        except Exception as exc:
            self._append_log(f"ROI original image load skipped: {exc}")
            return None

    def _refresh_calibration_overlay(self, *_args) -> None:
        if not hasattr(self, "_calibration_overlay_label"):
            return
        sample = self._calibration_comparison_sample.currentText()
        axis = self._calibration_comparison_axis.currentText()
        entry = self._calibration_comparison_data.get(sample, {}).get(axis)
        if entry is None:
            self._calibration_overlay_source_pixmap = None
            self._calibration_overlay_label.clear()
            self._calibration_overlay_label.setText("Run Model Calibration to compare ROI images.")
            return
        mode = (
            self._calibration_comparison_mode.currentText()
            if hasattr(self, "_calibration_comparison_mode")
            else "Overlay"
        )
        self._calibration_overlay_source_pixmap = self._roi_comparison_pixmap(entry, mode)
        self._fit_calibration_overlay_pixmap()

    def _roi_comparison_pixmap(self, entry, mode: str):
        import numpy as np
        from PySide6.QtGui import QImage, QPixmap

        target_array = np.asarray(entry["target"], dtype=bool)
        simulated_array = self._resize_mask_to_shape(entry["simulated"], target_array.shape)
        original_array = entry.get("original")
        if mode == "Outline":
            rgb = _roi_outline_rgb(target_array, simulated_array)
        elif mode == "Original + Outline":
            if original_array is None:
                rgb = _roi_outline_rgb(target_array, simulated_array)
            else:
                rgb = _roi_outline_rgb(target_array, simulated_array, original_array)
        elif mode == "Target Mask":
            rgb = _roi_mask_rgb(target_array, foreground=(44, 120, 84))
        elif mode == "Simulated Mask":
            rgb = _roi_mask_rgb(simulated_array, foreground=(35, 101, 158))
        else:
            rgb = _roi_overlay_rgb(target_array, simulated_array)
        height, width, _ = rgb.shape
        qimage = QImage(
            rgb.data,
            width,
            height,
            width * 3,
            QImage.Format.Format_RGB888,
        ).copy()
        return QPixmap.fromImage(qimage)

    def _fit_calibration_overlay_pixmap(self) -> None:
        if self._calibration_overlay_source_pixmap is None:
            return
        target_size = self._calibration_overlay_label.contentsRect().size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            target_size = self._calibration_overlay_label.size()
        self._calibration_overlay_label.setPixmap(
            self._calibration_overlay_source_pixmap.scaled(
                target_size,
                self._Qt.AspectRatioMode.KeepAspectRatio,
                self._Qt.TransformationMode.FastTransformation,
            )
        )

    def _export_calibration_research_artifacts(self) -> None:
        if self._busy:
            self._append_log("A task is already in progress.")
            return
        result = self._last_calibration_result
        if result is None or not result.samples:
            self._QMessageBox.warning(
                self._window,
                "No calibration result",
                "Run Model Calibration before exporting research artifacts.",
            )
            return
        output_root = Path(self._calibration_output_dir.text().strip())
        preset_name = self._machine_map_name.text().strip() or "Machine Map"
        output_dir = result.output_dir or _model_calibration_preset_output_dir(
            output_root,
            preset_name,
        )
        geometry_path = self._last_calibration_geometry_path or Path(
            self._calibration_geometry.text().strip()
        )
        self._append_log(f"Exporting Model Calibration research artifacts: {output_dir}")
        self._set_busy(True, "Exporting research artifacts...", task="research_export")
        worker = _ResearchArtifactExportWorker(str(output_dir), result, str(geometry_path))
        worker.signals.progress.connect(self._set_research_export_progress)
        worker.signals.finished.connect(self._research_artifacts_finished)
        worker.signals.failed.connect(self._research_artifacts_failed)
        self._research_artifact_worker = worker
        self._thread_pool.start(worker)

    def _set_research_export_progress(self, percent: int, message: str) -> None:
        self._set_task_progress(percent, message)
        self._calibration_progress_bar.setValue(max(0, min(100, int(percent))))
        self._calibration_progress_bar.setFormat(f"{max(0, min(100, int(percent)))}%")
        self._calibration_progress_message.setText(message)

    def _research_artifacts_finished(self, output_path) -> None:
        self._research_artifact_worker = None
        self._append_log(f"Research artifacts exported: {output_path}")
        self._calibration_progress_message.setText(f"Research artifacts exported: {output_path}")
        self._set_busy(False, "Research artifacts exported")

    def _research_artifacts_failed(self, message: str) -> None:
        self._research_artifact_worker = None
        self._append_log(f"Research artifact export failed: {message}")
        self._set_busy(False, "Research artifact export failed")
        self._QMessageBox.critical(self._window, "Research artifact export failed", message)

    def _generate_machine_map(self) -> None:
        if self._busy:
            self._append_log("A task is already in progress.")
            return
        try:
            calibration_result = self._last_calibration_result
            weights_csv = self._machine_map_weights_csv_path()
            coordinates_xlsx = Path(self._machine_map_coordinates.text().strip())
            if calibration_result is None and not weights_csv.exists():
                raise ValueError(
                    "Run Model Calibration first or select an output dir with saved weights."
                )
            if not coordinates_xlsx.exists():
                raise ValueError("Select a valid SP coordinate workbook.")
            resolution = int(self._machine_map_resolution.value())
            preset_name = self._machine_map_name.text().strip() or "Machine Map"
            voxel_spacing = self._float(self._calibration_spacing, "Grid spacing")
        except Exception as exc:
            self._QMessageBox.warning(self._window, "Invalid input", str(exc))
            return

        self._append_log(
            "Generating machine parameter map: "
            f"weights={weights_csv}, coordinates={coordinates_xlsx}, "
            f"resolution={resolution}, voxel_spacing={voxel_spacing:g}"
        )
        self._last_machine_map_result = None
        self._last_machine_map_export_result = None
        self._save_machine_map_button.setEnabled(False)
        self._machine_map_status.setText("Starting machine parameter map generation")
        self._set_busy(True, "Generating machine parameter map...", task="machine_map")
        worker = _MachineMapWorker(
            str(weights_csv),
            str(coordinates_xlsx),
            resolution,
            preset_name,
            voxel_spacing,
            calibration_result=calibration_result,
        )
        worker.signals.progress.connect(self._set_machine_map_progress)
        worker.signals.finished.connect(self._machine_map_finished)
        worker.signals.failed.connect(self._machine_map_failed)
        self._machine_map_worker = worker
        self._thread_pool.start(worker)

    def _machine_map_weights_csv_path(self) -> Path:
        output_root = Path(self._calibration_output_dir.text().strip())
        preset_name = self._machine_map_name.text().strip() or "Machine Map"
        preset_csv = (
            _model_calibration_preset_output_dir(output_root, preset_name)
            / "model_calibration_weights.csv"
        )
        legacy_csv = output_root / "model_calibration_weights.csv"
        if preset_csv.exists() or not legacy_csv.exists():
            return preset_csv
        return legacy_csv

    def _set_machine_map_progress(self, percent: int, message: str) -> None:
        value = max(0, min(100, int(percent)))
        self._set_task_progress(value, message)
        self._calibration_progress_bar.setValue(value)
        self._calibration_progress_bar.setFormat(f"{value}%")
        self._calibration_progress_message.setText(message)
        self._machine_map_status.setText(message)

    def _machine_map_finished(self, result) -> None:
        self._machine_map_worker = None
        self._last_machine_map_result = result
        self._last_machine_map_export_result = None
        library_root = Path(self._calibration_output_dir.text().strip())
        map_dir = _machine_map_preset_output_dir(library_root, result.preset_name)
        self._machine_map_outputs_label.setText(
            "\n".join(
                [
                    f"Preset: {result.preset_name}",
                    "State: In memory",
                    f"Target folder: {map_dir}",
                    "Files: Not saved",
                ]
            )
        )
        self._machine_map_status.setText(
            f"{result.preset_name}: {result.sample_count} samples, {result.resolution} x "
            f"{result.resolution} grid, {result.elapsed_seconds:.2f}s. Save to use as preset."
        )
        self._machine_map_preset_name.setText(f"{result.preset_name} (not saved)")
        self._set_machine_map_contour_data_from_result(result)
        self._append_log(
            "Machine parameter map complete: "
            "result is ready in memory. Save Machine Map to write preset files."
        )
        self._save_machine_map_button.setEnabled(True)
        self._set_busy(False, "Machine parameter map complete")

    def _machine_map_failed(self, message: str) -> None:
        self._machine_map_worker = None
        self._machine_map_status.setText(message)
        self._append_log(f"Machine parameter map failed: {message}")
        self._set_busy(False, "Machine parameter map failed")
        self._QMessageBox.critical(self._window, "Machine parameter map failed", message)

    def _save_machine_map(self) -> None:
        if self._busy:
            self._append_log("A task is already in progress.")
            return
        result = self._last_machine_map_result
        if result is None:
            self._QMessageBox.warning(
                self._window,
                "No machine map",
                "Generate a Machine Parameter Map before saving.",
            )
            return
        try:
            library_root = Path(self._calibration_output_dir.text().strip())
            if not str(library_root):
                raise ValueError("Select a preset library.")
            preset_folder = _machine_map_preset_output_dir(library_root, result.preset_name)
        except Exception as exc:
            self._QMessageBox.warning(self._window, "Invalid output", str(exc))
            return

        self._append_log(f"Saving Machine Parameter Map to: {preset_folder}")
        self._set_busy(True, "Saving Machine Parameter Map...", task="machine_map_save")
        worker = _SaveMachineMapWorker(
            str(library_root),
            str(preset_folder),
            result,
            self._machine_map_run_configuration(library_root, result),
        )
        worker.signals.progress.connect(self._set_machine_map_progress)
        worker.signals.finished.connect(self._machine_map_save_finished)
        worker.signals.failed.connect(self._machine_map_save_failed)
        self._save_machine_map_worker = worker
        self._thread_pool.start(worker)

    def _machine_map_save_finished(self, result) -> None:
        self._save_machine_map_worker = None
        self._last_machine_map_export_result = result
        self._machine_map_outputs_label.setText(
            "\n".join(
                [
                    f"Preset: {result.preset_name}",
                    f"Folder: {result.output_dir}",
                    f"Map: {result.map_npz}",
                    f"Metadata: {result.metadata_json}",
                    f"Configuration: {result.configuration_json}",
                    f"Grid: {result.grid_csv}",
                    f"Samples: {result.sample_csv}",
                ]
            )
        )
        self._machine_map_status.setText(
            f"{result.preset_name}: saved {result.sample_count} samples, "
            f"{result.resolution} x {result.resolution} grid"
        )
        self._machine_map_path.setText(str(result.map_npz))
        self._refresh_machine_preset_list(select_path=result.map_npz, load_contour=True)
        self._append_log(
            "Machine parameter map saved: "
            f"{result.map_npz}, {result.metadata_json}, "
            f"{result.configuration_json}, {result.grid_csv}, {result.sample_csv}"
        )
        self._append_log(f"Machine preset set to saved map: {result.preset_name}.")
        self._set_busy(False, "Machine parameter map saved")

    def _machine_map_save_failed(self, message: str) -> None:
        self._save_machine_map_worker = None
        self._machine_map_status.setText(message)
        self._append_log(f"Machine parameter map save failed: {message}")
        self._set_busy(False, "Machine parameter map save failed")
        self._QMessageBox.critical(self._window, "Machine parameter map save failed", message)

    def _set_machine_map_contour_data_from_result(self, result) -> None:
        import numpy as np

        if not hasattr(self, "_machine_map_contour_label"):
            return
        self._machine_map_contour_data = {
            name: np.asarray(result.grid[name], dtype=np.float64)
            for name in ("NX", "PX", "NY", "PY", "EPS", "IDP")
            if name in result.grid
        }
        sample_x = []
        sample_y = []
        coordinate_by_sample = {coord.sample: coord for coord in result.coordinates}
        for row in result.parameters:
            coordinate = coordinate_by_sample.get(row.sample)
            if coordinate is None:
                continue
            x_value, y_value = result.model.normalizer.normalize(coordinate.x, coordinate.y)
            sample_x.append(x_value)
            sample_y.append(y_value)
        self._machine_map_contour_data["_sample_x"] = np.asarray(sample_x, dtype=np.float64)
        self._machine_map_contour_data["_sample_y"] = np.asarray(sample_y, dtype=np.float64)
        self._refresh_machine_map_contour()

    def _load_machine_map_contour(self, path, silent: bool = False) -> None:
        import numpy as np

        if not hasattr(self, "_machine_map_contour_label"):
            return
        map_path = Path(path)
        if not map_path.exists():
            if not silent:
                self._machine_map_contour_label.clear()
                self._machine_map_contour_label.setText("Machine parameter map file not found.")
            return
        try:
            with np.load(map_path, allow_pickle=False) as data:
                self._machine_map_contour_data = {
                    name: np.asarray(data[name], dtype=np.float64)
                    for name in ("NX", "PX", "NY", "PY", "EPS", "IDP")
                    if name in data.files
                }
                if "sample_x_normalized" in data.files and "sample_y_normalized" in data.files:
                    self._machine_map_contour_data["_sample_x"] = np.asarray(
                        data["sample_x_normalized"], dtype=np.float64
                    )
                    self._machine_map_contour_data["_sample_y"] = np.asarray(
                        data["sample_y_normalized"], dtype=np.float64
                    )
                else:
                    self._machine_map_contour_data["_sample_x"] = np.asarray([], dtype=np.float64)
                    self._machine_map_contour_data["_sample_y"] = np.asarray([], dtype=np.float64)
        except Exception as exc:
            self._machine_map_contour_data = None
            self._machine_map_contour_source_pixmap = None
            self._machine_map_contour_label.clear()
            self._machine_map_contour_label.setText(f"Machine map preview failed: {exc}")
            return
        self._refresh_machine_map_contour()

    def _refresh_machine_map_contour(self, *_args) -> None:
        if not hasattr(self, "_machine_map_contour_label"):
            return
        if not self._machine_map_contour_data:
            self._machine_map_contour_source_pixmap = None
            self._machine_map_contour_label.clear()
            self._machine_map_contour_label.setText("Generate or load a machine parameter map.")
            return
        variable = self._machine_map_contour_variable.currentText()
        values = self._machine_map_contour_data.get(variable)
        if values is None:
            self._machine_map_contour_label.clear()
            self._machine_map_contour_label.setText(f"{variable} is not available in this map.")
            return
        self._machine_map_contour_source_pixmap = self._machine_map_contour_pixmap(values)
        self._fit_machine_map_contour_pixmap()

    def _machine_map_contour_pixmap(self, values):
        import numpy as np
        from PySide6.QtCore import QPointF, QRectF, Qt
        from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
        from scipy.ndimage import zoom
        from skimage.measure import find_contours

        array = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(array)
        if not finite.any():
            normalized = np.zeros(array.shape, dtype=np.float64)
        else:
            lo = float(np.nanmin(array[finite]))
            hi = float(np.nanmax(array[finite]))
            if hi <= lo:
                normalized = np.zeros(array.shape, dtype=np.float64)
            else:
                normalized = np.clip((array - lo) / (hi - lo), 0.0, 1.0)

        plot_width = 640
        plot_height = 360
        canvas_width = 760
        canvas_height = 430
        left = 46
        top = 26
        bar_left = left + plot_width + 24
        bar_width = 18
        safe_normalized = np.nan_to_num(normalized, nan=0.0)
        if safe_normalized.size:
            zoom_factors = (
                max(1.0, plot_height / safe_normalized.shape[0]),
                max(1.0, plot_width / safe_normalized.shape[1]),
            )
            display_values = zoom(safe_normalized, zoom_factors, order=1)
            display_values = display_values[:plot_height, :plot_width]
            if display_values.shape != (plot_height, plot_width):
                padded = np.zeros((plot_height, plot_width), dtype=np.float64)
                padded[: display_values.shape[0], : display_values.shape[1]] = display_values
                display_values = padded
        else:
            display_values = np.zeros((plot_height, plot_width), dtype=np.float64)
        rgb = np.ascontiguousarray(_workbench_colormap(np.flipud(display_values)))
        height, width, _ = rgb.shape
        qimage = QImage(
            rgb.data,
            width,
            height,
            width * 3,
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap(canvas_width, canvas_height)
        pixmap.fill(QColor(248, 250, 252))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.drawImage(left, top, qimage)

        painter.setPen(QPen(QColor(226, 232, 240), 1))
        for step in range(1, 5):
            x = left + step * plot_width / 5
            y = top + step * plot_height / 5
            painter.drawLine(QPointF(x, top), QPointF(x, top + plot_height))
            painter.drawLine(QPointF(left, y), QPointF(left + plot_width, y))

        painter.setPen(QPen(QColor(15, 23, 42), 1))
        painter.drawRect(left, top, plot_width, plot_height)

        contour_levels = np.linspace(0.15, 0.85, 6)
        fill_value = np.nanmean(normalized[finite]) if finite.any() else 0.0
        contour_source = np.nan_to_num(normalized, nan=fill_value)
        source_height, source_width = contour_source.shape
        if source_height > 1 and source_width > 1:
            for level in contour_levels:
                alpha = int(80 + 90 * level)
                pen = QPen(QColor(15, 23, 42, alpha), 1)
                painter.setPen(pen)
                for contour in find_contours(contour_source, level):
                    if len(contour) < 2:
                        continue
                    points = [
                        QPointF(
                            left + float(col) * plot_width / max(1, source_width - 1),
                            top + (1.0 - float(row) / max(1, source_height - 1)) * plot_height,
                        )
                        for row, col in contour
                    ]
                    painter.drawPolyline(QPolygonF(points))

        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QColor(30, 41, 59))
        variable = self._machine_map_contour_variable.currentText()
        title = f"{variable} parameter field"
        painter.drawText(QRectF(left, 4, plot_width, 18), Qt.AlignmentFlag.AlignLeft, title)
        painter.drawText(
            QRectF(left, top + plot_height + 8, plot_width, 18),
            Qt.AlignmentFlag.AlignCenter,
            "Normalized base plate coordinate",
        )

        bar_values = np.linspace(1.0, 0.0, plot_height)[:, None]
        bar_rgb = np.ascontiguousarray(
            _workbench_colormap(np.repeat(bar_values, bar_width, axis=1))
        )
        bar_qimage = QImage(
            bar_rgb.data,
            bar_width,
            plot_height,
            bar_width * 3,
            QImage.Format.Format_RGB888,
        ).copy()
        painter.drawImage(bar_left, top, bar_qimage)
        painter.setPen(QPen(QColor(15, 23, 42), 1))
        painter.drawRect(bar_left, top, bar_width, plot_height)
        if finite.any():
            painter.drawText(
                QRectF(bar_left + 24, top - 5, 70, 18),
                Qt.AlignmentFlag.AlignLeft,
                f"{hi:.3g}",
            )
            painter.drawText(
                QRectF(bar_left + 24, top + plot_height / 2 - 9, 70, 18),
                Qt.AlignmentFlag.AlignLeft,
                f"{((lo + hi) / 2):.3g}",
            )
            painter.drawText(
                QRectF(bar_left + 24, top + plot_height - 13, 70, 18),
                Qt.AlignmentFlag.AlignLeft,
                f"{lo:.3g}",
            )

        sample_x = self._machine_map_contour_data.get("_sample_x", [])
        sample_y = self._machine_map_contour_data.get("_sample_y", [])
        painter.setBrush(QColor(255, 255, 255, 230))
        painter.setPen(QPen(QColor(15, 23, 42), 2))
        for x_value, y_value in zip(sample_x, sample_y, strict=False):
            if not np.isfinite(x_value) or not np.isfinite(y_value):
                continue
            x = left + float(np.clip(x_value, 0.0, 1.0)) * plot_width
            y = top + (1.0 - float(np.clip(y_value, 0.0, 1.0))) * plot_height
            painter.drawEllipse(QPointF(x, y), 4.2, 4.2)
        painter.end()
        return pixmap

    def _fit_machine_map_contour_pixmap(self) -> None:
        if self._machine_map_contour_source_pixmap is None:
            return
        target_size = self._machine_map_contour_label.contentsRect().size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            target_size = self._machine_map_contour_label.size()
        self._machine_map_contour_label.setPixmap(
            self._machine_map_contour_source_pixmap.scaled(
                target_size,
                self._Qt.AspectRatioMode.KeepAspectRatio,
                self._Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _run_simulation(self) -> None:
        if self._busy:
            self._append_log("A virtual printing run is already in progress.")
            return
        try:
            config = self._simulation_config_from_form()
        except Exception as exc:
            self._QMessageBox.warning(self._window, "Invalid input", str(exc))
            return

        if self._last_voxel_grid is None:
            self._QMessageBox.warning(
                self._window,
                "Voxelization required",
                "Voxelize the geometry before running virtual printing.",
            )
            return
        if self._last_voxel_signature != self._voxel_signature(config):
            self._QMessageBox.warning(
                self._window,
                "Voxelization out of date",
                "Geometry or grid spacing changed. Voxelize again before running.",
            )
            return

        self._append_log(f"Running {config.geometry_path}")
        self._log_run_config(config)
        self._set_busy(True, "Running virtual printing...", task="simulation")
        worker = _SimulationWorker(config, self._last_voxel_grid)
        worker.signals.progress.connect(self._set_task_progress)
        worker.signals.finished.connect(self._simulation_finished)
        worker.signals.failed.connect(self._simulation_failed)
        self._simulation_worker = worker
        self._thread_pool.start(worker)

    def _simulation_finished(self, config, result) -> None:
        try:
            self._window.statusBar().showMessage("Rendering result...")
            self._last_result = result
            self._last_result_config = config
            self._set_loaded_result_from_simulation(result, config.output_dir)
            self._append_log("Simulation result is ready in memory. Save outputs on request.")
            self._shape_label.setText(" x ".join(str(v) for v in result.probability.shape))
            self._spacing_label.setText(f"{result.spacing:g}")
            self._rest_label.setText(f"{result.rest_volume:.3f}%")
            self._density_label.setText(f"{result.probability_density:.3f}%")
            outside_voxels = int((result.binary & ~result.voxel).sum())
            self._outside_label.setText(str(outside_voxels))
            self._elapsed_label.setText(f"{result.elapsed_seconds:.3f} s")
            self._preview_result_button.setEnabled(True)
            self._save_result_button.setEnabled(True)
            self._save_loaded_result_button.setEnabled(True)
            self._preview_result()
            self._preview_loaded_result(show_error=False)
            self._append_log("Complete")
            self._append_log(f"Out-of-CAD voxels: {outside_voxels}")
            self._navigation.setCurrentRow(1)
        except Exception as exc:
            self._simulation_failed(config, str(exc))
            return
        finally:
            self._simulation_worker = None

        self._set_busy(False, "Complete: result in memory")

    def _simulation_failed(self, _config, message: str) -> None:
        self._simulation_worker = None
        self._set_busy(False, "Simulation failed")
        self._append_log(f"Error: {message}")
        self._QMessageBox.critical(self._window, "Simulation failed", message)

    def _current_result_for_save(self):
        if self._loaded_result is None:
            if self._last_result is None:
                raise ValueError("Run or load a simulation result first.")
            return self._last_result

        import numpy as np

        from capp.domain import SimulationResult

        probability = np.asarray(self._loaded_result["probability"], dtype=np.uint8)
        binary = np.asarray(self._loaded_result["binary"], dtype=bool)
        voxel = np.asarray(self._loaded_result["voxel"], dtype=bool)
        support_mask = self._loaded_support_mask()
        rest_volume = float(self._loaded_result.get("rest_volume", 0.0))
        probability_density = float(self._loaded_result.get("probability_density", 0.0))
        if self._result_support_removed():
            probability = probability.copy()
            probability[support_mask] = 0
            binary = binary & ~support_mask
            voxel = voxel & ~support_mask
            support_mask = np.zeros_like(voxel, dtype=bool)
            rest_volume, probability_density = self._result_volume_metrics(
                probability,
                binary,
                voxel,
            )

        return SimulationResult(
            probability=probability,
            binary=binary,
            voxel=voxel,
            spacing=float(self._loaded_result["spacing"]),
            origin=tuple(float(v) for v in self._loaded_result["origin"]),
            rest_volume=rest_volume,
            probability_density=probability_density,
            elapsed_seconds=float(self._loaded_result.get("elapsed_seconds", 0.0)),
            source_geometry=self._loaded_result.get("source_geometry"),
            support_mask=support_mask,
        )

    def _result_volume_metrics(self, probability, binary, voxel) -> tuple[float, float]:
        import numpy as np

        solid_count = float(np.count_nonzero(voxel))
        if solid_count <= 0.0:
            return 0.0, 0.0
        rest_volume = 100.0 * float(np.count_nonzero(binary)) / solid_count
        probability_density = float((probability * voxel.astype(np.uint8)).sum() / solid_count)
        return rest_volume, probability_density

    def _save_outputs(self) -> None:
        if self._busy:
            self._append_log("A task is already in progress.")
            return
        if self._loaded_result is None:
            self._QMessageBox.warning(
                self._window,
                "Missing result",
                "Run or load a simulation result first.",
            )
            return
        try:
            result = self._current_result_for_save()
        except Exception as exc:
            self._QMessageBox.warning(self._window, "Invalid result", str(exc))
            return

        output_text = self._output_dir.text().strip()
        if output_text:
            output_dir = Path(output_text)
        elif self._last_result_config is not None:
            output_dir = self._last_result_config.output_dir
        else:
            output_dir = Path("examples/outputs/gui_simulation")
        suffix = " with support removed" if self._result_support_removed() else ""
        self._append_log(f"Saving outputs{suffix} to: {output_dir}")
        self._set_busy(True, "Saving outputs...", task="save")
        worker = _SaveOutputsWorker(output_dir, result)
        worker.signals.progress.connect(self._set_task_progress)
        worker.signals.finished.connect(self._save_outputs_finished)
        worker.signals.failed.connect(self._save_outputs_failed)
        self._save_outputs_worker = worker
        self._thread_pool.start(worker)

    def _save_outputs_finished(self, output_dir) -> None:
        self._save_outputs_worker = None
        output_path = Path(output_dir)
        result_path = output_path / "simulation_result.npz"
        if self._loaded_result is not None:
            self._loaded_result["path"] = result_path
        self._result_npz_path.setText(str(result_path))
        self._output_label.setText(str(output_path))
        self._files_label.setText(
            "simulation_result.npz\nprobability.vtk\nbinary.vtk\nsupport_mask.vtk"
        )
        self._append_log(
            "Saved simulation_result.npz, probability.vtk, binary.vtk, support_mask.vtk"
        )
        self._set_busy(False, f"Saved: {output_path}")

    def _save_outputs_failed(self, _output_dir, message: str) -> None:
        self._save_outputs_worker = None
        self._append_log(f"Output save failed: {message}")
        self._set_busy(False, "Output save failed")
        self._QMessageBox.critical(self._window, "Output save failed", message)

    def _set_busy(self, busy: bool, message: str, task: str = "") -> None:
        from PySide6.QtWidgets import QApplication

        self._busy = busy
        self._window.statusBar().showMessage(message)
        self._status_progress.setVisible(busy)

        if busy:
            self._status_progress.setRange(0, 100)
            self._status_progress.setValue(0)
            self._status_progress.setFormat("%p%")
            if not self._cursor_busy:
                QApplication.setOverrideCursor(self._Qt.CursorShape.WaitCursor)
                self._cursor_busy = True
        else:
            self._status_progress.setRange(0, 100)
            self._status_progress.setValue(0)
            if self._cursor_busy:
                QApplication.restoreOverrideCursor()
                self._cursor_busy = False

        self._run_button.setEnabled(not busy)
        self._run_button.setText(
            "Running..." if busy and task == "simulation" else "Run Virtual Printing"
        )
        self._voxelize_button.setEnabled(not busy)
        self._voxelize_button.setText(
            "Voxelizing..." if busy and task == "voxelization" else "Voxelize Geometry"
        )
        if hasattr(self, "_generate_support_button"):
            can_generate_support = (
                (not busy)
                and self._part_type.currentText() == "Part & Support"
                and self._support_source.currentText() == "Generate from overhang"
            )
            self._generate_support_button.setEnabled(can_generate_support)
            self._generate_support_button.setText(
                "Generating..." if busy and task == "support_generation" else "Generate"
            )
        if hasattr(self, "_clear_support_button"):
            self._clear_support_button.setEnabled(
                (not busy) and self._part_type.currentText() == "Part & Support"
            )
        if hasattr(self, "_save_support_button"):
            can_save_support = (
                (not busy)
                and self._part_type.currentText() == "Part & Support"
                and self._support_source.currentText() == "Generate from overhang"
                and self._last_generated_support_grid is not None
            )
            self._save_support_button.setEnabled(can_save_support)
            self._save_support_button.setText(
                "Saving..." if busy and task == "support_save" else "Save STL"
            )
        self._run_button.setEnabled((not busy) and self._last_voxel_grid is not None)
        self._preview_result_button.setEnabled((not busy) and self._last_result is not None)
        if hasattr(self, "_save_voxel_grid_button"):
            self._save_voxel_grid_button.setEnabled(
                (not busy) and self._last_voxel_grid is not None
            )
            self._save_voxel_grid_button.setText(
                "Saving..." if busy and task == "voxel_save" else "Save Voxel Grid"
            )

        can_save = (not busy) and self._loaded_result is not None
        if hasattr(self, "_save_result_button"):
            self._save_result_button.setEnabled(can_save)
            self._save_result_button.setText(
                "Saving..." if busy and task == "save" else "Save Outputs"
            )
        if hasattr(self, "_save_loaded_result_button"):
            self._save_loaded_result_button.setEnabled(can_save)
            self._save_loaded_result_button.setText(
                "Saving..." if busy and task == "save" else "Save Current Result"
            )
        self._update_result_action_state()
        if hasattr(self, "_deviation_button"):
            self._deviation_button.setText(
                "Computing..."
                if busy and task == "geometry_deviation"
                else "Show Deviation Heatmap"
            )
        if hasattr(self, "_run_calibration_button"):
            self._run_calibration_button.setEnabled(not busy)
            self._run_calibration_button.setText(
                "Running..." if busy and task == "model_calibration" else "Run Model Calibration"
            )
        if hasattr(self, "_save_calibration_button"):
            can_save_calibration = (
                (not busy)
                and self._last_calibration_result is not None
                and bool(self._last_calibration_result.samples)
            )
            self._save_calibration_button.setEnabled(can_save_calibration)
            self._save_calibration_button.setText(
                "Saving..." if busy and task == "calibration_save" else "Save Calibration"
            )
        if hasattr(self, "_generate_machine_map_button"):
            try:
                can_generate_map = (not busy) and (
                    (
                        self._last_calibration_result is not None
                        and bool(self._last_calibration_result.samples)
                    )
                    or self._machine_map_weights_csv_path().exists()
                )
            except Exception:
                can_generate_map = False
            self._generate_machine_map_button.setEnabled(can_generate_map)
            self._generate_machine_map_button.setText(
                "Generating..." if busy and task == "machine_map" else "Generate Machine Map"
            )
        if hasattr(self, "_save_machine_map_button"):
            can_save_machine_map = (not busy) and self._last_machine_map_result is not None
            self._save_machine_map_button.setEnabled(can_save_machine_map)
            self._save_machine_map_button.setText(
                "Saving..." if busy and task == "machine_map_save" else "Save Machine Map"
            )
        if hasattr(self, "_export_research_artifacts_button"):
            can_export_research = (
                (not busy)
                and self._last_calibration_result is not None
                and bool(self._last_calibration_result.samples)
            )
            self._export_research_artifacts_button.setEnabled(can_export_research)
            self._export_research_artifacts_button.setText(
                "Exporting..."
                if busy and task == "research_export"
                else "Export Research Artifacts"
            )

    def _set_task_progress(self, percent: int, message: str) -> None:
        value = max(0, min(100, int(percent)))
        self._status_progress.setValue(value)
        self._status_progress.setFormat(f"{value}%")
        self._window.statusBar().showMessage(f"{message} ({value}%)")

    def _invalidate_voxelization(self, *_args) -> None:
        if self._last_voxel_grid is None:
            self._update_preview_source_availability()
            return
        self._last_voxel_grid = None
        self._last_voxel_preview_data = None
        self._last_voxel_signature = None
        self._run_button.setEnabled(False)
        if hasattr(self, "_save_voxel_grid_button"):
            self._save_voxel_grid_button.setEnabled(False)
        self._voxel_status_label.setText("Required")
        self._update_preview_source_availability()
        self._append_log("Voxelization cleared. Run voxelization again before simulation.")

    def _voxel_signature(self, config) -> tuple[object, ...]:
        support_path = getattr(config, "support_geometry_path", None)
        support_generation = getattr(config, "support_generation", None)
        generation_signature = None
        if support_generation is not None:
            generation_signature = (
                support_generation.support_type,
                float(support_generation.overhang_angle),
                float(support_generation.pitch),
                float(support_generation.thickness),
                float(support_generation.footprint_offset),
                float(support_generation.contact_depth),
                support_generation.build_plate_z,
            )
        return (
            str(config.geometry_path.resolve()),
            "" if support_path is None else str(Path(support_path).resolve()),
            str(getattr(config, "support_type", "")),
            generation_signature,
            float(config.voxel_spacing),
        )

    def _simulation_config_from_form(self):
        from capp.config import SimulationConfig
        from capp.domain import (
            MachineBiasMode,
            MachineMapCoordinateMode,
            NeighborhoodModel,
            SolverParameters,
            StochasticMode,
        )

        source_geometry_path = Path(self._part_geometry.text().strip())
        if not source_geometry_path.exists():
            raise ValueError("Select a valid part geometry STL file.")
        geometry_path = self._part_geometry_processing_path()
        support_geometry_path = None
        support_type = "Volume support"
        support_generation = None
        if self._part_type.currentText() == "Part & Support":
            support_type = self._support_type.currentText()
            if self._support_source.currentText() == "External STL":
                support_geometry_path = Path(self._support_geometry.text().strip())
                if not support_geometry_path.exists():
                    raise ValueError("Select a valid support geometry STL file.")
            else:
                generated_support = self._generated_support_path_and_voxel_type()
                if generated_support is not None:
                    support_geometry_path, support_type = generated_support

        neighborhood_text = self._neighborhood.currentText()
        if neighborhood_text == "SimpleVN":
            neighborhood = NeighborhoodModel.SIMPLE_VON_NEUMANN
            current_coefficients = self._float(self._coeff_current, "Equivalent coefficient")
            lower_coefficients = self._float(self._coeff_lower, "Lower coefficient")
        elif neighborhood_text == "DirectionalVN":
            neighborhood = NeighborhoodModel.DIRECTIONAL_VON_NEUMANN
            current_coefficients = (
                self._float(self._coeff_x_neg, "-x Neumann"),
                self._float(self._coeff_x_pos, "+x Neumann"),
                self._float(self._coeff_y_neg, "-y Neumann"),
                self._float(self._coeff_y_pos, "+y Neumann"),
            )
            lower_coefficients = self._float(self._coeff_lower, "Lower coefficient")
        else:
            neighborhood = NeighborhoodModel.SIMPLE_MOORE
            current_coefficients = self._float(self._coeff_current, "Moore coefficient")
            lower_coefficients = (
                self._float(self._coeff_moore_l, "Moore L coefficient"),
                self._float(self._coeff_moore_cl, "Moore CL coefficient"),
            )

        stochastic_mode = (
            StochasticMode.IN_LAYER
            if self._stochastic_mode.currentText() == "In-layer"
            else StochasticMode.IN_VOLUME
        )
        selected_machine_map_path = self._selected_machine_map_path()
        machine_bias = (
            MachineBiasMode.NONE
            if selected_machine_map_path is None
            else MachineBiasMode.PRESET
        )
        machine_map_path = None
        machine_map_position = (0.0, 0.0)
        machine_map_coordinate_mode = MachineMapCoordinateMode.FULL_BASE_PLATE
        machine_map_bounds = None
        if machine_bias is MachineBiasMode.PRESET:
            machine_map_path = selected_machine_map_path
            if not machine_map_path.exists():
                raise ValueError("Select a valid machine parameter map .npz file.")
            coordinate_mode_text = self._machine_map_coordinate_mode.currentText()
            if coordinate_mode_text == "Explicit bounds":
                machine_map_coordinate_mode = MachineMapCoordinateMode.EXPLICIT_BOUNDS
                machine_map_bounds = (
                    self._float(self._machine_map_x_min, "Map X min"),
                    self._float(self._machine_map_x_max, "Map X max"),
                    self._float(self._machine_map_y_min, "Map Y min"),
                    self._float(self._machine_map_y_max, "Map Y max"),
                )
            elif coordinate_mode_text == "Part center":
                machine_map_coordinate_mode = MachineMapCoordinateMode.PART_CENTER
                machine_map_position = (
                    self._float(self._machine_map_x, "Part center X"),
                    self._float(self._machine_map_y, "Part center Y"),
                )
        backend = self._selected_solver_backend()

        solver = SolverParameters(
            neighborhood=neighborhood,
            current_coefficients=current_coefficients,
            lower_coefficients=lower_coefficients,
            residual_criteria=(
                self._float(self._residual_avg, "MAE residual"),
                self._float(self._residual_max, "MaxE residual"),
                1e-4,
                1e-3,
            ),
            overwrap_criterion=self._float(self._overwrap, "Overwrap criterion"),
            iteration_bound=self._int(self._iteration_bound, "Max iteration"),
            min_bias=self._float(self._min_bias, "Minimum bias"),
            stochastic_mode=stochastic_mode,
            machine_bias=machine_bias,
            machine_map_path=machine_map_path,
            machine_map_coordinate_mode=machine_map_coordinate_mode,
            machine_map_position=machine_map_position,
            machine_map_bounds=machine_map_bounds,
            initial_deviation=self._float(self._idp, "IDP value"),
            backend=backend,
            rng_seed=1000,
        )

        return SimulationConfig(
            geometry_path=geometry_path,
            output_dir=Path(self._output_dir.text().strip()),
            voxel_spacing=self._float(self._grid_spacing, "Grid spacing"),
            solver=solver,
            support_geometry_path=support_geometry_path,
            support_type=support_type,
            support_generation=support_generation,
        )

    def _log_run_config(self, config) -> None:
        solver = config.solver
        self._append_log(f"Geometry path used: {config.geometry_path}")
        if getattr(config, "support_geometry_path", None) is not None:
            self._append_log(
                f"Support path used: {config.support_geometry_path} ({config.support_type})"
            )
        support_generation = getattr(config, "support_generation", None)
        if support_generation is not None:
            self._append_log(
                "Generated support: "
                f"{support_generation.support_type}, "
                f"overhang <= {support_generation.overhang_angle:g} deg, "
                f"pitch {support_generation.pitch:g} mm, "
                f"thickness {support_generation.thickness:g} mm, "
                f"contact overlap {support_generation.contact_depth:g} mm"
            )
        self._append_log(f"Output directory: {config.output_dir}")
        self._append_log(f"Voxel spacing: {config.voxel_spacing:g} mm")
        self._append_log(f"Neighborhood: {solver.neighborhood.value}")
        self._append_log(f"Current coefficients: {solver.current_coefficients}")
        self._append_log(f"Lower coefficients: {solver.lower_coefficients}")
        self._append_log(f"Stochastic mode: {solver.stochastic_mode.value}")
        self._append_log(f"Machine preset: {solver.machine_bias.value}")
        if solver.machine_map_path is not None:
            self._append_log(f"Machine map: {solver.machine_map_path}")
            self._append_log(f"Machine map name: {self._machine_map_preset_name.text()}")
            self._append_log(f"Machine map coordinates: {solver.machine_map_coordinate_mode.value}")
            if solver.machine_map_bounds is not None:
                self._append_log(f"Machine map bounds: {solver.machine_map_bounds}")
            else:
                self._append_log(f"Machine map part center: {solver.machine_map_position}")
        self._append_log(f"Solver backend: {self._solver_label(solver.backend.value)}")

    def _float(self, field, label: str) -> float:
        try:
            return float(field.text().strip())
        except ValueError as exc:
            raise ValueError(f"{label} must be numeric.") from exc

    def _int(self, field, label: str) -> int:
        value = self._float(field, label)
        if value <= 0 or int(value) != value:
            raise ValueError(f"{label} must be a positive integer.")
        return int(value)

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._log.appendPlainText(f"[{timestamp}] {message}")
