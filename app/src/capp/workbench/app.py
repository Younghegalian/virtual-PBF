from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


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
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.signals = _VoxelizationWorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            from capp.geometry.voxelizer import voxelize_mesh

            def progress(percent: int, message: str) -> None:
                self.signals.progress.emit(percent, message)

            grid = voxelize_mesh(
                self.config.geometry_path,
                self.config.voxel_spacing,
                progress_callback=progress,
            )
        except Exception as exc:
            self.signals.failed.emit(self.config, str(exc))
            return

        self.signals.progress.emit(100, "Voxelization complete")
        self.signals.finished.emit(self.config, grid, grid.data)


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
                output_dir=self.output_dir,
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
            from capp.calibration import export_model_calibration_research_artifacts

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
        output_dir: str,
        resolution: int,
        preset_name: str,
        voxel_spacing: float,
    ) -> None:
        super().__init__()
        self.weights_csv = weights_csv
        self.coordinates_xlsx = coordinates_xlsx
        self.output_dir = output_dir
        self.resolution = resolution
        self.preset_name = preset_name
        self.voxel_spacing = voxel_spacing
        self.signals = _MachineMapWorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            from capp.machine_map import generate_machine_parameter_map_from_files

            def progress(percent: int, message: str) -> None:
                self.signals.progress.emit(percent, message)

            result = generate_machine_parameter_map_from_files(
                weights_csv=self.weights_csv,
                coordinates_xlsx=self.coordinates_xlsx,
                output_dir=self.output_dir,
                resolution=self.resolution,
                preset_name=self.preset_name,
                voxel_spacing=self.voxel_spacing,
                progress_callback=progress,
            )
        except Exception as exc:
            self.signals.failed.emit(str(exc))
            return

        self.signals.finished.emit(result)


class WorkbenchMainWindow:
    def __init__(self) -> None:
        from PySide6.QtCore import QThreadPool, Qt
        from PySide6.QtGui import QIcon
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

        from capp.workbench.branding import APP_NAME, app_icon_path

        self._window = QMainWindow()
        self._window.setWindowTitle(APP_NAME)
        self._window.setWindowIcon(QIcon(str(app_icon_path())))
        self._window.resize(1600, 960)

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
        self._navigation.setFixedWidth(204)
        self._navigation.setSpacing(0)
        self._navigation.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._navigation.currentRowChanged.connect(self._set_page)
        self._last_result = None
        self._last_result_config = None
        self._last_voxel_grid = None
        self._last_voxel_preview_data = None
        self._last_voxel_signature = None
        self._loaded_result = None
        self._busy = False
        self._cursor_busy = False
        self._simulation_worker = None
        self._voxelization_worker = None
        self._stl_preview_worker = None
        self._last_stl_preview = None
        self._save_outputs_worker = None
        self._model_calibration_worker = None
        self._research_artifact_worker = None
        self._machine_map_worker = None
        self._last_calibration_result = None
        self._last_calibration_geometry_path = None
        self._last_machine_map_result = None
        self._calibration_comparison_data = {}
        self._calibration_overlay_source_pixmap = None
        self._machine_map_contour_data = None
        self._machine_map_contour_source_pixmap = None
        self._thread_pool = QThreadPool.globalInstance()
        self._log = self._make_log()

        self._stack = QStackedWidget()
        self._stack.setObjectName("FeatureStack")
        self._add_feature("simulation", "Virtual Printing", self._build_simulation_page())
        self._add_feature("results", "Result Display", self._build_results_page())
        self._add_feature("calibration", "Model Calibration", self._build_lab_page())
        self._add_feature("data", "Data & Models", self._build_placeholder_page("Data & Models"))
        self._add_feature("settings", "Preferences", self._build_placeholder_page("Preferences"))

        splitter = QSplitter(self._Qt.Orientation.Horizontal)
        splitter.setObjectName("MainSplitter")
        splitter.addWidget(self._navigation)
        splitter.addWidget(self._stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

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

    def show(self, maximized: bool = False) -> None:
        if maximized:
            self._window.showMaximized()
        else:
            self._window.show()

    def _add_feature(self, icon_key: str, title: str, widget: object) -> None:
        from PySide6.QtWidgets import QListWidgetItem

        item = QListWidgetItem(self._icons[icon_key], title)
        self._navigation.addItem(item)
        self._stack.addWidget(widget)

    def _apply_style(self) -> None:
        self._window.setStyleSheet(
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
                background: #f4f7fa;
                border-right: 1px solid #aeb8c4;
                padding: 4px;
            }
            #Navigation::item {
                min-height: 25px;
                padding: 2px 7px;
                border-radius: 1px;
                margin: 0;
            }
            #Navigation::item:selected {
                background: #dbe6f2;
                border-left: 3px solid #2563eb;
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
            ("data", "Data & Models", 3),
            ("settings", "Preferences", 4),
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

        self._support_geometry = QLineEdit()
        geometry_form.addRow(
            "Support geometry",
            self._file_row(self._support_geometry, self._browse_support),
        )

        self._support_type = QComboBox()
        self._support_type.addItems(["Line support", "Volume support"])
        self._support_type.setCurrentText("Volume support")
        geometry_form.addRow("Support type", self._support_type)

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
        self._machine_preset.addItems(["None", "Machine Map"])
        self._machine_preset.currentTextChanged.connect(self._update_machine_preset_controls)
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
        self._machine_map_path.editingFinished.connect(self._refresh_machine_map_name)
        self._refresh_machine_map_name()
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
        self._update_machine_preset_controls()

        self._processor = QComboBox()
        self._processor.currentIndexChanged.connect(self._update_backend_status_label)
        self._processor_status = QLabel()
        self._processor_status.setWordWrap(True)
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
        left.setMinimumWidth(400)
        left.setMaximumWidth(520)
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

        self._slice_axis = QComboBox()
        self._slice_axis.addItems(["Z", "X", "Y"])
        self._slice_axis.currentTextChanged.connect(self._update_result_slice)
        form.addRow("Slice axis", self._slice_axis)

        self._slice_slider = QSlider(Qt.Orientation.Horizontal)
        self._slice_slider.valueChanged.connect(self._update_result_slice)
        form.addRow("Slice", self._slice_slider)
        result_layout.addWidget(controls_box)

        preview_button = QPushButton("Preview 3D")
        preview_button.clicked.connect(self._preview_loaded_result)
        result_layout.addWidget(preview_button)

        self._slice_label = QLabel("-")
        self._slice_label.setObjectName("SliceView")
        self._slice_label.setMinimumSize(360, 300)
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
        splitter.setSizes([460, 1220])
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
        self._calibration_geometry = QLineEdit()
        input_form.addRow(
            "Calibration STL",
            self._file_row(self._calibration_geometry, self._browse_calibration_geometry),
        )
        self._calibration_sample_dir = QLineEdit()
        input_form.addRow(
            "ROI sample folder",
            self._file_row(self._calibration_sample_dir, self._browse_calibration_sample_dir),
        )
        self._calibration_spacing = QLineEdit("0.5")
        input_form.addRow("Grid spacing (mm)", self._calibration_spacing)
        self._calibration_sample_filter = QLineEdit()
        input_form.addRow("Sample filter", self._calibration_sample_filter)
        self._calibration_output_dir = QLineEdit("examples/outputs/model_calibration")
        input_form.addRow(
            "Output dir",
            self._file_row(self._calibration_output_dir, self._browse_calibration_output_dir),
        )
        left_layout.addWidget(input_box)

        solver_box = QGroupBox("Calibration Solver")
        solver_form = QFormLayout(solver_box)
        self._configure_form(solver_form)
        self._calibration_optimizer = QComboBox()
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
        self._calibration_processor_status.setWordWrap(True)
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
        self._machine_map_status = QLabel("Waiting for Model Calibration weights.")
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
        self._generate_machine_map_button = QPushButton("Generate Machine Map")
        self._generate_machine_map_button.setEnabled(False)
        self._generate_machine_map_button.clicked.connect(self._generate_machine_map)
        action_row.addWidget(self._generate_machine_map_button)
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
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(3, 3, 3, 3)
        right_layout.setSpacing(3)
        right_layout.addWidget(self._page_title("Calibration Review"))

        comparison_box = QGroupBox("ROI Comparison")
        comparison_layout = QVBoxLayout(comparison_box)
        comparison_layout.setContentsMargins(4, 4, 4, 4)
        comparison_layout.setSpacing(4)
        comparison_controls = QHBoxLayout()
        comparison_controls.setContentsMargins(0, 0, 0, 0)
        comparison_controls.setSpacing(4)
        self._calibration_comparison_sample = QComboBox()
        self._calibration_comparison_sample.currentTextChanged.connect(
            self._refresh_calibration_overlay
        )
        self._calibration_comparison_axis = QComboBox()
        self._calibration_comparison_axis.addItems(["X ROI", "Y ROI"])
        self._calibration_comparison_axis.currentTextChanged.connect(
            self._refresh_calibration_overlay
        )
        comparison_controls.addWidget(self._calibration_comparison_sample, 2)
        comparison_controls.addWidget(self._calibration_comparison_axis, 1)
        comparison_layout.addLayout(comparison_controls)
        self._calibration_overlay_label = QLabel("Run Model Calibration to compare ROI images.")
        self._calibration_overlay_label.setObjectName("SliceView")
        self._calibration_overlay_label.setAlignment(self._Qt.AlignmentFlag.AlignCenter)
        self._calibration_overlay_label.setMinimumHeight(280)
        comparison_layout.addWidget(self._calibration_overlay_label, 1)
        right_layout.addWidget(comparison_box, 1)

        contour_box = QGroupBox("Machine Map Contour")
        contour_layout = QVBoxLayout(contour_box)
        contour_layout.setContentsMargins(4, 4, 4, 4)
        contour_layout.setSpacing(4)
        self._machine_map_contour_variable = QComboBox()
        self._machine_map_contour_variable.addItems(["NX", "PX", "NY", "PY", "EPS", "IDP"])
        self._machine_map_contour_variable.currentTextChanged.connect(
            self._refresh_machine_map_contour
        )
        contour_layout.addWidget(self._machine_map_contour_variable)
        self._machine_map_contour_label = QLabel("Generate or load a machine parameter map.")
        self._machine_map_contour_label.setObjectName("SliceView")
        self._machine_map_contour_label.setAlignment(self._Qt.AlignmentFlag.AlignCenter)
        self._machine_map_contour_label.setMinimumHeight(260)
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

    def _file_row(self, line_edit, callback):
        from PySide6.QtWidgets import QHBoxLayout, QPushButton

        button = QPushButton("Open")
        button.setIcon(self._icons["open"])
        button.clicked.connect(callback)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(line_edit)
        row.addWidget(button)
        return row

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

    def _browse_part(self) -> None:
        path = self._open_file("Select part geometry", "STL files (*.stl);;All files (*.*)")
        if path:
            self._part_geometry.setText(path)
            self._append_log(f"Part geometry selected: {path}")
            self._estimate_grid_spacing_for_path(path)
            self._preview_part_geometry()

    def _browse_support(self) -> None:
        path = self._open_file("Select support geometry", "STL files (*.stl);;All files (*.*)")
        if path:
            self._support_geometry.setText(path)
            self._append_log(f"Support geometry selected: {path}")

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
            "Select Model Calibration output directory",
            str(Path.cwd()),
        )
        if path:
            self._calibration_output_dir.setText(path)
            self._append_log(f"Model Calibration output directory selected: {path}")

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
            self._machine_preset.setCurrentText("Machine Map")
            self._refresh_machine_map_name()
            self._append_log(f"Machine parameter map selected: {path}")

    def _update_machine_map_coordinate_fields(self) -> None:
        coordinate_mode = self._machine_map_coordinate_mode.currentText()
        center_mode = coordinate_mode == "Part center"
        bounds_mode = coordinate_mode == "Explicit bounds"
        for widget in self._machine_map_center_widgets:
            widget.setEnabled(center_mode)
        for widget in self._machine_map_bounds_widgets:
            widget.setEnabled(bounds_mode)

    def _refresh_machine_map_name(self) -> None:
        path = Path(self._machine_map_path.text().strip())
        if not path.exists():
            self._machine_map_preset_name.setText("-")
            return
        try:
            from capp.machine_map import read_machine_parameter_map_metadata

            metadata = read_machine_parameter_map_metadata(path)
            label = metadata.preset_name
            if metadata.voxel_spacing is not None:
                label = f"{label} ({metadata.voxel_spacing:g} mm)"
            self._machine_map_preset_name.setText(label)
            if self._machine_preset.currentText() == "Machine Map":
                self._apply_machine_map_spacing(metadata.voxel_spacing)
            if hasattr(self, "_machine_map_contour_label"):
                self._load_machine_map_contour(path, silent=True)
        except Exception as exc:
            self._machine_map_preset_name.setText(f"Unreadable map: {exc}")

    def _update_machine_preset_controls(self, *_args) -> None:
        machine_map_active = self._machine_preset.currentText() == "Machine Map"
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

        self._apply_machine_preset_lock_state(machine_map_active)

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

    def _default_machine_coordinate_path(self) -> Path:
        project_root = Path(__file__).resolve().parents[4]
        path = project_root / "data" / "machine_map" / "sp_coordinates.xlsx"
        if path.exists():
            return path
        return Path("..") / "data" / "machine_map" / "sp_coordinates.xlsx"

    def _default_machine_map_path(self) -> Path:
        path = (
            Path("examples")
            / "outputs"
            / "model_calibration"
            / "machine_presets"
            / "Machine_Map"
            / "machine_parameter_map.npz"
        )
        if path.exists():
            return path
        legacy = Path("examples") / "outputs" / "model_calibration" / "machine_parameter_map.npz"
        if legacy.exists():
            return legacy
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
        self._last_stl_preview = None
        self._set_preview_source("STL")
        self._append_log(f"Preparing STL preview: {path}")
        self._preview.show_message("Loading STL preview...")
        worker = _StlPreviewWorker(path)
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
        if self._last_stl_preview is None or not hasattr(self, "_preview"):
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
        current_path = self._part_geometry.text().strip()
        if str(Path(path).resolve()) != str(Path(current_path).resolve()):
            self._append_log("STL preview discarded because the selected geometry changed.")
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
        current_path = self._part_geometry.text().strip()
        if current_path and str(Path(path).resolve()) != str(Path(current_path).resolve()):
            return
        self._append_log(f"STL preview failed: {message}")
        self._preview.show_message(f"STL preview failed: {message}")
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
        worker = _VoxelizationWorker(config)
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
        self._append_log(f"Voxel grid ready: shape={grid.shape}, filled={grid.filled_count}")

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

    def _load_result_npz(self, path: str | Path) -> None:
        try:
            import numpy as np

            data = np.load(path)
            self._loaded_result = {
                "probability": data["probability"],
                "binary": data["binary"].astype(bool),
                "voxel": data["voxel"].astype(bool),
                "spacing": float(data["spacing"][0]),
                "origin": tuple(float(v) for v in data["origin"]),
                "path": Path(path),
            }
            self._output_label.setText(str(Path(path).parent))
            self._files_label.setText("\n".join(data.files))
            self._append_log(f"Loaded result NPZ: {path}")
            self._refresh_result_views()
        except Exception as exc:
            self._append_log(f"Result load failed: {exc}")
            self._QMessageBox.critical(self._window, "Result load failed", str(exc))

    def _set_loaded_result_from_simulation(self, result, output_dir: Path) -> None:
        self._loaded_result = {
            "probability": result.probability,
            "binary": result.binary,
            "voxel": result.voxel,
            "spacing": result.spacing,
            "origin": result.origin,
            "path": None,
        }
        self._result_npz_path.setText("")
        self._output_label.setText(f"In memory; target: {output_dir}")
        self._files_label.setText("Not saved")
        self._refresh_result_views()

    def _refresh_result_views(self, *_args) -> None:
        if self._loaded_result is None:
            return
        volume = self._selected_result_volume()
        axis = self._slice_axis.currentText()
        axis_index = {"X": 0, "Y": 1, "Z": 2}[axis]
        self._slice_slider.blockSignals(True)
        self._slice_slider.setMinimum(0)
        self._slice_slider.setMaximum(max(0, volume.shape[axis_index] - 1))
        self._slice_slider.setValue(volume.shape[axis_index] // 2)
        self._slice_slider.blockSignals(False)
        self._update_result_slice()

    def _preview_loaded_result(self) -> None:
        if self._loaded_result is None:
            self._QMessageBox.warning(self._window, "Missing result", "Load or run a result first.")
            return
        volume = self._selected_result_volume()
        name = self._result_volume_choice.currentText()
        if name == "Probability":
            preview_volume = volume
            label = "Probability"
        else:
            preview_volume = volume.astype(bool)
            label = name
        self._append_log(f"Previewing result volume: {label}")
        self._result_preview.show_voxels(
            preview_volume,
            spacing=self._loaded_result["spacing"],
            origin=self._loaded_result["origin"],
            label=label,
        )

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
        self._slice_label.setPixmap(self._array_to_pixmap(image))

    def _selected_result_volume(self):
        if self._loaded_result is None:
            raise ValueError("No result loaded.")
        choice = self._result_volume_choice.currentText()
        if choice.startswith("Binary"):
            return self._loaded_result["binary"]
        return self._loaded_result["probability"]

    def _array_to_pixmap(self, image):
        import numpy as np
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QImage, QPixmap

        array = np.asarray(image)
        if array.dtype == bool:
            array = array.astype(np.uint8) * 255
        else:
            array = array.astype(np.float32)
            if array.size and array.max() > array.min():
                array = 255.0 * (array - array.min()) / (array.max() - array.min())
            array = array.astype(np.uint8)

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
        return pixmap.scaled(
            self._slice_label.size(),
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
            return
        state = "Available" if status.available else "Unavailable"
        self._processor_status.setText(f"{state}. {status.detail}")

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
            return
        state = "Available" if status.available else "Unavailable"
        self._calibration_processor_status.setText(f"{state}. {status.detail}")

    def _selected_calibration_backend(self):
        from capp.domain import SolverBackend

        return SolverBackend(
            self._calibration_processor.currentData() or SolverBackend.CPU_REFERENCE.value
        )

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
        enabled = value == "Part & Support"
        self._support_geometry.setEnabled(enabled)
        self._support_type.setEnabled(enabled)

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
                raise ValueError("Select an output directory.")
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
        self._machine_map_status.setText("Waiting for Model Calibration weights.")
        self._generate_machine_map_button.setEnabled(False)
        self._export_research_artifacts_button.setEnabled(False)
        self._calibration_details.setText("-")
        self._clear_calibration_comparison()
        self._calibration_progress_bar.setValue(0)
        self._calibration_progress_bar.setFormat("0%")
        self._calibration_progress_message.setText("Starting Model Calibration")
        self._set_busy(True, "Running Model Calibration...", task="model_calibration")
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

    def _model_calibration_sample_filter(self) -> set[str] | None:
        text = self._calibration_sample_filter.text().strip()
        if not text:
            return None
        return {part.strip() for part in text.split(",") if part.strip()}

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
        output_dir = result.output_dir or Path(self._calibration_output_dir.text().strip())
        csv_path = output_dir / "model_calibration_weights.csv"
        self._calibration_samples_label.setText(str(len(result.samples)))
        self._calibration_progress_bar.setValue(100)
        self._calibration_progress_bar.setFormat("100%")
        self._calibration_progress_message.setText(
            f"Complete: load {result.target_load_seconds:.2f}s, "
            f"voxelize {result.voxelization_seconds:.2f}s, "
            f"solver CPU time {result.solver_seconds:.2f}s, "
            f"ROI/loss {result.roi_seconds + result.loss_seconds:.2f}s, "
            f"save {result.save_seconds:.2f}s"
        )
        self._calibration_loss_label.setText(f"{result.average_loss:.6g}")
        self._calibration_elapsed_label.setText(f"{result.elapsed_seconds:.3f} s")
        self._calibration_csv_label.setText(str(csv_path))
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
        self._calibration_details.setText("\n".join(details) if details else "-")
        self._populate_calibration_comparison(result)
        self._append_log(f"Model Calibration complete: {len(result.samples)} sample(s)")
        self._append_log(f"Model Calibration weights: {csv_path}")
        self._machine_map_status.setText("Weights ready. Generate the machine parameter map.")
        self._generate_machine_map_button.setEnabled(True)
        self._export_research_artifacts_button.setEnabled(True)
        self._set_busy(False, "Model Calibration complete")

    def _model_calibration_failed(self, message: str) -> None:
        self._model_calibration_worker = None
        self._calibration_progress_message.setText(message)
        self._append_log(f"Model Calibration failed: {message}")
        self._set_busy(False, "Model Calibration failed")
        self._QMessageBox.critical(self._window, "Model Calibration failed", message)

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
            if target is None:
                target_x = np.zeros(simulated_x.shape, dtype=bool)
                target_y = np.zeros(simulated_y.shape, dtype=bool)
            else:
                target_x = self._resize_mask_to_shape(target.roi_x, simulated_x.shape)
                target_y = self._resize_mask_to_shape(target.roi_y, simulated_y.shape)
            comparison[sample.sample] = {
                "X ROI": (target_x, simulated_x),
                "Y ROI": (target_y, simulated_y),
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

    def _refresh_calibration_overlay(self, *_args) -> None:
        if not hasattr(self, "_calibration_overlay_label"):
            return
        sample = self._calibration_comparison_sample.currentText()
        axis = self._calibration_comparison_axis.currentText()
        pair = self._calibration_comparison_data.get(sample, {}).get(axis)
        if pair is None:
            self._calibration_overlay_source_pixmap = None
            self._calibration_overlay_label.clear()
            self._calibration_overlay_label.setText("Run Model Calibration to compare ROI images.")
            return
        self._calibration_overlay_source_pixmap = self._roi_overlay_pixmap(*pair)
        self._fit_calibration_overlay_pixmap()

    def _roi_overlay_pixmap(self, target, simulated):
        import numpy as np
        from PySide6.QtGui import QImage, QPixmap

        target_array = np.asarray(target, dtype=bool)
        simulated_array = self._resize_mask_to_shape(simulated, target_array.shape)
        rgb = np.full((*target_array.shape, 3), 248, dtype=np.uint8)
        target_only = target_array & ~simulated_array
        simulated_only = simulated_array & ~target_array
        overlap = target_array & simulated_array
        rgb[target_only] = (224, 132, 58)
        rgb[simulated_only] = (56, 116, 196)
        rgb[overlap] = (62, 153, 101)
        rgb = np.ascontiguousarray(np.flipud(rgb))
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
        self._calibration_overlay_label.setPixmap(
            self._calibration_overlay_source_pixmap.scaled(
                self._calibration_overlay_label.size(),
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
        output_dir = result.output_dir or Path(self._calibration_output_dir.text().strip())
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
            weights_csv = self._machine_map_weights_csv_path()
            coordinates_xlsx = Path(self._machine_map_coordinates.text().strip())
            output_dir = Path(self._calibration_output_dir.text().strip())
            if not weights_csv.exists():
                raise ValueError(
                    "Run Model Calibration first or select an output dir with weights."
                )
            if not coordinates_xlsx.exists():
                raise ValueError("Select a valid SP coordinate workbook.")
            if not str(output_dir):
                raise ValueError("Select an output directory.")
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
        self._machine_map_status.setText("Starting machine parameter map generation")
        self._set_busy(True, "Generating machine parameter map...", task="machine_map")
        worker = _MachineMapWorker(
            str(weights_csv),
            str(coordinates_xlsx),
            str(output_dir),
            resolution,
            preset_name,
            voxel_spacing,
        )
        worker.signals.progress.connect(self._set_machine_map_progress)
        worker.signals.finished.connect(self._machine_map_finished)
        worker.signals.failed.connect(self._machine_map_failed)
        self._machine_map_worker = worker
        self._thread_pool.start(worker)

    def _machine_map_weights_csv_path(self) -> Path:
        output_dir = Path(self._calibration_output_dir.text().strip())
        return output_dir / "model_calibration_weights.csv"

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
        self._machine_map_outputs_label.setText(
            "\n".join(
                [
                    f"Preset: {result.preset_name}",
                    f"Folder: {result.output_dir}",
                    f"Map: {result.map_npz}",
                    f"Metadata: {result.metadata_json}",
                    f"Grid: {result.grid_csv}",
                    f"Samples: {result.sample_csv}",
                ]
            )
        )
        self._machine_map_status.setText(
            f"{result.preset_name}: {result.sample_count} samples, {result.resolution} x "
            f"{result.resolution} grid, {result.elapsed_seconds:.2f}s"
        )
        self._machine_map_path.setText(str(result.map_npz))
        self._machine_preset.setCurrentText("Machine Map")
        self._machine_map_preset_name.setText(result.preset_name)
        self._load_machine_map_contour(result.map_npz)
        self._append_log(
            "Machine parameter map complete: "
            f"{result.map_npz}, {result.metadata_json}, {result.grid_csv}, {result.sample_csv}"
        )
        self._append_log(f"Machine preset set to generated map: {result.preset_name}.")
        self._set_busy(False, "Machine parameter map complete")

    def _machine_map_failed(self, message: str) -> None:
        self._machine_map_worker = None
        self._machine_map_status.setText(message)
        self._append_log(f"Machine parameter map failed: {message}")
        self._set_busy(False, "Machine parameter map failed")
        self._QMessageBox.critical(self._window, "Machine parameter map failed", message)

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
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap

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
        low = np.array([41, 98, 156], dtype=np.float64)
        mid = np.array([238, 241, 236], dtype=np.float64)
        high = np.array([200, 103, 54], dtype=np.float64)
        rgb = np.empty((*normalized.shape, 3), dtype=np.uint8)
        lower_half = normalized <= 0.5
        t_low = np.clip(normalized * 2.0, 0.0, 1.0)[..., None]
        t_high = np.clip((normalized - 0.5) * 2.0, 0.0, 1.0)[..., None]
        rgb[lower_half] = (low * (1.0 - t_low[lower_half]) + mid * t_low[lower_half]).astype(
            np.uint8
        )
        rgb[~lower_half] = (
            mid * (1.0 - t_high[~lower_half]) + high * t_high[~lower_half]
        ).astype(np.uint8)
        rgb[~finite] = (220, 224, 230)
        rgb = np.ascontiguousarray(np.flipud(rgb))
        height, width, _ = rgb.shape
        qimage = QImage(
            rgb.data,
            width,
            height,
            width * 3,
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(qimage)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor(24, 24, 27), 2))
        sample_x = self._machine_map_contour_data.get("_sample_x", [])
        sample_y = self._machine_map_contour_data.get("_sample_y", [])
        for x_value, y_value in zip(sample_x, sample_y):
            if not np.isfinite(x_value) or not np.isfinite(y_value):
                continue
            x = float(np.clip(x_value, 0.0, 1.0)) * (width - 1)
            y = (1.0 - float(np.clip(y_value, 0.0, 1.0))) * (height - 1)
            painter.drawEllipse(QPointF(x, y), 3.0, 3.0)
        painter.end()
        return pixmap

    def _fit_machine_map_contour_pixmap(self) -> None:
        if self._machine_map_contour_source_pixmap is None:
            return
        self._machine_map_contour_label.setPixmap(
            self._machine_map_contour_source_pixmap.scaled(
                self._machine_map_contour_label.size(),
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

        if self._part_type.currentText() == "Part & Support":
            self._QMessageBox.warning(
                self._window,
                "Not implemented",
                "Part & Support UI is restored, but support voxelization is not migrated yet.",
            )
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
            self._preview_loaded_result()
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

    def _save_outputs(self) -> None:
        if self._busy:
            self._append_log("A task is already in progress.")
            return
        if self._last_result is None:
            self._QMessageBox.warning(self._window, "Missing result", "Run a simulation first.")
            return

        output_text = self._output_dir.text().strip()
        if output_text:
            output_dir = Path(output_text)
        elif self._last_result_config is not None:
            output_dir = self._last_result_config.output_dir
        else:
            output_dir = Path("examples/outputs/gui_simulation")
        self._append_log(f"Saving outputs to: {output_dir}")
        self._set_busy(True, "Saving outputs...", task="save")
        worker = _SaveOutputsWorker(output_dir, self._last_result)
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
        self._files_label.setText("simulation_result.npz\nprobability.vtk\nbinary.vtk")
        self._append_log("Saved simulation_result.npz, probability.vtk, binary.vtk")
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
        self._run_button.setEnabled((not busy) and self._last_voxel_grid is not None)
        self._preview_result_button.setEnabled((not busy) and self._last_result is not None)

        can_save = (not busy) and self._last_result is not None
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
        if hasattr(self, "_run_calibration_button"):
            self._run_calibration_button.setEnabled(not busy)
            self._run_calibration_button.setText(
                "Running..." if busy and task == "model_calibration" else "Run Model Calibration"
            )
        if hasattr(self, "_generate_machine_map_button"):
            try:
                can_generate_map = (not busy) and self._machine_map_weights_csv_path().exists()
            except Exception:
                can_generate_map = False
            self._generate_machine_map_button.setEnabled(can_generate_map)
            self._generate_machine_map_button.setText(
                "Generating..." if busy and task == "machine_map" else "Generate Machine Map"
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
        self._voxel_status_label.setText("Required")
        self._update_preview_source_availability()
        self._append_log("Voxelization cleared. Run voxelization again before simulation.")

    def _voxel_signature(self, config) -> tuple[str, float]:
        return (str(config.geometry_path.resolve()), float(config.voxel_spacing))

    def _simulation_config_from_form(self):
        from capp.config import SimulationConfig
        from capp.domain import (
            MachineBiasMode,
            MachineMapCoordinateMode,
            NeighborhoodModel,
            SolverParameters,
            StochasticMode,
        )

        geometry_path = Path(self._part_geometry.text().strip())
        if not geometry_path.exists():
            raise ValueError("Select a valid part geometry STL file.")

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
        machine_bias = (
            MachineBiasMode.NONE
            if self._machine_preset.currentText() == "None"
            else MachineBiasMode.PRESET
        )
        machine_map_path = None
        machine_map_position = (0.0, 0.0)
        machine_map_coordinate_mode = MachineMapCoordinateMode.FULL_BASE_PLATE
        machine_map_bounds = None
        if machine_bias is MachineBiasMode.PRESET:
            machine_map_path = Path(self._machine_map_path.text().strip())
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
        )

    def _log_run_config(self, config) -> None:
        solver = config.solver
        self._append_log(f"Geometry path used: {config.geometry_path}")
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
        self._append_log(f"Solver backend: {solver.backend.value}")

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
