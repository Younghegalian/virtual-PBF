from __future__ import annotations

from datetime import datetime
from pathlib import Path


class WorkbenchMainWindow:
    def __init__(self) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QFileDialog,
            QLabel,
            QListWidget,
            QMainWindow,
            QMessageBox,
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
            "simulation": style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay),
            "calibration": style.standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton),
            "results": style.standardIcon(QStyle.StandardPixmap.SP_FileIcon),
            "data": style.standardIcon(QStyle.StandardPixmap.SP_DriveHDIcon),
            "settings": style.standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView),
            "open": style.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon),
        }

        self._navigation = QListWidget()
        self._navigation.setFixedWidth(220)
        self._navigation.setSpacing(2)
        self._navigation.currentRowChanged.connect(self._set_page)
        self._last_result = None
        self._last_voxel_grid = None
        self._loaded_result = None
        self._log = self._make_log()

        self._stack = QStackedWidget()
        self._add_feature("home", "Workspace", self._build_start_page())
        self._add_feature("simulation", "Virtual Printing", self._build_simulation_page())
        self._add_feature("results", "Result Display", self._build_results_page())
        self._add_feature("calibration", "Lab Features", self._build_lab_page())
        self._add_feature("data", "Data & Models", self._build_placeholder_page("Data & Models"))
        self._add_feature("settings", "Preferences", self._build_placeholder_page("Preferences"))

        splitter = QSplitter(self._Qt.Orientation.Horizontal)
        splitter.addWidget(self._navigation)
        splitter.addWidget(self._stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(4)
        central_layout.addWidget(splitter, 1)
        central_layout.addWidget(QLabel("Log"))
        central_layout.addWidget(self._log)
        self._window.setCentralWidget(central)
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

    def _build_start_page(self):
        from PySide6.QtWidgets import QGridLayout, QPushButton, QVBoxLayout, QWidget

        panel = QWidget()
        panel.setObjectName("Page")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)
        layout.addWidget(self._page_title("Workspace"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)

        entries = [
            ("simulation", "Virtual Printing", 1),
            ("results", "Result Display", 2),
            ("calibration", "Lab Features", 3),
            ("data", "Data & Models", 4),
            ("settings", "Preferences", 5),
        ]
        for position, (icon_key, label, page_index) in enumerate(entries):
            button = QPushButton(label)
            button.setIcon(self._icons[icon_key])
            button.setMinimumHeight(72)
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
        outer = QHBoxLayout(panel)

        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.addWidget(QLabel("Virtual Printing"))

        geometry_box = QGroupBox("Geometry In")
        geometry_form = QFormLayout(geometry_box)
        self._part_type = QComboBox()
        self._part_type.addItems(["Part only", "Part & Support"])
        self._part_type.currentTextChanged.connect(self._on_part_type_changed)
        geometry_form.addRow("Part type", self._part_type)

        self._part_geometry = QLineEdit()
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
        estimate = QPushButton("Estimate")
        estimate.clicked.connect(self._estimate_grid_spacing)
        spacing_row = QHBoxLayout()
        spacing_row.addWidget(self._grid_spacing)
        spacing_row.addWidget(estimate)
        geometry_form.addRow("Grid spacing (mm)", spacing_row)

        left_layout.addWidget(geometry_box)

        computation_box = QGroupBox("Computation Preset")
        computation_form = QFormLayout(computation_box)
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
        self._stochastic_mode = QComboBox()
        self._stochastic_mode.addItems(["In-layer", "In-volume"])
        process_form.addRow("Stochastic process", self._stochastic_mode)

        self._machine_preset = QComboBox()
        self._machine_preset.addItems(["None", "Machine Map"])
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

        self._processor = QComboBox()
        self._processor.currentIndexChanged.connect(self._update_backend_status_label)
        self._processor_status = QLabel()
        self._processor_status.setWordWrap(True)
        self._processor_status.setObjectName("BackendStatus")
        refresh_devices = QPushButton("Validate")
        refresh_devices.clicked.connect(lambda: self._refresh_compute_backends(log=True))
        processor_row = QHBoxLayout()
        processor_row.addWidget(self._processor, 1)
        processor_row.addWidget(refresh_devices)
        process_form.addRow("Processor", processor_row)
        process_form.addRow("Device status", self._processor_status)
        self._refresh_compute_backends(log=False)

        self._output_dir = QLineEdit("examples/outputs/gui_simulation")
        process_form.addRow("Output dir", self._file_row(self._output_dir, self._browse_output_dir))
        left_layout.addWidget(process_box)

        action_row = QHBoxLayout()
        preview_stl = QPushButton("Preview STL")
        preview_stl.setIcon(self._icons["open"])
        preview_stl.clicked.connect(self._preview_stl)
        action_row.addWidget(preview_stl)

        preview_voxel = QPushButton("Preview Voxel Grid")
        preview_voxel.clicked.connect(self._preview_voxel_grid)
        action_row.addWidget(preview_voxel)

        self._run_button = QPushButton("Run Virtual Printing")
        self._run_button.setIcon(self._icons["simulation"])
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
        left_layout.addLayout(action_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(430)
        scroll.setWidget(left_content)

        right = QWidget()
        right.setMinimumWidth(680)
        right_layout = QVBoxLayout(right)
        from capp.workbench.preview import PreviewPane

        self._preview = PreviewPane()
        right_layout.addWidget(self._preview.widget, 6)

        right_layout.addWidget(QLabel("Summary"))
        summary = QFormLayout()
        self._shape_label = QLabel("-")
        self._spacing_label = QLabel("-")
        self._rest_label = QLabel("-")
        self._density_label = QLabel("-")
        self._elapsed_label = QLabel("-")
        summary.addRow("Shape", self._shape_label)
        summary.addRow("Spacing", self._spacing_label)
        summary.addRow("Rest Volume", self._rest_label)
        summary.addRow("Probability Density", self._density_label)
        summary.addRow("Elapsed", self._elapsed_label)
        right_layout.addLayout(summary)

        splitter = QSplitter(self._Qt.Orientation.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([520, 980])
        outer.addWidget(splitter)

        self._on_part_type_changed(self._part_type.currentText())
        self._set_parameter_defaults(self._neighborhood.currentText())
        return panel

    def _build_results_page(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QComboBox,
            QFormLayout,
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
        layout = QHBoxLayout(panel)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Result Display"))

        self._result_npz_path = QLineEdit()
        load_row = QHBoxLayout()
        load_row.addWidget(self._result_npz_path)
        browse = QPushButton("Open NPZ")
        browse.setIcon(self._icons["open"])
        browse.clicked.connect(self._browse_result_npz)
        load_row.addWidget(browse)
        left_layout.addLayout(load_row)

        form = QFormLayout()
        self._result_volume_choice = QComboBox()
        self._result_volume_choice.addItems(["Voxel (raw CAD)", "Binary (cleaned)", "Probability"])
        self._result_volume_choice.currentTextChanged.connect(self._refresh_result_views)
        form.addRow("Volume", self._result_volume_choice)

        self._slice_axis = QComboBox()
        self._slice_axis.addItems(["Z", "X", "Y"])
        self._slice_axis.currentTextChanged.connect(self._update_result_slice)
        form.addRow("Slice axis", self._slice_axis)

        self._slice_slider = QSlider(Qt.Orientation.Horizontal)
        self._slice_slider.valueChanged.connect(self._update_result_slice)
        form.addRow("Slice", self._slice_slider)
        left_layout.addLayout(form)

        preview_button = QPushButton("Preview 3D")
        preview_button.clicked.connect(self._preview_loaded_result)
        left_layout.addWidget(preview_button)

        self._slice_label = QLabel("-")
        self._slice_label.setMinimumSize(360, 300)
        self._slice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(QLabel("Slice View"))
        left_layout.addWidget(self._slice_label, 1)

        self._output_label = QLabel("-")
        self._files_label = QLabel("-")
        left_layout.addWidget(QLabel("Output"))
        left_layout.addWidget(self._output_label)
        left_layout.addWidget(QLabel("Files"))
        left_layout.addWidget(self._files_label)

        from capp.workbench.preview import PreviewPane

        self._result_preview = PreviewPane()

        splitter = QSplitter(self._Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self._result_preview.widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([500, 1000])
        layout.addWidget(splitter)
        return panel

    def _build_lab_page(self):
        from PySide6.QtWidgets import QGridLayout, QPushButton, QVBoxLayout, QWidget

        panel = QWidget()
        panel.setObjectName("Page")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self._page_title("Lab Features"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        names = [
            "Topology Refinement",
            "Roughness Contour",
            "CT Data Transform",
            "RMC",
        ]
        for position, name in enumerate(names):
            button = QPushButton(name)
            button.setMinimumHeight(32)
            button.setEnabled(False)
            grid.addWidget(button, position // 2, position % 2)

        layout.addLayout(grid)
        layout.addStretch(1)
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
            self._estimate_grid_spacing()

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

    def _preview_stl(self) -> None:
        path = self._part_geometry.text().strip()
        if not path:
            self._QMessageBox.warning(self._window, "Missing geometry", "Select an STL first.")
            return
        if not Path(path).exists():
            self._QMessageBox.warning(
                self._window,
                "Missing geometry",
                "Selected STL does not exist.",
            )
            return
        self._append_log(f"Previewing STL: {path}")
        self._preview.show_stl(path)

    def _preview_result(self) -> None:
        if self._last_result is None:
            self._QMessageBox.warning(self._window, "Missing result", "Run a simulation first.")
            return
        self._append_log("Previewing cleaned binary voxel result.")
        self._preview.show_voxels(
            self._last_result.binary,
            spacing=self._last_result.spacing,
            origin=self._last_result.origin,
            label="Binary (cleaned)",
        )

    def _preview_voxel_grid(self) -> None:
        try:
            config = self._simulation_config_from_form()
        except Exception as exc:
            self._QMessageBox.warning(self._window, "Invalid input", str(exc))
            return

        try:
            from capp.geometry.voxelizer import voxelize_mesh

            self._append_log(
                f"Voxelizing selected geometry: {config.geometry_path} "
                f"at {config.voxel_spacing:g} mm"
            )
            grid = voxelize_mesh(config.geometry_path, config.voxel_spacing)
            self._last_voxel_grid = grid
            self._preview.show_voxels(
                grid.data,
                spacing=grid.spacing,
                origin=grid.origin,
                label="Voxel",
            )
            self._append_log(f"Voxel grid ready: shape={grid.shape}, filled={grid.filled_count}")
        except Exception as exc:
            self._append_log(f"Voxel preview failed: {exc}")
            self._QMessageBox.critical(self._window, "Voxel preview failed", str(exc))

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
        self._slice_source_pixmap = self._array_to_pixmap(image)
        self._fit_slice_pixmap()

    def _slice_label_resize_event(self, event) -> None:
        from PySide6.QtWidgets import QLabel

        QLabel.resizeEvent(self._slice_label, event)
        self._fit_slice_pixmap()

    def _fit_slice_pixmap(self) -> None:
        if self._slice_source_pixmap is None:
            return
        from PySide6.QtCore import Qt

        target_size = self._slice_label.contentsRect().size()
        if target_size.width() <= 2 or target_size.height() <= 2:
            return
        scaled = self._slice_source_pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._slice_label.setPixmap(scaled)

    def _selected_result_volume(self):
        if self._loaded_result is None:
            raise ValueError("No result loaded.")
        choice = self._result_volume_choice.currentText()
        if choice.startswith("Voxel"):
            return self._loaded_result["voxel"]
        if choice.startswith("Binary"):
            return self._loaded_result["binary"]
        return self._loaded_result["probability"]

    def _array_to_pixmap(self, image):
        import numpy as np
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
        return QPixmap.fromImage(qimage)

    def _refresh_calibration_backends(self, log: bool = False) -> None:
        from capp.compute.devices import solver_backend_statuses

        statuses = solver_backend_statuses()
        current_backend = self._calibration_processor.currentData()
        self._calibration_backend_statuses = {status.backend.value: status for status in statuses}

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
            self._processor_status.setText("Solver status not validated.")
            return
        state = "Available" if status.available else "Unavailable"
        self._processor_status.setText(f"{state}. {status.detail}")

    def _selected_solver_backend(self):
        from capp.domain import SolverBackend

        return SolverBackend(self._processor.currentData() or SolverBackend.CPU_REFERENCE.value)

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

    def _run_simulation(self) -> None:
        from PySide6.QtWidgets import QApplication

        from capp.simulation.runner import run_simulation_config, save_default_outputs

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

        self._run_button.setEnabled(False)
        QApplication.setOverrideCursor(self._Qt.CursorShape.WaitCursor)
        try:
            self._append_log(f"Running {config.geometry_path}")
            self._log_run_config(config)
            result = run_simulation_config(config)
            self._last_result = result
            save_default_outputs(config.output_dir, result)
            self._set_loaded_result_from_simulation(result, config.output_dir)
            self._append_log("Saved simulation_result.npz, probability.vtk, binary.vtk")
            self._shape_label.setText(" x ".join(str(v) for v in result.probability.shape))
            self._spacing_label.setText(f"{result.spacing:g}")
            self._rest_label.setText(f"{result.rest_volume:.3f}%")
            self._density_label.setText(f"{result.probability_density:.3f}%")
            self._elapsed_label.setText(f"{result.elapsed_seconds:.3f} s")
            self._output_label.setText(str(config.output_dir))
            self._files_label.setText("simulation_result.npz\nprobability.vtk\nbinary.vtk")
            self._preview_result_button.setEnabled(True)
            self._preview_result()
            self._preview_loaded_result()
            self._append_log(f"Complete {config.output_dir}")
            self._navigation.setCurrentRow(1)
        except Exception as exc:
            self._append_log(f"Error: {exc}")
            self._QMessageBox.critical(self._window, "Simulation failed", str(exc))
        finally:
            QApplication.restoreOverrideCursor()
            self._run_button.setEnabled(True)

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
        self._append_log(f"Solver: {self._solver_label(solver.backend.value)}")
        self._append_log("Noise cleanup: keeping 6-connected components >= 50 voxels.")

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
