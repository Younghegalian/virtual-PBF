from __future__ import annotations

import argparse
import os
from pathlib import Path

from capp.config import load_simulation_config
from capp.data.matlab import inspect_mat_file
from capp.domain import SolverBackend
from capp.simulation.runner import run_simulation_config, save_default_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="capp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="Run a simulation from a YAML config.")
    simulate.add_argument("--config", required=True, type=Path)

    calibrate = subparsers.add_parser(
        "model-calibrate",
        help="Run Model Calibration from ROI samples and a calibration STL.",
    )
    calibrate.add_argument("--geometry", required=True, type=Path)
    calibrate.add_argument("--samples", required=True, type=Path)
    calibrate.add_argument("--spacing", required=True, type=float)
    calibrate.add_argument("--output", required=True, type=Path)
    calibrate.add_argument("--max-evaluations", default=24, type=int)
    calibrate.add_argument(
        "--optimizer",
        default="adaptive_sobol",
        choices=["adaptive_sobol", "sobol", "latin_hypercube"],
        help="Model Calibration candidate search strategy.",
    )
    calibrate.add_argument(
        "--parallel-samples",
        default=min(4, os.cpu_count() or 1),
        type=int,
        help="Number of shared candidate simulations to run concurrently.",
    )
    calibrate.add_argument(
        "--backend",
        default=SolverBackend.CPU_NATIVE.value,
        choices=[backend.value for backend in SolverBackend],
    )
    calibrate.add_argument(
        "--sample",
        action="append",
        help="Optional sample name to include. Repeat to calibrate multiple selected samples.",
    )
    calibrate.add_argument(
        "--save-research-artifacts",
        action="store_true",
        help="Export per-sample ROI TIFFs and best 3D STL/VTK calibration artifacts.",
    )

    machine_map = subparsers.add_parser(
        "machine-map",
        help="Generate a build-plate machine parameter map from Model Calibration weights.",
    )
    machine_map.add_argument("--weights", required=True, type=Path)
    machine_map.add_argument("--coordinates", required=True, type=Path)
    machine_map.add_argument("--output", required=True, type=Path)
    machine_map.add_argument("--resolution", default=200, type=int)
    machine_map.add_argument("--name", default="Machine Map")
    machine_map.add_argument("--voxel-spacing", type=float)

    inspect_mat = subparsers.add_parser("inspect-mat", help="Inspect MATLAB data file headers.")
    inspect_mat.add_argument("paths", nargs="+", type=Path)

    args = parser.parse_args(argv)

    if args.command == "simulate":
        return _simulate(args.config)
    if args.command == "model-calibrate":
        return _model_calibrate(args)
    if args.command == "machine-map":
        return _machine_map(args)
    if args.command == "inspect-mat":
        return _inspect_mat(args.paths)

    parser.error(f"Unknown command: {args.command}")
    return 2


def _simulate(config_path: Path) -> int:
    config = load_simulation_config(config_path)
    result = run_simulation_config(config)
    save_default_outputs(config.output_dir, result)
    print(f"Simulation complete: {config.output_dir}")
    return 0


def _model_calibrate(args: argparse.Namespace) -> int:
    from capp.calibration.model_calibration import (
        ModelCalibrationOptions,
        run_model_calibration_from_paths,
    )

    def progress(percent: int, message: str) -> None:
        print(f"[{percent:3d}%] {message}")

    result = run_model_calibration_from_paths(
        geometry_path=args.geometry,
        sample_dir=args.samples,
        voxel_spacing=args.spacing,
        output_dir=args.output,
        options=ModelCalibrationOptions(
            max_evaluations=args.max_evaluations,
            backend=SolverBackend(args.backend),
            max_workers=args.parallel_samples,
            optimizer=args.optimizer,
            save_research_artifacts=args.save_research_artifacts,
        ),
        sample_names=set(args.sample) if args.sample else None,
        progress_callback=progress,
    )
    print(
        "Model Calibration complete: "
        f"{len(result.samples)} sample(s), average loss {result.average_loss:.6g}, "
        f"outputs {args.output}"
    )
    return 0


def _machine_map(args: argparse.Namespace) -> int:
    from capp.machine_map import generate_machine_parameter_map_from_files

    def progress(percent: int, message: str) -> None:
        print(f"[{percent:3d}%] {message}")

    result = generate_machine_parameter_map_from_files(
        weights_csv=args.weights,
        coordinates_xlsx=args.coordinates,
        output_dir=args.output,
        resolution=args.resolution,
        preset_name=args.name,
        voxel_spacing=args.voxel_spacing,
        progress_callback=progress,
    )
    print(
        "Machine parameter map complete: "
        f"{result.sample_count} sample(s), resolution {result.resolution}, "
        f"outputs {result.output_dir}"
    )
    return 0


def _inspect_mat(paths: list[Path]) -> int:
    for path in paths:
        info = inspect_mat_file(path)
        print(f"{info.path}: {info.kind.value} ({info.size_bytes} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

