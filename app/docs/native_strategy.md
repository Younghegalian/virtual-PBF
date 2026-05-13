# Native C++/CUDA Strategy

This document records which parts of CA-PP should use native code and which should remain in Python.

## Guiding Rule

Use C++/CUDA only where it buys clear performance, memory control, or deterministic low-level behavior.

Python remains the orchestration layer for configuration, UI integration, optimization, I/O, and research iteration.

## Strong C++/CUDA Candidates

| Area | Native Target | Why |
| --- | --- | --- |
| Layer-wise solver update | C++ CPU first, CUDA later | Main repeated numerical workload. Needs memory layout control and acceleration. |
| Stochastic sampling | C++/CUDA | Large volume-wide RNG and thresholding. Should share memory layout with solver. |
| Binary smoothing | C++/CUDA | Repeated local neighborhood operation over dense 3D tensors. |
| Connected component cleanup | C++ first, CUDA optional | Current MATLAB path removes small components. Can become expensive for large volumes. |
| Voxel tensor utilities | C++ | Padding, cropping, typed buffers, layout conversion, stats. Good shared infrastructure. |
| Batch calibration evaluator | C++/CUDA solver behind Python optimizer | Optimizer calls simulation many times; the solver bottleneck compounds. |

## Possible Later Native Candidates

| Area | Recommendation |
| --- | --- |
| Voxelization | Start with Python `trimesh`/VTK tooling. Move to C++ only if parity/performance becomes a blocker. |
| ROI loss | Keep in Python first. Move to C++ only if calibration profiling shows it matters. |
| VTK/STL export | Keep in Python libraries first. Native export is not a first-order bottleneck. |
| Neural decoder inference | Depends on final model format. Prefer ONNX Runtime or Torch before custom C++. |

## Poor C++ Candidates

These should stay out of native code unless there is a proven reason:

- Desktop UI logic.
- Project/config management.
- File browsing and workflow state.
- Optimizer orchestration.
- Data migration scripts.
- Plotting and visual inspection.

## Native Build Direction

Initial native work should be staged:

1. Define C++ data structures for dense 3D volumes.
2. Port the Python reference Von Neumann solver to C++ CPU.
3. Bind it to Python with nanobind or pybind11.
4. Add parity tests against Python reference fixtures.
5. Move the hot layer update kernel to CUDA.
6. Add CUDA sampling and smoothing only after solver parity is stable.

Current implementation note:

- The workbench has explicit solver backend choices for CPU reference, CPU native C++, and CUDA GPU.
- CPU native C++ and CUDA GPU are enabled only after local module/device validation passes.
- The CPU native C++ backend is implemented as `capp_native.solve_von_neumann()` through pybind11 and is the preferred CPU path.
- The current CUDA backend uses CuPy CUDA array operations and NVIDIA runtime wheels. A lower-level custom CUDA kernel remains a later optimization step.
- The Python/NumPy reference solver remains the correctness reference and fallback.
- That solver is layer-vectorized, not voxel-looped, but each Z layer can still run many convergence iterations.
- Large high-density volumes should not be judged by this reference solver's speed; the production path is the native solver staged above.

## Memory Layout

The native core should explicitly define a single dense volume layout.

Recommendation:

- Store volumes as flat contiguous buffers.
- Use row-major indexing on the C++ side.
- Document conversion to and from NumPy arrays.
- Avoid hidden transposes at API boundaries.
- Keep probability as `float32` during computation and `uint8` only for persisted result artifacts.
- Keep binary and voxel data as byte-backed booleans in native code to avoid bit-packed complexity.

## CUDA Notes

The solver has layer dependency along the Z axis. That means:

- Z layers remain sequential at the high level.
- X/Y cells within a layer are strong CUDA candidates.
- Convergence loops remain per-layer.
- GPU kernels should avoid reallocating temporary arrays per iteration.
- RNG should be explicitly seeded and documented; bit-perfect MATLAB equivalence is not expected.

The target is scientific/statistical parity, not identical MATLAB random samples.

