#include <pybind11/pybind11.h>

#include "capp_native/solver.hpp"
#include "capp_native/version.hpp"

namespace py = pybind11;

PYBIND11_MODULE(capp_native, module) {
  module.doc() = "Native acceleration module for CA-PP rebuild.";
  module.def("version", &capp_native::version);
  module.def("solve_von_neumann", &capp_native::solve_von_neumann, py::arg("voxel"),
             py::arg("spacing"), py::arg("parameters"),
             py::arg("progress_callback") = py::none());
}

