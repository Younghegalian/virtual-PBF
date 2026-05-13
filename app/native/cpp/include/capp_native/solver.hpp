#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace capp_native {

pybind11::dict solve_von_neumann(
    pybind11::array_t<bool, pybind11::array::c_style | pybind11::array::forcecast> voxel,
    double spacing,
    pybind11::dict parameters,
    pybind11::object progress_callback = pybind11::none());

}
