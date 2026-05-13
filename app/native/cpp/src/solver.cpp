#include "capp_native/solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace capp_native {
namespace {

struct Shape3 {
  std::size_t x;
  std::size_t y;
  std::size_t z;
};

struct Coefficients {
  float neg_y;
  float pos_x;
  float pos_y;
  float neg_x;
  float lower;
};

struct Parameters {
  std::string neighborhood;
  std::string stochastic_mode;
  std::string machine_bias;
  Coefficients coeffs;
  float residual_avg;
  float residual_max;
  int iteration_bound;
  float min_bias;
  float initial_deviation;
  std::uint32_t rng_seed;
};

struct SpatialParameters {
  bool enabled = false;
  bool has_initial_deviation = false;
  std::vector<float> neg_y;
  std::vector<float> pos_x;
  std::vector<float> pos_y;
  std::vector<float> neg_x;
  std::vector<float> min_bias;
  std::vector<float> initial_deviation;
};

std::size_t idx(std::size_t i, std::size_t j, std::size_t k, Shape3 shape) {
  return (i * shape.y + j) * shape.z + k;
}

std::size_t idx2(std::size_t i, std::size_t j, Shape3 shape) {
  return i * shape.y + j;
}

float clamp01(float value) {
  return std::max(0.0F, std::min(1.0F, value));
}

std::vector<float> read_float_grid(py::handle value, Shape3 shape, const std::string& name) {
  auto array = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(value);
  if (!array) {
    throw std::runtime_error(name + " must be convertible to a float array.");
  }
  auto info = array.request();
  if (info.ndim != 2 || info.shape[0] != static_cast<py::ssize_t>(shape.x) ||
      info.shape[1] != static_cast<py::ssize_t>(shape.y)) {
    throw std::runtime_error(name + " must be a 2D float array matching voxel X/Y shape.");
  }
  const float* data = static_cast<const float*>(info.ptr);
  return std::vector<float>(data, data + shape.x * shape.y);
}

SpatialParameters parse_spatial_parameters(py::dict parameters, Shape3 cropped) {
  SpatialParameters spatial{};
  if (parameters.contains("spatial_current_coefficients") &&
      !parameters["spatial_current_coefficients"].is_none()) {
    py::sequence seq =
        py::reinterpret_borrow<py::sequence>(parameters["spatial_current_coefficients"]);
    if (seq.size() != 4) {
      throw std::runtime_error("spatial_current_coefficients requires four grids.");
    }
    spatial.enabled = true;
    spatial.neg_y = read_float_grid(seq[0], cropped, "spatial NX");
    spatial.pos_x = read_float_grid(seq[1], cropped, "spatial PX");
    spatial.pos_y = read_float_grid(seq[2], cropped, "spatial NY");
    spatial.neg_x = read_float_grid(seq[3], cropped, "spatial PY");
  }
  if (parameters.contains("spatial_min_bias") && !parameters["spatial_min_bias"].is_none()) {
    spatial.enabled = true;
    spatial.min_bias = read_float_grid(parameters["spatial_min_bias"], cropped, "spatial EPS");
  }
  if (parameters.contains("spatial_initial_deviation") &&
      !parameters["spatial_initial_deviation"].is_none()) {
    spatial.enabled = true;
    spatial.initial_deviation =
        read_float_grid(parameters["spatial_initial_deviation"], cropped, "spatial IDP");
    spatial.has_initial_deviation = std::any_of(
        spatial.initial_deviation.begin(), spatial.initial_deviation.end(),
        [](float value) { return value > 0.0F; });
  }
  return spatial;
}

float scalar_or_last(py::handle value) {
  if (py::isinstance<py::sequence>(value) && !py::isinstance<py::str>(value)) {
    py::sequence seq = py::reinterpret_borrow<py::sequence>(value);
    if (seq.size() == 0) {
      throw std::runtime_error("Coefficient sequence cannot be empty.");
    }
    return seq[seq.size() - 1].cast<float>();
  }
  return value.cast<float>();
}

std::tuple<float, float, float, float> directional_coefficients(py::handle value) {
  if (py::isinstance<py::sequence>(value) && !py::isinstance<py::str>(value)) {
    py::sequence seq = py::reinterpret_borrow<py::sequence>(value);
    if (seq.size() != 4) {
      throw std::runtime_error("Directional current coefficients require four values.");
    }
    return {seq[0].cast<float>(), seq[1].cast<float>(), seq[2].cast<float>(),
            seq[3].cast<float>()};
  }
  float coeff = value.cast<float>();
  return {coeff, coeff, coeff, coeff};
}

Parameters parse_parameters(py::dict parameters) {
  Parameters parsed{};
  parsed.neighborhood = parameters["neighborhood"].cast<std::string>();
  parsed.stochastic_mode = parameters["stochastic_mode"].cast<std::string>();
  parsed.machine_bias = parameters["machine_bias"].cast<std::string>();
  if (parsed.machine_bias != "none") {
    throw std::runtime_error("Machine bias presets are not migrated to native solver yet.");
  }
  if (parsed.neighborhood == "simple_moore") {
    throw std::runtime_error("Moore neighborhood parity is pending for native solver.");
  }

  auto [a, b, c, d] = directional_coefficients(parameters["current_coefficients"]);
  float lower = scalar_or_last(parameters["lower_coefficients"]);
  if (parsed.neighborhood == "simple_von_neumann") {
    parsed.coeffs = {a, a, a, a, lower};
  } else {
    parsed.coeffs = {a, b, c, d, lower};
  }

  py::sequence residual = py::reinterpret_borrow<py::sequence>(parameters["residual_criteria"]);
  parsed.residual_avg = residual[0].cast<float>();
  parsed.residual_max = residual[1].cast<float>();
  parsed.iteration_bound = parameters["iteration_bound"].cast<int>();
  parsed.min_bias = parameters["min_bias"].cast<float>();
  parsed.initial_deviation = parameters["initial_deviation"].cast<float>();
  if (parameters["rng_seed"].is_none()) {
    parsed.rng_seed = 1000;
  } else {
    parsed.rng_seed = parameters["rng_seed"].cast<std::uint32_t>();
  }
  return parsed;
}

std::vector<std::uint8_t> remove_small_components(
    const std::vector<std::uint8_t>& input,
    Shape3 shape,
    std::size_t min_size,
    int connectivity) {
  std::vector<std::uint8_t> output(input.size(), 0);
  std::vector<std::uint8_t> visited(input.size(), 0);
  std::vector<std::size_t> component;
  std::queue<std::size_t> pending;

  std::vector<std::tuple<int, int, int>> offsets;
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        if (dx == 0 && dy == 0 && dz == 0) {
          continue;
        }
        int distance = std::abs(dx) + std::abs(dy) + std::abs(dz);
        if ((connectivity == 1 && distance == 1) || connectivity == 3) {
          offsets.emplace_back(dx, dy, dz);
        }
      }
    }
  }

  for (std::size_t i = 0; i < shape.x; ++i) {
    for (std::size_t j = 0; j < shape.y; ++j) {
      for (std::size_t k = 0; k < shape.z; ++k) {
        std::size_t start = idx(i, j, k, shape);
        if (!input[start] || visited[start]) {
          continue;
        }

        component.clear();
        pending.push(start);
        visited[start] = 1;
        while (!pending.empty()) {
          std::size_t current = pending.front();
          pending.pop();
          component.push_back(current);

          std::size_t ci = current / (shape.y * shape.z);
          std::size_t rem = current % (shape.y * shape.z);
          std::size_t cj = rem / shape.z;
          std::size_t ck = rem % shape.z;

          for (auto [dx, dy, dz] : offsets) {
            int ni = static_cast<int>(ci) + dx;
            int nj = static_cast<int>(cj) + dy;
            int nk = static_cast<int>(ck) + dz;
            if (ni < 0 || nj < 0 || nk < 0 || ni >= static_cast<int>(shape.x) ||
                nj >= static_cast<int>(shape.y) || nk >= static_cast<int>(shape.z)) {
              continue;
            }
            std::size_t next = idx(static_cast<std::size_t>(ni), static_cast<std::size_t>(nj),
                                   static_cast<std::size_t>(nk), shape);
            if (input[next] && !visited[next]) {
              visited[next] = 1;
              pending.push(next);
            }
          }
        }

        if (component.size() >= min_size) {
          for (std::size_t cell : component) {
            output[cell] = 1;
          }
        }
      }
    }
  }
  return output;
}

std::vector<std::uint8_t> crop_binary(
    const std::vector<std::uint8_t>& input,
    Shape3 padded,
    Shape3 cropped) {
  std::vector<std::uint8_t> output(cropped.x * cropped.y * cropped.z, 0);
  for (std::size_t i = 0; i < cropped.x; ++i) {
    for (std::size_t j = 0; j < cropped.y; ++j) {
      for (std::size_t k = 0; k < cropped.z; ++k) {
        output[idx(i, j, k, cropped)] = input[idx(i + 1, j + 1, k + 1, padded)];
      }
    }
  }
  return output;
}

void paste_binary(
    std::vector<std::uint8_t>& target,
    const std::vector<std::uint8_t>& source,
    Shape3 padded,
    Shape3 cropped) {
  for (std::size_t i = 0; i < cropped.x; ++i) {
    for (std::size_t j = 0; j < cropped.y; ++j) {
      for (std::size_t k = 0; k < cropped.z; ++k) {
        target[idx(i + 1, j + 1, k + 1, padded)] = source[idx(i, j, k, cropped)];
      }
    }
  }
}

void smooth_binary(std::vector<std::uint8_t>& binary, Shape3 padded, int max_iterations) {
  float threshold = 5.0F;
  std::vector<std::uint8_t> previous(binary.size(), 0);
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    previous = binary;
    bool changed = false;
    for (std::size_t i = 1; i < padded.x - 1; ++i) {
      for (std::size_t j = 1; j < padded.y - 1; ++j) {
        for (std::size_t k = 1; k < padded.z - 1; ++k) {
          int neighbors =
              previous[idx(i - 1, j - 1, k, padded)] +
              2 * previous[idx(i, j - 1, k, padded)] +
              previous[idx(i + 1, j - 1, k, padded)] +
              2 * previous[idx(i - 1, j, k, padded)] +
              2 * previous[idx(i + 1, j, k, padded)] +
              previous[idx(i - 1, j + 1, k, padded)] +
              2 * previous[idx(i, j + 1, k, padded)] +
              previous[idx(i + 1, j + 1, k, padded)] +
              3 * previous[idx(i, j, k - 1, padded)] +
              3 * previous[idx(i, j, k + 1, padded)];
          std::uint8_t next = previous[idx(i, j, k, padded)] && neighbors >= threshold;
          if (next != binary[idx(i, j, k, padded)]) {
            changed = true;
          }
          binary[idx(i, j, k, padded)] = next;
        }
      }
    }
    threshold -= 0.2F;
    if (!changed) {
      break;
    }
  }
}

std::vector<std::uint8_t> postprocess_binary(
    std::vector<std::uint8_t> binary,
    Shape3 padded,
    Shape3 cropped,
    bool apply_area_open) {
  if (apply_area_open) {
    auto inner = crop_binary(binary, padded, cropped);
    inner = remove_small_components(inner, cropped, 8, 3);
    paste_binary(binary, inner, padded, cropped);
  }
  auto filtered = remove_small_components(binary, padded, 50, 1);
  auto cropped_binary = crop_binary(filtered, padded, cropped);
  return remove_small_components(cropped_binary, cropped, 50, 1);
}

}  // namespace

py::dict solve_von_neumann(
    py::array_t<bool, py::array::c_style | py::array::forcecast> voxel_array,
    double spacing,
    py::dict parameters,
    py::object progress_callback) {
  auto voxel_info = voxel_array.request();
  if (voxel_info.ndim != 3) {
    throw std::runtime_error("voxel must be a 3D boolean array.");
  }
  if (spacing <= 0.0) {
    throw std::runtime_error("spacing must be positive.");
  }

  Parameters params = parse_parameters(parameters);
  Shape3 cropped{static_cast<std::size_t>(voxel_info.shape[0]),
                 static_cast<std::size_t>(voxel_info.shape[1]),
                 static_cast<std::size_t>(voxel_info.shape[2])};
  Shape3 padded{cropped.x + 2, cropped.y + 2, cropped.z + 2};
  SpatialParameters spatial = parse_spatial_parameters(parameters, cropped);
  if (spatial.enabled && params.neighborhood != "directional_von_neumann") {
    throw std::runtime_error("Spatial machine parameter maps require DirectionalVN.");
  }
  const bool* voxel = static_cast<const bool*>(voxel_info.ptr);

  std::vector<float> voxel_calc(padded.x * padded.y * padded.z, 0.0F);
  std::vector<float> probability(voxel_calc.size(), 0.0F);
  std::vector<float> probability_export(voxel_calc.size(), 0.0F);
  std::vector<std::uint8_t> binary(voxel_calc.size(), 0);
  std::vector<float> random_field(voxel_calc.size(), 0.0F);

  auto started = std::chrono::steady_clock::now();
  bool has_progress = !progress_callback.is_none();
  int last_progress = -1;
  auto report_progress = [&](int percent, const std::string& message) {
    if (!has_progress) {
      return;
    }
    percent = std::max(0, std::min(100, percent));
    if (percent == last_progress) {
      return;
    }
    last_progress = percent;
    py::gil_scoped_acquire acquire;
    progress_callback(percent, message);
  };

  {
    py::gil_scoped_release release;
    report_progress(0, "Preparing PBF Q solver");

    for (std::size_t i = 0; i < padded.x; ++i) {
      for (std::size_t j = 0; j < padded.y; ++j) {
        voxel_calc[idx(i, j, 0, padded)] = 1.0F;
        probability[idx(i, j, 0, padded)] = 1.0F;
      }
    }

    for (std::size_t i = 0; i < cropped.x; ++i) {
      for (std::size_t j = 0; j < cropped.y; ++j) {
        for (std::size_t k = 0; k < cropped.z; ++k) {
          bool value = voxel[idx(i, j, k, cropped)];
          voxel_calc[idx(i + 1, j + 1, k + 1, padded)] = value ? 1.0F : 0.0F;
        }
      }
    }
    report_progress(2, "Generating native stochastic field");

    std::mt19937 rng(params.rng_seed);
    std::uniform_real_distribution<float> distribution(0.0F, 1.0F);
    for (float& value : random_field) {
      value = distribution(rng);
    }

    float spacing_scale =
        params.stochastic_mode == "in_layer"
            ? static_cast<float>(spacing)
            : static_cast<float>(std::pow(spacing, 12.0));
    float min_val = params.min_bias * spacing_scale;
    int beta = static_cast<int>(std::ceil(0.1 / spacing) + 1.0);
    float idp_model = params.initial_deviation / static_cast<float>(spacing);
    Shape3 layer_shape{cropped.x, cropped.y, 1};
    std::vector<float> updated(cropped.x * cropped.y, 0.0F);
    std::vector<float> previous(cropped.x * cropped.y, 0.0F);

    for (std::size_t layer = 1; layer <= cropped.z; ++layer) {
      int percent = 3 + static_cast<int>(
                            (static_cast<float>(layer - 1) /
                             std::max<std::size_t>(1, cropped.z)) *
                            89.0F);
      report_progress(percent, "Running PBF Q solver");

      if (static_cast<int>(layer) <= beta) {
        for (std::size_t i = 0; i < cropped.x; ++i) {
          for (std::size_t j = 0; j < cropped.y; ++j) {
            probability[idx(i + 1, j + 1, layer, padded)] =
                voxel_calc[idx(i + 1, j + 1, layer, padded)];
          }
        }
      } else {
        float ma_residual = 1.0F;
        float max_residual = 1.0F;
        int iteration = 0;
        while ((ma_residual > params.residual_avg || max_residual > params.residual_max) &&
               iteration < params.iteration_bound) {
          for (std::size_t i = 0; i < cropped.x; ++i) {
            for (std::size_t j = 0; j < cropped.y; ++j) {
              previous[idx(i, j, 0, layer_shape)] =
                  probability[idx(i + 1, j + 1, layer, padded)];
            }
          }

          float delta_sum = 0.0F;
          max_residual = 0.0F;
          std::size_t changed = 0;
          for (std::size_t i = 0; i < cropped.x; ++i) {
            for (std::size_t j = 0; j < cropped.y; ++j) {
              float center_voxel = voxel_calc[idx(i + 1, j + 1, layer, padded)];
              std::size_t map_cell = idx2(i, j, cropped);
              float neg_y = spatial.neg_y.empty() ? params.coeffs.neg_y : spatial.neg_y[map_cell];
              float pos_x = spatial.pos_x.empty() ? params.coeffs.pos_x : spatial.pos_x[map_cell];
              float pos_y = spatial.pos_y.empty() ? params.coeffs.pos_y : spatial.pos_y[map_cell];
              float neg_x = spatial.neg_x.empty() ? params.coeffs.neg_x : spatial.neg_x[map_cell];
              float min_val_cell =
                  spatial.min_bias.empty() ? min_val : spatial.min_bias[map_cell] * spacing_scale;
              float idp_model_cell = spatial.initial_deviation.empty()
                                         ? idp_model
                                         : spatial.initial_deviation[map_cell] /
                                               static_cast<float>(spacing);
              float left = probability[idx(i + 1, j, layer, padded)] * neg_y;
              float upper = probability[idx(i, j + 1, layer, padded)] * pos_x;
              float lower_x =
                  probability[idx(i + 2, j + 1, layer, padded)] * pos_y;
              float right = probability[idx(i + 1, j + 2, layer, padded)] * neg_x;
              float below = probability[idx(i + 1, j + 1, layer - 1, padded)] * center_voxel *
                            params.coeffs.lower;

              float no_growth =
                  (1.0F - left) * (1.0F - upper) * (1.0F - lower_x) * (1.0F - right) *
                  (1.0F - below);
              std::size_t inner = idx(i, j, 0, layer_shape);
              float epsilon =
                  previous[inner] == 0.0F ? min_val_cell * center_voxel : 0.0F;
              float value = ((center_voxel + idp_model_cell) / (1.0F + idp_model_cell)) *
                            (1.0F - no_growth + epsilon);
              value = clamp01(value);
              updated[inner] = value;
              float delta = std::abs(previous[inner] - value);
              if (delta != 0.0F) {
                ++changed;
              }
              delta_sum += delta;
              max_residual = std::max(max_residual, delta);
            }
          }

          for (std::size_t i = 0; i < cropped.x; ++i) {
            for (std::size_t j = 0; j < cropped.y; ++j) {
              probability[idx(i + 1, j + 1, layer, padded)] =
                  updated[idx(i, j, 0, layer_shape)];
            }
          }
          ma_residual = delta_sum / static_cast<float>(changed + 1);
          ++iteration;
        }
      }

      for (std::size_t i = 0; i < cropped.x; ++i) {
        for (std::size_t j = 0; j < cropped.y; ++j) {
          probability_export[idx(i + 1, j + 1, layer, padded)] =
              probability[idx(i + 1, j + 1, layer, padded)];
        }
      }

      if (params.stochastic_mode == "in_layer") {
        for (std::size_t i = 0; i < padded.x; ++i) {
          for (std::size_t j = 0; j < padded.y; ++j) {
            std::size_t cell = idx(i, j, layer, padded);
            binary[cell] = random_field[cell] <= probability[cell] ? 1 : 0;
            probability[cell] = binary[cell] ? 1.0F : 0.0F;
          }
        }
      }
    }

    if (params.stochastic_mode == "in_volume") {
      report_progress(92, "Sampling native in-volume stochastic field");
      for (std::size_t cell = 0; cell < binary.size(); ++cell) {
        binary[cell] = random_field[cell] <= probability[cell] && voxel_calc[cell] > 0.0F ? 1 : 0;
      }
      report_progress(94, "Smoothing native sampled volume");
      smooth_binary(binary, padded, 20);
    }

    report_progress(96, "Post-processing native connected components");
    bool apply_area_open =
        params.stochastic_mode == "in_layer" &&
        (spatial.initial_deviation.empty() ? params.initial_deviation > 0.0F
                                           : spatial.has_initial_deviation);
    binary = postprocess_binary(binary, padded, cropped, apply_area_open);
  }

  auto ended = std::chrono::steady_clock::now();
  double elapsed =
      std::chrono::duration_cast<std::chrono::duration<double>>(ended - started).count();

  py::array_t<std::uint8_t> probability_result({cropped.x, cropped.y, cropped.z});
  py::array_t<std::uint8_t> binary_result({cropped.x, cropped.y, cropped.z});
  auto probability_buffer = probability_result.mutable_unchecked<3>();
  auto binary_buffer = binary_result.mutable_unchecked<3>();

  std::size_t solid_count = 0;
  std::uint64_t probability_sum = 0;
  std::size_t binary_count = 0;
  for (std::size_t i = 0; i < cropped.x; ++i) {
    for (std::size_t j = 0; j < cropped.y; ++j) {
      for (std::size_t k = 0; k < cropped.z; ++k) {
        float probability_value = probability_export[idx(i + 1, j + 1, k + 1, padded)];
        auto encoded = static_cast<std::uint8_t>(std::floor(std::round(probability_value * 100.0F)));
        probability_buffer(i, j, k) = encoded;
        std::uint8_t binary_value = binary[idx(i, j, k, cropped)];
        binary_buffer(i, j, k) = binary_value;
        if (voxel[idx(i, j, k, cropped)]) {
          ++solid_count;
          probability_sum += encoded;
        }
        if (binary_value) {
          ++binary_count;
        }
      }
    }
  }

  if (solid_count == 0) {
    throw std::runtime_error("Cannot simulate an empty voxel grid.");
  }

  py::dict result;
  result["probability"] = probability_result;
  result["binary"] = binary_result;
  result["rest_volume"] = 100.0 * static_cast<double>(binary_count) / static_cast<double>(solid_count);
  result["probability_density"] =
      static_cast<double>(probability_sum) / static_cast<double>(solid_count);
  result["elapsed_seconds"] = elapsed;
  if (has_progress) {
    progress_callback(100, "PBF Q solver complete");
  }
  return result;
}

}  // namespace capp_native
