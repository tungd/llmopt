#include <torch/extension.h>

#include <ATen/native/mps/MetalShaderLibrary.h>

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr uint64_t kTile = 16;

struct Q8Params {
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t has_bias;
};

struct KernelEntry {
  std::shared_ptr<at::native::mps::PrecompiledMetalShaderLibrary> library;
  std::shared_ptr<at::native::mps::MetalKernelFunction> half_kernel;
  std::shared_ptr<at::native::mps::MetalKernelFunction> float_kernel;
  std::shared_ptr<at::native::mps::MetalKernelFunction> half_dequant_kernel;
  std::shared_ptr<at::native::mps::MetalKernelFunction> float_dequant_kernel;
  std::shared_ptr<at::native::mps::MetalKernelFunction> w4a16_kernel;
  std::unordered_map<
      std::string,
      std::shared_ptr<at::native::mps::MetalKernelFunction>> half_q8_kernels;
  std::unordered_map<
      std::string,
      std::shared_ptr<at::native::mps::MetalKernelFunction>> float_q8_kernels;
};

std::mutex cache_mutex;
std::unordered_map<std::string, std::shared_ptr<KernelEntry>> kernel_cache;

std::shared_ptr<KernelEntry> load_kernel(const std::string& library_path) {
  std::lock_guard<std::mutex> lock(cache_mutex);
  const auto cached = kernel_cache.find(library_path);
  if (cached != kernel_cache.end()) {
    return cached->second;
  }

  auto entry = std::make_shared<KernelEntry>();
  entry->library = std::make_shared<at::native::mps::PrecompiledMetalShaderLibrary>(
      library_path);
  const auto function = [&](const std::string& name) {
    return entry->library->hasFunction(name)
        ? entry->library->getKernelFunction(name)
        : std::shared_ptr<at::native::mps::MetalKernelFunction>{};
  };
  entry->half_kernel = function("llmopt_q8_linear");
  entry->float_kernel = function("llmopt_q8_linear_f32");
  entry->half_dequant_kernel = function("llmopt_q8_dequantize");
  entry->float_dequant_kernel = function("llmopt_q8_dequantize_f32");
  entry->w4a16_kernel = function("llmopt_w4a16_linear_f16_g64");
  if (entry->half_kernel) {
    entry->half_q8_kernels.emplace("llmopt_q8_linear", entry->half_kernel);
  }
  if (entry->float_kernel) {
    entry->float_q8_kernels.emplace("llmopt_q8_linear_f32", entry->float_kernel);
  }
  kernel_cache.emplace(library_path, entry);
  return entry;
}

std::shared_ptr<at::native::mps::MetalKernelFunction> q8_kernel(
    const std::shared_ptr<KernelEntry>& entry,
    bool input_is_half,
    const std::string& name) {
  std::lock_guard<std::mutex> lock(cache_mutex);
  auto& kernels = input_is_half ? entry->half_q8_kernels : entry->float_q8_kernels;
  const auto cached = kernels.find(name);
  if (cached != kernels.end()) {
    return cached->second;
  }
  auto kernel = entry->library->getKernelFunction(name);
  TORCH_CHECK(kernel, "generated Metal library has no ", name, " function");
  kernels.emplace(name, kernel);
  return kernel;
}

bool is_mps(const at::Tensor& tensor) {
  return tensor.device().type() == c10::DeviceType::MPS;
}

uint64_t round_up_to_tile(int64_t value, uint64_t tile) {
  return (static_cast<uint64_t>(value) + tile - 1) / tile * tile;
}

} // namespace

at::Tensor q8_linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& scale,
    pybind11::object bias_object,
    const std::string& library_path,
    bool exact,
    const std::string& kernel_name,
    int64_t tile_m,
    int64_t tile_n,
    int64_t tile_k) {
  TORCH_CHECK(is_mps(input), "llmopt Metal runtime expects an MPS input");
  TORCH_CHECK(is_mps(weight), "llmopt Metal runtime expects an MPS weight");
  TORCH_CHECK(is_mps(scale), "llmopt Metal runtime expects an MPS scale");
  const bool input_is_half = input.scalar_type() == at::kHalf;
  const bool input_is_float = input.scalar_type() == at::kFloat;
  TORCH_CHECK(
      input_is_half || input_is_float,
      "llmopt Metal runtime expects float16 or float32 input");
  TORCH_CHECK(weight.scalar_type() == at::kChar, "llmopt Metal runtime expects int8 weight");
  TORCH_CHECK(scale.scalar_type() == at::kHalf, "llmopt Metal runtime expects float16 scale");
  TORCH_CHECK(input.dim() >= 2, "llmopt Metal runtime expects at least a 2-D input");
  TORCH_CHECK(weight.dim() == 2, "llmopt Metal runtime expects a 2-D weight");
  TORCH_CHECK(scale.numel() == weight.size(0), "Q8 scale length must equal output channels");

  const bool has_bias = !bias_object.is_none();
  at::Tensor bias;
  if (has_bias) {
    bias = bias_object.cast<at::Tensor>();
    TORCH_CHECK(is_mps(bias), "llmopt Metal runtime expects an MPS bias");
    TORCH_CHECK(bias.scalar_type() == at::kHalf, "llmopt Metal runtime expects float16 bias");
    TORCH_CHECK(bias.numel() == weight.size(0), "Q8 bias length must equal output channels");
  }

  const int64_t k = input.size(-1);
  const int64_t n = weight.size(0);
  TORCH_CHECK(weight.size(1) == k, "Q8 input and weight K dimensions must match");
  TORCH_CHECK(k > 0 && n > 0, "Q8 dimensions must be positive");
  TORCH_CHECK(
      tile_m > 0 && tile_n > 0 && tile_k > 0 && tile_k % 4 == 0,
      "Q8 tile dimensions must be positive and tile_k must be a multiple of four");

  at::Tensor input_2d = input.contiguous().reshape({-1, k});
  at::Tensor weight_2d = weight.contiguous();
  at::Tensor scale_1d = scale.contiguous().reshape({n});
  at::Tensor bias_1d = has_bias ? bias.contiguous().reshape({n}) : scale_1d;
  const int64_t m = input_2d.size(0);
  TORCH_CHECK(m > 0, "Q8 input must contain at least one row");
  TORCH_CHECK(m <= UINT32_MAX && n <= UINT32_MAX && k <= UINT32_MAX,
              "Q8 dimensions exceed the Metal runtime ABI");
  std::vector<int64_t> output_shape(input.sizes().begin(), input.sizes().end());
  output_shape.back() = n;

  const auto entry = load_kernel(library_path);

  if (exact) {
    const auto& dequant_kernel = input_is_half ? entry->half_dequant_kernel
                                               : entry->float_dequant_kernel;
    TORCH_CHECK(
        dequant_kernel,
        "generated Metal library has no exact Q8 dequantization kernel: ",
        library_path);
    at::Tensor dequantized_weight = at::empty({n, k}, input.options());
    const Q8Params dequant_params{
        0u,
        static_cast<uint32_t>(n),
        static_cast<uint32_t>(k),
        0u};
    const std::array<uint64_t, 2> dequant_grid = {
        round_up_to_tile(k, kTile), round_up_to_tile(n, kTile)};
    const std::array<uint64_t, 2> dequant_group = {kTile, kTile};

    dequant_kernel->runCommandBlock([&] {
      dequant_kernel->startEncoding();
      dequant_kernel->setArg(0, weight_2d);
      dequant_kernel->setArg(1, scale_1d);
      dequant_kernel->setArg(2, dequantized_weight);
      dequant_kernel->setArg(3, dequant_params);
      dequant_kernel->dispatch(dequant_grid, dequant_group);
    });

    c10::optional<at::Tensor> reference_bias =
        has_bias ? c10::optional<at::Tensor>(bias_1d) : c10::nullopt;
    return at::linear(input_2d, dequantized_weight, reference_bias)
        .reshape(output_shape);
  }

  at::Tensor output_2d = at::empty({m, n}, input.options());
  const auto kernel = q8_kernel(entry, input_is_half, kernel_name);
  TORCH_CHECK(
      kernel,
      "generated Metal library has no float32 Q8 kernel: ",
      library_path);
  const Q8Params params{
      static_cast<uint32_t>(m),
      static_cast<uint32_t>(n),
      static_cast<uint32_t>(k),
      has_bias ? 1u : 0u};
  // dispatchThreads can create a partial final threadgroup when the grid is
  // not tile-aligned. The kernel's cooperative loads need every 16x16 lane
  // to participate, so round the launch grid up and let the kernel bounds
  // checks suppress out-of-range stores for the selected tile geometry.
  const std::array<uint64_t, 2> grid = {
      round_up_to_tile(n, static_cast<uint64_t>(tile_n)),
      round_up_to_tile(m, static_cast<uint64_t>(tile_m))};
  const std::array<uint64_t, 2> group = {
      static_cast<uint64_t>(tile_n), static_cast<uint64_t>(tile_m)};

  kernel->runCommandBlock([&] {
    kernel->startEncoding();
    kernel->setArg(0, input_2d);
    kernel->setArg(1, weight_2d);
    kernel->setArg(2, scale_1d);
    kernel->setArg(3, bias_1d);
    kernel->setArg(4, output_2d);
    kernel->setArg(5, params);
    kernel->dispatch(grid, group);
  });

  return output_2d.reshape(output_shape);
}

at::Tensor w4a16_linear(
    const at::Tensor& input,
    const at::Tensor& packed_weight,
    const at::Tensor& scale,
    pybind11::object bias_object,
    const std::string& library_path) {
  TORCH_CHECK(is_mps(input), "W4A16 Metal runtime expects an MPS input");
  TORCH_CHECK(is_mps(packed_weight), "W4A16 Metal runtime expects an MPS weight");
  TORCH_CHECK(is_mps(scale), "W4A16 Metal runtime expects an MPS scale");
  TORCH_CHECK(input.scalar_type() == at::kHalf,
              "W4A16 Metal runtime expects float16 input");
  TORCH_CHECK(packed_weight.scalar_type() == at::kByte,
              "W4A16 Metal runtime expects uint8 packed weight");
  TORCH_CHECK(scale.scalar_type() == at::kHalf,
              "W4A16 Metal runtime expects float16 scale");
  TORCH_CHECK(input.dim() >= 2, "W4A16 input must have at least two dimensions");
  TORCH_CHECK(packed_weight.dim() == 2, "W4A16 weight must be rank two");
  TORCH_CHECK(scale.dim() == 2, "W4A16 scale must be rank two");

  const int64_t k = input.size(-1);
  const int64_t n = packed_weight.size(0);
  TORCH_CHECK(k > 0 && n > 0 && k % 64 == 0,
              "W4A16 K must be positive and divisible by 64");
  TORCH_CHECK(packed_weight.size(1) == k / 2,
              "W4A16 packed weight must have shape [N,K/2]");
  TORCH_CHECK(scale.size(0) == n && scale.size(1) == k / 64,
              "W4A16 scale must have shape [N,K/64]");

  const bool has_bias = !bias_object.is_none();
  at::Tensor bias;
  if (has_bias) {
    bias = bias_object.cast<at::Tensor>();
    TORCH_CHECK(is_mps(bias), "W4A16 Metal runtime expects an MPS bias");
    TORCH_CHECK(bias.scalar_type() == at::kHalf,
                "W4A16 Metal runtime expects float16 bias");
    TORCH_CHECK(bias.numel() == n,
                "W4A16 bias length must equal output channels");
  }

  at::Tensor input_2d = input.contiguous().reshape({-1, k});
  at::Tensor weight_2d = packed_weight.contiguous();
  at::Tensor scale_2d = scale.contiguous();
  at::Tensor bias_1d = has_bias ? bias.contiguous().reshape({n}) : scale_2d;
  const int64_t m = input_2d.size(0);
  TORCH_CHECK(m > 0, "W4A16 input must contain at least one row");
  TORCH_CHECK(m <= UINT32_MAX && n <= UINT32_MAX && k <= UINT32_MAX,
              "W4A16 dimensions exceed the Metal runtime ABI");

  std::vector<int64_t> output_shape(input.sizes().begin(), input.sizes().end());
  output_shape.back() = n;
  at::Tensor output_2d = at::empty({m, n}, input.options());
  const auto entry = load_kernel(library_path);
  TORCH_CHECK(entry->w4a16_kernel,
              "generated Metal library has no llmopt_w4a16_linear_f16_g64 function: ",
              library_path);
  const Q8Params params{
      static_cast<uint32_t>(m), static_cast<uint32_t>(n),
      static_cast<uint32_t>(k), has_bias ? 1u : 0u};
  const std::array<uint64_t, 2> grid = {round_up_to_tile(m * n, 256), 1};
  const std::array<uint64_t, 2> group = {256, 1};

  entry->w4a16_kernel->runCommandBlock([&] {
    entry->w4a16_kernel->startEncoding();
    entry->w4a16_kernel->setArg(0, input_2d);
    entry->w4a16_kernel->setArg(1, weight_2d);
    entry->w4a16_kernel->setArg(2, scale_2d);
    entry->w4a16_kernel->setArg(3, bias_1d);
    entry->w4a16_kernel->setArg(4, output_2d);
    entry->w4a16_kernel->setArg(5, params);
    entry->w4a16_kernel->dispatch(grid, group);
  });

  return output_2d.reshape(output_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "w4a16_linear",
      &w4a16_linear,
      pybind11::arg("input"),
      pybind11::arg("packed_weight"),
      pybind11::arg("scale"),
      pybind11::arg("bias"),
      pybind11::arg("library_path"));
  module.def(
      "q8_linear",
      &q8_linear,
      pybind11::arg("input"),
      pybind11::arg("weight"),
      pybind11::arg("scale"),
      pybind11::arg("bias"),
      pybind11::arg("library_path"),
      pybind11::arg("exact") = false,
      pybind11::arg("kernel_name") = "llmopt_q8_linear",
      pybind11::arg("tile_m") = 16,
      pybind11::arg("tile_n") = 16,
      pybind11::arg("tile_k") = 64);
}
