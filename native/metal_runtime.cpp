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

struct W4A16Params {
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t has_bias;
};

struct KernelEntry {
  std::shared_ptr<at::native::mps::PrecompiledMetalShaderLibrary> library;
  std::shared_ptr<at::native::mps::MetalKernelFunction> w4a16_kernel;
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
  entry->w4a16_kernel = function("llmopt_w4a16_linear_f16_g64");
  kernel_cache.emplace(library_path, entry);
  return entry;
}

bool is_mps(const at::Tensor& tensor) {
  return tensor.device().type() == c10::DeviceType::MPS;
}

uint64_t round_up_to_tile(int64_t value, uint64_t tile) {
  return (static_cast<uint64_t>(value) + tile - 1) / tile * tile;
}

} // namespace

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
  const W4A16Params params{
      static_cast<uint32_t>(m), static_cast<uint32_t>(n),
      static_cast<uint32_t>(k), has_bias ? 1u : 0u};
  const uint64_t total_threads = static_cast<uint64_t>(m) * n * 32;
  const std::array<uint64_t, 2> grid = {round_up_to_tile(total_threads, 256), 1};
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
}
