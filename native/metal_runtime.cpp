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
  std::shared_ptr<at::native::mps::MetalKernelFunction> kernel;
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
  entry->kernel = entry->library->getKernelFunction("llmopt_q8_linear");
  TORCH_CHECK(
      entry->kernel,
      "generated Metal library has no llmopt_q8_linear function: ",
      library_path);
  kernel_cache.emplace(library_path, entry);
  return entry;
}

bool is_mps(const at::Tensor& tensor) {
  return tensor.device().type() == c10::DeviceType::MPS;
}

uint64_t round_up_to_tile(int64_t value) {
  const uint64_t tile = kTile;
  return (static_cast<uint64_t>(value) + tile - 1) / tile * tile;
}

} // namespace

at::Tensor q8_linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& scale,
    pybind11::object bias_object,
    const std::string& library_path) {
  TORCH_CHECK(is_mps(input), "llmopt Metal runtime expects an MPS input");
  TORCH_CHECK(is_mps(weight), "llmopt Metal runtime expects an MPS weight");
  TORCH_CHECK(is_mps(scale), "llmopt Metal runtime expects an MPS scale");
  TORCH_CHECK(input.scalar_type() == at::kHalf, "llmopt Metal runtime expects float16 input");
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

  at::Tensor input_2d = input.contiguous().reshape({-1, k});
  at::Tensor weight_2d = weight.contiguous();
  at::Tensor scale_1d = scale.contiguous().reshape({n});
  at::Tensor bias_1d = has_bias ? bias.contiguous().reshape({n}) : scale_1d;
  const int64_t m = input_2d.size(0);
  TORCH_CHECK(m > 0, "Q8 input must contain at least one row");
  TORCH_CHECK(m <= UINT32_MAX && n <= UINT32_MAX && k <= UINT32_MAX,
              "Q8 dimensions exceed the Metal runtime ABI");

  at::Tensor output_2d = at::empty({m, n}, input.options());
  const auto entry = load_kernel(library_path);
  const Q8Params params{
      static_cast<uint32_t>(m),
      static_cast<uint32_t>(n),
      static_cast<uint32_t>(k),
      has_bias ? 1u : 0u};
  // dispatchThreads can create a partial final threadgroup when the grid is
  // not tile-aligned. The kernel's cooperative loads need every 16x16 lane
  // to participate, so round the launch grid up and let the kernel bounds
  // checks suppress out-of-range stores.
  const std::array<uint64_t, 2> grid = {
      round_up_to_tile(n), round_up_to_tile(m)};
  const std::array<uint64_t, 2> group = {kTile, kTile};

  entry->kernel->runCommandBlock([&] {
    entry->kernel->startEncoding();
    entry->kernel->setArg(0, input_2d);
    entry->kernel->setArg(1, weight_2d);
    entry->kernel->setArg(2, scale_1d);
    entry->kernel->setArg(3, bias_1d);
    entry->kernel->setArg(4, output_2d);
    entry->kernel->setArg(5, params);
    entry->kernel->dispatch(grid, group);
  });

  std::vector<int64_t> output_shape(input.sizes().begin(), input.sizes().end());
  output_shape.back() = n;
  return output_2d.reshape(output_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "q8_linear",
      &q8_linear,
      pybind11::arg("input"),
      pybind11::arg("weight"),
      pybind11::arg("scale"),
      pybind11::arg("bias"),
      pybind11::arg("library_path"));
}
