#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(VkResult result, const char *operation) {
  if (result != VK_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with Vulkan status " +
                             std::to_string(result));
  }
}

std::vector<uint32_t> read_spirv(const std::string &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("cannot open SPIR-V module: " + path);
  const auto size = input.tellg();
  if (size <= 0 || (size % 4) != 0) {
    throw std::runtime_error("invalid SPIR-V module length: " + path);
  }
  std::vector<uint32_t> words(static_cast<size_t>(size) / 4);
  input.seekg(0);
  input.read(reinterpret_cast<char *>(words.data()), size);
  if (!input) throw std::runtime_error("cannot read SPIR-V module: " + path);
  return words;
}

std::string json_string(const char *text) {
  std::string result = "\"";
  for (const unsigned char c : std::string(text)) {
    if (c == '\\' || c == '"') result.push_back('\\');
    result.push_back(static_cast<char>(c));
  }
  result.push_back('"');
  return result;
}

bool has_extension(const std::vector<VkExtensionProperties> &extensions,
                   const char *name) {
  return std::any_of(extensions.begin(), extensions.end(), [name](const auto &extension) {
    return std::strcmp(extension.extensionName, name) == 0;
  });
}

struct Buffer {
  VkDevice device = VK_NULL_HANDLE;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
  VkMemoryPropertyFlags properties = 0;

  Buffer() = default;
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&other) noexcept { *this = std::move(other); }
  Buffer &operator=(Buffer &&other) noexcept {
    if (this != &other) {
      release();
      device = other.device;
      buffer = other.buffer;
      memory = other.memory;
      size = other.size;
      properties = other.properties;
      other.buffer = VK_NULL_HANDLE;
      other.memory = VK_NULL_HANDLE;
    }
    return *this;
  }
  ~Buffer() { release(); }
  void release() {
    if (buffer) vkDestroyBuffer(device, buffer, nullptr);
    if (memory) vkFreeMemory(device, memory, nullptr);
    buffer = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
  }
};

struct Context {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queue_family = 0;
  VkPhysicalDeviceProperties properties{};
  VkPhysicalDeviceMemoryProperties memory{};
  VkCommandPool command_pool = VK_NULL_HANDLE;

  Context() = default;
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  Context(Context &&other) noexcept { *this = std::move(other); }
  Context &operator=(Context &&other) noexcept {
    if (this != &other) {
      instance = other.instance;
      physical = other.physical;
      device = other.device;
      queue = other.queue;
      queue_family = other.queue_family;
      properties = other.properties;
      memory = other.memory;
      command_pool = other.command_pool;
      other.instance = VK_NULL_HANDLE;
      other.device = VK_NULL_HANDLE;
      other.command_pool = VK_NULL_HANDLE;
    }
    return *this;
  }

  ~Context() {
    if (device) vkDeviceWaitIdle(device);
    if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
    if (device) vkDestroyDevice(device, nullptr);
    if (instance) vkDestroyInstance(instance, nullptr);
  }
};

Context create_context() {
  Context context;
  uint32_t instance_extension_count = 0;
  check(vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count, nullptr),
        "enumerate instance extensions");
  std::vector<VkExtensionProperties> instance_extensions(instance_extension_count);
  check(vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count,
                                               instance_extensions.data()),
        "enumerate instance extensions");
  std::vector<const char *> enabled_instance_extensions;
  VkInstanceCreateFlags instance_flags = 0;
  if (has_extension(instance_extensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
    enabled_instance_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    instance_flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
  }
  VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  application.pApplicationName = "llmopt-vulkan-probe";
  application.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  application.pEngineName = "llmopt";
  application.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  application.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  instance_info.flags = instance_flags;
  instance_info.pApplicationInfo = &application;
  instance_info.enabledExtensionCount =
      static_cast<uint32_t>(enabled_instance_extensions.size());
  instance_info.ppEnabledExtensionNames = enabled_instance_extensions.data();
  check(vkCreateInstance(&instance_info, nullptr, &context.instance), "create instance");

  uint32_t physical_count = 0;
  check(vkEnumeratePhysicalDevices(context.instance, &physical_count, nullptr),
        "enumerate physical devices");
  if (physical_count == 0) throw std::runtime_error("no Vulkan physical device found");
  std::vector<VkPhysicalDevice> physical_devices(physical_count);
  check(vkEnumeratePhysicalDevices(context.instance, &physical_count, physical_devices.data()),
        "enumerate physical devices");
  for (const auto physical : physical_devices) {
    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families.data());
    for (uint32_t index = 0; index < family_count; ++index) {
      if ((families[index].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
          families[index].timestampValidBits > 0) {
        context.physical = physical;
        context.queue_family = index;
        break;
      }
    }
    if (context.physical) break;
  }
  if (!context.physical) {
    throw std::runtime_error("no compute queue with timestamp support found");
  }
  vkGetPhysicalDeviceProperties(context.physical, &context.properties);
  vkGetPhysicalDeviceMemoryProperties(context.physical, &context.memory);

  uint32_t device_extension_count = 0;
  check(vkEnumerateDeviceExtensionProperties(context.physical, nullptr,
                                             &device_extension_count, nullptr),
        "enumerate device extensions");
  std::vector<VkExtensionProperties> device_extensions(device_extension_count);
  check(vkEnumerateDeviceExtensionProperties(context.physical, nullptr,
                                             &device_extension_count,
                                             device_extensions.data()),
        "enumerate device extensions");
  std::vector<const char *> enabled_device_extensions;
  constexpr const char *portability_subset = "VK_KHR_portability_subset";
  if (has_extension(device_extensions, portability_subset)) {
    enabled_device_extensions.push_back(portability_subset);
  }
  const float priority = 1.0f;
  VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queue_info.queueFamilyIndex = context.queue_family;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &priority;
  VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;
  device_info.enabledExtensionCount = static_cast<uint32_t>(enabled_device_extensions.size());
  device_info.ppEnabledExtensionNames = enabled_device_extensions.data();
  check(vkCreateDevice(context.physical, &device_info, nullptr, &context.device),
        "create device");
  vkGetDeviceQueue(context.device, context.queue_family, 0, &context.queue);
  VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = context.queue_family;
  check(vkCreateCommandPool(context.device, &pool_info, nullptr, &context.command_pool),
        "create command pool");
  return context;
}

uint32_t memory_type(const Context &context, uint32_t bits,
                     VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred = 0) {
  uint32_t fallback = std::numeric_limits<uint32_t>::max();
  for (uint32_t index = 0; index < context.memory.memoryTypeCount; ++index) {
    const auto flags = context.memory.memoryTypes[index].propertyFlags;
    if ((bits & (1u << index)) && (flags & required) == required) {
      if ((flags & preferred) == preferred) return index;
      if (fallback == std::numeric_limits<uint32_t>::max()) fallback = index;
    }
  }
  if (fallback == std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("no compatible Vulkan memory type");
  }
  return fallback;
}

Buffer create_buffer(const Context &context, VkDeviceSize size, VkBufferUsageFlags usage,
                     VkMemoryPropertyFlags required,
                     VkMemoryPropertyFlags preferred = 0) {
  Buffer result;
  result.device = context.device;
  result.size = size;
  VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  info.size = size;
  info.usage = usage;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  check(vkCreateBuffer(context.device, &info, nullptr, &result.buffer), "create buffer");
  VkMemoryRequirements requirements{};
  vkGetBufferMemoryRequirements(context.device, result.buffer, &requirements);
  const uint32_t type = memory_type(context, requirements.memoryTypeBits, required, preferred);
  result.properties = context.memory.memoryTypes[type].propertyFlags;
  VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocation.allocationSize = requirements.size;
  allocation.memoryTypeIndex = type;
  check(vkAllocateMemory(context.device, &allocation, nullptr, &result.memory),
        "allocate buffer memory");
  check(vkBindBufferMemory(context.device, result.buffer, result.memory, 0),
        "bind buffer memory");
  return result;
}

VkShaderModule create_shader(VkDevice device, const std::string &path) {
  const auto words = read_spirv(path);
  VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  info.codeSize = words.size() * sizeof(uint32_t);
  info.pCode = words.data();
  VkShaderModule shader = VK_NULL_HANDLE;
  check(vkCreateShaderModule(device, &info, nullptr, &shader), "create shader module");
  return shader;
}

VkCommandBuffer allocate_command(const Context &context) {
  VkCommandBufferAllocateInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  info.commandPool = context.command_pool;
  info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  info.commandBufferCount = 1;
  VkCommandBuffer command = VK_NULL_HANDLE;
  check(vkAllocateCommandBuffers(context.device, &info, &command), "allocate command buffer");
  return command;
}

void submit_and_wait(const Context &context, VkCommandBuffer command) {
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &command;
  check(vkQueueSubmit(context.queue, 1, &submit, VK_NULL_HANDLE), "submit command buffer");
  check(vkQueueWaitIdle(context.queue), "wait for queue");
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) * 0.5;
}

double measure(const Context &context, VkPipeline pipeline, VkPipelineLayout layout,
               VkDescriptorSet descriptors, uint32_t vector_count, uint32_t groups_x,
               uint32_t groups_y, int warmup, int repeats) {
  VkQueryPoolCreateInfo query_info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
  query_info.queryCount = 2;
  VkQueryPool queries = VK_NULL_HANDLE;
  check(vkCreateQueryPool(context.device, &query_info, nullptr, &queries),
        "create timestamp query pool");
  VkCommandBuffer command = allocate_command(context);
  std::vector<double> samples;
  for (int iteration = 0; iteration < warmup + repeats; ++iteration) {
    check(vkResetCommandBuffer(command, 0), "reset command buffer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command, &begin), "begin command buffer");
    vkCmdResetQueryPool(command, queries, 0, 2);
    vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    if (descriptors) {
      vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1,
                              &descriptors, 0, nullptr);
      vkCmdPushConstants(command, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                         sizeof(vector_count), &vector_count);
    }
    vkCmdDispatch(command, groups_x, groups_y, 1);
    vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
    check(vkEndCommandBuffer(command), "end command buffer");
    submit_and_wait(context, command);
    uint64_t timestamps[2]{};
    check(vkGetQueryPoolResults(context.device, queries, 0, 2, sizeof(timestamps),
                                timestamps, sizeof(uint64_t),
                                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
          "read timestamp queries");
    if (iteration >= warmup) {
      const double seconds = static_cast<double>(timestamps[1] - timestamps[0]) *
                             context.properties.limits.timestampPeriod * 1e-9;
      samples.push_back(seconds);
    }
  }
  vkFreeCommandBuffers(context.device, context.command_pool, 1, &command);
  vkDestroyQueryPool(context.device, queries, nullptr);
  return median(std::move(samples));
}

struct Measurement {
  uint64_t bytes;
  double seconds;
  double bandwidth_gbps;
};

Measurement measure_copy(const Context &context, VkPipeline pipeline,
                         VkPipelineLayout layout, VkDescriptorSetLayout descriptor_layout,
                         uint64_t bytes, int warmup, int repeats) {
  const auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  Buffer source = create_buffer(context, bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  Buffer destination =
      create_buffer(context, bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  Buffer staging = create_buffer(context, 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
  VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pool_info.maxSets = 1;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &pool_size;
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  check(vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool),
        "create descriptor pool");
  VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  set_info.descriptorPool = descriptor_pool;
  set_info.descriptorSetCount = 1;
  set_info.pSetLayouts = &descriptor_layout;
  VkDescriptorSet descriptors = VK_NULL_HANDLE;
  check(vkAllocateDescriptorSets(context.device, &set_info, &descriptors),
        "allocate descriptor set");
  VkDescriptorBufferInfo source_info{source.buffer, 0, bytes};
  VkDescriptorBufferInfo destination_info{destination.buffer, 0, bytes};
  VkWriteDescriptorSet writes[2]{};
  for (auto &write : writes) write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].dstSet = descriptors;
  writes[0].dstBinding = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[0].pBufferInfo = &source_info;
  writes[1].dstSet = descriptors;
  writes[1].dstBinding = 1;
  writes[1].descriptorCount = 1;
  writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[1].pBufferInfo = &destination_info;
  vkUpdateDescriptorSets(context.device, 2, writes, 0, nullptr);

  VkCommandBuffer initialize = allocate_command(context);
  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check(vkBeginCommandBuffer(initialize, &begin), "begin initialization command");
  vkCmdFillBuffer(initialize, source.buffer, 0, bytes, 0x5a17c0deu);
  vkCmdFillBuffer(initialize, destination.buffer, 0, bytes, 0u);
  check(vkEndCommandBuffer(initialize), "end initialization command");
  submit_and_wait(context, initialize);
  vkFreeCommandBuffers(context.device, context.command_pool, 1, &initialize);

  const uint32_t vector_count = static_cast<uint32_t>(bytes / 16);
  const uint32_t total_groups = (vector_count + 255) / 256;
  const uint32_t groups_x = std::min(total_groups,
                                     context.properties.limits.maxComputeWorkGroupCount[0]);
  const uint32_t groups_y = (total_groups + groups_x - 1) / groups_x;
  if (groups_y > context.properties.limits.maxComputeWorkGroupCount[1]) {
    throw std::runtime_error("copy dispatch exceeds Vulkan workgroup-count limits");
  }
  const double seconds = measure(context, pipeline, layout, descriptors, vector_count,
                                 groups_x, groups_y, warmup, repeats);

  VkCommandBuffer validate = allocate_command(context);
  check(vkBeginCommandBuffer(validate, &begin), "begin validation command");
  VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.buffer = destination.buffer;
  barrier.offset = 0;
  barrier.size = 16;
  vkCmdPipelineBarrier(validate, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &barrier, 0,
                       nullptr);
  VkBufferCopy copy{0, 0, 16};
  vkCmdCopyBuffer(validate, destination.buffer, staging.buffer, 1, &copy);
  check(vkEndCommandBuffer(validate), "end validation command");
  submit_and_wait(context, validate);
  vkFreeCommandBuffers(context.device, context.command_pool, 1, &validate);
  void *mapped = nullptr;
  check(vkMapMemory(context.device, staging.memory, 0, 16, 0, &mapped),
        "map validation buffer");
  if (!(staging.properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = staging.memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    check(vkInvalidateMappedMemoryRanges(context.device, 1, &range),
          "invalidate validation buffer");
  }
  const auto *words = static_cast<const uint32_t *>(mapped);
  const bool correct = std::all_of(words, words + 4,
                                   [](uint32_t value) { return value == 0x5a17c0deu; });
  vkUnmapMemory(context.device, staging.memory);
  vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
  if (!correct) throw std::runtime_error("Vulkan copy validation failed");
  return {bytes, seconds, (2.0 * static_cast<double>(bytes)) / seconds / 1e9};
}

VkPipeline create_pipeline(VkDevice device, VkShaderModule shader, VkPipelineLayout layout) {
  VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = shader;
  stage.pName = "main";
  VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  info.stage = stage;
  info.layout = layout;
  VkPipeline pipeline = VK_NULL_HANDLE;
  check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline),
        "create compute pipeline");
  return pipeline;
}

struct Options {
  std::string copy_shader;
  std::string empty_shader;
  std::string output;
  int warmup = 5;
  int repeats = 21;
  std::vector<uint64_t> byte_counts{16ull << 20, 64ull << 20, 256ull << 20};
};

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto value = [&]() -> std::string {
      if (++index >= argc) throw std::runtime_error("missing value after " + argument);
      return argv[index];
    };
    if (argument == "--copy-shader") options.copy_shader = value();
    else if (argument == "--empty-shader") options.empty_shader = value();
    else if (argument == "--output") options.output = value();
    else if (argument == "--warmup") options.warmup = std::stoi(value());
    else if (argument == "--repeats") options.repeats = std::stoi(value());
    else if (argument == "--bytes") options.byte_counts = {std::stoull(value())};
    else throw std::runtime_error("unknown argument: " + argument);
  }
  if (options.copy_shader.empty() || options.empty_shader.empty() || options.output.empty()) {
    throw std::runtime_error(
        "usage: llmopt-vulkan-probe --copy-shader path --empty-shader path "
        "--output path [--warmup count] [--repeats count] [--bytes count]");
  }
  if (options.warmup < 0 || options.repeats <= 0) {
    throw std::runtime_error("warmup must be non-negative and repeats must be positive");
  }
  return options;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    Context context = create_context();
    VkDescriptorSetLayoutBinding bindings[2]{};
    for (uint32_t index = 0; index < 2; ++index) {
      bindings[index].binding = index;
      bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[index].descriptorCount = 1;
      bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo descriptor_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptor_info.bindingCount = 2;
    descriptor_info.pBindings = bindings;
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    check(vkCreateDescriptorSetLayout(context.device, &descriptor_info, nullptr,
                                      &descriptor_layout),
          "create descriptor layout");
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo copy_layout_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    copy_layout_info.setLayoutCount = 1;
    copy_layout_info.pSetLayouts = &descriptor_layout;
    copy_layout_info.pushConstantRangeCount = 1;
    copy_layout_info.pPushConstantRanges = &push;
    VkPipelineLayout copy_layout = VK_NULL_HANDLE;
    check(vkCreatePipelineLayout(context.device, &copy_layout_info, nullptr, &copy_layout),
          "create copy pipeline layout");
    VkPipelineLayoutCreateInfo empty_layout_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPipelineLayout empty_layout = VK_NULL_HANDLE;
    check(vkCreatePipelineLayout(context.device, &empty_layout_info, nullptr, &empty_layout),
          "create empty pipeline layout");
    VkShaderModule copy_shader = create_shader(context.device, options.copy_shader);
    VkShaderModule empty_shader = create_shader(context.device, options.empty_shader);
    VkPipeline copy_pipeline = create_pipeline(context.device, copy_shader, copy_layout);
    VkPipeline empty_pipeline = create_pipeline(context.device, empty_shader, empty_layout);

    const double dispatch_seconds = measure(context, empty_pipeline, empty_layout,
                                            VK_NULL_HANDLE, 0, 1, 1, options.warmup,
                                            options.repeats);
    std::vector<Measurement> measurements;
    for (const uint64_t bytes : options.byte_counts) {
      if (bytes < 4096 || (bytes % 16) != 0 || bytes / 16 > UINT32_MAX) {
        throw std::runtime_error("stream byte count must be 16-byte aligned and valid");
      }
      measurements.push_back(measure_copy(context, copy_pipeline, copy_layout,
                                          descriptor_layout, bytes, options.warmup,
                                          options.repeats));
    }

    std::ofstream output(options.output);
    if (!output) throw std::runtime_error("cannot write receipt: " + options.output);
    output << std::setprecision(17);
    output << "{\n  \"schema_version\": 1,\n  \"backend\": \"vulkan\",\n"
           << "  \"implementation\": \""
#if defined(__APPLE__)
           << "moltenvk"
#else
           << "native-vulkan"
#endif
           << "\",\n  \"device_name\": " << json_string(context.properties.deviceName)
           << ",\n  \"api_version\": \"" << VK_VERSION_MAJOR(context.properties.apiVersion)
           << "." << VK_VERSION_MINOR(context.properties.apiVersion) << "."
           << VK_VERSION_PATCH(context.properties.apiVersion) << "\",\n"
           << "  \"measurement\": {\n    \"warmup\": " << options.warmup
           << ",\n    \"repeats\": " << options.repeats
           << ",\n    \"dispatch_overhead_seconds\": " << dispatch_seconds
           << ",\n    \"memory_bandwidth_gbps\": "
           << measurements.back().bandwidth_gbps
           << ",\n    \"memory_bandwidth_stream_bytes\": " << measurements.back().bytes
           << ",\n    \"copy_sample_validated\": true,\n    \"streaming_copy_sweep\": [\n";
    for (size_t index = 0; index < measurements.size(); ++index) {
      const auto &measurement = measurements[index];
      output << "      {\"stream_bytes\": " << measurement.bytes
             << ", \"gpu_seconds\": " << measurement.seconds
             << ", \"effective_bandwidth_gbps\": " << measurement.bandwidth_gbps
             << "}" << (index + 1 == measurements.size() ? "\n" : ",\n");
    }
    output << "    ]\n  }\n}\n";
    output.close();

    std::cout << std::fixed << std::setprecision(3) << "device="
              << context.properties.deviceName << " backend=vulkan implementation="
#if defined(__APPLE__)
              << "moltenvk"
#else
              << "native-vulkan"
#endif
              << " dispatch_us=" << dispatch_seconds * 1e6
              << " sustained_effective_bandwidth_gbps="
              << measurements.back().bandwidth_gbps << " stream_bytes="
              << measurements.back().bytes << " output=" << options.output << "\n";
    for (const auto &measurement : measurements) {
      std::cout << "stream_bytes=" << measurement.bytes
                << " effective_bandwidth_gbps=" << measurement.bandwidth_gbps << "\n";
    }

    vkDestroyPipeline(context.device, empty_pipeline, nullptr);
    vkDestroyPipeline(context.device, copy_pipeline, nullptr);
    vkDestroyShaderModule(context.device, empty_shader, nullptr);
    vkDestroyShaderModule(context.device, copy_shader, nullptr);
    vkDestroyPipelineLayout(context.device, empty_layout, nullptr);
    vkDestroyPipelineLayout(context.device, copy_layout, nullptr);
    vkDestroyDescriptorSetLayout(context.device, descriptor_layout, nullptr);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "llmopt-vulkan-probe: " << error.what() << "\n";
    return 1;
  }
}
