#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <mach/mach_time.h>

typedef struct {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
} llmopt_metal_context;

#define LLMOPT_PIPELINE_CACHE_SIZE 64

typedef struct {
  const char *name;
  id<MTLComputePipelineState> pipeline;
} llmopt_pipeline_entry;

typedef struct {
  id<MTLDevice> device;
  id<MTLLibrary> library;
  id<MTLCommandQueue> queue;
  NSMutableDictionary *pipelines;
  llmopt_pipeline_entry cache[LLMOPT_PIPELINE_CACHE_SIZE];
  uint32_t cache_count;
} llmopt_metal_library;

typedef struct {
  id<MTLBuffer> buffer;
  NSUInteger offset;
  NSUInteger length;
} llmopt_metal_buffer;

typedef struct {
  id<MTLCommandBuffer> command;
  id<MTLComputeCommandEncoder> compute;
  BOOL finished;
} llmopt_metal_batch;

typedef struct {
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t has_bias;
} llmopt_q8_params;

#include <stdatomic.h>
#include <pthread.h>

#define LLMOPT_RING_CAPACITY 256

typedef struct {
  uint32_t request_id;
  int32_t token;
  int32_t past_tokens;
  uint32_t flags;
} llmopt_sq_entry;

typedef struct {
  uint32_t request_id;
  int32_t token;
  uint32_t status;
} llmopt_cq_entry;

typedef struct {
  id<MTLComputePipelineState> pipeline;
  NSUInteger buffer_count;
  id<MTLBuffer> buffers[16];
  NSUInteger offsets[16];
  NSUInteger lengths[16];
  char kernel_name[128];
  uint8_t parameters[256];
  NSUInteger parameter_length;
  MTLSize grid;
  MTLSize group;
  bool barrier_before;
  bool is_paged_attention;
  bool is_checkpoint_pack;
  int8_t token_buffer_index;
  int8_t output_buffer_index;
} llmopt_dispatch_record;

typedef struct {
  _Atomic uint32_t sq_head;
  _Atomic uint32_t sq_tail;
  llmopt_sq_entry sq[LLMOPT_RING_CAPACITY];

  _Atomic uint32_t cq_head;
  _Atomic uint32_t cq_tail;
  llmopt_cq_entry cq[LLMOPT_RING_CAPACITY];

  pthread_mutex_t mutex;
  pthread_cond_t sq_cond;
  pthread_cond_t cq_cond;
  _Atomic uint32_t running;

  pthread_t worker_thread;
  bool worker_started;
  id<MTLCommandQueue> queue;
  llmopt_metal_library *library;
  llmopt_dispatch_record *records;
  size_t record_count;
  id<MTLBuffer> token_buffers[2];
  NSUInteger token_buffer_offsets[2];
  id<MTLBuffer> output_buffers[2];
  NSUInteger output_buffer_offsets[2];
  id<MTLComputePipelineState> chain_pipeline;
} llmopt_ring_queue;

typedef struct {
  id<MTLCommandQueue> queue;
  llmopt_metal_library *library;
  llmopt_dispatch_record *records;
  size_t record_count;
  size_t barrier_count;
  id<MTLBuffer> token_buffer;
  NSUInteger token_buffer_offset;
  id<MTLBuffer> output_buffer;
  NSUInteger output_buffer_offset;
} llmopt_prebaked_plan;

static bool is_paged_attention_kernel(const char *name) {
  static const char prefix[] = "llmopt_attention_q8_paged_";
  return strncmp(name, prefix, sizeof(prefix) - 1) == 0;
}

static llmopt_prebaked_plan *Prebaked_val(value handle) {
  return *((llmopt_prebaked_plan **)Data_custom_val(handle));
}

static llmopt_ring_queue *Ring_val(value handle) {
  return *((llmopt_ring_queue **)Data_custom_val(handle));
}

static llmopt_metal_context *Context_val(value handle) {
  return *((llmopt_metal_context **)Data_custom_val(handle));
}

static llmopt_metal_library *Library_val(value handle) {
  return *((llmopt_metal_library **)Data_custom_val(handle));
}

static llmopt_metal_buffer *Buffer_val(value handle) {
  return *((llmopt_metal_buffer **)Data_custom_val(handle));
}

static llmopt_metal_batch *Batch_val(value handle) {
  return *((llmopt_metal_batch **)Data_custom_val(handle));
}

static void finalize_context(value handle) {
  llmopt_metal_context **slot =
      (llmopt_metal_context **)Data_custom_val(handle);
  if (*slot != NULL) {
    [(*slot)->queue release];
    [(*slot)->device release];
    free(*slot);
    *slot = NULL;
  }
}

static void finalize_library(value handle) {
  llmopt_metal_library **slot =
      (llmopt_metal_library **)Data_custom_val(handle);
  if (*slot != NULL) {
    for (uint32_t i = 0; i < (*slot)->cache_count; i++) {
      free((void *)(*slot)->cache[i].name);
    }
    [(*slot)->pipelines release];
    [(*slot)->queue release];
    [(*slot)->library release];
    [(*slot)->device release];
    free(*slot);
    *slot = NULL;
  }
}

static void finalize_buffer(value handle) {
  llmopt_metal_buffer **slot = (llmopt_metal_buffer **)Data_custom_val(handle);
  if (*slot != NULL) {
    [(*slot)->buffer release];
    free(*slot);
    *slot = NULL;
  }
}

static void end_batch_compute(llmopt_metal_batch *batch) {
  if (batch->compute != nil) {
    [batch->compute endEncoding];
    [batch->compute release];
    batch->compute = nil;
  }
}

static void finalize_batch(value handle) {
  llmopt_metal_batch **slot =
      (llmopt_metal_batch **)Data_custom_val(handle);
  if (*slot != NULL) {
    end_batch_compute(*slot);
    [(*slot)->command release];
    free(*slot);
    *slot = NULL;
  }
}

static struct custom_operations context_operations = {
    "llmopt.metal.context",
    finalize_context,
    custom_compare_default,
    custom_hash_default,
    custom_serialize_default,
    custom_deserialize_default,
    custom_compare_ext_default,
    custom_fixed_length_default};

static struct custom_operations library_operations = {
    "llmopt.metal.library",
    finalize_library,
    custom_compare_default,
    custom_hash_default,
    custom_serialize_default,
    custom_deserialize_default,
    custom_compare_ext_default,
    custom_fixed_length_default};

static struct custom_operations buffer_operations = {
    "llmopt.metal.buffer",
    finalize_buffer,
    custom_compare_default,
    custom_hash_default,
    custom_serialize_default,
    custom_deserialize_default,
    custom_compare_ext_default,
    custom_fixed_length_default};

static struct custom_operations batch_operations = {
    "llmopt.metal.batch",
    finalize_batch,
    custom_compare_default,
    custom_hash_default,
    custom_serialize_default,
    custom_deserialize_default,
    custom_compare_ext_default,
    custom_fixed_length_default};

static value alloc_context(llmopt_metal_context *context) {
  value result =
      caml_alloc_custom(&context_operations, sizeof(context), 0, 1);
  *((llmopt_metal_context **)Data_custom_val(result)) = context;
  return result;
}

static value alloc_library(llmopt_metal_library *library) {
  value result =
      caml_alloc_custom(&library_operations, sizeof(library), 0, 1);
  *((llmopt_metal_library **)Data_custom_val(result)) = library;
  return result;
}

static value alloc_buffer(llmopt_metal_buffer *buffer) {
  value result = caml_alloc_custom(&buffer_operations, sizeof(buffer), 0, 1);
  *((llmopt_metal_buffer **)Data_custom_val(result)) = buffer;
  return result;
}

static value alloc_batch(llmopt_metal_batch *batch) {
  value result =
      caml_alloc_custom(&batch_operations, sizeof(batch), 0, 1);
  *((llmopt_metal_batch **)Data_custom_val(result)) = batch;
  return result;
}

static void finalize_ring(value handle) {
  llmopt_ring_queue **slot = (llmopt_ring_queue **)Data_custom_val(handle);
  if (*slot != NULL) {
    llmopt_ring_queue *q = *slot;
    atomic_store_explicit(&q->running, 0, memory_order_release);
    pthread_mutex_lock(&q->mutex);
    pthread_cond_broadcast(&q->sq_cond);
    pthread_cond_broadcast(&q->cq_cond);
    pthread_mutex_unlock(&q->mutex);
    if (q->worker_started) {
      pthread_join(q->worker_thread, NULL);
    }
    if (q->records != NULL) {
      for (size_t i = 0; i < q->record_count; i++) {
        [q->records[i].pipeline release];
        for (NSUInteger b = 0; b < q->records[i].buffer_count; b++) {
          [q->records[i].buffers[b] release];
        }
      }
      free(q->records);
    }
    [q->queue release];
    if (q->token_buffers[0] != nil) [q->token_buffers[0] release];
    if (q->token_buffers[1] != nil) [q->token_buffers[1] release];
    if (q->output_buffers[0] != nil) [q->output_buffers[0] release];
    if (q->output_buffers[1] != nil) [q->output_buffers[1] release];
    [q->chain_pipeline release];
    pthread_mutex_destroy(&q->mutex);
    pthread_cond_destroy(&q->sq_cond);
    pthread_cond_destroy(&q->cq_cond);
    free(q);
    *slot = NULL;
  }
}

static struct custom_operations ring_operations = {
    "llmopt.ring.queue",
    finalize_ring,
    custom_compare_default,
    custom_hash_default,
    custom_serialize_default,
    custom_deserialize_default,
    custom_compare_ext_default,
    custom_fixed_length_default};

static value alloc_ring(llmopt_ring_queue *ring) {
  value result = caml_alloc_custom(&ring_operations, sizeof(ring), 0, 1);
  *((llmopt_ring_queue **)Data_custom_val(result)) = ring;
  return result;
}

static void finalize_prebaked(value v) {
  llmopt_prebaked_plan **slot = (llmopt_prebaked_plan **)Data_custom_val(v);
  llmopt_prebaked_plan *plan = *slot;
  if (plan != NULL) {
    if (plan->records != NULL) {
      for (size_t i = 0; i < plan->record_count; i++) {
        [plan->records[i].pipeline release];
        for (NSUInteger b = 0; b < plan->records[i].buffer_count; b++) {
          [plan->records[i].buffers[b] release];
        }
      }
      free(plan->records);
    }
    [plan->queue release];
    [plan->token_buffer release];
    [plan->output_buffer release];
    free(plan);
    *slot = NULL;
  }
}

static struct custom_operations prebaked_operations = {
    "llmopt.prebaked.plan",
    finalize_prebaked,
    custom_compare_default,
    custom_hash_default,
    custom_serialize_default,
    custom_deserialize_default,
    custom_compare_ext_default,
    custom_fixed_length_default};

static value alloc_prebaked(llmopt_prebaked_plan *plan) {
  value result = caml_alloc_custom(&prebaked_operations, sizeof(plan), 0, 1);
  *((llmopt_prebaked_plan **)Data_custom_val(result)) = plan;
  return result;
}

static void fail_with_error(const char *prefix, NSError *error) {
  NSString *detail = error == nil ? @"unknown Metal error" : error.localizedDescription;
  NSString *message = [NSString stringWithFormat:@"%s: %@", prefix, detail];
  caml_failwith(message.UTF8String);
}

CAMLprim value caml_llmopt_metal_create_context(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(result);
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      caml_failwith("Metal has no default device");
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      caml_failwith("Metal could not create a command queue");
    }
    llmopt_metal_context *context = calloc(1, sizeof(*context));
    if (context == NULL) {
      [queue release];
      caml_raise_out_of_memory();
    }
    context->device = [device retain];
    context->queue = queue;
    result = alloc_context(context);
  }
  CAMLreturn(result);
}

CAMLprim value caml_llmopt_metal_device_name(value context_value) {
  CAMLparam1(context_value);
  CAMLlocal1(result);
  llmopt_metal_context *context = Context_val(context_value);
  if (context == NULL) {
    caml_failwith("Metal context has been finalized");
  }
  @autoreleasepool {
    result = caml_copy_string(context->device.name.UTF8String);
  }
  CAMLreturn(result);
}

static double llmopt_command_duration(id<MTLCommandBuffer> command,
                                      uint64_t started,
                                      uint64_t finished) {
  if (@available(macOS 10.15, *)) {
    if (command.GPUEndTime > command.GPUStartTime) {
      return command.GPUEndTime - command.GPUStartTime;
    }
  }
  mach_timebase_info_data_t info;
  mach_timebase_info(&info);
  return ((double)(finished - started) * (double)info.numer /
          (double)info.denom) * 1e-9;
}

static double llmopt_median(double *values, int count) {
  for (int i = 1; i < count; ++i) {
    double value = values[i];
    int j = i - 1;
    while (j >= 0 && values[j] > value) {
      values[j + 1] = values[j];
      --j;
    }
    values[j + 1] = value;
  }
  if ((count & 1) != 0) {
    return values[count / 2];
  }
  return (values[count / 2 - 1] + values[count / 2]) * 0.5;
}

CAMLprim value caml_llmopt_metal_measure_hardware(value warmup_value,
                                                   value repeat_value,
                                                   value bytes_value) {
  CAMLparam3(warmup_value, repeat_value, bytes_value);
  CAMLlocal1(result);
  int warmup = Int_val(warmup_value);
  int repeats = Int_val(repeat_value);
  int byte_count = Int_val(bytes_value);
  if (warmup < 0 || repeats <= 0 || byte_count < 4096 ||
      (byte_count % 16) != 0) {
    caml_invalid_argument(
        "Metal hardware probe requires warmup >= 0, repeats > 0, and "
        "16-byte-aligned bytes >= 4096");
  }

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (device == nil || queue == nil) {
      caml_failwith("Metal hardware probe could not create a device queue");
    }
    NSString *source =
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "kernel void llmopt_probe_empty(uint gid [[thread_position_in_grid]]) { "
         "(void)gid; }\n"
         "kernel void llmopt_probe_copy(device const uint4 *src [[buffer(0)]], "
         "device uint4 *dst [[buffer(1)]], uint gid "
         "[[thread_position_in_grid]], uint count [[threads_per_grid]]) { "
         "if (gid < count) dst[gid] = src[gid]; }\n";
    NSError *error = nil;
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    options.fastMathEnabled = NO;
    id<MTLLibrary> library =
        [device newLibraryWithSource:source options:options error:&error];
    [options release];
    if (library == nil) {
      fail_with_error("Metal hardware probe could not compile kernels", error);
    }
    id<MTLFunction> empty_function = [library newFunctionWithName:@"llmopt_probe_empty"];
    id<MTLFunction> copy_function = [library newFunctionWithName:@"llmopt_probe_copy"];
    id<MTLComputePipelineState> empty_pipeline =
        [device newComputePipelineStateWithFunction:empty_function error:&error];
    if (empty_pipeline == nil) {
      fail_with_error("Metal hardware probe could not create empty pipeline", error);
    }
    id<MTLComputePipelineState> copy_pipeline =
        [device newComputePipelineStateWithFunction:copy_function error:&error];
    if (copy_pipeline == nil) {
      fail_with_error("Metal hardware probe could not create copy pipeline", error);
    }
    id<MTLBuffer> input = [device newBufferWithLength:(NSUInteger)byte_count
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:(NSUInteger)byte_count
                                               options:MTLResourceStorageModeShared];
    if (input == nil || output == nil) {
      caml_raise_out_of_memory();
    }

    double *dispatch_samples = calloc((size_t)repeats, sizeof(double));
    double *copy_samples = calloc((size_t)repeats, sizeof(double));
    if (dispatch_samples == NULL || copy_samples == NULL) {
      free(dispatch_samples);
      free(copy_samples);
      caml_raise_out_of_memory();
    }
    NSUInteger vectors = (NSUInteger)byte_count / 16;
    NSUInteger copy_width = MIN(copy_pipeline.maxTotalThreadsPerThreadgroup, 256);
    int total = warmup + repeats;
    for (int iteration = 0; iteration < total; ++iteration) {
      id<MTLCommandBuffer> empty_command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> empty_encoder =
          [empty_command computeCommandEncoder];
      [empty_encoder setComputePipelineState:empty_pipeline];
      [empty_encoder dispatchThreads:MTLSizeMake(1, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [empty_encoder endEncoding];
      uint64_t started = mach_continuous_time();
      [empty_command commit];
      [empty_command waitUntilCompleted];
      uint64_t finished = mach_continuous_time();
      if (empty_command.status != MTLCommandBufferStatusCompleted) {
        fail_with_error("Metal empty dispatch probe failed", empty_command.error);
      }

      id<MTLCommandBuffer> copy_command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> copy_encoder =
          [copy_command computeCommandEncoder];
      [copy_encoder setComputePipelineState:copy_pipeline];
      [copy_encoder setBuffer:input offset:0 atIndex:0];
      [copy_encoder setBuffer:output offset:0 atIndex:1];
      [copy_encoder dispatchThreads:MTLSizeMake(vectors, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(copy_width, 1, 1)];
      [copy_encoder endEncoding];
      uint64_t copy_started = mach_continuous_time();
      [copy_command commit];
      [copy_command waitUntilCompleted];
      uint64_t copy_finished = mach_continuous_time();
      if (copy_command.status != MTLCommandBufferStatusCompleted) {
        fail_with_error("Metal copy bandwidth probe failed", copy_command.error);
      }
      if (iteration >= warmup) {
        int sample = iteration - warmup;
        dispatch_samples[sample] =
            llmopt_command_duration(empty_command, started, finished);
        copy_samples[sample] =
            llmopt_command_duration(copy_command, copy_started, copy_finished);
      }
    }

    double dispatch_seconds = llmopt_median(dispatch_samples, repeats);
    double copy_seconds = llmopt_median(copy_samples, repeats);
    double bandwidth_gbps =
        (2.0 * (double)byte_count) / copy_seconds / 1e9;
    free(dispatch_samples);
    free(copy_samples);

    result = caml_alloc_tuple(4);
    Store_field(result, 0, caml_copy_string(device.name.UTF8String));
    Store_field(result, 1,
                Val_int((int)copy_pipeline.maxTotalThreadsPerThreadgroup));
    Store_field(result, 2, caml_copy_double(dispatch_seconds));
    Store_field(result, 3, caml_copy_double(bandwidth_gbps));

    [input release];
    [output release];
    [empty_pipeline release];
    [copy_pipeline release];
    [empty_function release];
    [copy_function release];
    [library release];
    [queue release];
  }
  CAMLreturn(result);
}

CAMLprim value caml_llmopt_metal_load_library(value context_value,
                                               value path_value) {
  CAMLparam2(context_value, path_value);
  CAMLlocal1(result);
  llmopt_metal_context *context = Context_val(context_value);
  if (context == NULL) {
    caml_failwith("Metal context has been finalized");
  }
  @autoreleasepool {
    NSString *path = [NSString stringWithUTF8String:String_val(path_value)];
    NSError *error = nil;
    id<MTLLibrary> metal_library = nil;
    if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
      NSDictionary *attrs = [[NSFileManager defaultManager] attributesOfItemAtPath:path error:nil];
      if (attrs && [attrs fileSize] > 0) {
        metal_library = [context->device newLibraryWithFile:path error:&error];
      }
    }
    if (metal_library == nil) {
      NSString *metalPath = [[path stringByDeletingPathExtension] stringByAppendingPathExtension:@"metal"];
      if ([[NSFileManager defaultManager] fileExistsAtPath:metalPath]) {
        NSString *source = [NSString stringWithContentsOfFile:metalPath encoding:NSUTF8StringEncoding error:&error];
        if (source != nil) {
          MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
          options.fastMathEnabled = YES;
          metal_library = [context->device newLibraryWithSource:source options:options error:&error];
          [options release];
        }
      }
    }
    if (metal_library == nil) {
      fail_with_error("cannot load Metal library", error);
    }
    llmopt_metal_library *library = calloc(1, sizeof(*library));
    if (library == NULL) {
      [metal_library release];
      caml_raise_out_of_memory();
    }
    library->device = [context->device retain];
    library->library = metal_library;
    library->queue = [context->queue retain];
    library->pipelines = [[NSMutableDictionary alloc] init];
    if (library->pipelines == nil) {
      [library->queue release];
      [library->device release];
      [metal_library release];
      free(library);
      caml_raise_out_of_memory();
    }
    result = alloc_library(library);
  }
  CAMLreturn(result);
}

CAMLprim value caml_llmopt_metal_has_function(value library_value,
                                              value name_value) {
  CAMLparam2(library_value, name_value);
  llmopt_metal_library *library = Library_val(library_value);
  if (library == NULL) {
    caml_failwith("Metal library has been finalized");
  }
  BOOL found = NO;
  @autoreleasepool {
    NSString *name = [NSString stringWithUTF8String:String_val(name_value)];
    id<MTLFunction> function = [library->library newFunctionWithName:name];
    found = function != nil;
    [function release];
  }
  CAMLreturn(Val_bool(found));
}

CAMLprim value caml_llmopt_metal_buffer_of_bytes(value context_value,
                                                 value bytes_value) {
  CAMLparam2(context_value, bytes_value);
  CAMLlocal1(result);
  llmopt_metal_context *context = Context_val(context_value);
  mlsize_t length = caml_string_length(bytes_value);
  if (context == NULL) {
    caml_failwith("Metal context has been finalized");
  }
  if (length == 0) {
    caml_invalid_argument("Metal buffer cannot be empty");
  }
  @autoreleasepool {
    id<MTLBuffer> metal_buffer =
        [context->device newBufferWithBytes:Bytes_val(bytes_value)
                                    length:length
                                   options:MTLResourceStorageModeShared];
    if (metal_buffer == nil) {
      caml_failwith("Metal could not allocate and initialize a buffer");
    }
    llmopt_metal_buffer *buffer = calloc(1, sizeof(*buffer));
    if (buffer == NULL) {
      [metal_buffer release];
      caml_raise_out_of_memory();
    }
    buffer->buffer = metal_buffer;
    buffer->offset = 0;
    buffer->length = length;
    result = alloc_buffer(buffer);
  }
  CAMLreturn(result);
}

CAMLprim value caml_llmopt_metal_create_buffer(value context_value,
                                               value length_value) {
  CAMLparam2(context_value, length_value);
  CAMLlocal1(result);
  llmopt_metal_context *context = Context_val(context_value);
  intnat requested = Long_val(length_value);
  if (context == NULL) {
    caml_failwith("Metal context has been finalized");
  }
  if (requested <= 0) {
    requested = 16;
  }
  @autoreleasepool {
    id<MTLBuffer> metal_buffer =
        [context->device newBufferWithLength:(NSUInteger)requested
                                     options:MTLResourceStorageModeShared];
    if (metal_buffer == nil) {
      caml_failwith("Metal could not allocate a buffer");
    }
    memset(metal_buffer.contents, 0, (size_t)requested);
    llmopt_metal_buffer *buffer = calloc(1, sizeof(*buffer));
    if (buffer == NULL) {
      [metal_buffer release];
      caml_raise_out_of_memory();
    }
    buffer->buffer = metal_buffer;
    buffer->offset = 0;
    buffer->length = (NSUInteger)requested;
    result = alloc_buffer(buffer);
  }
  CAMLreturn(result);
}

CAMLprim value caml_llmopt_metal_buffer_contents(value buffer_value) {
  CAMLparam1(buffer_value);
  CAMLlocal1(result);
  llmopt_metal_buffer *buffer = Buffer_val(buffer_value);
  if (buffer == NULL) {
    caml_failwith("Metal buffer has been finalized");
  }
  NSUInteger length = buffer->length;
  result = caml_alloc_string(length);
  memcpy(Bytes_val(result),
         (const uint8_t *)buffer->buffer.contents + buffer->offset, length);
  CAMLreturn(result);
}

CAMLprim value caml_llmopt_metal_buffer_argmax_f16(value buffer_value,
                                                    value vocabulary_value) {
  CAMLparam2(buffer_value, vocabulary_value);
  llmopt_metal_buffer *buffer = Buffer_val(buffer_value);
  intnat requested = Long_val(vocabulary_value);
  if (buffer == NULL) {
    caml_failwith("Metal buffer has been finalized");
  }
  if (requested <= 0) {
    caml_invalid_argument("float16 logits vocabulary must be positive");
  }
  NSUInteger vocabulary = (NSUInteger)requested;
  if (vocabulary > NSUIntegerMax / sizeof(_Float16)) {
    caml_invalid_argument("float16 logits vocabulary byte length overflows");
  }
  NSUInteger row_bytes = vocabulary * sizeof(_Float16);
  if (buffer->length == 0 || buffer->length % row_bytes != 0) {
    caml_invalid_argument("float16 logits buffer does not contain complete rows");
  }
  const uint8_t *base =
      (const uint8_t *)buffer->buffer.contents + buffer->offset;
  const _Float16 *ptr =
      (const _Float16 *)(base + buffer->length - row_bytes);
  _Float16 max_val = ptr[0];
  NSUInteger max_idx = 0;
  for (NSUInteger i = 1; i < vocabulary; i++) {
    if (ptr[i] > max_val) {
      max_val = ptr[i];
      max_idx = i;
    }
  }
  CAMLreturn(Val_long((intnat)max_idx));
}

CAMLprim value caml_llmopt_metal_buffer_length(value buffer_value) {
  CAMLparam1(buffer_value);
  llmopt_metal_buffer *buffer = Buffer_val(buffer_value);
  if (buffer == NULL) {
    caml_failwith("Metal buffer has been finalized");
  }
  CAMLreturn(Val_long(buffer->length));
}

CAMLprim value caml_llmopt_metal_buffer_copy(value source_value,
                                              value destination_value) {
  CAMLparam2(source_value, destination_value);
  llmopt_metal_buffer *source = Buffer_val(source_value);
  llmopt_metal_buffer *destination = Buffer_val(destination_value);
  if (source == NULL || destination == NULL) {
    caml_failwith("Metal buffer has been finalized");
  }
  if (source->length != destination->length) {
    caml_invalid_argument("Metal copy requires equal buffer lengths");
  }
  memmove((uint8_t *)destination->buffer.contents + destination->offset,
          (const uint8_t *)source->buffer.contents + source->offset,
          source->length);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_llmopt_metal_buffer_set_int64(value buffer_value,
                                                  value offset_value,
                                                  value int64_value) {
  CAMLparam3(buffer_value, offset_value, int64_value);
  llmopt_metal_buffer *buffer = Buffer_val(buffer_value);
  intnat offset = Long_val(offset_value);
  if (buffer == NULL) {
    caml_failwith("Metal buffer has been finalized");
  }
  if (offset < 0 || (NSUInteger)(offset + 8) > buffer->length) {
    caml_invalid_argument("buffer offset out of bounds");
  }
  int64_t *ptr =
      (int64_t *)((uint8_t *)buffer->buffer.contents + buffer->offset + offset);
  *ptr = (int64_t)Int64_val(int64_value);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_llmopt_metal_buffer_set_u32_array(value buffer_value,
                                                      value offset_value,
                                                      value array_value) {
  CAMLparam3(buffer_value, offset_value, array_value);
  llmopt_metal_buffer *buffer = Buffer_val(buffer_value);
  intnat offset = Long_val(offset_value);
  if (buffer == NULL) {
    caml_failwith("Metal buffer has been finalized");
  }
  mlsize_t count = Wosize_val(array_value);
  if (offset < 0 || (NSUInteger)(offset + count * 4) > buffer->length) {
    caml_invalid_argument("buffer set_u32_array out of bounds");
  }
  uint32_t *ptr =
      (uint32_t *)((uint8_t *)buffer->buffer.contents + buffer->offset + offset);
  for (mlsize_t i = 0; i < count; i++) {
    ptr[i] = (uint32_t)Long_val(Field(array_value, i));
  }
  CAMLreturn(Val_unit);
}

CAMLprim value caml_llmopt_metal_map_file(value context_value,
                                          value path_value) {
  CAMLparam2(context_value, path_value);
  CAMLlocal1(result);
  llmopt_metal_context *context = Context_val(context_value);
  if (context == NULL) {
    caml_failwith("Metal context has been finalized");
  }
  const char *path = String_val(path_value);
  int descriptor = open(path, O_RDONLY);
  if (descriptor < 0) {
    char message[512];
    snprintf(message, sizeof(message), "cannot open tensor store %s: %s", path,
             strerror(errno));
    caml_failwith(message);
  }
  struct stat stats;
  if (fstat(descriptor, &stats) != 0) {
    int saved_errno = errno;
    close(descriptor);
    char message[512];
    snprintf(message, sizeof(message), "cannot stat tensor store %s: %s", path,
             strerror(saved_errno));
    caml_failwith(message);
  }
  if (stats.st_size <= 0) {
    close(descriptor);
    caml_failwith("tensor store is empty");
  }
  size_t file_length = (size_t)stats.st_size;
  if ((off_t)file_length != stats.st_size) {
    close(descriptor);
    caml_failwith("tensor store is too large for this process");
  }
  long page_size_value = sysconf(_SC_PAGESIZE);
  if (page_size_value <= 0) {
    close(descriptor);
    caml_failwith("cannot determine the host page size");
  }
  size_t page_size = (size_t)page_size_value;
  if (file_length > SIZE_MAX - (page_size - 1)) {
    close(descriptor);
    caml_failwith("tensor store mapping length overflows");
  }
  size_t mapping_length =
      ((file_length + page_size - 1) / page_size) * page_size;
  void *mapping = mmap(NULL, mapping_length, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE, descriptor, 0);
  int saved_errno = errno;
  close(descriptor);
  if (mapping == MAP_FAILED) {
    char message[512];
    snprintf(message, sizeof(message), "cannot map tensor store %s: %s", path,
             strerror(saved_errno));
    caml_failwith(message);
  }
  @autoreleasepool {
    id<MTLBuffer> metal_buffer =
        [context->device newBufferWithBytesNoCopy:mapping
                                           length:mapping_length
                                          options:MTLResourceStorageModeShared
                                      deallocator:^(void *pointer,
                                                    NSUInteger length) {
                                        munmap(pointer, length);
                                      }];
    if (metal_buffer == nil) {
      munmap(mapping, mapping_length);
      caml_failwith("Metal could not wrap the mapped tensor store");
    }
    llmopt_metal_buffer *buffer = calloc(1, sizeof(*buffer));
    if (buffer == NULL) {
      [metal_buffer release];
      caml_raise_out_of_memory();
    }
    buffer->buffer = metal_buffer;
    buffer->offset = 0;
    buffer->length = file_length;
    result = alloc_buffer(buffer);
  }
  CAMLreturn(result);
}

CAMLprim value caml_llmopt_metal_buffer_view(value buffer_value,
                                             value offset_value,
                                             value length_value) {
  CAMLparam3(buffer_value, offset_value, length_value);
  CAMLlocal1(result);
  llmopt_metal_buffer *parent = Buffer_val(buffer_value);
  intnat requested_offset = Long_val(offset_value);
  intnat requested_length = Long_val(length_value);
  if (parent == NULL) {
    caml_failwith("Metal buffer has been finalized");
  }
  if (requested_offset < 0 || requested_length <= 0 ||
      (uintnat)requested_offset > parent->length ||
      (uintnat)requested_length > parent->length - (uintnat)requested_offset) {
    caml_invalid_argument("Metal buffer view is outside its parent buffer");
  }
  llmopt_metal_buffer *view = calloc(1, sizeof(*view));
  if (view == NULL) {
    caml_raise_out_of_memory();
  }
  view->buffer = [parent->buffer retain];
  view->offset = parent->offset + (NSUInteger)requested_offset;
  view->length = (NSUInteger)requested_length;
  result = alloc_buffer(view);
  CAMLreturn(result);
}

static void require_buffer_size(llmopt_metal_buffer *buffer, uint64_t required,
                                const char *name) {
  if (buffer == NULL) {
    caml_failwith("Metal buffer has been finalized");
  }
  if ((uint64_t)buffer->length < required) {
    char message[192];
    snprintf(message, sizeof(message),
             "Metal %s buffer is too small: required=%llu actual=%llu", name,
             (unsigned long long)required,
             (unsigned long long)buffer->length);
    caml_invalid_argument(message);
  }
}

static id<MTLComputePipelineState>
pipeline_for_name(llmopt_metal_library *library, const char *kernel_name) {
  for (uint32_t i = 0; i < library->cache_count; i++) {
    if (library->cache[i].name == kernel_name ||
        strcmp(library->cache[i].name, kernel_name) == 0) {
      return library->cache[i].pipeline;
    }
  }

  NSString *name = [NSString stringWithUTF8String:kernel_name];
  id<MTLComputePipelineState> pipeline =
      [library->pipelines objectForKey:name];
  if (pipeline == nil) {
    id<MTLFunction> function = [library->library newFunctionWithName:name];
    if (function == nil) {
      caml_failwith("Metal library does not contain the selected kernel");
    }
    NSError *pipeline_error = nil;
    pipeline = [library->device newComputePipelineStateWithFunction:function
                                                               error:&pipeline_error];
    [function release];
    if (pipeline == nil) {
      fail_with_error("cannot create Metal compute pipeline", pipeline_error);
    }
    [library->pipelines setObject:pipeline forKey:name];
  }

  if (library->cache_count < LLMOPT_PIPELINE_CACHE_SIZE) {
    library->cache[library->cache_count].name = strdup(kernel_name);
    library->cache[library->cache_count].pipeline = pipeline;
    library->cache_count++;
  }
  return pipeline;
}

static id<MTLComputePipelineState>
pipeline_for_name_opt(llmopt_metal_library *library, const char *kernel_name) {
  if (library == NULL || library->library == nil) return nil;
  for (uint32_t i = 0; i < library->cache_count; i++) {
    if (library->cache[i].name == kernel_name ||
        strcmp(library->cache[i].name, kernel_name) == 0) {
      return library->cache[i].pipeline;
    }
  }

  NSString *name = [NSString stringWithUTF8String:kernel_name];
  id<MTLComputePipelineState> pipeline =
      [library->pipelines objectForKey:name];
  if (pipeline == nil) {
    id<MTLFunction> function = [library->library newFunctionWithName:name];
    if (function == nil) {
      return nil;
    }
    NSError *pipeline_error = nil;
    pipeline = [library->device newComputePipelineStateWithFunction:function
                                                               error:&pipeline_error];
    [function release];
    if (pipeline == nil) {
      return nil;
    }
    [library->pipelines setObject:pipeline forKey:name];
  }

  if (library->cache_count < LLMOPT_PIPELINE_CACHE_SIZE) {
    library->cache[library->cache_count].name = strdup(kernel_name);
    library->cache[library->cache_count].pipeline = pipeline;
    library->cache_count++;
  }
  return pipeline;
}

CAMLprim value caml_llmopt_metal_preload_pipeline(value library_value,
                                                   value name_value) {
  CAMLparam2(library_value, name_value);
  llmopt_metal_library *library = Library_val(library_value);
  if (library == NULL) {
    caml_failwith("Metal library has been finalized");
  }
  @autoreleasepool {
    (void)pipeline_for_name(library, String_val(name_value));
  }
  CAMLreturn(Val_unit);
}

static NSUInteger positive_size(value encoded, const char *name) {
  intnat decoded = Long_val(encoded);
  if (decoded <= 0) {
    char message[160];
    snprintf(message, sizeof(message), "Metal %s must be positive", name);
    caml_invalid_argument(message);
  }
  return (NSUInteger)decoded;
}

static llmopt_metal_batch *require_active_batch(value handle) {
  llmopt_metal_batch *batch = Batch_val(handle);
  if (batch == NULL) {
    caml_failwith("Metal batch has been finalized");
  }
  if (batch->finished || batch->command == nil) {
    caml_invalid_argument("Metal batch is already finished");
  }
  return batch;
}

static id<MTLComputeCommandEncoder>
batch_compute_encoder(llmopt_metal_batch *batch) {
  if (batch->compute == nil) {
    batch->compute = [[batch->command computeCommandEncoder] retain];
    if (batch->compute == nil) {
      caml_failwith("Metal could not create a batched compute encoder");
    }
  }
  return batch->compute;
}

CAMLprim value caml_llmopt_metal_begin_batch(value library_value) {
  CAMLparam1(library_value);
  CAMLlocal1(result);
  llmopt_metal_library *library = Library_val(library_value);
  if (library == NULL) {
    caml_failwith("Metal library has been finalized");
  }
  @autoreleasepool {
    id<MTLCommandBuffer> command = [library->queue commandBuffer];
    if (command == nil) {
      caml_failwith("Metal could not create a batched command buffer");
    }
    llmopt_metal_batch *batch = calloc(1, sizeof(*batch));
    if (batch == NULL) {
      caml_raise_out_of_memory();
    }
    batch->command = [command retain];
    batch->compute = nil;
    batch->finished = NO;
    result = alloc_batch(batch);
  }
  CAMLreturn(result);
}

CAMLprim value caml_llmopt_metal_batch_dispatch(value arguments) {
  CAMLparam1(arguments);
  llmopt_metal_batch *batch =
      require_active_batch(Field(arguments, 0));
  llmopt_metal_library *library = Library_val(Field(arguments, 1));
  const char *kernel_name = String_val(Field(arguments, 2));
  value buffers = Field(arguments, 3);
  value parameters = Field(arguments, 4);
  MTLSize grid = MTLSizeMake(
      positive_size(Field(arguments, 5), "grid width"),
      positive_size(Field(arguments, 6), "grid height"),
      positive_size(Field(arguments, 7), "grid depth"));
  MTLSize group = MTLSizeMake(
      positive_size(Field(arguments, 8), "threadgroup width"),
      positive_size(Field(arguments, 9), "threadgroup height"),
      positive_size(Field(arguments, 10), "threadgroup depth"));
  if (library == NULL) {
    caml_failwith("Metal library has been finalized");
  }

  id<MTLComputePipelineState> pipeline =
      pipeline_for_name(library, kernel_name);
  if (group.width * group.height * group.depth >
      pipeline.maxTotalThreadsPerThreadgroup) {
    caml_invalid_argument("Metal threadgroup exceeds pipeline capacity");
  }
  id<MTLComputeCommandEncoder> encoder = batch_compute_encoder(batch);
  [encoder setComputePipelineState:pipeline];

  NSUInteger buffer_index = 0;
  value remaining = buffers;
  while (remaining != Val_emptylist) {
    llmopt_metal_buffer *buffer = Buffer_val(Field(remaining, 0));
    if (buffer == NULL) {
      caml_failwith("Metal buffer has been finalized");
    }
    [encoder setBuffer:buffer->buffer
                offset:buffer->offset
               atIndex:buffer_index];
    buffer_index += 1;
    remaining = Field(remaining, 1);
  }

  mlsize_t parameter_length = caml_string_length(parameters);
  if (parameter_length > 0) {
    [encoder setBytes:Bytes_val(parameters)
               length:(NSUInteger)parameter_length
              atIndex:buffer_index];
  }
  [encoder dispatchThreads:grid threadsPerThreadgroup:group];
  CAMLreturn(Val_unit);
}

CAMLprim value caml_llmopt_metal_batch_dispatch_all(value batch_value,
                                                    value library_value,
                                                    value dispatches_value) {
  CAMLparam3(batch_value, library_value, dispatches_value);
  llmopt_metal_batch *batch = require_active_batch(batch_value);
  llmopt_metal_library *library = Library_val(library_value);
  if (library == NULL) {
    caml_failwith("Metal library has been finalized");
  }
  id<MTLComputeCommandEncoder> encoder = batch_compute_encoder(batch);
  mlsize_t total = Wosize_val(dispatches_value);

  id<MTLComputePipelineState> current_pipeline = nil;
  for (mlsize_t i = 0; i < total; i++) {
    value item = Field(dispatches_value, i);
    const char *kernel_name = String_val(Field(item, 0));
    value buffers = Field(item, 1);
    value parameters = Field(item, 2);
    MTLSize grid = MTLSizeMake(
        Long_val(Field(item, 3)),
        Long_val(Field(item, 4)),
        Long_val(Field(item, 5)));
    MTLSize group = MTLSizeMake(
        Long_val(Field(item, 6)),
        Long_val(Field(item, 7)),
        Long_val(Field(item, 8)));

    id<MTLComputePipelineState> pipeline =
        pipeline_for_name(library, kernel_name);
    if (pipeline != current_pipeline) {
      [encoder setComputePipelineState:pipeline];
      current_pipeline = pipeline;
    }

    id<MTLBuffer> bufs[16];
    NSUInteger offs[16];
    NSUInteger buffer_index = 0;
    value remaining = buffers;
    while (remaining != Val_emptylist && buffer_index < 16) {
      llmopt_metal_buffer *buffer = Buffer_val(Field(remaining, 0));
      if (buffer != NULL) {
        bufs[buffer_index] = buffer->buffer;
        offs[buffer_index] = buffer->offset;
      } else {
        bufs[buffer_index] = nil;
        offs[buffer_index] = 0;
      }
      buffer_index += 1;
      remaining = Field(remaining, 1);
    }
    if (buffer_index > 0) {
      [encoder setBuffers:bufs offsets:offs withRange:NSMakeRange(0, buffer_index)];
    }

    mlsize_t parameter_length = caml_string_length(parameters);
    if (parameter_length > 0) {
      [encoder setBytes:Bytes_val(parameters)
                 length:(NSUInteger)parameter_length
                atIndex:buffer_index];
    }
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
  }

  CAMLreturn(Val_unit);
}


CAMLprim value caml_llmopt_metal_batch_copy(value batch_value,
                                             value source_value,
                                             value destination_value) {
  CAMLparam3(batch_value, source_value, destination_value);
  llmopt_metal_batch *batch = require_active_batch(batch_value);
  llmopt_metal_buffer *source = Buffer_val(source_value);
  llmopt_metal_buffer *destination = Buffer_val(destination_value);
  if (source == NULL || destination == NULL) {
    caml_failwith("Metal buffer has been finalized");
  }
  if (source->length != destination->length) {
    caml_invalid_argument("Metal copy requires equal buffer lengths");
  }
  if (source->buffer == destination->buffer) {
    NSUInteger source_end = source->offset + source->length;
    NSUInteger destination_end = destination->offset + destination->length;
    if (source->offset == destination->offset) {
      CAMLreturn(Val_unit);
    }
    if (source->offset < destination_end && destination->offset < source_end) {
      caml_invalid_argument("batched Metal copy buffers overlap");
    }
  }
  @autoreleasepool {
    end_batch_compute(batch);
    id<MTLBlitCommandEncoder> blit =
        [[batch->command blitCommandEncoder] retain];
    if (blit == nil) {
      caml_failwith("Metal could not create a batched blit encoder");
    }
    [blit copyFromBuffer:source->buffer
            sourceOffset:source->offset
                toBuffer:destination->buffer
       destinationOffset:destination->offset
                    size:source->length];
    [blit endEncoding];
    [blit release];
  }
  CAMLreturn(Val_unit);
}

CAMLprim value caml_llmopt_metal_batch_barrier(value batch_value) {
  CAMLparam1(batch_value);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_llmopt_metal_commit_batch(value batch_value) {
  CAMLparam1(batch_value);
  llmopt_metal_batch *batch = require_active_batch(batch_value);
  @autoreleasepool {
    uint64_t started = mach_absolute_time();
    end_batch_compute(batch);
    id<MTLCommandBuffer> command = [batch->command retain];
    [batch->command commit];
    batch->finished = YES;
    [batch->command release];
    batch->command = nil;
    caml_enter_blocking_section();
    [command waitUntilCompleted];
    caml_leave_blocking_section();
    uint64_t finished = mach_absolute_time();
    if (getenv("LLMOPT_PROFILE_BATCH") != NULL) {
      double gpu_seconds = llmopt_command_duration(command, started, finished);
      mach_timebase_info_data_t info;
      mach_timebase_info(&info);
      double wall_seconds =
          ((double)(finished - started) * (double)info.numer /
           (double)info.denom) * 1e-9;
      fprintf(stderr,
              "llmopt-metal-profile: gpu_ms=%.6f commit_wait_ms=%.6f\n",
              gpu_seconds * 1000.0, wall_seconds * 1000.0);
    }
    MTLCommandBufferStatus status = command.status;
    NSError *command_error = [command.error retain];
    [command release];
    if (status != MTLCommandBufferStatusCompleted) {
      fail_with_error("Metal batch failed", command_error);
    }
    [command_error release];
  }
  CAMLreturn(Val_unit);
}

CAMLprim value caml_llmopt_metal_commit_batch_async(value batch_value) {
  CAMLparam1(batch_value);
  llmopt_metal_batch *batch = require_active_batch(batch_value);
  @autoreleasepool {
    end_batch_compute(batch);
    [batch->command commit];
    batch->finished = YES;
    [batch->command release];
    batch->command = nil;
  }
  CAMLreturn(Val_unit);
}

CAMLprim value caml_llmopt_metal_abort_batch(value batch_value) {
  CAMLparam1(batch_value);
  llmopt_metal_batch *batch = Batch_val(batch_value);
  if (batch != NULL && !batch->finished) {
    end_batch_compute(batch);
    [batch->command release];
    batch->command = nil;
    batch->finished = YES;
  }
  CAMLreturn(Val_unit);
}

CAMLprim value caml_llmopt_metal_dispatch(value arguments) {
  CAMLparam1(arguments);
  llmopt_metal_library *library = Library_val(Field(arguments, 0));
  const char *kernel_name = String_val(Field(arguments, 1));
  value buffers = Field(arguments, 2);
  value parameters = Field(arguments, 3);
  MTLSize grid = MTLSizeMake(
      positive_size(Field(arguments, 4), "grid width"),
      positive_size(Field(arguments, 5), "grid height"),
      positive_size(Field(arguments, 6), "grid depth"));
  MTLSize group = MTLSizeMake(
      positive_size(Field(arguments, 7), "threadgroup width"),
      positive_size(Field(arguments, 8), "threadgroup height"),
      positive_size(Field(arguments, 9), "threadgroup depth"));

  if (library == NULL) {
    caml_failwith("Metal library has been finalized");
  }

  @autoreleasepool {
    id<MTLComputePipelineState> pipeline =
        pipeline_for_name(library, kernel_name);
    if (group.width * group.height * group.depth >
        pipeline.maxTotalThreadsPerThreadgroup) {
      caml_invalid_argument("Metal threadgroup exceeds pipeline capacity");
    }

    id<MTLCommandBuffer> command = [library->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (command == nil || encoder == nil) {
      caml_failwith("Metal could not create a command buffer or encoder");
    }
    [encoder setComputePipelineState:pipeline];

    NSUInteger buffer_index = 0;
    value remaining = buffers;
    while (remaining != Val_emptylist) {
      llmopt_metal_buffer *buffer = Buffer_val(Field(remaining, 0));
      if (buffer == NULL) {
        caml_failwith("Metal buffer has been finalized");
      }
      [encoder setBuffer:buffer->buffer
                  offset:buffer->offset
                 atIndex:buffer_index];
      buffer_index += 1;
      remaining = Field(remaining, 1);
    }

    mlsize_t parameter_length = caml_string_length(parameters);
    if (parameter_length > 0) {
      [encoder setBytes:Bytes_val(parameters)
                 length:(NSUInteger)parameter_length
                atIndex:buffer_index];
    }
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    [command retain];
    [command commit];
    caml_enter_blocking_section();
    [command waitUntilCompleted];
    caml_leave_blocking_section();
    MTLCommandBufferStatus status = command.status;
    NSError *command_error = [command.error retain];
    [command release];
    if (status != MTLCommandBufferStatusCompleted) {
      fail_with_error("Metal dispatch failed", command_error);
    }
    [command_error release];
  }
  CAMLreturn(Val_unit);
}

static inline bool llmopt_enable_serial_barriers(void) {
  static int cached = -1;
  if (cached == -1) {
    const char *env = getenv("LLMOPT_ENABLE_SERIAL_BARRIERS");
    cached = (env != NULL && strcmp(env, "1") == 0) ? 1 : 0;
  }
  return cached == 1;
}

static void *llmopt_worker_loop(void *arg) {
  llmopt_ring_queue *q = (llmopt_ring_queue *)arg;

  int pipeline_depth = 1;
  const char *depth_env = getenv("LLMOPT_QUEUE_DEPTH");
  if (depth_env != NULL) {
    int d = atoi(depth_env);
    if (d >= 1) pipeline_depth = d;
  }
  if (getenv("LLMOPT_PROFILE_BATCH") != NULL) {
    pipeline_depth = 1;
  }

  id<MTLCommandBuffer> prev_cmd = nil;
  llmopt_sq_entry prev_entry;
  uint32_t prev_slot = 0;
  bool has_prev = false;

  while (atomic_load_explicit(&q->running, memory_order_acquire)) {
    uint32_t head = atomic_load_explicit(&q->sq_head, memory_order_relaxed);
    uint32_t tail = atomic_load_explicit(&q->sq_tail, memory_order_acquire);

    if (head == tail) {
      bool found = false;
      for (int i = 0; i < 1000; i++) {
        tail = atomic_load_explicit(&q->sq_tail, memory_order_acquire);
        if (head != tail) {
          found = true;
          break;
        }
        sched_yield();
      }
      if (!found) {
        if (has_prev) {
          [prev_cmd waitUntilCompleted];
          int32_t next_token = 0;
          if (q->output_buffers[prev_slot] != nil) {
            int32_t *out_ptr = (int32_t *)((uint8_t *)[q->output_buffers[prev_slot] contents] + q->output_buffer_offsets[prev_slot]);
            next_token = *out_ptr;
            if (prev_slot != 0 && q->output_buffers[0] != nil) {
              int32_t *out0 = (int32_t *)((uint8_t *)[q->output_buffers[0] contents] + q->output_buffer_offsets[0]);
              *out0 = next_token;
            }
          }
          uint32_t cq_tail = atomic_load_explicit(&q->cq_tail, memory_order_relaxed);
          q->cq[cq_tail % LLMOPT_RING_CAPACITY].request_id = prev_entry.request_id;
          q->cq[cq_tail % LLMOPT_RING_CAPACITY].token = next_token;
          q->cq[cq_tail % LLMOPT_RING_CAPACITY].status = (prev_cmd.status == MTLCommandBufferStatusCompleted) ? 0 : 1;
          atomic_store_explicit(&q->cq_tail, cq_tail + 1, memory_order_release);

          pthread_mutex_lock(&q->mutex);
          pthread_cond_signal(&q->cq_cond);
          pthread_mutex_unlock(&q->mutex);

          [prev_cmd release];
          prev_cmd = nil;
          has_prev = false;
        }

        pthread_mutex_lock(&q->mutex);
        while (atomic_load_explicit(&q->sq_head, memory_order_relaxed) ==
               atomic_load_explicit(&q->sq_tail, memory_order_relaxed) &&
               atomic_load_explicit(&q->running, memory_order_relaxed)) {
          pthread_cond_wait(&q->sq_cond, &q->mutex);
        }
        pthread_mutex_unlock(&q->mutex);
        continue;
      }
    }

    llmopt_sq_entry entry = q->sq[head % LLMOPT_RING_CAPACITY];
    atomic_store_explicit(&q->sq_head, head + 1, memory_order_release);

    if (entry.flags == 2 || !atomic_load_explicit(&q->running, memory_order_relaxed)) {
      break;
    }

    uint32_t slot = head % 2;
    bool gpu_chain = false;
    if (entry.token >= 0) {
      if (q->token_buffers[slot] != nil) {
        int64_t *ptr = (int64_t *)((uint8_t *)[q->token_buffers[slot] contents] + q->token_buffer_offsets[slot]);
        *ptr = (int64_t)entry.token;
      }
    } else {
      uint32_t last_slot = (slot == 0) ? 1 : 0;
      if (q->chain_pipeline != nil && q->output_buffers[last_slot] != nil && q->token_buffers[slot] != nil) {
        gpu_chain = true;
      } else if (q->output_buffers[last_slot] != nil && q->token_buffers[slot] != nil) {
        if (has_prev) {
          [prev_cmd waitUntilCompleted];
        }
        int32_t *out_ptr = (int32_t *)((uint8_t *)[q->output_buffers[last_slot] contents] + q->output_buffer_offsets[last_slot]);
        int64_t *ptr = (int64_t *)((uint8_t *)[q->token_buffers[slot] contents] + q->token_buffer_offsets[slot]);
        *ptr = (int64_t)(*out_ptr);
      }
    }

    id<MTLCommandBuffer> cmd = nil;
    @autoreleasepool {
      cmd = [[q->queue commandBuffer] retain];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

      if (gpu_chain) {
        uint32_t last_slot = (slot == 0) ? 1 : 0;
        [enc setComputePipelineState:q->chain_pipeline];
        [enc setBuffer:q->output_buffers[last_slot] offset:q->output_buffer_offsets[last_slot] atIndex:0];
        [enc setBuffer:q->token_buffers[slot] offset:q->token_buffer_offsets[slot] atIndex:1];
        [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        if (llmopt_enable_serial_barriers()) {
          [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
      }

      id<MTLComputePipelineState> current_pipeline = nil;
      for (size_t i = 0; i < q->record_count; i++) {
        llmopt_dispatch_record *r = &q->records[i];
        if (r->barrier_before && llmopt_enable_serial_barriers()) {
          [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        if (r->pipeline != current_pipeline) {
          [enc setComputePipelineState:r->pipeline];
          current_pipeline = r->pipeline;
        }
        if (r->buffer_count > 0) {
          [enc setBuffers:r->buffers offsets:r->offsets withRange:NSMakeRange(0, r->buffer_count)];
          if (r->token_buffer_index >= 0) {
            NSUInteger off = (slot == 0) ? r->offsets[r->token_buffer_index] : (r->offsets[r->token_buffer_index] - q->token_buffer_offsets[0] + q->token_buffer_offsets[1]);
            [enc setBuffer:q->token_buffers[slot] offset:off atIndex:(NSUInteger)r->token_buffer_index];
          }
          if (r->output_buffer_index >= 0) {
            NSUInteger off = (slot == 0) ? r->offsets[r->output_buffer_index] : (r->offsets[r->output_buffer_index] - q->output_buffer_offsets[0] + q->output_buffer_offsets[1]);
            [enc setBuffer:q->output_buffers[slot] offset:off atIndex:(NSUInteger)r->output_buffer_index];
          }
        }
        if (r->parameter_length > 0) {
          if (r->is_paged_attention && r->parameter_length >= 16) {
            *(uint32_t *)(&r->parameters[12]) = (uint32_t)entry.past_tokens;
          }
          [enc setBytes:r->parameters length:r->parameter_length atIndex:r->buffer_count];
        }
        [enc dispatchThreads:r->grid threadsPerThreadgroup:r->group];
      }

      [enc endEncoding];
      [cmd commit];
    }

    if (pipeline_depth <= 1) {
      [cmd waitUntilCompleted];
      int32_t next_token = 0;
      if (q->output_buffers[slot] != nil) {
        int32_t *out_ptr = (int32_t *)((uint8_t *)[q->output_buffers[slot] contents] + q->output_buffer_offsets[slot]);
        next_token = *out_ptr;
        if (slot != 0 && q->output_buffers[0] != nil) {
          int32_t *out0 = (int32_t *)((uint8_t *)[q->output_buffers[0] contents] + q->output_buffer_offsets[0]);
          *out0 = next_token;
        }
      }
      uint32_t cq_tail = atomic_load_explicit(&q->cq_tail, memory_order_relaxed);
      q->cq[cq_tail % LLMOPT_RING_CAPACITY].request_id = entry.request_id;
      q->cq[cq_tail % LLMOPT_RING_CAPACITY].token = next_token;
      q->cq[cq_tail % LLMOPT_RING_CAPACITY].status = (cmd.status == MTLCommandBufferStatusCompleted) ? 0 : 1;
      atomic_store_explicit(&q->cq_tail, cq_tail + 1, memory_order_release);

      pthread_mutex_lock(&q->mutex);
      pthread_cond_signal(&q->cq_cond);
      pthread_mutex_unlock(&q->mutex);

      [cmd release];
      prev_cmd = nil;
      has_prev = false;
    } else {
      if (has_prev) {
        [prev_cmd waitUntilCompleted];
        int32_t next_token = 0;
        if (q->output_buffers[prev_slot] != nil) {
          int32_t *out_ptr = (int32_t *)((uint8_t *)[q->output_buffers[prev_slot] contents] + q->output_buffer_offsets[prev_slot]);
          next_token = *out_ptr;
          if (prev_slot != 0 && q->output_buffers[0] != nil) {
            int32_t *out0 = (int32_t *)((uint8_t *)[q->output_buffers[0] contents] + q->output_buffer_offsets[0]);
            *out0 = next_token;
          }
        }
        uint32_t cq_tail = atomic_load_explicit(&q->cq_tail, memory_order_relaxed);
        q->cq[cq_tail % LLMOPT_RING_CAPACITY].request_id = prev_entry.request_id;
        q->cq[cq_tail % LLMOPT_RING_CAPACITY].token = next_token;
        q->cq[cq_tail % LLMOPT_RING_CAPACITY].status = (prev_cmd.status == MTLCommandBufferStatusCompleted) ? 0 : 1;
        atomic_store_explicit(&q->cq_tail, cq_tail + 1, memory_order_release);

        pthread_mutex_lock(&q->mutex);
        pthread_cond_signal(&q->cq_cond);
        pthread_mutex_unlock(&q->mutex);

        [prev_cmd release];
        prev_cmd = nil;
        has_prev = false;
      }
      prev_cmd = cmd;
      prev_entry = entry;
      prev_slot = slot;
      has_prev = true;
    }
  }

  if (has_prev) {
    [prev_cmd waitUntilCompleted];
    int32_t next_token = 0;
    if (q->output_buffers[prev_slot] != nil) {
      int32_t *out_ptr = (int32_t *)((uint8_t *)[q->output_buffers[prev_slot] contents] + q->output_buffer_offsets[prev_slot]);
      next_token = *out_ptr;
      if (prev_slot != 0 && q->output_buffers[0] != nil) {
        int32_t *out0 = (int32_t *)((uint8_t *)[q->output_buffers[0] contents] + q->output_buffer_offsets[0]);
        *out0 = next_token;
      }
    }
    uint32_t cq_tail = atomic_load_explicit(&q->cq_tail, memory_order_relaxed);
    q->cq[cq_tail % LLMOPT_RING_CAPACITY].request_id = prev_entry.request_id;
    q->cq[cq_tail % LLMOPT_RING_CAPACITY].token = next_token;
    q->cq[cq_tail % LLMOPT_RING_CAPACITY].status = (prev_cmd.status == MTLCommandBufferStatusCompleted) ? 0 : 1;
    atomic_store_explicit(&q->cq_tail, cq_tail + 1, memory_order_release);

    pthread_mutex_lock(&q->mutex);
    pthread_cond_signal(&q->cq_cond);
    pthread_mutex_unlock(&q->mutex);

    [prev_cmd release];
    prev_cmd = nil;
    has_prev = false;
  }

  return NULL;
}

CAMLprim value caml_llmopt_ring_create(value unit) {
  CAMLparam1(unit);
  llmopt_ring_queue *q = calloc(1, sizeof(*q));
  if (q == NULL) {
    caml_raise_out_of_memory();
  }
  pthread_mutex_init(&q->mutex, NULL);
  pthread_cond_init(&q->sq_cond, NULL);
  pthread_cond_init(&q->cq_cond, NULL);
  atomic_store_explicit(&q->running, 1, memory_order_release);
  CAMLreturn(alloc_ring(q));
}

CAMLprim value caml_llmopt_ring_submit(value v_ring, value v_req_id, value v_token, value v_past, value v_flags) {
  CAMLparam5(v_ring, v_req_id, v_token, v_past, v_flags);
  llmopt_ring_queue *q = Ring_val(v_ring);
  uint32_t tail = atomic_load_explicit(&q->sq_tail, memory_order_relaxed);
  uint32_t head = atomic_load_explicit(&q->sq_head, memory_order_acquire);
  if (tail - head >= LLMOPT_RING_CAPACITY) {
    CAMLreturn(Val_bool(0));
  }
  q->sq[tail % LLMOPT_RING_CAPACITY].request_id = (uint32_t)Long_val(v_req_id);
  q->sq[tail % LLMOPT_RING_CAPACITY].token = (int32_t)Long_val(v_token);
  q->sq[tail % LLMOPT_RING_CAPACITY].past_tokens = (int32_t)Long_val(v_past);
  q->sq[tail % LLMOPT_RING_CAPACITY].flags = (uint32_t)Long_val(v_flags);
  atomic_store_explicit(&q->sq_tail, tail + 1, memory_order_release);

  pthread_mutex_lock(&q->mutex);
  pthread_cond_signal(&q->sq_cond);
  pthread_mutex_unlock(&q->mutex);

  CAMLreturn(Val_bool(1));
}

CAMLprim value caml_llmopt_ring_wait_completion(value v_ring) {
  CAMLparam1(v_ring);
  CAMLlocal1(v_result);
  llmopt_ring_queue *q = Ring_val(v_ring);

  uint32_t head = atomic_load_explicit(&q->cq_head, memory_order_relaxed);
  uint32_t tail = atomic_load_explicit(&q->cq_tail, memory_order_acquire);

  if (head == tail) {
    caml_enter_blocking_section();
    pthread_mutex_lock(&q->mutex);
    while (atomic_load_explicit(&q->cq_head, memory_order_relaxed) ==
           atomic_load_explicit(&q->cq_tail, memory_order_relaxed) &&
           atomic_load_explicit(&q->running, memory_order_relaxed)) {
      pthread_cond_wait(&q->cq_cond, &q->mutex);
    }
    pthread_mutex_unlock(&q->mutex);
    caml_leave_blocking_section();
  }

  head = atomic_load_explicit(&q->cq_head, memory_order_relaxed);
  tail = atomic_load_explicit(&q->cq_tail, memory_order_acquire);
  if (head == tail) {
    caml_failwith("Ring queue terminated with no completion");
  }

  llmopt_cq_entry entry = q->cq[head % LLMOPT_RING_CAPACITY];
  atomic_store_explicit(&q->cq_head, head + 1, memory_order_release);

  v_result = caml_alloc_tuple(3);
  Store_field(v_result, 0, Val_long(entry.request_id));
  Store_field(v_result, 1, Val_long(entry.token));
  Store_field(v_result, 2, Val_long(entry.status));

  CAMLreturn(v_result);
}

CAMLprim value caml_llmopt_ring_poll_completion(value v_ring) {
  CAMLparam1(v_ring);
  CAMLlocal2(v_result, v_tuple);
  llmopt_ring_queue *q = Ring_val(v_ring);

  uint32_t head = atomic_load_explicit(&q->cq_head, memory_order_relaxed);
  uint32_t tail = atomic_load_explicit(&q->cq_tail, memory_order_acquire);

  if (head == tail) {
    CAMLreturn(Val_int(0)); // None
  }

  llmopt_cq_entry entry = q->cq[head % LLMOPT_RING_CAPACITY];
  atomic_store_explicit(&q->cq_head, head + 1, memory_order_release);

  v_tuple = caml_alloc_tuple(3);
  Store_field(v_tuple, 0, Val_long(entry.request_id));
  Store_field(v_tuple, 1, Val_long(entry.token));
  Store_field(v_tuple, 2, Val_long(entry.status));

  v_result = caml_alloc(1, 0); // Some
  Store_field(v_result, 0, v_tuple);

  CAMLreturn(v_result);
}

CAMLprim value caml_llmopt_ring_start_worker(value v_ring,
                                             value v_library,
                                             value v_dispatches,
                                             value v_token_buf,
                                             value v_out_buf) {
  CAMLparam5(v_ring, v_library, v_dispatches, v_token_buf, v_out_buf);
  llmopt_ring_queue *q = Ring_val(v_ring);
  llmopt_metal_library *lib = Library_val(v_library);
  llmopt_metal_buffer *t_buf = Buffer_val(v_token_buf);
  llmopt_metal_buffer *o_buf = Buffer_val(v_out_buf);

  mlsize_t total = Wosize_val(v_dispatches);
  q->records = calloc(total, sizeof(llmopt_dispatch_record));
  q->record_count = total;
  q->library = lib;
  q->queue = [lib->queue retain];

  if (t_buf != NULL) {
    q->token_buffers[0] = [t_buf->buffer retain];
    q->token_buffer_offsets[0] = t_buf->offset;
    q->token_buffers[1] = [lib->device newBufferWithLength:t_buf->buffer.length options:MTLResourceStorageModeShared];
    q->token_buffer_offsets[1] = 0;
  }
  if (o_buf != NULL) {
    q->output_buffers[0] = [o_buf->buffer retain];
    q->output_buffer_offsets[0] = o_buf->offset;
    q->output_buffers[1] = [lib->device newBufferWithLength:o_buf->buffer.length options:MTLResourceStorageModeShared];
    q->output_buffer_offsets[1] = 0;
  }
  q->chain_pipeline = [pipeline_for_name_opt(lib, "llmopt_chain_token_id") retain];

  for (mlsize_t i = 0; i < total; i++) {
    value item = Field(v_dispatches, i);
    const char *kname = String_val(Field(item, 0));
    value buffers = Field(item, 1);
    value params = Field(item, 2);
    q->records[i].grid = MTLSizeMake(Long_val(Field(item, 3)), Long_val(Field(item, 4)), Long_val(Field(item, 5)));
    q->records[i].group = MTLSizeMake(Long_val(Field(item, 6)), Long_val(Field(item, 7)), Long_val(Field(item, 8)));
    q->records[i].barrier_before = Bool_val(Field(item, 9));
    q->records[i].pipeline = [pipeline_for_name(lib, kname) retain];
    q->records[i].is_paged_attention = is_paged_attention_kernel(kname);
    q->records[i].token_buffer_index = -1;
    q->records[i].output_buffer_index = -1;

    NSUInteger b_idx = 0;
    value rem = buffers;
    while (rem != Val_emptylist && b_idx < 16) {
      llmopt_metal_buffer *b = Buffer_val(Field(rem, 0));
      if (b != NULL) {
        q->records[i].buffers[b_idx] = [b->buffer retain];
        q->records[i].offsets[b_idx] = b->offset;
        if (q->token_buffers[0] != nil && b->buffer == q->token_buffers[0]) {
          q->records[i].token_buffer_index = (int8_t)b_idx;
        }
        if (q->output_buffers[0] != nil && b->buffer == q->output_buffers[0]) {
          q->records[i].output_buffer_index = (int8_t)b_idx;
        }
      }
      b_idx++;
      rem = Field(rem, 1);
    }
    q->records[i].buffer_count = b_idx;

    mlsize_t plen = caml_string_length(params);
    if (plen > 0 && plen <= 256) {
      memcpy(q->records[i].parameters, Bytes_val(params), plen);
      q->records[i].parameter_length = plen;
    }
  }

  atomic_store_explicit(&q->running, 1, memory_order_release);
  q->worker_started = true;
  pthread_create(&q->worker_thread, NULL, llmopt_worker_loop, q);

  CAMLreturn(Val_unit);
}

CAMLprim value caml_llmopt_ring_destroy(value v_ring) {
  CAMLparam1(v_ring);
  finalize_ring(v_ring);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_llmopt_prebaked_create(value v_library,
                                           value v_dispatches,
                                           value v_token_buf,
                                           value v_out_buf) {
  CAMLparam4(v_library, v_dispatches, v_token_buf, v_out_buf);
  llmopt_metal_library *lib = Library_val(v_library);
  llmopt_metal_buffer *t_buf = Buffer_val(v_token_buf);
  llmopt_metal_buffer *o_buf = Buffer_val(v_out_buf);

  if (lib == NULL) {
    caml_failwith("Metal library has been finalized");
  }

  mlsize_t total = Wosize_val(v_dispatches);
  if (total == 0) {
    caml_invalid_argument("prebaked Metal plan requires at least one dispatch");
  }
  for (mlsize_t i = 0; i < total; i++) {
    value item = Field(v_dispatches, i);
    value rem = Field(item, 1);
    NSUInteger buffer_count = 0;
    while (rem != Val_emptylist) {
      buffer_count++;
      rem = Field(rem, 1);
    }
    if (buffer_count > 16) {
      caml_invalid_argument("prebaked Metal dispatch exceeds 16 buffers");
    }
    if (caml_string_length(Field(item, 2)) > 256) {
      caml_invalid_argument("prebaked Metal dispatch parameters exceed 256 bytes");
    }
  }

  llmopt_prebaked_plan *plan = calloc(1, sizeof(*plan));
  if (plan == NULL) {
    caml_raise_out_of_memory();
  }

  plan->records = calloc(total, sizeof(llmopt_dispatch_record));
  plan->record_count = total;
  plan->library = lib;
  plan->queue = [lib->queue retain];

  if (t_buf != NULL) {
    plan->token_buffer = [t_buf->buffer retain];
    plan->token_buffer_offset = t_buf->offset;
  }
  if (o_buf != NULL) {
    plan->output_buffer = [o_buf->buffer retain];
    plan->output_buffer_offset = o_buf->offset;
  }

  for (mlsize_t i = 0; i < total; i++) {
    value item = Field(v_dispatches, i);
    const char *kname = String_val(Field(item, 0));
    snprintf(plan->records[i].kernel_name, sizeof(plan->records[i].kernel_name), "%s", kname);
    value buffers = Field(item, 1);
    value params = Field(item, 2);
    plan->records[i].grid = MTLSizeMake(Long_val(Field(item, 3)), Long_val(Field(item, 4)), Long_val(Field(item, 5)));
    plan->records[i].group = MTLSizeMake(Long_val(Field(item, 6)), Long_val(Field(item, 7)), Long_val(Field(item, 8)));
    if (getenv("LLMOPT_PROFILE_PLAN") != NULL) {
      fprintf(stderr, "llmopt-prebaked-kernel[%zu]: %s grid=(%lu,%lu,%lu) group=(%lu,%lu,%lu)\n",
              i, kname, (unsigned long)plan->records[i].grid.width,
              (unsigned long)plan->records[i].grid.height, (unsigned long)plan->records[i].grid.depth,
              (unsigned long)plan->records[i].group.width,
              (unsigned long)plan->records[i].group.height, (unsigned long)plan->records[i].group.depth);
    }
    plan->records[i].barrier_before = Bool_val(Field(item, 9));
    if (plan->records[i].barrier_before) {
      plan->barrier_count += 1;
    }
    plan->records[i].pipeline = [pipeline_for_name(lib, kname) retain];
    plan->records[i].is_paged_attention = is_paged_attention_kernel(kname);
    plan->records[i].is_checkpoint_pack = (strcmp(kname, "llmopt_cache_pack_checkpoint_q8_simd") == 0);

    NSUInteger b_idx = 0;
    value rem = buffers;
    while (rem != Val_emptylist && b_idx < 16) {
      llmopt_metal_buffer *b = Buffer_val(Field(rem, 0));
      if (b != NULL) {
        plan->records[i].buffers[b_idx] = [b->buffer retain];
        plan->records[i].offsets[b_idx] = b->offset;
        plan->records[i].lengths[b_idx] = b->length;
      }
      b_idx++;
      rem = Field(rem, 1);
    }
    plan->records[i].buffer_count = b_idx;

    mlsize_t plen = caml_string_length(params);
    if (plen > 0 && plen <= 256) {
      memcpy(plan->records[i].parameters, Bytes_val(params), plen);
      plan->records[i].parameter_length = plen;
    }
  }

  CAMLreturn(alloc_prebaked(plan));
}

static bool range_has_nan(const char *kname, id<MTLBuffer> buf, NSUInteger offset, NSUInteger length) {
  if (buf == nil || length == 0) return false;
  if (strstr(kname, "_i64") != NULL || strstr(kname, "cache_pack") != NULL) return false;
  NSUInteger total_len = [buf length];
  if (offset >= total_len) return false;
  if (offset + length > total_len) {
    length = total_len - offset;
  }
  const uint8_t *raw = (const uint8_t *)[buf contents] + offset;
  if (strcmp(kname, "llmopt_matmul") == 0 || strstr(kname, "_f32") != NULL) {
    const uint32_t *u32 = (const uint32_t *)raw;
    NSUInteger count32 = length / 4;
    for (NSUInteger k = 0; k < count32; k++) {
      uint32_t v = u32[k];
      if ((v & 0x7F800000) == 0x7F800000 && (v & 0x007FFFFF) != 0) {
        return true;
      }
    }
  } else {
    const uint16_t *u16 = (const uint16_t *)raw;
    NSUInteger count16 = length / 2;
    for (NSUInteger k = 0; k < count16; k++) {
      uint16_t v = u16[k];
      if ((v & 0x7C00) == 0x7C00 && (v & 0x03FF) != 0) {
        return true;
      }
    }
  }
  return false;
}

CAMLprim value caml_llmopt_prebaked_execute(value v_plan, value v_token, value v_past_tokens, value v_checkpoint) {
  CAMLparam4(v_plan, v_token, v_past_tokens, v_checkpoint);
  llmopt_prebaked_plan *plan = Prebaked_val(v_plan);
  int64_t token = (int64_t)Long_val(v_token);
  int32_t past_tokens = (int32_t)Long_val(v_past_tokens);
  int32_t checkpoint = (int32_t)Long_val(v_checkpoint);
  if (plan->token_buffer != nil) {
    int64_t *ptr = (int64_t *)((uint8_t *)[plan->token_buffer contents] + plan->token_buffer_offset);
    *ptr = token;
  }

  const char *nan_check_env = getenv("LLMOPT_DEBUG_NAN_CHECK");
  if (nan_check_env != NULL && (past_tokens > 0 || getenv("LLMOPT_CHECK_PREFILL") != NULL)) {
    @autoreleasepool {
      fprintf(stderr, "[DEBUG_NAN_CHECK] Starting check across %zu records (past_tokens=%d)...\n",
              plan->record_count, past_tokens);
      for (size_t i = 0; i < plan->record_count; i++) {
        llmopt_dispatch_record *r = &plan->records[i];
        bool had_bad_before[16] = {false};
        for (NSUInteger b = 0; b < r->buffer_count; b++) {
          if (range_has_nan(r->kernel_name, r->buffers[b], r->offsets[b], r->lengths[b])) {
            had_bad_before[b] = true;
          }
        }

        id<MTLCommandBuffer> cmd = [plan->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:r->pipeline];
        if (r->buffer_count > 0) {
          [enc setBuffers:r->buffers offsets:r->offsets
                withRange:NSMakeRange(0, r->buffer_count)];
        }
        if (r->parameter_length > 0) {
          if (r->is_paged_attention && r->parameter_length >= 16) {
            *(uint32_t *)(&r->parameters[12]) = (uint32_t)past_tokens;
          }
          if (r->is_checkpoint_pack && r->parameter_length >= 4) {
            *(uint32_t *)(&r->parameters[0]) = (uint32_t)checkpoint;
          }
          [enc setBytes:r->parameters length:r->parameter_length
                atIndex:r->buffer_count];
        }
        [enc dispatchThreads:r->grid threadsPerThreadgroup:r->group];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.status != MTLCommandBufferStatusCompleted) {
          fprintf(stderr, "[DEBUG_NAN_CHECK] Dispatch %zu '%s' failed on Metal: %s\n",
                  i, r->kernel_name, [[cmd.error localizedDescription] UTF8String]);
        }

        for (NSUInteger b = 0; b < r->buffer_count; b++) {
          if (range_has_nan(r->kernel_name, r->buffers[b], r->offsets[b], r->lengths[b])) {
            if (!had_bad_before[b]) {
              fprintf(stderr, "\n==================================================\n");
              fprintf(stderr, "[DEBUG_NAN_CHECK] FIRST NaN PRODUCED AT RECORD %zu:\n", i);
              fprintf(stderr, "  kernel: '%s'\n", r->kernel_name);
              fprintf(stderr, "  buffer: %lu (offset=%lu, length=%lu, total_buf_len=%lu)\n",
                      (unsigned long)b, (unsigned long)r->offsets[b], (unsigned long)r->lengths[b],
                      (unsigned long)[r->buffers[b] length]);
              fprintf(stderr, "  grid: (%lu, %lu, %lu), group: (%lu, %lu, %lu)\n",
                      r->grid.width, r->grid.height, r->grid.depth,
                      r->group.width, r->group.height, r->group.depth);
              if (r->parameter_length > 0) {
                fprintf(stderr, "  parameters (%lu bytes, u32):", r->parameter_length);
                for (NSUInteger p = 0; p + 4 <= r->parameter_length && p < 48; p += 4) {
                  fprintf(stderr, " %u", *(uint32_t *)(&r->parameters[p]));
                }
                fprintf(stderr, "\n");
              }
              const uint16_t *words = (const uint16_t *)((const uint8_t *)[r->buffers[b] contents] + r->offsets[b]);
              fprintf(stderr, "  first 16 half words (hex):");
              for (int k = 0; k < 16 && (NSUInteger)k * 2 < r->lengths[b]; k++) {
                fprintf(stderr, " 0x%04x", words[k]);
              }
              fprintf(stderr, "\n");
              for (NSUInteger in_b = 0; in_b < r->buffer_count; in_b++) {
                fprintf(stderr, "  buffer[%lu]: offset=%lu, len=%lu, total=%lu\n",
                        (unsigned long)in_b, (unsigned long)r->offsets[in_b],
                        (unsigned long)r->lengths[in_b], (unsigned long)[r->buffers[in_b] length]);
              }
              fprintf(stderr, "==================================================\n\n");
              fflush(stderr);
              exit(1);
            }
          }
        }
      }
    }
  } else if (getenv("LLMOPT_PROFILE_BREAKDOWN") != NULL && past_tokens > 0) {
    @autoreleasepool {
      typedef struct {
        char name[128];
        double total_ms;
        size_t count;
      } kernel_stat;
      kernel_stat stats[128];
      size_t stat_count = 0;
      double overall_gpu_ms = 0.0;

      for (size_t i = 0; i < plan->record_count; i++) {
        llmopt_dispatch_record *r = &plan->records[i];
        id<MTLCommandBuffer> cmd = [plan->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:r->pipeline];
        if (r->buffer_count > 0) {
          [enc setBuffers:r->buffers offsets:r->offsets
                withRange:NSMakeRange(0, r->buffer_count)];
        }
        if (r->parameter_length > 0) {
          if (r->is_paged_attention && r->parameter_length >= 16) {
            *(uint32_t *)(&r->parameters[12]) = (uint32_t)past_tokens;
          }
          if (r->is_checkpoint_pack && r->parameter_length >= 4) {
            *(uint32_t *)(&r->parameters[0]) = (uint32_t)checkpoint;
          }
          [enc setBytes:r->parameters length:r->parameter_length
                atIndex:r->buffer_count];
        }
        [enc dispatchThreads:r->grid threadsPerThreadgroup:r->group];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        double d_ms = 0.0;
        if (cmd.GPUEndTime > cmd.GPUStartTime) {
          d_ms = (cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0;
        }
        overall_gpu_ms += d_ms;
        size_t found = stat_count;
        for (size_t s = 0; s < stat_count; s++) {
          if (strcmp(stats[s].name, r->kernel_name) == 0) {
            found = s;
            break;
          }
        }
        if (found < stat_count) {
          stats[found].total_ms += d_ms;
          stats[found].count += 1;
        } else if (stat_count < 128) {
          snprintf(stats[stat_count].name, sizeof(stats[stat_count].name), "%s", r->kernel_name);
          stats[stat_count].total_ms = d_ms;
          stats[stat_count].count = 1;
          stat_count++;
        }
      }
      for (size_t i = 0; i < stat_count; i++) {
        for (size_t j = i + 1; j < stat_count; j++) {
          if (stats[j].total_ms > stats[i].total_ms) {
            kernel_stat tmp = stats[i];
            stats[i] = stats[j];
            stats[j] = tmp;
          }
        }
      }
      fprintf(stderr, "\n========== DECODE KERNEL BREAKDOWN (past_tokens=%d) ==========\n", past_tokens);
      fprintf(stderr, "%-48s | %5s | %10s | %10s | %6s\n", "Kernel Name", "Calls", "Total (ms)", "Avg (us)", "Pct");
      fprintf(stderr, "-------------------------------------------------+-------+------------+------------+-------\n");
      for (size_t s = 0; s < stat_count; s++) {
        double avg_us = (stats[s].total_ms / stats[s].count) * 1000.0;
        double pct = overall_gpu_ms > 0 ? (stats[s].total_ms / overall_gpu_ms) * 100.0 : 0.0;
        fprintf(stderr, "%-48s | %5zu | %10.3f | %10.2f | %5.1f%%\n",
                stats[s].name, stats[s].count, stats[s].total_ms, avg_us, pct);
      }
      fprintf(stderr, "-------------------------------------------------+-------+------------+------------+-------\n");
      fprintf(stderr, "%-48s | %5zu | %10.3f |            | 100.0%%\n", "TOTAL DISPATCHES", plan->record_count, overall_gpu_ms);
      fprintf(stderr, "===============================================================\n\n");
    }
  } else {
    @autoreleasepool {
    uint64_t started = mach_absolute_time();
    id<MTLCommandBuffer> cmd = [plan->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    id<MTLComputePipelineState> current_pipeline = nil;
    for (size_t i = 0; i < plan->record_count; i++) {
      llmopt_dispatch_record *r = &plan->records[i];
      if (r->barrier_before && llmopt_enable_serial_barriers()) {
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
      if (r->pipeline != current_pipeline) {
        [enc setComputePipelineState:r->pipeline];
        current_pipeline = r->pipeline;
      }
      if (r->buffer_count > 0) {
        [enc setBuffers:r->buffers offsets:r->offsets
              withRange:NSMakeRange(0, r->buffer_count)];
      }
      if (r->parameter_length > 0) {
        if (r->is_paged_attention && r->parameter_length >= 16) {
          *(uint32_t *)(&r->parameters[12]) = (uint32_t)past_tokens;
        }
        if (r->is_checkpoint_pack && r->parameter_length >= 4) {
          *(uint32_t *)(&r->parameters[0]) = (uint32_t)checkpoint;
        }
        [enc setBytes:r->parameters length:r->parameter_length
              atIndex:r->buffer_count];
      }
      [enc dispatchThreads:r->grid threadsPerThreadgroup:r->group];
    }

    [enc endEncoding];
    [cmd commit];
    caml_enter_blocking_section();
    [cmd waitUntilCompleted];
    caml_leave_blocking_section();
    uint64_t finished = mach_absolute_time();
    if (getenv("LLMOPT_PROFILE_BATCH") != NULL) {
      double gpu_seconds = llmopt_command_duration(cmd, started, finished);
      mach_timebase_info_data_t info;
      mach_timebase_info(&info);
      double wall_seconds =
          ((double)(finished - started) * (double)info.numer /
           (double)info.denom) * 1e-9;
      fprintf(stderr,
              "llmopt-prebaked-profile: dispatches=%zu barriers=%zu gpu_ms=%.6f wait_ms=%.6f\n",
              plan->record_count, plan->barrier_count,
              gpu_seconds * 1000.0, wall_seconds * 1000.0);
    }
    if (cmd.status != MTLCommandBufferStatusCompleted) {
      fail_with_error("Prebaked Metal batch failed", cmd.error);
    }
  }
  }

  int32_t next_token = 0;
  if (plan->output_buffer != nil) {
    int32_t *ptr = (int32_t *)((uint8_t *)[plan->output_buffer contents] + plan->output_buffer_offset);
    next_token = *ptr;
  }
  CAMLreturn(Val_long((intptr_t)next_token));
}

CAMLprim value caml_llmopt_f16_argmax(value v_bytes, value v_count) {
  CAMLparam2(v_bytes, v_count);
  const _Float16 *ptr = (const _Float16 *)String_val(v_bytes);
  size_t count = (size_t)Long_val(v_count);
  if (count == 0) CAMLreturn(Val_long(0));
  _Float16 max_val = ptr[0];
  size_t max_idx = 0;
  for (size_t i = 1; i < count; i++) {
    if (ptr[i] > max_val) {
      max_val = ptr[i];
      max_idx = i;
    }
  }
  CAMLreturn(Val_long((intptr_t)max_idx));
}

typedef struct {
  uint32_t token;
  float logit;
} LLMOptTokenLogit;

static inline void llmopt_heap_sift_down(LLMOptTokenLogit *heap, int size, int idx) {
  while (1) {
    int smallest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;
    if (left < size && heap[left].logit < heap[smallest].logit) smallest = left;
    if (right < size && heap[right].logit < heap[smallest].logit) smallest = right;
    if (smallest == idx) break;
    LLMOptTokenLogit tmp = heap[idx];
    heap[idx] = heap[smallest];
    heap[smallest] = tmp;
    idx = smallest;
  }
}

static inline void llmopt_heap_sift_up(LLMOptTokenLogit *heap, int idx) {
  while (idx > 0) {
    int parent = (idx - 1) / 2;
    if (heap[idx].logit < heap[parent].logit) {
      LLMOptTokenLogit tmp = heap[idx];
      heap[idx] = heap[parent];
      heap[parent] = tmp;
      idx = parent;
    } else {
      break;
    }
  }
}

static inline uint64_t llmopt_xoroshiro128plus(uint64_t s[2]) {
  const uint64_t s0 = s[0];
  uint64_t s1 = s[1];
  const uint64_t result = s0 + s1;
  s1 ^= s0;
  s[0] = ((s0 << 24) | (s0 >> 40)) ^ s1 ^ (s1 << 16);
  s[1] = (s1 << 37) | (s1 >> 27);
  return result;
}

static inline float llmopt_random_uniform_01(uint64_t s[2]) {
  return (float)(llmopt_xoroshiro128plus(s) >> 11) * (1.0f / 9007199254740992.0f);
}

CAMLprim value caml_llmopt_sample_f16_native(
    value v_bytes, value v_count, value v_temp, value v_top_k,
    value v_top_p, value v_min_p, value v_seed) {
  CAMLparam5(v_bytes, v_count, v_temp, v_top_k, v_top_p);
  CAMLxparam2(v_min_p, v_seed);

  const _Float16 *ptr = (const _Float16 *)String_val(v_bytes);
  size_t count = (size_t)Long_val(v_count);
  if (count == 0) CAMLreturn(Val_long(0));

  float temp = (float)Double_val(v_temp);
  int top_k = (int)Long_val(v_top_k);
  float top_p = (float)Double_val(v_top_p);
  float min_p = (float)Double_val(v_min_p);
  uint64_t seed = (uint64_t)Long_val(v_seed);

  // Fast-path for greedy
  if (temp <= 0.0001f || top_k == 1) {
    _Float16 max_val = ptr[0];
    size_t max_idx = 0;
    for (size_t i = 1; i < count; i++) {
      if (ptr[i] > max_val) {
        max_val = ptr[i];
        max_idx = i;
      }
    }
    CAMLreturn(Val_long((intptr_t)max_idx));
  }

  // Bound Top-K for stack allocation
  int eff_k = 64;
  if (top_k > 0 && top_k <= 256) eff_k = top_k;
  else if (top_k > 256) eff_k = 256;

  LLMOptTokenLogit heap[256];
  int heap_size = 0;

  // Single-pass Top-K streaming scan
  for (size_t i = 0; i < count; i++) {
    float val = (float)ptr[i];
    if (heap_size < eff_k) {
      heap[heap_size].token = (uint32_t)i;
      heap[heap_size].logit = val;
      llmopt_heap_sift_up(heap, heap_size);
      heap_size++;
    } else if (val > heap[0].logit) {
      heap[0].token = (uint32_t)i;
      heap[0].logit = val;
      llmopt_heap_sift_down(heap, eff_k, 0);
    }
  }

  // Extract heap and sort descending
  LLMOptTokenLogit sorted[256];
  int n_candidates = heap_size;
  for (int i = n_candidates - 1; i >= 0; i--) {
    sorted[i] = heap[0];
    heap[0] = heap[heap_size - 1];
    heap_size--;
    llmopt_heap_sift_down(heap, heap_size, 0);
  }

  float max_logit = sorted[0].logit;
  float inv_temp = 1.0f / temp;

  // Compute exp((logit - max_logit) / temp)
  float weights[256];
  float sum_weights = 0.0f;
  for (int i = 0; i < n_candidates; i++) {
    float scaled = (sorted[i].logit - max_logit) * inv_temp;
    float w = expf(scaled);
    weights[i] = w;
    sum_weights += w;
  }

  // Min-P filtering
  int valid_candidates = n_candidates;
  if (min_p > 0.0f && sum_weights > 0.0f) {
    float max_prob = weights[0] / sum_weights;
    float prob_thresh = max_prob * min_p;
    float filtered_sum = 0.0f;
    int count_valid = 0;
    for (int i = 0; i < n_candidates; i++) {
      if (weights[i] / sum_weights >= prob_thresh) {
        filtered_sum += weights[i];
        count_valid++;
      } else {
        break;
      }
    }
    if (count_valid > 0) {
      valid_candidates = count_valid;
      sum_weights = filtered_sum;
    }
  }

  // Top-P (Nucleus) Truncation
  if (top_p > 0.0f && top_p < 1.0f && sum_weights > 0.0f) {
    float cum_p = 0.0f;
    float p_sum = 0.0f;
    int p_cutoff = valid_candidates;
    for (int i = 0; i < valid_candidates; i++) {
      float p = weights[i] / sum_weights;
      cum_p += p;
      p_sum += weights[i];
      if (cum_p >= top_p) {
        p_cutoff = i + 1;
        break;
      }
    }
    valid_candidates = p_cutoff;
    sum_weights = p_sum;
  }

  // Sample token from distribution
  uint64_t rng[2];
  if (seed == 0) {
    seed = (uint64_t)mach_absolute_time();
  }
  rng[0] = seed ^ 0x9E3779B97F4A7C15ULL;
  rng[1] = (seed << 31) | (seed >> 33);
  if (rng[0] == 0 && rng[1] == 0) { rng[0] = 1; rng[1] = 2; }

  float r = llmopt_random_uniform_01(rng) * sum_weights;
  float acc = 0.0f;
  uint32_t chosen_token = sorted[0].token;

  for (int i = 0; i < valid_candidates; i++) {
    acc += weights[i];
    if (acc >= r) {
      chosen_token = sorted[i].token;
      break;
    }
  }

  CAMLreturn(Val_long((intptr_t)chosen_token));
}

CAMLprim value caml_llmopt_sample_f16_bytecode(value *argv, int argn) {
  (void)argn;
  return caml_llmopt_sample_f16_native(
      argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]);
}
