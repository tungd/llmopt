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
  uint8_t parameters[256];
  NSUInteger parameter_length;
  MTLSize grid;
  MTLSize group;
  bool is_paged_attention;
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
  id<MTLBuffer> token_buffer;
  NSUInteger token_buffer_offset;
  id<MTLBuffer> output_buffer;
  NSUInteger output_buffer_offset;
} llmopt_ring_queue;

typedef struct {
  id<MTLCommandQueue> queue;
  llmopt_metal_library *library;
  llmopt_dispatch_record *records;
  size_t record_count;
  id<MTLBuffer> token_buffer;
  NSUInteger token_buffer_offset;
  id<MTLBuffer> output_buffer;
  NSUInteger output_buffer_offset;
} llmopt_prebaked_plan;

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
    [q->token_buffer release];
    [q->output_buffer release];
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
    id<MTLLibrary> metal_library =
        [context->device newLibraryWithFile:path error:&error];
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
    caml_invalid_argument("Metal buffer size must be positive");
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
    [pipeline release];
  }

  if (library->cache_count < LLMOPT_PIPELINE_CACHE_SIZE) {
    library->cache[library->cache_count].name = strdup(kernel_name);
    library->cache[library->cache_count].pipeline = pipeline;
    library->cache_count++;
  }
  return pipeline;
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
    [encoder setComputePipelineState:pipeline];

    NSUInteger buffer_index = 0;
    value remaining = buffers;
    while (remaining != Val_emptylist) {
      llmopt_metal_buffer *buffer = Buffer_val(Field(remaining, 0));
      if (buffer != NULL) {
        [encoder setBuffer:buffer->buffer
                    offset:buffer->offset
                   atIndex:buffer_index];
      }
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
    end_batch_compute(batch);
    id<MTLCommandBuffer> command = [batch->command retain];
    [batch->command commit];
    batch->finished = YES;
    [batch->command release];
    batch->command = nil;
    caml_enter_blocking_section();
    [command waitUntilCompleted];
    caml_leave_blocking_section();
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

CAMLprim value caml_llmopt_metal_dispatch_q8(value arguments) {
  CAMLparam1(arguments);
  llmopt_metal_library *library = Library_val(Field(arguments, 0));
  const char *kernel_name = String_val(Field(arguments, 1));
  llmopt_metal_buffer *input = Buffer_val(Field(arguments, 2));
  llmopt_metal_buffer *weight = Buffer_val(Field(arguments, 3));
  llmopt_metal_buffer *scale = Buffer_val(Field(arguments, 4));
  llmopt_metal_buffer *bias = Buffer_val(Field(arguments, 5));
  llmopt_metal_buffer *output = Buffer_val(Field(arguments, 6));
  intnat m_value = Long_val(Field(arguments, 7));
  intnat n_value = Long_val(Field(arguments, 8));
  intnat k_value = Long_val(Field(arguments, 9));
  BOOL has_bias = Bool_val(Field(arguments, 10));

  if (library == NULL) {
    caml_failwith("Metal library has been finalized");
  }
  if (m_value <= 0 || n_value <= 0 || k_value <= 0 ||
      (uintnat)m_value > UINT32_MAX || (uintnat)n_value > UINT32_MAX ||
      (uintnat)k_value > UINT32_MAX) {
    caml_invalid_argument("Q8 Metal dimensions must be positive uint32 values");
  }

  BOOL is_float = strstr(kernel_name, "_f32") != NULL;
  uint64_t element_size = is_float ? 4 : 2;
  uint64_t m = (uint64_t)m_value;
  uint64_t n = (uint64_t)n_value;
  uint64_t k = (uint64_t)k_value;
  require_buffer_size(input, m * k * element_size, "input");
  require_buffer_size(weight, n * k, "weight");
  require_buffer_size(scale, n * 2, "scale");
  require_buffer_size(bias, n * 2, "bias");
  require_buffer_size(output, m * n * element_size, "output");

  @autoreleasepool {
    id<MTLComputePipelineState> pipeline =
        pipeline_for_name(library, kernel_name);

    id<MTLCommandBuffer> command = [library->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (command == nil || encoder == nil) {
      caml_failwith("Metal could not create a command buffer or encoder");
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input->buffer offset:input->offset atIndex:0];
    [encoder setBuffer:weight->buffer offset:weight->offset atIndex:1];
    [encoder setBuffer:scale->buffer offset:scale->offset atIndex:2];
    [encoder setBuffer:bias->buffer offset:bias->offset atIndex:3];
    [encoder setBuffer:output->buffer offset:output->offset atIndex:4];
    llmopt_q8_params params = {
        (uint32_t)m, (uint32_t)n, (uint32_t)k, has_bias ? 1u : 0u};
    [encoder setBytes:&params length:sizeof(params) atIndex:5];
    MTLSize grid = MTLSizeMake(((n + 15) / 16) * 16,
                               ((m + 15) / 16) * 16, 1);
    MTLSize group = MTLSizeMake(16, 16, 1);
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
      fail_with_error("Metal Q8 dispatch failed", command_error);
    }
    [command_error release];
  }
  CAMLreturn(Val_unit);
}

static void *llmopt_worker_loop(void *arg) {
  llmopt_ring_queue *q = (llmopt_ring_queue *)arg;

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

    if (q->token_buffer != nil) {
      int64_t *ptr = (int64_t *)((uint8_t *)[q->token_buffer contents] + q->token_buffer_offset);
      *ptr = (int64_t)entry.token;
    }

    @autoreleasepool {
      id<MTLCommandBuffer> cmd = [q->queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

      for (size_t i = 0; i < q->record_count; i++) {
        llmopt_dispatch_record *r = &q->records[i];
        [enc setComputePipelineState:r->pipeline];
        for (NSUInteger b = 0; b < r->buffer_count; b++) {
          [enc setBuffer:r->buffers[b] offset:r->offsets[b] atIndex:b];
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
      [cmd waitUntilCompleted];
    }

    int32_t next_token = 0;
    if (q->output_buffer != nil) {
      int32_t *out_ptr = (int32_t *)((uint8_t *)[q->output_buffer contents] + q->output_buffer_offset);
      next_token = *out_ptr;
    }

    uint32_t cq_tail = atomic_load_explicit(&q->cq_tail, memory_order_relaxed);
    q->cq[cq_tail % LLMOPT_RING_CAPACITY].request_id = entry.request_id;
    q->cq[cq_tail % LLMOPT_RING_CAPACITY].token = next_token;
    q->cq[cq_tail % LLMOPT_RING_CAPACITY].status = 0;
    atomic_store_explicit(&q->cq_tail, cq_tail + 1, memory_order_release);

    pthread_mutex_lock(&q->mutex);
    pthread_cond_signal(&q->cq_cond);
    pthread_mutex_unlock(&q->mutex);
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
    q->token_buffer = [t_buf->buffer retain];
    q->token_buffer_offset = t_buf->offset;
  }
  if (o_buf != NULL) {
    q->output_buffer = [o_buf->buffer retain];
    q->output_buffer_offset = o_buf->offset;
  }

  for (mlsize_t i = 0; i < total; i++) {
    value item = Field(v_dispatches, i);
    const char *kname = String_val(Field(item, 0));
    value buffers = Field(item, 1);
    value params = Field(item, 2);
    q->records[i].grid = MTLSizeMake(Long_val(Field(item, 3)), Long_val(Field(item, 4)), Long_val(Field(item, 5)));
    q->records[i].group = MTLSizeMake(Long_val(Field(item, 6)), Long_val(Field(item, 7)), Long_val(Field(item, 8)));
    q->records[i].pipeline = [pipeline_for_name(lib, kname) retain];
    q->records[i].is_paged_attention = (strcmp(kname, "llmopt_attention_q8_paged_simd_h64") == 0);

    NSUInteger b_idx = 0;
    value rem = buffers;
    while (rem != Val_emptylist && b_idx < 16) {
      llmopt_metal_buffer *b = Buffer_val(Field(rem, 0));
      if (b != NULL) {
        q->records[i].buffers[b_idx] = [b->buffer retain];
        q->records[i].offsets[b_idx] = b->offset;
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

  llmopt_prebaked_plan *plan = calloc(1, sizeof(*plan));
  if (plan == NULL) {
    caml_raise_out_of_memory();
  }

  mlsize_t total = Wosize_val(v_dispatches);
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
    value buffers = Field(item, 1);
    value params = Field(item, 2);
    plan->records[i].grid = MTLSizeMake(Long_val(Field(item, 3)), Long_val(Field(item, 4)), Long_val(Field(item, 5)));
    plan->records[i].group = MTLSizeMake(Long_val(Field(item, 6)), Long_val(Field(item, 7)), Long_val(Field(item, 8)));
    plan->records[i].pipeline = [pipeline_for_name(lib, kname) retain];
    plan->records[i].is_paged_attention = (strcmp(kname, "llmopt_attention_q8_paged_simd_h64") == 0);

    NSUInteger b_idx = 0;
    value rem = buffers;
    while (rem != Val_emptylist && b_idx < 16) {
      llmopt_metal_buffer *b = Buffer_val(Field(rem, 0));
      if (b != NULL) {
        plan->records[i].buffers[b_idx] = [b->buffer retain];
        plan->records[i].offsets[b_idx] = b->offset;
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

CAMLprim value caml_llmopt_prebaked_execute(value v_plan, value v_token, value v_past_tokens) {
  CAMLparam3(v_plan, v_token, v_past_tokens);
  llmopt_prebaked_plan *plan = Prebaked_val(v_plan);
  int64_t token = (int64_t)Long_val(v_token);
  int32_t past_tokens = (int32_t)Long_val(v_past_tokens);
  if (plan->token_buffer != nil) {
    int64_t *ptr = (int64_t *)((uint8_t *)[plan->token_buffer contents] + plan->token_buffer_offset);
    *ptr = token;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> cmd = [plan->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    for (size_t i = 0; i < plan->record_count; i++) {
      llmopt_dispatch_record *r = &plan->records[i];
      [enc setComputePipelineState:r->pipeline];
      for (NSUInteger b = 0; b < r->buffer_count; b++) {
        if (r->buffers[b] != nil) {
          [enc setBuffer:r->buffers[b] offset:r->offsets[b] atIndex:b];
        }
      }
      if (r->parameter_length > 0) {
        if (r->is_paged_attention && r->parameter_length >= 16) {
          *(uint32_t *)(&r->parameters[12]) = (uint32_t)past_tokens;
        }
        [enc setBytes:r->parameters length:r->parameter_length atIndex:r->buffer_count];
      }
      [enc dispatchThreads:r->grid threadsPerThreadgroup:r->group];
    }

    [enc endEncoding];
    [cmd commit];
    caml_enter_blocking_section();
    [cmd waitUntilCompleted];
    caml_leave_blocking_section();
    if (cmd.status != MTLCommandBufferStatusCompleted) {
      fail_with_error("Prebaked Metal batch failed", cmd.error);
    }
  }

  int32_t next_token = 0;
  if (plan->output_buffer != nil) {
    int32_t *out_ptr = (int32_t *)((uint8_t *)[plan->output_buffer contents] + plan->output_buffer_offset);
    next_token = *out_ptr;
  }

  CAMLreturn(Val_long(next_token));
}
