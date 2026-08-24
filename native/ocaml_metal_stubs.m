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

typedef struct {
  id<MTLDevice> device;
  id<MTLLibrary> library;
  id<MTLCommandQueue> queue;
  NSMutableDictionary *pipelines;
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
  NSString *name = [NSString stringWithUTF8String:kernel_name];
  id<MTLComputePipelineState> pipeline =
      [library->pipelines objectForKey:name];
  if (pipeline != nil) {
    return pipeline;
  }
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
  return [library->pipelines objectForKey:name];
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

  @autoreleasepool {
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
