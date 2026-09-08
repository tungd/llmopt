(** Metal source catalog common helpers: the four kernel_entry* entry
   constructors and the shared metal_header preamble.  Split out of
   metal.ml (LOOP item 5). *)

let kernel_entry_with_threadgroup ~threadgroup ~name ~operation ~input_dtype
    ~output_dtype =
  match
    Kernel_abi.Entry.create ~name ~operation ~input_dtype ~output_dtype
      ~threadgroup
  with
  | Ok entry -> entry
  | Error message -> invalid_arg ("invalid built-in Metal kernel: " ^ message)

let kernel_entry ~name ~operation ~input_dtype ~output_dtype =
  kernel_entry_with_threadgroup ~threadgroup:(16, 16, 1) ~name ~operation
    ~input_dtype ~output_dtype

let kernel_entry_with_threadgroup_and_tile ~tile ~threadgroup ~name ~operation
    ~input_dtype ~output_dtype =
  match
    Kernel_abi.Entry.create_with_tile ~tile ~name ~operation ~input_dtype
      ~output_dtype ~threadgroup
  with
  | Ok entry -> entry
  | Error message -> invalid_arg ("invalid parameterized Metal kernel: " ^ message)

let kernel_entry_with_tile ~tile ~name ~operation ~input_dtype ~output_dtype =
  kernel_entry_with_threadgroup_and_tile ~tile ~threadgroup:(16, 16, 1) ~name
    ~operation ~input_dtype ~output_dtype

let metal_header =
  "#include <metal_stdlib>\n#include <metal_matrix>\nusing namespace metal;\n"
