(* Metal_block_codegen: public facade for constructive block-quantized
   Linear code generation. Dispatches to format-specific modules:
   - Metal_q4_0_codegen (Q4_0 linear/qkv/epilogue/multi/add)
   - Metal_block_codegen_formats (non-Q4_0 formats and add variants)
*)

let linear_kernels ~target schema =
  match Block_schema.quant_type schema with
  | Some Ir.Dtype.Q8_0 -> Metal_block_codegen_formats.q8_0_linear ~target schema
  | Some Ir.Dtype.Q4_K -> Metal_block_codegen_formats.q4_k_linear ~target schema
  | Some Ir.Dtype.Q5_K -> Metal_block_codegen_formats.q5_k_linear ~target schema
  | Some Ir.Dtype.Q6_K -> Metal_block_codegen_formats.q6_k_linear ~target schema
  | Some Ir.Dtype.Q5_0 -> Metal_block_codegen_formats.q5_0_linear ~target schema
  | Some Ir.Dtype.Q4_0 -> Metal_q4_0_codegen.q4_0_linear ~target schema
  | Some Ir.Dtype.IQ4_XS -> Metal_block_codegen_formats.iq4_xs_linear ~target schema
  | None -> Error "constructive Metal Linear emission requires block-quant storage"

let qkv_kernels ~target schema =
  match Block_schema.quant_type schema with
  | Some Ir.Dtype.Q4_0 -> Metal_q4_0_codegen.q4_0_qkv ~target schema
  | _ -> Error "constructive QKV emission only implemented for Q4_0"

let q4_0_x8_epilogue_kernel = Metal_q4_0_codegen.q4_0_x8_epilogue_kernel
let q4_0_x4_epilogue_kernel = Metal_q4_0_codegen.q4_0_x4_epilogue_kernel
let q4_0_m16_epilogue_kernel = Metal_q4_0_codegen.q4_0_m16_epilogue_kernel
let q4_0_m32_epilogue_kernel = Metal_q4_0_codegen.q4_0_m32_epilogue_kernel
let q4_0_multi_linear_f16_x8_kernel = Metal_q4_0_codegen.q4_0_multi_linear_f16_x8_kernel
let q4_0_multi_export_linear_f16_x8_kernel = Metal_q4_0_codegen.q4_0_multi_export_linear_f16_x8_kernel
let q4_0_linear_add_kernel = Metal_q4_0_codegen.q4_0_linear_add_kernel
let q4_0_m2_add_kernel = Metal_q4_0_codegen.q4_0_m2_add_kernel

let q8_0_linear_add_kernel = Metal_block_codegen_formats.q8_0_linear_add_kernel
let q8_0_m2_add_kernel = Metal_block_codegen_formats.q8_0_m2_add_kernel
let q5_0_linear_add_kernel = Metal_block_codegen_formats.q5_0_linear_add_kernel
let q5_0_m2_add_kernel = Metal_block_codegen_formats.q5_0_m2_add_kernel
let q4_k_linear_add_kernel = Metal_block_codegen_formats.q4_k_linear_add_kernel
let q5_k_linear_add_kernel = Metal_block_codegen_formats.q5_k_linear_add_kernel
let q6_k_linear_add_kernel = Metal_block_codegen_formats.q6_k_linear_add_kernel
let iq4_xs_linear_add_kernel = Metal_block_codegen_formats.iq4_xs_linear_add_kernel
let iq4_xs_m2_add_kernel = Metal_block_codegen_formats.iq4_xs_m2_add_kernel
let q4_k_m2_x4_add_kernel = Metal_block_codegen_formats.q4_k_m2_x4_add_kernel
let q5_k_m2_x4_add_kernel = Metal_block_codegen_formats.q5_k_m2_x4_add_kernel
let q6_k_m2_x4_add_kernel = Metal_block_codegen_formats.q6_k_m2_x4_add_kernel
