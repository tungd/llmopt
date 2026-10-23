(** Metal quant source selection and entry assembly for the canonical
    Tile32_Affine{4,8} runtime family and Q4_0 baseline. *)

open Metal_source_common

let ( let* ) = Result.bind

module Tactic = Metal_selection.Tactic

let q4_0_source = Metal_q4_0_source.q4_0_source

let quant_common_source = {|
struct QuantBlock32Params {
    uint total_blocks;
};

struct QuantLinearParams {
    uint m;
    uint n;
    uint k;
    uint has_bias;
};

inline half llmopt_gated_product(
    float gate_acc, float up_acc, uint activation) {
  const half gate = half(gate_acc);
  const half up = half(up_acc);
  const float gate_value = float(gate);
  float activated = 0.0f;
  if (activation == 0u) {
    activated = gate_value / (1.0f + exp(-gate_value));
  } else if (activation == 1u) {
    activated = 0.5f * gate_value
      * (1.0f + tanh(clamp(0.7978845608f
        * (gate_value + 0.044715f * gate_value * gate_value * gate_value),
        -10.0f, 10.0f)));
  } else {
    activated = 1.0f / (1.0f + exp(-gate_value));
  }
  return half(half(activated) * up);
}
|}

let find_substring_from source pattern offset =
  let source_length = String.length source in
  let pattern_length = String.length pattern in
  let rec search index =
    if index + pattern_length > source_length then None
    else if String.sub source index pattern_length = pattern then Some index
    else search (index + 1)
  in
  search offset

let replace_all source ~pattern ~replacement =
  let pattern_length = String.length pattern in
  let buffer = Buffer.create (String.length source) in
  let rec copy offset =
    match find_substring_from source pattern offset with
    | None ->
        Buffer.add_substring buffer source offset (String.length source - offset)
    | Some index ->
        Buffer.add_substring buffer source offset (index - offset);
        Buffer.add_string buffer replacement;
        copy (index + pattern_length)
  in
  copy 0;
  Buffer.contents buffer

let extract_kernel source name =
  let signature = "kernel void " ^ name ^ "(" in
  match find_substring_from source signature 0 with
  | None -> invalid_arg ("missing quantized Linear kernel source: " ^ name)
  | Some start ->
      let stop =
        match find_substring_from source "\nkernel void " (start + 1) with
        | Some stop -> stop
        | None -> String.length source
      in
      String.sub source start (stop - start)

let remove_kernel name source =
  let signature = "kernel void " ^ name ^ "(" in
  match find_substring_from source signature 0 with
  | None -> source
  | Some start ->
      let stop =
        match find_substring_from source "\nkernel void " (start + 1) with
        | Some stop -> stop
        | None -> String.length source
      in
      String.sub source 0 start
      ^ String.sub source stop (String.length source - stop)

let quant_source ~target = function
  | (Ir.Dtype.Tile32_Affine4 | Ir.Dtype.Tile32_Affine8) as quant ->
      let* schema =
        Block_schema.of_storage
          (Ir.Linear_storage.Block_quantized quant)
        |> Result.map_error Block_schema.Error.to_string
      in
      Metal_block_codegen.linear_kernels ~target schema
  | Ir.Dtype.Q4_0 ->
      let* schema =
        Block_schema.of_storage
          (Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_0)
        |> Result.map_error Block_schema.Error.to_string
      in
      let* generated = Metal_block_codegen.linear_kernels ~target schema in
      Ok
        (generated
        ^ (q4_0_source
          |> remove_kernel "llmopt_q4_0_linear_f16"
          |> remove_kernel "llmopt_q4_0_linear_f16_m2"))
  | ( Ir.Dtype.Q8_0
    | Ir.Dtype.Q5_0
    | Ir.Dtype.Q4_K
    | Ir.Dtype.Q5_K
    | Ir.Dtype.Q6_K
    | Ir.Dtype.IQ4_XS ) as quant ->
      Error
        (Printf.sprintf
           "legacy GGUF format %s must be transcoded to Tile32_Affine{4,8} before Metal codegen"
           (Ir.Dtype.quant_to_string quant))

let emit_dequant_q4_0 () = metal_header ^ quant_common_source ^ q4_0_source

let emit_tile32_affine4 ?(target = Target_hardware.default) () =
  match quant_source ~target Ir.Dtype.Tile32_Affine4 with
  | Ok src -> metal_header ^ quant_common_source ^ src
  | Error msg -> invalid_arg msg

let emit_tile32_affine8 ?(target = Target_hardware.default) () =
  match quant_source ~target Ir.Dtype.Tile32_Affine8 with
  | Ok src -> metal_header ^ quant_common_source ^ src
  | Error msg -> invalid_arg msg
