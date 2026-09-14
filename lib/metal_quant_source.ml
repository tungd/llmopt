(** Metal quant source selection and entry assembly: quant_common_source,
   the find/replace/extract text patchers, per-format source selection
   (legacy + Metal_block_codegen structural), quant linear-_add shim,
   kquant_entries, and the public emit_dequant_* wrappers.  Split out of
   metal.ml (LOOP item 5). *)

open Metal_source_common

let ( let* ) = Result.bind

module Tactic = Metal_selection.Tactic

let q8_0_source = Metal_q8_0_source.q8_0_source
let q5_0_source = Metal_q5_0_source.q5_0_source
let q4_0_source = Metal_q4_0_source.q4_0_source
let q4_k_source = Metal_q4_k_source.q4_k_source
let q5_k_source = Metal_q5_k_source.q5_k_source
let q6_k_source = Metal_q6_k_source.q6_k_source
let iq4_xs_source = Metal_iq4_xs_source.iq4_xs_source

let quant_common_source = {|
struct QuantBlock32Params {
    uint total_blocks;
};

struct QuantBlock256Params {
    uint total_superblocks;
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

let legacy_quant_source = function
  | Ir.Dtype.Q8_0 -> q8_0_source
  | Ir.Dtype.Q4_K -> q4_k_source
  | Ir.Dtype.Q5_K -> q5_k_source
  | Ir.Dtype.Q6_K -> q6_k_source
  | Ir.Dtype.Q5_0 -> q5_0_source
  | Ir.Dtype.Q4_0 -> q4_0_source
  | Ir.Dtype.IQ4_XS -> iq4_xs_source

let quant_source ~target = function
  | (Ir.Dtype.Q8_0 | Ir.Dtype.Q4_K | Ir.Dtype.Q5_K | Ir.Dtype.Q6_K
    | Ir.Dtype.Q5_0 | Ir.Dtype.Q4_0 | Ir.Dtype.IQ4_XS) as quant ->
      let* schema =
        Block_schema.of_storage
          (Ir.Linear_storage.Block_quantized quant)
        |> Result.map_error Block_schema.Error.to_string
      in
      let* generated = Metal_block_codegen.linear_kernels ~target schema in
      Ok
        (generated
        ^ (legacy_quant_source quant
          |> remove_kernel
               (Printf.sprintf "llmopt_%s_linear_f16"
                  (Ir.Dtype.quant_to_string quant |> String.lowercase_ascii))
          |> remove_kernel
               (Printf.sprintf "llmopt_%s_linear_f16_m2"
                  (Ir.Dtype.quant_to_string quant |> String.lowercase_ascii))))

(* G4P12-04d-A2 + G4P12-05: this text-patcher is the defensive fallback of
   the quant-add fold.  Every served _add base is structural today: the
   block32 and superblock families' generic/_m2 twins (and IQ4_XS m2), the K
   families' m2_x4 twins (G4P12-05-2), and the Q4_0 m4/m16/m32 zoo statics
   all come from Metal_block_codegen / the parametric family emission before
   this shim is reached; Q4_0 x8/x4 bases fail loudly in the fold.  A future
   base reaching this fallback still gets the legacy llama-style extract +
   store rewrite when its zoo text exists, so no served surface can silently
   lose its twin. *)
let quant_linear_add_source ~target (quant, tactic) =
  ignore target;
  let name = Tactic.name tactic in
  if quant = Ir.Dtype.Q4_0 then
    invalid_arg
      "quant_linear_add_source: Q4_0 _add twins come from Metal_block_codegen \
       (G4P12-04d-A2); dispatch Q4_0 bases before reaching this shim"
  else
    let source = extract_kernel (legacy_quant_source quant) name in
    let replacements =
      [ ( "kernel void " ^ name ^ "(",
          "kernel void " ^ name ^ "_add(" );
      ( "    if (params.has_bias != 0u) acc += float(bias[col]);\n",
        "" );
      ( "    if (params.has_bias != 0u) accumulator += float(bias[col]);\n",
        "" );
      ( "    const float bias_value = params.has_bias != 0u ? float(bias[col]) : 0.0f;\n",
        "" );
      ( "      const float bias_value = params.has_bias != 0u ? float(bias[col]) : 0.0f;\n",
        "" );
      ( "    output[row * params.n + col] = half(acc);\n",
        "    const uint output_index = row * params.n + col;\n"
        ^ "    output[output_index] = half(half(acc) + bias[output_index]);\n" );
      ( "    output[row * params.n + col] = half(accumulator);\n",
        "    const uint output_index = row * params.n + col;\n"
        ^ "    output[output_index] = half(half(accumulator) + bias[output_index]);\n" );
      ( "    output[col] = half(acc0 + bias_value);\n",
        "    output[col] = half(half(acc0) + bias[col]);\n" );
      ( "    output[params.n + col] = half(acc1 + bias_value);\n",
        "    output[params.n + col] = half(half(acc1) + bias[params.n + col]);\n" );
      ( "      output[row * params.n + col] = half(sum + bias_value);\n",
        "      const uint output_index = row * params.n + col;\n"
        ^ "      output[output_index] = half(half(sum) + bias[output_index]);\n" );
      ( "    output[row * params.n + col] = half(sum + bias_value);\n",
        "    const uint output_index = row * params.n + col;\n"
        ^ "    output[output_index] = half(half(sum) + bias[output_index]);\n" );
      ( "      output[col] = half(sum0 + bias_value);\n",
        "      output[col] = half(half(sum0) + bias[col]);\n" );
      ( "      output[params.n + col] = half(sum1 + bias_value);\n",
        "      output[params.n + col] = half(half(sum1) + bias[params.n + col]);\n" );
      ( "    output[col] = half(sum0 + bias_value);\n",
        "    output[col] = half(half(sum0) + bias[col]);\n" );
      ( "    output[params.n + col] = half(sum1 + bias_value);\n",
        "    output[params.n + col] = half(half(sum1) + bias[params.n + col]);\n" );
      ( "    if (params.has_bias != 0u) total += float(bias[col]);\n",
        "" );
      ( "    output[col] = half(total);\n",
        "    output[col] = half(half(total) + bias[col]);\n" );
      ( "    output[col] = half(total0 + bias_value);\n",
        "    output[col] = half(half(total0) + bias[col]);\n" );
      ( "    output[params.n + col] = half(total1 + bias_value);\n",
        "    output[params.n + col] = half(half(total1) + bias[params.n + col]);\n" ) ]
  in
  let fused =
    List.fold_left
      (fun source (pattern, replacement) ->
        replace_all source ~pattern ~replacement)
      source replacements
  in
  fused ^ "\n"

let kquant_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_dequant_q4_k"
      ~operation:Kernel_abi.Operation.Cast
      ~input_dtype:(Ir.Dtype.Quant Q4_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_dequant_q5_k"
      ~operation:Kernel_abi.Operation.Cast
      ~input_dtype:(Ir.Dtype.Quant Q5_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_dequant_q6_k"
      ~operation:Kernel_abi.Operation.Cast
      ~input_dtype:(Ir.Dtype.Quant Q6_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_k_linear_f16"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_k_linear_f16_m2"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_k_linear_f16_splitk"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_k_linear_f16_splitk_m2"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_q4_k_linear_f16_llama_mv_add"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_q4_k_linear_f16_m2_x2_l32"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_q4_k_linear_f16_m2_n2_l32"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_q4_k_linear_f16_m2_x4"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q5_k_linear_f16"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q5_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q5_k_linear_f16_m2"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q5_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q5_k_linear_f16_splitk"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q5_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q5_k_linear_f16_splitk_m2"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q5_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_q5_k_linear_f16_m2_x1_l32"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q5_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_q5_k_linear_f16_m2_n1_l32"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q5_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_q5_k_linear_f16_m2_x4"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q5_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q6_k_linear_f16"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q6_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q6_k_linear_f16_m2"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q6_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q6_k_linear_f16_splitk"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q6_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q6_k_linear_f16_splitk_m2"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q6_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_q6_k_linear_f16_llama_mv_add"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q6_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_q6_k_linear_f16_m2_x4"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q6_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_iq4_xs_linear_f16"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant IQ4_XS) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_iq4_xs_linear_f16_m2"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant IQ4_XS) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_k_gated_linear_f16"
      ~operation:Kernel_abi.Operation.Gated_linear
      ~input_dtype:(Ir.Dtype.Quant Q4_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_q4_k_gated_linear_f16_m2_x1_l32"
      ~operation:Kernel_abi.Operation.Gated_linear
      ~input_dtype:(Ir.Dtype.Quant Q4_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_q5_k_gated_linear_f16_m2_x1_l32"
      ~operation:Kernel_abi.Operation.Gated_linear
      ~input_dtype:(Ir.Dtype.Quant Q5_K) ~output_dtype:Ir.Dtype.Float16 ]

let emit_dequant_q8_0 () = metal_header ^ quant_common_source ^ q8_0_source
let emit_dequant_q5_0 () = metal_header ^ quant_common_source ^ q5_0_source
let emit_dequant_q4_0 () = metal_header ^ quant_common_source ^ q4_0_source
let emit_dequant_q4_k () = metal_header ^ quant_common_source ^ q4_k_source
let emit_dequant_q5_k () = metal_header ^ quant_common_source ^ q5_k_source
let emit_dequant_q6_k () = metal_header ^ quant_common_source ^ q6_k_source
