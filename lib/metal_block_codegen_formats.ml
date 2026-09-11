(* Internal module split (LOOP item 7): non-Q4_0 block code generation formats
   and their residual/add variants. *)

open Metal_block_codegen_common

let ( let* ) = Result.bind

let q4_k_linear ~target schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* scale_offset = field_offset physical Block_schema.Physical_layout.Field.Block_scale in
  let* minimum_offset = field_offset physical Block_schema.Physical_layout.Field.Block_minimum in
  let* table_offset = field_offset physical Block_schema.Physical_layout.Field.Scale_minimum_table in
  let* quant_offset = field_offset physical Block_schema.Physical_layout.Field.Quantized in
  let* () = require_superblock ~target physical logical "Q4_K" in
  if Block_schema.Logical_decode.code_bits equation <> 4 then
    Error "constructive Q4_K emitter requires four-bit codes"
  else
    let value_function = "llmopt_schema_q4_k_value" in
    let block_elements = Block_schema.Physical_layout.block_elements physical in
    let block_bytes = Block_schema.Physical_layout.block_bytes physical in
    let subblock_elements =
      Block_schema.Logical_decode.subblock_elements logical
    in
    Ok
      (emit_superblock ~m2_x4:true schema ~name:"q4_k" ~value_function
         (Printf.sprintf {|
inline float %s(device const uchar* block, uint lane, uint subblock) {
  const float d = float(*reinterpret_cast<device const half*>(block + %du));
  const float dmin = float(*reinterpret_cast<device const half*>(block + %du));
  device const uchar* table = block + %du;
  const uint j = subblock & 3u;
  const uint scale = subblock < 4u
      ? (table[j] & 63u)
      : ((table[j + 8u] & 15u) | ((table[j] >> 6u) << 4u));
  const uint minimum = subblock < 4u
      ? (table[j + 4u] & 63u)
      : ((table[j + 8u] >> 4u) | ((table[j + 4u] >> 6u) << 4u));
  const uchar packed = block[%du + (subblock >> 1u) * 32u + lane];
  const uint code = (subblock & 1u) == 0u ? (packed & 15u) : (packed >> 4u);
  return d * float(scale) * float(code) - dmin * float(minimum);
}
|} value_function scale_offset minimum_offset table_offset quant_offset)
      ^ emit_superblock_gated_m2 ~name:"q4_k" ~value_function
          ~block_elements ~block_bytes ~subblock_elements)

let q5_k_linear ~target schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* scale_offset = field_offset physical Block_schema.Physical_layout.Field.Block_scale in
  let* minimum_offset = field_offset physical Block_schema.Physical_layout.Field.Block_minimum in
  let* table_offset = field_offset physical Block_schema.Physical_layout.Field.Scale_minimum_table in
  let* high_offset = field_offset physical Block_schema.Physical_layout.Field.Quantized_high in
  let* low_offset = field_offset physical Block_schema.Physical_layout.Field.Quantized_low in
  let* () = require_superblock ~target physical logical "Q5_K" in
  if Block_schema.Logical_decode.code_bits equation <> 5 then
    Error "constructive Q5_K emitter requires five-bit codes"
  else
    let value_function = "llmopt_schema_q5_k_value" in
    let block_elements = Block_schema.Physical_layout.block_elements physical in
    let block_bytes = Block_schema.Physical_layout.block_bytes physical in
    let subblock_elements =
      Block_schema.Logical_decode.subblock_elements logical
    in
    Ok
      (emit_superblock ~m2_x4:true schema ~name:"q5_k" ~value_function
         (Printf.sprintf {|
inline float %s(device const uchar* block, uint lane, uint subblock) {
  const float d = float(*reinterpret_cast<device const half*>(block + %du));
  const float dmin = float(*reinterpret_cast<device const half*>(block + %du));
  device const uchar* table = block + %du;
  const uint j = subblock & 3u;
  const uint scale = subblock < 4u
      ? (table[j] & 63u)
      : ((table[j + 8u] & 15u) | ((table[j] >> 6u) << 4u));
  const uint minimum = subblock < 4u
      ? (table[j + 4u] & 63u)
      : ((table[j + 8u] >> 4u) | ((table[j + 4u] >> 6u) << 4u));
  const uchar packed = block[%du + (subblock >> 1u) * 32u + lane];
  const uint nibble = (subblock & 1u) == 0u ? (packed & 15u) : (packed >> 4u);
  const uint high = (block[%du + lane] >> subblock) & 1u;
  return d * float(scale) * float(nibble | (high << 4u))
      - dmin * float(minimum);
}
|} value_function scale_offset minimum_offset table_offset low_offset high_offset)
      ^ emit_superblock_gated_m2 ~name:"q5_k" ~value_function
          ~block_elements ~block_bytes ~subblock_elements)

let q6_k_linear ~target schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* zero = offset_zero (Block_schema.Logical_decode.code_signedness equation) in
  let* low_offset = field_offset physical Block_schema.Physical_layout.Field.Quantized_low in
  let* high_offset = field_offset physical Block_schema.Physical_layout.Field.Quantized_high in
  let* table_offset = field_offset physical Block_schema.Physical_layout.Field.Scale_table in
  let* scale_offset = field_offset physical Block_schema.Physical_layout.Field.Block_scale in
  let* () = require_superblock ~target physical logical "Q6_K" in
  if Block_schema.Logical_decode.code_bits equation <> 6 then
    Error "constructive Q6_K emitter requires six-bit codes"
  else
    let value_function = "llmopt_schema_q6_k_value" in
    let block_elements = Block_schema.Physical_layout.block_elements physical in
    let block_bytes = Block_schema.Physical_layout.block_bytes physical in
    let subblock_elements =
      Block_schema.Logical_decode.subblock_elements logical
    in
    Ok
      (emit_superblock ~m2_x4:true schema ~name:"q6_k" ~value_function
         (Printf.sprintf {|
inline float %s(device const uchar* block, uint lane, uint subblock) {
  const uint half_block = subblock >> 2u;
  const uint within = subblock & 3u;
  const uint low_index = half_block * 64u + (within & 1u) * 32u + lane;
  const uint low_shift = (within >> 1u) * 4u;
  const uint high_index = half_block * 32u + lane;
  const uint high_shift = within * 2u;
  const uint code = ((block[%du + low_index] >> low_shift) & 15u)
      | (((block[%du + high_index] >> high_shift) & 3u) << 4u);
  const uint scale_index = half_block * 8u + (lane >> 4u) + within * 2u;
  const char subscale = *reinterpret_cast<device const char*>(block + %du + scale_index);
  const float scale = float(*reinterpret_cast<device const half*>(block + %du));
  return scale * float(subscale) * float(int(code) - %d);
}
|} value_function low_offset high_offset table_offset scale_offset zero)
      ^ emit_superblock_gated_m2 ~name:"q6_k" ~value_function
          ~block_elements ~block_bytes ~subblock_elements)

let iq4_xs_linear ~target schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* scale_offset = field_offset physical Block_schema.Physical_layout.Field.Block_scale in
  let* high_offset = field_offset physical Block_schema.Physical_layout.Field.Codebook_scale_high in
  let* low_offset = field_offset physical Block_schema.Physical_layout.Field.Codebook_scale_low in
  let* quant_offset = field_offset physical Block_schema.Physical_layout.Field.Quantized in
  let* () = require_superblock ~target physical logical "IQ4_XS" in
  match equation, Block_schema.Logical_decode.scale equation with
  | Block_schema.Logical_decode.Codebook { values; code_bits = 4; _ },
    Block_schema.Logical_decode.Product
      [ Block_schema.Logical_decode.Per_block _;
        Block_schema.Logical_decode.Packed_power_of_two { bias; denominator; _ } ] ->
      let values = values |> List.map string_of_int |> String.concat ", " in
      let value_function = "llmopt_schema_iq4_xs_value" in
      let block_elements = Block_schema.Physical_layout.block_elements physical in
      let block_bytes = Block_schema.Physical_layout.block_bytes physical in
      let subblock_elements =
        Block_schema.Logical_decode.subblock_elements logical
      in
      Ok
        (emit_superblock ~m2_x4:false schema ~name:"iq4_xs" ~value_function
           (Printf.sprintf {|
constant int llmopt_schema_iq4_xs_values[16] = { %s };
inline float %s(device const uchar* block, uint lane, uint subblock) {
  const float d = float(*reinterpret_cast<device const half*>(block + %du));
  const ushort high = *reinterpret_cast<device const ushort*>(block + %du);
  const uchar low_byte = block[%du + (subblock >> 1u)];
  const uint low = (low_byte >> (4u * (subblock & 1u))) & 15u;
  const uint high_bits = (high >> (2u * subblock)) & 3u;
  const int exponent = int(low | (high_bits << 4u)) - %d;
  const uchar packed = block[%du + (subblock >> 1u) * 32u + lane];
  const uint code = (subblock & 1u) == 0u ? (packed & 15u) : (packed >> 4u);
  return d * exp2(float(exponent) / float(%d))
      * float(llmopt_schema_iq4_xs_values[code]);
}
|} values value_function scale_offset high_offset low_offset bias quant_offset
              denominator)
        ^ emit_superblock_gated_m2 ~name:"iq4_xs" ~value_function
            ~block_elements ~block_bytes ~subblock_elements)
  | _ -> Error "constructive IQ4_XS emitter requires the codebook power-of-two schema"

let q8_0_linear ~target schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* scale_offset =
    field_offset physical Block_schema.Physical_layout.Field.Block_scale
  in
  let* quant_offset =
    field_offset physical Block_schema.Physical_layout.Field.Quantized
  in
  let* () = require_block32 ~target physical "Q8_0" in
  if Block_schema.quant_type schema <> Some Ir.Dtype.Q8_0 then
    Error "constructive Q8_0 emitter received another storage schema"
  else if Block_schema.Logical_decode.code_bits equation <> 8
       || Block_schema.Logical_decode.code_signedness equation
          <> Block_schema.Physical_layout.Signedness.Signed
  then Error "constructive Q8_0 emitter requires signed eight-bit codes"
  else
    let value_function = "llmopt_schema_q8_0_value" in
    Ok
      (Printf.sprintf {|
inline float %s(device const uchar* block, uint lane) {
  const half scale = *reinterpret_cast<device const half*>(block + %du);
  const char code = *reinterpret_cast<device const char*>(block + %du + lane);
  return float(scale) * float(code);
}
|} value_function scale_offset quant_offset
      ^ emit_linear_pair ~name:"q8_0" ~value_function
          ~block_elements:(Block_schema.Physical_layout.block_elements physical)
          ~block_bytes:(Block_schema.Physical_layout.block_bytes physical))

let q5_0_linear ~target schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* zero = offset_zero (Block_schema.Logical_decode.code_signedness equation) in
  let* scale_offset =
    field_offset physical Block_schema.Physical_layout.Field.Block_scale
  in
  let* high_offset =
    field_offset physical Block_schema.Physical_layout.Field.Quantized_high
  in
  let* low_offset =
    field_offset physical Block_schema.Physical_layout.Field.Quantized_low
  in
  let* () = require_block32 ~target physical "Q5_0" in
  if Block_schema.quant_type schema <> Some Ir.Dtype.Q5_0 then
    Error "constructive Q5_0 emitter received another storage schema"
  else if Block_schema.Logical_decode.code_bits equation <> 5 then
    Error "constructive Q5_0 emitter requires five-bit codes"
  else
    let value_function = "llmopt_schema_q5_0_value" in
    Ok
      (Printf.sprintf {|
inline float %s(device const uchar* block, uint lane) {
  const half scale = *reinterpret_cast<device const half*>(block + %du);
  const uchar high_byte = block[%du + (lane >> 3u)];
  const uint high_bit = (high_byte >> (lane & 7u)) & 1u;
  const uchar low_byte = block[%du + (lane & 15u)];
  const uint low_nibble = lane < 16u ? (low_byte & 15u) : (low_byte >> 4u);
  return float(scale) * float(int(low_nibble | (high_bit << 4u)) - %d);
}
|} value_function scale_offset high_offset low_offset zero
      ^ emit_linear_pair ~name:"q5_0" ~value_function
          ~block_elements:(Block_schema.Physical_layout.block_elements physical)
          ~block_bytes:(Block_schema.Physical_layout.block_bytes physical)
      ^ emit_block32_gated ~name:"q5_0" ~value_function
          ~block_elements:(Block_schema.Physical_layout.block_elements physical)
          ~block_bytes:(Block_schema.Physical_layout.block_bytes physical))


let q8_0_twin_schema_geometry ~label ~target schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* _scale_offset =
    field_offset physical Block_schema.Physical_layout.Field.Block_scale
  in
  let* _quant_offset =
    field_offset physical Block_schema.Physical_layout.Field.Quantized
  in
  let* () = require_block32 ~target physical "Q8_0" in
  if Block_schema.quant_type schema <> Some Ir.Dtype.Q8_0 then
    Error
      (Printf.sprintf
         "constructive Q8_0 %s twin emitter received another storage schema"
         label)
  else if Block_schema.Logical_decode.code_bits equation <> 8
       || Block_schema.Logical_decode.code_signedness equation
          <> Block_schema.Physical_layout.Signedness.Signed
  then Error "constructive Q8_0 twin emitter requires signed eight-bit codes"
  else
    let block_elements = Block_schema.Physical_layout.block_elements physical in
    let block_bytes = Block_schema.Physical_layout.block_bytes physical in
    Ok (block_elements, block_bytes)

let q8_0_linear_add_kernel ~name ~target schema =
  let* block_elements, block_bytes =
    q8_0_twin_schema_geometry ~label:"linear-add" ~target schema
  in
  Ok
    (emit_block32_linear_add ~name ~value_function:"llmopt_schema_q8_0_value"
       ~block_elements ~block_bytes)

let q8_0_m2_add_kernel ~name ~target schema =
  let* block_elements, block_bytes =
    q8_0_twin_schema_geometry ~label:"m2-add" ~target schema
  in
  Ok
    (emit_block32_m2_add ~name ~value_function:"llmopt_schema_q8_0_value"
       ~block_elements ~block_bytes)

(* G4P12-05-1: same block32 twin admissions for Q5_0 (five-bit offset codes;
   the schema-value decode reads the scale/high/low offsets directly, so the
   twin body again carries no offsets of its own). *)
let q5_0_twin_schema_geometry ~label ~target schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* _scale_offset =
    field_offset physical Block_schema.Physical_layout.Field.Block_scale
  in
  let* _high_offset =
    field_offset physical Block_schema.Physical_layout.Field.Quantized_high
  in
  let* _low_offset =
    field_offset physical Block_schema.Physical_layout.Field.Quantized_low
  in
  let* () = require_block32 ~target physical "Q5_0" in
  if Block_schema.quant_type schema <> Some Ir.Dtype.Q5_0 then
    Error
      (Printf.sprintf
         "constructive Q5_0 %s twin emitter received another storage schema"
         label)
  else if Block_schema.Logical_decode.code_bits equation <> 5 then
    Error "constructive Q5_0 twin emitter requires five-bit codes"
  else
    let block_elements = Block_schema.Physical_layout.block_elements physical in
    let block_bytes = Block_schema.Physical_layout.block_bytes physical in
    Ok (block_elements, block_bytes)

let q5_0_linear_add_kernel ~name ~target schema =
  let* block_elements, block_bytes =
    q5_0_twin_schema_geometry ~label:"linear-add" ~target schema
  in
  Ok
    (emit_block32_linear_add ~name ~value_function:"llmopt_schema_q5_0_value"
       ~block_elements ~block_bytes)

let q5_0_m2_add_kernel ~name ~target schema =
  let* block_elements, block_bytes =
    q5_0_twin_schema_geometry ~label:"m2-add" ~target schema
  in
  Ok
    (emit_block32_m2_add ~name ~value_function:"llmopt_schema_q5_0_value"
       ~block_elements ~block_bytes)

(* G4P12-05-1b: superblock twin schema admissions for Q4_K/Q5_K/Q6_K/IQ4_XS.
   The family value function reads the block (scale/min/table offsets resolved
   inside the value function from the schema that built it), so the geometry
   checks the superblock shape, the quant type and the code width, plus
   presence of the distinguishing physical fields. *)
let superblock_twin_geometry ~label ~target ~expected_quant ~code_bits schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* () = require_superblock ~target physical logical "twin" in
  if Block_schema.quant_type schema <> Some expected_quant then
    Error
      (Printf.sprintf
         "constructive superblock %s twin emitter received another storage schema"
         label)
  else if Block_schema.Logical_decode.code_bits equation <> code_bits then
    Error
      (Printf.sprintf
         "constructive %s twin emitter requires %d-bit codes" label code_bits)
  else
    let block_elements = Block_schema.Physical_layout.block_elements physical in
    let block_bytes = Block_schema.Physical_layout.block_bytes physical in
    let subblock_elements =
      Block_schema.Logical_decode.subblock_elements logical
    in
    Ok (block_elements, block_bytes, subblock_elements)

let q4_k_twin_schema_geometry ~label ~target schema =
  let physical = Block_schema.physical_layout schema in
  let* _quant_offset =
    field_offset physical Block_schema.Physical_layout.Field.Quantized
  in
  let* _table_offset =
    field_offset physical Block_schema.Physical_layout.Field.Scale_minimum_table
  in
  superblock_twin_geometry ~label ~target ~expected_quant:Ir.Dtype.Q4_K
    ~code_bits:4 schema

let q5_k_twin_schema_geometry ~label ~target schema =
  let physical = Block_schema.physical_layout schema in
  let* _high_offset =
    field_offset physical Block_schema.Physical_layout.Field.Quantized_high
  in
  let* _low_offset =
    field_offset physical Block_schema.Physical_layout.Field.Quantized_low
  in
  superblock_twin_geometry ~label ~target ~expected_quant:Ir.Dtype.Q5_K
    ~code_bits:5 schema

let q6_k_twin_schema_geometry ~label ~target schema =
  let physical = Block_schema.physical_layout schema in
  let* _high_offset =
    field_offset physical Block_schema.Physical_layout.Field.Quantized_high
  in
  let* _low_offset =
    field_offset physical Block_schema.Physical_layout.Field.Quantized_low
  in
  let* _scale_table =
    field_offset physical Block_schema.Physical_layout.Field.Scale_table
  in
  superblock_twin_geometry ~label ~target ~expected_quant:Ir.Dtype.Q6_K
    ~code_bits:6 schema

let iq4_xs_twin_schema_geometry ~label ~target schema =
  let physical = Block_schema.physical_layout schema in
  let* _high_offset =
    field_offset physical Block_schema.Physical_layout.Field.Codebook_scale_high
  in
  let* _low_offset =
    field_offset physical Block_schema.Physical_layout.Field.Codebook_scale_low
  in
  superblock_twin_geometry ~label ~target ~expected_quant:Ir.Dtype.IQ4_XS
    ~code_bits:4 schema

let superblock_linear_add ~name ~value_function ~geometry schema =
  let* block_elements, block_bytes, subblock_elements = geometry schema in
  Ok
    (emit_superblock_linear_add ~name ~value_function ~block_elements
       ~block_bytes ~subblock_elements)

let q4_k_linear_add_kernel ~name ~target schema =
  let* body =
    superblock_linear_add ~name ~value_function:"llmopt_schema_q4_k_value"
      ~geometry:(q4_k_twin_schema_geometry ~label:"linear-add" ~target) schema
  in
  Ok body

let q5_k_linear_add_kernel ~name ~target schema =
  let* body =
    superblock_linear_add ~name ~value_function:"llmopt_schema_q5_k_value"
      ~geometry:(q5_k_twin_schema_geometry ~label:"linear-add" ~target) schema
  in
  Ok body

let q6_k_linear_add_kernel ~name ~target schema =
  let* body =
    superblock_linear_add ~name ~value_function:"llmopt_schema_q6_k_value"
      ~geometry:(q6_k_twin_schema_geometry ~label:"linear-add" ~target) schema
  in
  Ok body

let iq4_xs_linear_add_kernel ~name ~target schema =
  let* body =
    superblock_linear_add ~name ~value_function:"llmopt_schema_iq4_xs_value"
      ~geometry:(iq4_xs_twin_schema_geometry ~label:"linear-add" ~target)
      schema
  in
  Ok body

let iq4_xs_m2_add_kernel ~name ~target schema =
  let* block_elements, block_bytes, subblock_elements =
    iq4_xs_twin_schema_geometry ~label:"m2-add" ~target schema
  in
  Ok
    (emit_superblock_m2_add ~name ~value_function:"llmopt_schema_iq4_xs_value"
       ~block_elements ~block_bytes ~subblock_elements)

(* G4P12-05-2: on-demand _add twins for the parametric m2_x4 tile.  The flat
   m2_x4 kernel ships unconditionally in the K-family emission (emit_superblock
   ~m2_x4:true); the fold appends the twin here when a Linear-add m=2 base
   selects the m2_x4 tile, replacing the llama-shim patch. *)
let k_m2_x4_add_kernel ~name ~target ~value_function schema geometry =
  let* block_elements, block_bytes, subblock_elements = geometry schema in
  Ok
    (emit_superblock_m2_x4_add ~name ~value_function ~block_elements
       ~block_bytes ~subblock_elements)

let q4_k_m2_x4_add_kernel ~name ~target schema =
  k_m2_x4_add_kernel ~name ~target ~value_function:"llmopt_schema_q4_k_value"
    schema (q4_k_twin_schema_geometry ~label:"m2_x4-add" ~target)

let q5_k_m2_x4_add_kernel ~name ~target schema =
  k_m2_x4_add_kernel ~name ~target ~value_function:"llmopt_schema_q5_k_value"
    schema (q5_k_twin_schema_geometry ~label:"m2_x4-add" ~target)

let q6_k_m2_x4_add_kernel ~name ~target schema =
  k_m2_x4_add_kernel ~name ~target ~value_function:"llmopt_schema_q6_k_value"
    schema (q6_k_twin_schema_geometry ~label:"m2_x4-add" ~target)

