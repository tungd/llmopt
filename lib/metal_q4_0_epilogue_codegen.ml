open Metal_block_codegen_common

let ( let* ) = Result.bind

(** The ABI distinguishes the optional flat bias buffer from algebraic epilogue
    inputs, including an identity epilogue with no inputs. *)
module Writeback = struct
  type t = Bias | Epilogue of Epilogue.t

  let of_epilogue epilogue =
    if Epilogue.equal epilogue Epilogue.none then Bias else Epilogue epilogue

  let header ~name = function
    | Bias ->
        Printf.sprintf "\nkernel void %s(\n\
        \    device const half* input [[buffer(0)]],\n\
        \    device const uchar* weight [[buffer(1)]],\n\
        \    device const half* bias [[buffer(2)]],\n\
        \    device half* output [[buffer(3)]],\n\
        \    constant QuantLinearParams& params [[buffer(4)]]," name
    | Epilogue epilogue ->
        let count = List.length (Epilogue.inputs epilogue) in
        "kernel void " ^ name ^ "(\n"
        ^ "    device const half* input [[buffer(0)]],\n"
        ^ "    device const uchar* weight [[buffer(1)]]"
        ^ Epilogue.msl_buffer_decls ~start_index:2 epilogue
        ^ Printf.sprintf
            ",\n    device half* output [[buffer(%d)]],\n    constant QuantLinearParams& params [[buffer(%d)]],"
            (2 + count) (3 + count)

  let q4_0_single_row_epilogue_store ~columns epilogue =
    List.init columns (fun j ->
        let acc = Printf.sprintf "accumulator%d" j in
        let idx =
          if j = 0 then "base_col" else Printf.sprintf "(base_col + %du)" j
        in
        let expr = Epilogue.emit_msl_writeback ~acc ~idx epilogue in
        if j = 0 then Printf.sprintf "    output[base_col] = half(%s);" expr
        else
          Printf.sprintf
            "    if (base_col + %du < params.n) output[base_col + %du] = half(%s);"
            j j expr)
    |> String.concat "\n"

  let q4_0_row_block_epilogue_stores ~rows ~row_guard epilogue =
    List.init rows (fun r ->
        let acc = Printf.sprintf "acc[%d]" r in
        let idx = Printf.sprintf "out_idx%d" r in
        let expr = Epilogue.emit_msl_writeback ~acc ~idx epilogue in
        Printf.sprintf
          "    if (%s + %du < params.m) {\n      const uint %s = (%s + %du) * params.n + column;\n      output[%s] = half(%s);\n    }"
          row_guard r idx row_guard r idx expr)
    |> String.concat "\n"

  let single_row ?row ~columns writeback =
    let stores = match writeback with
      | Epilogue epilogue -> q4_0_single_row_epilogue_store ~columns epilogue
      | Bias ->
          let declaration, buffer = match row with
            | None -> "", "bias"
            | Some row ->
                Printf.sprintf "      device const half* bias_row = bias + %s * params.n;\n" row,
                "bias_row"
          in
          let bias = List.init columns (fun j ->
            if j = 0 then Printf.sprintf "      accumulator0 += float(%s[base_col]);" buffer
            else Printf.sprintf
              "      if (base_col + %du < params.n) accumulator%d += float(%s[base_col + %du]);" j j buffer j)
            |> String.concat "\n" in
          "    if (params.has_bias != 0u) {\n" ^ declaration ^ bias ^ "\n    }\n"
          ^ q4_0_single_row_epilogue_store ~columns Epilogue.none
    in
    "  if (lane == 0u && base_col < params.n) {\n" ^ stores ^ "\n  }"

  let row_block ~rows ~row_guard writeback =
    let stores = match writeback with
      | Epilogue epilogue -> q4_0_row_block_epilogue_stores ~rows ~row_guard epilogue
      | Bias -> Printf.sprintf {|    const float b_val = (params.has_bias != 0u) ? float(bias[column]) : 0.0f;
    #pragma unroll
    for (uint r = 0; r < %d; ++r) {
      if (%s + r < params.m) {
        output[(%s + r) * params.n + column] = half(acc[r] + b_val);
      }
    }|} rows row_guard row_guard
    in
    "  if (lane == 0u) {\n" ^ stores ^ "\n  }"
end

let q4_0_decode_epilogue_kernel ~label ~emit ~name ~epilogue ~target schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* zero = offset_zero (Block_schema.Logical_decode.code_signedness equation) in
  let* scale_offset =
    field_offset physical Block_schema.Physical_layout.Field.Block_scale
  in
  let* quant_offset =
    field_offset physical Block_schema.Physical_layout.Field.Quantized
  in
  let block_elements = Block_schema.Physical_layout.block_elements physical in
  let block_bytes = Block_schema.Physical_layout.block_bytes physical in
  let simd_lanes = target.Target_hardware.memory.simd_lanes in
  if Block_schema.quant_type schema <> Some Ir.Dtype.Q4_0 then
    Error
      (Printf.sprintf
         "constructive Q4_0 %s digest emitter received another storage schema"
         label)
  else if Block_schema.Logical_decode.code_bits equation <> 4 then
    Error
      (Printf.sprintf
         "constructive Q4_0 %s digest emitter requires four-bit codes" label)
  else if block_elements <> 32 || simd_lanes <> 32 then
    Error
      (Printf.sprintf
         "constructive Q4_0 %s digest emission requires block=32 and SIMD=32 \
          (got block=%d SIMD=%d)"
         label block_elements simd_lanes)
  else
    Ok (emit ~name ~writeback:(Writeback.Epilogue epilogue)
          ~scale_offset ~quant_offset ~zero ~block_elements ~block_bytes ())
