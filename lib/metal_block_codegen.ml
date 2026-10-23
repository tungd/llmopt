(* Metal_block_codegen: public facade for constructive block-quantized
   Linear code generation. Dispatches to format-specific modules:
   - Metal_q4_0_codegen (Q4_0 linear/qkv/epilogue/multi/add)
   - Metal_block_codegen_formats (non-Q4_0 formats and add variants)
   - Parametric Tile32_Affine (B in {4, 8}) unified affine contraction codegen
*)

let ( let* ) = Result.bind

let tile32_affine_geometry ~target schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* scale_offset =
    Metal_block_codegen_common.field_offset physical
      Block_schema.Physical_layout.Field.Block_scale
  in
  let* minimum_offset =
    Metal_block_codegen_common.field_offset physical
      Block_schema.Physical_layout.Field.Block_minimum
  in
  let* quant_offset =
    Metal_block_codegen_common.field_offset physical
      Block_schema.Physical_layout.Field.Quantized
  in
  let* () =
    Metal_block_codegen_common.require_block32 ~target physical "Tile32_Affine"
  in
  match equation with
  | Block_schema.Logical_decode.Affine
      {
        code_bits = (4 | 8) as code_bits;
        code_signedness = Block_schema.Physical_layout.Signedness.Unsigned;
        _;
      } ->
      let block_elements =
        Block_schema.Physical_layout.block_elements physical
      in
      let block_bytes = Block_schema.Physical_layout.block_bytes physical in
      Ok
        ( code_bits,
          scale_offset,
          minimum_offset,
          quant_offset,
          block_elements,
          block_bytes )
  | _ ->
      Error
        "constructive Tile32_Affine emitter requires unsigned 4-bit or 8-bit \
         affine block=32 schema"

let emit_tile32_affine_cluster_loop ~code_bits ~scale_offset ~minimum_offset
    ~quant_offset ~block_elements ~block_bytes ~in_ptr ~acc_body =
  if code_bits = 4 then
    Printf.sprintf
      {|  const uint cluster_blocks = (blocks + 7u) >> 3u;
  const uint b_sub = lane >> 2u;
  const uint t_in_b = lane & 3u;
  const uint byte_offset = %du + (t_in_b << 2u);

  for (uint c = 0u; c < cluster_blocks; ++c) {
    const uint block_idx = (c << 3u) + b_sub;
    if (block_idx < blocks) {
      const uint blk_byte_off = block_idx * %du;
      const uint l0 = t_in_b << 2u;
      const uint k_idx = block_idx * %du + l0;
      const float a0_lo = float(%s[k_idx]);
      const float a0_hi = float(%s[k_idx + 16u]);
      const float a1_lo = float(%s[k_idx + 1u]);
      const float a1_hi = float(%s[k_idx + 17u]);
      const float a2_lo = float(%s[k_idx + 2u]);
      const float a2_hi = float(%s[k_idx + 18u]);
      const float a3_lo = float(%s[k_idx + 3u]);
      const float a3_hi = float(%s[k_idx + 19u]);
      const float a_sum = a0_lo + a0_hi + a1_lo + a1_hi + a2_lo + a2_hi + a3_lo + a3_hi;
%s
    }
  }|}
      quant_offset block_bytes block_elements in_ptr in_ptr in_ptr in_ptr in_ptr
      in_ptr in_ptr in_ptr
      (acc_body ~scale_offset ~minimum_offset ~code_bits)
  else
    Printf.sprintf
      {|  const uint cluster_blocks = (blocks + 3u) >> 2u;
  const uint b_sub = lane >> 3u;
  const uint t_in_b = lane & 7u;
  const uint byte_offset = %du + (t_in_b << 2u);

  for (uint c = 0u; c < cluster_blocks; ++c) {
    const uint block_idx = (c << 2u) + b_sub;
    if (block_idx < blocks) {
      const uint blk_byte_off = block_idx * %du;
      const uint l0 = t_in_b << 2u;
      const uint k_idx = block_idx * %du + l0;
      const float a0 = float(%s[k_idx]);
      const float a1 = float(%s[k_idx + 1u]);
      const float a2 = float(%s[k_idx + 2u]);
      const float a3 = float(%s[k_idx + 3u]);
      const float a_sum = a0 + a1 + a2 + a3;
%s
    }
  }|}
      quant_offset block_bytes block_elements in_ptr in_ptr in_ptr in_ptr
      (acc_body ~scale_offset ~minimum_offset ~code_bits)

let emit_tile32_affine_slot_acc ~row_var ~acc_var ~scale_offset ~minimum_offset
    ~code_bits =
  if code_bits = 4 then
    Printf.sprintf
      {|      {
        device const uchar* blk = %s + blk_byte_off;
        const float scale = float(*reinterpret_cast<device const half*>(blk + %du));
        const float minimum = float(*reinterpret_cast<device const half*>(blk + %du));
        const ushort p_lo = *reinterpret_cast<device const ushort*>(blk + byte_offset);
        const ushort p_hi = *reinterpret_cast<device const ushort*>(blk + byte_offset + 2u);
        %s += scale * (
            a0_lo * float(p_lo & 15u) +
            a0_hi * float((p_lo >> 4u) & 15u) +
            a1_lo * float((p_lo >> 8u) & 15u) +
            a1_hi * float((p_lo >> 12u) & 15u) +
            a2_lo * float(p_hi & 15u) +
            a2_hi * float((p_hi >> 4u) & 15u) +
            a3_lo * float((p_hi >> 8u) & 15u) +
            a3_hi * float((p_hi >> 12u) & 15u)) + minimum * a_sum;
      }|}
      row_var scale_offset minimum_offset acc_var
  else
    Printf.sprintf
      {|      {
        device const uchar* blk = %s + blk_byte_off;
        const float scale = float(*reinterpret_cast<device const half*>(blk + %du));
        const float minimum = float(*reinterpret_cast<device const half*>(blk + %du));
        const uchar4 q4 = *reinterpret_cast<device const uchar4*>(blk + byte_offset);
        %s += scale * (
            a0 * float(q4.x) +
            a1 * float(q4.y) +
            a2 * float(q4.z) +
            a3 * float(q4.w)) + minimum * a_sum;
      }|}
      row_var scale_offset minimum_offset acc_var

let tile32_affine_linear_f16_x8_kernel ?name ?(epilogue = Epilogue.none) ~target
    schema =
  let* ( code_bits,
         scale_offset,
         minimum_offset,
         quant_offset,
         block_elements,
         block_bytes ) =
    tile32_affine_geometry ~target schema
  in
  let default_name =
    Printf.sprintf "llmopt_tile32_affine%d_linear_f16_x8" code_bits
  in
  let kernel_name = Option.value name ~default:default_name in
  let acc_body ~scale_offset ~minimum_offset ~code_bits =
    List.init 8 (fun j ->
        emit_tile32_affine_slot_acc
          ~row_var:(Printf.sprintf "row_blocks%d" j)
          ~acc_var:(Printf.sprintf "accumulator%d" j)
          ~scale_offset ~minimum_offset ~code_bits)
    |> String.concat "\n"
  in
  let cluster_loop =
    emit_tile32_affine_cluster_loop ~code_bits ~scale_offset ~minimum_offset
      ~quant_offset ~block_elements ~block_bytes ~in_ptr:"activation" ~acc_body
  in
  let flat =
    Printf.sprintf
      {|
kernel void %s(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_in_tg = tid >> 5u;
  const uint tg_idx = (gid - tid) >> 8u;
  const uint col_group = tg_idx * 8u + simd_in_tg;
  const uint base_col = col_group << 3u;
  if (base_col >= params.n) return;
  const uint blocks = params.k / %du;
  const uint col_stride = blocks * %du;
  device const uchar* row_blocks0 = weight + min(base_col, params.n - 1u) * col_stride;
  device const uchar* row_blocks1 = weight + min(base_col + 1u, params.n - 1u) * col_stride;
  device const uchar* row_blocks2 = weight + min(base_col + 2u, params.n - 1u) * col_stride;
  device const uchar* row_blocks3 = weight + min(base_col + 3u, params.n - 1u) * col_stride;
  device const uchar* row_blocks4 = weight + min(base_col + 4u, params.n - 1u) * col_stride;
  device const uchar* row_blocks5 = weight + min(base_col + 5u, params.n - 1u) * col_stride;
  device const uchar* row_blocks6 = weight + min(base_col + 6u, params.n - 1u) * col_stride;
  device const uchar* row_blocks7 = weight + min(base_col + 7u, params.n - 1u) * col_stride;
  device const half* activation = input;
  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;
  float accumulator2 = 0.0f;
  float accumulator3 = 0.0f;
  float accumulator4 = 0.0f;
  float accumulator5 = 0.0f;
  float accumulator6 = 0.0f;
  float accumulator7 = 0.0f;

%s
  accumulator0 = simd_sum(accumulator0);
  accumulator1 = simd_sum(accumulator1);
  accumulator2 = simd_sum(accumulator2);
  accumulator3 = simd_sum(accumulator3);
  accumulator4 = simd_sum(accumulator4);
  accumulator5 = simd_sum(accumulator5);
  accumulator6 = simd_sum(accumulator6);
  accumulator7 = simd_sum(accumulator7);

  if (lane == 0u && base_col < params.n) {
    if (params.has_bias != 0u) {
      accumulator0 += float(bias[base_col]);
      if (base_col + 1u < params.n) accumulator1 += float(bias[base_col + 1u]);
      if (base_col + 2u < params.n) accumulator2 += float(bias[base_col + 2u]);
      if (base_col + 3u < params.n) accumulator3 += float(bias[base_col + 3u]);
      if (base_col + 4u < params.n) accumulator4 += float(bias[base_col + 4u]);
      if (base_col + 5u < params.n) accumulator5 += float(bias[base_col + 5u]);
      if (base_col + 6u < params.n) accumulator6 += float(bias[base_col + 6u]);
      if (base_col + 7u < params.n) accumulator7 += float(bias[base_col + 7u]);
    }
    output[base_col] = half(accumulator0);
    if (base_col + 1u < params.n) output[base_col + 1u] = half(accumulator1);
    if (base_col + 2u < params.n) output[base_col + 2u] = half(accumulator2);
    if (base_col + 3u < params.n) output[base_col + 3u] = half(accumulator3);
    if (base_col + 4u < params.n) output[base_col + 4u] = half(accumulator4);
    if (base_col + 5u < params.n) output[base_col + 5u] = half(accumulator5);
    if (base_col + 6u < params.n) output[base_col + 6u] = half(accumulator6);
    if (base_col + 7u < params.n) output[base_col + 7u] = half(accumulator7);
  }
}
|}
      kernel_name block_elements block_bytes cluster_loop
  in
  if Epilogue.equal epilogue Epilogue.none then Ok flat
  else Metal_q4_0_epilogue_codegen.q4_0_x8_digest_of_flat ~name:kernel_name epilogue flat

let tile32_affine_multi_linear_f16_x8_kernel ~name ~epilogue ?(num_slots = 2)
    ~target schema =
  let* ( code_bits,
         scale_offset,
         minimum_offset,
         quant_offset,
         block_elements,
         block_bytes ) =
    tile32_affine_geometry ~target schema
  in
  let canonical_epilogue = Applicative_expr.canonicalize epilogue in
  let aux_values = Applicative_expr.required_values canonical_epilogue in
  let num_aux = List.length aux_values in
  let buf_idx_out = 1 + num_slots + num_aux in
  let buf_idx_params = 2 + num_slots + num_aux in
  let weight_buffers =
    List.init num_slots (fun s ->
        Printf.sprintf "    device const uchar* weight_%d [[buffer(%d)]]," s
          (1 + s))
    |> String.concat "\n"
  in
  let aux_buffers =
    if num_aux = 0 then ""
    else
      aux_values
      |> List.mapi (fun k v ->
             let dtype_str =
               match Ir.Value.dtype v with
               | Ir.Dtype.Float16 -> "half"
               | _ -> "float"
             in
             Printf.sprintf "    device const %s* buf_%d [[buffer(%d)]],\n"
               dtype_str k (1 + num_slots + k))
      |> String.concat ""
  in
  let weight_bases =
    List.init num_slots (fun s ->
        let rows =
          List.init 8 (fun j ->
              if j = 0 then
                Printf.sprintf
                  "  device const uchar* row_%d_0 = base_%d + base_col * col_stride;"
                  s s
              else
                Printf.sprintf
                  "  device const uchar* row_%d_%d = base_%d + min(base_col + %du, params.n - 1u) * col_stride;"
                  s j s j)
          |> String.concat "\n"
        in
        Printf.sprintf "  device const uchar* base_%d = weight_%d;\n%s" s s rows)
    |> String.concat "\n\n"
  in
  let accumulators =
    List.init num_slots (fun s ->
        Printf.sprintf
          "  float acc%d_0 = 0.0f, acc%d_1 = 0.0f, acc%d_2 = 0.0f, acc%d_3 = 0.0f;\n\
          \  float acc%d_4 = 0.0f, acc%d_5 = 0.0f, acc%d_6 = 0.0f, acc%d_7 = 0.0f;"
          s s s s s s s s)
    |> String.concat "\n"
  in
  let acc_body ~scale_offset ~minimum_offset ~code_bits =
    List.init num_slots (fun s ->
        List.init 8 (fun j ->
            emit_tile32_affine_slot_acc
              ~row_var:(Printf.sprintf "row_%d_%d" s j)
              ~acc_var:(Printf.sprintf "acc%d_%d" s j)
              ~scale_offset ~minimum_offset ~code_bits)
        |> String.concat "\n")
    |> String.concat "\n\n"
  in
  let cluster_loop =
    emit_tile32_affine_cluster_loop ~code_bits ~scale_offset ~minimum_offset
      ~quant_offset ~block_elements ~block_bytes ~in_ptr:"in_ptr" ~acc_body
  in
  let simd_sums =
    List.init num_slots (fun s ->
        List.init 8 (fun j ->
            Printf.sprintf "  acc%d_%d = simd_sum(acc%d_%d);" s j s j)
        |> String.concat "\n")
    |> String.concat "\n\n"
  in
  let in_bufs =
    if num_aux = 0 then ""
    else
      aux_values
      |> List.mapi (fun k v ->
             let dtype_str =
               match Ir.Value.dtype v with
               | Ir.Dtype.Float16 -> "half"
               | _ -> "float"
             in
             Printf.sprintf
               "    device const %s* in_buf_%d = buf_%d + row * params.n;\n"
               dtype_str k k)
      |> String.concat ""
  in
  let value_var v =
    Printf.sprintf "in_buf_%d" (Ir.Value_id.to_int (Ir.Value.id v))
  in
  let writebacks =
    List.init 8 (fun j ->
        let idx =
          if j = 0 then "base_col" else Printf.sprintf "(base_col + %du)" j
        in
        let expr =
          Applicative_expr.emit_msl_expr ~value_var
            ~slot_emitter:(fun s -> Printf.sprintf "acc%d_%d" s j)
            ~index_expr:idx canonical_epilogue
        in
        if j = 0 then Printf.sprintf "    out_row[base_col] = half(%s);" expr
        else
          Printf.sprintf
            "    if (base_col + %du < params.n) out_row[base_col + %du] = half(%s);"
            j j expr)
    |> String.concat "\n"
  in
  Ok
    (Printf.sprintf
       {|
kernel void %s(
    device const half* input [[buffer(0)]],
%s
%s    device half* output [[buffer(%d)]],
    constant QuantLinearParams& params [[buffer(%d)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint col_groups = (params.n + 7u) >> 3u;
  const uint total_elements = params.m * col_groups;
  if (simd_idx >= total_elements) return;

  const uint row = simd_idx / col_groups;
  const uint group_idx = simd_idx - row * col_groups;
  const uint base_col = group_idx << 3u;

  device const half* in_ptr = input + row * params.k;

  const uint blocks = params.k / %du;
  const uint col_stride = blocks * %du;
%s

%s

%s

%s

  if (lane == 0u) {
    device half* out_row = output + row * params.n;
%s%s
  }
}
|}
       name weight_buffers aux_buffers buf_idx_out buf_idx_params
       block_elements block_bytes weight_bases accumulators cluster_loop
       simd_sums in_bufs writebacks)

let tile32_affine_multi_export_linear_f16_x8_kernel ~name ~epilogues
    ?(num_slots = 2) ~target schema =
  let* ( code_bits,
         scale_offset,
         minimum_offset,
         quant_offset,
         block_elements,
         block_bytes ) =
    tile32_affine_geometry ~target schema
  in
  let num_exports = List.length epilogues in
  let canonical_epilogues = List.map Applicative_expr.canonicalize epilogues in
  let aux_values =
    List.concat_map Applicative_expr.required_values canonical_epilogues
    |> List.fold_left
         (fun acc v ->
           if List.exists (fun v' -> Ir.Value.equal v v') acc then acc
           else acc @ [ v ])
         []
  in
  let num_aux = List.length aux_values in
  let first_output_buf = 1 + num_slots + num_aux in
  let buf_idx_params = first_output_buf + num_exports in
  let weight_buffers =
    List.init num_slots (fun s ->
        Printf.sprintf "    device const uchar* weight_%d [[buffer(%d)]]," s
          (1 + s))
    |> String.concat "\n"
  in
  let aux_buffers =
    if num_aux = 0 then ""
    else
      aux_values
      |> List.mapi (fun k v ->
             let dtype_str =
               match Ir.Value.dtype v with
               | Ir.Dtype.Float16 -> "half"
               | _ -> "float"
             in
             Printf.sprintf "    device const %s* buf_%d [[buffer(%d)]],\n"
               dtype_str k (1 + num_slots + k))
      |> String.concat ""
  in
  let output_buffers =
    List.init num_exports (fun e ->
        Printf.sprintf "    device half* output_%d [[buffer(%d)]]," e
          (first_output_buf + e))
    |> String.concat "\n"
  in
  let weight_bases =
    List.init num_slots (fun s ->
        let rows =
          List.init 8 (fun j ->
              if j = 0 then
                Printf.sprintf
                  "  device const uchar* row_%d_0 = base_%d + base_col * col_stride;"
                  s s
              else
                Printf.sprintf
                  "  device const uchar* row_%d_%d = base_%d + min(base_col + %du, params.n - 1u) * col_stride;"
                  s j s j)
          |> String.concat "\n"
        in
        Printf.sprintf "  device const uchar* base_%d = weight_%d;\n%s" s s rows)
    |> String.concat "\n\n"
  in
  let accumulators =
    List.init num_slots (fun s ->
        Printf.sprintf
          "  float acc%d_0 = 0.0f, acc%d_1 = 0.0f, acc%d_2 = 0.0f, acc%d_3 = 0.0f;\n\
          \  float acc%d_4 = 0.0f, acc%d_5 = 0.0f, acc%d_6 = 0.0f, acc%d_7 = 0.0f;"
          s s s s s s s s)
    |> String.concat "\n"
  in
  let acc_body ~scale_offset ~minimum_offset ~code_bits =
    List.init num_slots (fun s ->
        List.init 8 (fun j ->
            emit_tile32_affine_slot_acc
              ~row_var:(Printf.sprintf "row_%d_%d" s j)
              ~acc_var:(Printf.sprintf "acc%d_%d" s j)
              ~scale_offset ~minimum_offset ~code_bits)
        |> String.concat "\n")
    |> String.concat "\n\n"
  in
  let cluster_loop =
    emit_tile32_affine_cluster_loop ~code_bits ~scale_offset ~minimum_offset
      ~quant_offset ~block_elements ~block_bytes ~in_ptr:"in_ptr" ~acc_body
  in
  let simd_sums =
    List.init num_slots (fun s ->
        List.init 8 (fun j ->
            Printf.sprintf "  acc%d_%d = simd_sum(acc%d_%d);" s j s j)
        |> String.concat "\n")
    |> String.concat "\n\n"
  in
  let in_bufs =
    if num_aux = 0 then ""
    else
      aux_values
      |> List.mapi (fun k v ->
             let dtype_str =
               match Ir.Value.dtype v with
               | Ir.Dtype.Float16 -> "half"
               | _ -> "float"
             in
             Printf.sprintf
               "    device const %s* in_buf_%d = buf_%d + row * params.n;\n"
               dtype_str k k)
      |> String.concat ""
  in
  let value_var v =
    Printf.sprintf "in_buf_%d" (Ir.Value_id.to_int (Ir.Value.id v))
  in
  let writebacks =
    canonical_epilogues
    |> List.mapi (fun e epil ->
           let row_decl =
             Printf.sprintf
               "    device half* out_row_%d = output_%d + row * params.n;" e e
           in
           let col_writes =
             List.init 8 (fun j ->
                 let idx =
                   if j = 0 then "base_col"
                   else Printf.sprintf "(base_col + %du)" j
                 in
                 let expr =
                   Applicative_expr.emit_msl_expr ~value_var
                     ~slot_emitter:(fun s -> Printf.sprintf "acc%d_%d" s j)
                     ~index_expr:idx epil
                 in
                 if j = 0 then
                   Printf.sprintf "    out_row_%d[base_col] = half(%s);" e expr
                 else
                   Printf.sprintf
                     "    if (base_col + %du < params.n) out_row_%d[base_col + %du] = half(%s);"
                     j e j expr)
             |> String.concat "\n"
           in
           row_decl ^ "\n" ^ col_writes)
    |> String.concat "\n"
  in
  Ok
    (Printf.sprintf
       {|
kernel void %s(
    device const half* input [[buffer(0)]],
%s
%s%s
    constant QuantLinearParams& params [[buffer(%d)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint col_groups = (params.n + 7u) >> 3u;
  const uint total_elements = params.m * col_groups;
  if (simd_idx >= total_elements) return;

  const uint row = simd_idx / col_groups;
  const uint group_idx = simd_idx - row * col_groups;
  const uint base_col = group_idx << 3u;

  device const half* in_ptr = input + row * params.k;

  const uint blocks = params.k / %du;
  const uint col_stride = blocks * %du;
%s

%s

%s

%s

  if (lane == 0u) {
%s%s
  }
}
|}
       name weight_buffers aux_buffers output_buffers buf_idx_params
       block_elements block_bytes weight_bases accumulators cluster_loop
       simd_sums in_bufs writebacks)

let tile32_affine_linear ~target schema =
  let* ( code_bits,
         scale_offset,
         minimum_offset,
         quant_offset,
         block_elements,
         block_bytes ) =
    tile32_affine_geometry ~target schema
  in
  let name = Printf.sprintf "tile32_affine%d" code_bits in
  let value_function =
    Printf.sprintf "llmopt_schema_tile32_affine%d_value" code_bits
  in
  let value_fn_src =
    if code_bits = 4 then
      Printf.sprintf
        {|
inline float %s(device const uchar* block, uint lane) {
  const float scale = float(*reinterpret_cast<device const half*>(block + %du));
  const float minimum = float(*reinterpret_cast<device const half*>(block + %du));
  const uchar packed = block[%du + (lane & 15u)];
  const uint code = lane < 16u ? (packed & 15u) : (packed >> 4u);
  return scale * float(code) + minimum;
}
|}
        value_function scale_offset minimum_offset quant_offset
    else
      Printf.sprintf
        {|
inline float %s(device const uchar* block, uint lane) {
  const float scale = float(*reinterpret_cast<device const half*>(block + %du));
  const float minimum = float(*reinterpret_cast<device const half*>(block + %du));
  const uint code = uint(block[%du + lane]);
  return scale * float(code) + minimum;
}
|}
        value_function scale_offset minimum_offset quant_offset
  in
  let* x8_src = tile32_affine_linear_f16_x8_kernel ~target schema in
  Ok
    (value_fn_src
    ^ Metal_block_codegen_common.emit_linear_pair ~name ~value_function
        ~block_elements ~block_bytes
    ^ x8_src)

let tile32_affine_linear_add_kernel ~name ~target schema =
  let* ( code_bits,
         _scale_offset,
         _minimum_offset,
         _quant_offset,
         block_elements,
         block_bytes ) =
    tile32_affine_geometry ~target schema
  in
  let value_function =
    Printf.sprintf "llmopt_schema_tile32_affine%d_value" code_bits
  in
  Ok
    (Metal_block_codegen_common.emit_block32_linear_add ~name ~value_function
       ~block_elements ~block_bytes)

let tile32_affine_m2_add_kernel ~name ~target schema =
  let* ( code_bits,
         _scale_offset,
         _minimum_offset,
         _quant_offset,
         block_elements,
         block_bytes ) =
    tile32_affine_geometry ~target schema
  in
  let value_function =
    Printf.sprintf "llmopt_schema_tile32_affine%d_value" code_bits
  in
  Ok
    (Metal_block_codegen_common.emit_block32_m2_add ~name ~value_function
       ~block_elements ~block_bytes)

let linear_kernels ~target schema =
  match Block_schema.quant_type schema with
  | Some (Ir.Dtype.Tile32_Affine4 | Ir.Dtype.Tile32_Affine8) ->
      tile32_affine_linear ~target schema
  | Some Ir.Dtype.Q4_0 -> Metal_q4_0_codegen.q4_0_linear ~target schema
  | Some
      ( Ir.Dtype.Q8_0
      | Ir.Dtype.Q5_0
      | Ir.Dtype.Q4_K
      | Ir.Dtype.Q5_K
      | Ir.Dtype.Q6_K
      | Ir.Dtype.IQ4_XS ) ->
      Error
        "legacy GGUF format must be transcoded to Tile32_Affine{4,8} before Metal codegen"
  | None ->
      (match
         Block_schema.Logical_decode.equation
           (Block_schema.logical_decode schema)
       with
      | Block_schema.Logical_decode.Affine _ ->
          tile32_affine_linear ~target schema
      | _ ->
          Error
            "constructive Metal Linear emission requires block-quant storage")

let qkv_kernels ~target schema =
  match Block_schema.quant_type schema with
  | Some Ir.Dtype.Q4_0 -> Metal_q4_0_codegen.q4_0_qkv ~target schema
  | _ -> Error "constructive QKV emission only implemented for Q4_0"

let q4_0_x16_epilogue_kernel = Metal_q4_0_codegen.q4_0_x16_epilogue_kernel
let q4_0_x8_epilogue_kernel = Metal_q4_0_codegen.q4_0_x8_epilogue_kernel
let q4_0_x4_epilogue_kernel = Metal_q4_0_codegen.q4_0_x4_epilogue_kernel
let q4_0_m16_epilogue_kernel = Metal_q4_0_codegen.q4_0_m16_epilogue_kernel
let q4_0_m32_epilogue_kernel = Metal_q4_0_codegen.q4_0_m32_epilogue_kernel
let q4_0_multi_linear_f16_x8_kernel = Metal_q4_0_codegen.q4_0_multi_linear_f16_x8_kernel
let q4_0_multi_linear_f16_x16_kernel = Metal_q4_0_codegen.q4_0_multi_linear_f16_x16_kernel
let q4_0_multi_export_linear_f16_x8_kernel = Metal_q4_0_codegen.q4_0_multi_export_linear_f16_x8_kernel
let q4_0_multi_export_linear_f16_x16_kernel = Metal_q4_0_codegen.q4_0_multi_export_linear_f16_x16_kernel
let q4_0_linear_add_kernel = Metal_q4_0_codegen.q4_0_linear_add_kernel
let q4_0_m2_add_kernel = Metal_q4_0_codegen.q4_0_m2_add_kernel

