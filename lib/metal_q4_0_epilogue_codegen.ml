open Metal_block_codegen_common

let ( let* ) = Result.bind

let find_substring_from haystack needle start =
  let hlen = String.length haystack in
  let nlen = String.length needle in
  let rec go i =
    if i + nlen > hlen then None
    else if String.sub haystack i nlen = needle then Some i
    else go (i + 1)
  in
  if nlen = 0 then None else go (max start 0)

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

let q4_0_single_row_digest_of_flat ~columns ~label ~name epilogue kernel =
  (* The tail of the plain kernel's argument list (the thread-position
     attributes through the closing ") {") is shared verbatim by its digest
     variant; everything before it is re-emitted with the canonicalized
     epilogue buffers inserted between the weight and the output workspace. *)
  let attr_marker = "    uint gid [[thread_position_in_grid]]," in
  match find_substring_from kernel attr_marker 0 with
  | None ->
      Error
        (Printf.sprintf "Q4_0 %s digest emission: decode kernel thread \
                         attributes are absent"
           label)
  | Some attr_start -> (
      match find_substring_from kernel ") {" attr_start with
      | None ->
          Error
            (Printf.sprintf "Q4_0 %s digest emission: decode kernel header is \
                             unterminated"
               label)
      | Some close ->
          let header_stop = close + 3 in
          let attributes =
            String.sub kernel attr_start (header_stop - attr_start)
          in
          let body =
            String.sub kernel header_stop (String.length kernel - header_stop)
          in
          let buffer_count = List.length (Epilogue.inputs epilogue) in
          let header =
            "kernel void " ^ name ^ "(\n"
            ^ "    device const half* input [[buffer(0)]],\n"
            ^ "    device const uchar* weight [[buffer(1)]]"
            ^ Epilogue.msl_buffer_decls ~start_index:2 epilogue
            ^ Printf.sprintf
                ",\n    device half* output [[buffer(%d)]],\n    constant \
                 QuantLinearParams& params [[buffer(%d)]],\n"
                (2 + buffer_count) (3 + buffer_count)
            ^ attributes
          in
          let tail_marker = "  if (lane == 0u && base_col < params.n) {" in
          let full = header ^ body in
          match find_substring_from full tail_marker 0 with
          | None ->
              Error
                (Printf.sprintf "Q4_0 %s digest emission: writeback tail is \
                                 absent from the decode kernel"
                   label)
          | Some tail_start ->
              let head = String.sub full 0 tail_start in
              Ok
                (head ^ tail_marker ^ "\n"
                ^ q4_0_single_row_epilogue_store ~columns epilogue
                ^ "\n  }\n}\n"))

let q4_0_x16_digest_of_flat ~name epilogue kernel =
  q4_0_single_row_digest_of_flat ~columns:16 ~label:"x16" ~name epilogue kernel

let q4_0_x8_digest_of_flat ~name epilogue kernel =
  q4_0_single_row_digest_of_flat ~columns:8 ~label:"x8" ~name epilogue kernel

let q4_0_x4_digest_of_flat ~name epilogue kernel =
  q4_0_single_row_digest_of_flat ~columns:4 ~label:"x4" ~name epilogue kernel

(* Row-block digest variants (m16/m32): these kernels run one simdgroup per
   (row tile x column) and keep a [float acc[rows]] array, so the digest
   writeback replaces the flat store loop with one guarded store per
   accumulator row instead of a single-row column span.  The accumulator
   slot name is the flat kernel's array ([acc[r]]); the row-guard variable
   is the flat kernel's base-row register ([row_base] for m16,
   [sub_row_base] for m32).  The digest header layout matches the
   single-row variant exactly: epilogue buffers at buffer(2..2+p), the
   output workspace at buffer(2+p) and the params at buffer(3+p). *)

let q4_0_row_block_epilogue_stores ~rows ~row_guard epilogue =
  List.init rows (fun r ->
      let acc = Printf.sprintf "acc[%d]" r in
      let idx = Printf.sprintf "out_idx%d" r in
      let expr = Epilogue.emit_msl_writeback ~acc ~idx epilogue in
      Printf.sprintf
        "    if (%s + %du < params.m) {\n      const uint %s = (%s + %du) * params.n + column;\n      output[%s] = half(%s);\n    }"
        row_guard r idx row_guard r idx expr)
  |> String.concat "\n"

let q4_0_row_block_digest_of_flat ~rows ~row_guard ~label ~name epilogue
    kernel =
  let attr_marker = "    uint gid [[thread_position_in_grid]]," in
  match find_substring_from kernel attr_marker 0 with
  | None ->
      Error
        (Printf.sprintf "Q4_0 %s digest emission: decode kernel thread \
                         attributes are absent"
           label)
  | Some attr_start -> (
      match find_substring_from kernel ") {" attr_start with
      | None ->
          Error
            (Printf.sprintf "Q4_0 %s digest emission: decode kernel header is \
                             unterminated"
               label)
      | Some close ->
          let header_stop = close + 3 in
          let attributes =
            String.sub kernel attr_start (header_stop - attr_start)
          in
          let body =
            String.sub kernel header_stop (String.length kernel - header_stop)
          in
          let buffer_count = List.length (Epilogue.inputs epilogue) in
          let header =
            "kernel void " ^ name ^ "(\n"
            ^ "    device const half* input [[buffer(0)]],\n"
            ^ "    device const uchar* weight [[buffer(1)]]"
            ^ Epilogue.msl_buffer_decls ~start_index:2 epilogue
            ^ Printf.sprintf
                ",\n    device half* output [[buffer(%d)]],\n    constant \
                 QuantLinearParams& params [[buffer(%d)]],\n"
                (2 + buffer_count) (3 + buffer_count)
            ^ attributes
          in
          let tail_marker = "  if (lane == 0u) {" in
          let full = header ^ body in
          match find_substring_from full tail_marker 0 with
          | None ->
              Error
                (Printf.sprintf "Q4_0 %s digest emission: writeback tail is \
                                 absent from the decode kernel"
                   label)
          | Some tail_start ->
              let head = String.sub full 0 tail_start in
              Ok
                (head ^ tail_marker ^ "\n"
                ^ q4_0_row_block_epilogue_stores ~rows ~row_guard epilogue
                ^ "\n  }\n}\n"))

let q4_0_m16_digest_of_flat ~name epilogue kernel =
  q4_0_row_block_digest_of_flat ~rows:16 ~row_guard:"row_base" ~label:"m16"
    ~name epilogue kernel

(* The m32 tile runs two simdgroups per 32-row tile; each simdgroup
   [simd_in_tile] covers a 16-row half through [sub_row_base], so its
   writeback is structurally the same sixteen guarded stores as m16 but
   guarded against [sub_row_base]. *)

let q4_0_m32_digest_of_flat ~name epilogue kernel =
  q4_0_row_block_digest_of_flat ~rows:16 ~row_guard:"sub_row_base" ~label:"m32"
    ~name epilogue kernel

let q4_0_decode_epilogue_kernel ?(rows = 0) ?(row_guard = "") ~columns ~label
    ~emit_flat ~name ~epilogue ~target schema =
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
    let flat =
      emit_flat ~scale_offset ~quant_offset ~zero ~block_elements ~block_bytes
    in
    let digest_result =
      if rows = 0 then
        q4_0_single_row_digest_of_flat ~columns ~label ~name epilogue flat
      else
        q4_0_row_block_digest_of_flat ~rows ~row_guard ~label ~name epilogue
          flat
    in
    match digest_result with
    | Ok text -> Ok text
    | Error message ->
        Error
          (Printf.sprintf "constructive Q4_0 %s digest emission: %s" label message)
