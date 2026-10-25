let ( let* ) = Result.bind

let max_rank = 8
let max_u32 = 0xffff_ffffL

let set_u32 bytes offset value =
  let encoded = Int64.of_int value in
  if value < 0 || Int64.compare encoded max_u32 > 0 then
    Error (Printf.sprintf "Metal parameter does not fit uint32: %d" value)
  else (
    Bytes.set_int32_le bytes offset (Int64.to_int32 encoded);
    Ok ())

let u32s values =
  let bytes = Bytes.make (4 * List.length values) '\000' in
  let rec write offset = function
    | [] -> Ok bytes
    | value :: rest ->
        let* () = set_u32 bytes offset value in
        write (offset + 4) rest
  in
  write 0 values

let rms_norm ~rows ~width ~epsilon =
  let bytes = Bytes.make 12 '\000' in
  let* () = set_u32 bytes 0 rows in
  let* () = set_u32 bytes 4 width in
  Bytes.set_int32_le bytes 8 (Int32.bits_of_float epsilon);
  Ok bytes

let l2_norm_slice ~input_rows ~groups ~width ~input_width ~offset ~epsilon =
  let bytes = Bytes.make 24 '\000' in
  let* encoded = u32s [ input_rows; groups; width; input_width; offset ] in
  Bytes.blit encoded 0 bytes 0 20;
  Bytes.set_int32_le bytes 20 (Int32.bits_of_float epsilon);
  Ok bytes

let w4a16_lm_head_argmax ~m ~n ~k ~epsilon =
  let bytes = Bytes.make 16 '\000' in
  let* encoded = u32s [ m; n; k ] in
  Bytes.blit encoded 0 bytes 0 12;
  Bytes.set_int32_le bytes 12 (Int32.bits_of_float epsilon);
  Ok bytes

let w4a16_swiglu_ffn ~m ~n ~k ~epsilon =
  let bytes = Bytes.make 16 '\000' in
  let* encoded = u32s [ m; n; k ] in
  Bytes.blit encoded 0 bytes 0 12;
  Bytes.set_int32_le bytes 12 (Int32.bits_of_float epsilon);
  Ok bytes

let rms_rope ~batches ~tokens ~heads ~width ~half_dimension ~trig_batches
    ~epsilon =
  let bytes = Bytes.make 28 '\000' in
  let* () =
    u32s [ batches; tokens; heads; width; half_dimension; trig_batches ]
    |> Result.map (fun encoded -> Bytes.blit encoded 0 bytes 0 24)
  in
  Bytes.set_int32_le bytes 24 (Int32.bits_of_float epsilon);
  Ok bytes

let rms_rope_qk ~batches ~tokens ~q_heads ~k_heads ~width ~half_dimension
    ~trig_batches ~epsilon =
  let bytes = Bytes.make 32 '\000' in
  let* () =
    u32s [ batches; tokens; q_heads; k_heads; width; half_dimension; trig_batches ]
    |> Result.map (fun encoded -> Bytes.blit encoded 0 bytes 0 28)
  in
  Bytes.set_int32_le bytes 28 (Int32.bits_of_float epsilon);
  Ok bytes

let rotary_qk ~batches ~tokens ~q_heads ~k_heads ~width ~half_dimension
    ~trig_batches =
  u32s
    [ batches; tokens; q_heads; k_heads; width; half_dimension; trig_batches ]

let arange ~count ~start ~step =
  let bytes = Bytes.make 24 '\000' in
  let* () = set_u32 bytes 0 count in
  Bytes.set_int64_le bytes 8 (Int64.of_int start);
  Bytes.set_int64_le bytes 16 (Int64.of_int step);
  Ok bytes

let fill ~count scalar =
  let bytes = Bytes.make 8 '\000' in
  let* () = set_u32 bytes 0 count in
  (match scalar with
  | Ir.Scalar.Bool value ->
      Bytes.set_int32_le bytes 4 (if value then 1l else 0l)
  | Ir.Scalar.Float value ->
      Bytes.set_int32_le bytes 4 (Int32.bits_of_float value)
  | Ir.Scalar.Int value -> Bytes.set_int32_le bytes 4 (Int32.of_int value));
  Ok bytes

let attention ~batches ~heads ~query_length ~key_length ~head_dimension
    ~mask_batches ~mask_heads ~causal ~scale ?(kv_heads = heads)
    ?(token_first_output = 0) ?(past_length = 0)
    ?(token_first_value = 0) ?(mask_dtype = 0) () =
  let bytes = Bytes.make 56 '\000' in
  let values =
    [ batches; heads; query_length; key_length; head_dimension; mask_batches;
      mask_heads; (if causal then 1 else 0) ]
  in
  let rec write offset = function
    | [] -> Ok ()
    | value :: rest ->
        let* () = set_u32 bytes offset value in
        write (offset + 4) rest
  in
  let* () = write 0 values in
  Bytes.set_int32_le bytes 32 (Int32.bits_of_float scale);
  let* () = set_u32 bytes 36 kv_heads in
  let* () = set_u32 bytes 40 token_first_output in
  let* () = set_u32 bytes 44 past_length in
  let* () = set_u32 bytes 48 token_first_value in
  let* () = set_u32 bytes 52 mask_dtype in
  Ok bytes

let paged_attention_q8 ~batches ~query_heads ~kv_heads ~past_length
    ~head_dimension ~mask_batches ~mask_heads ~token_elements ~key_offset
    ~value_offset ~group_size ~token_stride ~scale ~mask_dtype
    ?(query_length = 1) ?(window = 0) () =
  let bytes = Bytes.make 64 '\000' in
  let values =
    [ batches; query_heads; kv_heads; past_length; head_dimension; mask_batches;
      mask_heads; token_elements; key_offset; value_offset; group_size;
      token_stride ]
  in
  let rec write offset = function
    | [] -> Ok ()
    | value :: rest ->
        let* () = set_u32 bytes offset value in
        write (offset + 4) rest
  in
  let* () = write 0 values in
  Bytes.set_int32_le bytes 48 (Int32.bits_of_float scale);
  let* () = set_u32 bytes 52 query_length in
  let* () = set_u32 bytes 56 mask_dtype in
  let* () = set_u32 bytes 60 window in
  Ok bytes

let speculative_attention ~batches ~heads ~query_length ~key_length
    ~head_dimension ~past_length ~scale ?(kv_heads = heads)
    ?(tree_masks = [| 1; 3; 7; 15; 0; 0; 0; 0 |]) () =
  let bytes = Bytes.make 64 '\000' in
  let* () = set_u32 bytes 0 batches in
  let* () = set_u32 bytes 4 heads in
  let* () = set_u32 bytes 8 query_length in
  let* () = set_u32 bytes 12 key_length in
  let* () = set_u32 bytes 16 head_dimension in
  let* () = set_u32 bytes 20 kv_heads in
  let* () = set_u32 bytes 24 past_length in
  Bytes.set_int32_le bytes 28 (Int32.bits_of_float scale);
  for i = 0 to min 7 (Array.length tree_masks - 1) do
    Bytes.set_int32_le bytes (32 + i * 4) (Int32.of_int tree_masks.(i))
  done;
  Ok bytes

let scalar_i64 = function
  | Ir.Scalar.Bool value -> if value then 1L else 0L
  | Ir.Scalar.Int value -> Int64.of_int value
  | Ir.Scalar.Float value -> Int64.of_float value

let scalar_f32 scalar =
  scalar |> Ir.Scalar.to_float |> Int32.bits_of_float

let rec write_shape bytes offset = function
  | [] -> Ok ()
  | dimension :: rest ->
      let* () = set_u32 bytes offset dimension in
      write_shape bytes (offset + 4) rest

let dimensions ~operation shape =
  let dimensions = Tensor_shape.dimensions shape in
  if List.length dimensions > max_rank then
    Error
      (Printf.sprintf "Metal %s supports rank at most %d" operation max_rank)
  else Ok dimensions

let pad_shape dimensions =
  dimensions @ List.init (max_rank - List.length dimensions) (Fun.const 1)

let write_i64s bytes offset values =
  List.iteri
    (fun index value ->
      Bytes.set_int64_le bytes (offset + (8 * index)) (Int64.of_int value))
    values

let movement ~input_dimensions ~output_dimensions ~axis0 ~axis1 =
  let rank = List.length output_dimensions in
  let bytes = Bytes.make 80 '\000' in
  let* () = set_u32 bytes 0 (List.fold_left ( * ) 1 output_dimensions) in
  let* () = set_u32 bytes 4 rank in
  let* () = set_u32 bytes 8 axis0 in
  let* () = set_u32 bytes 12 axis1 in
  let* () = write_shape bytes 16 (pad_shape input_dimensions) in
  let* () = write_shape bytes 48 (pad_shape output_dimensions) in
  Ok bytes

let transpose ~input_shape ~output_shape ~axis0 ~axis1 =
  let* input_dimensions = dimensions ~operation:"transpose" input_shape in
  let* output_dimensions = dimensions ~operation:"transpose" output_shape in
  let* axis0 =
    Tensor_shape.normalize_axis input_shape axis0
    |> Result.map_error Tensor_shape.error_to_string
  in
  let* axis1 =
    Tensor_shape.normalize_axis input_shape axis1
    |> Result.map_error Tensor_shape.error_to_string
  in
  let* inferred =
    Tensor_shape.transpose input_shape ~axis0 ~axis1
    |> Result.map_error Tensor_shape.error_to_string
  in
  if not (Tensor_shape.equal inferred output_shape) then
    Error "Metal transpose output shape is inconsistent"
  else
    movement ~input_dimensions ~output_dimensions ~axis0 ~axis1

let expand ~input_shape ~output_shape =
  let* input_dimensions = dimensions ~operation:"expand" input_shape in
  let* output_dimensions = dimensions ~operation:"expand" output_shape in
  let* _ =
    Tensor_shape.expand input_shape ~target:output_shape
    |> Result.map_error Tensor_shape.error_to_string
  in
  let rank = List.length output_dimensions in
  let aligned_input =
    List.init (rank - List.length input_dimensions) (Fun.const 1)
    @ input_dimensions
  in
  movement ~input_dimensions:aligned_input ~output_dimensions ~axis0:0 ~axis1:0

let index_parameters ~count ~input_shape ~output_shape index =
  let* input_dimensions = dimensions ~operation:"index" input_shape in
  let* output_dimensions = dimensions ~operation:"index" output_shape in
  let selectors = Tensor_shape.Index.selectors index in
  if List.length selectors > max_rank then
    Error
      (Printf.sprintf "Metal index supports at most %d selectors" max_rank)
  else
    let selector_kind, starts, steps =
      List.fold_left
        (fun (kinds, starts, steps) selector ->
          match selector with
          | Tensor_shape.Index.At value ->
              0 :: kinds, value :: starts, 0 :: steps
          | Tensor_shape.Index.Slice { start; step; length = _ } ->
              1 :: kinds, start :: starts, step :: steps
          | Tensor_shape.Index.New_axis ->
              2 :: kinds, 0 :: starts, 0 :: steps)
        ([], [], []) selectors
    in
    let selector_kind = List.rev selector_kind in
    let starts = List.rev starts in
    let steps = List.rev steps in
    let pad values =
      values @ List.init (max_rank - List.length values) (Fun.const 0)
    in
    let bytes = Bytes.make 240 '\000' in
    let* () = set_u32 bytes 0 count in
    let* () = set_u32 bytes 4 (List.length input_dimensions) in
    let* () = set_u32 bytes 8 (List.length output_dimensions) in
    let* () = set_u32 bytes 12 (List.length selectors) in
    let* () = write_shape bytes 16 (pad_shape input_dimensions) in
    let* () = write_shape bytes 48 (pad_shape output_dimensions) in
    let* () = write_shape bytes 80 (pad selector_kind) in
    write_i64s bytes 112 (pad starts);
    write_i64s bytes 176 (pad steps);
    Ok bytes

let index ~input_shape ~output_shape index =
  index_parameters ~count:(Tensor_shape.numel output_shape) ~input_shape
    ~output_shape index

let update_slice ~destination_shape ~source_shape index =
  let* inferred =
    Tensor_shape.apply_index destination_shape index
    |> Result.map_error Tensor_shape.error_to_string
  in
  if not (Tensor_shape.equal inferred source_shape) then
    Error "Metal slice-update source shape is inconsistent"
  else
    index_parameters ~count:(Tensor_shape.numel destination_shape)
      ~input_shape:destination_shape ~output_shape:source_shape index

let concat ~left_shape ~right_shape ~output_shape ~axis =
  let* left_dimensions = dimensions ~operation:"concat" left_shape in
  let* _right_dimensions = dimensions ~operation:"concat" right_shape in
  let* output_dimensions = dimensions ~operation:"concat" output_shape in
  let* axis =
    Tensor_shape.normalize_axis output_shape axis
    |> Result.map_error Tensor_shape.error_to_string
  in
  let* inferred =
    Tensor_shape.concat [ left_shape; right_shape ] ~axis
    |> Result.map_error Tensor_shape.error_to_string
  in
  if not (Tensor_shape.equal inferred output_shape) then
    Error "Metal concat output shape is inconsistent"
  else
    let left_dimensions = Array.of_list left_dimensions in
    let bytes = Bytes.make 48 '\000' in
    let* () = set_u32 bytes 0 (Tensor_shape.numel output_shape) in
    let* () = set_u32 bytes 4 (List.length output_dimensions) in
    let* () = set_u32 bytes 8 axis in
    let* () = set_u32 bytes 12 left_dimensions.(axis) in
    let* () = write_shape bytes 16 (pad_shape output_dimensions) in
    Ok bytes

let roll ~shape ~axis ~shift =
  let* shape_dimensions = dimensions ~operation:"roll" shape in
  let* axis =
    Tensor_shape.normalize_axis shape axis
    |> Result.map_error Tensor_shape.error_to_string
  in
  let encoded_shift = Int64.of_int shift in
  if
    Int64.compare encoded_shift (Int64.of_int32 Int32.min_int) < 0
    || Int64.compare encoded_shift (Int64.of_int32 Int32.max_int) > 0
  then Error "Metal roll shift does not fit int32"
  else
    let bytes = Bytes.make 48 '\000' in
    let* () = set_u32 bytes 0 (Tensor_shape.numel shape) in
    let* () = set_u32 bytes 4 (List.length shape_dimensions) in
    let* () = set_u32 bytes 8 axis in
    Bytes.set_int32_le bytes 12 (Int64.to_int32 encoded_shift);
    let* () = write_shape bytes 16 (pad_shape shape_dimensions) in
    Ok bytes

let align_broadcast_shape ~output_dimensions = function
  | None -> Ok (List.init (List.length output_dimensions) (Fun.const 1))
  | Some input_dimensions ->
      let output_rank = List.length output_dimensions in
      let input_rank = List.length input_dimensions in
      if input_rank > output_rank then
        Error "pointwise input rank exceeds output rank"
      else
        let aligned =
          List.init (output_rank - input_rank) (Fun.const 1)
          @ input_dimensions
        in
        if
          List.for_all2
            (fun input output -> input = 1 || input = output)
            aligned output_dimensions
        then Ok aligned
        else Error "pointwise input shape is not broadcastable to output"

let pointwise ~output_shape ~left_shape ~right_shape ~left_scalar
    ~right_scalar =
  let output_dimensions = Tensor_shape.dimensions output_shape in
  let rank = List.length output_dimensions in
  if rank > 8 then Error "Metal pointwise kernels support rank at most eight"
  else
    let* left_dimensions =
      align_broadcast_shape ~output_dimensions left_shape
    in
    let* right_dimensions =
      align_broadcast_shape ~output_dimensions right_shape
    in
    let pad dimensions = dimensions @ List.init (8 - rank) (Fun.const 1) in
    let bytes = Bytes.make 136 '\000' in
    let count = Tensor_shape.numel output_shape in
    let* () = set_u32 bytes 0 count in
    let* () = set_u32 bytes 4 rank in
    let* () = set_u32 bytes 8 (if Option.is_some left_scalar then 1 else 0) in
    let* () = set_u32 bytes 12 (if Option.is_some right_scalar then 1 else 0) in
    let* () = write_shape bytes 16 (pad output_dimensions) in
    let* () = write_shape bytes 48 (pad left_dimensions) in
    let* () = write_shape bytes 80 (pad right_dimensions) in
    Option.iter
      (fun scalar ->
        Bytes.set_int64_le bytes 112 (scalar_i64 scalar);
        Bytes.set_int32_le bytes 128 (scalar_f32 scalar))
      left_scalar;
    Option.iter
      (fun scalar ->
        Bytes.set_int64_le bytes 120 (scalar_i64 scalar);
        Bytes.set_int32_le bytes 132 (scalar_f32 scalar))
      right_scalar;
    Ok bytes

let masked_fill ~output_shape ~input_shape ~mask_shape scalar =
  let* bytes =
    pointwise ~output_shape
      ~left_shape:(Some (Tensor_shape.dimensions input_shape))
      ~right_shape:(Some (Tensor_shape.dimensions mask_shape))
      ~left_scalar:None ~right_scalar:None
  in
  Bytes.set_int32_le bytes 132 (scalar_f32 scalar);
  Ok bytes

let batched_matmul ~lhs_shape ~rhs_shape ~output_shape =
  let lhs = Tensor_shape.dimensions lhs_shape in
  let rhs = Tensor_shape.dimensions rhs_shape in
  let output = Tensor_shape.dimensions output_shape in
  let split_matrix dimensions =
    match List.rev dimensions with
    | last :: penultimate :: leading ->
        Ok (List.rev leading, penultimate, last)
    | _ -> Error "batched matmul inputs must have rank at least two"
  in
  let* lhs_batch, m, k = split_matrix lhs in
  let* rhs_batch, rhs_k, n = split_matrix rhs in
  let* output_batch, output_m, output_n = split_matrix output in
  let rank = List.length output_batch in
  if rank > 3 then Error "Metal batched matmul supports at most three batch axes"
  else if k <> rhs_k || m <> output_m || n <> output_n then
    Error "batched matmul matrix dimensions are inconsistent"
  else
    let align dimensions =
      List.init (rank - List.length dimensions) (Fun.const 1) @ dimensions
    in
    let lhs_batch = align lhs_batch in
    let rhs_batch = align rhs_batch in
    if
      not
        (List.for_all2
           (fun dimension output -> dimension = 1 || dimension = output)
           lhs_batch output_batch
        && List.for_all2
             (fun dimension output -> dimension = 1 || dimension = output)
             rhs_batch output_batch)
    then Error "batched matmul batch dimensions do not broadcast"
    else
      let pad dimensions = dimensions @ List.init (3 - rank) (Fun.const 1) in
      let values =
        [ Tensor_shape.numel output_shape; m; n; k; rank ]
        @ pad output_batch @ pad lhs_batch @ pad rhs_batch
      in
      u32s values

let moe_topk ~tokens ~num_experts ~top_k ~norm_topk_prob () =
  u32s [ tokens; num_experts; top_k; (if norm_topk_prob then 1 else 0) ]

let moe_gate_up ~tokens ~top_k ~hidden_dim ~intermediate_dim () =
  u32s [ tokens; top_k; hidden_dim; intermediate_dim ]

let moe_down ~tokens ~top_k ~hidden_dim ~intermediate_dim () =
  u32s [ tokens; top_k; hidden_dim; intermediate_dim ]
