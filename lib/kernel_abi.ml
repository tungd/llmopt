module Operation = struct
  type t =
    | Matmul
    | Fused_linear
    | Linear
    | W4a16_linear
    | W4a16_lm_head_argmax
    | Rms_norm
    | Rms_rope
    | Short_conv
    | Short_conv_step
    | Short_conv_prefill
    | Attention
    | Embedding
    | Arange
    | Diff
    | Cumsum
    | Fill
    | Gather2
    | Cast
    | Pointwise
    | Movement
    | Reduction
    | Update_slice
    | Cache

  let to_string = function
    | Matmul -> "matmul"
    | Fused_linear -> "fused-linear"
    | Linear -> "linear"
    | W4a16_linear -> "w4a16-linear-g64"
    | W4a16_lm_head_argmax -> "w4a16-lm-head-argmax"
    | Rms_norm -> "rms-norm"
    | Rms_rope -> "rms-rope"
    | Short_conv -> "short-conv"
    | Short_conv_step -> "short-conv-step"
    | Short_conv_prefill -> "short-conv-prefill"
    | Attention -> "attention"
    | Embedding -> "embedding"
    | Arange -> "arange"
    | Diff -> "diff"
    | Cumsum -> "cumsum"
    | Fill -> "fill"
    | Gather2 -> "gather2"
    | Cast -> "cast"
    | Pointwise -> "pointwise"
    | Movement -> "movement"
    | Reduction -> "reduction"
    | Update_slice -> "update-slice"
    | Cache -> "cache"

  let of_string = function
    | "matmul" -> Ok Matmul
    | "fused-linear" -> Ok Fused_linear
    | "linear" -> Ok Linear
    | "w4a16-linear-g64" -> Ok W4a16_linear
    | "w4a16-lm-head-argmax" -> Ok W4a16_lm_head_argmax
    | "rms-norm" -> Ok Rms_norm
    | "rms-rope" -> Ok Rms_rope
    | "short-conv" -> Ok Short_conv
    | "short-conv-step" -> Ok Short_conv_step
    | "short-conv-prefill" -> Ok Short_conv_prefill
    | "attention" -> Ok Attention
    | "embedding" -> Ok Embedding
    | "arange" -> Ok Arange
    | "diff" -> Ok Diff
    | "cumsum" -> Ok Cumsum
    | "fill" -> Ok Fill
    | "gather2" -> Ok Gather2
    | "cast" -> Ok Cast
    | "pointwise" -> Ok Pointwise
    | "movement" -> Ok Movement
    | "reduction" -> Ok Reduction
    | "update-slice" -> Ok Update_slice
    | "cache" -> Ok Cache
    | value -> Error ("unsupported kernel operation: " ^ value)
end

module Entry = struct
  type tile = int * int * int

  type t = {
    name : string;
    operation : Operation.t;
    input_dtype : Ir.Dtype.t;
    output_dtype : Ir.Dtype.t;
    threadgroup : int * int * int;
    tile : tile option;
  }

  let make ~tile ~name ~operation ~input_dtype ~output_dtype
      ~threadgroup:((x, y, z) as threadgroup) =
    let valid_tile =
      match tile with
      | None -> true
      | Some (tile_m, tile_n, tile_k) ->
          tile_m > 0 && tile_n > 0 && tile_k > 0 && tile_k mod 4 = 0
    in
    if String.trim name = "" then Error "kernel entry-point name cannot be empty"
    else if x <= 0 || y <= 0 || z <= 0 then
      Error "kernel threadgroup dimensions must be positive"
    else if not valid_tile then
      Error "kernel tile dimensions must be positive and tile_k must be divisible by four"
    else Ok { name; operation; input_dtype; output_dtype; threadgroup; tile }

  let create ~name ~operation ~input_dtype ~output_dtype ~threadgroup =
    make ~tile:None ~name ~operation ~input_dtype ~output_dtype ~threadgroup

  let create_with_tile ~tile ~name ~operation ~input_dtype ~output_dtype
      ~threadgroup =
    make ~tile:(Some tile) ~name ~operation ~input_dtype ~output_dtype
      ~threadgroup

  let name entry = entry.name
  let operation entry = entry.operation
  let input_dtype entry = entry.input_dtype
  let output_dtype entry = entry.output_dtype
  let threadgroup entry = entry.threadgroup
  let tile entry = entry.tile

end
