module Operation = struct
  type t =
    | Matmul
    | Fused_linear
    | Linear
    | Q8_linear
    | Q8_linear_silu
    | Q8_linear_add
    | Q8_linear_mul_add
    | Q8_dequantize
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
    | Q8_linear -> "q8-linear"
    | Q8_linear_silu -> "q8-linear-silu"
    | Q8_linear_add -> "q8-linear-add"
    | Q8_linear_mul_add -> "q8-linear-mul-add"
    | Q8_dequantize -> "q8-dequantize"
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
    | "q8-linear" -> Ok Q8_linear
    | "q8-linear-silu" -> Ok Q8_linear_silu
    | "q8-linear-add" -> Ok Q8_linear_add
    | "q8-linear-mul-add" -> Ok Q8_linear_mul_add
    | "q8-dequantize" -> Ok Q8_dequantize
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
