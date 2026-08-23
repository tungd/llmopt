module Operation = struct
  type t =
    | Matmul
    | Fused_linear
    | Linear
    | Q8_linear
    | Q8_dequantize
    | Rms_norm
    | Short_conv
    | Attention
    | Embedding

  let to_string = function
    | Matmul -> "matmul"
    | Fused_linear -> "fused-linear"
    | Linear -> "linear"
    | Q8_linear -> "q8-linear"
    | Q8_dequantize -> "q8-dequantize"
    | Rms_norm -> "rms-norm"
    | Short_conv -> "short-conv"
    | Attention -> "attention"
    | Embedding -> "embedding"

  let of_string = function
    | "matmul" -> Ok Matmul
    | "fused-linear" -> Ok Fused_linear
    | "linear" -> Ok Linear
    | "q8-linear" -> Ok Q8_linear
    | "q8-dequantize" -> Ok Q8_dequantize
    | "rms-norm" -> Ok Rms_norm
    | "short-conv" -> Ok Short_conv
    | "attention" -> Ok Attention
    | "embedding" -> Ok Embedding
    | value -> Error ("unsupported kernel operation: " ^ value)
end

module Entry = struct
  type t = {
    name : string;
    operation : Operation.t;
    input_dtype : Ir.Dtype.t;
    output_dtype : Ir.Dtype.t;
    threadgroup : int * int * int;
  }

  let create ~name ~operation ~input_dtype ~output_dtype
      ~threadgroup:((x, y, z) as threadgroup) =
    if String.trim name = "" then Error "kernel entry-point name cannot be empty"
    else if x <= 0 || y <= 0 || z <= 0 then
      Error "kernel threadgroup dimensions must be positive"
    else Ok { name; operation; input_dtype; output_dtype; threadgroup }

  let name entry = entry.name
  let operation entry = entry.operation
  let input_dtype entry = entry.input_dtype
  let output_dtype entry = entry.output_dtype
  let threadgroup entry = entry.threadgroup

end
