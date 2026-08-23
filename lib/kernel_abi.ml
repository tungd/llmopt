open Yojson.Basic.Util

module Operation = struct
  type t = Matmul | Fused_linear | Linear | Q8_linear | Q8_dequantize

  let to_string = function
    | Matmul -> "matmul"
    | Fused_linear -> "fused-linear"
    | Linear -> "linear"
    | Q8_linear -> "q8-linear"
    | Q8_dequantize -> "q8-dequantize"

  let of_string = function
    | "matmul" -> Ok Matmul
    | "fused-linear" -> Ok Fused_linear
    | "linear" -> Ok Linear
    | "q8-linear" -> Ok Q8_linear
    | "q8-dequantize" -> Ok Q8_dequantize
    | value -> Error ("unsupported kernel operation: " ^ value)
end

let dtype_of_string = function
  | "f32" -> Ok Ir.Dtype.Float32
  | "f16" -> Ok Ir.Dtype.Float16
  | "bf16" -> Ok Ir.Dtype.Bfloat16
  | "i64" -> Ok Ir.Dtype.Int64
  | "i32" -> Ok Ir.Dtype.Int32
  | "i8" -> Ok Ir.Dtype.Int8
  | "bool" -> Ok Ir.Dtype.Bool
  | value -> Error ("unsupported kernel dtype: " ^ value)

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

  let to_yojson entry =
    let x, y, z = entry.threadgroup in
    `Assoc
      [ ("name", `String entry.name);
        ("operation", `String (Operation.to_string entry.operation));
        ("input_dtype", `String (Ir.Dtype.to_string entry.input_dtype));
        ("output_dtype", `String (Ir.Dtype.to_string entry.output_dtype));
        ("threadgroup", `List [ `Int x; `Int y; `Int z ]) ]

  let of_yojson json =
    try
      let name = json |> member "name" |> to_string in
      let operation = json |> member "operation" |> to_string |> Operation.of_string in
      let input_dtype =
        json |> member "input_dtype" |> to_string |> dtype_of_string
      in
      let output_dtype =
        json |> member "output_dtype" |> to_string |> dtype_of_string
      in
      let threadgroup =
        match json |> member "threadgroup" |> to_list with
        | [ `Int x; `Int y; `Int z ] -> Ok (x, y, z)
        | _ -> Error "kernel threadgroup must contain three integers"
      in
      match operation, input_dtype, output_dtype, threadgroup with
      | Ok operation, Ok input_dtype, Ok output_dtype, Ok threadgroup ->
          create ~name ~operation ~input_dtype ~output_dtype ~threadgroup
      | Error message, _, _, _
      | _, Error message, _, _
      | _, _, Error message, _
      | _, _, _, Error message -> Error message
    with
    | Type_error (message, _) -> Error ("invalid kernel entry: " ^ message)
end
