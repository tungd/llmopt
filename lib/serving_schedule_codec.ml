(** Serving schedule binary serialization ([to_bytes]/[of_bytes]).
    Depends only on [Serving_schedule_core] and lower domains; the
    duplicated write/read_kernel_index definitions are preserved
    verbatim (see LOOP item 3). *)

open Serving_schedule_core

let dtype_tag = function
  | Ir.Dtype.Float32 -> 0
  | Ir.Dtype.Float16 -> 1
  | Ir.Dtype.Bfloat16 -> 2
  | Ir.Dtype.Int64 -> 3
  | Ir.Dtype.Int32 -> 4
  | Ir.Dtype.Int8 -> 5
  | Ir.Dtype.Bool -> 6
  | Ir.Dtype.UInt8 -> 7
  | Ir.Dtype.Quant Q8_0 -> 8
  | Ir.Dtype.Quant Q4_K -> 9
  | Ir.Dtype.Quant Q5_K -> 10
  | Ir.Dtype.Quant Q6_K -> 11
  | Ir.Dtype.Quant Q5_0 -> 12
  | Ir.Dtype.Quant Q4_0 -> 13
  | Ir.Dtype.Quant IQ4_XS -> 14

let dtype_of_tag = function
  | 0 -> Ok Ir.Dtype.Float32
  | 1 -> Ok Ir.Dtype.Float16
  | 2 -> Ok Ir.Dtype.Bfloat16
  | 3 -> Ok Ir.Dtype.Int64
  | 4 -> Ok Ir.Dtype.Int32
  | 5 -> Ok Ir.Dtype.Int8
  | 6 -> Ok Ir.Dtype.Bool
  | 7 -> Ok Ir.Dtype.UInt8
  | 8 -> Ok (Ir.Dtype.Quant Q8_0)
  | 9 -> Ok (Ir.Dtype.Quant Q4_K)
  | 10 -> Ok (Ir.Dtype.Quant Q5_K)
  | 11 -> Ok (Ir.Dtype.Quant Q6_K)
  | 12 -> Ok (Ir.Dtype.Quant Q5_0)
  | 13 -> Ok (Ir.Dtype.Quant Q4_0)
  | 14 -> Ok (Ir.Dtype.Quant IQ4_XS)
  | tag -> Error (Printf.sprintf "unknown schedule dtype tag: %d" tag)

let write_shape writer shape =
  let dimensions = Tensor_shape.dimensions shape in
  Binary.Writer.u16 writer (List.length dimensions);
  List.iter (Binary.Writer.u64 writer) dimensions

let read_shape reader =
  let* rank = Binary.Reader.u16 reader in
  let rec dimensions acc remaining =
    if remaining = 0 then
      (match Tensor_shape.create (List.rev acc) with
      | Ok shape -> Ok shape
      | Error error -> Error (Tensor_shape.error_to_string error))
    else
      let* dimension = Binary.Reader.u64 reader in
      dimensions (dimension :: acc) (remaining - 1)
  in
  dimensions [] rank

let write_value writer value =
  Binary.Writer.u32 writer (value_id value);
  Binary.Writer.u8 writer (dtype_tag (Ir.Value.dtype value));
  write_shape writer (Ir.Value.logical_shape value)

let read_value reader =
  let* id = Binary.Reader.u32 reader in
  let* dtype_tag = Binary.Reader.u8 reader in
  let* dtype = dtype_of_tag dtype_tag in
  let* shape = read_shape reader in
  try Ok (Ir.Value.make_tensor ~id ~shape ~dtype)
  with Invalid_argument message -> Error ("invalid schedule value: " ^ message)

let write_memory_space writer = function
  | Ir.Memory_space.Global -> Binary.Writer.u8 writer 0
  | Ir.Memory_space.Shared -> Binary.Writer.u8 writer 1
  | Ir.Memory_space.Register -> Binary.Writer.u8 writer 2
  | Ir.Memory_space.Private -> Binary.Writer.u8 writer 3

let read_memory_space reader =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 -> Ok Ir.Memory_space.Global
  | 1 -> Ok Ir.Memory_space.Shared
  | 2 -> Ok Ir.Memory_space.Register
  | 3 -> Ok Ir.Memory_space.Private
  | tag -> Error (Printf.sprintf "unknown schedule memory-space tag: %d" tag)

let write_layout writer = function
  | Ir.Layout.Row_major -> Binary.Writer.u8 writer 0
  | Ir.Layout.Col_major -> Binary.Writer.u8 writer 1
  | Ir.Layout.Xor_swizzle mask ->
      Binary.Writer.u8 writer 2;
      Binary.Writer.i64 writer mask

let read_layout reader =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 -> Ok Ir.Layout.Row_major
  | 1 -> Ok Ir.Layout.Col_major
  | 2 -> Binary.Reader.i64 reader |> Result.map (fun mask -> Ir.Layout.Xor_swizzle mask)
  | _ -> Error (Printf.sprintf "unknown schedule layout tag: %d" tag)

let write_option writer write = function
  | None -> Binary.Writer.u8 writer 0
  | Some value ->
      Binary.Writer.u8 writer 1;
      write writer value

let read_option reader read =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 -> Ok None
  | 1 -> read reader |> Result.map Option.some
  | _ -> Error (Printf.sprintf "unknown schedule option tag: %d" tag)

let rec write_argument writer = function
  | Ir.Argument.Value value ->
      Binary.Writer.u8 writer 0;
      Binary.Writer.u32 writer (value_id value)
  | Ir.Argument.Null -> Binary.Writer.u8 writer 1
  | Ir.Argument.Ellipsis -> Binary.Writer.u8 writer 11
  | Ir.Argument.Bool value ->
      Binary.Writer.u8 writer 2;
      Binary.Writer.bool writer value
  | Ir.Argument.Int value ->
      Binary.Writer.u8 writer 3;
      Binary.Writer.i64 writer value
  | Ir.Argument.Float value ->
      Binary.Writer.u8 writer 4;
      Binary.Writer.float64 writer value
  | Ir.Argument.String value ->
      Binary.Writer.u8 writer 5;
      Binary.Writer.string writer value
  | Ir.Argument.Symbol value ->
      Binary.Writer.u8 writer 6;
      Binary.Writer.string writer value
  | Ir.Argument.List values ->
      Binary.Writer.u8 writer 7;
      write_arguments writer values
  | Ir.Argument.Tuple values ->
      Binary.Writer.u8 writer 8;
      write_arguments writer values
  | Ir.Argument.Mapping fields ->
      Binary.Writer.u8 writer 9;
      Binary.Writer.u32 writer (List.length fields);
      List.iter
        (fun (name, value) ->
          Binary.Writer.string writer name;
          write_argument writer value)
        fields
  | Ir.Argument.Slice { start; stop; step } ->
      Binary.Writer.u8 writer 10;
      write_argument writer start;
      write_argument writer stop;
      write_argument writer step

and write_arguments writer values =
  Binary.Writer.u32 writer (List.length values);
  List.iter (write_argument writer) values

let find_value values id =
  match Hashtbl.find_opt values id with
  | Some value -> Ok value
  | None -> Error (Printf.sprintf "schedule argument references undefined value %d" id)

let write_slot writer slot =
  Binary.Writer.u32 writer (Kernel_region.Slot.to_int slot)

let read_slot reader =
  let* slot = Binary.Reader.u32 reader in
  Kernel_region.Slot.of_int slot

let write_linear_storage writer = function
  | Ir.Linear_storage.Dense dtype ->
      Binary.Writer.u8 writer 0;
      Binary.Writer.u8 writer (dtype_tag dtype)
  | Ir.Linear_storage.Block_quantized quant ->
      Binary.Writer.u8 writer 1;
      Binary.Writer.u8 writer (dtype_tag (Ir.Dtype.Quant quant))
  | Ir.Linear_storage.Groupwise_packed { bits; group_elements } ->
      Binary.Writer.u8 writer 2;
      Binary.Writer.u8 writer bits;
      Binary.Writer.u32 writer group_elements

let read_linear_storage reader =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 ->
      let* dtype_tag = Binary.Reader.u8 reader in
      let* dtype = dtype_of_tag dtype_tag in
      Ok (Ir.Linear_storage.Dense dtype)
  | 1 ->
      let* dtype_tag = Binary.Reader.u8 reader in
      let* dtype = dtype_of_tag dtype_tag in
      (match dtype with
      | Ir.Dtype.Quant quant -> Ok (Ir.Linear_storage.Block_quantized quant)
      | _ -> Error "kernel assignment block storage is not quantized")
  | 2 ->
      let* bits = Binary.Reader.u8 reader in
      let* group_elements = Binary.Reader.u32 reader in
      Ok (Ir.Linear_storage.Groupwise_packed { bits; group_elements })
  | _ -> Error (Printf.sprintf "unknown kernel assignment storage tag: %d" tag)

let write_kernel_scalar writer = function
  | Ir.Scalar.Bool value ->
      Binary.Writer.u8 writer 0;
      Binary.Writer.bool writer value
  | Ir.Scalar.Int value ->
      Binary.Writer.u8 writer 1;
      Binary.Writer.i64 writer value
  | Ir.Scalar.Float value ->
      Binary.Writer.u8 writer 2;
      Binary.Writer.float64 writer value

let read_kernel_scalar reader =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 -> Binary.Reader.bool reader |> Result.map (fun value -> Ir.Scalar.Bool value)
  | 1 -> Binary.Reader.i64 reader |> Result.map (fun value -> Ir.Scalar.Int value)
  | 2 -> Binary.Reader.float64 reader |> Result.map (fun value -> Ir.Scalar.Float value)
  | _ -> Error (Printf.sprintf "unknown kernel scalar tag: %d" tag)

let write_kernel_unary writer unary =
  let tag =
    match unary with
    | Kernel_region.Primitive.Silu -> 0
    | Kernel_region.Primitive.Gelu -> 1
    | Kernel_region.Primitive.Sigmoid -> 2
    | Kernel_region.Primitive.Neg -> 3
    | Kernel_region.Primitive.Rsqrt -> 4
    | Kernel_region.Primitive.Cos -> 5
    | Kernel_region.Primitive.Sin -> 6
    | Kernel_region.Primitive.Tanh -> 7
    | Kernel_region.Primitive.Exp -> 8
    | Kernel_region.Primitive.Softplus -> 9
    | Kernel_region.Primitive.Pow _ -> 10
  in
  Binary.Writer.u8 writer tag;
  match unary with
  | Kernel_region.Primitive.Pow scalar -> write_kernel_scalar writer scalar
  | _ -> ()

let read_kernel_unary reader =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 -> Ok Kernel_region.Primitive.Silu
  | 1 -> Ok Kernel_region.Primitive.Gelu
  | 2 -> Ok Kernel_region.Primitive.Sigmoid
  | 3 -> Ok Kernel_region.Primitive.Neg
  | 4 -> Ok Kernel_region.Primitive.Rsqrt
  | 5 -> Ok Kernel_region.Primitive.Cos
  | 6 -> Ok Kernel_region.Primitive.Sin
  | 7 -> Ok Kernel_region.Primitive.Tanh
  | 8 -> Ok Kernel_region.Primitive.Exp
  | 9 -> Ok Kernel_region.Primitive.Softplus
  | 10 ->
      read_kernel_scalar reader
      |> Result.map (fun scalar -> Kernel_region.Primitive.Pow scalar)
  | _ -> Error (Printf.sprintf "unknown kernel unary tag: %d" tag)

let write_kernel_binary writer binary =
  Binary.Writer.u8 writer
    (match binary with
    | Kernel_region.Primitive.Mul -> 0
    | Kernel_region.Primitive.Add -> 1
    | Kernel_region.Primitive.Sub -> 2
    | Kernel_region.Primitive.Div -> 3
    | Kernel_region.Primitive.Logical_and -> 4
    | Kernel_region.Primitive.Equal -> 5
    | Kernel_region.Primitive.Not_equal -> 6
    | Kernel_region.Primitive.Less_equal -> 7
    | Kernel_region.Primitive.Greater -> 8)

let read_kernel_binary reader =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 -> Ok Kernel_region.Primitive.Mul
  | 1 -> Ok Kernel_region.Primitive.Add
  | 2 -> Ok Kernel_region.Primitive.Sub
  | 3 -> Ok Kernel_region.Primitive.Div
  | 4 -> Ok Kernel_region.Primitive.Logical_and
  | 5 -> Ok Kernel_region.Primitive.Equal
  | 6 -> Ok Kernel_region.Primitive.Not_equal
  | 7 -> Ok Kernel_region.Primitive.Less_equal
  | 8 -> Ok Kernel_region.Primitive.Greater
  | _ -> Error (Printf.sprintf "unknown kernel binary tag: %d" tag)

let write_kernel_index writer index =
  let selectors = Tensor_shape.Index.selectors index in
  Binary.Writer.u16 writer (List.length selectors);
  List.iter
    (function
      | Tensor_shape.Index.At index ->
          Binary.Writer.u8 writer 0;
          Binary.Writer.u64 writer index
      | Tensor_shape.Index.Slice { start; step; length } ->
          Binary.Writer.u8 writer 1;
          Binary.Writer.i64 writer start;
          Binary.Writer.i64 writer step;
          Binary.Writer.u64 writer length
      | Tensor_shape.Index.New_axis -> Binary.Writer.u8 writer 2)
    selectors

let read_kernel_index reader =
  let* count = Binary.Reader.u16 reader in
  let rec loop reversed remaining =
    if remaining = 0 then Tensor_shape.Index.of_selectors (List.rev reversed)
    else
      let* tag = Binary.Reader.u8 reader in
      let* selector =
        match tag with
        | 0 ->
            Binary.Reader.u64 reader
            |> Result.map (fun index -> Tensor_shape.Index.At index)
        | 1 ->
            let* start = Binary.Reader.i64 reader in
            let* step = Binary.Reader.i64 reader in
            let* length = Binary.Reader.u64 reader in
            Ok (Tensor_shape.Index.Slice { start; step; length })
        | 2 -> Ok Tensor_shape.Index.New_axis
        | _ -> Error (Printf.sprintf "unknown kernel index tag: %d" tag)
      in
      loop (selector :: reversed) (remaining - 1)
  in
  loop [] count

let write_kernel_movement writer = function
  | Ir.Movement.View -> Binary.Writer.u8 writer 0
  | Ir.Movement.Reshape -> Binary.Writer.u8 writer 1
  | Ir.Movement.Transpose { axis0; axis1 } ->
      Binary.Writer.u8 writer 2;
      Binary.Writer.i64 writer axis0;
      Binary.Writer.i64 writer axis1
  | Ir.Movement.Unsqueeze axis ->
      Binary.Writer.u8 writer 3;
      Binary.Writer.i64 writer axis
  | Ir.Movement.Expand -> Binary.Writer.u8 writer 4
  | Ir.Movement.Contiguous -> Binary.Writer.u8 writer 5
  | Ir.Movement.Index index ->
      Binary.Writer.u8 writer 6;
      write_kernel_index writer index
  | Ir.Movement.Concat { axis } ->
      Binary.Writer.u8 writer 7;
      Binary.Writer.i64 writer axis
  | Ir.Movement.Roll { axis; shift } ->
      Binary.Writer.u8 writer 8;
      Binary.Writer.i64 writer axis;
      Binary.Writer.i64 writer shift

let read_kernel_movement reader =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 -> Ok Ir.Movement.View
  | 1 -> Ok Ir.Movement.Reshape
  | 2 ->
      let* axis0 = Binary.Reader.i64 reader in
      let* axis1 = Binary.Reader.i64 reader in
      Ok (Ir.Movement.Transpose { axis0; axis1 })
  | 3 ->
      Binary.Reader.i64 reader
      |> Result.map (fun axis -> Ir.Movement.Unsqueeze axis)
  | 4 -> Ok Ir.Movement.Expand
  | 5 -> Ok Ir.Movement.Contiguous
  | 6 ->
      read_kernel_index reader
      |> Result.map (fun index -> Ir.Movement.Index index)
  | 7 ->
      Binary.Reader.i64 reader
      |> Result.map (fun axis -> Ir.Movement.Concat { axis })
  | 8 ->
      let* axis = Binary.Reader.i64 reader in
      let* shift = Binary.Reader.i64 reader in
      Ok (Ir.Movement.Roll { axis; shift })
  | _ -> Error (Printf.sprintf "unknown kernel movement tag: %d" tag)

let write_kernel_index writer index =
  let selectors = Tensor_shape.Index.selectors index in
  Binary.Writer.u16 writer (List.length selectors);
  List.iter
    (function
      | Tensor_shape.Index.At index ->
          Binary.Writer.u8 writer 0;
          Binary.Writer.u64 writer index
      | Tensor_shape.Index.Slice { start; step; length } ->
          Binary.Writer.u8 writer 1;
          Binary.Writer.i64 writer start;
          Binary.Writer.i64 writer step;
          Binary.Writer.u64 writer length
      | Tensor_shape.Index.New_axis -> Binary.Writer.u8 writer 2)
    selectors

let read_kernel_index reader =
  let* count = Binary.Reader.u16 reader in
  let rec selectors reversed remaining =
    if remaining = 0 then Tensor_shape.Index.of_selectors (List.rev reversed)
    else
      let* tag = Binary.Reader.u8 reader in
      let* selector =
        match tag with
        | 0 ->
            Binary.Reader.u64 reader
            |> Result.map (fun index -> Tensor_shape.Index.At index)
        | 1 ->
            let* start = Binary.Reader.i64 reader in
            let* step = Binary.Reader.i64 reader in
            let* length = Binary.Reader.u64 reader in
            Ok (Tensor_shape.Index.Slice { start; step; length })
        | 2 -> Ok Tensor_shape.Index.New_axis
        | _ -> Error (Printf.sprintf "unknown kernel tensor-index tag: %d" tag)
      in
      selectors (selector :: reversed) (remaining - 1)
  in
  selectors [] count

let write_kernel_primitive writer = function
  | Kernel_region.Primitive.Rms_norm { epsilon } ->
      Binary.Writer.u8 writer 0;
      Binary.Writer.float64 writer epsilon
  | Kernel_region.Primitive.Linear { m; n; k; bias; storage } ->
      Binary.Writer.u8 writer 1;
      Binary.Writer.u32 writer m;
      Binary.Writer.u32 writer n;
      Binary.Writer.u32 writer k;
      Binary.Writer.bool writer bias;
      write_linear_storage writer storage
  | Kernel_region.Primitive.Unary unary ->
      Binary.Writer.u8 writer 2;
      write_kernel_unary writer unary
  | Kernel_region.Primitive.Binary binary ->
      Binary.Writer.u8 writer 3;
      write_kernel_binary writer binary
  | Kernel_region.Primitive.Scalar_binary { operator; scalar; scalar_on_left } ->
      Binary.Writer.u8 writer 4;
      write_kernel_binary writer operator;
      write_kernel_scalar writer scalar;
      Binary.Writer.bool writer scalar_on_left
  | Kernel_region.Primitive.Reduce reduction ->
      Binary.Writer.u8 writer 5;
      Binary.Writer.u8 writer
        (match reduction.Ir.Reduction.operator with
        | Ir.Reduction.Mean -> 0
        | Ir.Reduction.Sum -> 1
        | Ir.Reduction.Argmax -> 2);
      Binary.Writer.u16 writer (List.length reduction.axes);
      List.iter (Binary.Writer.i64 writer) reduction.axes;
      Binary.Writer.bool writer reduction.keepdim
  | Kernel_region.Primitive.Cast dtype ->
      Binary.Writer.u8 writer 6;
      Binary.Writer.u8 writer (dtype_tag dtype)
  | Kernel_region.Primitive.Movement movement ->
      Binary.Writer.u8 writer 7;
      write_kernel_movement writer movement
  | Kernel_region.Primitive.Short_conv convolution ->
      Binary.Writer.u8 writer 8;
      Binary.Writer.u32 writer (Ir.Short_conv.stride convolution);
      Binary.Writer.u32 writer (Ir.Short_conv.padding convolution);
      Binary.Writer.u32 writer (Ir.Short_conv.dilation convolution);
      Binary.Writer.u32 writer (Ir.Short_conv.groups convolution)
  | Kernel_region.Primitive.Batched_matmul -> Binary.Writer.u8 writer 9
  | Kernel_region.Primitive.Pad_right_zero { axis } ->
      Binary.Writer.u8 writer 10;
      Binary.Writer.u16 writer axis
  | Kernel_region.Primitive.Update_slice index ->
      Binary.Writer.u8 writer 11;
      write_kernel_index writer index
  | Kernel_region.Primitive.Triangular_recurrence config ->
      Binary.Writer.u8 writer 12;
      Binary.Writer.u16 writer (Ir.Triangular_recurrence.axis config);
      Binary.Writer.u16 writer (Ir.Triangular_recurrence.start config);
      Binary.Writer.u16 writer (Ir.Triangular_recurrence.stop config)

let read_kernel_primitive reader =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 ->
      let* epsilon = Binary.Reader.float64 reader in
      if Float.is_finite epsilon then Ok (Kernel_region.Primitive.rms_norm ~epsilon)
      else Error "kernel assignment RMSNorm epsilon is not finite"
  | 1 ->
      let* m = Binary.Reader.u32 reader in
      let* n = Binary.Reader.u32 reader in
      let* k = Binary.Reader.u32 reader in
      let* bias = Binary.Reader.bool reader in
      let* storage = read_linear_storage reader in
      Ok (Kernel_region.Primitive.linear ~m ~n ~k ~bias ~storage)
  | 2 -> read_kernel_unary reader |> Result.map Kernel_region.Primitive.unary
  | 3 -> read_kernel_binary reader |> Result.map Kernel_region.Primitive.binary
  | 4 ->
      let* operator = read_kernel_binary reader in
      let* scalar = read_kernel_scalar reader in
      let* scalar_on_left = Binary.Reader.bool reader in
      Ok
        (Kernel_region.Primitive.scalar_binary ~operator ~scalar ~scalar_on_left)
  | 5 ->
      let* operator_tag = Binary.Reader.u8 reader in
      let* operator =
        match operator_tag with
        | 0 -> Ok Ir.Reduction.Mean
        | 1 -> Ok Ir.Reduction.Sum
        | 2 -> Ok Ir.Reduction.Argmax
        | _ ->
            Error
              (Printf.sprintf "unknown kernel reduction tag: %d" operator_tag)
      in
      let* count = Binary.Reader.u16 reader in
      let rec axes reversed remaining =
        if remaining = 0 then Ok (List.rev reversed)
        else
          let* axis = Binary.Reader.i64 reader in
          axes (axis :: reversed) (remaining - 1)
      in
      let* axes = axes [] count in
      let* keepdim = Binary.Reader.bool reader in
      Ok
        (Kernel_region.Primitive.reduce
           { Ir.Reduction.operator; axes; keepdim })
  | 6 ->
      let* dtype_tag = Binary.Reader.u8 reader in
      let* dtype = dtype_of_tag dtype_tag in
      Ok (Kernel_region.Primitive.cast dtype)
  | 7 ->
      read_kernel_movement reader |> Result.map Kernel_region.Primitive.movement
  | 8 ->
      let* stride = Binary.Reader.u32 reader in
      let* padding = Binary.Reader.u32 reader in
      let* dilation = Binary.Reader.u32 reader in
      let* groups = Binary.Reader.u32 reader in
      let* convolution =
        Ir.Short_conv.create ~stride ~padding ~dilation ~groups
      in
      Ok (Kernel_region.Primitive.short_conv convolution)
  | 9 -> Ok Kernel_region.Primitive.batched_matmul
  | 10 ->
      Binary.Reader.u16 reader
      |> Result.map (fun axis -> Kernel_region.Primitive.pad_right_zero ~axis)
  | 11 ->
      read_kernel_index reader |> Result.map Kernel_region.Primitive.update_slice
  | 12 ->
      let* axis = Binary.Reader.u16 reader in
      let* start = Binary.Reader.u16 reader in
      let* stop = Binary.Reader.u16 reader in
      Ir.Triangular_recurrence.create ~axis ~start ~stop
      |> Result.map Kernel_region.Primitive.triangular_recurrence
  | _ -> Error (Printf.sprintf "unknown kernel assignment primitive tag: %d" tag)

let write_slots writer slots =
  Binary.Writer.u32 writer (List.length slots);
  List.iter (write_slot writer) slots

let read_slots reader =
  let* count = Binary.Reader.u32 reader in
  let rec loop reversed remaining =
    if remaining = 0 then Ok (List.rev reversed)
    else
      let* slot = read_slot reader in
      loop (slot :: reversed) (remaining - 1)
  in
  loop [] count

let write_fusion_region writer region =
  Binary.Writer.string writer (Kernel_region.name region);
  Binary.Writer.u32 writer (List.length (Kernel_region.member_node_ids region));
  List.iter (Binary.Writer.u32 writer) (Kernel_region.member_node_ids region);
  Binary.Writer.u32 writer (List.length (Kernel_region.inputs region));
  Kernel_region.inputs region
  |> List.iter (fun ({ Kernel_region.slot; value } : Kernel_region.input) ->
         write_slot writer slot;
         Binary.Writer.u32 writer (value_id value));
  Binary.Writer.u32 writer (List.length (Kernel_region.bindings region));
  Kernel_region.bindings region
  |> List.iter (fun binding ->
         let outputs = Kernel_region.binding_outputs binding in
         Binary.Writer.u32 writer (List.length outputs);
         outputs
         |> List.iter (fun output ->
                write_slot writer (Kernel_region.output_slot output);
                Binary.Writer.u8 writer
                  (dtype_tag (Kernel_region.output_dtype output));
                write_shape writer (Kernel_region.output_shape output));
         write_kernel_primitive writer (Kernel_region.binding_primitive binding);
         write_slots writer (Kernel_region.binding_inputs binding));
  Binary.Writer.u32 writer (List.length (Kernel_region.results region));
  Kernel_region.results region
  |> List.iter (fun result ->
         write_slot writer (Kernel_region.result_slot result);
         Binary.Writer.u32 writer
           (value_id (Kernel_region.result_value result));
         match Kernel_region.result_storage result with
         | Kernel_region.Fresh -> Binary.Writer.u8 writer 0
         | Kernel_region.Alias_input slot ->
             Binary.Writer.u8 writer 1;
             write_slot writer slot
         | Kernel_region.In_place_input slot ->
             Binary.Writer.u8 writer 2;
             write_slot writer slot);
  let effects = Kernel_region.effects region in
  let accesses = Kernel_region.Effect.accesses effects in
  Binary.Writer.u32 writer (List.length accesses);
  accesses
  |> List.iter (fun (access : Kernel_region.Effect.access) ->
         write_slot writer access.slot;
         Binary.Writer.u8 writer
           (match access.mode with
           | Kernel_region.Effect.Read -> 0
           | Kernel_region.Effect.Write -> 1
           | Kernel_region.Effect.Read_write -> 2);
         match access.alias with
         | Kernel_region.Effect.Distinct -> Binary.Writer.u8 writer 0
         | Kernel_region.Effect.May_alias -> Binary.Writer.u8 writer 1
         | Kernel_region.Effect.Alias_of slot ->
             Binary.Writer.u8 writer 2;
             write_slot writer slot);
  write_slots writer (Kernel_region.Effect.state_inputs effects);
  write_slots writer (Kernel_region.Effect.state_outputs effects);
  Binary.Writer.u32 writer (List.length (Kernel_region.Effect.barriers effects));
  List.iter (Binary.Writer.u32 writer) (Kernel_region.Effect.barriers effects);
  let resource = Kernel_region.resource region in
  Binary.Writer.u64_int64 writer (Kernel_region.Resource.scalar_ops resource);
  Binary.Writer.u64_int64 writer (Kernel_region.Resource.bytes_read resource);
  Binary.Writer.u64_int64 writer (Kernel_region.Resource.bytes_written resource);
  Binary.Writer.u64_int64 writer (Kernel_region.Resource.temporary_bytes resource);
  Binary.Writer.u32 writer
    (Kernel_region.Resource.synchronization_points resource)

let read_fusion_region values reader =
  let* name = Binary.Reader.string reader in
  let* member_count = Binary.Reader.u32 reader in
  let rec members reversed remaining =
    if remaining = 0 then Ok (List.rev reversed)
    else
      let* member = Binary.Reader.u32 reader in
      members (member :: reversed) (remaining - 1)
  in
  let* member_node_ids = members [] member_count in
  let* input_count = Binary.Reader.u32 reader in
  let rec inputs reversed remaining =
    if remaining = 0 then Ok (List.rev reversed)
    else
      let* slot = read_slot reader in
      let* value_id = Binary.Reader.u32 reader in
      let* value = find_value values value_id in
      inputs (({ Kernel_region.slot; value } : Kernel_region.input) :: reversed)
        (remaining - 1)
  in
  let* inputs = inputs [] input_count in
  let* binding_count = Binary.Reader.u32 reader in
  let rec bindings reversed remaining =
    if remaining = 0 then Ok (List.rev reversed)
    else
      let* output_count = Binary.Reader.u32 reader in
      let rec outputs reversed remaining =
        if remaining = 0 then Ok (List.rev reversed)
        else
          let* slot = read_slot reader in
          let* dtype_tag = Binary.Reader.u8 reader in
          let* dtype = dtype_of_tag dtype_tag in
          let* shape = read_shape reader in
          outputs
            (({ Kernel_region.slot; shape; dtype; value = None } : Kernel_region.output_spec)
            :: reversed)
            (remaining - 1)
      in
      let* outputs = outputs [] output_count in
      let* primitive = read_kernel_primitive reader in
      let* input_slots = read_slots reader in
      let* binding =
        Kernel_region.make_binding ~outputs ~primitive ~inputs:input_slots
      in
      bindings (binding :: reversed) (remaining - 1)
  in
  let* bindings = bindings [] binding_count in
  let* result_count = Binary.Reader.u32 reader in
  let rec results reversed remaining =
    if remaining = 0 then Ok (List.rev reversed)
    else
      let* slot = read_slot reader in
      let* value_id = Binary.Reader.u32 reader in
      let* value = find_value values value_id in
      let* storage_tag = Binary.Reader.u8 reader in
      let* storage =
        match storage_tag with
        | 0 -> Ok Kernel_region.Fresh
        | 1 -> read_slot reader |> Result.map (fun slot -> Kernel_region.Alias_input slot)
        | 2 ->
            read_slot reader
            |> Result.map (fun slot -> Kernel_region.In_place_input slot)
        | _ -> Error (Printf.sprintf "unknown kernel result storage tag: %d" storage_tag)
      in
      results (({ Kernel_region.slot; value; storage } : Kernel_region.result) :: reversed)
        (remaining - 1)
  in
  let* results = results [] result_count in
  let* access_count = Binary.Reader.u32 reader in
  let rec accesses reversed remaining =
    if remaining = 0 then Ok (List.rev reversed)
    else
      let* slot = read_slot reader in
      let* mode_tag = Binary.Reader.u8 reader in
      let* mode =
        match mode_tag with
        | 0 -> Ok Kernel_region.Effect.Read
        | 1 -> Ok Kernel_region.Effect.Write
        | 2 -> Ok Kernel_region.Effect.Read_write
        | _ -> Error (Printf.sprintf "unknown kernel effect mode tag: %d" mode_tag)
      in
      let* alias_tag = Binary.Reader.u8 reader in
      let* alias =
        match alias_tag with
        | 0 -> Ok Kernel_region.Effect.Distinct
        | 1 -> Ok Kernel_region.Effect.May_alias
        | 2 -> read_slot reader |> Result.map (fun slot -> Kernel_region.Effect.Alias_of slot)
        | _ -> Error (Printf.sprintf "unknown kernel effect alias tag: %d" alias_tag)
      in
      accesses
        (({ Kernel_region.Effect.slot; mode; alias } : Kernel_region.Effect.access)
        :: reversed)
        (remaining - 1)
  in
  let* accesses = accesses [] access_count in
  let* state_inputs = read_slots reader in
  let* state_outputs = read_slots reader in
  let* barrier_count = Binary.Reader.u32 reader in
  let rec barriers reversed remaining =
    if remaining = 0 then Ok (List.rev reversed)
    else
      let* barrier = Binary.Reader.u32 reader in
      barriers (barrier :: reversed) (remaining - 1)
  in
  let* barriers = barriers [] barrier_count in
  let* effects =
    Kernel_region.Effect.create ~accesses ~state_inputs ~state_outputs ~barriers ()
  in
  let* scalar_ops = Binary.Reader.u64_int64 reader in
  let* bytes_read = Binary.Reader.u64_int64 reader in
  let* bytes_written = Binary.Reader.u64_int64 reader in
  let* temporary_bytes = Binary.Reader.u64_int64 reader in
  let* synchronization_points = Binary.Reader.u32 reader in
  let* resource =
    Kernel_region.Resource.create ~scalar_ops ~bytes_read ~bytes_written
      ~temporary_bytes ~synchronization_points
  in
  Kernel_region.create ~name ~member_node_ids ~inputs ~bindings ~results ~effects
    ~resource
  |> Result.map_error Kernel_region.error_to_string

let rec read_argument ~depth values reader =
  if depth > 64 then Error "schedule argument nesting is too deep"
  else
    let* tag = Binary.Reader.u8 reader in
    match tag with
    | 0 ->
        let* id = Binary.Reader.u32 reader in
        find_value values id |> Result.map (fun value -> Ir.Argument.Value value)
    | 1 -> Ok Ir.Argument.Null
    | 11 -> Ok Ir.Argument.Ellipsis
    | 2 -> Binary.Reader.bool reader |> Result.map (fun value -> Ir.Argument.Bool value)
    | 3 -> Binary.Reader.i64 reader |> Result.map (fun value -> Ir.Argument.Int value)
    | 4 ->
        let* value = Binary.Reader.float64 reader in
        if Float.is_finite value then Ok (Ir.Argument.Float value)
        else Error "schedule contains a non-finite float argument"
    | 5 -> Binary.Reader.string reader |> Result.map (fun value -> Ir.Argument.String value)
    | 6 -> Binary.Reader.string reader |> Result.map (fun value -> Ir.Argument.Symbol value)
    | 7 ->
        read_arguments ~depth:(depth + 1) values reader
        |> Result.map (fun values -> Ir.Argument.List values)
    | 8 ->
        read_arguments ~depth:(depth + 1) values reader
        |> Result.map (fun values -> Ir.Argument.Tuple values)
    | 9 ->
        let* count = Binary.Reader.u32 reader in
        let rec fields acc remaining =
          if remaining = 0 then Ok (Ir.Argument.Mapping (List.rev acc))
          else
            let* name = Binary.Reader.string reader in
            let* value = read_argument ~depth:(depth + 1) values reader in
            fields ((name, value) :: acc) (remaining - 1)
        in
        fields [] count
    | 10 ->
        let* start = read_argument ~depth:(depth + 1) values reader in
        let* stop = read_argument ~depth:(depth + 1) values reader in
        let* step = read_argument ~depth:(depth + 1) values reader in
        Ok (Ir.Argument.Slice { start; stop; step })
    | _ -> Error (Printf.sprintf "unknown schedule argument tag: %d" tag)

and read_arguments ~depth values reader =
  let* count = Binary.Reader.u32 reader in
  let rec loop acc remaining =
    if remaining = 0 then Ok (List.rev acc)
    else
      let* value = read_argument ~depth values reader in
      loop (value :: acc) (remaining - 1)
  in
  loop [] count

let write_named_arguments writer fields =
  Binary.Writer.u32 writer (List.length fields);
  List.iter
    (fun (name, value) ->
      Binary.Writer.string writer name;
      write_argument writer value)
    fields

let read_named_arguments values reader =
  let* count = Binary.Reader.u32 reader in
  let rec loop acc remaining =
    if remaining = 0 then Ok (List.rev acc)
    else
      let* name = Binary.Reader.string reader in
      let* value = read_argument ~depth:0 values reader in
      loop ((name, value) :: acc) (remaining - 1)
  in
  loop [] count

let write_scalar writer = function
  | Ir.Scalar.Bool value ->
      Binary.Writer.u8 writer 0;
      Binary.Writer.bool writer value
  | Ir.Scalar.Int value ->
      Binary.Writer.u8 writer 1;
      Binary.Writer.i64 writer value
  | Ir.Scalar.Float value ->
      Binary.Writer.u8 writer 2;
      Binary.Writer.float64 writer value

let read_scalar reader =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 -> Binary.Reader.bool reader |> Result.map (fun value -> Ir.Scalar.Bool value)
  | 1 -> Binary.Reader.i64 reader |> Result.map (fun value -> Ir.Scalar.Int value)
  | 2 ->
      let* value = Binary.Reader.float64 reader in
      if Float.is_finite value then Ok (Ir.Scalar.Float value)
      else Error "schedule contains a non-finite pointwise scalar"
  | _ -> Error (Printf.sprintf "unknown pointwise scalar tag: %d" tag)

let write_pointwise_operand writer = function
  | Ir.Pointwise.Tensor value ->
      Binary.Writer.u8 writer 0;
      Binary.Writer.u32 writer (value_id value)
  | Ir.Pointwise.Scalar scalar ->
      Binary.Writer.u8 writer 1;
      write_scalar writer scalar

let read_pointwise_operand values reader =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 ->
      let* id = Binary.Reader.u32 reader in
      find_value values id |> Result.map (fun value -> Ir.Pointwise.Tensor value)
  | 1 ->
      read_scalar reader |> Result.map (fun scalar -> Ir.Pointwise.Scalar scalar)
  | _ -> Error (Printf.sprintf "unknown pointwise operand tag: %d" tag)

let write_pointwise writer = function
  | Ir.Pointwise.Unary (operator, input) ->
      Binary.Writer.u8 writer 0;
      (match operator with
      | Ir.Pointwise.Neg -> Binary.Writer.u8 writer 0
      | Ir.Pointwise.Rsqrt -> Binary.Writer.u8 writer 1
      | Ir.Pointwise.Silu -> Binary.Writer.u8 writer 2
      | Ir.Pointwise.Cos -> Binary.Writer.u8 writer 3
      | Ir.Pointwise.Sin -> Binary.Writer.u8 writer 4
      | Ir.Pointwise.Tanh -> Binary.Writer.u8 writer 6
      | Ir.Pointwise.Exp -> Binary.Writer.u8 writer 7
      | Ir.Pointwise.Sigmoid -> Binary.Writer.u8 writer 8
      | Ir.Pointwise.Softplus -> Binary.Writer.u8 writer 9
      | Ir.Pointwise.Pow exponent ->
          Binary.Writer.u8 writer 5;
          write_scalar writer exponent);
      Binary.Writer.u32 writer (value_id input)
  | Ir.Pointwise.Binary (operator, left, right) ->
      Binary.Writer.u8 writer 1;
      Binary.Writer.u8 writer
        (match operator with
        | Ir.Pointwise.Add -> 0
        | Ir.Pointwise.Mul -> 1
        | Ir.Pointwise.Silu_mul -> 9
        | Ir.Pointwise.Gelu_mul -> 10
        | Ir.Pointwise.Sigmoid_mul -> 11
        | Ir.Pointwise.Sub -> 2
        | Ir.Pointwise.Logical_and -> 3
        | Ir.Pointwise.Equal -> 4
        | Ir.Pointwise.Not_equal -> 5
        | Ir.Pointwise.Less_equal -> 6
        | Ir.Pointwise.Div -> 7
        | Ir.Pointwise.Greater -> 8);
      write_pointwise_operand writer left;
      write_pointwise_operand writer right

let read_pointwise values reader =
  let* kind = Binary.Reader.u8 reader in
  match kind with
  | 0 ->
      let* tag = Binary.Reader.u8 reader in
      let* operator =
        match tag with
        | 0 -> Ok Ir.Pointwise.Neg
        | 1 -> Ok Ir.Pointwise.Rsqrt
        | 2 -> Ok Ir.Pointwise.Silu
        | 3 -> Ok Ir.Pointwise.Cos
        | 4 -> Ok Ir.Pointwise.Sin
        | 5 -> read_scalar reader |> Result.map (fun scalar -> Ir.Pointwise.Pow scalar)
        | 6 -> Ok Ir.Pointwise.Tanh
        | 7 -> Ok Ir.Pointwise.Exp
        | 8 -> Ok Ir.Pointwise.Sigmoid
        | 9 -> Ok Ir.Pointwise.Softplus
        | _ -> Error (Printf.sprintf "unknown pointwise unary tag: %d" tag)
      in
      let* id = Binary.Reader.u32 reader in
      let* input = find_value values id in
      Ok (Ir.Pointwise.Unary (operator, input))
  | 1 ->
      let* tag = Binary.Reader.u8 reader in
      let* operator =
        match tag with
        | 0 -> Ok Ir.Pointwise.Add
        | 1 -> Ok Ir.Pointwise.Mul
        | 9 -> Ok Ir.Pointwise.Silu_mul
        | 10 -> Ok Ir.Pointwise.Gelu_mul
        | 11 -> Ok Ir.Pointwise.Sigmoid_mul
        | 2 -> Ok Ir.Pointwise.Sub
        | 3 -> Ok Ir.Pointwise.Logical_and
        | 4 -> Ok Ir.Pointwise.Equal
        | 5 -> Ok Ir.Pointwise.Not_equal
        | 6 -> Ok Ir.Pointwise.Less_equal
        | 7 -> Ok Ir.Pointwise.Div
        | 8 -> Ok Ir.Pointwise.Greater
        | _ -> Error (Printf.sprintf "unknown pointwise binary tag: %d" tag)
      in
      let* left = read_pointwise_operand values reader in
      let* right = read_pointwise_operand values reader in
      Ok (Ir.Pointwise.Binary (operator, left, right))
  | _ -> Error (Printf.sprintf "unknown pointwise operation tag: %d" kind)

let write_index writer index =
  let selectors = Tensor_shape.Index.selectors index in
  Binary.Writer.u16 writer (List.length selectors);
  List.iter
    (function
      | Tensor_shape.Index.At index ->
          Binary.Writer.u8 writer 0;
          Binary.Writer.u64 writer index
      | Tensor_shape.Index.Slice { start; step; length } ->
          Binary.Writer.u8 writer 1;
          Binary.Writer.i64 writer start;
          Binary.Writer.i64 writer step;
          Binary.Writer.u64 writer length
      | Tensor_shape.Index.New_axis -> Binary.Writer.u8 writer 2)
    selectors

let read_index reader =
  let* count = Binary.Reader.u16 reader in
  let rec selectors acc remaining =
    if remaining = 0 then
      Tensor_shape.Index.of_selectors (List.rev acc)
    else
      let* tag = Binary.Reader.u8 reader in
      let* selector =
        match tag with
        | 0 ->
            Binary.Reader.u64 reader
            |> Result.map (fun index -> Tensor_shape.Index.At index)
        | 1 ->
            let* start = Binary.Reader.i64 reader in
            let* step = Binary.Reader.i64 reader in
            let* length = Binary.Reader.u64 reader in
            Ok (Tensor_shape.Index.Slice { start; step; length })
        | 2 -> Ok Tensor_shape.Index.New_axis
        | _ -> Error (Printf.sprintf "unknown tensor-index tag: %d" tag)
      in
      selectors (selector :: acc) (remaining - 1)
  in
  selectors [] count

let write_primitive writer = function
  | Ir.Primitive.Pointwise operation ->
      Binary.Writer.u8 writer 0;
      write_pointwise writer operation
  | Ir.Primitive.Cast dtype ->
      Binary.Writer.u8 writer 1;
      Binary.Writer.u8 writer (dtype_tag dtype)
  | Ir.Primitive.Reduce { operator = Ir.Reduction.Mean; axes; keepdim } ->
      Binary.Writer.u8 writer 2;
      Binary.Writer.u16 writer (List.length axes);
      List.iter (Binary.Writer.u16 writer) axes;
      Binary.Writer.bool writer keepdim
  | Ir.Primitive.Reduce { operator = Ir.Reduction.Sum; axes; keepdim } ->
      Binary.Writer.u8 writer 12;
      Binary.Writer.u16 writer (List.length axes);
      List.iter (Binary.Writer.u16 writer) axes;
      Binary.Writer.bool writer keepdim
  | Ir.Primitive.Reduce { operator = Ir.Reduction.Argmax; axes; keepdim } ->
      Binary.Writer.u8 writer 25;
      Binary.Writer.u16 writer (List.length axes);
      List.iter (Binary.Writer.u16 writer) axes;
      Binary.Writer.bool writer keepdim
  | Ir.Primitive.Movement movement ->
      Binary.Writer.u8 writer 3;
      (match movement with
      | Ir.Movement.View -> Binary.Writer.u8 writer 0
      | Ir.Movement.Reshape -> Binary.Writer.u8 writer 1
      | Ir.Movement.Transpose { axis0; axis1 } ->
          Binary.Writer.u8 writer 2;
          Binary.Writer.u16 writer axis0;
          Binary.Writer.u16 writer axis1
      | Ir.Movement.Unsqueeze axis ->
          Binary.Writer.u8 writer 3;
          Binary.Writer.u16 writer axis
      | Ir.Movement.Expand -> Binary.Writer.u8 writer 4
      | Ir.Movement.Contiguous -> Binary.Writer.u8 writer 5
      | Ir.Movement.Index index ->
          Binary.Writer.u8 writer 6;
          write_index writer index
      | Ir.Movement.Concat { axis } ->
          Binary.Writer.u8 writer 7;
          Binary.Writer.u16 writer axis
      | Ir.Movement.Roll { axis; shift } ->
          Binary.Writer.u8 writer 8;
          Binary.Writer.u16 writer axis;
          Binary.Writer.i64 writer shift)
  | Ir.Primitive.Short_conv config ->
      Binary.Writer.u8 writer 4;
      Binary.Writer.u32 writer (Ir.Short_conv.stride config);
      Binary.Writer.u32 writer (Ir.Short_conv.padding config);
      Binary.Writer.u32 writer (Ir.Short_conv.dilation config);
      Binary.Writer.u32 writer (Ir.Short_conv.groups config)
  | Ir.Primitive.Short_conv_silu config ->
      let convolution = Ir.Short_conv_silu.convolution config in
      Binary.Writer.u8 writer 23;
      Binary.Writer.u32 writer (Ir.Short_conv.stride convolution);
      Binary.Writer.u32 writer (Ir.Short_conv.padding convolution);
      Binary.Writer.u32 writer (Ir.Short_conv.dilation convolution);
      Binary.Writer.u32 writer (Ir.Short_conv.groups convolution);
      Binary.Writer.u32 writer (Ir.Short_conv_silu.output_start config)
  | Ir.Primitive.Attention config ->
      Binary.Writer.u8 writer 5;
      Binary.Writer.float64 writer (Ir.Attention.scale config);
      Binary.Writer.bool writer (Ir.Attention.causal config)
  | Ir.Primitive.Paged_attention_q8 config ->
      Binary.Writer.u8 writer 14;
      Binary.Writer.float64 writer (Ir.Paged_attention_q8.scale config);
      List.iter (Binary.Writer.u32 writer)
        [ Ir.Paged_attention_q8.cache_layer config;
          Ir.Paged_attention_q8.kv_heads config;
          Ir.Paged_attention_q8.head_dim config;
          Ir.Paged_attention_q8.token_elements config;
          Ir.Paged_attention_q8.key_offset config;
          Ir.Paged_attention_q8.value_offset config;
          Ir.Paged_attention_q8.group_size config;
          Ir.Paged_attention_q8.token_stride config ]
      ; Binary.Writer.u32 writer
          (Option.fold ~none:0 ~some:Fun.id
             (Ir.Paged_attention_q8.window config))
  | Ir.Primitive.Embedding -> Binary.Writer.u8 writer 6
  | Ir.Primitive.Arange config ->
      Binary.Writer.u8 writer 7;
      Binary.Writer.i64 writer (Ir.Arange.start config);
      Binary.Writer.i64 writer (Ir.Arange.stop config);
      Binary.Writer.i64 writer (Ir.Arange.step config)
  | Ir.Primitive.Diff config ->
      Binary.Writer.u8 writer 8;
      Binary.Writer.u16 writer (Ir.Diff.axis config)
  | Ir.Primitive.Cumsum config ->
      Binary.Writer.u8 writer 9;
      Binary.Writer.u16 writer (Ir.Cumsum.axis config)
  | Ir.Primitive.Triangular_recurrence config ->
      Binary.Writer.u8 writer 20;
      Binary.Writer.u16 writer (Ir.Triangular_recurrence.axis config);
      Binary.Writer.u16 writer (Ir.Triangular_recurrence.start config);
      Binary.Writer.u16 writer (Ir.Triangular_recurrence.stop config)
  | Ir.Primitive.Fill scalar ->
      Binary.Writer.u8 writer 10;
      write_scalar writer scalar
  | Ir.Primitive.Gather2 -> Binary.Writer.u8 writer 11
  | Ir.Primitive.Update_slice index ->
      Binary.Writer.u8 writer 13;
      write_index writer index
  | Ir.Primitive.Pad_right_zero { axis } ->
      Binary.Writer.u8 writer 15;
      Binary.Writer.u16 writer axis
  | Ir.Primitive.Triangular { upper; diagonal } ->
      Binary.Writer.u8 writer 16;
      Binary.Writer.bool writer upper;
      Binary.Writer.i64 writer diagonal
  | Ir.Primitive.Masked_fill scalar ->
      Binary.Writer.u8 writer 17;
      write_scalar writer scalar
  | Ir.Primitive.Eye -> Binary.Writer.u8 writer 18
  | Ir.Primitive.Batched_matmul -> Binary.Writer.u8 writer 19
  | Ir.Primitive.Gated_delta -> Binary.Writer.u8 writer 21
  | Ir.Primitive.L2_norm { epsilon } ->
      Binary.Writer.u8 writer 22;
      Binary.Writer.float64 writer epsilon
  | Ir.Primitive.L2_norm_slice config ->
      Binary.Writer.u8 writer 24;
      Binary.Writer.float64 writer (Ir.L2_norm_slice.epsilon config);
      Binary.Writer.u32 writer (Ir.L2_norm_slice.offset config)

let read_primitive values reader =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 -> read_pointwise values reader |> Result.map (fun value -> Ir.Primitive.Pointwise value)
  | 1 ->
      let* tag = Binary.Reader.u8 reader in
      dtype_of_tag tag |> Result.map (fun dtype -> Ir.Primitive.Cast dtype)
  | 2 ->
      let* count = Binary.Reader.u16 reader in
      let rec axes acc remaining =
        if remaining = 0 then Ok (List.rev acc)
        else
          let* axis = Binary.Reader.u16 reader in
          axes (axis :: acc) (remaining - 1)
      in
      let* axes = axes [] count in
      let* keepdim = Binary.Reader.bool reader in
      Ok
        (Ir.Primitive.Reduce
           { Ir.Reduction.operator = Mean; axes; keepdim })
  | 3 ->
      let* movement_tag = Binary.Reader.u8 reader in
      let* movement =
        match movement_tag with
        | 0 -> Ok Ir.Movement.View
        | 1 -> Ok Ir.Movement.Reshape
        | 2 ->
            let* axis0 = Binary.Reader.u16 reader in
            let* axis1 = Binary.Reader.u16 reader in
            Ok (Ir.Movement.Transpose { axis0; axis1 })
        | 3 ->
            Binary.Reader.u16 reader
            |> Result.map (fun axis -> Ir.Movement.Unsqueeze axis)
        | 4 -> Ok Ir.Movement.Expand
        | 5 -> Ok Ir.Movement.Contiguous
        | 6 ->
            read_index reader
            |> Result.map (fun index -> Ir.Movement.Index index)
        | 7 ->
            Binary.Reader.u16 reader
            |> Result.map (fun axis -> Ir.Movement.Concat { axis })
        | 8 ->
            let* axis = Binary.Reader.u16 reader in
            let* shift = Binary.Reader.i64 reader in
            Ok (Ir.Movement.Roll { axis; shift })
        | _ -> Error (Printf.sprintf "unknown movement tag: %d" movement_tag)
      in
      Ok (Ir.Primitive.Movement movement)
  | 4 ->
      let* stride = Binary.Reader.u32 reader in
      let* padding = Binary.Reader.u32 reader in
      let* dilation = Binary.Reader.u32 reader in
      let* groups = Binary.Reader.u32 reader in
      Ir.Short_conv.create ~stride ~padding ~dilation ~groups
      |> Result.map (fun config -> Ir.Primitive.Short_conv config)
  | 5 ->
      let* scale = Binary.Reader.float64 reader in
      let* causal = Binary.Reader.bool reader in
      Ir.Attention.create ~scale ~causal
      |> Result.map (fun config -> Ir.Primitive.Attention config)
  | 6 -> Ok Ir.Primitive.Embedding
  | 7 ->
      let* start = Binary.Reader.i64 reader in
      let* stop = Binary.Reader.i64 reader in
      let* step = Binary.Reader.i64 reader in
      Ir.Arange.create ~start ~stop ~step
      |> Result.map (fun config -> Ir.Primitive.Arange config)
  | 8 ->
      let* axis = Binary.Reader.u16 reader in
      Ir.Diff.create ~axis |> Result.map (fun config -> Ir.Primitive.Diff config)
  | 9 ->
      let* axis = Binary.Reader.u16 reader in
      Ir.Cumsum.create ~axis
      |> Result.map (fun config -> Ir.Primitive.Cumsum config)
  | 10 -> read_scalar reader |> Result.map (fun scalar -> Ir.Primitive.Fill scalar)
  | 11 -> Ok Ir.Primitive.Gather2
  | 12 ->
      let* count = Binary.Reader.u16 reader in
      let rec axes acc remaining =
        if remaining = 0 then Ok (List.rev acc)
        else
          let* axis = Binary.Reader.u16 reader in
          axes (axis :: acc) (remaining - 1)
      in
      let* axes = axes [] count in
      let* keepdim = Binary.Reader.bool reader in
      Ok
        (Ir.Primitive.Reduce
           { Ir.Reduction.operator = Sum; axes; keepdim })
  | 13 ->
      read_index reader |> Result.map (fun index -> Ir.Primitive.Update_slice index)
  | 14 ->
      let* scale = Binary.Reader.float64 reader in
      let* cache_layer = Binary.Reader.u32 reader in
      let* kv_heads = Binary.Reader.u32 reader in
      let* head_dim = Binary.Reader.u32 reader in
      let* token_elements = Binary.Reader.u32 reader in
      let* key_offset = Binary.Reader.u32 reader in
      let* value_offset = Binary.Reader.u32 reader in
      let* group_size = Binary.Reader.u32 reader in
      let* token_stride = Binary.Reader.u32 reader in
      let* encoded_window = Binary.Reader.u32 reader in
      let window = if encoded_window = 0 then None else Some encoded_window in
      Ir.Paged_attention_q8.create ~window ~scale ~cache_layer ~kv_heads ~head_dim
        ~token_elements ~key_offset ~value_offset ~group_size ~token_stride
      |> Result.map (fun config -> Ir.Primitive.Paged_attention_q8 config)
  | 15 ->
      Binary.Reader.u16 reader
      |> Result.map (fun axis -> Ir.Primitive.Pad_right_zero { axis })
  | 16 ->
      let* upper = Binary.Reader.bool reader in
      let* diagonal = Binary.Reader.i64 reader in
      Ok (Ir.Primitive.Triangular { upper; diagonal })
  | 17 ->
      read_scalar reader
      |> Result.map (fun scalar -> Ir.Primitive.Masked_fill scalar)
  | 18 -> Ok Ir.Primitive.Eye
  | 19 -> Ok Ir.Primitive.Batched_matmul
  | 20 ->
      let* axis = Binary.Reader.u16 reader in
      let* start = Binary.Reader.u16 reader in
      let* stop = Binary.Reader.u16 reader in
      Ir.Triangular_recurrence.create ~axis ~start ~stop
      |> Result.map (fun config -> Ir.Primitive.Triangular_recurrence config)
  | 21 -> Ok Ir.Primitive.Gated_delta
  | 22 ->
      let* epsilon = Binary.Reader.float64 reader in
      if Float.is_finite epsilon && epsilon >= 0.0 then
        Ok (Ir.Primitive.L2_norm { epsilon })
      else Error "invalid L2 normalization epsilon"
  | 23 ->
      let* stride = Binary.Reader.u32 reader in
      let* padding = Binary.Reader.u32 reader in
      let* dilation = Binary.Reader.u32 reader in
      let* groups = Binary.Reader.u32 reader in
      let* output_start = Binary.Reader.u32 reader in
      let* convolution =
        Ir.Short_conv.create ~stride ~padding ~dilation ~groups
      in
      Ir.Short_conv_silu.create ~convolution ~output_start
      |> Result.map (fun config -> Ir.Primitive.Short_conv_silu config)
  | 24 ->
      let* epsilon = Binary.Reader.float64 reader in
      let* offset = Binary.Reader.u32 reader in
      Ir.L2_norm_slice.create ~epsilon ~offset
      |> Result.map (fun config -> Ir.Primitive.L2_norm_slice config)
  | 25 ->
      let* count = Binary.Reader.u16 reader in
      let rec axes acc remaining =
        if remaining = 0 then Ok (List.rev acc)
        else
          let* axis = Binary.Reader.u16 reader in
          axes (axis :: acc) (remaining - 1)
      in
      let* axes = axes [] count in
      let* keepdim = Binary.Reader.bool reader in
      Ok
        (Ir.Primitive.Reduce
           { Ir.Reduction.operator = Argmax; axes; keepdim })
  | _ -> Error (Printf.sprintf "unknown primitive tag: %d" tag)

let write_op writer = function
  | Ir.Op.Input { name; source } ->
      Binary.Writer.u8 writer 0;
      Binary.Writer.string writer name;
      (match source with
      | Ir.Input_source.Runtime -> Binary.Writer.u8 writer 0
      | Ir.Input_source.Tensor_store { key } ->
          Binary.Writer.u8 writer 1;
          Binary.Writer.string writer key)
  | Ir.Op.Alloc { space; layout } ->
      Binary.Writer.u8 writer 1;
      write_memory_space writer space;
      write_layout writer layout
  | Ir.Op.Copy { asynchronous; barrier } ->
      Binary.Writer.u8 writer 2;
      Binary.Writer.bool writer asynchronous;
      write_option writer Binary.Writer.i64 barrier
  | Ir.Op.Matmul { m; n; k } ->
      Binary.Writer.u8 writer 3;
      List.iter (Binary.Writer.u64 writer) [ m; n; k ]
  | Ir.Op.Linear { m; n; k; bias } ->
      Binary.Writer.u8 writer 4;
      List.iter (Binary.Writer.u64 writer) [ m; n; k ];
      Binary.Writer.bool writer bias
  | Ir.Op.Linear_add { m; n; k } ->
      Binary.Writer.u8 writer 40;
      List.iter (Binary.Writer.u64 writer) [ m; n; k ]
  | Ir.Op.Add { broadcast } ->
      Binary.Writer.u8 writer 5;
      Binary.Writer.u8 writer (match broadcast with Shape.Same -> 0 | Shape.Row -> 1)
  | Ir.Op.Gelu -> Binary.Writer.u8 writer 6
  | Ir.Op.Relu -> Binary.Writer.u8 writer 7
  | Ir.Op.Rms_norm { epsilon } ->
      Binary.Writer.u8 writer 16;
      Binary.Writer.float64 writer epsilon
  | Ir.Op.Rms_norm_add { epsilon } ->
      Binary.Writer.u8 writer 37;
      Binary.Writer.float64 writer epsilon
  | Ir.Op.Gated_linear { m; n; k; activation } ->
      Binary.Writer.u8 writer 38;
      List.iter (Binary.Writer.u64 writer) [ m; n; k ];
      Binary.Writer.u8 writer
        (match activation with
        | Ir.Gated_activation.Silu -> 0
        | Ir.Gated_activation.Gelu -> 1
        | Ir.Gated_activation.Sigmoid -> 2)
  | Ir.Op.Primitive primitive ->
      Binary.Writer.u8 writer 15;
      write_primitive writer primitive
  | Ir.Op.Opaque { op; target; reason; arguments; keyword_arguments } ->
      Binary.Writer.u8 writer
        (if String.trim reason = "" then 8 else 43);
      Binary.Writer.string writer op;
      Binary.Writer.string writer target;
      if String.trim reason <> "" then Binary.Writer.string writer reason;
      write_arguments writer arguments;
      write_named_arguments writer keyword_arguments
  | Ir.Op.Output { name } ->
      Binary.Writer.u8 writer 9;
      Binary.Writer.string writer name
  | Ir.Op.Barrier_create { id; name } ->
      Binary.Writer.u8 writer 10;
      Binary.Writer.i64 writer id;
      Binary.Writer.string writer name
  | Ir.Op.Barrier_arrive id ->
      Binary.Writer.u8 writer 11;
      Binary.Writer.i64 writer id
  | Ir.Op.Barrier_wait id ->
      Binary.Writer.u8 writer 12;
      Binary.Writer.i64 writer id
  | Ir.Op.Fused_matmul_bias { m; n; k } ->
      Binary.Writer.u8 writer 13;
      List.iter (Binary.Writer.u64 writer) [ m; n; k ]
  | Ir.Op.W4a16_linear { m; n; k; bias } ->
      Binary.Writer.u8 writer 34;
      List.iter (Binary.Writer.u64 writer) [ m; n; k ];
      Binary.Writer.bool writer bias
  | Ir.Op.W4a16_swiglu_ffn { m; n; k; epsilon } ->
      Binary.Writer.u8 writer 35;
      List.iter (Binary.Writer.u64 writer) [ m; n; k ];
      Binary.Writer.float64 writer epsilon
  | Ir.Op.W4a16_qkv_linear { m; k; n_q; n_k; n_v; extra_outputs } ->
      Binary.Writer.u8 writer 36;
      List.iter (Binary.Writer.u64 writer) [ m; k; n_q; n_k; n_v ];
      Binary.Writer.u8 writer (List.length extra_outputs);
      List.iter (write_value writer) extra_outputs
  | Ir.Op.Quant_qkv_linear { m; k; n_q; n_k; n_v; extra_outputs } ->
      Binary.Writer.u8 writer 41;
      List.iter (Binary.Writer.u64 writer) [ m; k; n_q; n_k; n_v ];
      Binary.Writer.u8 writer (List.length extra_outputs);
      List.iter (write_value writer) extra_outputs
  | Ir.Op.W4a16_lm_head_argmax { m; n; k; epsilon; extra_outputs } ->
      Binary.Writer.u8 writer (if extra_outputs = [] then 29 else 33);
      List.iter (Binary.Writer.u64 writer) [ m; n; k ];
      Binary.Writer.float64 writer epsilon;
      if extra_outputs <> [] then begin
        Binary.Writer.u8 writer (List.length extra_outputs);
        List.iter (write_value writer) extra_outputs
      end
  | Ir.Op.Rms_rope config ->
      Binary.Writer.u8 writer 20;
      Binary.Writer.float64 writer (Ir.Rms_rope.epsilon config);
      Binary.Writer.u64 writer (Ir.Rms_rope.half_dimension config)
  | Ir.Op.Rms_rope_qk
      { q_heads; k_heads; width; half_dimension; epsilon; extra_outputs } ->
      Binary.Writer.u8 writer 26;
      Binary.Writer.u64 writer q_heads;
      Binary.Writer.u64 writer k_heads;
      Binary.Writer.u64 writer width;
      Binary.Writer.u64 writer half_dimension;
      Binary.Writer.float64 writer epsilon;
      Binary.Writer.u8 writer (List.length extra_outputs);
      List.iter (write_value writer) extra_outputs
  | Ir.Op.Rotary_qk
      { q_heads; k_heads; width; half_dimension; extra_outputs } ->
      Binary.Writer.u8 writer 39;
      Binary.Writer.u64 writer q_heads;
      Binary.Writer.u64 writer k_heads;
      Binary.Writer.u64 writer width;
      Binary.Writer.u64 writer half_dimension;
      Binary.Writer.u8 writer (List.length extra_outputs);
      List.iter (write_value writer) extra_outputs
  | Ir.Op.Short_conv_step config ->
      Binary.Writer.u8 writer 21;
      Binary.Writer.u64 writer (Ir.Short_conv_step.channels config);
      Binary.Writer.u64 writer (Ir.Short_conv_step.window config)
  | Ir.Op.Short_conv_step_fused config ->
      Binary.Writer.u8 writer 25;
      Binary.Writer.u64 writer (Ir.Short_conv_step.channels config);
      Binary.Writer.u64 writer (Ir.Short_conv_step.window config)
  | Ir.Op.Short_conv_prefill config ->
      Binary.Writer.u8 writer 22;
      Binary.Writer.u64 writer (Ir.Short_conv_prefill.channels config);
      Binary.Writer.u64 writer (Ir.Short_conv_prefill.window config)
  | Ir.Op.Fused_region { extra_outputs } ->
      Binary.Writer.u8 writer 42;
      Binary.Writer.u8 writer (List.length extra_outputs);
      List.iter (write_value writer) extra_outputs

let read_three_dimensions reader =
  let* m = Binary.Reader.u64 reader in
  let* n = Binary.Reader.u64 reader in
  let* k = Binary.Reader.u64 reader in
  Ok (m, n, k)

let read_op values reader =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 ->
      let* name = Binary.Reader.string reader in
      let* source_tag = Binary.Reader.u8 reader in
      let* source =
        match source_tag with
        | 0 -> Ok Ir.Input_source.Runtime
        | 1 ->
            let* key = Binary.Reader.string reader in
            Ok (Ir.Input_source.Tensor_store { key })
        | _ ->
            Error (Printf.sprintf "unknown input source tag: %d" source_tag)
      in
      Ok (Ir.Op.Input { name; source })
  | 1 ->
      let* space = read_memory_space reader in
      let* layout = read_layout reader in
      Ok (Ir.Op.Alloc { space; layout })
  | 2 ->
      let* asynchronous = Binary.Reader.bool reader in
      let* barrier = read_option reader Binary.Reader.i64 in
      Ok (Ir.Op.Copy { asynchronous; barrier })
  | 3 ->
      let* m, n, k = read_three_dimensions reader in
      Ok (Ir.Op.Matmul { m; n; k })
  | 4 ->
      let* m, n, k = read_three_dimensions reader in
      let* bias = Binary.Reader.bool reader in
      Ok (Ir.Op.Linear { m; n; k; bias })
  | 40 ->
      let* m, n, k = read_three_dimensions reader in
      if m > 0 && n > 0 && k > 0 then Ok (Ir.Op.Linear_add { m; n; k })
      else Error "schedule contains invalid Linear-add dimensions"
  | 5 ->
      let* broadcast_tag = Binary.Reader.u8 reader in
      let* broadcast =
        match broadcast_tag with
        | 0 -> Ok Shape.Same
        | 1 -> Ok Shape.Row
        | _ -> Error (Printf.sprintf "unknown broadcast tag: %d" broadcast_tag)
      in
      Ok (Ir.Op.Add { broadcast })
  | 6 -> Ok Ir.Op.Gelu
  | 7 -> Ok Ir.Op.Relu
  | 8 ->
      let* op = Binary.Reader.string reader in
      let* target = Binary.Reader.string reader in
      let* arguments = read_arguments ~depth:0 values reader in
      let* keyword_arguments = read_named_arguments values reader in
      Ok
        (Ir.Op.Opaque
           { op; target; reason = "legacy opaque operation (diagnostic unavailable)";
             arguments; keyword_arguments })
  | 43 ->
      let* op = Binary.Reader.string reader in
      let* target = Binary.Reader.string reader in
      let* reason = Binary.Reader.string reader in
      let* arguments = read_arguments ~depth:0 values reader in
      let* keyword_arguments = read_named_arguments values reader in
      Ok (Ir.Op.Opaque { op; target; reason; arguments; keyword_arguments })
  | 9 -> Binary.Reader.string reader |> Result.map (fun name -> Ir.Op.Output { name })
  | 10 ->
      let* id = Binary.Reader.i64 reader in
      let* name = Binary.Reader.string reader in
      Ok (Ir.Op.Barrier_create { id; name })
  | 11 -> Binary.Reader.i64 reader |> Result.map (fun id -> Ir.Op.Barrier_arrive id)
  | 12 -> Binary.Reader.i64 reader |> Result.map (fun id -> Ir.Op.Barrier_wait id)
  | 13 ->
      let* m, n, k = read_three_dimensions reader in
      Ok (Ir.Op.Fused_matmul_bias { m; n; k })
  | 34 ->
      let* m, n, k = read_three_dimensions reader in
      let* bias = Binary.Reader.bool reader in
      Ok (Ir.Op.W4a16_linear { m; n; k; bias })
  | 35 ->
      let* m, n, k = read_three_dimensions reader in
      let* epsilon = Binary.Reader.float64 reader in
      if Float.is_finite epsilon && epsilon > 0.0 then
        Ok (Ir.Op.W4a16_swiglu_ffn { m; n; k; epsilon })
      else Error "schedule contains an invalid W4A16 SwiGLU epsilon"
  | 36 ->
      let* m = Binary.Reader.u64 reader in
      let* k = Binary.Reader.u64 reader in
      let* n_q = Binary.Reader.u64 reader in
      let* n_k = Binary.Reader.u64 reader in
      let* n_v = Binary.Reader.u64 reader in
      let* count = Binary.Reader.u8 reader in
      let rec outputs acc remaining =
        if remaining = 0 then Ok (List.rev acc)
        else
          let* output = read_value reader in
          outputs (output :: acc) (remaining - 1)
      in
      let* extra_outputs = outputs [] count in
      Ok (Ir.Op.W4a16_qkv_linear { m; k; n_q; n_k; n_v; extra_outputs })
  | 41 ->
      let* m = Binary.Reader.u64 reader in
      let* k = Binary.Reader.u64 reader in
      let* n_q = Binary.Reader.u64 reader in
      let* n_k = Binary.Reader.u64 reader in
      let* n_v = Binary.Reader.u64 reader in
      let* count = Binary.Reader.u8 reader in
      let rec outputs acc remaining =
        if remaining = 0 then Ok (List.rev acc)
        else
          let* output = read_value reader in
          outputs (output :: acc) (remaining - 1)
      in
      let* extra_outputs = outputs [] count in
      Ok (Ir.Op.Quant_qkv_linear { m; k; n_q; n_k; n_v; extra_outputs })
  | 15 -> read_primitive values reader |> Result.map (fun value -> Ir.Op.Primitive value)
  | 16 ->
      let* epsilon = Binary.Reader.float64 reader in
      if Float.is_finite epsilon then Ok (Ir.Op.Rms_norm { epsilon })
      else Error "schedule contains a non-finite RMSNorm epsilon"
  | 37 ->
      let* epsilon = Binary.Reader.float64 reader in
      if Float.is_finite epsilon then Ok (Ir.Op.Rms_norm_add { epsilon })
      else Error "schedule contains a non-finite RMSNorm-add epsilon"
  | 38 ->
      let* m, n, k = read_three_dimensions reader in
      let* activation_tag = Binary.Reader.u8 reader in
      let* activation =
        match activation_tag with
        | 0 -> Ok Ir.Gated_activation.Silu
        | 1 -> Ok Ir.Gated_activation.Gelu
        | 2 -> Ok Ir.Gated_activation.Sigmoid
        | _ -> Error "schedule contains an invalid gated activation"
      in
      if m > 0 && n > 0 && k > 0 then
        Ok (Ir.Op.Gated_linear { m; n; k; activation })
      else Error "schedule contains invalid gated Linear dimensions"
  | 29 ->
      let* m, n, k = read_three_dimensions reader in
      let* epsilon = Binary.Reader.float64 reader in
      if Float.is_finite epsilon && epsilon > 0.0 then
        Ok (Ir.Op.W4a16_lm_head_argmax { m; n; k; epsilon; extra_outputs = [] })
      else Error "schedule contains a non-finite LM-head argmax epsilon"
  | 33 ->
      let* m, n, k = read_three_dimensions reader in
      let* epsilon = Binary.Reader.float64 reader in
      let* count = Binary.Reader.u8 reader in
      let rec outputs acc remaining =
        if remaining = 0 then Ok (List.rev acc)
        else
          let* output = read_value reader in
          outputs (output :: acc) (remaining - 1)
      in
      let* extra_outputs = outputs [] count in
      if Float.is_finite epsilon && epsilon > 0.0 && count = 1 then
        Ok (Ir.Op.W4a16_lm_head_argmax { m; n; k; epsilon; extra_outputs })
      else Error "schedule contains invalid LM-head argmax secondary outputs"
  | 20 ->
      let* epsilon = Binary.Reader.float64 reader in
      let* half_dimension = Binary.Reader.u64 reader in
      Ir.Rms_rope.create ~epsilon ~half_dimension
      |> Result.map (fun config -> Ir.Op.Rms_rope config)
  | 21 ->
      let* channels = Binary.Reader.u64 reader in
      let* window = Binary.Reader.u64 reader in
      Ir.Short_conv_step.create ~channels ~window
      |> Result.map (fun config -> Ir.Op.Short_conv_step config)
  | 22 ->
      let* channels = Binary.Reader.u64 reader in
      let* window = Binary.Reader.u64 reader in
      Ir.Short_conv_prefill.create ~channels ~window
      |> Result.map (fun config -> Ir.Op.Short_conv_prefill config)
  | 25 ->
      let* channels = Binary.Reader.u64 reader in
      let* window = Binary.Reader.u64 reader in
      Ir.Short_conv_step.create ~channels ~window
      |> Result.map (fun config -> Ir.Op.Short_conv_step_fused config)
  | 26 ->
      let* q_heads = Binary.Reader.u64 reader in
      let* k_heads = Binary.Reader.u64 reader in
      let* width = Binary.Reader.u64 reader in
      let* half_dimension = Binary.Reader.u64 reader in
      let* epsilon = Binary.Reader.float64 reader in
      let* count = Binary.Reader.u8 reader in
      let rec outputs acc remaining =
        if remaining = 0 then Ok (List.rev acc)
        else
          let* output = read_value reader in
          outputs (output :: acc) (remaining - 1)
      in
      let* extra_outputs = outputs [] count in
      Ok (Ir.Op.Rms_rope_qk { q_heads; k_heads; width; half_dimension; epsilon; extra_outputs })
  | 39 ->
      let* q_heads = Binary.Reader.u64 reader in
      let* k_heads = Binary.Reader.u64 reader in
      let* width = Binary.Reader.u64 reader in
      let* half_dimension = Binary.Reader.u64 reader in
      let* count = Binary.Reader.u8 reader in
      let rec outputs acc remaining =
        if remaining = 0 then Ok (List.rev acc)
        else
          let* output = read_value reader in
          outputs (output :: acc) (remaining - 1)
      in
      let* extra_outputs = outputs [] count in
      if count = 1 then
        Ok
          (Ir.Op.Rotary_qk
             { q_heads; k_heads; width; half_dimension; extra_outputs })
      else Error "schedule contains invalid rotary-QK secondary outputs"
  | 42 ->
      let* count = Binary.Reader.u8 reader in
      let rec outputs acc remaining =
        if remaining = 0 then Ok (List.rev acc)
        else
          let* output = read_value reader in
          outputs (output :: acc) (remaining - 1)
      in
      let* extra_outputs = outputs [] count in
      Ok (Ir.Op.Fused_region { extra_outputs })
  | _ -> Error (Printf.sprintf "unknown schedule opcode: %d" tag)

let magic = "LLMOSCH\000"

let to_bytes schedule =
  let writer = Binary.Writer.create () in
  Binary.Writer.raw_string writer magic;
  Binary.Writer.u16 writer 29;
  Binary.Writer.u32 writer (List.length schedule.commands);
  List.iter
    (fun command ->
      Binary.Writer.u32 writer command.Command.node_id;
      Binary.Writer.u16 writer (List.length command.Command.inputs);
      List.iter
        (fun value -> Binary.Writer.u32 writer (value_id value))
        command.Command.inputs;
      write_option writer write_value command.Command.output;
      write_op writer command.Command.op;
      write_option writer write_fusion_region command.Command.fusion_region)
    schedule.commands;
  Binary.Writer.contents writer

let of_bytes bytes =
  let reader = Binary.Reader.create bytes in
  let* actual_magic = Binary.Reader.raw_string reader ~length:(String.length magic) in
  if actual_magic <> magic then Error "invalid serving schedule magic"
  else
    let* version = Binary.Reader.u16 reader in
    if version <> 29 then
      Error (Printf.sprintf "unsupported serving schedule version: %d" version)


    else
      let* count = Binary.Reader.u32 reader in
      let values = Hashtbl.create count in
      let seen_values = ref Int_set.empty in
      let rec read_values acc remaining =
        if remaining = 0 then Ok (List.rev acc)
        else
          let* node_id = Binary.Reader.u32 reader in
          let* input_count = Binary.Reader.u16 reader in
          let rec inputs acc remaining =
            if remaining = 0 then Ok (List.rev acc)
            else
              let* id = Binary.Reader.u32 reader in
              let* value = find_value values id in
              inputs (value :: acc) (remaining - 1)
          in
          let* inputs = inputs [] input_count in
          let* output = read_option reader read_value in
          let* op = read_op values reader in
          List.iter
            (fun value -> Hashtbl.replace values (value_id value) value)
            (Option.to_list output @ Ir.Op.additional_outputs op);
          let* fusion_region =
            read_option reader (read_fusion_region values)
          in
          let command = { Command.node_id; op; inputs; output; fusion_region } in
          let* () = validate_command !seen_values command
          in
          List.iter
            (fun value ->
              let id = value_id value in
              Hashtbl.add values id value;
              seen_values := Int_set.add id !seen_values)
            (Option.to_list output @ Ir.Op.additional_outputs op);
          read_values (command :: acc) (remaining - 1)
      in
      let* commands = read_values [] count in
      let* () = Binary.Reader.finish reader in
      create commands
