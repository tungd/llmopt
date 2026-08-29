module Slot = struct
  type t = int

  let of_int n =
    if n < 0 then Error "kernel IR slots must be non-negative" else Ok n

  let of_int_exn n =
    match of_int n with Ok n -> n | Error message -> invalid_arg message

  let to_int n = n
  let compare (left : t) right = Int.compare left right
  let equal left right = left = right
end

module Primitive = struct
  type unary = Silu
  type binary = Mul | Add
  type w4a16_linear = { m : int; n : int; k : int; bias : bool }

  type t =
    | Rms_norm of { epsilon : float }
    | W4a16_linear of w4a16_linear
    | Unary of unary
    | Binary of binary

  let rms_norm ~epsilon = Rms_norm { epsilon }
  let w4a16_linear ~m ~n ~k ~bias = W4a16_linear { m; n; k; bias }
  let unary operation = Unary operation
  let binary operation = Binary operation

  let to_string = function
    | Rms_norm { epsilon } -> Printf.sprintf "rms-norm(eps=%.9g)" epsilon
    | W4a16_linear { m; n; k; bias = false } ->
        Printf.sprintf "w4a16-linear-g64[%dx%dx%d]" m n k
    | W4a16_linear { m; n; k; bias = true } ->
        Printf.sprintf "w4a16-linear-g64+bias[%dx%dx%d]" m n k
    | Unary Silu -> "silu"
    | Binary Mul -> "mul"
    | Binary Add -> "add"
end

module Effect = struct
  type mode = Read | Write | Read_write
  type alias = Distinct | May_alias | Alias_of of Slot.t
  type access = { slot : Slot.t; mode : mode; alias : alias }

  type t = {
    accesses : access list;
    state_inputs : Slot.t list;
    state_outputs : Slot.t list;
    barriers : int list;
  }

  let empty =
    { accesses = []; state_inputs = []; state_outputs = []; barriers = [] }

  let duplicate xs =
    let rec loop seen = function
      | [] -> None
      | x :: rest ->
          if List.mem x seen then Some x else loop (x :: seen) rest
    in
    loop [] xs

  let create ~accesses ?(state_inputs = []) ?(state_outputs = [])
      ?(barriers = []) () =
    let negative_access =
      List.find_opt (fun access -> Slot.to_int access.slot < 0) accesses
    in
    match negative_access with
    | Some access ->
        Error
          (Printf.sprintf "negative effect slot %d"
             (Slot.to_int access.slot))
    | None -> (
        match duplicate state_inputs with
        | Some slot ->
            Error
              (Printf.sprintf "duplicate effect state input slot %d"
                 (Slot.to_int slot))
        | None -> (
            match duplicate state_outputs with
            | Some slot ->
                Error
                  (Printf.sprintf "duplicate effect state output slot %d"
                     (Slot.to_int slot))
            | None when List.exists (fun id -> id < 0) barriers ->
                Error "effect barrier IDs must be non-negative"
            | None -> Ok { accesses; state_inputs; state_outputs; barriers }))

  let accesses effects = effects.accesses
  let state_inputs effects = effects.state_inputs
  let state_outputs effects = effects.state_outputs
  let barriers effects = effects.barriers
end

module Resource = struct
  type t = {
    scalar_ops : int64;
    bytes_read : int64;
    bytes_written : int64;
    temporary_bytes : int64;
    synchronization_points : int;
  }

  let zero =
    {
      scalar_ops = 0L;
      bytes_read = 0L;
      bytes_written = 0L;
      temporary_bytes = 0L;
      synchronization_points = 0;
    }

  let create ~scalar_ops ~bytes_read ~bytes_written ~temporary_bytes
      ~synchronization_points =
    if scalar_ops < 0L then Error "resource scalar_ops must be non-negative"
    else if bytes_read < 0L then Error "resource bytes_read must be non-negative"
    else if bytes_written < 0L then
      Error "resource bytes_written must be non-negative"
    else if temporary_bytes < 0L then
      Error "resource temporary_bytes must be non-negative"
    else if synchronization_points < 0 then
      Error "resource synchronization points must be non-negative"
    else
      Ok
        {
          scalar_ops;
          bytes_read;
          bytes_written;
          temporary_bytes;
          synchronization_points;
        }

  let scalar_ops resource = resource.scalar_ops
  let bytes_read resource = resource.bytes_read
  let bytes_written resource = resource.bytes_written
  let temporary_bytes resource = resource.temporary_bytes
  let synchronization_points resource = resource.synchronization_points
end

type input = { slot : Slot.t; value : Ir.Value.t }

type output_spec = {
  slot : Slot.t;
  shape : Tensor_shape.t;
  dtype : Ir.Dtype.t;
}

type binding = {
  outputs : output_spec list;
  primitive : Primitive.t;
  inputs : Slot.t list;
}

let make_binding ~outputs ~primitive ~inputs =
  let output_slots = List.map (fun output -> output.slot) outputs in
  let duplicate slots =
    let rec loop seen = function
      | [] -> None
      | slot :: rest ->
          if List.exists (Slot.equal slot) seen then Some slot
          else loop (slot :: seen) rest
    in
    loop [] slots
  in
  if outputs = [] then Error "binding must have at least one output"
  else if List.exists (fun slot -> Slot.to_int slot < 0) inputs then
    Error "binding input slots must be non-negative"
  else if List.exists (fun output -> Slot.to_int output.slot < 0) outputs then
    Error "binding output slots must be non-negative"
  else
    match duplicate output_slots with
    | Some slot ->
        Error
          (Printf.sprintf "binding output slot %d is duplicated"
             (Slot.to_int slot))
    | None when List.exists (fun slot -> List.mem slot inputs) output_slots ->
        Error "binding output cannot be an input"
    | None -> Ok { outputs; primitive; inputs }

let binding_outputs binding = binding.outputs
let binding_inputs binding = binding.inputs
let binding_primitive binding = binding.primitive
let output_slot output = output.slot
let output_shape output = output.shape
let output_dtype output = output.dtype

type storage = Fresh | Alias_input of Slot.t | In_place_input of Slot.t
type result = { slot : Slot.t; value : Ir.Value.t; storage : storage }

type t = {
  name : string;
  member_node_ids : int list;
  inputs : input list;
  bindings : binding list;
  results : result list;
  effects : Effect.t;
  resource : Resource.t;
}

type error =
  | Empty_name
  | Invalid_member_node_id of int
  | Duplicate_member_node_id of int
  | Duplicate_slot of Slot.t
  | Unknown_slot of Slot.t
  | Forward_reference of Slot.t
  | Invalid_binding of string
  | Invalid_result of string
  | Invalid_effect of string

module Slot_map = Map.Make (Int)
type metadata = Tensor_shape.t * Ir.Dtype.t

let is_float = function
  | Ir.Dtype.Float32 | Ir.Dtype.Float16 | Ir.Dtype.Bfloat16 -> true
  | Ir.Dtype.Int64 | Ir.Dtype.Int32 | Ir.Dtype.Int8 | Ir.Dtype.UInt8
  | Ir.Dtype.Bool | Ir.Dtype.Quant _ -> false

let dims shape = Tensor_shape.dimensions shape
let numel shape = Tensor_shape.numel shape
let same_shape left right = Tensor_shape.equal left right

let output_metadata ({ shape; dtype; _ } : output_spec) = (shape, dtype)

let metadata inputs bindings =
  let map =
    List.fold_left
      (fun map ({ slot; value } : input) ->
        Slot_map.add slot (Ir.Value.logical_shape value, Ir.Value.dtype value) map)
      Slot_map.empty inputs
  in
  List.fold_left
    (fun map (binding : binding) ->
      List.fold_left
        (fun map (output : output_spec) ->
          Slot_map.add output.slot (output_metadata output) map)
        map binding.outputs)
    map bindings

let one_output binding =
  match binding.outputs with
  | [ output ] -> Ok output
  | _ -> Error (Invalid_binding "primitive requires exactly one output")

let input_metadata map slot =
  match Slot_map.find_opt slot map with
  | Some metadata -> Ok metadata
  | None -> Error (Unknown_slot slot)

let expected_matrix_shape shape ~rows ~columns = dims shape = [ rows; columns ]

let validate_primitive map binding =
  let invalid message = Error (Invalid_binding message) in
  let one = function [ x ] -> Ok x | _ -> invalid "expected one input slot" in
  let two = function [ x; y ] -> Ok (x, y) | _ -> invalid "expected two input slots" in
  match binding.primitive with
  | Primitive.Rms_norm { epsilon } -> (
      if not (Float.is_finite epsilon) || epsilon <= 0.0 then
        invalid "RMSNorm epsilon must be finite and positive"
      else
        match one_output binding, two binding.inputs with
        | Error error, _ | _, Error error -> Error error
        | Ok output, Ok (input, weight) -> (
            match input_metadata map input, input_metadata map weight with
            | Error error, _ | _, Error error -> Error error
            | Ok (input_shape, input_dtype), Ok (weight_shape, weight_dtype) ->
                let last =
                  match List.rev (dims input_shape) with
                  | last :: _ -> Some last
                  | [] -> None
                in
                if
                  is_float input_dtype && is_float weight_dtype
                  && Option.exists
                       (fun last -> dims weight_shape = [ last ]) last
                  && same_shape input_shape output.shape
                  && is_float output.dtype
                then Ok ()
                else invalid "RMSNorm metadata is inconsistent"))
  | Primitive.W4a16_linear { m; n; k; bias } -> (
      if m <= 0 || n <= 0 || k <= 0 then
        invalid "W4A16 dimensions must be positive"
      else if k mod 64 <> 0 then
        invalid
          "W4A16 input dimension must be divisible by the fixed group size 64"
      else
        match one_output binding, binding.inputs with
        | Error error, _ -> Error error
        | Ok _, _ when List.length binding.inputs <> (if bias then 4 else 3) ->
            invalid
              (Printf.sprintf
                 "W4A16 linear expects activation, packed weight, scale%s"
                 (if bias then ", and bias" else ""))
        | Ok output, activation :: weight :: scale :: rest -> (
            match
              input_metadata map activation,
              input_metadata map weight,
              input_metadata map scale
            with
            | Error error, _, _ | _, Error error, _ | _, _, Error error ->
                Error error
            | Ok (activation_shape, activation_dtype),
              Ok (weight_shape, weight_dtype),
              Ok (scale_shape, scale_dtype) ->
                let packed_k = (k + 1) / 2 in
                let groups = (k + 63) / 64 in
                let activation_ok =
                  activation_dtype = Ir.Dtype.Float16
                  && numel activation_shape = m * k
                  && (match List.rev (dims activation_shape) with
                     | last :: _ -> last = k
                     | [] -> false)
                in
                let bias_ok =
                  match rest with
                  | [] -> not bias
                  | [ bias_slot ] when bias -> (
                      match input_metadata map bias_slot with
                      | Ok (bias_shape, bias_dtype) ->
                          bias_dtype = Ir.Dtype.Float16
                          &&
                          (dims bias_shape = [ n ]
                          || dims bias_shape = [ 1; n ])
                      | Error _ -> false)
                  | _ -> false
                in
                if
                  activation_ok
                  && weight_dtype = Ir.Dtype.UInt8
                  && expected_matrix_shape weight_shape ~rows:n ~columns:packed_k
                  && scale_dtype = Ir.Dtype.Float16
                  && dims scale_shape = [ n; groups ]
                  && bias_ok
                  && numel output.shape = m * n
                  && (match List.rev (dims output.shape) with
                     | last :: _ -> last = n
                     | [] -> false)
                  && output.dtype = Ir.Dtype.Float16
                then Ok ()
                else invalid "W4A16 metadata is inconsistent")
        | Ok _, _ -> invalid "W4A16 linear expects activation, packed weight, and scale")
  | Primitive.Unary Primitive.Silu -> (
      match one_output binding, one binding.inputs with
      | Error error, _ | _, Error error -> Error error
      | Ok output, Ok input -> (
          match input_metadata map input with
          | Error error -> Error error
          | Ok (shape, dtype)
            when dtype = Ir.Dtype.Float16 && is_float dtype
                 && same_shape shape output.shape && dtype = output.dtype ->
              Ok ()
          | Ok _ -> invalid "SiLU metadata is inconsistent"))
  | Primitive.Binary _ -> (
      match one_output binding, two binding.inputs with
      | Error error, _ | _, Error error -> Error error
      | Ok output, Ok (left, right) -> (
          match input_metadata map left, input_metadata map right with
          | Error error, _ | _, Error error -> Error error
          | Ok (left_shape, left_dtype), Ok (right_shape, right_dtype) ->
              if
                left_dtype = Ir.Dtype.Float16
                && right_dtype = left_dtype
                && same_shape left_shape right_shape
                && same_shape left_shape output.shape
                && output.dtype = left_dtype
              then Ok ()
              else invalid "binary metadata is inconsistent"))

let check_unique slots =
  let rec loop seen = function
    | [] -> Ok ()
    | slot :: rest ->
        if Slot_map.mem slot seen then Error (Duplicate_slot slot)
        else loop (Slot_map.add slot () seen) rest
  in
  loop Slot_map.empty slots

let validate_members ids =
  let rec loop seen = function
    | [] -> Ok ()
    | id :: rest ->
        if id < 0 then Error (Invalid_member_node_id id)
        else if List.mem id seen then Error (Duplicate_member_node_id id)
        else loop (id :: seen) rest
  in
  loop [] ids

let validate_effect available effects =
  let known slot = Slot_map.mem slot available in
  let check slots kind =
    match List.find_opt (fun slot -> not (known slot)) slots with
    | None -> Ok ()
    | Some slot ->
        Error
          (Invalid_effect
             (Printf.sprintf "%s references unknown slot %d" kind
                (Slot.to_int slot)))
  in
  let check_alias access =
    match access.Effect.alias with
    | Effect.Distinct | Effect.May_alias -> Ok ()
    | Effect.Alias_of target when known target -> Ok ()
    | Effect.Alias_of target ->
        Error
          (Invalid_effect
             (Printf.sprintf "effect alias references unknown slot %d"
                (Slot.to_int target)))
  in
  match
    List.find_map
      (fun access ->
        match
          check [ access.Effect.slot ] "effect access",
          check_alias access
        with
        | Ok (), Ok () -> None
        | Error error, _ | _, Error error -> Some error)
      (Effect.accesses effects)
  with
  | Some error -> Error error
  | None -> (
      match check (Effect.state_inputs effects) "state input" with
      | Error _ as error -> error
      | Ok () -> check (Effect.state_outputs effects) "state output")

let create ~name ~member_node_ids ~inputs ~bindings ~results ~effects ~resource =
  if String.trim name = "" then Error Empty_name
  else
    match validate_members member_node_ids with
    | Error _ as error -> error
    | Ok () ->
        let input_slots = List.map (fun ({ slot; _ } : input) -> slot) inputs in
        if List.exists (fun slot -> Slot.to_int slot < 0) input_slots then
          Error (Invalid_effect "input slots must be non-negative")
        else
          match check_unique input_slots with
          | Error _ as error -> error
          | Ok () ->
              let initial =
                List.fold_left
                  (fun map slot -> Slot_map.add slot () map)
                  Slot_map.empty input_slots
              in
              let rec validate_bindings defined = function
                | [] -> Ok defined
                | (binding : binding) :: rest ->
                    let output_slots =
                      List.map
                        (fun (output : output_spec) -> output.slot)
                        binding.outputs
                    in
                    (match
                       List.find_opt
                         (fun slot -> Slot_map.mem slot defined)
                         output_slots
                     with
                    | Some slot -> Error (Duplicate_slot slot)
                    | None -> (
                        match
                          List.find_opt
                            (fun slot -> not (Slot_map.mem slot defined))
                            binding.inputs
                        with
                        | Some slot -> Error (Forward_reference slot)
                        | None -> (
                            match
                              validate_primitive (metadata inputs bindings)
                                binding
                            with
                            | Error _ as error -> error
                            | Ok () ->
                                let defined =
                                  List.fold_left
                                    (fun defined slot ->
                                      Slot_map.add slot () defined)
                                    defined output_slots
                                in
                                validate_bindings defined rest)))
              in
              (match validate_bindings initial bindings with
              | Error _ as error -> error
              | Ok defined ->
                  let result_slots = List.map (fun result -> result.slot) results in
                  (match check_unique result_slots with
                  | Error _ as error -> error
                  | Ok () ->
                      let all_metadata = metadata inputs bindings in
                      let check_result ({ slot; value; storage } : result) =
                        if not (Slot_map.mem slot defined) then Error (Unknown_slot slot)
                        else
                          match Slot_map.find_opt slot all_metadata with
                          | Some (shape, dtype)
                            when same_shape shape (Ir.Value.logical_shape value)
                                 && dtype = Ir.Value.dtype value -> (
                              match storage with
                              | Fresh -> Ok ()
                              | Alias_input source | In_place_input source ->
                                  if not (List.mem source input_slots) then
                                    Error
                                      (Invalid_result
                                         "result storage must name an input slot")
                                  else
                                    (match Slot_map.find_opt source all_metadata with
                                    | Some (source_shape, source_dtype)
                                      when same_shape source_shape shape
                                           && source_dtype = dtype -> Ok ()
                                    | Some _ ->
                                        Error
                                          (Invalid_result
                                             "aliased result metadata differs from its input")
                                    | None -> Error (Unknown_slot source)))
                          | Some _ -> Error (Invalid_result "result metadata is inconsistent")
                          | None -> Error (Unknown_slot slot)
                      in
                      (match
                         List.find_map
                           (fun result ->
                             match check_result result with
                             | Ok () -> None
                             | Error error -> Some error)
                           results
                       with
                      | Some error -> Error error
                      | None -> (
                          match validate_effect defined effects with
                          | Error _ as error -> error
                          | Ok () ->
                              Ok
                                {
                                  name;
                                  member_node_ids;
                                  inputs;
                                  bindings;
                                  results;
                                  effects;
                                  resource;
                                }))))

let name region = region.name
let member_node_ids region = region.member_node_ids
let inputs region = region.inputs
let bindings region = region.bindings
let results region = region.results
let effects region = region.effects
let resource region = region.resource
let result_slot result = result.slot
let result_value result = result.value
let result_storage result = result.storage

let error_to_string = function
  | Empty_name -> "kernel IR region name cannot be empty"
  | Invalid_member_node_id id ->
      Printf.sprintf "kernel IR member node ID %d is negative" id
  | Duplicate_member_node_id id ->
      Printf.sprintf "kernel IR member node ID %d is duplicated" id
  | Duplicate_slot slot ->
      Printf.sprintf "kernel IR slot %d is duplicated" (Slot.to_int slot)
  | Unknown_slot slot ->
      Printf.sprintf "kernel IR slot %d is unknown" (Slot.to_int slot)
  | Forward_reference slot ->
      Printf.sprintf "kernel IR slot %d is a forward reference" (Slot.to_int slot)
  | Invalid_binding message -> "invalid kernel IR binding: " ^ message
  | Invalid_result message -> "invalid kernel IR result: " ^ message
  | Invalid_effect message -> "invalid kernel IR effect: " ^ message

let pp_slot formatter slot = Format.fprintf formatter "s%d" (Slot.to_int slot)

let pp_slots formatter slots =
  Format.fprintf formatter "[";
  List.iteri
    (fun index slot ->
      if index > 0 then Format.fprintf formatter ",";
      pp_slot formatter slot)
    slots;
  Format.fprintf formatter "]"

let pp_binding formatter (binding : binding) =
  Format.fprintf formatter "[";
  List.iteri
    (fun index (output : output_spec) ->
      if index > 0 then Format.fprintf formatter ",";
      pp_slot formatter output.slot)
    binding.outputs;
  Format.fprintf formatter "]=%s%a" (Primitive.to_string binding.primitive)
    pp_slots binding.inputs

let pp_result formatter ({ slot; value; storage } : result) =
  let storage =
    match storage with
    | Fresh -> "fresh"
    | Alias_input source -> Printf.sprintf "alias(s%d)" (Slot.to_int source)
    | In_place_input source -> Printf.sprintf "in-place(s%d)" (Slot.to_int source)
  in
  Format.fprintf formatter "%a=v%d/%s" pp_slot slot
    (Ir.Value.id value |> Ir.Value_id.to_int) storage

let pp formatter region =
  Format.fprintf formatter "region %S members=[" region.name;
  List.iteri
    (fun index id ->
      if index > 0 then Format.fprintf formatter ",";
      Format.fprintf formatter "%d" id)
    region.member_node_ids;
  Format.fprintf formatter "] inputs=[";
  List.iteri
    (fun index ({ slot; value } : input) ->
      if index > 0 then Format.fprintf formatter ";";
      Format.fprintf formatter "s%d:v%d" (Slot.to_int slot)
        (Ir.Value.id value |> Ir.Value_id.to_int))
    region.inputs;
  Format.fprintf formatter "] bindings=[";
  List.iteri
    (fun index binding ->
      if index > 0 then Format.fprintf formatter ";";
      pp_binding formatter binding)
    region.bindings;
  Format.fprintf formatter "] results=[";
  List.iteri
    (fun index result ->
      if index > 0 then Format.fprintf formatter ";";
      pp_result formatter result)
    region.results;
  Format.fprintf formatter "]"

let to_string region = Format.asprintf "%a" pp region
