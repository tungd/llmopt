module Capture = struct
  type t = string

  let of_string name =
    let name = String.trim name in
    if name = "" then Error "capture name cannot be empty"
    else if String.exists (function ' ' | '\t' | '\n' | '\r' -> true | _ -> false) name
    then Error "capture name cannot contain whitespace"
    else Ok name

  let of_string_exn name =
    match of_string name with Ok name -> name | Error message -> invalid_arg message

  let to_string name = name
  let compare (left : t) right = String.compare left right
end

module Shape = struct
  type dim = Any | Exact of int | Var of string
  type t = Any_shape | Rank of int | Dims of dim list
end

type operation = Any_operation | W4a16_linear | Rms_norm | Silu | Mul | Add
type effect_kind = Pure | Stateful | Synchronizing | Opaque | Any_effect
type use_count = Exactly of int | At_most of int | At_least of int

type predicate =
  | Op of operation
  | Shape of Capture.t * Shape.t
  | Dtype of Capture.t * Ir.Dtype.t
  | Effect of effect_kind
  | Uses of Capture.t * use_count

type pattern =
  | Node of node_pattern
  | Or of pattern list
  | Optional of pattern

and node_pattern = {
  op : operation;
  inputs : input_pattern list;
  outputs : output_pattern list;
  predicates : predicate list;
  capture : Capture.t option;
}

and input_pattern = Any_input | Capture_input of Capture.t | Produced_by of pattern
and output_pattern = Ignore_output | Capture_output of Capture.t

module Reader = struct
  type token = Left | Right | Token_atom of string
  type sexp = Atom of string | List of sexp list

  let ( let* ) = Result.bind

  let tokenize source =
    let length = String.length source in
    let rec loop index current tokens =
      if index = length then
        let tokens = if current = "" then tokens else Token_atom current :: tokens in
        Ok (List.rev tokens)
      else
        match source.[index] with
        | '(' ->
            let tokens = if current = "" then tokens else Token_atom current :: tokens in
            loop (index + 1) "" (Left :: tokens)
        | ')' ->
            let tokens = if current = "" then tokens else Token_atom current :: tokens in
            loop (index + 1) "" (Right :: tokens)
        | ' ' | '\t' | '\n' | '\r' ->
            let tokens = if current = "" then tokens else Token_atom current :: tokens in
            loop (index + 1) "" tokens
        | ';' ->
            let rec skip index =
              if index = length then index
              else if source.[index] = '\n' then index + 1
              else skip (index + 1)
            in
            let tokens = if current = "" then tokens else Token_atom current :: tokens in
            loop (skip (index + 1)) "" tokens
        | character ->
            loop (index + 1) (current ^ String.make 1 character) tokens
    in
    loop 0 "" []

  let parse_sexp source =
    let* tokens = tokenize source in
    let rec one = function
      | [] -> Error "expected an s-expression"
        | Token_atom atom :: rest -> Ok (Atom atom, rest)
      | Right :: _ -> Error "unexpected ')'"
      | Left :: rest ->
          let rec list acc = function
            | [] -> Error "unterminated s-expression"
            | Right :: rest -> Ok (List (List.rev acc), rest)
            | tokens ->
                let* value, rest = one tokens in
                list (value :: acc) rest
          in
          list [] rest
    in
    let* value, rest = one tokens in
    match rest with [] -> Ok value | _ -> Error "multiple s-expressions"
end

(* The parser is kept below the token reader so it can use the same result
   syntax without exposing the intermediate s-expression representation. *)
module Parser = struct
  open Reader

  let ( let* ) = Result.bind

  let atom = function Atom value -> Ok value | List _ -> Error "expected an atom"

  let list = function List values -> Ok values | Atom _ -> Error "expected a list"

  let capture sexp =
    let* name = atom sexp in
    let name = if String.length name > 0 && name.[0] = '$' then String.sub name 1 (String.length name - 1) else name in
    Capture.of_string name

  let operation = function
    | "any" | "*" -> Ok Any_operation
    | "w4a16-linear" | "w4a16_linear" | "w4a16-linear-g64" -> Ok W4a16_linear
    | "rms-norm" | "rms_norm" -> Ok Rms_norm
    | "silu" -> Ok Silu
    | "mul" -> Ok Mul
    | "add" -> Ok Add
    | value -> Error ("unknown fusion-query operation: " ^ value)

  let parse_effect = function
    | "pure" -> Ok Pure
    | "stateful" -> Ok Stateful
    | "synchronizing" | "sync" -> Ok Synchronizing
    | "opaque" -> Ok Opaque
    | "any" | "*" -> Ok Any_effect
    | value -> Error ("unknown fusion-query effect: " ^ value)

  let dtype = function
    | "f32" | "float32" -> Ok Ir.Dtype.Float32
    | "f16" | "float16" -> Ok Ir.Dtype.Float16
    | "bf16" | "bfloat16" -> Ok Ir.Dtype.Bfloat16
    | "i64" | "int64" -> Ok Ir.Dtype.Int64
    | "i32" | "int32" -> Ok Ir.Dtype.Int32
    | "i8" | "int8" -> Ok Ir.Dtype.Int8
    | "u8" | "uint8" -> Ok Ir.Dtype.UInt8
    | "bool" -> Ok Ir.Dtype.Bool
    | value -> Error ("unknown fusion-query dtype: " ^ value)

  let integer sexp =
    let* value = atom sexp in
    match int_of_string_opt value with
    | Some value -> Ok value
    | None -> Error ("expected integer, got " ^ value)

  let parse_list parser values =
    let rec loop acc = function
      | [] -> Ok (List.rev acc)
      | value :: rest ->
          let* parsed = parser value in
          loop (parsed :: acc) rest
    in
    loop [] values

  let shape sexp =
    match sexp with
    | Atom value when value = "any" || value = "*" -> Ok Shape.Any_shape
    | List [ Atom "rank"; rank ] ->
        integer rank |> Result.map (fun rank -> Shape.Rank rank)
    | List (Atom "dims" :: dims) ->
        let parse_dim = function
          | Atom "_" | Atom "*" -> Ok Shape.Any
          | Atom value -> (
              match int_of_string_opt value with
              | Some value when value >= 0 -> Ok (Shape.Exact value)
              | Some _ -> Error "shape dimensions cannot be negative"
              | None when String.length value > 0 && value.[0] = '?' ->
                  Ok (Shape.Var (String.sub value 1 (String.length value - 1)))
              | None -> Error ("invalid shape dimension: " ^ value))
          | List _ -> Error "shape dimension must be an atom"
        in
        let rec parse_dims acc = function
          | [] -> Ok (Shape.Dims (List.rev acc))
          | value :: rest ->
              let* value = parse_dim value in
              parse_dims (value :: acc) rest
        in
        parse_dims [] dims
    | _ -> Error "shape predicate expects any, (rank N), or (dims ...)"

  let rec value_pattern = function
    | Atom "$" -> Error "capture name cannot be empty"
    | Atom value when String.length value > 0 && value.[0] = '$' ->
        capture (Atom value) |> Result.map (fun value -> Capture_input value)
    | List [ Atom "any" ] -> Ok Any_input
    | List [ Atom "tensor"; value ] ->
        capture value |> Result.map (fun value -> Capture_input value)
    | List [ Atom "ref"; value ] ->
        capture value |> Result.map (fun value -> Capture_input value)
    | List [ Atom "produced-by"; pattern ] ->
        parse_pattern pattern |> Result.map (fun pattern -> Produced_by pattern)
    | List (Atom "node" :: _) as value ->
        parse_pattern value |> Result.map (fun pattern -> Produced_by pattern)
    | _ -> Error "invalid fusion-query input pattern"

  and output_pattern = function
    | List [ Atom "ignore" ] | Atom "_" -> Ok Ignore_output
    | Atom value when String.length value > 0 && value.[0] = '$' ->
        capture (Atom value) |> Result.map (fun value -> Capture_output value)
    | List [ Atom "tensor"; value ] ->
        capture value |> Result.map (fun value -> Capture_output value)
    | _ -> Error "invalid fusion-query output pattern"

  and use_count values =
    match values with
    | [ Atom "exactly"; count ] -> integer count |> Result.map (fun n -> Exactly n)
    | [ Atom "at-most"; count ] -> integer count |> Result.map (fun n -> At_most n)
    | [ Atom "at-least"; count ] -> integer count |> Result.map (fun n -> At_least n)
    | _ -> Error "uses predicate expects exactly/at-most/at-least N"

  and predicate = function
    | List [ Atom "op"; Atom value ] -> operation value |> Result.map (fun op -> Op op)
    | List [ Atom "effect"; Atom value ] ->
        parse_effect value |> Result.map (fun effect_kind -> Effect effect_kind)
    | List [ Atom "shape"; value; shape_sexp ] ->
        let* value = capture value in
        shape shape_sexp |> Result.map (fun shape -> Shape (value, shape))
    | List [ Atom "dtype"; value; Atom dtype_name ] ->
        let* value = capture value in
        dtype dtype_name |> Result.map (fun dtype -> Dtype (value, dtype))
    | List (Atom "uses" :: value :: rest) ->
        let* value = capture value in
        use_count rest |> Result.map (fun count -> Uses (value, count))
    | _ -> Error "invalid fusion-query predicate"

  and parse_pattern = function
    | List (Atom "node" :: fields) -> parse_node fields
    | List (Atom "or" :: patterns) -> parse_list parse_pattern patterns |> Result.map (fun patterns -> Or patterns)
    | List [ Atom "optional"; pattern ] -> parse_pattern pattern |> Result.map (fun pattern -> Optional pattern)
    | _ -> Error "fusion-query pattern must be node, or, or optional"

  and parse_node fields =
    let rec loop operation_field inputs outputs predicates node_capture = function
      | [] -> (
          match operation_field with
          | None -> Error "fusion-query node has no op"
          | Some op -> Ok (Node { op; inputs; outputs; predicates; capture = node_capture }))
      | List [ Atom "op"; Atom value ] :: rest ->
          let* op = operation value in
          loop (Some op) inputs outputs predicates node_capture rest
      | List (Atom "in" :: values) :: rest ->
          let* values = parse_list value_pattern values in
          loop operation_field values outputs predicates node_capture rest
      | List (Atom "out" :: values) :: rest ->
          let* values = parse_list output_pattern values in
          loop operation_field inputs values predicates node_capture rest
      | List (Atom "where" :: values) :: rest ->
          let* values = parse_list predicate values in
          loop operation_field inputs outputs (predicates @ values) node_capture rest
      | List [ Atom "capture"; value ] :: rest ->
          let* value = capture value in
          loop operation_field inputs outputs predicates (Some value) rest
      | _ -> Error "unknown fusion-query node field"
    in
    loop None [] [] [] None fields

  let parse source =
    let* sexp = Reader.parse_sexp source in
    parse_pattern sexp
end

module Sexp = struct
  let parse = Parser.parse
end

let parse = Sexp.parse

module Match = struct
  type value = Node of Ir.node | Tensor of Ir.Value.t

  type t = {
    root : Ir.node;
    captures : (Capture.t * value) list;
    member_node_ids : int list;
  }

  let root match_ = match_.root

  let capture name match_ =
    List.find_map
      (fun (candidate, value) ->
        if Capture.compare candidate name = 0 then Some value else None)
      match_.captures

  let member_node_ids match_ = match_.member_node_ids
end

module Node_id_set = Set.Make (Int)

let value_equal = Ir.Value.equal

let contains text needle =
  let text_length = String.length text in
  let needle_length = String.length needle in
  let rec at index =
    if index + needle_length > text_length then false
    else if String.sub text index needle_length = needle then true
    else at (index + 1)
  in
  if needle_length = 0 then true else if text_length < needle_length then false else at 0

let operation_of_node node =
  match Ir.node_op node with
  | Ir.Op.Rms_norm _ -> Rms_norm
  | Ir.Op.Add _ -> Add
  | Ir.Op.Primitive (Ir.Primitive.Pointwise (Ir.Pointwise.Unary (Ir.Pointwise.Silu, _))) -> Silu
  | Ir.Op.Primitive (Ir.Primitive.Pointwise (Ir.Pointwise.Binary (Ir.Pointwise.Mul, _, _))) -> Mul
  | Ir.Op.Primitive (Ir.Primitive.Pointwise (Ir.Pointwise.Binary (Ir.Pointwise.Add, _, _))) -> Add
  | Ir.Op.Opaque { op; _ } when
      let op = String.lowercase_ascii op in
      op = "w4a16_linear" || op = "w4a16-linear-g64" || contains op "w4a16" ->
      W4a16_linear
  | operation ->
      let operation = String.lowercase_ascii (Ir.Op.to_string operation) in
      if contains operation "w4a16" then W4a16_linear else Any_operation

let matches_operation expected actual =
  match expected with Any_operation -> true | expected -> expected = actual

let effect_of_node node =
  match Ir.node_op node with
  | Ir.Op.Opaque _ -> Opaque
  | Ir.Op.Barrier_create _ | Ir.Op.Barrier_arrive _ | Ir.Op.Barrier_wait _ -> Synchronizing
  | Ir.Op.Alloc _ | Ir.Op.Copy _ | Ir.Op.Short_conv_step _
  | Ir.Op.Short_conv_step_fused _ | Ir.Op.Short_conv_prefill _
  | Ir.Op.Primitive (Ir.Primitive.Paged_attention_q8 _) -> Stateful
  | _ -> Pure

let matches_effect expected actual =
  match expected with Any_effect -> true | expected -> expected = actual

let node_outputs node =
  Option.to_list (Ir.node_output node)
  @ Ir.Op.additional_outputs (Ir.node_op node)

let producer nodes value =
  List.find_opt
    (fun node -> List.exists (value_equal value) (node_outputs node))
    nodes

let uses graph value =
  Ir.Graph.nodes graph
  |> List.fold_left
       (fun count node ->
         count
         + List.fold_left
             (fun count input -> if value_equal input value then count + 1 else count)
             0 (Ir.node_inputs node))
       0

type capture_value = Match.value

type state = {
  captures : (Capture.t * capture_value) list;
  shape_variables : (string * int) list;
  member_node_ids : int list;
  visiting : Node_id_set.t;
}

let initial_state =
  { captures = []; shape_variables = []; member_node_ids = []; visiting = Node_id_set.empty }

let find_capture name captures =
  List.find_map
    (fun (candidate, value) ->
      if Capture.compare candidate name = 0 then Some value else None)
    captures

let same_capture_value left right =
  match left, right with
  | Match.Node left, Match.Node right -> Ir.node_id left = Ir.node_id right
  | Match.Tensor left, Match.Tensor right -> value_equal left right
  | _ -> false

let bind_capture name value state =
  match find_capture name state.captures with
  | Some existing when same_capture_value existing value -> Some state
  | Some _ -> None
  | None -> Some { state with captures = (name, value) :: state.captures }

let add_member node state =
  let id = Ir.node_id node in
  if List.mem id state.member_node_ids then state
  else { state with member_node_ids = state.member_node_ids @ [ id ] }

let shape_matches shape value state =
  let dimensions = Tensor_shape.dimensions (Ir.Value.logical_shape value) in
  let bind_variable name dimension state =
    match List.assoc_opt name state.shape_variables with
    | Some expected when expected <> dimension -> None
    | Some _ -> Some state
    | None -> Some { state with shape_variables = (name, dimension) :: state.shape_variables }
  in
  match shape with
  | Shape.Any_shape -> Some state
  | Shape.Rank rank when List.length dimensions = rank -> Some state
  | Shape.Rank _ -> None
  | Shape.Dims expected when List.length expected = List.length dimensions ->
      List.fold_left2
        (fun state expected actual ->
          match state, expected with
          | None, _ -> None
          | Some state, Shape.Any -> Some state
          | Some state, Shape.Exact expected when expected = actual -> Some state
          | Some state, Shape.Exact _ -> None
          | Some state, Shape.Var name -> bind_variable name actual state)
        (Some state) expected dimensions
  | Shape.Dims _ -> None

let capture_tensor name state =
  match find_capture name state.captures with
  | Some (Match.Tensor value) -> Some value
  | _ -> None

let matches_use_count expected actual =
  match expected with
  | Exactly expected -> actual = expected
  | At_most expected -> actual <= expected
  | At_least expected -> actual >= expected

let predicate_matches graph node predicate state =
  match predicate with
  | Op operation ->
      if matches_operation operation (operation_of_node node) then Some state else None
  | Shape (capture, shape) ->
      Option.bind (capture_tensor capture state) (fun value ->
          shape_matches shape value state)
  | Dtype (capture, dtype) ->
      Option.bind (capture_tensor capture state) (fun value ->
          if Ir.Value.dtype value = dtype then Some state else None)
  | Effect expected ->
      if matches_effect expected (effect_of_node node) then Some state else None
  | Uses (capture, expected) ->
      Option.bind (capture_tensor capture state) (fun value ->
          if matches_use_count expected (uses graph value) then Some state else None)

let predicates_match graph node predicates state =
  List.fold_left
    (fun state predicate -> Option.bind state (predicate_matches graph node predicate))
    (Some state) predicates

let rec match_pattern graph nodes pattern node state =
  match pattern with
  | Or patterns ->
      List.concat_map (fun pattern -> match_pattern graph nodes pattern node state) patterns
  | Optional pattern ->
      state :: match_pattern graph nodes pattern node state
  | Node specification -> (
      match node with
      | None -> []
      | Some node -> match_node graph nodes specification node state)

and match_node graph nodes specification node state =
  let node_id = Ir.node_id node in
  if Node_id_set.mem node_id state.visiting then []
  else if not (matches_operation specification.op (operation_of_node node)) then []
  else
    let state =
      { (add_member node state) with visiting = Node_id_set.add node_id state.visiting }
    in
    let node_capture =
      match specification.capture with
      | None -> Some state
      | Some capture -> bind_capture capture (Match.Node node) state
    in
    let match_outputs state =
      let outputs = node_outputs node in
      if specification.outputs = [] then [ state ]
      else if List.length outputs <> List.length specification.outputs then []
      else
        List.fold_left2
          (fun states output_pattern output ->
            List.concat_map
              (fun state ->
                match output_pattern with
                | Ignore_output -> [ state ]
                | Capture_output capture ->
                    Option.to_list (bind_capture capture (Match.Tensor output) state))
              states)
          [ state ] specification.outputs outputs
    in
    let match_inputs_exact state inputs =
      let rec loop states patterns inputs =
        match patterns, inputs with
        | [], [] -> states
        | pattern :: patterns, input :: inputs ->
            let states =
              List.concat_map
                (fun state -> match_input graph nodes pattern input state)
                states
            in
            loop states patterns inputs
        | _ -> []
      in
      loop [ state ] specification.inputs inputs
    in
    let match_inputs state =
      let inputs = Ir.node_inputs node in
      let direct = match_inputs_exact state inputs in
      if direct <> [] then direct
      else
        match specification.op, inputs with
        | (Add | Mul), [ left; right ] ->
            match_inputs_exact state [ right; left ]
        | _ -> []
    in
    let states =
      match node_capture with
      | None -> []
      | Some state -> match_inputs state
    in
    let states = List.concat_map match_outputs states in
    let states =
      List.filter_map
        (fun state -> predicates_match graph node specification.predicates state)
        states
    in
    List.map
      (fun state -> { state with visiting = Node_id_set.remove node_id state.visiting })
      states

and match_input graph nodes pattern value state =
  match pattern with
  | Any_input -> [ state ]
  | Capture_input capture ->
      Option.to_list (bind_capture capture (Match.Tensor value) state)
  | Produced_by pattern ->
      match producer nodes value with
      | Some node -> match_pattern graph nodes pattern (Some node) state
      | None -> match_pattern graph nodes pattern None state

let match_graph pattern graph =
  let nodes = Ir.Graph.nodes graph in
  List.concat_map
    (fun node ->
      match_pattern graph nodes pattern (Some node) initial_state
      |> List.map (fun state ->
             {
               Match.root = node;
               captures = state.captures;
               member_node_ids = state.member_node_ids;
             }))
    nodes

let construct_region ~name ~match_ ~inputs ~bindings ~results ~effects ~resource =
  Kernel_ir.create ~name ~member_node_ids:(Match.member_node_ids match_) ~inputs ~bindings
    ~results ~effects ~resource
  |> Result.map_error Kernel_ir.error_to_string

module Rule = struct
  let ( let* ) = Result.bind

  type t = {
    pattern : pattern;
    result_captures : Capture.t list;
    emit : Match.t -> (Kernel_ir.t, string) result;
  }

  let create ~pattern ~result_captures ~emit = { pattern; result_captures; emit }

  let apply rule graph =
    let matches = match_graph rule.pattern graph in
    List.fold_left
      (fun regions match_ ->
        let* regions = regions in
        match
          List.find_opt
            (fun capture -> Match.capture capture match_ = None)
            rule.result_captures
        with
        | Some capture ->
            Error
              (Printf.sprintf "fusion rule result capture %s is not bound"
                 (Capture.to_string capture))
        | None ->
            let* region = rule.emit match_ in
            Ok (region :: regions))
      (Ok []) matches
    |> Result.map List.rev
end
