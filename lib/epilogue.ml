open Applicative_expr

type slot = Acc

type t =
  | None
  | Bias of { bias : Ir.Value.t }
  | Silu
  | Gelu
  | Silu_mul of { up : Ir.Value.t }
  | Gelu_mul of { up : Ir.Value.t }
  | Residual_add of { residual : Ir.Value.t }
  | Bias_residual_add of { bias : Ir.Value.t; residual : Ir.Value.t }
  | Silu_mul_residual_add of { up : Ir.Value.t; residual : Ir.Value.t }
  | Gelu_mul_residual_add of { up : Ir.Value.t; residual : Ir.Value.t }
  | Expr of slot Applicative_expr.t

let none = None

let bias b = Bias { bias = b }
let silu = Silu
let gelu = Gelu
let silu_mul ~up = Silu_mul { up }
let gelu_mul ~up = Gelu_mul { up }
let residual_add ~residual = Residual_add { residual }
let bias_residual_add ~bias ~residual = Bias_residual_add { bias; residual }
let silu_mul_residual_add ~up ~residual = Silu_mul_residual_add { up; residual }
let gelu_mul_residual_add ~up ~residual = Gelu_mul_residual_add { up; residual }

let mul ~factor =
  Expr (Binop (Mul, Slot Acc, Read { value = factor; coord = Coordinate_map.identity }))

let of_expr e = Expr e

let to_expr = function
  | None -> Slot Acc
  | Bias { bias } ->
      Binop (Add, Slot Acc, Read { value = bias; coord = Coordinate_map.identity })
  | Silu -> Unop (Silu, Slot Acc)
  | Gelu -> Unop (Gelu, Slot Acc)
  | Silu_mul { up } ->
      Binop (Mul, Unop (Silu, Slot Acc), Read { value = up; coord = Coordinate_map.identity })
  | Gelu_mul { up } ->
      Binop (Mul, Unop (Gelu, Slot Acc), Read { value = up; coord = Coordinate_map.identity })
  | Residual_add { residual } ->
      Binop (Add, Slot Acc, Read { value = residual; coord = Coordinate_map.identity })
  | Bias_residual_add { bias; residual } ->
      Binop (Add, Binop (Add, Slot Acc, Read { value = bias; coord = Coordinate_map.identity }),
             Read { value = residual; coord = Coordinate_map.identity })
  | Silu_mul_residual_add { up; residual } ->
      Binop (Add, Binop (Mul, Unop (Silu, Slot Acc), Read { value = up; coord = Coordinate_map.identity }),
             Read { value = residual; coord = Coordinate_map.identity })
  | Gelu_mul_residual_add { up; residual } ->
      Binop (Add, Binop (Mul, Unop (Gelu, Slot Acc), Read { value = up; coord = Coordinate_map.identity }),
             Read { value = residual; coord = Coordinate_map.identity })
  | Expr e -> e

let inputs = function
  | None | Silu | Gelu -> []
  | Bias { bias } -> [ bias ]
  | Silu_mul { up } | Gelu_mul { up } -> [ up ]
  | Residual_add { residual } -> [ residual ]
  | Bias_residual_add { bias; residual } -> [ bias; residual ]
  | Silu_mul_residual_add { up; residual }
  | Gelu_mul_residual_add { up; residual } -> [ up; residual ]
  | Expr e -> Applicative_expr.required_values e

let equal left right =
  (* Algebraic equality: tags and expressions are compared through their
     [to_expr] forms so that equal terms with different raw trees compare
     equal, congruent with [Monoid.equal]. *)
  Applicative_expr.equal (fun Acc Acc -> true) (to_expr left) (to_expr right)

let to_string = function
  | None -> "none"
  | Bias _ -> "bias"
  | Silu -> "silu"
  | Gelu -> "gelu"
  | Silu_mul _ -> "silu_mul"
  | Gelu_mul _ -> "gelu_mul"
  | Residual_add _ -> "residual_add"
  | Bias_residual_add _ -> "bias_residual_add"
  | Silu_mul_residual_add _ -> "silu_mul_residual_add"
  | Gelu_mul_residual_add _ -> "gelu_mul_residual_add"
  | Expr e -> Applicative_expr.to_string (fun Acc -> "acc") e

module Monoid = struct
  (* The scalar sublanguage of [Epilogue.t]: composition over scalar trees is
     this monoid's append (pure substitution), which [compose] below reduces
     to before it reconstructs the legacy composite tags. *)
  type t = slot Applicative_expr.t

  let empty = Applicative_expr.Slot Acc

  let append producer consumer =
    Applicative_expr.simplify
      (Applicative_expr.subst (function Acc -> producer) consumer)

  let is_empty = function
    | Applicative_expr.Slot Acc -> true
    | _ -> false

  let equal a b =
    Applicative_expr.equal (fun Acc Acc -> true) a b

  let to_string e =
    Applicative_expr.to_string (fun Acc -> "acc") e

  let emit_msl e ~acc ~idx =
    Applicative_expr.emit_msl_expr ~slot_emitter:(fun Acc -> acc) ~index_expr:idx (Applicative_expr.simplify e)
end

(* Composition over the scalar sublanguage is the monoid append (pure
   substitution); tags never participate in the composition step.  Composite
   tags are recognised normal forms of the composed operand pair and are
   reconstructed afterwards, with their roles read from the operand tags
   (roles are explicit only on the constructors, never recoverable from a
   scalar tree).  The four legacy composite templates below are the only
   pairings that carry a tag today: each owns its FP association order, so
   the reconstruction returns the template (not the raw substitution tree) to
   keep byte-identical emission; every other composition is a pure [Expr]. *)
let tag_of_composition producer consumer =
  match producer, consumer with
  | Bias { bias }, Residual_add { residual }
  | Residual_add { residual }, Bias { bias } ->
      Some (Bias_residual_add { bias; residual })
  | Silu_mul { up }, Residual_add { residual } ->
      Some (Silu_mul_residual_add { up; residual })
  | Gelu_mul { up }, Residual_add { residual } ->
      Some (Gelu_mul_residual_add { up; residual })
  | _ -> None

let compose ~producer ~consumer =
  match producer, consumer with
  | None, consumer -> consumer
  | producer, None -> producer
  | _ ->
      (* Graph-level composition: producer ⋄ consumer substitutes the
         producer's scalar tree at the accumulator slot of the consumer's
         scalar tree -- exactly [Monoid.append] -- with no typed compatibility
         branch.  The final reconstruction re-tags the legacy composite
         pairings above; all other results are the pure [Expr] normal form. *)
      let composed = Monoid.append (to_expr producer) (to_expr consumer) in
      (match tag_of_composition producer consumer with
      | Some tag -> tag
      | None -> Expr composed)

let emit_msl_writeback ~acc ~idx = function
  | None -> acc
  | Bias _ ->
      Printf.sprintf "(%s + (float)bias[%s])" acc idx
  | Silu ->
      Printf.sprintf "(%s * (1.0f / (1.0f + metal::exp(-%s))))" acc acc
  | Gelu ->
      Printf.sprintf "(0.5f * %s * (1.0f + metal::tanh(0.79788456f * (%s + 0.044715f * %s * %s * %s))))"
        acc acc acc acc acc
  | Silu_mul _ ->
      Printf.sprintf "((%s * (1.0f / (1.0f + metal::exp(-%s)))) * (float)up[%s])"
        acc acc idx
  | Gelu_mul _ ->
      Printf.sprintf "((0.5f * %s * (1.0f + metal::tanh(0.79788456f * (%s + 0.044715f * %s * %s * %s)))) * (float)up[%s])"
        acc acc acc acc acc idx
  | Residual_add _ ->
      Printf.sprintf "(%s + (float)residual[%s])" acc idx
  | Bias_residual_add _ ->
      Printf.sprintf "((%s + (float)bias[%s]) + (float)residual[%s])" acc idx idx
  | Silu_mul_residual_add _ ->
      Printf.sprintf "(((%s * (1.0f / (1.0f + metal::exp(-%s)))) * (float)up[%s]) + (float)residual[%s])"
        acc acc idx idx
  | Gelu_mul_residual_add _ ->
      Printf.sprintf "(((0.5f * %s * (1.0f + metal::tanh(0.79788456f * (%s + 0.044715f * %s * %s * %s)))) * (float)up[%s]) + (float)residual[%s])"
        acc acc acc acc acc idx idx
  | Expr expr ->
      Applicative_expr.emit_msl_expr ~slot_emitter:(fun Acc -> acc) ~index_expr:idx expr

let map_reads ~rename expr =
  let rec go expr =
    match expr with
    | Slot _ | Const _ as leaf -> leaf
    | Read { value; coord } -> Read { value = rename value; coord }
    | Unop (op, operand) -> Unop (op, go operand)
    | Binop (op, left, right) -> Binop (op, go left, go right)
    | Triop (op, a, b, c) -> Triop (op, go a, go b, go c)
  in
  go expr

let canonicalize epilogue =
  let expr = to_expr epilogue in
  let required = Applicative_expr.required_values expr in
  let ordinal value =
    let rec find index = function
      | [] ->
          invalid_arg
            "Epilogue.canonicalize: tensor not required by the epilogue expression"
      | candidate :: rest ->
          if Ir.Value.equal value candidate then index else find (index + 1) rest
    in
    find 0 required
  in
  let rename value =
    Ir.Value.make_tensor ~id:(ordinal value)
      ~shape:(Ir.Value.logical_shape value) ~dtype:(Ir.Value.dtype value)
  in
  Expr (map_reads ~rename expr)

let msl_buffer_decls ~start_index epilogue =
  (* Buffer order and identity must agree exactly with the tensor reads of
     [emit_msl_writeback]: the typed variants read their role-named buffers
     ([bias]/[up]/[residual]) while [Expr] trees read [buf_<Value_id>]
     (matching [Applicative_expr.emit_msl_expr]).  Enumerating [inputs] here
     keeps the declaration layout and the readback names in one place, ordered
     by the stable first-occurrence order of [Epilogue.inputs]. *)
  let bindings =
    match epilogue with
    | None | Silu | Gelu -> []
    | Bias { bias } -> [ (bias, "bias") ]
    | Silu_mul { up } | Gelu_mul { up } -> [ (up, "up") ]
    | Residual_add { residual } -> [ (residual, "residual") ]
    | Bias_residual_add { bias; residual } ->
        [ (bias, "bias"); (residual, "residual") ]
    | Silu_mul_residual_add { up; residual }
    | Gelu_mul_residual_add { up; residual } ->
        [ (up, "up"); (residual, "residual") ]
    | Expr expr ->
        Applicative_expr.required_values expr
        |> List.map (fun value ->
               ( value,
                 Printf.sprintf "buf_%d"
                   (Ir.Value_id.to_int (Ir.Value.id value)) ))
  in
  bindings
  |> List.mapi (fun offset (value, name) ->
         let type_name =
           match Ir.Value.dtype value with
           | Ir.Dtype.Float16 -> "half"
           | _ -> "float"
         in
         Printf.sprintf ",\n    device const %s* %s [[buffer(%d)]]" type_name
           name (start_index + offset))
  |> String.concat ""

let algebraic_kernel_name ~base epilogue =
  let canonical = to_expr (canonicalize epilogue) in
  let writeback =
    Applicative_expr.emit_msl_expr
      ~slot_emitter:(fun Acc -> "acc") ~index_expr:"col" canonical
  in
  let dtypes =
    Applicative_expr.required_values canonical
    |> List.map (fun value -> Ir.Dtype.to_string (Ir.Value.dtype value))
    |> String.concat ","
  in
  let digest = Digest.string (dtypes ^ "|" ^ writeback) in
  base ^ "_" ^ String.sub (Digest.to_hex digest) 0 8
