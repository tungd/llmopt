type unop =
  | Neg
  | Abs
  | Silu
  | Gelu
  | Exp
  | Log
  | Rsqrt
  | Cast of Ir.Dtype.t

type binop =
  | Add
  | Sub
  | Mul
  | Div
  | Min
  | Max

type triop =
  | Clamp
  | Where

type 'var t =
  | Slot of 'var
  | Const of float
  | Read of { value : Ir.Value.t; coord : Coordinate_map.t }
  | Unop of unop * 'var t
  | Binop of binop * 'var t * 'var t
  | Triop of triop * 'var t * 'var t * 'var t

let pure x = Const x

let fmap op e =
  match op, e with
  | Cast t, Read { value; _ } when Ir.Value.dtype value = t -> e
  | Cast t1, Unop (Cast t2, inner) when t1 = t2 -> Unop (Cast t1, inner)
  | _ -> Unop (op, e)

let lift2 op a b = Binop (op, a, b)
let lift3 op a b c = Triop (op, a, b, c)

let rec subst f = function
  | Slot v -> f v
  | Const c -> Const c
  | Read r -> Read r
  | Unop (op, a) -> Unop (op, subst f a)
  | Binop (op, a, b) -> Binop (op, subst f a, subst f b)
  | Triop (op, a, b, c) -> Triop (op, subst f a, subst f b, subst f c)

let rec simplify = function
  | Slot v -> Slot v
  | Const c -> Const c
  | Read { value; coord } ->
      Read { value; coord = Coordinate_map.simplify coord }
  | Unop (Cast t, Read { value; coord }) when Ir.Value.dtype value = t ->
      Read { value; coord = Coordinate_map.simplify coord }
  | Unop (Cast t1, Unop (Cast t2, inner)) when t1 = t2 ->
      simplify (Unop (Cast t1, inner))
  | Unop (Neg, Unop (Neg, inner)) -> simplify inner
  | Unop (op, a) -> (
      let sa = simplify a in
      match op, sa with
      | Cast t, Read { value; _ } when Ir.Value.dtype value = t -> sa
      | Neg, Const c -> Const (-.c)
      | Abs, Const c -> Const (abs_float c)
      | Exp, Const c -> Const (exp c)
      | Log, Const c -> Const (log c)
      | Rsqrt, Const c when c > 0.0 -> Const (1.0 /. sqrt c)
      | _ -> Unop (op, sa))
  | Binop (op, a, b) -> (
      let sa = simplify a in
      let sb = simplify b in
      match op, sa, sb with
      | Add, Const c1, Const c2 -> Const (c1 +. c2)
      | Sub, Const c1, Const c2 -> Const (c1 -. c2)
      | Mul, Const c1, Const c2 -> Const (c1 *. c2)
      | Div, Const c1, Const c2 when c2 <> 0.0 -> Const (c1 /. c2)
      | Min, Const c1, Const c2 -> Const (min c1 c2)
      | Max, Const c1, Const c2 -> Const (max c1 c2)
      | Add, x, Const 0.0 | Add, Const 0.0, x -> x
      | Sub, x, Const 0.0 -> x
      | Mul, x, Const 1.0 | Mul, Const 1.0, x -> x
      | Mul, _, Const 0.0 | Mul, Const 0.0, _ -> Const 0.0
      | Div, x, Const 1.0 -> x
      | _ -> Binop (op, sa, sb))
  | Triop (Clamp, x, Const min_c, Const max_c) ->
      let sx = simplify x in
      Triop (Clamp, sx, Const min_c, Const max_c)
  | Triop (op, a, b, c) ->
      Triop (op, simplify a, simplify b, simplify c)

let rec structural_equal cmp_slot a b =
  match a, b with
  | Slot s1, Slot s2 -> cmp_slot s1 s2
  | Const c1, Const c2 -> Float.equal c1 c2
  | Read r1, Read r2 ->
      Ir.Value.equal r1.value r2.value
      && Coordinate_map.equal r1.coord r2.coord
  | Unop (op1, a1), Unop (op2, a2) -> op1 = op2 && structural_equal cmp_slot a1 a2
  | Binop (op1, a1, b1), Binop (op2, a2, b2) ->
      op1 = op2 && structural_equal cmp_slot a1 a2 && structural_equal cmp_slot b1 b2
  | Triop (op1, a1, b1, c1), Triop (op2, a2, b2, c2) ->
      op1 = op2 && structural_equal cmp_slot a1 a2
      && structural_equal cmp_slot b1 b2 && structural_equal cmp_slot c1 c2
  | _ -> false

(* Algebraic equality: both sides are normalized by the idempotent [simplify]
   before structural comparison, so e.g. [Add x (Const 0.0)] equals [x]. *)
let equal cmp_slot a b =
  structural_equal cmp_slot (simplify a) (simplify b)

module Value_id_set = Set.Make (struct
    type t = Ir.Value_id.t

    let compare = Ir.Value_id.compare
  end)

(* Collects unique tensor inputs in stable first-syntactic-occurrence order
   (left-to-right, pre-order), deduplicating by [Value_id]. *)
let required_values expr =
  let rec walk seen acc = function
    | Slot _ | Const _ -> (seen, acc)
    | Read { value; _ } ->
        let id = Ir.Value.id value in
        if Value_id_set.mem id seen then (seen, acc)
        else (Value_id_set.add id seen, value :: acc)
    | Unop (_, a) -> walk seen acc a
    | Binop (_, a, b) ->
        let seen, acc = walk seen acc a in
        walk seen acc b
    | Triop (_, a, b, c) ->
        let seen, acc = walk seen acc a in
        let seen, acc = walk seen acc b in
        walk seen acc c
  in
  let _, reversed = walk Value_id_set.empty [] expr in
  List.rev reversed

let unop_to_string = function
  | Neg -> "-"
  | Abs -> "abs"
  | Silu -> "silu"
  | Gelu -> "gelu"
  | Exp -> "exp"
  | Log -> "log"
  | Rsqrt -> "rsqrt"
  | Cast t -> "cast<" ^ Ir.Dtype.to_string t ^ ">"

let binop_to_string = function
  | Add -> "+"
  | Sub -> "-"
  | Mul -> "*"
  | Div -> "/"
  | Min -> "min"
  | Max -> "max"

let triop_to_string = function
  | Clamp -> "clamp"
  | Where -> "where"

let rec to_string slot_to_string = function
  | Slot s -> slot_to_string s
  | Const c -> Printf.sprintf "%.4g" c
  | Read { value; coord } ->
      let vname = Printf.sprintf "v%d" (Ir.Value_id.to_int (Ir.Value.id value)) in
      if Coordinate_map.is_identity coord then vname
      else Printf.sprintf "%s[%s]" vname (Coordinate_map.to_string coord)
  | Unop (op, a) -> Printf.sprintf "%s(%s)" (unop_to_string op) (to_string slot_to_string a)
  | Binop (op, a, b) ->
      Printf.sprintf "(%s %s %s)" (to_string slot_to_string a) (binop_to_string op)
        (to_string slot_to_string b)
  | Triop (op, a, b, c) ->
      Printf.sprintf "%s(%s, %s, %s)" (triop_to_string op) (to_string slot_to_string a)
        (to_string slot_to_string b) (to_string slot_to_string c)

let map_reads ~rename expr =
  let rec go = function
    | Slot _ | Const _ as leaf -> leaf
    | Read { value; coord } -> Read { value = rename value; coord }
    | Unop (op, operand) -> Unop (op, go operand)
    | Binop (op, left, right) -> Binop (op, go left, go right)
    | Triop (op, a, b, c) -> Triop (op, go a, go b, go c)
  in
  go expr

let canonicalize expr =
  let required = required_values expr in
  let ordinal value =
    let rec find index = function
      | [] ->
          invalid_arg
            "Applicative_expr.canonicalize: tensor not required by the \
             expression"
      | candidate :: rest ->
          if Ir.Value.equal value candidate then index else find (index + 1) rest
    in
    find 0 required
  in
  let rename value =
    Ir.Value.make_tensor ~id:(ordinal value)
      ~shape:(Ir.Value.logical_shape value) ~dtype:(Ir.Value.dtype value)
  in
  map_reads ~rename expr

let rec emit_msl_expr ?value_var ~slot_emitter ~index_expr expr =
  let get_var =
    match value_var with
    | Some f -> f
    | None -> fun v -> Printf.sprintf "buf_%d" (Ir.Value_id.to_int (Ir.Value.id v))
  in
  match expr with
  | Slot s -> slot_emitter s
  | Const c -> Printf.sprintf "%.7ff" c
  | Read { value; coord } ->
      let shape_dims =
        Tensor_shape.dimensions (Ir.Value.logical_shape value)
      in
      let idx = Coordinate_map.emit_msl_index coord ~index_var:index_expr ~shape:shape_dims in
      Printf.sprintf "((float)%s[%s])" (get_var value) idx
  | Unop (Neg, a) ->
      Printf.sprintf "(-%s)" (emit_msl_expr ?value_var ~slot_emitter ~index_expr a)
  | Unop (Abs, a) ->
      Printf.sprintf "metal::abs(%s)" (emit_msl_expr ?value_var ~slot_emitter ~index_expr a)
  | Unop (Silu, a) ->
      let e = emit_msl_expr ?value_var ~slot_emitter ~index_expr a in
      Printf.sprintf "(%s * (1.0f / (1.0f + metal::exp(-%s))))" e e
  | Unop (Gelu, a) ->
      let e = emit_msl_expr ?value_var ~slot_emitter ~index_expr a in
      Printf.sprintf "(0.5f * %s * (1.0f + metal::tanh(metal::clamp(0.7978845608f * (%s + 0.044715f * %s * %s * %s), -10.0f, 10.0f))))"
        e e e e e
  | Unop (Exp, a) ->
      Printf.sprintf "metal::exp(%s)" (emit_msl_expr ?value_var ~slot_emitter ~index_expr a)
  | Unop (Log, a) ->
      Printf.sprintf "metal::log(%s)" (emit_msl_expr ?value_var ~slot_emitter ~index_expr a)
  | Unop (Rsqrt, a) ->
      Printf.sprintf "metal::rsqrt(%s)" (emit_msl_expr ?value_var ~slot_emitter ~index_expr a)
  | Unop (Cast Ir.Dtype.Float16, a) ->
      Printf.sprintf "((half)(%s))" (emit_msl_expr ?value_var ~slot_emitter ~index_expr a)
  | Unop (Cast Ir.Dtype.Float32, a) ->
      (* All arithmetic in this emitter is float; a float32 cast of the inner
         expression is an identity in this context (half operands promote
         implicitly), so the cast is intentionally elided. *)
      emit_msl_expr ?value_var ~slot_emitter ~index_expr a
  | Unop (Cast dtype, _) ->
      invalid_arg
        (Printf.sprintf
           "Applicative_expr.emit_msl_expr: cast to %s is not representable \
            (only Float16 and Float32 casts are supported)"
           (Ir.Dtype.to_string dtype))
  | Binop (Add, a, b) ->
      Printf.sprintf "(%s + %s)"
        (emit_msl_expr ?value_var ~slot_emitter ~index_expr a)
        (emit_msl_expr ?value_var ~slot_emitter ~index_expr b)
  | Binop (Sub, a, b) ->
      Printf.sprintf "(%s - %s)"
        (emit_msl_expr ?value_var ~slot_emitter ~index_expr a)
        (emit_msl_expr ?value_var ~slot_emitter ~index_expr b)
  | Binop (Mul, a, b) ->
      Printf.sprintf "(%s * %s)"
        (emit_msl_expr ?value_var ~slot_emitter ~index_expr a)
        (emit_msl_expr ?value_var ~slot_emitter ~index_expr b)
  | Binop (Div, a, b) ->
      Printf.sprintf "(%s / %s)"
        (emit_msl_expr ?value_var ~slot_emitter ~index_expr a)
        (emit_msl_expr ?value_var ~slot_emitter ~index_expr b)
  | Binop (Min, a, b) ->
      Printf.sprintf "metal::min(%s, %s)"
        (emit_msl_expr ?value_var ~slot_emitter ~index_expr a)
        (emit_msl_expr ?value_var ~slot_emitter ~index_expr b)
  | Binop (Max, a, b) ->
      Printf.sprintf "metal::max(%s, %s)"
        (emit_msl_expr ?value_var ~slot_emitter ~index_expr a)
        (emit_msl_expr ?value_var ~slot_emitter ~index_expr b)
  | Triop (Clamp, x, min_val, max_val) ->
      Printf.sprintf "metal::clamp(%s, %s, %s)"
        (emit_msl_expr ?value_var ~slot_emitter ~index_expr x)
        (emit_msl_expr ?value_var ~slot_emitter ~index_expr min_val)
        (emit_msl_expr ?value_var ~slot_emitter ~index_expr max_val)
  | Triop (Where, cond, true_val, false_val) ->
      Printf.sprintf "(%s ? %s : %s)"
        (emit_msl_expr ?value_var ~slot_emitter ~index_expr cond)
        (emit_msl_expr ?value_var ~slot_emitter ~index_expr true_val)
        (emit_msl_expr ?value_var ~slot_emitter ~index_expr false_val)
