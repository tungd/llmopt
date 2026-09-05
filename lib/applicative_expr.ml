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

let rec equal cmp_slot a b =
  match a, b with
  | Slot s1, Slot s2 -> cmp_slot s1 s2
  | Const c1, Const c2 -> Float.equal c1 c2
  | Read r1, Read r2 ->
      Ir.Value.equal r1.value r2.value && Coordinate_map.equal r1.coord r2.coord
  | Unop (op1, a1), Unop (op2, a2) -> op1 = op2 && equal cmp_slot a1 a2
  | Binop (op1, a1, b1), Binop (op2, a2, b2) ->
      op1 = op2 && equal cmp_slot a1 a2 && equal cmp_slot b1 b2
  | Triop (op1, a1, b1, c1), Triop (op2, a2, b2, c2) ->
      op1 = op2 && equal cmp_slot a1 a2 && equal cmp_slot b1 b2 && equal cmp_slot c1 c2
  | _ -> false

let rec required_values = function
  | Slot _ | Const _ -> []
  | Read { value; _ } -> [ value ]
  | Unop (_, a) -> required_values a
  | Binop (_, a, b) ->
      let v1 = required_values a in
      let v2 = required_values b in
      List.fold_left
        (fun acc v -> if List.exists (Ir.Value.equal v) acc then acc else v :: acc)
        v1 v2
  | Triop (_, a, b, c) ->
      let v12 = required_values (Binop (Add, a, b)) in
      let v3 = required_values c in
      List.fold_left
        (fun acc v -> if List.exists (Ir.Value.equal v) acc then acc else v :: acc)
        v12 v3

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
      Printf.sprintf "(0.5f * %s * (1.0f + metal::tanh(0.79788456f * (%s + 0.044715f * %s * %s * %s))))"
        e e e e e
  | Unop (Exp, a) ->
      Printf.sprintf "metal::exp(%s)" (emit_msl_expr ?value_var ~slot_emitter ~index_expr a)
  | Unop (Log, a) ->
      Printf.sprintf "metal::log(%s)" (emit_msl_expr ?value_var ~slot_emitter ~index_expr a)
  | Unop (Rsqrt, a) ->
      Printf.sprintf "metal::rsqrt(%s)" (emit_msl_expr ?value_var ~slot_emitter ~index_expr a)
  | Unop (Cast Ir.Dtype.Float16, a) ->
      Printf.sprintf "((half)(%s))" (emit_msl_expr ?value_var ~slot_emitter ~index_expr a)
  | Unop (Cast _, a) -> emit_msl_expr ?value_var ~slot_emitter ~index_expr a
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
