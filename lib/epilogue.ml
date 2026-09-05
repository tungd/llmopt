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

let compose ~producer ~consumer =
  match producer, consumer with
  | None, consumer -> consumer
  | producer, None -> producer
  | Bias { bias }, Residual_add { residual } ->
      Bias_residual_add { bias; residual }
  | Residual_add { residual }, Bias { bias } ->
      Bias_residual_add { bias; residual }
  | Silu_mul { up }, Residual_add { residual } ->
      Silu_mul_residual_add { up; residual }
  | Gelu_mul { up }, Residual_add { residual } ->
      Gelu_mul_residual_add { up; residual }
  | p, c ->
      let ep = to_expr p in
      let ec = to_expr c in
      let composed = Applicative_expr.subst (function Acc -> ep) ec in
      Expr (Applicative_expr.simplify composed)

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

module Monoid = struct
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
