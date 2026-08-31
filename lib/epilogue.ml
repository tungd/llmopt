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

let inputs = function
  | None | Silu | Gelu -> []
  | Bias { bias } -> [ bias ]
  | Silu_mul { up } | Gelu_mul { up } -> [ up ]
  | Residual_add { residual } -> [ residual ]
  | Bias_residual_add { bias; residual } -> [ bias; residual ]
  | Silu_mul_residual_add { up; residual }
  | Gelu_mul_residual_add { up; residual } -> [ up; residual ]

let value_equal a b = Ir.Value.equal a b

let equal left right =
  match left, right with
  | None, None -> true
  | Silu, Silu -> true
  | Gelu, Gelu -> true
  | Bias { bias = b1 }, Bias { bias = b2 } -> value_equal b1 b2
  | Silu_mul { up = u1 }, Silu_mul { up = u2 } -> value_equal u1 u2
  | Gelu_mul { up = u1 }, Gelu_mul { up = u2 } -> value_equal u1 u2
  | Residual_add { residual = r1 }, Residual_add { residual = r2 } -> value_equal r1 r2
  | Bias_residual_add { bias = b1; residual = r1 }, Bias_residual_add { bias = b2; residual = r2 } ->
      value_equal b1 b2 && value_equal r1 r2
  | Silu_mul_residual_add { up = u1; residual = r1 }, Silu_mul_residual_add { up = u2; residual = r2 } ->
      value_equal u1 u2 && value_equal r1 r2
  | Gelu_mul_residual_add { up = u1; residual = r1 }, Gelu_mul_residual_add { up = u2; residual = r2 } ->
      value_equal u1 u2 && value_equal r1 r2
  | _ -> false

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

let compose ~producer ~consumer =
  match producer, consumer with
  | None, consumer -> Ok consumer
  | producer, None -> Ok producer
  | Bias { bias }, Residual_add { residual } ->
      Ok (Bias_residual_add { bias; residual })
  | Residual_add { residual }, Bias { bias } ->
      Ok (Bias_residual_add { bias; residual })
  | Silu_mul { up }, Residual_add { residual } ->
      Ok (Silu_mul_residual_add { up; residual })
  | Gelu_mul { up }, Residual_add { residual } ->
      Ok (Gelu_mul_residual_add { up; residual })
  | p, c ->
      Error (Printf.sprintf "Cannot compose epilogue %s with %s" (to_string p) (to_string c))

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
