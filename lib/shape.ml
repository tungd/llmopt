type t = { rows : int; cols : int }

type error =
  | Non_positive_dimension of string * int
  | Not_matrix_multipliable of t * t
  | Not_broadcastable of t * t

type broadcast = Same | Row

let to_string shape = Printf.sprintf "%dx%d" shape.rows shape.cols

let error_to_string = function
  | Non_positive_dimension (name, value) ->
      Printf.sprintf "%s must be positive, got %d" name value
  | Not_matrix_multipliable (left, right) ->
      Printf.sprintf "cannot multiply %s by %s" (to_string left) (to_string right)
  | Not_broadcastable (left, right) ->
      Printf.sprintf "cannot add %s and %s" (to_string left) (to_string right)

let create ~rows ~cols =
  if rows <= 0 then Error (Non_positive_dimension ("rows", rows))
  else if cols <= 0 then Error (Non_positive_dimension ("cols", cols))
  else Ok { rows; cols }

let of_ints_exn ~rows ~cols =
  match create ~rows ~cols with
  | Ok shape -> shape
  | Error error -> invalid_arg (error_to_string error)

let rows shape = shape.rows
let cols shape = shape.cols
let equal left right = left = right

let matmul left right =
  if left.cols <> right.rows then Error (Not_matrix_multipliable (left, right))
  else Ok { rows = left.rows; cols = right.cols }

let add left right =
  if equal left right then Ok (left, Same)
  else if left.rows > 0 && left.cols = right.cols && right.rows = 1 then
    Ok (left, Row)
  else if right.rows > 0 && left.cols = right.cols && left.rows = 1 then
    Ok (right, Row)
  else Error (Not_broadcastable (left, right))
