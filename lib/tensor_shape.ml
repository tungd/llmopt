type t = {
  dimensions : int array;
  numel : int;
}

type error =
  | Negative_dimension of { axis : int; value : int }
  | Element_count_overflow of int list
  | Matrix_projection_has_zero_elements of t
  | Matrix_projection_overflow of t

let to_string shape =
  match Array.to_list shape.dimensions with
  | [] -> "[]"
  | dimensions ->
      dimensions |> List.map string_of_int |> String.concat "x"
      |> Printf.sprintf "[%s]"

let error_to_string = function
  | Negative_dimension { axis; value } ->
      Printf.sprintf "tensor dimension %d must be non-negative, got %d" axis value
  | Element_count_overflow dimensions ->
      Printf.sprintf "tensor element count overflows for shape [%s]"
        (dimensions |> List.map string_of_int |> String.concat "x")
  | Matrix_projection_has_zero_elements shape ->
      Printf.sprintf "cannot project zero-element tensor %s to a tile matrix"
        (to_string shape)
  | Matrix_projection_overflow shape ->
      Printf.sprintf "matrix projection overflows for tensor %s" (to_string shape)

let create dimensions =
  let rec validate axis product = function
    | [] -> Ok product
    | value :: _ when value < 0 -> Error (Negative_dimension { axis; value })
    | 0 :: rest -> validate (axis + 1) 0 rest
    | value :: rest ->
        if product <> 0 && value > max_int / product then
          Error (Element_count_overflow dimensions)
        else validate (axis + 1) (product * value) rest
  in
  match validate 0 1 dimensions with
  | Error _ as error -> error
  | Ok numel -> Ok { dimensions = Array.of_list dimensions; numel }

let of_ints_exn dimensions =
  match create dimensions with
  | Ok shape -> shape
  | Error error -> invalid_arg (error_to_string error)

let scalar = of_ints_exn []

let of_matrix shape =
  of_ints_exn [ Shape.rows shape; Shape.cols shape ]

let dimensions shape = Array.to_list shape.dimensions
let rank shape = Array.length shape.dimensions
let numel shape = shape.numel
let equal left right = left = right

let matrix shape =
  if shape.numel = 0 then Error (Matrix_projection_has_zero_elements shape)
  else
    let dimensions = dimensions shape in
    let rows, cols =
      match List.rev dimensions with
      | [] -> 1, 1
      | [ only ] -> 1, only
      | last :: leading ->
          let rows = List.fold_left ( * ) 1 leading in
          rows, last
    in
    match Shape.create ~rows ~cols with
    | Ok matrix -> Ok matrix
    | Error (Shape.Non_positive_dimension _) ->
        Error (Matrix_projection_has_zero_elements shape)
    | Error _ -> Error (Matrix_projection_overflow shape)

let matrix_exn shape =
  match matrix shape with
  | Ok matrix -> matrix
  | Error error -> invalid_arg (error_to_string error)
