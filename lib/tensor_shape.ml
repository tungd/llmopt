type t = {
  dimensions : int array;
  numel : int;
}

type error =
  | Negative_dimension of { axis : int; value : int }
  | Element_count_overflow of int list
  | Matrix_projection_has_zero_elements of t
  | Matrix_projection_overflow of t
  | Invalid_axis of { rank : int; axis : int; allow_end : bool }
  | Duplicate_axis of int
  | Incompatible_broadcast of { left : t; right : t }
  | Element_count_mismatch of { source : t; target : t }
  | Invalid_expansion of { source : t; target : t }

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
  | Invalid_axis { rank; axis; allow_end } ->
      let upper = if allow_end then rank else rank - 1 in
      Printf.sprintf "tensor axis %d is outside [%d,%d] for rank %d" axis
        (-rank - if allow_end then 1 else 0) upper rank
  | Duplicate_axis axis ->
      Printf.sprintf "tensor axis %d appears more than once" axis
  | Incompatible_broadcast { left; right } ->
      Printf.sprintf "tensor shapes %s and %s do not broadcast" (to_string left)
        (to_string right)
  | Element_count_mismatch { source; target } ->
      Printf.sprintf "cannot reshape %s (%d elements) as %s (%d elements)"
        (to_string source) source.numel (to_string target) target.numel
  | Invalid_expansion { source; target } ->
      Printf.sprintf "cannot expand tensor %s to %s" (to_string source)
        (to_string target)

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

let normalize_axis ?(allow_end = false) shape axis =
  let rank = rank shape in
  let lower = -rank - if allow_end then 1 else 0 in
  let upper = if allow_end then rank else rank - 1 in
  if axis < lower || axis > upper then
    Error (Invalid_axis { rank; axis; allow_end })
  else if axis < 0 then Ok (axis + rank + if allow_end then 1 else 0)
  else Ok axis

let normalize_axes shape axes =
  let rec loop seen acc = function
    | [] -> Ok (List.sort Int.compare acc)
    | axis :: rest ->
        (match normalize_axis shape axis with
        | Error _ as error -> error
        | Ok axis when List.mem axis seen -> Error (Duplicate_axis axis)
        | Ok axis -> loop (axis :: seen) (axis :: acc) rest)
  in
  loop [] [] axes

let broadcast left right =
  let left_dimensions = dimensions left |> List.rev in
  let right_dimensions = dimensions right |> List.rev in
  let rec merge acc left_dimensions right_dimensions =
    match left_dimensions, right_dimensions with
    | [], [] -> create acc
    | left :: rest, [] | [], left :: rest -> merge (left :: acc) rest []
    | left :: left_rest, right :: right_rest
      when left = right || left = 1 || right = 1 ->
        merge (Int.max left right :: acc) left_rest right_rest
    | _ -> Error (Incompatible_broadcast { left; right })
  in
  merge [] left_dimensions right_dimensions

let reshape source target =
  if source.numel = target.numel then Ok target
  else Error (Element_count_mismatch { source; target })

let replace_nth index value values =
  List.mapi (fun current existing -> if current = index then value else existing) values

let transpose shape ~axis0 ~axis1 =
  match normalize_axis shape axis0, normalize_axis shape axis1 with
  | Error _ as error, _ | _, (Error _ as error) -> error
  | Ok axis0, Ok axis1 ->
      let dimensions = shape.dimensions in
      let left = dimensions.(axis0) in
      let right = dimensions.(axis1) in
      dimensions |> Array.to_list |> replace_nth axis0 right
      |> replace_nth axis1 left |> create

let unsqueeze shape ~axis =
  match normalize_axis ~allow_end:true shape axis with
  | Error _ as error -> error
  | Ok axis ->
      let rec insert index = function
        | values when index = 0 -> 1 :: values
        | value :: rest -> value :: insert (index - 1) rest
        | [] -> [ 1 ]
      in
      dimensions shape |> insert axis |> create

let expand source ~target =
  let source_dimensions = dimensions source |> List.rev in
  let target_dimensions = dimensions target |> List.rev in
  let rec compatible source_dimensions target_dimensions =
    match source_dimensions, target_dimensions with
    | [], _ -> true
    | _ :: _, [] -> false
    | source :: source_rest, target :: target_rest ->
        (source = target || source = 1)
        && compatible source_rest target_rest
  in
  if compatible source_dimensions target_dimensions then Ok target
  else Error (Invalid_expansion { source; target })

let reduce shape ~axes ~keepdim =
  match normalize_axes shape axes with
  | Error _ as error -> error
  | Ok axes ->
      dimensions shape
      |> List.mapi (fun axis dimension -> axis, dimension)
      |> List.filter_map (fun (axis, dimension) ->
             if List.mem axis axes then if keepdim then Some 1 else None
             else Some dimension)
      |> create

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
