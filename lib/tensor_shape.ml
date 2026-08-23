type t = {
  dimensions : int array;
  numel : int;
}

let ( let* ) = Result.bind

module Index = struct
  module Spec = struct
    type t =
      | At of int
      | Slice of {
          start : int option;
          stop : int option;
          step : int option;
        }
      | New_axis
      | Ellipsis
  end

  type selector =
    | At of int
    | Slice of {
        start : int;
        step : int;
        length : int;
      }
    | New_axis

  type t = selector list

  let of_selectors selectors =
    let rec validate = function
      | [] -> Ok selectors
      | At index :: _ when index < 0 ->
          Error "normalized tensor index contains a negative integer"
      | Slice { step = 0; _ } :: _ ->
          Error "normalized tensor index contains a zero slice step"
      | Slice { length; _ } :: _ when length < 0 ->
          Error "normalized tensor index contains a negative slice length"
      | _ :: rest -> validate rest
    in
    validate selectors

  let selectors index = index

  let selector_to_string = function
    | At index -> string_of_int index
    | Slice { start; step; length } ->
        Printf.sprintf "%d:%d:%d" start step length
    | New_axis -> "newaxis"

  let to_string index =
    index |> List.map selector_to_string |> String.concat ","
    |> Printf.sprintf "index[%s]"
end

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
  | Multiple_ellipsis
  | Too_many_indices of { rank : int; consumed : int }
  | Index_out_of_bounds of { axis : int; dimension : int; index : int }
  | Zero_slice_step
  | Empty_concat
  | Concat_rank_mismatch of { expected : int; actual : int }
  | Concat_dimension_mismatch of {
      axis : int;
      expected : int;
      actual : int;
    }
  | Concat_dimension_overflow of int
  | Invalid_chunk_count of int
  | Invalid_depthwise_conv1d of string
  | Convolution_dimension_overflow
  | Malformed_index of string

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
  | Multiple_ellipsis -> "tensor index contains more than one ellipsis"
  | Too_many_indices { rank; consumed } ->
      Printf.sprintf "tensor rank %d cannot consume %d index axes" rank consumed
  | Index_out_of_bounds { axis; dimension; index } ->
      Printf.sprintf "tensor index %d is out of bounds for axis %d of size %d"
        index axis dimension
  | Zero_slice_step -> "tensor slice step cannot be zero"
  | Empty_concat -> "tensor concat requires at least one input"
  | Concat_rank_mismatch { expected; actual } ->
      Printf.sprintf "tensor concat expected rank %d, got rank %d" expected actual
  | Concat_dimension_mismatch { axis; expected; actual } ->
      Printf.sprintf "tensor concat axis %d expected size %d, got %d" axis
        expected actual
  | Concat_dimension_overflow axis ->
      Printf.sprintf "tensor concat dimension overflows at axis %d" axis
  | Invalid_chunk_count chunks ->
      Printf.sprintf "tensor chunk count must be positive, got %d" chunks
  | Invalid_depthwise_conv1d message -> "invalid depthwise conv1d: " ^ message
  | Convolution_dimension_overflow ->
      "depthwise conv1d output dimension overflows"
  | Malformed_index message -> "malformed normalized tensor index: " ^ message

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

let depthwise_conv1d source weight ~stride ~padding ~dilation ~groups =
  let invalid message = Error (Invalid_depthwise_conv1d message) in
  match dimensions source, dimensions weight with
  | [ batch; channels; input_width ], [ output_channels; channels_per_group; kernel_width ] ->
      if stride <= 0 then invalid "stride must be positive"
      else if padding < 0 then invalid "padding must be non-negative"
      else if dilation <= 0 then invalid "dilation must be positive"
      else if groups <= 0 then invalid "groups must be positive"
      else if channels = 0 then invalid "input channels must be positive"
      else if groups <> channels then
        invalid "groups must equal the input channel count"
      else if output_channels <> channels then
        invalid "weight output channels must equal the input channel count"
      else if channels_per_group <> 1 then
        invalid "weight must contain one input channel per group"
      else if kernel_width <= 0 then invalid "kernel width must be positive"
      else if kernel_width - 1 > (max_int - 1) / dilation then
        Error Convolution_dimension_overflow
      else
        let effective_kernel = (dilation * (kernel_width - 1)) + 1 in
        if padding > max_int / 2 then Error Convolution_dimension_overflow
        else
          let double_padding = 2 * padding in
          if input_width > max_int - double_padding then
            Error Convolution_dimension_overflow
          else
            let padded_width = input_width + double_padding in
            if padded_width < effective_kernel then
              invalid "effective kernel exceeds the padded input width"
            else
              let output_width =
                ((padded_width - effective_kernel) / stride) + 1
              in
              create [ batch; channels; output_width ]
  | _ -> invalid "input and weight must both have rank three"

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

let clamp ~minimum ~maximum value = Int.max minimum (Int.min maximum value)

let normalize_slice dimension ~start ~stop ~step =
  let step = Option.value step ~default:1 in
  if step = 0 then Error Zero_slice_step
  else if step > 0 then
    let normalize endpoint default =
      match endpoint with
      | None -> default
      | Some value ->
          let value = if value < 0 then value + dimension else value in
          clamp ~minimum:0 ~maximum:dimension value
    in
    let start = normalize start 0 in
    let stop = normalize stop dimension in
    let length =
      if start >= stop then 0 else ((stop - start - 1) / step) + 1
    in
    Ok (start, step, length)
  else
    let normalize endpoint default =
      match endpoint with
      | None -> default
      | Some value ->
          let value = if value < 0 then value + dimension else value in
          clamp ~minimum:(-1) ~maximum:(dimension - 1) value
    in
    let start = normalize start (dimension - 1) in
    let stop = normalize stop (-1) in
    let length =
      if start <= stop then 0
      else
        let distance = Int64.of_int (start - stop - 1) in
        let magnitude = Int64.neg (Int64.of_int step) in
        Int64.(to_int (add (div distance magnitude) 1L))
    in
    Ok (start, step, length)

type expanded_index =
  | Expanded_at of int
  | Expanded_slice of {
      start : int option;
      stop : int option;
      step : int option;
    }
  | Expanded_new_axis

let index shape (specs : Index.Spec.t list) =
  let rank = rank shape in
  let consumes_axis (spec : Index.Spec.t) =
    match spec with
    | Index.Spec.At _ | Index.Spec.Slice _ -> 1
    | Index.Spec.New_axis | Index.Spec.Ellipsis -> 0
  in
  let consumed =
    specs |> List.map consumes_axis |> List.fold_left ( + ) 0
  in
  if consumed > rank then Error (Too_many_indices { rank; consumed })
  else
    let ellipsis_count =
      List.fold_left
        (fun count -> function Index.Spec.Ellipsis -> count + 1 | _ -> count)
        0 specs
    in
    if ellipsis_count > 1 then Error Multiple_ellipsis
    else
      let missing = rank - consumed in
      let expanded_full_slice =
        Expanded_slice { start = None; stop = None; step = None }
      in
      let rec expand acc = function
        | [] when ellipsis_count = 0 ->
            Ok
              (List.rev_append acc
                 (List.init missing (fun _ -> expanded_full_slice)))
        | [] -> Ok (List.rev acc)
        | Index.Spec.Ellipsis :: rest ->
            expand
              (List.rev_append
                 (List.init missing (fun _ -> expanded_full_slice))
                 acc)
              rest
        | Index.Spec.At index :: rest -> expand (Expanded_at index :: acc) rest
        | Index.Spec.Slice { start; stop; step } :: rest ->
            expand (Expanded_slice { start; stop; step } :: acc) rest
        | Index.Spec.New_axis :: rest -> expand (Expanded_new_axis :: acc) rest
      in
      let* specs = expand [] specs in
      let dimensions = shape.dimensions in
      let rec normalize source_axis selectors output_dimensions = function
        | [] ->
            let* output = create (List.rev output_dimensions) in
            Ok (List.rev selectors, output)
        | Expanded_new_axis :: rest ->
            normalize source_axis (Index.New_axis :: selectors)
              (1 :: output_dimensions) rest
        | Expanded_at raw_index :: rest ->
            let dimension = dimensions.(source_axis) in
            let index = if raw_index < 0 then raw_index + dimension else raw_index in
            if index < 0 || index >= dimension then
              Error
                (Index_out_of_bounds
                   { axis = source_axis; dimension; index = raw_index })
            else
              normalize (source_axis + 1) (Index.At index :: selectors)
                output_dimensions rest
        | Expanded_slice { start; stop; step } :: rest ->
            let dimension = dimensions.(source_axis) in
            let* start, step, length =
              normalize_slice dimension ~start ~stop ~step
            in
            let selector = Index.Slice { start; step; length } in
            normalize (source_axis + 1) (selector :: selectors)
              (length :: output_dimensions) rest
      in
      normalize 0 [] [] specs

let apply_index shape index =
  let selectors = Index.selectors index in
  let source_rank = rank shape in
  let rec apply source_axis output_dimensions = function
    | [] ->
        if source_axis <> source_rank then
          Error
            (Malformed_index
               (Printf.sprintf "consumes %d axes from rank %d" source_axis
                  source_rank))
        else create (List.rev output_dimensions)
    | Index.New_axis :: rest ->
        apply source_axis (1 :: output_dimensions) rest
    | Index.At index :: rest ->
        if source_axis >= source_rank then
          Error (Malformed_index "integer selector exceeds source rank")
        else
          let dimension = shape.dimensions.(source_axis) in
          if index < 0 || index >= dimension then
            Error
              (Index_out_of_bounds
                 { axis = source_axis; dimension; index })
          else apply (source_axis + 1) output_dimensions rest
    | Index.Slice { start; step; length } :: rest ->
        if source_axis >= source_rank then
          Error (Malformed_index "slice selector exceeds source rank")
        else if step = 0 then Error Zero_slice_step
        else if length < 0 then
          Error (Malformed_index "slice length is negative")
        else
          let dimension = shape.dimensions.(source_axis) in
          let valid =
            if length = 0 then start >= -1 && start <= dimension
            else
              let last =
                Int64.add (Int64.of_int start)
                  (Int64.mul (Int64.of_int (length - 1)) (Int64.of_int step))
              in
              start >= 0 && start < dimension && last >= 0L
              && last < Int64.of_int dimension
          in
          if not valid then
            Error
              (Malformed_index
                 (Printf.sprintf
                    "slice start=%d step=%d length=%d escapes axis %d of size %d"
                    start step length source_axis dimension))
          else apply (source_axis + 1) (length :: output_dimensions) rest
  in
  apply 0 [] selectors

let concat shapes ~axis =
  match shapes with
  | [] -> Error Empty_concat
  | first :: rest ->
      let* axis = normalize_axis first axis in
      let expected_rank = rank first in
      let output_dimensions = Array.copy first.dimensions in
      let rec add_dimension total = function
        | [] -> Ok total
        | shape :: remaining ->
            let actual_rank = rank shape in
            if actual_rank <> expected_rank then
              Error
                (Concat_rank_mismatch
                   { expected = expected_rank; actual = actual_rank })
            else
              let rec validate current_axis =
                if current_axis = expected_rank then Ok ()
                else if current_axis = axis then validate (current_axis + 1)
                else
                  let expected = first.dimensions.(current_axis) in
                  let actual = shape.dimensions.(current_axis) in
                  if expected = actual then validate (current_axis + 1)
                  else
                    Error
                      (Concat_dimension_mismatch
                         { axis = current_axis; expected; actual })
              in
              let* () = validate 0 in
              let dimension = shape.dimensions.(axis) in
              if dimension > max_int - total then
                Error (Concat_dimension_overflow axis)
              else add_dimension (total + dimension) remaining
      in
      let* dimension = add_dimension first.dimensions.(axis) rest in
      output_dimensions.(axis) <- dimension;
      create (Array.to_list output_dimensions)

let chunk shape ~chunks ~axis =
  if chunks <= 0 then Error (Invalid_chunk_count chunks)
  else
    let* axis = normalize_axis shape axis in
    let dimension = shape.dimensions.(axis) in
    let chunk_size =
      if dimension = 0 then 0 else 1 + ((dimension - 1) / chunks)
    in
    let output_count =
      if dimension = 0 then chunks
      else 1 + ((dimension - 1) / chunk_size)
    in
    let rec build acc chunk_index =
      if chunk_index = output_count then Ok (List.rev acc)
      else
        let start = chunk_index * chunk_size in
        let stop = Int.min dimension (start + chunk_size) in
        let specs =
          List.init (rank shape) (fun current_axis ->
              if current_axis = axis then
                Index.Spec.Slice
                  { start = Some start; stop = Some stop; step = Some 1 }
              else
                Index.Spec.Slice { start = None; stop = None; step = None })
        in
        let* selection, output = index shape specs in
        build ((selection, output) :: acc) (chunk_index + 1)
    in
    build [] 0

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
