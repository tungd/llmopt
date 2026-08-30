open Bigarray

module Tensor = struct
  type t = (float, float32_elt, c_layout) Array2.t

  let create shape = Array2.create float32 c_layout (Shape.rows shape) (Shape.cols shape)

  let of_rows rows =
    let row_count = Array.length rows in
    if row_count = 0 then invalid_arg "Tensor.of_rows: empty input";
    let col_count = Array.length rows.(0) in
    if col_count = 0 then invalid_arg "Tensor.of_rows: empty row";
    Array.iter
      (fun row ->
        if Array.length row <> col_count then invalid_arg "Tensor.of_rows: ragged input")
      rows;
    let tensor = create (Shape.of_ints_exn ~rows:row_count ~cols:col_count) in
    Array.iteri
      (fun row values -> Array.iteri (fun col value -> Array2.set tensor row col value) values)
      rows;
    tensor

  let get tensor row col = Array2.get tensor row col
  let set tensor row col value = Array2.set tensor row col value
  let get_linear tensor index =
    let cols = Array2.dim2 tensor in
    get tensor (index / cols) (index mod cols)
  let set_linear tensor index value =
    let cols = Array2.dim2 tensor in
    set tensor (index / cols) (index mod cols) value
  let fill tensor value = Array2.fill tensor value
  let shape tensor =
    Shape.of_ints_exn ~rows:(Array2.dim1 tensor) ~cols:(Array2.dim2 tensor)

  let to_rows tensor =
    Array.init (Array2.dim1 tensor) (fun row ->
        Array.init (Array2.dim2 tensor) (fun col -> get tensor row col))
end

type execution = { outputs : (string, Tensor.t) Hashtbl.t }

let output execution name = Hashtbl.find_opt execution.outputs name

let outputs execution =
  Hashtbl.to_seq execution.outputs |> List.of_seq

type state = {
  input_values : (string, Tensor.t) Hashtbl.t;
  values : (int, Tensor.t) Hashtbl.t;
  execution : execution;
  mutable next_value : int;
}

let failf format = Printf.ksprintf (fun message -> raise (Failure message)) format

let fresh_value state ~logical_shape ~shape ~dtype =
  let value =
    Ir.Value.make_tensor ~id:state.next_value ~shape:logical_shape ~dtype
  in
  if not (Shape.equal (Ir.Value.shape value) shape) then
    invalid_arg "effect logical shape does not match CPU tile shape";
  state.next_value <- state.next_value + 1;
  value

let bind state value tensor =
  let expected = Ir.Value.shape value in
  let actual = Tensor.shape tensor in
  if not (Shape.equal expected actual) then
    failf "CPU shape mismatch for value %d: expected %s, got %s"
      (Ir.Value_id.to_int (Ir.Value.id value)) (Shape.to_string expected)
      (Shape.to_string actual);
  Hashtbl.replace state.values (Ir.Value_id.to_int (Ir.Value.id value)) tensor

let find state value =
  match Hashtbl.find_opt state.values (Ir.Value_id.to_int (Ir.Value.id value)) with
  | Some tensor -> tensor
  | None ->
      failf "CPU value %d has no storage"
        (Ir.Value_id.to_int (Ir.Value.id value))

let copy_into source destination =
  let source_shape = Tensor.shape source in
  let destination_shape = Tensor.shape destination in
  if not (Shape.equal source_shape destination_shape) then
    failf "CPU copy shape mismatch: %s -> %s" (Shape.to_string source_shape)
      (Shape.to_string destination_shape);
  for row = 0 to Shape.rows source_shape - 1 do
    for col = 0 to Shape.cols source_shape - 1 do
      Tensor.set destination row col (Tensor.get source row col)
    done
  done

let matmul left right output =
  let left_shape = Tensor.shape left in
  let output_shape = Tensor.shape output in
  for row = 0 to Shape.rows output_shape - 1 do
    for col = 0 to Shape.cols output_shape - 1 do
      let accumulator = ref 0.0 in
      for inner = 0 to Shape.cols left_shape - 1 do
        accumulator :=
          !accumulator
          +. (Tensor.get left row inner *. Tensor.get right inner col)
      done;
      Tensor.set output row col !accumulator
    done
  done

let linear input weight output bias =
  let input_shape = Tensor.shape input in
  let output_shape = Tensor.shape output in
  for row = 0 to Shape.rows output_shape - 1 do
    for col = 0 to Shape.cols output_shape - 1 do
      let accumulator = ref 0.0 in
      for inner = 0 to Shape.cols input_shape - 1 do
        accumulator :=
          !accumulator
          +. (Tensor.get input row inner *. Tensor.get weight col inner)
      done;
      let value =
        match bias with
        | None -> !accumulator
        | Some bias -> !accumulator +. Tensor.get bias 0 col
      in
      Tensor.set output row col value
  done
  done

let w4a16_linear input packed_weight scale output bias =
  let input_shape = Tensor.shape input in
  let output_shape = Tensor.shape output in
  let signed_nibble nibble = if nibble land 0x8 = 0 then nibble else nibble - 16 in
  for row = 0 to Shape.rows output_shape - 1 do
    for col = 0 to Shape.cols output_shape - 1 do
      let accumulator = ref 0.0 in
      for inner = 0 to Shape.cols input_shape - 1 do
        let byte =
          Tensor.get packed_weight col (inner / 2) |> Float.round |> int_of_float
        in
        let nibble = if inner land 1 = 0 then byte land 0xF else byte lsr 4 land 0xF in
        let weight = Float.of_int (signed_nibble nibble) in
        let group_scale = Tensor.get scale col (inner / 64) in
        accumulator :=
          !accumulator +. (Tensor.get input row inner *. weight *. group_scale)
      done;
      let value =
        match bias with
        | None -> !accumulator
        | Some bias -> !accumulator +. Tensor.get bias 0 col
      in
      Tensor.set output row col value
    done
  done

let add ~left ~right ~output ~broadcast =
  let output_shape = Tensor.shape output in
  let left_shape = Tensor.shape left in
  let right_shape = Tensor.shape right in
  for row = 0 to Shape.rows output_shape - 1 do
    for col = 0 to Shape.cols output_shape - 1 do
      let left_row, right_row =
        match broadcast with
        | Shape.Same -> row, row
        | Shape.Row ->
            ((if Shape.rows left_shape = 1 then 0 else row),
             (if Shape.rows right_shape = 1 then 0 else row))
      in
      Tensor.set output row col
        (Tensor.get left left_row col +. Tensor.get right right_row col)
    done
  done

let gelu input output =
  let shape = Tensor.shape input in
  for row = 0 to Shape.rows shape - 1 do
    for col = 0 to Shape.cols shape - 1 do
      let value = Tensor.get input row col in
      let scaled = sqrt 2.0 *. (value +. 0.044715 *. value *. value *. value) in
      Tensor.set output row col (0.5 *. value *. (1.0 +. tanh (scaled *. 0.5)))
    done
  done

let relu input output =
  let shape = Tensor.shape input in
  for row = 0 to Shape.rows shape - 1 do
    for col = 0 to Shape.cols shape - 1 do
      Tensor.set output row col (Float.max 0.0 (Tensor.get input row col))
    done
  done

let coordinates dimensions linear =
  let coordinates = Array.make (Array.length dimensions) 0 in
  let remaining = ref linear in
  for axis = Array.length dimensions - 1 downto 0 do
    let dimension = dimensions.(axis) in
    coordinates.(axis) <- !remaining mod dimension;
    remaining := !remaining / dimension
  done;
  coordinates

let linear_index dimensions coordinates =
  let index = ref 0 in
  Array.iteri
    (fun axis coordinate -> index := (!index * dimensions.(axis)) + coordinate)
    coordinates;
  !index

let broadcast_index ~source_shape ~output_dimensions output_coordinates =
  let source_dimensions =
    Ir.Value.logical_shape source_shape |> Tensor_shape.dimensions |> Array.of_list
  in
  let rank_delta = Array.length output_dimensions - Array.length source_dimensions in
  let source_coordinates =
    Array.mapi
      (fun axis dimension ->
        if dimension = 1 then 0 else output_coordinates.(axis + rank_delta))
      source_dimensions
  in
  linear_index source_dimensions source_coordinates

let scalar_truth value = value <> 0.0

let pointwise_unary operator value =
  match operator with
  | Ir.Pointwise.Neg -> -.value
  | Ir.Pointwise.Rsqrt -> 1.0 /. sqrt value
  | Ir.Pointwise.Silu -> value /. (1.0 +. exp (-.value))
  | Ir.Pointwise.Cos -> cos value
  | Ir.Pointwise.Sin -> sin value
  | Ir.Pointwise.Tanh -> tanh value
  | Ir.Pointwise.Exp -> exp value
  | Ir.Pointwise.Sigmoid -> 1.0 /. (1.0 +. exp (-.value))
  | Ir.Pointwise.Softplus -> if value > 20.0 then value else log1p (exp value)
  | Ir.Pointwise.Pow exponent -> value ** Ir.Scalar.to_float exponent

let pointwise_binary operator left right =
  match operator with
  | Ir.Pointwise.Add -> left +. right
  | Ir.Pointwise.Mul -> left *. right
  | Ir.Pointwise.Silu_mul -> (left /. (1.0 +. exp (-.left))) *. right
  | Ir.Pointwise.Gelu_mul ->
      let scaled = sqrt 2.0 *. (left +. (0.044715 *. left *. left *. left)) in
      (0.5 *. left *. (1.0 +. tanh (scaled *. 0.5))) *. right
  | Ir.Pointwise.Sigmoid_mul -> (1.0 /. (1.0 +. exp (-.left))) *. right
  | Ir.Pointwise.Sub -> left -. right
  | Ir.Pointwise.Div -> left /. right
  | Ir.Pointwise.Logical_and ->
      if scalar_truth left && scalar_truth right then 1.0 else 0.0
  | Ir.Pointwise.Equal -> if left = right then 1.0 else 0.0
  | Ir.Pointwise.Not_equal -> if left <> right then 1.0 else 0.0
  | Ir.Pointwise.Less_equal -> if left <= right then 1.0 else 0.0
  | Ir.Pointwise.Greater -> if left > right then 1.0 else 0.0

let pointwise state operation output_value output =
  let output_shape = Ir.Value.logical_shape output_value in
  let output_dimensions = Tensor_shape.dimensions output_shape |> Array.of_list in
  let operand_value output_coordinates = function
    | Ir.Pointwise.Scalar scalar -> Ir.Scalar.to_float scalar
    | Ir.Pointwise.Tensor value ->
        let source_index =
          broadcast_index ~source_shape:value ~output_dimensions output_coordinates
        in
        Tensor.get_linear (find state value) source_index
  in
  for index = 0 to Tensor_shape.numel output_shape - 1 do
    let output_coordinates = coordinates output_dimensions index in
    let value =
      match operation with
      | Ir.Pointwise.Unary (operator, input) ->
          pointwise_unary operator
            (operand_value output_coordinates (Ir.Pointwise.Tensor input))
      | Ir.Pointwise.Binary (operator, left, right) ->
          pointwise_binary operator (operand_value output_coordinates left)
            (operand_value output_coordinates right)
    in
    Tensor.set_linear output index value
  done

let reduce state reduction input_value output_value output =
  let input_shape = Ir.Value.logical_shape input_value in
  let output_shape = Ir.Value.logical_shape output_value in
  let input_dimensions = Tensor_shape.dimensions input_shape |> Array.of_list in
  let output_dimensions = Tensor_shape.dimensions output_shape |> Array.of_list in
  let reduced = Array.make (Array.length input_dimensions) false in
  List.iter (fun axis -> reduced.(axis) <- true) reduction.Ir.Reduction.axes;
  let counts = Array.make (Tensor_shape.numel output_shape) 0 in
  Tensor.fill output 0.0;
  for input_index = 0 to Tensor_shape.numel input_shape - 1 do
    let input_coordinates = coordinates input_dimensions input_index in
    let output_coordinates =
      if reduction.keepdim then
        Array.mapi
          (fun axis coordinate -> if reduced.(axis) then 0 else coordinate)
          input_coordinates
      else
        input_coordinates |> Array.to_list
        |> List.mapi (fun axis coordinate -> axis, coordinate)
        |> List.filter_map (fun (axis, coordinate) ->
               if reduced.(axis) then None else Some coordinate)
        |> Array.of_list
    in
    let output_index = linear_index output_dimensions output_coordinates in
    Tensor.set_linear output output_index
      (Tensor.get_linear output output_index
      +. Tensor.get_linear (find state input_value) input_index);
    counts.(output_index) <- counts.(output_index) + 1
  done;
  match reduction.Ir.Reduction.operator with
  | Ir.Reduction.Sum -> ()
  | Ir.Reduction.Mean ->
      Array.iteri
        (fun index count ->
          Tensor.set_linear output index
            (Tensor.get_linear output index /. Float.of_int count))
        counts

let l2_norm state ~epsilon input_value output =
  let dimensions = Tensor_shape.dimensions (Ir.Value.logical_shape input_value) in
  let width =
    match List.rev dimensions with
    | width :: _ when width > 0 -> width
    | _ -> failf "L2 normalization requires a non-empty final dimension"
  in
  let input = find state input_value in
  let rows = Tensor_shape.numel (Ir.Value.logical_shape input_value) / width in
  for row = 0 to rows - 1 do
    let base = row * width in
    let square_sum = ref 0.0 in
    for column = 0 to width - 1 do
      let value = Tensor.get_linear input (base + column) in
      square_sum := !square_sum +. (value *. value)
    done;
    let inverse = 1.0 /. Float.sqrt (!square_sum +. epsilon) in
    for column = 0 to width - 1 do
      Tensor.set_linear output (base + column)
        (Tensor.get_linear input (base + column) *. inverse)
    done
  done

let apply_index state selection input_value output_value output =
  let output_shape = Ir.Value.logical_shape output_value in
  let output_dimensions = Tensor_shape.dimensions output_shape |> Array.of_list in
  let input_dimensions =
    Ir.Value.logical_shape input_value |> Tensor_shape.dimensions |> Array.of_list
  in
  let source = find state input_value in
  for index = 0 to Tensor_shape.numel output_shape - 1 do
    let output_coordinates = coordinates output_dimensions index in
    let input_coordinates = Array.make (Array.length input_dimensions) 0 in
    let source_axis = ref 0 in
    let output_axis = ref 0 in
    Tensor_shape.Index.selectors selection
    |> List.iter (function
         | Tensor_shape.Index.At selected ->
             input_coordinates.(!source_axis) <- selected;
             incr source_axis
         | Tensor_shape.Index.Slice { start; step; length = _ } ->
             input_coordinates.(!source_axis) <-
               start + (output_coordinates.(!output_axis) * step);
             incr source_axis;
             incr output_axis
         | Tensor_shape.Index.New_axis -> incr output_axis);
    let input_index = linear_index input_dimensions input_coordinates in
    Tensor.set_linear output index (Tensor.get_linear source input_index)
  done

let update_slice state selection destination_value source_value output =
  let destination_shape = Ir.Value.logical_shape destination_value in
  let source_shape = Ir.Value.logical_shape source_value in
  let destination_dimensions =
    Tensor_shape.dimensions destination_shape |> Array.of_list
  in
  let source_dimensions = Tensor_shape.dimensions source_shape |> Array.of_list in
  let destination = find state destination_value in
  let source = find state source_value in
  copy_into destination output;
  for index = 0 to Tensor_shape.numel source_shape - 1 do
    let source_coordinates = coordinates source_dimensions index in
    let destination_coordinates =
      Array.make (Array.length destination_dimensions) 0
    in
    let destination_axis = ref 0 in
    let source_axis = ref 0 in
    Tensor_shape.Index.selectors selection
    |> List.iter (function
         | Tensor_shape.Index.At selected ->
             destination_coordinates.(!destination_axis) <- selected;
             incr destination_axis
         | Tensor_shape.Index.Slice { start; step; length = _ } ->
             destination_coordinates.(!destination_axis) <-
               start + (source_coordinates.(!source_axis) * step);
             incr destination_axis;
             incr source_axis
         | Tensor_shape.Index.New_axis -> incr source_axis);
    let destination_index =
      linear_index destination_dimensions destination_coordinates
    in
    Tensor.set_linear output destination_index (Tensor.get_linear source index)
  done

let concatenate state axis input_values output_value output =
  let inputs = Array.of_list input_values in
  if Array.length inputs = 0 then failf "concat requires at least one input";
  let output_shape = Ir.Value.logical_shape output_value in
  let output_dimensions = Tensor_shape.dimensions output_shape |> Array.of_list in
  let input_dimensions =
    Array.map
      (fun value ->
        Ir.Value.logical_shape value |> Tensor_shape.dimensions |> Array.of_list)
      inputs
  in
  for index = 0 to Tensor_shape.numel output_shape - 1 do
    let output_coordinates = coordinates output_dimensions index in
    let selected_input = ref 0 in
    let selected_axis = ref output_coordinates.(axis) in
    while
      !selected_input < Array.length inputs - 1
      && !selected_axis >= input_dimensions.(!selected_input).(axis)
    do
      selected_axis :=
        !selected_axis - input_dimensions.(!selected_input).(axis);
      incr selected_input
    done;
    let input_coordinates = Array.copy output_coordinates in
    input_coordinates.(axis) <- !selected_axis;
    let input_index =
      linear_index input_dimensions.(!selected_input) input_coordinates
    in
    Tensor.set_linear output index
      (Tensor.get_linear (find state inputs.(!selected_input)) input_index)
  done

let apply_movement state movement input_value output_value output =
  let input_shape = Ir.Value.logical_shape input_value in
  let output_shape = Ir.Value.logical_shape output_value in
  let input_dimensions = Tensor_shape.dimensions input_shape |> Array.of_list in
  let output_dimensions = Tensor_shape.dimensions output_shape |> Array.of_list in
  let source = find state input_value in
  match movement with
  | Ir.Movement.View | Ir.Movement.Reshape | Ir.Movement.Unsqueeze _
  | Ir.Movement.Contiguous ->
      for index = 0 to Tensor_shape.numel output_shape - 1 do
        Tensor.set_linear output index (Tensor.get_linear source index)
      done
  | Ir.Movement.Expand ->
      for index = 0 to Tensor_shape.numel output_shape - 1 do
        let output_coordinates = coordinates output_dimensions index in
        let input_index =
          broadcast_index ~source_shape:input_value ~output_dimensions
            output_coordinates
        in
        Tensor.set_linear output index (Tensor.get_linear source input_index)
      done
  | Ir.Movement.Transpose { axis0; axis1 } ->
      for index = 0 to Tensor_shape.numel output_shape - 1 do
        let input_coordinates = coordinates output_dimensions index in
        let temporary = input_coordinates.(axis0) in
        input_coordinates.(axis0) <- input_coordinates.(axis1);
        input_coordinates.(axis1) <- temporary;
        let input_index = linear_index input_dimensions input_coordinates in
        Tensor.set_linear output index (Tensor.get_linear source input_index)
      done
  | Ir.Movement.Index selection ->
      apply_index state selection input_value output_value output
  | Ir.Movement.Concat _ -> failf "concat reached the unary movement handler"
  | Ir.Movement.Roll { axis; shift } ->
      let width = input_dimensions.(axis) in
      for index = 0 to Tensor_shape.numel output_shape - 1 do
        let output_coordinates = coordinates output_dimensions index in
        let input_coordinates = Array.copy output_coordinates in
        let shifted = (output_coordinates.(axis) - shift) mod width in
        input_coordinates.(axis) <- if shifted < 0 then shifted + width else shifted;
        let input_index = linear_index input_dimensions input_coordinates in
        Tensor.set_linear output index (Tensor.get_linear source input_index)
      done

let short_conv state config input_value weight_value output_value output =
  match
    Tensor_shape.dimensions (Ir.Value.logical_shape input_value),
    Tensor_shape.dimensions (Ir.Value.logical_shape weight_value),
    Tensor_shape.dimensions (Ir.Value.logical_shape output_value)
  with
  | ( [ batches; channels; input_width ],
      [ weight_channels; 1; kernel_width ],
      [ output_batches; output_channels; output_width ] )
    when batches = output_batches && channels = weight_channels
         && channels = output_channels ->
      let input = find state input_value in
      let weight = find state weight_value in
      let stride = Ir.Short_conv.stride config in
      let padding = Ir.Short_conv.padding config in
      let dilation = Ir.Short_conv.dilation config in
      for batch = 0 to batches - 1 do
        for channel = 0 to channels - 1 do
          for position = 0 to output_width - 1 do
            let accumulator = ref 0.0 in
            for kernel = 0 to kernel_width - 1 do
              let source_position =
                (position * stride) - padding + (kernel * dilation)
              in
              if source_position >= 0 && source_position < input_width then
                let input_index =
                  (((batch * channels) + channel) * input_width)
                  + source_position
                in
                let weight_index = (channel * kernel_width) + kernel in
                accumulator :=
                  !accumulator
                  +. (Tensor.get_linear input input_index
                     *. Tensor.get_linear weight weight_index)
            done;
            let output_index =
              (((batch * channels) + channel) * output_width) + position
            in
            Tensor.set_linear output output_index !accumulator
          done
        done
      done
  | _ -> failf "invalid CPU short-conv tensor metadata"

let short_conv_silu state config input_value weight_value output_value output =
  match
    Tensor_shape.dimensions (Ir.Value.logical_shape input_value),
    Tensor_shape.dimensions (Ir.Value.logical_shape weight_value),
    Tensor_shape.dimensions (Ir.Value.logical_shape output_value)
  with
  | ( [ batches; tokens; channels ],
      [ weight_channels; 1; kernel_width ],
      [ output_batches; output_tokens; output_channels ] )
    when batches = output_batches && tokens = output_tokens
         && channels = weight_channels && channels = output_channels ->
      let input = find state input_value in
      let weight = find state weight_value in
      let convolution = Ir.Short_conv_silu.convolution config in
      let stride = Ir.Short_conv.stride convolution in
      let padding = Ir.Short_conv.padding convolution in
      let dilation = Ir.Short_conv.dilation convolution in
      let output_start = Ir.Short_conv_silu.output_start config in
      for batch = 0 to batches - 1 do
        for token = 0 to tokens - 1 do
          for channel = 0 to channels - 1 do
            let accumulator = ref 0.0 in
            for kernel = 0 to kernel_width - 1 do
              let source_token =
                ((token + output_start) * stride) - padding
                + (kernel * dilation)
              in
              if source_token >= 0 && source_token < tokens then
                let input_index =
                  (((batch * tokens) + source_token) * channels) + channel
                in
                let weight_index = (channel * kernel_width) + kernel in
                accumulator :=
                  !accumulator
                  +. (Tensor.get_linear input input_index
                     *. Tensor.get_linear weight weight_index)
            done;
            let activated = pointwise_unary Ir.Pointwise.Silu !accumulator in
            let output_index =
              (((batch * tokens) + token) * channels) + channel
            in
            Tensor.set_linear output output_index activated
          done
        done
      done
  | _ -> failf "invalid token-major short-conv-SiLU shapes"

let attention state config query_value key_value value_value mask_value
    output_value output =
  match
    Tensor_shape.dimensions (Ir.Value.logical_shape query_value),
    Tensor_shape.dimensions (Ir.Value.logical_shape key_value),
    Tensor_shape.dimensions (Ir.Value.logical_shape value_value),
    Tensor_shape.dimensions (Ir.Value.logical_shape mask_value),
    Tensor_shape.dimensions (Ir.Value.logical_shape output_value)
  with
  | ( [ batches; heads; query_length; head_dimension ],
      [ key_batches; key_heads; key_length; key_dimension ],
      [ value_batches; value_heads; value_length; value_dimension ],
      [ mask_batches; mask_heads; mask_queries; mask_keys ],
      [ output_batches; output_heads; output_queries; output_dimension ] )
    when key_batches = batches && value_batches = batches
         && output_batches = batches && key_heads = heads
         && value_heads = heads && output_heads = heads
         && key_dimension = head_dimension && value_dimension = head_dimension
         && output_dimension = head_dimension && value_length = key_length
         && mask_queries = query_length && mask_keys = key_length
         && output_queries = query_length ->
      let query = find state query_value in
      let key = find state key_value in
      let value = find state value_value in
      let mask = find state mask_value in
      let scale = Ir.Attention.scale config in
      let allowed batch head query_position key_position =
        let mask_batch = if mask_batches = 1 then 0 else batch in
        let mask_head = if mask_heads = 1 then 0 else head in
        let mask_index =
          ((((mask_batch * mask_heads) + mask_head) * query_length)
          + query_position)
          * key_length
          + key_position
        in
        Tensor.get_linear mask mask_index <> 0.0
        && (not (Ir.Attention.causal config)
           || key_position <= query_position)
      in
      let score batch head query_position key_position =
        let accumulator = ref 0.0 in
        for dimension = 0 to head_dimension - 1 do
          let query_index =
            ((((batch * heads) + head) * query_length) + query_position)
            * head_dimension
            + dimension
          in
          let key_index =
            ((((batch * heads) + head) * key_length) + key_position)
            * head_dimension
            + dimension
          in
          accumulator :=
            !accumulator
            +. (Tensor.get_linear query query_index
               *. Tensor.get_linear key key_index)
        done;
        !accumulator *. scale
      in
      for batch = 0 to batches - 1 do
        for head = 0 to heads - 1 do
          for query_position = 0 to query_length - 1 do
            let maximum = ref Float.neg_infinity in
            let found = ref false in
            for key_position = 0 to key_length - 1 do
              if allowed batch head query_position key_position then (
                found := true;
                maximum :=
                  Float.max !maximum
                    (score batch head query_position key_position))
            done;
            let denominator = ref 0.0 in
            if !found then
              for key_position = 0 to key_length - 1 do
                if allowed batch head query_position key_position then
                  denominator :=
                    !denominator
                    +. exp
                         (score batch head query_position key_position
                         -. !maximum)
              done;
            for dimension = 0 to head_dimension - 1 do
              let result = ref 0.0 in
              if !denominator <> 0.0 then
                for key_position = 0 to key_length - 1 do
                  if allowed batch head query_position key_position then
                    let probability =
                      exp
                        (score batch head query_position key_position
                        -. !maximum)
                      /. !denominator
                    in
                    let value_index =
                      ((((batch * heads) + head) * key_length) + key_position)
                      * head_dimension
                      + dimension
                    in
                    result :=
                      !result
                      +. (probability *. Tensor.get_linear value value_index)
                done;
              let output_index =
                ((((batch * heads) + head) * query_length) + query_position)
                * head_dimension
                + dimension
              in
              Tensor.set_linear output output_index !result
            done
          done
        done
      done
  | _ -> failf "invalid CPU attention tensor metadata"

let embedding state indices_value weight_value output_value output =
  match
    Tensor_shape.dimensions (Ir.Value.logical_shape weight_value),
    List.rev (Tensor_shape.dimensions (Ir.Value.logical_shape output_value))
  with
  | [ vocabulary; width ], output_width :: _ when output_width = width ->
      let indices = find state indices_value in
      let weight = find state weight_value in
      let token_count = Tensor_shape.numel (Ir.Value.logical_shape indices_value) in
      for token = 0 to token_count - 1 do
        let raw_index = Tensor.get_linear indices token in
        if
          (not (Float.is_finite raw_index))
          || raw_index <> Float.round raw_index || raw_index < 0.0
          || raw_index >= Float.of_int vocabulary
        then failf "CPU embedding index %.9g is out of range" raw_index;
        let index = int_of_float raw_index in
        for dimension = 0 to width - 1 do
          Tensor.set_linear output ((token * width) + dimension)
            (Tensor.get_linear weight ((index * width) + dimension))
        done
      done
  | _ -> failf "invalid CPU embedding tensor metadata"

let arange config output_value output =
  let count = Tensor_shape.numel (Ir.Value.logical_shape output_value) in
  let start = Ir.Arange.start config |> Int64.of_int in
  let step = Ir.Arange.step config |> Int64.of_int in
  for index = 0 to count - 1 do
    let value = Int64.(add start (mul (of_int index) step)) in
    Tensor.set_linear output index (Int64.to_float value)
  done

let axis_factors shape axis =
  let dimensions = Tensor_shape.dimensions shape |> Array.of_list in
  let product from_index to_index =
    let result = ref 1 in
    for index = from_index to to_index do
      result := !result * dimensions.(index)
    done;
    !result
  in
  let outer = product 0 (axis - 1) in
  let width = dimensions.(axis) in
  let inner = product (axis + 1) (Array.length dimensions - 1) in
  outer, width, inner

let diff state config source_value prepend_value output_value output =
  let axis = Ir.Diff.axis config in
  let source_shape = Ir.Value.logical_shape source_value in
  let prepend_shape = Ir.Value.logical_shape prepend_value in
  let outer, source_width, inner = axis_factors source_shape axis in
  let prepend_outer, prepend_width, prepend_inner =
    axis_factors prepend_shape axis
  in
  let output_outer, output_width, output_inner =
    axis_factors (Ir.Value.logical_shape output_value) axis
  in
  if outer <> prepend_outer || outer <> output_outer || inner <> prepend_inner
     || inner <> output_inner || output_width <> source_width + prepend_width - 1
  then failf "invalid CPU diff tensor metadata";
  let source = find state source_value in
  let prepend = find state prepend_value in
  let combined outer_index axis_index inner_index =
    if axis_index < prepend_width then
      Tensor.get_linear prepend
        (((outer_index * prepend_width) + axis_index) * inner + inner_index)
    else
      Tensor.get_linear source
        (((outer_index * source_width) + axis_index - prepend_width) * inner
        + inner_index)
  in
  for outer_index = 0 to outer - 1 do
    for axis_index = 0 to output_width - 1 do
      for inner_index = 0 to inner - 1 do
        let output_index =
          (((outer_index * output_width) + axis_index) * inner) + inner_index
        in
        Tensor.set_linear output output_index
          (combined outer_index (axis_index + 1) inner_index
          -. combined outer_index axis_index inner_index)
      done
    done
  done

let cumsum state config input_value output_value output =
  let axis = Ir.Cumsum.axis config in
  let input_shape = Ir.Value.logical_shape input_value in
  let outer, width, inner = axis_factors input_shape axis in
  if
    not
      (Tensor_shape.equal input_shape (Ir.Value.logical_shape output_value))
  then failf "invalid CPU cumsum tensor metadata";
  let input = find state input_value in
  for outer_index = 0 to outer - 1 do
    for inner_index = 0 to inner - 1 do
      let accumulator = ref 0 in
      for axis_index = 0 to width - 1 do
        let index =
          (((outer_index * width) + axis_index) * inner) + inner_index
        in
        if Tensor.get_linear input index <> 0.0 then incr accumulator;
        Tensor.set_linear output index (Float.of_int !accumulator)
      done
    done
  done

let fill scalar output_value output =
  let count = Tensor_shape.numel (Ir.Value.logical_shape output_value) in
  let value = Ir.Scalar.to_float scalar in
  for index = 0 to count - 1 do
    Tensor.set_linear output index value
  done

let gather2 state source_value first_index_value second_index_value output_value
    output =
  let rows, cols =
    match Tensor_shape.dimensions (Ir.Value.logical_shape source_value) with
    | [ rows; cols ] -> rows, cols
    | _ -> failf "invalid CPU two-index gather source metadata"
  in
  let source = find state source_value in
  let first_index = find state first_index_value in
  let second_index = find state second_index_value in
  let output_shape = Ir.Value.logical_shape output_value in
  let output_dimensions = Tensor_shape.dimensions output_shape |> Array.of_list in
  let read_index tensor value bound name =
    let raw = Tensor.get_linear tensor value in
    if
      (not (Float.is_finite raw)) || raw <> Float.round raw || raw < 0.0
      || raw >= Float.of_int bound
    then failf "CPU two-index gather %s %.9g is out of range" name raw;
    int_of_float raw
  in
  for output_index = 0 to Tensor_shape.numel output_shape - 1 do
    let output_coordinates = coordinates output_dimensions output_index in
    let first_offset =
      broadcast_index ~source_shape:first_index_value ~output_dimensions
        output_coordinates
    in
    let second_offset =
      broadcast_index ~source_shape:second_index_value ~output_dimensions
        output_coordinates
    in
    let row = read_index first_index first_offset rows "row" in
    let col = read_index second_index second_offset cols "column" in
    Tensor.set_linear output output_index
      (Tensor.get_linear source ((row * cols) + col))
  done

let triangular_recurrence state config input_value output_value output =
  let input = find state input_value in
  let dimensions =
    Tensor_shape.dimensions (Ir.Value.logical_shape output_value)
  in
  let reversed = List.rev dimensions in
  let width = List.hd reversed in
  let rows = List.nth reversed 1 in
  if rows <> width then failf "CPU triangular recurrence state must be square";
  let outer =
    Tensor_shape.numel (Ir.Value.logical_shape output_value) / (width * width)
  in
  let round_f32 value = Int32.float_of_bits (Int32.bits_of_float value) in
  copy_into input output;
  for group = 0 to outer - 1 do
    let base = group * width * width in
    for row = Ir.Triangular_recurrence.start config
      to Ir.Triangular_recurrence.stop config - 1
    do
      let source =
        Array.init row (fun column ->
            Tensor.get_linear output (base + (row * width) + column))
      in
      for column = 0 to row - 1 do
        let accumulator = ref 0.0 in
        for inner = 0 to row - 1 do
          let product =
            round_f32
              (source.(inner)
              *. Tensor.get_linear output
                   (base + (inner * width) + column))
          in
          accumulator := round_f32 (!accumulator +. product)
        done;
        Tensor.set_linear output (base + (row * width) + column)
          (round_f32 (source.(column) +. !accumulator))
      done
    done
  done

let gated_delta state query_value key_value value_value gate_value beta_value
    output_value output =
  let query = find state query_value in
  let key = find state key_value in
  let value = find state value_value in
  let gate = find state gate_value in
  let beta = find state beta_value in
  let batch, heads, tokens, width, token_major =
    match
      Ir.Value.dtype query_value,
      Tensor_shape.dimensions (Ir.Value.logical_shape query_value)
    with
    | Ir.Dtype.Float32, [ batch; heads; tokens; width ] ->
        batch, heads, tokens, width, false
    | Ir.Dtype.Float16, [ batch; tokens; heads; width ] ->
        batch, heads, tokens, width, true
    | _ -> failf "CPU gated-delta query has an unsupported layout or dtype"
  in
  if
    Tensor_shape.dimensions (Ir.Value.logical_shape output_value)
    <> [ batch; tokens; heads; width ]
  then failf "CPU gated-delta output shape is inconsistent";
  let scale = 1.0 /. sqrt (Float.of_int width) in
  for batch_index = 0 to batch - 1 do
    for head = 0 to heads - 1 do
      let matrix = Array.make_matrix width width 0.0 in
      for token = 0 to tokens - 1 do
        let vector_base, scalar_index =
          if token_major then
            ( (((batch_index * tokens) + token) * heads + head) * width,
              ((batch_index * tokens) + token) * heads + head )
          else
            ( (((batch_index * heads) + head) * tokens + token) * width,
              ((batch_index * heads) + head) * tokens + token )
        in
        let decay = exp (Tensor.get_linear gate scalar_index) in
        let beta_value = Tensor.get_linear beta scalar_index in
        for row = 0 to width - 1 do
          let state_dot_key = ref 0.0 in
          for column = 0 to width - 1 do
            matrix.(row).(column) <- matrix.(row).(column) *. decay;
            state_dot_key :=
              !state_dot_key
              +. (matrix.(row).(column)
                 *. Tensor.get_linear key (vector_base + column))
          done;
          let delta =
            (Tensor.get_linear value (vector_base + row) -. !state_dot_key)
            *. beta_value
          in
          let state_dot_query = ref 0.0 in
          for column = 0 to width - 1 do
            matrix.(row).(column) <-
              matrix.(row).(column)
              +. (Tensor.get_linear key (vector_base + column) *. delta);
            state_dot_query :=
              !state_dot_query
              +. (matrix.(row).(column)
                 *. Tensor.get_linear query (vector_base + column))
          done;
          let output_index =
            (((batch_index * tokens) + token) * heads + head) * width + row
          in
          Tensor.set_linear output output_index (!state_dot_query *. scale)
        done
      done
    done
  done

let primitive state operation inputs output_value output =
  match operation, inputs with
  | Ir.Primitive.Pointwise operation, _ ->
      pointwise state operation output_value output
  | Ir.Primitive.Cast _, [ input ] -> copy_into (find state input) output
  | Ir.Primitive.Reduce reduction, [ input ] ->
      reduce state reduction input output_value output
  | Ir.Primitive.Movement (Ir.Movement.Concat { axis }), inputs ->
      concatenate state axis inputs output_value output
  | Ir.Primitive.Movement movement, [ input ] ->
      apply_movement state movement input output_value output
  | Ir.Primitive.Short_conv config, [ input; weight ] ->
      short_conv state config input weight output_value output
  | Ir.Primitive.Short_conv_silu config, [ input; weight ] ->
      short_conv_silu state config input weight output_value output
  | Ir.Primitive.Attention config, [ query; key; value; mask ] ->
      attention state config query key value mask output_value output
  | Ir.Primitive.Paged_attention_q8 _, _ ->
      failf "paged Q8 attention is a serving-runtime primitive"
  | Ir.Primitive.Embedding, [ indices; weight ] ->
      embedding state indices weight output_value output
  | Ir.Primitive.Arange config, [] -> arange config output_value output
  | Ir.Primitive.Diff config, [ source; prepend ] ->
      diff state config source prepend output_value output
  | Ir.Primitive.Cumsum config, [ input ] ->
      cumsum state config input output_value output
  | Ir.Primitive.Triangular_recurrence config, [ input ] ->
      triangular_recurrence state config input output_value output
  | Ir.Primitive.Gated_delta, [ query; key; value; gate; beta ] ->
      gated_delta state query key value gate beta output_value output
  | Ir.Primitive.L2_norm { epsilon }, [ input ] ->
      l2_norm state ~epsilon input output
  | Ir.Primitive.Fill scalar, [] -> fill scalar output_value output
  | Ir.Primitive.Gather2, [ source; first_index; second_index ] ->
      gather2 state source first_index second_index output_value output
  | Ir.Primitive.Update_slice index, [ destination; source ] ->
      update_slice state index destination source output
  | _ -> failf "invalid primitive input arity"

let run ~inputs thunk =
  let input_values = Hashtbl.create (List.length inputs) in
  List.iter (fun (name, tensor) -> Hashtbl.replace input_values name tensor) inputs;
  let state =
    {
      input_values;
      values = Hashtbl.create 32;
      execution = { outputs = Hashtbl.create 8 };
      next_value = 0;
    }
  in
  Effect.Deep.match_with thunk ()
    {
      retc = (fun value -> Ok (value, state.execution));
      exnc = (fun exception_value -> Error exception_value);
      effc =
        (fun (type a) (eff : a Effect.t) ->
          match eff with
          | Tile_effect.Input
              { name; source = _; shape; logical_shape; dtype } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let source =
                    match Hashtbl.find_opt state.input_values name with
                    | Some tensor -> tensor
                    | None -> failf "CPU input %s is not bound" name
                  in
                  let value = fresh_value state ~logical_shape ~shape ~dtype in
                  bind state value source;
                  Effect.Deep.continue continuation value)
          | Tile_effect.Alloc { shape; logical_shape; dtype; _ } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let value = fresh_value state ~logical_shape ~shape ~dtype in
                  let tensor = Tensor.create shape in
                  Tensor.fill tensor 0.0;
                  bind state value tensor;
                  Effect.Deep.continue continuation value)
          | Tile_effect.Copy { src; dst } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  copy_into (find state src) (find state dst);
                  Effect.Deep.continue continuation ())
          | Tile_effect.Async_copy { src; dst; _ } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  copy_into (find state src) (find state dst);
                  Effect.Deep.continue continuation ())
          | Tile_effect.Matmul { lhs; rhs; shape; logical_shape } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let value =
                    fresh_value state ~logical_shape ~shape
                      ~dtype:(Ir.Value.dtype lhs)
                  in
                  let tensor = Tensor.create shape in
                  matmul (find state lhs) (find state rhs) tensor;
                  bind state value tensor;
                  Effect.Deep.continue continuation value)
          | Tile_effect.Linear { input; weight; bias; shape; logical_shape } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let value =
                    fresh_value state ~logical_shape ~shape
                      ~dtype:(Ir.Value.dtype input)
                  in
                  let tensor = Tensor.create shape in
                  linear (find state input) (find state weight) tensor
                    (Option.map (find state) bias);
                  bind state value tensor;
                  Effect.Deep.continue continuation value)
          | Tile_effect.W4a16_linear
              { input; weight; scale; bias; shape; logical_shape } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let value =
                    fresh_value state ~logical_shape ~shape
                      ~dtype:(Ir.Value.dtype input)
                  in
                  let tensor = Tensor.create shape in
                  w4a16_linear (find state input) (find state weight) (find state scale)
                    tensor (Option.map (find state) bias);
                  bind state value tensor;
                  Effect.Deep.continue continuation value)
          | Tile_effect.Add { lhs; rhs; shape; logical_shape; broadcast } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let value =
                    fresh_value state ~logical_shape ~shape
                      ~dtype:(Ir.Value.dtype lhs)
                  in
                  let tensor = Tensor.create shape in
                  add ~left:(find state lhs) ~right:(find state rhs) ~output:tensor ~broadcast;
                  bind state value tensor;
                  Effect.Deep.continue continuation value)
          | Tile_effect.Gelu { input; shape; logical_shape } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let value =
                    fresh_value state ~logical_shape ~shape
                      ~dtype:(Ir.Value.dtype input)
                  in
                  let tensor = Tensor.create shape in
                  gelu (find state input) tensor;
                  bind state value tensor;
                  Effect.Deep.continue continuation value)
          | Tile_effect.Relu { input; shape; logical_shape } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let value =
                    fresh_value state ~logical_shape ~shape
                      ~dtype:(Ir.Value.dtype input)
                  in
                  let tensor = Tensor.create shape in
                  relu (find state input) tensor;
                  bind state value tensor;
                  Effect.Deep.continue continuation value)
          | Tile_effect.Primitive
              { operation; inputs; shape; logical_shape; dtype } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let value = fresh_value state ~logical_shape ~shape ~dtype in
                  let tensor = Tensor.create shape in
                  primitive state operation inputs value tensor;
                  bind state value tensor;
                  Effect.Deep.continue continuation value)
          | Tile_effect.Output { name; value } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  Hashtbl.replace state.execution.outputs name (find state value);
                  Effect.Deep.continue continuation ())
          | Tile_effect.Barrier_create _ ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  Effect.Deep.continue continuation (Tile_effect.Barrier.of_id 0))
          | Tile_effect.Barrier_arrive _ ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  Effect.Deep.continue continuation ())
          | Tile_effect.Barrier_wait _ ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  Effect.Deep.continue continuation ())
          | _ -> None);
    }
