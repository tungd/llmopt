module Space = struct
  type global
  type shared
  type register

  type _ t =
    | Global : global t
    | Shared : shared t
    | Register : register t

  let global = Global
  let shared = Shared
  let register = Register

  let to_ir : type space. space t -> Ir.Memory_space.t = function
    | Global -> Ir.Memory_space.Global
    | Shared -> Ir.Memory_space.Shared
    | Register -> Ir.Memory_space.Register
end

module Layout = struct
  type row_major
  type col_major
  type swizzled

  type _ t =
    | Row_major : row_major t
    | Col_major : col_major t
    | Xor_swizzled : int -> swizzled t

  let row_major = Row_major
  let col_major = Col_major
  let xor_swizzled mask = Xor_swizzled mask

  let to_ir : type layout. layout t -> Ir.Layout.t = function
    | Row_major -> Ir.Layout.Row_major
    | Col_major -> Ir.Layout.Col_major
    | Xor_swizzled mask -> Ir.Layout.Xor_swizzle mask
end

type ('space, 'layout) t = {
  value : Ir.Value.t;
  shape : Shape.t;
  dtype : Ir.Dtype.t;
  space : 'space Space.t;
  layout : 'layout Layout.t;
}

type any = Tile : ('space, 'layout) t -> any

let shape tile = tile.shape
let dtype tile = tile.dtype

let input ?(dtype = Ir.Dtype.Float32) ~name ~shape () =
  let value =
    Tile_effect.input ~name ~source:Ir.Input_source.Runtime ~shape ~dtype
  in
  { value; shape; dtype; space = Space.global; layout = Layout.row_major }

let alloc ?(dtype = Ir.Dtype.Float32) ~shape ~space ~layout () =
  let value =
    Tile_effect.alloc
      {
        shape;
        logical_shape = Tensor_shape.of_matrix shape;
        dtype;
        space = Space.to_ir space;
        layout = Layout.to_ir layout;
      }
  in
  { value; shape; dtype; space; layout }

let copy left right =
  if not (Shape.equal left.shape right.shape) then
    invalid_arg (Shape.error_to_string (Shape.Not_broadcastable (left.shape, right.shape)));
  Tile_effect.copy ~src:left.value ~dst:right.value

let async_copy source destination ~barrier =
  if not (Shape.equal source.shape destination.shape) then
    invalid_arg (Shape.error_to_string (Shape.Not_broadcastable (source.shape, destination.shape)));
  Tile_effect.async_copy ~src:source.value ~dst:destination.value ~barrier

let shape_or_invalid context = function
  | Ok shape -> shape
  | Error error -> invalid_arg (context ^ ": " ^ Shape.error_to_string error)

let matmul left right =
  let result_shape = shape_or_invalid "matmul" (Shape.matmul left.shape right.shape) in
  let value =
    Tile_effect.matmul
      {
        lhs = left.value;
        rhs = right.value;
        shape = result_shape;
        logical_shape = Tensor_shape.of_matrix result_shape;
      }
  in
  {
    value;
    shape = result_shape;
    dtype = left.dtype;
    space = Space.register;
    layout = Layout.row_major;
  }

let w4a16_linear input weight scale ?bias =
  let k = Shape.cols input.shape in
  let n = Shape.rows weight.shape in
  if input.dtype <> Ir.Dtype.Float16 then
    invalid_arg "w4a16_linear: activation must be float16"
  else if weight.dtype <> Ir.Dtype.UInt8 then
    invalid_arg "w4a16_linear: packed weight must have uint8 storage"
  else if scale.dtype <> Ir.Dtype.Float16 then
    invalid_arg "w4a16_linear: scale must be float16"
  else if k mod 64 <> 0 then
    invalid_arg "w4a16_linear: input width must be divisible by group size 64"
  else if Shape.cols weight.shape * 2 <> k then
    invalid_arg
      "w4a16_linear: packed weight shape must be [N,K/2]"
  else
    let output_shape =
      Shape.of_ints_exn ~rows:(Shape.rows input.shape) ~cols:n
    in
    let expected_scale =
      Shape.of_ints_exn ~rows:n ~cols:(k / 64)
    in
    if not (Shape.equal scale.shape expected_scale) then
      invalid_arg
        (Printf.sprintf "w4a16_linear: expected scale shape %s, got %s"
           (Shape.to_string expected_scale) (Shape.to_string scale.shape))
    else
      (match bias with
      | Some bias
        when not
               (Shape.equal bias.shape
                  (Shape.of_ints_exn ~rows:1 ~cols:n)) ->
          invalid_arg
            (Printf.sprintf "w4a16_linear: expected bias shape [1x%d], got %s" n
               (Shape.to_string bias.shape))
      | _ ->
          let value =
            Tile_effect.w4a16_linear
              {
                input = input.value;
                weight = weight.value;
                scale = scale.value;
                bias = Option.map (fun tile -> tile.value) bias;
                shape = output_shape;
                logical_shape = Tensor_shape.of_matrix output_shape;
              }
          in
          {
            value;
            shape = output_shape;
            dtype = input.dtype;
            space = Space.register;
            layout = Layout.row_major;
          })

let q8_linear input weight scale ?bias =
  if weight.dtype <> Ir.Dtype.Int8 then
    invalid_arg "q8_linear: weight must have int8 storage"
  else if scale.dtype <> Ir.Dtype.Float16 && scale.dtype <> Ir.Dtype.Float32 then
    invalid_arg "q8_linear: scale must be float16 or float32"
  else if Shape.cols input.shape <> Shape.cols weight.shape then
    invalid_arg
      (Shape.error_to_string
         (Shape.Not_broadcastable (input.shape, weight.shape)))
  else
    let output_shape =
      Shape.of_ints_exn ~rows:(Shape.rows input.shape) ~cols:(Shape.rows weight.shape)
    in
    let expected_scale =
      Shape.of_ints_exn ~rows:1 ~cols:(Shape.rows weight.shape)
    in
    if not (Shape.equal scale.shape expected_scale) then
      invalid_arg
        (Printf.sprintf "q8_linear: expected scale shape %s, got %s"
           (Shape.to_string expected_scale) (Shape.to_string scale.shape))
    else
      (match bias with
      | Some bias when not (Shape.equal bias.shape expected_scale) ->
          invalid_arg
            (Printf.sprintf "q8_linear: expected bias shape %s, got %s"
               (Shape.to_string expected_scale) (Shape.to_string bias.shape))
      | _ ->
          let value =
            Tile_effect.q8_linear
              {
                input = input.value;
                weight = weight.value;
                scale = scale.value;
                bias = Option.map (fun tile -> tile.value) bias;
                shape = output_shape;
                logical_shape = Tensor_shape.of_matrix output_shape;
              }
          in
          {
            value;
            shape = output_shape;
            dtype = input.dtype;
            space = Space.register;
            layout = Layout.row_major;
          })

let add left right =
  let result_shape, broadcast =
    match Shape.add left.shape right.shape with
    | Ok result -> result
    | Error error -> invalid_arg ("add: " ^ Shape.error_to_string error)
  in
  let value =
    Tile_effect.add
      {
        lhs = left.value;
        rhs = right.value;
        shape = result_shape;
        logical_shape = Tensor_shape.of_matrix result_shape;
        broadcast;
      }
  in
  {
    value;
    shape = result_shape;
    dtype = left.dtype;
    space = Space.register;
    layout = Layout.row_major;
  }

let gelu tile =
  let value =
    Tile_effect.gelu
      {
        input = tile.value;
        shape = tile.shape;
        logical_shape = Tensor_shape.of_matrix tile.shape;
      }
  in
  {
    value;
    shape = tile.shape;
    dtype = tile.dtype;
    space = Space.register;
    layout = Layout.row_major;
  }

let output ~name tile = Tile_effect.output ~name ~value:tile.value
let create_barrier name = Tile_effect.create_barrier name
let barrier_arrive barrier = Tile_effect.barrier_arrive barrier
let barrier_wait barrier = Tile_effect.barrier_wait barrier
