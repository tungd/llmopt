module Dtype = struct
  type quant_type =
    | Q8_0
    | Q4_K
    | Q5_K
    | Q6_K
    | Q5_0
    | Q4_0
    | IQ4_XS

  type t =
    | Float32
    | Float16
    | Bfloat16
    | Int64
    | Int32
    | Int8
    | UInt8
    | Bool
    | Quant of quant_type

  let quant_to_string = function
    | Q8_0 -> "Q8_0"
    | Q4_K -> "Q4_K"
    | Q5_K -> "Q5_K"
    | Q6_K -> "Q6_K"
    | Q5_0 -> "Q5_0"
    | Q4_0 -> "Q4_0"
    | IQ4_XS -> "IQ4_XS"

  let quant_of_string = function
    | "Q8_0" | "q8_0" -> Some Q8_0
    | "Q4_K" | "q4_k" -> Some Q4_K
    | "Q5_K" | "q5_k" -> Some Q5_K
    | "Q6_K" | "q6_k" -> Some Q6_K
    | "Q5_0" | "q5_0" -> Some Q5_0
    | "Q4_0" | "q4_0" -> Some Q4_0
    | "IQ4_XS" | "iq4_xs" -> Some IQ4_XS
    | _ -> None

  let to_string = function
    | Float32 -> "f32"
    | Float16 -> "f16"
    | Bfloat16 -> "bf16"
    | Int64 -> "i64"
    | Int32 -> "i32"
    | Int8 -> "i8"
    | UInt8 -> "u8"
    | Bool -> "bool"
    | Quant q -> quant_to_string q
end

module Tensor_layout = struct
  type t = Dense of Dtype.t | Block_quantized of Dtype.quant_type

  let of_dtype = function
    | Dtype.Quant quant -> Block_quantized quant
    | dtype -> Dense dtype

  let block_elements = function
    | Dtype.Q8_0 | Dtype.Q5_0 | Dtype.Q4_0 -> 32
    | Dtype.Q4_K | Dtype.Q5_K | Dtype.Q6_K | Dtype.IQ4_XS -> 256

  let block_bytes = function
    | Dtype.Q8_0 -> 34
    | Dtype.Q4_K -> 144
    | Dtype.Q5_K -> 176
    | Dtype.Q6_K -> 210
    | Dtype.Q5_0 -> 22
    | Dtype.Q4_0 -> 18
    | Dtype.IQ4_XS -> 136

  let physical_bytes layout shape =
    let elements = Tensor_shape.numel shape in
    if elements <= 0 then Error "tensor layout requires at least one element"
    else
      match layout with
      | Dense dtype ->
          let byte_width =
            match dtype with
            | Dtype.Float32 | Dtype.Int32 -> Ok 4
            | Dtype.Float16 | Dtype.Bfloat16 -> Ok 2
            | Dtype.Int64 -> Ok 8
            | Dtype.Int8 | Dtype.UInt8 | Dtype.Bool -> Ok 1
            | Dtype.Quant _ -> Error "dense tensor layout cannot contain quantized dtype"
          in
          Result.bind byte_width (fun byte_width ->
              if elements > max_int / byte_width then
                Error "tensor layout byte length overflows"
              else Ok (elements * byte_width))
      | Block_quantized quant ->
          Tensor_shape.physical_bytes shape
            ~block_size:(block_elements quant) ~bytes_per_block:(block_bytes quant)
end

module Memory_space = struct
  type t = Global | Shared | Register | Private

  let to_string = function
    | Global -> "global"
    | Shared -> "shared"
    | Register -> "register"
    | Private -> "private"
end

module Layout = struct
  type t = Row_major | Col_major | Xor_swizzle of int

  let to_string = function
    | Row_major -> "row-major"
    | Col_major -> "col-major"
    | Xor_swizzle mask -> Printf.sprintf "xor-swizzle(0x%x)" mask
end

module Input_source = struct
  type t = Runtime | Tensor_store of { key : string }

  let to_string = function
    | Runtime -> "runtime"
    | Tensor_store { key } -> "tensor:" ^ key
end

module Value_id = struct
  type t = int
  let compare = Int.compare
  let to_int value = value
end

module Value = struct
  type t = {
    id : Value_id.t;
    shape : Shape.t;
    logical_shape : Tensor_shape.t;
    dtype : Dtype.t;
  }

  let make ~id ~shape ~dtype =
    { id; shape; logical_shape = Tensor_shape.of_matrix shape; dtype }

  let make_tensor ~id ~shape ~dtype =
    { id; shape = Tensor_shape.matrix_exn shape; logical_shape = shape; dtype }

  let id value = value.id
  let shape value = value.shape
  let logical_shape value = value.logical_shape
  let dtype value = value.dtype
  let layout value = Tensor_layout.of_dtype value.dtype
  let equal left right = left.id = right.id
end

module Linear_storage = struct
  type layout =
    | Dense of Dtype.t
    | Block_quantized of Dtype.quant_type
    | Groupwise_packed of { bits : int; group_elements : int }

  type t = {
    layout : layout;
    weight : Value.t;
    scale : Value.t option;
    bias : Value.t option;
  }

  let classify ~has_bias ~weight ~parameters =
    let split_bias parameters =
      match has_bias, List.rev parameters with
      | false, _ -> Ok (parameters, None)
      | true, bias :: reversed -> Ok (List.rev reversed, Some bias)
      | true, [] -> Error "biased Linear has no bias parameter"
    in
    Result.bind (split_bias parameters) (fun (storage_parameters, bias) ->
        match Value.dtype weight, storage_parameters with
        | Dtype.UInt8, [ scale ] when Value.dtype scale = Dtype.Float16 ->
            Ok
              { layout = Groupwise_packed { bits = 4; group_elements = 64 };
                weight;
                scale = Some scale;
                bias }
        | Dtype.Quant quant, [] ->
            Ok { layout = Block_quantized quant; weight; scale = None; bias }
        | dtype, [] -> (
            match Tensor_layout.of_dtype dtype with
            | Tensor_layout.Dense dtype ->
                Ok { layout = Dense dtype; weight; scale = None; bias }
            | Tensor_layout.Block_quantized _ -> assert false)
        | Dtype.UInt8, _ ->
            Error "packed groupwise Linear requires one float16 scale tensor"
        | Dtype.Quant _, _ ->
            Error "block-quantized Linear has unexpected storage parameters"
        | _ -> Error "dense Linear has unexpected storage parameters")

  let is_quantized storage =
    match storage.layout with Dense _ -> false | Block_quantized _ | Groupwise_packed _ -> true

  let has_separate_scale storage = Option.is_some storage.scale
end

module Scalar = struct
  type t = Bool of bool | Int of int | Float of float

  let to_float = function
    | Bool false -> 0.0
    | Bool true -> 1.0
    | Int value -> Float.of_int value
    | Float value -> value

  let to_string = function
    | Bool value -> string_of_bool value
    | Int value -> string_of_int value
    | Float value -> Printf.sprintf "%.17g" value
end

module Pointwise = struct
  type operand = Tensor of Value.t | Scalar of Scalar.t
  type unary =
    | Neg
    | Rsqrt
    | Silu
    | Cos
    | Sin
    | Tanh
    | Exp
    | Sigmoid
    | Softplus
    | Pow of Scalar.t
  type binary =
    | Add
    | Mul
    | Silu_mul
    | Gelu_mul
    | Sigmoid_mul
    | Sub
    | Div
    | Logical_and
    | Equal
    | Not_equal
    | Less_equal
    | Greater
  type t = Unary of unary * Value.t | Binary of binary * operand * operand

  let values = function
    | Unary (_, value) -> [ value ]
    | Binary (_, left, right) ->
        [ left; right ]
        |> List.filter_map (function Tensor value -> Some value | Scalar _ -> None)

  let unary_to_string = function
    | Neg -> "neg"
    | Rsqrt -> "rsqrt"
    | Silu -> "silu"
    | Cos -> "cos"
    | Sin -> "sin"
    | Tanh -> "tanh"
    | Exp -> "exp"
    | Sigmoid -> "sigmoid"
    | Softplus -> "softplus"
    | Pow exponent -> "pow(" ^ Scalar.to_string exponent ^ ")"

  let binary_to_string = function
    | Add -> "add"
    | Mul -> "mul"
    | Silu_mul -> "silu-mul"
    | Gelu_mul -> "gelu-mul"
    | Sigmoid_mul -> "sigmoid-mul"
    | Sub -> "sub"
    | Div -> "div"
    | Logical_and -> "and"
    | Equal -> "eq"
    | Not_equal -> "ne"
    | Less_equal -> "le"
    | Greater -> "gt"

  let to_string = function
    | Unary (operator, _) -> unary_to_string operator
    | Binary (operator, _, _) -> binary_to_string operator
end

module Gated_activation = struct
  type t = Silu | Gelu | Sigmoid

  let to_string = function
    | Silu -> "silu"
    | Gelu -> "gelu"
    | Sigmoid -> "sigmoid"
end

module Reduction = struct
  type operator = Mean | Sum
  type t = { operator : operator; axes : int list; keepdim : bool }

  let to_string reduction =
    let axes = reduction.axes |> List.map string_of_int |> String.concat "," in
    let operator = match reduction.operator with Mean -> "mean" | Sum -> "sum" in
    Printf.sprintf "%s(axes=[%s],keepdim=%b)" operator axes reduction.keepdim
end

module Movement = struct
  type t =
    | View
    | Reshape
    | Transpose of { axis0 : int; axis1 : int }
    | Unsqueeze of int
    | Expand
    | Contiguous
    | Index of Tensor_shape.Index.t
    | Concat of { axis : int }
    | Roll of { axis : int; shift : int }

  let to_string = function
    | View -> "view"
    | Reshape -> "reshape"
    | Transpose { axis0; axis1 } ->
        Printf.sprintf "transpose(%d,%d)" axis0 axis1
    | Unsqueeze axis -> Printf.sprintf "unsqueeze(%d)" axis
    | Expand -> "expand"
    | Contiguous -> "contiguous"
    | Index index -> Tensor_shape.Index.to_string index
    | Concat { axis } -> Printf.sprintf "concat(axis=%d)" axis
    | Roll { axis; shift } -> Printf.sprintf "roll(axis=%d,shift=%d)" axis shift
end

module Short_conv = struct
  type t = {
    stride : int;
    padding : int;
    dilation : int;
    groups : int;
  }

  let create ~stride ~padding ~dilation ~groups =
    if stride <= 0 then Error "short-conv stride must be positive"
    else if padding < 0 then Error "short-conv padding must be non-negative"
    else if dilation <= 0 then Error "short-conv dilation must be positive"
    else if groups <= 0 then Error "short-conv groups must be positive"
    else Ok { stride; padding; dilation; groups }

  let stride config = config.stride
  let padding config = config.padding
  let dilation config = config.dilation
  let groups config = config.groups

  let to_string config =
    Printf.sprintf "short-conv(stride=%d,padding=%d,dilation=%d,groups=%d)"
      config.stride config.padding config.dilation config.groups
end

module Short_conv_silu = struct
  type t = { convolution : Short_conv.t; output_start : int }

  let create ~convolution ~output_start =
    if output_start < 0 then
      Error "short-conv-SiLU output start must be non-negative"
    else Ok { convolution; output_start }

  let convolution config = config.convolution
  let output_start config = config.output_start

  let to_string config =
    Printf.sprintf "short-conv-silu-tm(%s,output-start=%d)"
      (Short_conv.to_string config.convolution) config.output_start
end

module L2_norm_slice = struct
  type t = { epsilon : float; offset : int }

  let create ~epsilon ~offset =
    if not (Float.is_finite epsilon) || epsilon < 0.0 then
      Error "L2-normalization slice epsilon must be finite and non-negative"
    else if offset < 0 then
      Error "L2-normalization slice offset must be non-negative"
    else Ok { epsilon; offset }

  let epsilon config = config.epsilon
  let offset config = config.offset

  let to_string config =
    Printf.sprintf "l2-norm-slice(eps=%.9g,offset=%d)" config.epsilon
      config.offset
end

module Attention = struct
  type t = { scale : float; causal : bool }

  let create ~scale ~causal =
    if Float.is_finite scale then Ok { scale; causal }
    else Error "attention scale must be finite"

  let scale config = config.scale
  let causal config = config.causal

  let to_string config =
    Printf.sprintf "attention(scale=%.9g,causal=%b)" config.scale config.causal
end

module Rms_rope = struct
  type t = { epsilon : float; half_dimension : int }

  let create ~epsilon ~half_dimension =
    if not (Float.is_finite epsilon) then
      Error "RMSNorm-RoPE epsilon must be finite"
    else if half_dimension <= 0 then
      Error "RMSNorm-RoPE half dimension must be positive"
    else Ok { epsilon; half_dimension }

  let epsilon config = config.epsilon
  let half_dimension config = config.half_dimension

  let to_string config =
    Printf.sprintf "rms-rope(eps=%.9g,half=%d)" config.epsilon
      config.half_dimension
end

module Short_conv_step = struct
  type t = { channels : int; window : int }

  let create ~channels ~window =
    if channels <= 0 then Error "short-conv-step channels must be positive"
    else if window <= 0 then Error "short-conv-step window must be positive"
    else Ok { channels; window }

  let channels config = config.channels
  let window config = config.window

  let to_string config =
    Printf.sprintf "short-conv-step(channels=%d,window=%d)" config.channels
      config.window
end

module Short_conv_prefill = struct
  type t = { channels : int; window : int }

  let create ~channels ~window =
    if channels <= 0 then Error "short-conv-prefill channels must be positive"
    else if window <= 0 then Error "short-conv-prefill window must be positive"
    else Ok { channels; window }

  let channels config = config.channels
  let window config = config.window

  let to_string config =
    Printf.sprintf "short-conv-prefill(channels=%d,window=%d)" config.channels
      config.window
end

module Paged_attention_q8 = struct
  type t = {
    scale : float;
    cache_layer : int;
    attention_layers : int;
    kv_heads : int;
    group_size : int;
    token_stride : int;
  }

  let create ~scale ~cache_layer ~attention_layers ~kv_heads ~group_size
      ~token_stride =
    if not (Float.is_finite scale) then Error "paged Q8 attention scale must be finite"
    else if attention_layers <= 0 then
      Error "paged Q8 attention requires at least one attention layer"
    else if cache_layer < 0 || cache_layer >= attention_layers then
      Error "paged Q8 attention cache layer is outside the attention layout"
    else if kv_heads <= 0 then Error "paged Q8 attention requires positive KV heads"
    else if group_size <> 64 then
      Error "paged Q8 attention group size must be 64"
    else if
      token_stride <> 2 * attention_layers * kv_heads * (64 + 2)
    then
      Error
        "paged Q8 attention token stride disagrees with the fixed head-64 layout"
    else
      Ok
        {
          scale;
          cache_layer;
          attention_layers;
          kv_heads;
          group_size;
          token_stride;
        }

  let scale config = config.scale
  let cache_layer config = config.cache_layer
  let attention_layers config = config.attention_layers
  let kv_heads config = config.kv_heads
  let group_size config = config.group_size
  let token_stride config = config.token_stride

  let to_string config =
    Printf.sprintf
      "paged-attention-q8(scale=%.9g,layer=%d/%d,kv-heads=%d,group=%d,stride=%d)"
      config.scale config.cache_layer config.attention_layers config.kv_heads
      config.group_size config.token_stride
end

module Arange = struct
  type t = { start : int; stop : int; step : int }

  let create ~start ~stop ~step =
    if step = 0 then Error "arange step must be non-zero"
    else Ok { start; stop; step }

  let start config = config.start
  let stop config = config.stop
  let step config = config.step

  let to_string config =
    Printf.sprintf "arange(start=%d,stop=%d,step=%d)" config.start config.stop
      config.step
end

module Diff = struct
  type t = { axis : int }

  let create ~axis =
    if axis < 0 then Error "diff axis must be normalized"
    else Ok { axis }

  let axis config = config.axis
  let to_string config = Printf.sprintf "diff(axis=%d,prepend=true)" config.axis
end

module Cumsum = struct
  type t = { axis : int }

  let create ~axis =
    if axis < 0 then Error "cumsum axis must be normalized"
    else Ok { axis }

  let axis config = config.axis
  let to_string config = Printf.sprintf "cumsum(axis=%d)" config.axis
end

module Triangular_recurrence = struct
  type t = { axis : int; start : int; stop : int }

  let create ~axis ~start ~stop =
    if axis < 0 then Error "triangular recurrence axis must be normalized"
    else if start < 1 then Error "triangular recurrence start must be positive"
    else if stop <= start then
      Error "triangular recurrence stop must be greater than start"
    else Ok { axis; start; stop }

  let axis config = config.axis
  let start config = config.start
  let stop config = config.stop

  let to_string config =
    Printf.sprintf "triangular-recurrence(axis=%d,start=%d,stop=%d)"
      config.axis config.start config.stop
end

module Primitive = struct
  type t =
    | Pointwise of Pointwise.t
    | Cast of Dtype.t
    | Reduce of Reduction.t
    | Movement of Movement.t
    | Short_conv of Short_conv.t
    | Short_conv_silu of Short_conv_silu.t
    | Attention of Attention.t
    | Paged_attention_q8 of Paged_attention_q8.t
    | Embedding
    | Arange of Arange.t
    | Diff of Diff.t
    | Cumsum of Cumsum.t
    | Triangular_recurrence of Triangular_recurrence.t
    | Gated_delta
    | L2_norm of { epsilon : float }
    | L2_norm_slice of L2_norm_slice.t
    | Fill of Scalar.t
    | Gather2
    | Update_slice of Tensor_shape.Index.t
    | Pad_right_zero of { axis : int }
    | Triangular of { upper : bool; diagonal : int }
    | Masked_fill of Scalar.t
    | Eye
    | Batched_matmul

  let values = function
    | Pointwise operation -> Pointwise.values operation
    | Cast _ | Reduce _ | Movement _ | Short_conv _ | Short_conv_silu _
    | Attention _
    | Paged_attention_q8 _ | Embedding | Arange _ | Diff _ | Cumsum _
    | Triangular_recurrence _ | Gated_delta | L2_norm _ | L2_norm_slice _
    | Fill _ | Gather2 | Update_slice _
    | Pad_right_zero _ | Triangular _
    | Masked_fill _ | Eye | Batched_matmul -> []

  let to_string = function
    | Pointwise operation -> Pointwise.to_string operation
    | Cast dtype -> "cast(" ^ Dtype.to_string dtype ^ ")"
    | Reduce reduction -> Reduction.to_string reduction
    | Movement movement -> Movement.to_string movement
    | Short_conv config -> Short_conv.to_string config
    | Short_conv_silu config -> Short_conv_silu.to_string config
    | Attention config -> Attention.to_string config
    | Paged_attention_q8 config -> Paged_attention_q8.to_string config
    | Embedding -> "embedding"
    | Arange config -> Arange.to_string config
    | Diff config -> Diff.to_string config
    | Cumsum config -> Cumsum.to_string config
    | Triangular_recurrence config -> Triangular_recurrence.to_string config
    | Gated_delta -> "gated-delta"
    | L2_norm { epsilon } -> Printf.sprintf "l2-norm(eps=%.9g)" epsilon
    | L2_norm_slice config -> L2_norm_slice.to_string config
    | Fill scalar -> "fill(" ^ Scalar.to_string scalar ^ ")"
    | Gather2 -> "gather2"
    | Update_slice index -> "update-" ^ Tensor_shape.Index.to_string index
    | Pad_right_zero { axis } -> Printf.sprintf "pad-right-zero(axis=%d)" axis
    | Triangular { upper; diagonal } ->
        Printf.sprintf "%s(diagonal=%d)" (if upper then "triu" else "tril")
          diagonal
    | Masked_fill scalar -> "masked-fill(" ^ Scalar.to_string scalar ^ ")"
    | Eye -> "eye"
    | Batched_matmul -> "batched-matmul"
end

module Argument = struct
  type t =
    | Value of Value.t
    | Null
    | Ellipsis
    | Bool of bool
    | Int of int
    | Float of float
    | String of string
    | Symbol of string
    | List of t list
    | Tuple of t list
    | Mapping of (string * t) list
    | Slice of { start : t; stop : t; step : t }

  let rec values = function
    | Value value -> [ value ]
    | Null | Ellipsis | Bool _ | Int _ | Float _ | String _ | Symbol _ -> []
    | List arguments | Tuple arguments -> List.concat_map values arguments
    | Mapping fields -> fields |> List.map snd |> List.concat_map values
    | Slice { start; stop; step } -> values start @ values stop @ values step
end

module Op = struct
  type t =
    | Input of { name : string; source : Input_source.t }
    | Alloc of { space : Memory_space.t; layout : Layout.t }
    | Copy of { asynchronous : bool; barrier : int option }
    | Matmul of { m : int; n : int; k : int }
    | Linear of { m : int; n : int; k : int; bias : bool }
    | Linear_add of { m : int; n : int; k : int }
    | Add of { broadcast : Shape.broadcast }
    | Gelu
    | Relu
    | Rms_norm of { epsilon : float }
    | Rms_norm_add of { epsilon : float }
    | Gated_linear of {
        m : int;
        n : int;
        k : int;
        activation : Gated_activation.t;
      }
    | Rms_rope of Rms_rope.t
    | Short_conv_step of Short_conv_step.t
    | Short_conv_step_fused of Short_conv_step.t
    | Short_conv_prefill of Short_conv_prefill.t
    | Primitive of Primitive.t
    | Opaque of {
        op : string;
        target : string;
        arguments : Argument.t list;
        keyword_arguments : (string * Argument.t) list;
      }
    | Output of { name : string }
    | Barrier_create of { id : int; name : string }
    | Barrier_arrive of int
    | Barrier_wait of int
    | Fused_matmul_bias of { m : int; n : int; k : int }
    | W4a16_linear of { m : int; n : int; k : int; bias : bool }
    | W4a16_qkv_linear of {
        m : int;
        k : int;
        n_q : int;
        n_k : int;
        n_v : int;
        extra_outputs : Value.t list;
      }
    | W4a16_swiglu_ffn of { m : int; n : int; k : int; epsilon : float }
    | Rms_rope_qk of {
        q_heads : int;
        k_heads : int;
        width : int;
        half_dimension : int;
        epsilon : float;
        extra_outputs : Value.t list;
      }
    | Rotary_qk of {
        q_heads : int;
        k_heads : int;
        width : int;
        half_dimension : int;
        extra_outputs : Value.t list;
      }
    | W4a16_lm_head_argmax of {
        m : int;
        n : int;
        k : int;
        epsilon : float;
        extra_outputs : Value.t list;
      }
  let additional_outputs = function
    | W4a16_lm_head_argmax { extra_outputs; _ } -> extra_outputs
    | W4a16_qkv_linear { extra_outputs; _ } -> extra_outputs
    | Rms_rope_qk { extra_outputs; _ } -> extra_outputs
    | Rotary_qk { extra_outputs; _ } -> extra_outputs
    | _ -> []

  let to_string = function
    | Input { name; source } ->
        Printf.sprintf "input(%s,%s)" name (Input_source.to_string source)
    | Alloc { space; layout } ->
        Printf.sprintf "alloc(%s,%s)" (Memory_space.to_string space)
          (Layout.to_string layout)
    | Copy { asynchronous; barrier } ->
        let kind = if asynchronous then "async-copy" else "copy" in
        let barrier =
          match barrier with
          | None -> ""
          | Some id -> Printf.sprintf ",barrier=%d" id
        in
        kind ^ barrier
    | Matmul { m; n; k } -> Printf.sprintf "matmul[%dx%dx%d]" m n k
    | Linear { m; n; k; bias = false } -> Printf.sprintf "linear[%dx%dx%d]" m n k
    | Linear { m; n; k; bias = true } -> Printf.sprintf "linear+bias[%dx%dx%d]" m n k
    | Linear_add { m; n; k } ->
        Printf.sprintf "linear-add[%dx%dx%d]" m n k
    | Add { broadcast = Shape.Same } -> "add[same]"
    | Add { broadcast = Shape.Row } -> "add[row-broadcast]"
    | Gelu -> "gelu"
    | Relu -> "relu"
    | Rms_norm { epsilon } -> Printf.sprintf "rms-norm(eps=%.9g)" epsilon
    | Rms_norm_add { epsilon } ->
        Printf.sprintf "rms-norm-add(eps=%.9g)" epsilon
    | Gated_linear { m; n; k; activation } ->
        Printf.sprintf "gated-linear-%s[%dx%dx%d]"
          (Gated_activation.to_string activation) m n k
    | Rms_rope config -> Rms_rope.to_string config
    | Rms_rope_qk { q_heads; k_heads; width; half_dimension; epsilon; _ } ->
        Printf.sprintf "rms-rope-qk(q=%d,k=%d,w=%d,half=%d,eps=%.9g)" q_heads k_heads width half_dimension epsilon
    | Rotary_qk { q_heads; k_heads; width; half_dimension; _ } ->
        Printf.sprintf "rotary-qk(q=%d,k=%d,w=%d,half=%d)" q_heads k_heads
          width half_dimension
    | Short_conv_step config -> Short_conv_step.to_string config
    | Short_conv_step_fused config ->
        "short-conv-step-fused(" ^ Short_conv_step.to_string config ^ ")"
    | Short_conv_prefill config -> Short_conv_prefill.to_string config
    | Primitive primitive -> Primitive.to_string primitive
    | Opaque { op; target; _ } -> Printf.sprintf "opaque(%s,%s)" op target
    | Output { name } -> Printf.sprintf "output(%s)" name
    | Barrier_create { id; name } -> Printf.sprintf "barrier-create(%d,%s)" id name
    | Barrier_arrive id -> Printf.sprintf "barrier-arrive(%d)" id
    | Barrier_wait id -> Printf.sprintf "barrier-wait(%d)" id
    | Fused_matmul_bias { m; n; k } ->
        Printf.sprintf "fused-matmul-bias[%dx%dx%d]" m n k
    | W4a16_linear { m; n; k; bias = false } ->
        Printf.sprintf "w4a16-linear-g64[%dx%dx%d]" m n k
    | W4a16_linear { m; n; k; bias = true } ->
        Printf.sprintf "w4a16-linear-g64+bias[%dx%dx%d]" m n k
    | W4a16_qkv_linear { m; k; n_q; n_k; n_v; _ } ->
        Printf.sprintf "w4a16-qkv-linear-g64[%dx%dx(%d+%d+%d)]" m k n_q n_k n_v
    | W4a16_swiglu_ffn { m; n; k; epsilon } ->
        Printf.sprintf "w4a16-swiglu-ffn-g64[%dx%dx%d,eps=%.9g]" m n k
          epsilon
    | W4a16_lm_head_argmax { m; n; k; epsilon; extra_outputs } ->
        Printf.sprintf "w4a16-lm-head-argmax%s[%dx%dx%d,eps=%.9g]"
          (if extra_outputs = [] then "" else "+logits-output") m n k
          epsilon
end

type node = {
  id : int;
  op : Op.t;
  inputs : Value.t list;
  output : Value.t option;
}

let node_id node = node.id
let node_op node = node.op
let node_inputs node = node.inputs
let node_output node = node.output
let node_replace node ~op ~inputs = { node with op; inputs }
let node_reindex node id = { node with id }
let node_create ~id ~op ~inputs ~output = { id; op; inputs; output }

module Graph = struct
  type t = {
    mutable next_value : int;
    mutable next_node : int;
    mutable nodes_rev : node list;
    mutable outputs_rev : (string * Value.t) list;
  }

  let create () =
    { next_value = 0; next_node = 0; nodes_rev = []; outputs_rev = [] }

  let fresh_value graph ~shape ~dtype =
    let value = Value.make ~id:graph.next_value ~shape ~dtype in
    graph.next_value <- graph.next_value + 1;
    value

  let fresh_tensor_value graph ~shape ~dtype =
    let value = Value.make_tensor ~id:graph.next_value ~shape ~dtype in
    graph.next_value <- graph.next_value + 1;
    value

  let append graph ~op ~inputs ~output =
    let node = { id = graph.next_node; op; inputs; output } in
    graph.next_node <- graph.next_node + 1;
    graph.nodes_rev <- node :: graph.nodes_rev

  let input graph ~name ~source ~shape ~dtype =
    let value = fresh_value graph ~shape ~dtype in
    append graph ~op:(Op.Input { name; source }) ~inputs:[] ~output:(Some value);
    value

  let tensor_input graph ~name ~source ~shape ~dtype =
    let value = fresh_tensor_value graph ~shape ~dtype in
    append graph ~op:(Op.Input { name; source }) ~inputs:[] ~output:(Some value);
    value

  let allocate graph ~shape ~dtype ~space ~layout =
    let value = fresh_value graph ~shape ~dtype in
    append graph ~op:(Op.Alloc { space; layout }) ~inputs:[] ~output:(Some value);
    value

  let add_output graph ~name value =
    append graph ~op:(Op.Output { name }) ~inputs:[ value ] ~output:None;
    graph.outputs_rev <- (name, value) :: graph.outputs_rev

  let nodes graph = List.rev graph.nodes_rev
  let outputs graph = List.rev graph.outputs_rev

  let with_nodes graph nodes = { graph with nodes_rev = List.rev nodes }

  let with_nodes_and_outputs graph nodes outputs =
    { graph with nodes_rev = List.rev nodes; outputs_rev = List.rev outputs }

  let pp formatter graph =
    let pp_value formatter value =
      Format.fprintf formatter "%%%d:%s:%s"
        (Value_id.to_int (Value.id value))
        (Tensor_shape.to_string (Value.logical_shape value))
        (Dtype.to_string (Value.dtype value))
    in
    List.iter
      (fun node ->
        Format.fprintf formatter "n%d %s (" node.id (Op.to_string node.op);
        List.iteri
          (fun index value ->
            if index > 0 then Format.fprintf formatter ", ";
            pp_value formatter value)
          node.inputs;
        Format.fprintf formatter ")";
        (match node.output with
        | None -> ()
        | Some value -> Format.fprintf formatter " -> %a" pp_value value);
        Format.fprintf formatter "@.")
      (nodes graph)
end

module Debug = struct
  open Sexplib0.Sexp_conv

  type value = {
    value_id : int;
    logical_shape : int list;
    dtype : string;
  }
  [@@deriving sexp]

  type node = {
    node_id : int;
    operation : string;
    inputs : int list;
    embedded_inputs : int list;
    outputs : int list;
  }
  [@@deriving sexp]

  type named_output = { name : string; value_id : int } [@@deriving sexp]

  type snapshot = {
    values : value list;
    nodes : node list;
    named_outputs : named_output list;
  }
  [@@deriving sexp]

  let value_id value = Value.id value |> Value_id.to_int

  let embedded_inputs op =
    match op with
    | Op.Opaque { arguments; keyword_arguments; _ } ->
        arguments @ List.map snd keyword_arguments
        |> List.concat_map Argument.values
    | Op.Primitive (Primitive.Pointwise operation) ->
        Pointwise.values operation
    | _ -> []

  let value value =
    { value_id = value_id value;
      logical_shape = Tensor_shape.dimensions (Value.logical_shape value);
      dtype = Dtype.to_string (Value.dtype value) }

  let node node =
    let op = node_op node in
    { node_id = node_id node;
      operation = Op.to_string op;
      inputs = List.map value_id (node_inputs node);
      embedded_inputs = List.map value_id (embedded_inputs op);
      outputs =
        List.map value_id
          (Option.to_list (node_output node) @ Op.additional_outputs op) }

  let snapshot graph =
    let graph_nodes = Graph.nodes graph in
    let all_values =
      graph_nodes
      |> List.concat_map (fun node ->
             let op = node_op node in
             node_inputs node @ embedded_inputs op
             @ Option.to_list (node_output node)
             @ Op.additional_outputs op)
      |> List.sort_uniq (fun left right ->
             Value_id.compare (Value.id left) (Value.id right))
      |> List.map value
    in
    { values = all_values;
      nodes = List.map node graph_nodes;
      named_outputs =
        Graph.outputs graph
        |> List.map (fun (name, value) -> { name; value_id = value_id value }) }

  let to_sexp_string graph =
    graph |> snapshot |> sexp_of_snapshot |> Sexplib0.Sexp.to_string_hum
end
