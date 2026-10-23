module Error = struct
  type t =
    | Unsupported_storage of Ir.Linear_storage.layout
    | Invalid_groupwise_bits of int
    | Invalid_groupwise_group_elements of int

  let to_string = function
    | Unsupported_storage (Ir.Linear_storage.Dense dtype) ->
        Printf.sprintf "block schema does not support dense storage (%s)"
          (Ir.Dtype.to_string dtype)
    | Unsupported_storage (Ir.Linear_storage.Block_quantized quant) ->
        Printf.sprintf "block schema does not support block dtype %s"
          (Ir.Dtype.quant_to_string quant)
    | Unsupported_storage (Ir.Linear_storage.Groupwise_packed { bits; group_elements }) ->
        Printf.sprintf
          "block schema does not support groupwise storage (bits=%d, group-elements=%d)"
          bits group_elements
    | Invalid_groupwise_bits bits ->
        Printf.sprintf "groupwise packed storage requires four-bit values (got %d)"
          bits
    | Invalid_groupwise_group_elements group_elements ->
        Printf.sprintf
          "groupwise packed storage requires groups of 64 values (got %d)"
          group_elements
end

module Physical_layout = struct
  module Signedness = struct
    type t =
      | Unsigned
      | Signed
      | Twos_complement
      | Offset of int
      | Floating

    let to_string = function
      | Unsigned -> "unsigned"
      | Signed -> "signed"
      | Twos_complement -> "two's-complement"
      | Offset zero -> Printf.sprintf "offset(%d)" zero
      | Floating -> "floating"
  end

  module Field = struct
    type id =
      | Block_scale
      | Block_minimum
      | Scale_minimum_table
      | Scale_table
      | Quantized
      | Quantized_low
      | Quantized_high
      | Codebook_scale_low
      | Codebook_scale_high
      | Group_scale

    type storage = Inline | Separate
    type scalar = F16 | U8 | I8 | U16

    type t = {
      id : id;
      storage : storage;
      offset : int;
      bytes : int;
      scalar : scalar;
      alignment : int;
    }

    let make ~id ~storage ~offset ~bytes ~scalar ~alignment =
      { id; storage; offset; bytes; scalar; alignment }

    let id field = field.id
    let storage field = field.storage
    let offset field = field.offset
    let bytes field = field.bytes
    let scalar field = field.scalar
    let alignment field = field.alignment

    let id_to_string = function
      | Block_scale -> "block-scale"
      | Block_minimum -> "block-minimum"
      | Scale_minimum_table -> "scale-minimum-table"
      | Scale_table -> "scale-table"
      | Quantized -> "quantized"
      | Quantized_low -> "quantized-low"
      | Quantized_high -> "quantized-high"
      | Codebook_scale_low -> "codebook-scale-low"
      | Codebook_scale_high -> "codebook-scale-high"
      | Group_scale -> "group-scale"

    let scalar_to_string = function
      | F16 -> "f16"
      | U8 -> "u8"
      | I8 -> "i8"
      | U16 -> "u16"

    let to_string field =
      Printf.sprintf "%s[%d:%d] %s %s"
        (id_to_string field.id) field.offset field.bytes
        (scalar_to_string field.scalar)
        (if field.storage = Inline then "inline" else "separate")
  end

  module Bit_range = struct
    type t = {
      field : Field.id;
      byte_offset : int;
      byte_stride : int;
      bit_offset : int;
      bit_width : int;
      count : int;
      signedness : Signedness.t;
    }

    let make ~field ~byte_offset ~byte_stride ~bit_offset ~bit_width ~count
        ~signedness =
      { field; byte_offset; byte_stride; bit_offset; bit_width; count; signedness }

    let field range = range.field
    let byte_offset range = range.byte_offset
    let byte_stride range = range.byte_stride
    let bit_offset range = range.bit_offset
    let bit_width range = range.bit_width
    let count range = range.count
    let signedness range = range.signedness

    let to_string range =
      Printf.sprintf "%s + %d*i, bits %d..%d x%d, %s"
        (Field.id_to_string range.field)
        range.byte_offset range.bit_offset
        (range.bit_offset + range.bit_width - 1) range.count
        (Signedness.to_string range.signedness)
  end

  type t = {
    block_elements : int;
    block_bytes : int;
    storage_bytes : int;
    alignment : int;
    fields : Field.t list;
    bit_ranges : Bit_range.t list;
  }

  let make ~block_elements ~block_bytes ~storage_bytes ~alignment ~fields
      ~bit_ranges =
    { block_elements; block_bytes; storage_bytes; alignment; fields; bit_ranges }

  let block_elements layout = layout.block_elements
  let block_bytes layout = layout.block_bytes
  let storage_bytes layout = layout.storage_bytes
  let alignment layout = layout.alignment
  let fields layout = layout.fields
  let field layout id =
    List.find_opt (fun field -> Field.id field = id) layout.fields
  let bit_ranges layout = layout.bit_ranges

  let to_string layout =
    Printf.sprintf "%d values, %d inline bytes (%d storage bytes), alignment %d"
      layout.block_elements layout.block_bytes layout.storage_bytes layout.alignment
end

module Logical_decode = struct
  type scale =
    | No_scale
    | Per_block of {
        field : Physical_layout.Field.id;
        bits : int;
      }
    | Per_subblock of {
        field : Physical_layout.Field.id;
        bits : int;
      }
    | Separate_group of {
        field : Physical_layout.Field.id;
        bits : int;
      }
    | Product of scale list
    | Packed_power_of_two of {
        low_field : Physical_layout.Field.id;
        low_bits : int;
        high_field : Physical_layout.Field.id;
        high_bits : int;
        bias : int;
        denominator : int;
      }

  type equation =
    | Signed_integer of {
        code_bits : int;
        code_signedness : Physical_layout.Signedness.t;
        scale : scale;
      }
    | Affine of {
        code_bits : int;
        code_signedness : Physical_layout.Signedness.t;
        scale : scale;
        minimum : scale;
      }
    | Codebook of {
        code_bits : int;
        code_signedness : Physical_layout.Signedness.t;
        values : int list;
        scale : scale;
      }

  module Subblock = struct
    type t = { index : int; offset : int; elements : int; equation : equation }

    let make ~index ~offset ~elements ~equation =
      { index; offset; elements; equation }

    let index subblock = subblock.index
    let offset subblock = subblock.offset
    let elements subblock = subblock.elements
    let equation subblock = subblock.equation
  end

  type t = {
    block_elements : int;
    subblock_elements : int;
    subblocks : Subblock.t list;
    equation : equation;
  }

  let make ~block_elements ~subblock_elements ~equation =
    let subblocks =
      List.init (block_elements / subblock_elements) (fun index ->
          Subblock.make ~index ~offset:(index * subblock_elements)
            ~elements:subblock_elements ~equation)
    in
    { block_elements; subblock_elements; subblocks; equation }

  let block_elements decode = decode.block_elements
  let subblock_elements decode = decode.subblock_elements
  let subblocks decode = decode.subblocks
  let equation decode = decode.equation

  let code_bits = function
    | Signed_integer { code_bits; _ }
    | Affine { code_bits; _ }
    | Codebook { code_bits; _ } ->
        code_bits

  let code_signedness = function
    | Signed_integer { code_signedness; _ }
    | Affine { code_signedness; _ }
    | Codebook { code_signedness; _ } ->
        code_signedness

  let scale = function
    | Signed_integer { scale; _ }
    | Affine { scale; _ }
    | Codebook { scale; _ } ->
        scale

  let minimum = function
    | Affine { minimum; _ } -> Some minimum
    | Signed_integer _ | Codebook _ -> None

  let rec scale_to_string = function
    | No_scale -> "none"
    | Per_block { bits; _ } -> Printf.sprintf "per-block(%d-bit)" bits
    | Per_subblock { bits; _ } -> Printf.sprintf "per-subblock(%d-bit)" bits
    | Separate_group { bits; _ } ->
        Printf.sprintf "separate-group(%d-bit)" bits
    | Product factors ->
        Printf.sprintf "product(%s)"
          (String.concat " * " (List.map scale_to_string factors))
    | Packed_power_of_two { low_bits; high_bits; bias; denominator; _ } ->
        Printf.sprintf "packed-power-of-two(%d+%d-bit,bias=%d,denominator=%d)"
          low_bits high_bits bias denominator

  let to_string decode =
    Printf.sprintf "%d values, subblocks %d x %d, %d-bit %s"
      decode.block_elements (List.length decode.subblocks)
      decode.subblock_elements (code_bits decode.equation)
      (scale_to_string (scale decode.equation))
end

module Target_tactic = struct
  module Primitive = struct
    type t =
      | Portable_scalar
      | Metal_extract_bits
      | Metal_half2
      | Metal_simd_reduce
      | Cuda_byte_permute
      | Cuda_shuffle
      | Hip_bitfield
      | Hip_wave_reduce

    let to_string = function
      | Portable_scalar -> "portable-scalar"
      | Metal_extract_bits -> "metal-extract-bits"
      | Metal_half2 -> "metal-half2"
      | Metal_simd_reduce -> "metal-simd-reduce"
      | Cuda_byte_permute -> "cuda-byte-permute"
      | Cuda_shuffle -> "cuda-shuffle"
      | Hip_bitfield -> "hip-bitfield"
      | Hip_wave_reduce -> "hip-wave-reduce"
  end

  type backend = Portable | Metal | Cuda | Hip
  type execution = Scalar | Simd of int | Warp of int | Wave of int

  type decode =
    | Direct_integer
    | Packed_nibble
    | Packed_bitfield
    | Codebook

  type t = {
    backend : backend;
    execution : execution;
    decode : decode;
    primitives : Primitive.t list;
  }

  let make ~backend ~execution ~decode ~primitives =
    { backend; execution; decode; primitives }
  let backend tactic = tactic.backend
  let execution tactic = tactic.execution

  let vector_width tactic =
    match tactic.execution with
    | Scalar -> 1
    | Simd width | Warp width | Wave width -> width

  let decode tactic = tactic.decode
  let primitives tactic = tactic.primitives

  let backend_name = function
    | Portable -> "portable"
    | Metal -> "metal"
    | Cuda -> "cuda"
    | Hip -> "hip"

  let execution_name = function
    | Scalar -> "scalar"
    | Simd width -> Printf.sprintf "simd%d" width
    | Warp width -> Printf.sprintf "warp%d" width
    | Wave width -> Printf.sprintf "wave%d" width

  let name tactic =
    backend_name tactic.backend ^ "-" ^ execution_name tactic.execution
end

type t = {
  storage : Ir.Linear_storage.layout;
  quant_type : Ir.Dtype.quant_type option;
  physical : Physical_layout.t;
  logical : Logical_decode.t;
  tactics : Target_tactic.t list;
}

let inline_field id offset bytes scalar =
  Physical_layout.Field.make ~id ~storage:Physical_layout.Field.Inline ~offset
    ~bytes ~scalar ~alignment:(match scalar with Physical_layout.Field.F16 | Physical_layout.Field.U16 -> 2 | Physical_layout.Field.U8 | Physical_layout.Field.I8 -> 1)

let separate_field id bytes scalar =
  Physical_layout.Field.make ~id ~storage:Physical_layout.Field.Separate ~offset:0
    ~bytes ~scalar
    ~alignment:(match scalar with Physical_layout.Field.F16 | Physical_layout.Field.U16 -> 2 | Physical_layout.Field.U8 | Physical_layout.Field.I8 -> 1)

let bit_range ~field ~byte_offset ~byte_stride ~bit_offset ~bit_width ~count
    ~signedness =
  Physical_layout.Bit_range.make ~field ~byte_offset ~byte_stride ~bit_offset
    ~bit_width ~count ~signedness

let nibble_ranges ~field ~bytes ~signedness =
  [ bit_range ~field ~byte_offset:0 ~byte_stride:1 ~bit_offset:0 ~bit_width:4
      ~count:bytes ~signedness;
    bit_range ~field ~byte_offset:0 ~byte_stride:1 ~bit_offset:4 ~bit_width:4
      ~count:bytes ~signedness ]

let byte_bit_ranges ~field ~bytes ~bits ~signedness =
  List.init bytes (fun byte ->
      List.init bits (fun bit ->
          bit_range ~field ~byte_offset:byte ~byte_stride:0 ~bit_offset:bit
            ~bit_width:1 ~count:1 ~signedness))
  |> List.concat

let q4_k_scale_ranges () =
  [ bit_range ~field:Physical_layout.Field.Scale_minimum_table ~byte_offset:0
      ~byte_stride:1 ~bit_offset:0 ~bit_width:6 ~count:4
      ~signedness:Physical_layout.Signedness.Unsigned;
    bit_range ~field:Physical_layout.Field.Scale_minimum_table ~byte_offset:4
      ~byte_stride:1 ~bit_offset:0 ~bit_width:6 ~count:4
      ~signedness:Physical_layout.Signedness.Unsigned;
    bit_range ~field:Physical_layout.Field.Scale_minimum_table ~byte_offset:0
      ~byte_stride:1 ~bit_offset:6 ~bit_width:2 ~count:4
      ~signedness:Physical_layout.Signedness.Unsigned;
    bit_range ~field:Physical_layout.Field.Scale_minimum_table ~byte_offset:4
      ~byte_stride:1 ~bit_offset:6 ~bit_width:2 ~count:4
      ~signedness:Physical_layout.Signedness.Unsigned;
    bit_range ~field:Physical_layout.Field.Scale_minimum_table ~byte_offset:8
      ~byte_stride:1 ~bit_offset:0 ~bit_width:4 ~count:4
      ~signedness:Physical_layout.Signedness.Unsigned;
    bit_range ~field:Physical_layout.Field.Scale_minimum_table ~byte_offset:8
      ~byte_stride:1 ~bit_offset:4 ~bit_width:4 ~count:4
      ~signedness:Physical_layout.Signedness.Unsigned ]

let q4_k_physical ~quant ~with_high_plane =
  let fields =
    [ inline_field Physical_layout.Field.Block_scale 0 2
        Physical_layout.Field.F16;
      inline_field Physical_layout.Field.Block_minimum 2 2
        Physical_layout.Field.F16;
      inline_field Physical_layout.Field.Scale_minimum_table 4 12
        Physical_layout.Field.U8 ]
    @ if with_high_plane then
        [ inline_field Physical_layout.Field.Quantized_high 16 32
            Physical_layout.Field.U8;
          inline_field Physical_layout.Field.Quantized_low 48 128
            Physical_layout.Field.U8 ]
      else
        [ inline_field Physical_layout.Field.Quantized 16 128
            Physical_layout.Field.U8 ]
  in
  let quant_ranges =
    if with_high_plane then
      byte_bit_ranges ~field:Physical_layout.Field.Quantized_high ~bytes:32
        ~bits:8 ~signedness:Physical_layout.Signedness.Unsigned
      @ nibble_ranges ~field:Physical_layout.Field.Quantized_low ~bytes:128
          ~signedness:Physical_layout.Signedness.Unsigned
    else
      nibble_ranges ~field:Physical_layout.Field.Quantized ~bytes:128
        ~signedness:Physical_layout.Signedness.Unsigned
  in
  Physical_layout.make
    ~block_elements:(Ir.Tensor_layout.block_elements quant)
    ~block_bytes:(Ir.Tensor_layout.block_bytes quant)
    ~storage_bytes:(Ir.Tensor_layout.block_bytes quant)
    ~alignment:2 ~fields
    ~bit_ranges:(q4_k_scale_ranges () @ quant_ranges)

let q6_k_physical quant =
  let fields =
    [ inline_field Physical_layout.Field.Quantized_low 0 128
        Physical_layout.Field.U8;
      inline_field Physical_layout.Field.Quantized_high 128 64
        Physical_layout.Field.U8;
      inline_field Physical_layout.Field.Scale_table 192 16
        Physical_layout.Field.I8;
      inline_field Physical_layout.Field.Block_scale 208 2
        Physical_layout.Field.F16 ]
  in
  let low_ranges =
    nibble_ranges ~field:Physical_layout.Field.Quantized_low ~bytes:128
      ~signedness:Physical_layout.Signedness.Unsigned
  in
  let high_ranges =
    List.init 2 (fun plane ->
        byte_bit_ranges ~field:Physical_layout.Field.Quantized_high ~bytes:32
          ~bits:4 ~signedness:Physical_layout.Signedness.Unsigned
        |> List.map (fun range ->
               bit_range ~field:(Physical_layout.Bit_range.field range)
                 ~byte_offset:(plane * 32 + Physical_layout.Bit_range.byte_offset range)
                 ~byte_stride:(Physical_layout.Bit_range.byte_stride range)
                 ~bit_offset:(Physical_layout.Bit_range.bit_offset range * 2)
                 ~bit_width:2 ~count:1
                 ~signedness:Physical_layout.Signedness.Unsigned))
    |> List.concat
  in
  let scale_ranges =
    [ bit_range ~field:Physical_layout.Field.Scale_table ~byte_offset:0
        ~byte_stride:1 ~bit_offset:0 ~bit_width:8 ~count:16
        ~signedness:Physical_layout.Signedness.Signed ]
  in
  Physical_layout.make
    ~block_elements:(Ir.Tensor_layout.block_elements quant)
    ~block_bytes:(Ir.Tensor_layout.block_bytes quant)
    ~storage_bytes:(Ir.Tensor_layout.block_bytes quant)
    ~alignment:2 ~fields ~bit_ranges:(low_ranges @ high_ranges @ scale_ranges)

let q8_0_physical quant =
  Physical_layout.make
    ~block_elements:(Ir.Tensor_layout.block_elements quant)
    ~block_bytes:(Ir.Tensor_layout.block_bytes quant)
    ~storage_bytes:(Ir.Tensor_layout.block_bytes quant)
    ~alignment:2
    ~fields:
      [ inline_field Physical_layout.Field.Block_scale 0 2
          Physical_layout.Field.F16;
        inline_field Physical_layout.Field.Quantized 2 32
          Physical_layout.Field.I8 ]
    ~bit_ranges:
      [ bit_range ~field:Physical_layout.Field.Quantized ~byte_offset:0
          ~byte_stride:1 ~bit_offset:0 ~bit_width:8 ~count:32
          ~signedness:Physical_layout.Signedness.Signed ]

let q4_0_physical quant =
  Physical_layout.make
    ~block_elements:(Ir.Tensor_layout.block_elements quant)
    ~block_bytes:(Ir.Tensor_layout.block_bytes quant)
    ~storage_bytes:(Ir.Tensor_layout.block_bytes quant)
    ~alignment:2
    ~fields:
      [ inline_field Physical_layout.Field.Block_scale 0 2
          Physical_layout.Field.F16;
        inline_field Physical_layout.Field.Quantized 2 16
          Physical_layout.Field.U8 ]
    ~bit_ranges:
      (nibble_ranges ~field:Physical_layout.Field.Quantized ~bytes:16
         ~signedness:Physical_layout.Signedness.Unsigned)

let q5_0_physical quant =
  Physical_layout.make
    ~block_elements:(Ir.Tensor_layout.block_elements quant)
    ~block_bytes:(Ir.Tensor_layout.block_bytes quant)
    ~storage_bytes:(Ir.Tensor_layout.block_bytes quant)
    ~alignment:2
    ~fields:
      [ inline_field Physical_layout.Field.Block_scale 0 2
          Physical_layout.Field.F16;
        inline_field Physical_layout.Field.Quantized_high 2 4
          Physical_layout.Field.U8;
        inline_field Physical_layout.Field.Quantized_low 6 16
          Physical_layout.Field.U8 ]
    ~bit_ranges:
      (byte_bit_ranges ~field:Physical_layout.Field.Quantized_high ~bytes:4
         ~bits:8 ~signedness:Physical_layout.Signedness.Unsigned
       @ nibble_ranges ~field:Physical_layout.Field.Quantized_low ~bytes:16
           ~signedness:Physical_layout.Signedness.Unsigned)

let iq4_xs_physical quant =
  let scale_high_ranges =
    List.init 8 (fun index ->
        bit_range ~field:Physical_layout.Field.Codebook_scale_high
          ~byte_offset:0 ~byte_stride:0 ~bit_offset:(index * 2) ~bit_width:2
          ~count:1 ~signedness:Physical_layout.Signedness.Unsigned)
  in
  let scale_low_ranges =
    nibble_ranges ~field:Physical_layout.Field.Codebook_scale_low ~bytes:4
      ~signedness:Physical_layout.Signedness.Unsigned
  in
  Physical_layout.make
    ~block_elements:(Ir.Tensor_layout.block_elements quant)
    ~block_bytes:(Ir.Tensor_layout.block_bytes quant)
    ~storage_bytes:(Ir.Tensor_layout.block_bytes quant)
    ~alignment:2
    ~fields:
      [ inline_field Physical_layout.Field.Block_scale 0 2
          Physical_layout.Field.F16;
        inline_field Physical_layout.Field.Codebook_scale_high 2 2
          Physical_layout.Field.U16;
        inline_field Physical_layout.Field.Codebook_scale_low 4 4
          Physical_layout.Field.U8;
        inline_field Physical_layout.Field.Quantized 8 128
          Physical_layout.Field.U8 ]
    ~bit_ranges:
      (scale_high_ranges @ scale_low_ranges
       @ nibble_ranges ~field:Physical_layout.Field.Quantized ~bytes:128
           ~signedness:Physical_layout.Signedness.Unsigned)

let w4_physical ~bits ~group_elements =
  let block_elements = group_elements in
  let block_bytes = (block_elements * bits) / 8 in
  Physical_layout.make ~block_elements ~block_bytes
    ~storage_bytes:(block_bytes + 2)
    ~alignment:2
    ~fields:
      [ inline_field Physical_layout.Field.Quantized 0 block_bytes
          Physical_layout.Field.U8;
        separate_field Physical_layout.Field.Group_scale 2
          Physical_layout.Field.F16 ]
    ~bit_ranges:
      (nibble_ranges ~field:Physical_layout.Field.Quantized ~bytes:block_bytes
         ~signedness:Physical_layout.Signedness.Unsigned)

let q4_k_decode quant =
  let equation =
    Logical_decode.Affine
      { code_bits = 4;
        code_signedness = Physical_layout.Signedness.Unsigned;
        scale =
          Logical_decode.Product
            [ Logical_decode.Per_block
                { field = Physical_layout.Field.Block_scale; bits = 16 };
              Logical_decode.Per_subblock
                { field = Physical_layout.Field.Scale_minimum_table; bits = 6 } ];
        minimum =
          Logical_decode.Product
            [ Logical_decode.Per_block
                { field = Physical_layout.Field.Block_minimum; bits = 16 };
              Logical_decode.Per_subblock
                { field = Physical_layout.Field.Scale_minimum_table; bits = 6 } ] }
  in
  Logical_decode.make
    ~block_elements:(Ir.Tensor_layout.block_elements quant)
    ~subblock_elements:32 ~equation

let q5_k_decode quant =
  let equation =
    Logical_decode.Affine
      { code_bits = 5;
        code_signedness = Physical_layout.Signedness.Unsigned;
        scale =
          Logical_decode.Product
            [ Logical_decode.Per_block
                { field = Physical_layout.Field.Block_scale; bits = 16 };
              Logical_decode.Per_subblock
                { field = Physical_layout.Field.Scale_minimum_table; bits = 6 } ];
        minimum =
          Logical_decode.Product
            [ Logical_decode.Per_block
                { field = Physical_layout.Field.Block_minimum; bits = 16 };
              Logical_decode.Per_subblock
                { field = Physical_layout.Field.Scale_minimum_table; bits = 6 } ] }
  in
  Logical_decode.make
    ~block_elements:(Ir.Tensor_layout.block_elements quant)
    ~subblock_elements:32 ~equation

let q6_k_decode quant =
  let equation =
    Logical_decode.Signed_integer
      { code_bits = 6;
        code_signedness = Physical_layout.Signedness.Offset 32;
        scale =
          Logical_decode.Product
            [ Logical_decode.Per_block
                { field = Physical_layout.Field.Block_scale; bits = 16 };
              Logical_decode.Per_subblock
                { field = Physical_layout.Field.Scale_table; bits = 8 } ] }
  in
  Logical_decode.make
    ~block_elements:(Ir.Tensor_layout.block_elements quant)
    ~subblock_elements:32 ~equation

let q8_0_decode quant =
  let equation =
    Logical_decode.Signed_integer
      { code_bits = 8;
        code_signedness = Physical_layout.Signedness.Signed;
        scale =
          Logical_decode.Per_block
            { field = Physical_layout.Field.Block_scale; bits = 16 } }
  in
  Logical_decode.make
    ~block_elements:(Ir.Tensor_layout.block_elements quant)
    ~subblock_elements:32 ~equation

let q4_0_decode quant =
  let equation =
    Logical_decode.Signed_integer
      { code_bits = 4;
        code_signedness = Physical_layout.Signedness.Offset 8;
        scale =
          Logical_decode.Per_block
            { field = Physical_layout.Field.Block_scale; bits = 16 } }
  in
  Logical_decode.make
    ~block_elements:(Ir.Tensor_layout.block_elements quant)
    ~subblock_elements:32 ~equation

let q5_0_decode quant =
  let equation =
    Logical_decode.Signed_integer
      { code_bits = 5;
        code_signedness = Physical_layout.Signedness.Offset 16;
        scale =
          Logical_decode.Per_block
            { field = Physical_layout.Field.Block_scale; bits = 16 } }
  in
  Logical_decode.make
    ~block_elements:(Ir.Tensor_layout.block_elements quant)
    ~subblock_elements:32 ~equation

let iq4_xs_decode quant =
  let equation =
    Logical_decode.Codebook
      { code_bits = 4;
        code_signedness = Physical_layout.Signedness.Unsigned;
        values =
          [ -127; -104; -83; -65; -49; -35; -22; -10;
            1; 13; 25; 38; 53; 69; 89; 113 ];
        scale =
          Logical_decode.Product
            [ Logical_decode.Per_block
                { field = Physical_layout.Field.Block_scale; bits = 16 };
              Logical_decode.Packed_power_of_two
                { low_field = Physical_layout.Field.Codebook_scale_low;
                  low_bits = 4;
                  high_field = Physical_layout.Field.Codebook_scale_high;
                  high_bits = 2;
                  bias = 32;
                  denominator = 8 } ] }
  in
  Logical_decode.make
    ~block_elements:(Ir.Tensor_layout.block_elements quant)
    ~subblock_elements:32 ~equation

let w4_decode ~group_elements =
  let equation =
    Logical_decode.Signed_integer
      { code_bits = 4;
        code_signedness = Physical_layout.Signedness.Twos_complement;
        scale =
          Logical_decode.Separate_group
            { field = Physical_layout.Field.Group_scale; bits = 16 } }
  in
  Logical_decode.make ~block_elements:group_elements
    ~subblock_elements:group_elements ~equation

let tile32_affine_physical ~bits =
  let quant_bytes = (32 * bits) / 8 in
  let block_bytes = 4 + quant_bytes in
  let bit_ranges =
    if bits = 4 then
      nibble_ranges ~field:Physical_layout.Field.Quantized ~bytes:quant_bytes
        ~signedness:Physical_layout.Signedness.Unsigned
    else
      [ bit_range ~field:Physical_layout.Field.Quantized ~byte_offset:0
          ~byte_stride:1 ~bit_offset:0 ~bit_width:8 ~count:32
          ~signedness:Physical_layout.Signedness.Unsigned ]
  in
  Physical_layout.make ~block_elements:32 ~block_bytes ~storage_bytes:block_bytes
    ~alignment:2
    ~fields:
      [ inline_field Physical_layout.Field.Block_scale 0 2
          Physical_layout.Field.F16;
        inline_field Physical_layout.Field.Block_minimum 2 2
          Physical_layout.Field.F16;
        inline_field Physical_layout.Field.Quantized 4 quant_bytes
          Physical_layout.Field.U8 ]
    ~bit_ranges

let tile32_affine_decode ~bits =
  let equation =
    Logical_decode.Affine
      { code_bits = bits;
        code_signedness = Physical_layout.Signedness.Unsigned;
        scale =
          Logical_decode.Per_block
            { field = Physical_layout.Field.Block_scale; bits = 16 };
        minimum =
          Logical_decode.Per_block
            { field = Physical_layout.Field.Block_minimum; bits = 16 } }
  in
  Logical_decode.make ~block_elements:32 ~subblock_elements:32 ~equation

let primitives ~decode ~backend =
  match backend, decode with
  | Target_tactic.Portable, _ -> [ Target_tactic.Primitive.Portable_scalar ]
  | Target_tactic.Metal, Target_tactic.Direct_integer ->
      [ Target_tactic.Primitive.Metal_half2;
        Target_tactic.Primitive.Metal_simd_reduce ]
  | Target_tactic.Metal, _ ->
      [ Target_tactic.Primitive.Metal_extract_bits;
        Target_tactic.Primitive.Metal_half2;
        Target_tactic.Primitive.Metal_simd_reduce ]
  | Target_tactic.Cuda, Target_tactic.Direct_integer ->
      [ Target_tactic.Primitive.Cuda_shuffle ]
  | Target_tactic.Cuda, _ ->
      [ Target_tactic.Primitive.Cuda_byte_permute;
        Target_tactic.Primitive.Cuda_shuffle ]
  | Target_tactic.Hip, Target_tactic.Direct_integer ->
      [ Target_tactic.Primitive.Hip_wave_reduce ]
  | Target_tactic.Hip, _ ->
      [ Target_tactic.Primitive.Hip_bitfield;
        Target_tactic.Primitive.Hip_wave_reduce ]

let tactics decode =
  [ Target_tactic.make ~backend:Target_tactic.Portable
      ~execution:Target_tactic.Scalar ~decode
      ~primitives:(primitives ~decode ~backend:Target_tactic.Portable);
    Target_tactic.make ~backend:Target_tactic.Metal
      ~execution:(Target_tactic.Simd 32) ~decode
      ~primitives:(primitives ~decode ~backend:Target_tactic.Metal);
    Target_tactic.make ~backend:Target_tactic.Cuda
      ~execution:(Target_tactic.Warp 32) ~decode
      ~primitives:(primitives ~decode ~backend:Target_tactic.Cuda);
    Target_tactic.make ~backend:Target_tactic.Hip
      ~execution:(Target_tactic.Wave 32) ~decode
      ~primitives:(primitives ~decode ~backend:Target_tactic.Hip) ]

let make ~storage ~quant_type ~physical ~logical ~decode =
  { storage; quant_type; physical; logical; tactics = tactics decode }

let of_quant quant =
  match quant with
  | Ir.Dtype.Q8_0 ->
      make ~storage:(Ir.Linear_storage.Block_quantized quant)
        ~quant_type:(Some quant) ~physical:(q8_0_physical quant)
        ~logical:(q8_0_decode quant) ~decode:Target_tactic.Direct_integer
  | Ir.Dtype.Q4_K ->
      make ~storage:(Ir.Linear_storage.Block_quantized quant)
        ~quant_type:(Some quant)
        ~physical:(q4_k_physical ~quant ~with_high_plane:false)
        ~logical:(q4_k_decode quant) ~decode:Target_tactic.Packed_bitfield
  | Ir.Dtype.Q5_K ->
      make ~storage:(Ir.Linear_storage.Block_quantized quant)
        ~quant_type:(Some quant)
        ~physical:(q4_k_physical ~quant ~with_high_plane:true)
        ~logical:(q5_k_decode quant) ~decode:Target_tactic.Packed_bitfield
  | Ir.Dtype.Q6_K ->
      make ~storage:(Ir.Linear_storage.Block_quantized quant)
        ~quant_type:(Some quant) ~physical:(q6_k_physical quant)
        ~logical:(q6_k_decode quant) ~decode:Target_tactic.Packed_bitfield
  | Ir.Dtype.Q5_0 ->
      make ~storage:(Ir.Linear_storage.Block_quantized quant)
        ~quant_type:(Some quant) ~physical:(q5_0_physical quant)
        ~logical:(q5_0_decode quant) ~decode:Target_tactic.Packed_nibble
  | Ir.Dtype.Q4_0 ->
      make ~storage:(Ir.Linear_storage.Block_quantized quant)
        ~quant_type:(Some quant) ~physical:(q4_0_physical quant)
        ~logical:(q4_0_decode quant) ~decode:Target_tactic.Packed_nibble
  | Ir.Dtype.IQ4_XS ->
      make ~storage:(Ir.Linear_storage.Block_quantized quant)
        ~quant_type:(Some quant) ~physical:(iq4_xs_physical quant)
        ~logical:(iq4_xs_decode quant) ~decode:Target_tactic.Codebook
  | Ir.Dtype.Tile32_Affine4 ->
      make ~storage:(Ir.Linear_storage.Block_quantized quant)
        ~quant_type:(Some quant) ~physical:(tile32_affine_physical ~bits:4)
        ~logical:(tile32_affine_decode ~bits:4)
        ~decode:Target_tactic.Packed_nibble
  | Ir.Dtype.Tile32_Affine8 ->
      make ~storage:(Ir.Linear_storage.Block_quantized quant)
        ~quant_type:(Some quant) ~physical:(tile32_affine_physical ~bits:8)
        ~logical:(tile32_affine_decode ~bits:8)
        ~decode:Target_tactic.Direct_integer

let of_storage storage =
  match storage with
  | Ir.Linear_storage.Block_quantized quant -> Ok (of_quant quant)
  | Ir.Linear_storage.Dense _ -> Error (Error.Unsupported_storage storage)
  | Ir.Linear_storage.Groupwise_packed { bits; group_elements } ->
      if bits <> 4 then Error (Error.Invalid_groupwise_bits bits)
      else if group_elements <> 64 then
        Error (Error.Invalid_groupwise_group_elements group_elements)
      else
        Ok
          (make ~storage ~quant_type:None
             ~physical:(w4_physical ~bits ~group_elements)
             ~logical:(w4_decode ~group_elements) ~decode:Target_tactic.Packed_nibble)

let tile32_affine_schema ~bits =
  if bits <> 4 && bits <> 8 then Error (Error.Invalid_groupwise_bits bits)
  else
    let quant =
      if bits = 4 then Ir.Dtype.Tile32_Affine4 else Ir.Dtype.Tile32_Affine8
    in
    Ok (of_quant quant)

let storage schema = schema.storage
let quant_type schema = schema.quant_type
let physical_layout schema = schema.physical
let logical_decode schema = schema.logical
let target_tactics schema = schema.tactics
let block_elements schema = Physical_layout.block_elements schema.physical
let block_bytes schema = Physical_layout.block_bytes schema.physical
let to_string schema =
  match schema.quant_type, Logical_decode.equation schema.logical with
  | Some quant, _ -> Ir.Dtype.quant_to_string quant
  | None, Logical_decode.Affine { code_bits; _ } ->
      Printf.sprintf "Tile32_Affine%d" code_bits
  | None, _ -> "W4A16-G64"
