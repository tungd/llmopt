(** Validated physical and logical contracts for quantized Linear storage.

    A schema is the single immutable description of one storage layout: its
    byte fields and packed ranges, the numerical decode equation, and the
    target tactics that can interpret that equation.  The schema deliberately
    stays independent of any backend compiler package. *)

module Error : sig
  type t =
    | Unsupported_storage of Ir.Linear_storage.layout
    | Invalid_groupwise_bits of int
    | Invalid_groupwise_group_elements of int

  val to_string : t -> string
end

module Physical_layout : sig
  module Signedness : sig
    type t =
      | Unsigned
      | Signed
      | Twos_complement
      | Offset of int
      | Floating

    val to_string : t -> string
  end

  module Field : sig
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

    type t

    val id : t -> id
    val storage : t -> storage
    val offset : t -> int
    val bytes : t -> int
    val scalar : t -> scalar
    val alignment : t -> int
    val to_string : t -> string
  end

  module Bit_range : sig
    type t

    val field : t -> Field.id
    val byte_offset : t -> int
    val byte_stride : t -> int
    val bit_offset : t -> int
    val bit_width : t -> int
    val count : t -> int
    val signedness : t -> Signedness.t
    val to_string : t -> string
  end

  type t

  val block_elements : t -> int
  val block_bytes : t -> int
  val storage_bytes : t -> int
  val alignment : t -> int
  val fields : t -> Field.t list
  val field : t -> Field.id -> Field.t option
  val bit_ranges : t -> Bit_range.t list
  val to_string : t -> string
end

module Logical_decode : sig
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

  module Subblock : sig
    type t

    val index : t -> int
    val offset : t -> int
    val elements : t -> int
    val equation : t -> equation
  end

  type t

  val block_elements : t -> int
  val subblock_elements : t -> int
  val subblocks : t -> Subblock.t list
  val equation : t -> equation
  val code_bits : equation -> int
  val code_signedness : equation -> Physical_layout.Signedness.t
  val scale : equation -> scale
  val minimum : equation -> scale option
  val to_string : t -> string
end

module Target_tactic : sig
  module Primitive : sig
    type t =
      | Portable_scalar
      | Metal_extract_bits
      | Metal_half2
      | Metal_simd_reduce
      | Cuda_byte_permute
      | Cuda_shuffle
      | Hip_bitfield
      | Hip_wave_reduce

    val to_string : t -> string
  end

  type backend = Portable | Metal | Cuda | Hip

  type execution = Scalar | Simd of int | Warp of int | Wave of int

  type decode =
    | Direct_integer
    | Packed_nibble
    | Packed_bitfield
    | Codebook

  type t

  val backend : t -> backend
  val execution : t -> execution
  val vector_width : t -> int
  val decode : t -> decode
  val primitives : t -> Primitive.t list
  val name : t -> string
end

type t

val of_storage : Ir.Linear_storage.layout -> (t, Error.t) result

val storage : t -> Ir.Linear_storage.layout
val quant_type : t -> Ir.Dtype.quant_type option
val physical_layout : t -> Physical_layout.t
val logical_decode : t -> Logical_decode.t
val target_tactics : t -> Target_tactic.t list
val block_elements : t -> int
val block_bytes : t -> int
val to_string : t -> string
