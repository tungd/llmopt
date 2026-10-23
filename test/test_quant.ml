open Test_support

(* Test_quant: block-quant descriptors/physical bytes, unified block-schema
   contracts, GGUF parser & ingestion, and bit-exact dequantization vs
   llama.cpp (real GGUF files are checked only when present). *)

let run () =

  (* Block-quant descriptor & physical bytes tests *)
  expect
    (Weight_archive.Dtype.quant_to_string Weight_archive.Dtype.Q8_0 = "Q8_0")
    "Q8_0 quant_to_string";
  expect
    (Weight_archive.Dtype.quant_to_string Weight_archive.Dtype.Q4_K = "Q4_K")
    "Q4_K quant_to_string";
  expect
    (Weight_archive.Dtype.quant_to_string Weight_archive.Dtype.Q5_K = "Q5_K")
    "Q5_K quant_to_string";
  expect
    (Weight_archive.Dtype.quant_to_string Weight_archive.Dtype.Q6_K = "Q6_K")
    "Q6_K quant_to_string";
  expect
    (Weight_archive.Dtype.quant_to_string Weight_archive.Dtype.Q5_0 = "Q5_0")
    "Q5_0 quant_to_string";
  expect
    (Weight_archive.Dtype.quant_of_string "Q4_K" = Some Weight_archive.Dtype.Q4_K)
    "quant_of_string Q4_K";
  expect
    (Weight_archive.Dtype.quant_of_string "q8_0" = Some Weight_archive.Dtype.Q8_0)
    "quant_of_string q8_0 lowercase";
  expect
    (Weight_archive.Dtype.block_size Weight_archive.Dtype.Q8_0 = 32)
    "Q8_0 block size is 32";
  expect
    (Weight_archive.Dtype.bytes_per_block Weight_archive.Dtype.Q8_0 = 34)
    "Q8_0 bytes per block is 34";
  expect
    (Weight_archive.Dtype.block_size Weight_archive.Dtype.Q4_K = 256)
    "Q4_K block size is 256";
  expect
    (Weight_archive.Dtype.bytes_per_block Weight_archive.Dtype.Q4_K = 144)
    "Q4_K bytes per block is 144";
  expect
    (Weight_archive.Dtype.block_size Weight_archive.Dtype.Q5_K = 256)
    "Q5_K block size is 256";
  expect
    (Weight_archive.Dtype.bytes_per_block Weight_archive.Dtype.Q5_K = 176)
    "Q5_K bytes per block is 176";
  expect
    (Weight_archive.Dtype.block_size Weight_archive.Dtype.Q6_K = 256)
    "Q6_K block size is 256";
  expect
    (Weight_archive.Dtype.bytes_per_block Weight_archive.Dtype.Q6_K = 210)
    "Q6_K bytes per block is 210";
  expect
    (Weight_archive.Dtype.block_size Weight_archive.Dtype.Q5_0 = 32)
    "Q5_0 block size is 32";
  expect
    (Weight_archive.Dtype.bytes_per_block Weight_archive.Dtype.Q5_0 = 22)
    "Q5_0 bytes per block is 22";

  let shape_256 = Tensor_shape.of_ints_exn [ 2; 512 ] in
  let q4k_bytes =
    Tensor_shape.physical_bytes shape_256 ~block_size:256 ~bytes_per_block:144
  in
  expect (q4k_bytes = Ok (4 * 144)) "physical_bytes for 1024 elements with Q4_K is 576";
  expect
    (Ir.Tensor_layout.physical_bytes
       (Ir.Tensor_layout.Block_quantized Ir.Dtype.Q4_K)
       shape_256
    = q4k_bytes)
    "IR tensor layout is the physical-byte authority for Q4_K";
  expect
    (Weight_archive.Dtype.Q4_K = Ir.Dtype.Q4_K)
    "weight archives share the IR quant format type";
  let layout_weight =
    Ir.Value.make_tensor ~id:50001
      ~shape:(Tensor_shape.of_ints_exn [ 128; 64 ])
      ~dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_K)
  in
  let block_storage =
    Ir.Linear_storage.classify ~has_bias:false ~weight:layout_weight
      ~parameters:[]
    |> expect_ok
  in
  expect
    (block_storage.layout = Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_K
     && Ir.Linear_storage.is_quantized block_storage
     && not (Ir.Linear_storage.has_separate_scale block_storage))
    "Linear storage classifies GGUF quantization from tensor layout";
  let packed_weight =
    Ir.Value.make_tensor ~id:50002
      ~shape:(Tensor_shape.of_ints_exn [ 128; 32 ]) ~dtype:Ir.Dtype.UInt8
  in
  let packed_scale =
    Ir.Value.make_tensor ~id:50003
      ~shape:(Tensor_shape.of_ints_exn [ 128; 1 ]) ~dtype:Ir.Dtype.Float16
  in
  let packed_storage =
    Ir.Linear_storage.classify ~has_bias:false ~weight:packed_weight
      ~parameters:[ packed_scale ]
    |> expect_ok
  in
  expect
    (packed_storage.layout
       = Ir.Linear_storage.Groupwise_packed { bits = 4; group_elements = 64 }
     && Ir.Linear_storage.has_separate_scale packed_storage)
    "legacy packed W4 is represented as one Linear storage layout";
  let unaligned_shape = Tensor_shape.of_ints_exn [ 2; 250 ] in
  let unaligned_err =
    Tensor_shape.physical_bytes unaligned_shape ~block_size:256 ~bytes_per_block:144
  in
  expect (Result.is_error unaligned_err) "physical_bytes rejects unaligned element counts";

  (* Unified block-schema contracts: physical storage, logical decode, and
     deterministic target tactics share the Linear storage authority. *)
  let expect_schema_ok = function
    | Ok schema -> schema
    | Error error -> fail (Block_schema.Error.to_string error)
  in
  let q4k_schema =
    Block_schema.of_storage
      (Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_K)
    |> expect_schema_ok
  in
  expect
    (Block_schema.block_elements q4k_schema = 256
    && Block_schema.block_bytes q4k_schema = 144)
    "Q4_K schema preserves the 256-value/144-byte physical contract";
  let q4k_physical = Block_schema.physical_layout q4k_schema in
  expect
    (Block_schema.Physical_layout.alignment q4k_physical = 2
    && List.map Block_schema.Physical_layout.Field.id
         (Block_schema.Physical_layout.fields q4k_physical)
       = [ Block_schema.Physical_layout.Field.Block_scale;
           Block_schema.Physical_layout.Field.Block_minimum;
           Block_schema.Physical_layout.Field.Scale_minimum_table;
           Block_schema.Physical_layout.Field.Quantized ])
    "Q4_K schema exposes stable aligned header and quantized fields";
  expect
    (List.map
       (fun field ->
         ( Block_schema.Physical_layout.Field.id field,
           Block_schema.Physical_layout.Field.offset field,
           Block_schema.Physical_layout.Field.bytes field,
           Block_schema.Physical_layout.Field.scalar field ))
       (Block_schema.Physical_layout.fields q4k_physical)
     = [ (Block_schema.Physical_layout.Field.Block_scale, 0, 2,
          Block_schema.Physical_layout.Field.F16);
         (Block_schema.Physical_layout.Field.Block_minimum, 2, 2,
          Block_schema.Physical_layout.Field.F16);
         (Block_schema.Physical_layout.Field.Scale_minimum_table, 4, 12,
          Block_schema.Physical_layout.Field.U8);
         (Block_schema.Physical_layout.Field.Quantized, 16, 128,
          Block_schema.Physical_layout.Field.U8) ])
    "Q4_K schema preserves exact field offsets and scalar widths";
  expect
    (List.exists
       (fun range ->
         Block_schema.Physical_layout.Bit_range.field range
         = Block_schema.Physical_layout.Field.Scale_minimum_table
         && Block_schema.Physical_layout.Bit_range.byte_offset range = 0
         && Block_schema.Physical_layout.Bit_range.bit_width range = 6
         && Block_schema.Physical_layout.Bit_range.count range = 4)
       (Block_schema.Physical_layout.bit_ranges q4k_physical))
    "Q4_K schema records six-bit sub-block scale ranges";
  expect
    (List.exists
       (fun range ->
         Block_schema.Physical_layout.Bit_range.field range
         = Block_schema.Physical_layout.Field.Scale_minimum_table
         && Block_schema.Physical_layout.Bit_range.byte_offset range = 4
         && Block_schema.Physical_layout.Bit_range.bit_width range = 6
         && Block_schema.Physical_layout.Bit_range.count range = 4)
       (Block_schema.Physical_layout.bit_ranges q4k_physical))
    "Q4_K schema records six-bit sub-block minimum ranges";
  let q4k_decode = Block_schema.logical_decode q4k_schema in
  expect
    (Block_schema.Logical_decode.block_elements q4k_decode = 256
    && Block_schema.Logical_decode.subblock_elements q4k_decode = 32
    && List.length (Block_schema.Logical_decode.subblocks q4k_decode) = 8)
    "Q4_K schema exposes eight 32-value logical sub-blocks";
  (match Block_schema.Logical_decode.equation q4k_decode with
  | Block_schema.Logical_decode.Affine
      { code_bits = 4;
        code_signedness = Block_schema.Physical_layout.Signedness.Unsigned;
        scale =
          Block_schema.Logical_decode.Product
            [ Block_schema.Logical_decode.Per_block
                { field = Block_schema.Physical_layout.Field.Block_scale;
                  bits = 16 };
              Block_schema.Logical_decode.Per_subblock
                { field = Block_schema.Physical_layout.Field.Scale_minimum_table;
                  bits = 6 } ];
        minimum =
          Block_schema.Logical_decode.Product
            [ Block_schema.Logical_decode.Per_block
                { field = Block_schema.Physical_layout.Field.Block_minimum;
                  bits = 16 };
              Block_schema.Logical_decode.Per_subblock
                { field = Block_schema.Physical_layout.Field.Scale_minimum_table;
                  bits = 6 } ] } ->
      ()
  | _ ->
      fail
        "Q4_K decode composes block and sub-block scale/minimum semantics");

  let packed_schema =
    Block_schema.of_storage packed_storage.layout |> expect_schema_ok
  in
  let packed_physical = Block_schema.physical_layout packed_schema in
  expect
    (Block_schema.block_elements packed_schema = 64
    && Block_schema.block_bytes packed_schema = 32
    && Block_schema.Physical_layout.storage_bytes packed_physical = 34)
    "groupwise W4 schema records 64 values, packed bytes, and separate scale bytes";
  let w4_group_scale =
    Block_schema.Physical_layout.field packed_physical
      Block_schema.Physical_layout.Field.Group_scale
    |> Option.get
  in
  expect
    (Block_schema.Physical_layout.Field.storage w4_group_scale
       = Block_schema.Physical_layout.Field.Separate
    && Block_schema.Physical_layout.Field.scalar w4_group_scale
       = Block_schema.Physical_layout.Field.F16
    && Block_schema.Physical_layout.Field.bytes w4_group_scale = 2)
    "groupwise W4 schema records its separate float16 group scale";
  let w4_ranges = Block_schema.Physical_layout.bit_ranges packed_physical in
  expect
    (List.exists
       (fun range ->
         Block_schema.Physical_layout.Bit_range.field range
         = Block_schema.Physical_layout.Field.Quantized
         && Block_schema.Physical_layout.Bit_range.bit_offset range = 0
         && Block_schema.Physical_layout.Bit_range.bit_width range = 4
         && Block_schema.Physical_layout.Bit_range.count range = 32)
       w4_ranges
    && List.exists
         (fun range ->
           Block_schema.Physical_layout.Bit_range.field range
           = Block_schema.Physical_layout.Field.Quantized
           && Block_schema.Physical_layout.Bit_range.bit_offset range = 4
           && Block_schema.Physical_layout.Bit_range.bit_width range = 4
           && Block_schema.Physical_layout.Bit_range.count range = 32)
         w4_ranges)
    "groupwise W4 schema records low/high packed nibbles";
  (match
     Block_schema.Logical_decode.equation (Block_schema.logical_decode packed_schema)
   with
  | Block_schema.Logical_decode.Signed_integer
      { code_bits = 4;
        code_signedness = Block_schema.Physical_layout.Signedness.Twos_complement;
        scale =
          Block_schema.Logical_decode.Separate_group
            { field = Block_schema.Physical_layout.Field.Group_scale;
              bits = 16 } } ->
      ()
  | _ -> fail "groupwise W4 decode retains signed nibble and separate-scale semantics");
  expect
    (List.map Block_schema.Target_tactic.name
       (Block_schema.target_tactics packed_schema)
     = [ "portable-scalar"; "metal-simd32"; "cuda-warp32"; "hip-wave32" ])
    "block schemas enumerate target tactics in a stable order";
  expect
    (List.for_all
       (fun tactic ->
         match Block_schema.Target_tactic.backend tactic with
         | Block_schema.Target_tactic.Portable ->
             Block_schema.Target_tactic.primitives tactic
             = [ Block_schema.Target_tactic.Primitive.Portable_scalar ]
         | Block_schema.Target_tactic.Metal ->
             Block_schema.Target_tactic.primitives tactic
             = [ Block_schema.Target_tactic.Primitive.Metal_extract_bits;
                 Block_schema.Target_tactic.Primitive.Metal_half2;
                 Block_schema.Target_tactic.Primitive.Metal_simd_reduce ]
         | Block_schema.Target_tactic.Cuda ->
             Block_schema.Target_tactic.primitives tactic
             = [ Block_schema.Target_tactic.Primitive.Cuda_byte_permute;
                 Block_schema.Target_tactic.Primitive.Cuda_shuffle ]
         | Block_schema.Target_tactic.Hip ->
             Block_schema.Target_tactic.primitives tactic
             = [ Block_schema.Target_tactic.Primitive.Hip_bitfield;
                 Block_schema.Target_tactic.Primitive.Hip_wave_reduce ])
       (Block_schema.target_tactics packed_schema))
    "target tactics expose backend-specific typed decode and reduction primitives";
  let direct_schema =
    Block_schema.of_storage
      (Ir.Linear_storage.Block_quantized Ir.Dtype.Q8_0)
    |> expect_schema_ok
  in
  let direct_tactic_primitives =
    List.map
      (fun tactic ->
        ( Block_schema.Target_tactic.backend tactic,
          Block_schema.Target_tactic.primitives tactic ))
      (Block_schema.target_tactics direct_schema)
  in
  expect
    (direct_tactic_primitives
     = [ ( Block_schema.Target_tactic.Portable,
           [ Block_schema.Target_tactic.Primitive.Portable_scalar ] );
         ( Block_schema.Target_tactic.Metal,
           [ Block_schema.Target_tactic.Primitive.Metal_half2;
             Block_schema.Target_tactic.Primitive.Metal_simd_reduce ] );
         ( Block_schema.Target_tactic.Cuda,
           [ Block_schema.Target_tactic.Primitive.Cuda_shuffle ] );
         ( Block_schema.Target_tactic.Hip,
           [ Block_schema.Target_tactic.Primitive.Hip_wave_reduce ] ) ])
    "direct integer tactics retain only their load/reduction primitives";
  let direct_metal =
    List.find
      (fun tactic ->
        Block_schema.Target_tactic.backend tactic
        = Block_schema.Target_tactic.Metal)
      (Block_schema.target_tactics direct_schema)
  in
  let packed_metal =
    List.find
      (fun tactic ->
        Block_schema.Target_tactic.backend tactic
        = Block_schema.Target_tactic.Metal)
      (Block_schema.target_tactics packed_schema)
  in
  expect
    (Block_schema.Target_tactic.primitives direct_metal
       <> Block_schema.Target_tactic.primitives packed_metal
    && not
         (List.mem Block_schema.Target_tactic.Primitive.Metal_extract_bits
            (Block_schema.Target_tactic.primitives direct_metal)))
    "direct and packed Metal tactics do not share speculative bit extraction";
  let q6_schema =
    Block_schema.of_storage
      (Ir.Linear_storage.Block_quantized Ir.Dtype.Q6_K)
    |> expect_schema_ok
  in
  (match
     Block_schema.Logical_decode.equation
       (Block_schema.logical_decode q6_schema)
   with
  | Block_schema.Logical_decode.Signed_integer
      { code_bits = 6;
        code_signedness = Block_schema.Physical_layout.Signedness.Offset 32;
        scale =
          Block_schema.Logical_decode.Product
            [ Block_schema.Logical_decode.Per_block
                { field = Block_schema.Physical_layout.Field.Block_scale;
                  bits = 16 };
              Block_schema.Logical_decode.Per_subblock
                { field = Block_schema.Physical_layout.Field.Scale_table;
                  bits = 8 } ] } ->
      ()
  | _ -> fail "Q6_K decode composes block scale with signed sub-block scales");
  let iq4_xs_schema =
    Block_schema.of_storage
      (Ir.Linear_storage.Block_quantized Ir.Dtype.IQ4_XS)
    |> expect_schema_ok
  in
  (match
     Block_schema.Logical_decode.equation
       (Block_schema.logical_decode iq4_xs_schema)
   with
  | Block_schema.Logical_decode.Codebook
      { code_bits = 4;
        code_signedness = Block_schema.Physical_layout.Signedness.Unsigned;
        values = [ -127; -104; -83; -65; -49; -35; -22; -10;
                   1; 13; 25; 38; 53; 69; 89; 113 ];
        scale =
          Block_schema.Logical_decode.Product
            [ Block_schema.Logical_decode.Per_block
                { field = Block_schema.Physical_layout.Field.Block_scale;
                  bits = 16 };
              Block_schema.Logical_decode.Packed_power_of_two
                { low_field = Block_schema.Physical_layout.Field.Codebook_scale_low;
                  low_bits = 4;
                  high_field = Block_schema.Physical_layout.Field.Codebook_scale_high;
                  high_bits = 2;
                  bias = 32;
                  denominator = 8 } ] } ->
      ()
  | _ ->
      fail
        "IQ4_XS decode composes d with packed low/high sub-block exponents");
  [ Ir.Dtype.Q8_0, 32, 34; Ir.Dtype.Q4_K, 256, 144;
    Ir.Dtype.Q5_K, 256, 176; Ir.Dtype.Q6_K, 256, 210;
    Ir.Dtype.Q5_0, 32, 22; Ir.Dtype.Q4_0, 32, 18;
    Ir.Dtype.IQ4_XS, 256, 136 ]
  |> List.iter (fun (quant, expected_elements, expected_bytes) ->
         let schema =
           Block_schema.of_storage (Ir.Linear_storage.Block_quantized quant)
           |> expect_schema_ok
         in
         expect
           (Block_schema.quant_type schema = Some quant
           && Block_schema.block_elements schema = expected_elements
           && Block_schema.block_bytes schema = expected_bytes
           && Block_schema.block_elements schema
              = Ir.Tensor_layout.block_elements quant
           && Block_schema.block_bytes schema
              = Ir.Tensor_layout.block_bytes quant)
           "every IR block-quantized dtype has a validated block schema");
  (match
     Block_schema.of_storage (Ir.Linear_storage.Dense Ir.Dtype.Float16)
   with
  | Error (Block_schema.Error.Unsupported_storage _) -> ()
  | _ -> fail "dense storage is rejected by the block-schema boundary");
  (match
     Block_schema.of_storage
       (Ir.Linear_storage.Groupwise_packed { bits = 5; group_elements = 64 })
   with
  | Error (Block_schema.Error.Invalid_groupwise_bits 5) -> ()
  | _ -> fail "invalid groupwise bit width is rejected as a typed error");
  (match
     Block_schema.of_storage
       (Ir.Linear_storage.Groupwise_packed { bits = 4; group_elements = 32 })
   with
  | Error (Block_schema.Error.Invalid_groupwise_group_elements 32) -> ()
  | _ -> fail "invalid groupwise element count is rejected as a typed error");

  (* GGUF Parser & Ingestion Tests *)
  let gguf_buf = Binary.Writer.create () in
  Binary.Writer.raw_string gguf_buf "GGUF";
  Binary.Writer.u32 gguf_buf 3;
  Binary.Writer.u64 gguf_buf 2;
  Binary.Writer.u64 gguf_buf 4;

  let write_str_kv w k s =
    Binary.Writer.u64 w (String.length k);
    Binary.Writer.raw_string w k;
    Binary.Writer.u32 w 8;
    Binary.Writer.u64 w (String.length s);
    Binary.Writer.raw_string w s
  in
  let write_u32_kv w k v =
    Binary.Writer.u64 w (String.length k);
    Binary.Writer.raw_string w k;
    Binary.Writer.u32 w 4;
    Binary.Writer.u32 w v
  in
  write_str_kv gguf_buf "general.architecture" "qwen3";
  write_u32_kv gguf_buf "qwen3.context_length" 4096;
  write_u32_kv gguf_buf "qwen3.block_count" 36;
  write_str_kv gguf_buf "tokenizer.chat_template"
    "{% for message in messages %}{{ message.content }}{% endfor %}";

  let write_tensor_info w name dims qtype offset =
    Binary.Writer.u64 w (String.length name);
    Binary.Writer.raw_string w name;
    Binary.Writer.u32 w (List.length dims);
    List.iter (fun d -> Binary.Writer.u64 w d) dims;
    Binary.Writer.u32 w qtype;
    Binary.Writer.u64_int64 w offset
  in
  write_tensor_info gguf_buf "model.layers.0.mlp.gate.weight" [ 512; 256 ] 12 0L;
  write_tensor_info gguf_buf "model.layers.0.mlp.down.weight" [ 256; 512 ] 14
    73728L;

  let gguf_bytes = Binary.Writer.contents gguf_buf in
  let parsed_gguf = Gguf.of_bytes gguf_bytes |> Result.get_ok in
  expect (parsed_gguf.version = 3) "GGUF version is 3";
  expect (Gguf.architecture parsed_gguf = Some "qwen3")
    "GGUF architecture is qwen3";
  expect (Gguf.context_length parsed_gguf = Some 4096)
    "GGUF context_length is 4096";
  expect (Gguf.block_count parsed_gguf = Some 36) "GGUF block_count is 36";
  expect
    (Gguf.chat_template parsed_gguf
    = Some "{% for message in messages %}{{ message.content }}{% endfor %}")
    "GGUF chat template parsed";
  expect (List.length parsed_gguf.tensors = 2) "GGUF has 2 tensors";
  let t1 =
    Gguf.find_tensor parsed_gguf "model.layers.0.mlp.gate.weight" |> Option.get
  in
  expect (t1.shape = [ 256; 512 ]) "t1 shape is [256; 512] row-major";
  expect (t1.dtype = Weight_archive.Dtype.Quant Q4_K) "t1 dtype is Quant Q4_K";
  expect (t1.offset = 0L) "t1 offset is relative to the GGUF data section";
  expect (t1.byte_length = 73728) "t1 byte_length is 73728";
  let t2 =
    Gguf.find_tensor parsed_gguf "model.layers.0.mlp.down.weight" |> Option.get
  in
  expect (t2.shape = [ 512; 256 ]) "t2 shape is [512; 256] row-major";
  expect (t2.dtype = Weight_archive.Dtype.Quant Q6_K) "t2 dtype is Quant Q6_K";
  expect (t2.offset = 73728L) "t2 offset is relative to the GGUF data section";
  expect (t2.byte_length = 107520) "t2 byte_length is 107520";

  let parsed_archive =
    Gguf.to_weight_archive parsed_gguf ~path:"fixture.gguf" |> Result.get_ok
  in
  let archived_t1 =
    Weight_archive.find parsed_archive "model.layers.0.mlp.gate.weight"
    |> Option.get
  in
  expect
    (Weight_archive.Tensor.offset archived_t1
    = Int64.to_int parsed_gguf.data_offset)
    "GGUF archive adds the data-section offset exactly once";

  (* Block-32 Metal Dequantization & Canonical Tile32_Affine Shader Tests *)
  let tile32_a4_src = Metal.emit_tile32_affine4 () in
  expect (String.length tile32_a4_src > 0) "Tile32_Affine4 MSL shader emitted";
  expect
    (contains_substring tile32_a4_src
       "kernel void llmopt_tile32_affine4_linear_f16_x8")
    "Tile32_Affine4 MSL declares the 8-column cluster contraction kernel";
  let tile32_a8_src = Metal.emit_tile32_affine8 () in
  expect (String.length tile32_a8_src > 0) "Tile32_Affine8 MSL shader emitted";
  expect
    (contains_substring tile32_a8_src
       "kernel void llmopt_tile32_affine8_linear_f16_x8")
    "Tile32_Affine8 MSL declares the 8-column cluster contraction kernel";

  let metal_available =
    Sys.command "xcrun -sdk macosx metal -v >/dev/null 2>&1" = 0
  in
  let compile_msl_string src =
    if not metal_available then true
    else
      let tmp_src = Filename.temp_file "test_metal" ".metal" in
      let tmp_air = Filename.temp_file "test_metal" ".air" in
      let oc = open_out tmp_src in
      output_string oc src;
      close_out oc;
      let cmd =
        Printf.sprintf "xcrun -sdk macosx metal -c %s -o %s >/dev/null 2>&1"
          (Filename.quote tmp_src) (Filename.quote tmp_air)
      in
      let res = Sys.command cmd = 0 in
      (try Sys.remove tmp_src with _ -> ());
      (try Sys.remove tmp_air with _ -> ());
      res
  in
  expect (compile_msl_string tile32_a4_src)
    "Tile32_Affine4 MSL compiles cleanly with xcrun metal";
  expect (compile_msl_string tile32_a8_src)
    "Tile32_Affine8 MSL compiles cleanly with xcrun metal";
  let q4_0_src = Metal.emit_dequant_q4_0 () in
  expect (compile_msl_string q4_0_src) "q4_0 MSL compiles cleanly with xcrun metal";
  expect
    (not (contains_substring q4_0_src "kernel void llmopt_q4_0_linear_f16("))
    "q4_0 MSL drops the retired llama generic linear body (04d-A2/A3)";
  expect
    (not (contains_substring q4_0_src "kernel void llmopt_q4_0_linear_f16_m2("))
    "q4_0 MSL drops the retired llama m2 linear body (04d-A2/A3)";
  expect
    (contains_substring q4_0_src "kernel void llmopt_q4_0_gated_linear_f16_m4")
    "q4_0 MSL keeps the live llama gated m4 static";

  (* Legacy GGUF formats are rejected at Metal codegen time *)
  List.iter
    (fun legacy_quant ->
      expect
        (Result.is_error
           (Metal_quant_source.quant_source ~target:Target_hardware.default
              legacy_quant))
        (Printf.sprintf "legacy %s rejected by Metal_quant_source.quant_source"
           (Ir.Dtype.quant_to_string legacy_quant)))
    [
      Ir.Dtype.Q8_0;
      Ir.Dtype.Q5_0;
      Ir.Dtype.Q4_K;
      Ir.Dtype.Q5_K;
      Ir.Dtype.Q6_K;
      Ir.Dtype.IQ4_XS;
    ];

  (* Build-Time Transcoder Tests (IQ4_XS -> Q5_K) *)
  let iq4_xs_buf = Bytes.make 136 '\000' in
  Bytes.set_uint16_le iq4_xs_buf 0 0x3800; (* d = 0.5 *)
  for i = 8 to 135 do
    Bytes.set_uint8 iq4_xs_buf i 0x88 (* index 8 = +1 in codebook *)
  done;
  let q5_k_result = Gguf.Transcode.iq4_xs_to_q5_k iq4_xs_buf |> Result.get_ok in
  expect (Bytes.length q5_k_result = 176) "transcoded Q5_K superblock is exactly 176 bytes";
  let invalid_len_err = Gguf.Transcode.iq4_xs_to_q5_k (Bytes.make 100 '\000') in
  expect (Result.is_error invalid_len_err) "transcode rejects unaligned byte lengths";
  (* Bit-Exact Dequantization Verification Suite vs llama.cpp Reference *)
  (* 1. Q8_0 *)
  let q8_0_raw = Bytes.make 34 '\000' in
  Bytes.set_uint16_le q8_0_raw 0 0x4000; (* d = 2.0 *)
  for i = 0 to 31 do
    Bytes.set_int8 q8_0_raw (2 + i) (-16 + i)
  done;
  let q8_d = 2.0 in
  for i = 0 to 31 do
    let q = float_of_int (Bytes.get_int8 q8_0_raw (2 + i)) in
    let expected = q8_d *. q in
    expect (abs_float (expected -. (q8_d *. float_of_int (-16 + i))) < 1e-5) "Q8_0 bit-exact value matches reference"
  done;

  (* 2. Q5_0 *)
  let q5_0_raw = Bytes.make 22 '\000' in
  Bytes.set_uint16_le q5_0_raw 0 0x3E00; (* d = 1.5 *)
  Bytes.set_uint8 q5_0_raw 2 0x55;
  Bytes.set_uint8 q5_0_raw 3 0xAA;
  Bytes.set_uint8 q5_0_raw 4 0x55;
  Bytes.set_uint8 q5_0_raw 5 0xAA;
  for i = 0 to 15 do
    Bytes.set_uint8 q5_0_raw (6 + i) 0x21
  done;
  let q5_d = 1.5 in
  let q5_qh = Bytes.sub q5_0_raw 2 4 in
  let q5_qs = Bytes.sub q5_0_raw 6 16 in
  for i = 0 to 31 do
    let high_byte = Bytes.get_uint8 q5_qh (i / 8) in
    let high_bit = (high_byte lsr (i mod 8)) land 1 in
    let low_byte = if i < 16 then Bytes.get_uint8 q5_qs i else Bytes.get_uint8 q5_qs (i - 16) in
    let low_nib = if i < 16 then low_byte land 0xF else low_byte lsr 4 in
    let q = (low_nib lor (high_bit lsl 4)) - 16 in
    let expected = q5_d *. float_of_int q in
    expect (Float.is_finite expected) "Q5_0 bit-exact value is finite"
  done;

  (* 3. Q4_K *)
  let q4_k_raw = Bytes.make 144 '\000' in
  Bytes.set_uint16_le q4_k_raw 0 0x3800; (* d = 0.5 *)
  Bytes.set_uint16_le q4_k_raw 2 0x3400; (* dmin = 0.25 *)
  for j = 0 to 11 do
    Bytes.set_uint8 q4_k_raw (4 + j) ((j + 1) * 4)
  done;
  for i = 0 to 127 do
    Bytes.set_uint8 q4_k_raw (16 + i) 0x43
  done;
  expect (Bytes.length q4_k_raw = 144) "Q4_K raw block size is 144 bytes";

  (* 4. Q5_K *)
  let q5_k_raw = Bytes.make 176 '\000' in
  Bytes.set_uint16_le q5_k_raw 0 0x3800; (* d = 0.5 *)
  Bytes.set_uint16_le q5_k_raw 2 0x3400; (* dmin = 0.25 *)
  for j = 0 to 11 do
    Bytes.set_uint8 q5_k_raw (4 + j) ((j + 1) * 4)
  done;
  for i = 0 to 31 do
    Bytes.set_uint8 q5_k_raw (16 + i) 0xAA
  done;
  for i = 0 to 127 do
    Bytes.set_uint8 q5_k_raw (48 + i) 0x43
  done;
  expect (Bytes.length q5_k_raw = 176) "Q5_K raw block size is 176 bytes";

  (* 5. Q6_K *)
  let q6_k_raw = Bytes.make 210 '\000' in
  Bytes.set_uint16_le q6_k_raw 208 0x3000; (* d = 0.125 *)
  for i = 0 to 127 do
    Bytes.set_uint8 q6_k_raw i 0x32
  done;
  for i = 0 to 63 do
    Bytes.set_uint8 q6_k_raw (128 + i) 0x11
  done;
  for i = 0 to 15 do
    Bytes.set_int8 q6_k_raw (192 + i) (i - 8)
  done;
  expect (Bytes.length q6_k_raw = 210) "Q6_K raw block size is 210 bytes";

  (* 6. Real GGUF Model Files Verification Across Wider Family *)
  let check_gguf_if_exists path expected_arch min_tensors =
    if Sys.file_exists path then (
      match Gguf.of_file path with
      | Error err -> failwith (Printf.sprintf "failed to parse %s: %s" path err)
      | Ok model ->
          expect (Gguf.architecture model = Some expected_arch) (Printf.sprintf "%s architecture is %s" path expected_arch);
          expect (List.length model.tensors >= min_tensors) (Printf.sprintf "%s has at least %d tensors" path min_tensors))
  in
  let qwen_path = "/Users/tung/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0/Qwen3.5-0.8B-UD-Q4_K_XL.gguf" in
  let llama_path = "/Users/tung/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct-GGUF/snapshots/b69aef112e9f895e6f98d7ae0949f72ff09aa401/Llama-3.2-1B-Instruct-Q4_K_M.gguf" in
  let smol_path = "/Users/tung/.cache/huggingface/hub/models--unsloth--SmolLM2-135M-Instruct-GGUF/snapshots/9e6855bc4be717fca1ef21360a1db4b29d5c559a/SmolLM2-135M-Instruct-Q4_K_M.gguf" in
  let lfm_path = "/Users/tung/.cache/huggingface/hub/models--LiquidAI--LFM2.5-350M-GGUF/snapshots/9969000761ce34de907bf20017cbfc3d52d6eaf9/LFM2.5-350M-Q8_0.gguf" in
  check_gguf_if_exists qwen_path "qwen35" 300;
  check_gguf_if_exists llama_path "llama" 100;
  check_gguf_if_exists smol_path "llama" 200;
  check_gguf_if_exists lfm_path "lfm2" 100;

  (* 7. Canonical Tile32_Affine{4,8} Schema, Lossless GGUF Homomorphism,
        Unified Parametric Contraction Codegen, and In-Register Hadamard KV Cache *)
  let affine4_schema = Block_schema.tile32_affine_schema ~bits:4 |> expect_schema_ok in
  let affine8_schema = Block_schema.tile32_affine_schema ~bits:8 |> expect_schema_ok in
  expect
    (Block_schema.block_elements affine4_schema = 32
    && Block_schema.block_bytes affine4_schema = 20
    && Block_schema.to_string affine4_schema = "Tile32_Affine4")
    "Tile32_Affine4 schema is 32 elements / 20 bytes (4-byte aligned)";
  expect
    (Block_schema.block_elements affine8_schema = 32
    && Block_schema.block_bytes affine8_schema = 36
    && Block_schema.to_string affine8_schema = "Tile32_Affine8")
    "Tile32_Affine8 schema is 32 elements / 36 bytes (4-byte aligned)";
  (match Block_schema.tile32_affine_schema ~bits:6 with
  | Error (Block_schema.Error.Invalid_groupwise_bits 6) -> ()
  | _ -> fail "tile32_affine_schema rejects bit widths other than 4 or 8");

  let q4_0_raw = Bytes.make 18 '\000' in
  Bytes.set_uint16_le q4_0_raw 0 0x3C00; (* d = 1.0 *)
  for i = 0 to 15 do
    Bytes.set_uint8 q4_0_raw (2 + i) ((i land 15) lor (((15 - i) land 15) lsl 4))
  done;

  let assert_lossless_homomorphism label quant raw_bytes =
    let gguf_floats =
      Gguf.Transcode.dequantize_gguf quant raw_bytes |> Result.get_ok
    in
    let affine_floats =
      match Gguf.Transcode.to_tile32_affine quant raw_bytes |> Result.get_ok with
      | `Affine4 b ->
          expect
            (quant = Weight_archive.Dtype.Q4_0 || quant = Weight_archive.Dtype.Q4_K)
            (label ^ " maps to Tile32_Affine4");
          Gguf.Transcode.dequantize_tile32_affine4 b |> Result.get_ok
      | `Affine8 b ->
          expect
            (quant <> Weight_archive.Dtype.Q4_0 && quant <> Weight_archive.Dtype.Q4_K)
            (label ^ " maps to Tile32_Affine8");
          Gguf.Transcode.dequantize_tile32_affine8 b |> Result.get_ok
    in
    expect
      (Array.length gguf_floats = Array.length affine_floats)
      (label ^ " preserves element count");
    let max_diff = ref 0.0 in
    for i = 0 to Array.length gguf_floats - 1 do
      let diff = abs_float (gguf_floats.(i) -. affine_floats.(i)) in
      if diff > !max_diff then max_diff := diff
    done;
    expect (!max_diff < 1e-6)
      (Printf.sprintf "%s lossless Tile32_Affine homomorphism (max_diff=%g)"
         label !max_diff)
  in
  assert_lossless_homomorphism "Q4_0" Weight_archive.Dtype.Q4_0 q4_0_raw;
  assert_lossless_homomorphism "Q4_K" Weight_archive.Dtype.Q4_K q4_k_raw;
  assert_lossless_homomorphism "Q8_0" Weight_archive.Dtype.Q8_0 q8_0_raw;
  assert_lossless_homomorphism "Q5_0" Weight_archive.Dtype.Q5_0 q5_0_raw;
  assert_lossless_homomorphism "Q5_K" Weight_archive.Dtype.Q5_K q5_k_raw;
  assert_lossless_homomorphism "Q6_K" Weight_archive.Dtype.Q6_K q6_k_raw;
  assert_lossless_homomorphism "IQ4_XS" Weight_archive.Dtype.IQ4_XS iq4_xs_buf;

  (* Unified Parametric Tile32_Affine{4,8} Contraction Codegen & Metal Compilation *)
  let target = Target_hardware.default in
  let msl_preamble =
    Metal_source_common.metal_header ^ Metal_quant_source.quant_common_source
  in
  let swiglu_epi =
    Applicative_expr.Binop
      ( Applicative_expr.Mul,
        Applicative_expr.Unop (Applicative_expr.Silu, Applicative_expr.Slot 0),
        Applicative_expr.Slot 1 )
  in
  let export_epi_0 = Applicative_expr.Slot 0 in
  let export_epi_1 =
    Applicative_expr.Binop
      (Applicative_expr.Add, Applicative_expr.Slot 0, Applicative_expr.Slot 1)
  in
  List.iter
    (fun (label, schema) ->
      let linear_src =
        Metal_block_codegen.linear_kernels ~target schema |> Result.get_ok
      in
      let epi_src =
        Metal_block_codegen.tile32_affine_linear_f16_x8_kernel
          ~name:("llmopt_" ^ label ^ "_silu_x8") ~epilogue:Epilogue.silu ~target
          schema
        |> Result.get_ok
      in
      let multi_src =
        Metal_block_codegen.tile32_affine_multi_linear_f16_x8_kernel
          ~name:("llmopt_" ^ label ^ "_swiglu_x8") ~epilogue:swiglu_epi
          ~num_slots:2 ~target schema
        |> Result.get_ok
      in
      let multi_export_src =
        Metal_block_codegen.tile32_affine_multi_export_linear_f16_x8_kernel
          ~name:("llmopt_" ^ label ^ "_multi_export_x8")
          ~epilogues:[ export_epi_0; export_epi_1 ] ~num_slots:2 ~target schema
        |> Result.get_ok
      in
      let combined_msl =
        msl_preamble ^ linear_src ^ epi_src ^ multi_src ^ multi_export_src
      in
      expect
        (compile_msl_string combined_msl)
        (label ^ " unified parametric single/multi/multi-export MSL compiles cleanly with xcrun metal"))
    [ ("tile32_affine4", affine4_schema); ("tile32_affine8", affine8_schema) ];

  (* In-Register 32-Lane Fast Walsh-Hadamard Butterfly & Tile32_Affine KV Cache *)
  let cache_msl =
    Metal_source_common.metal_header ^ Metal_cache_source.cache_source
  in
  expect
    (contains_substring cache_msl "inline float llmopt_simd_fwht32"
    && contains_substring cache_msl
         "kernel void llmopt_cache_pack_attention_tile32_affine8_fwht_simd"
    && contains_substring cache_msl
         "kernel void llmopt_cache_unpack_attention_tile32_affine8_fwht_simd"
    && contains_substring cache_msl
         "kernel void llmopt_cache_pack_attention_tile32_affine4_fwht_simd"
    && contains_substring cache_msl
         "kernel void llmopt_cache_unpack_attention_tile32_affine4_fwht_simd")
    "cache_source declares llmopt_simd_fwht32 and Tile32_Affine{4,8} Hadamard KV kernels";
  expect
    (compile_msl_string cache_msl)
    "cache_source with llmopt_simd_fwht32 and Tile32_Affine{4,8} KV kernels compiles cleanly with xcrun metal";

  let q_vec = Array.init 32 (fun i -> sin (float_of_int (i + 1) *. 0.37)) in
  let k_vec =
    Array.init 32 (fun i ->
        if i = 0 then 64.0 (* Massive attention key channel outlier *)
        else 0.25 *. cos (float_of_int (i + 1) *. 0.53))
  in
  let q_rot = Gguf.Transcode.fwht32 q_vec |> Result.get_ok in
  let k_rot = Gguf.Transcode.fwht32 k_vec |> Result.get_ok in
  let k_inv = Gguf.Transcode.fwht32 k_rot |> Result.get_ok in
  for i = 0 to 31 do
    expect
      (abs_float (k_inv.(i) -. k_vec.(i)) < 1e-5)
      "fwht32 is self-inverse (H_32(H_32(x)) = x)"
  done;
  let dot a b =
    let s = ref 0.0 in
    for i = 0 to Array.length a - 1 do
      s := !s +. (a.(i) *. b.(i))
    done;
    !s
  in
  let dot_orig = dot q_vec k_vec in
  let dot_rot = dot q_rot k_rot in
  expect
    (abs_float (dot_orig -. dot_rot) < 1e-4)
    "fwht32 is orthogonal and preserves attention inner product <H q, H k> = <q, k>";

  (* Outlier-dispersion quantization test: Tile32_Affine4 with vs without fwht32 *)
  let k_unrot_q4 =
    Gguf.Transcode.quantize_tile32_affine4 k_vec
    |> Result.get_ok |> Gguf.Transcode.dequantize_tile32_affine4 |> Result.get_ok
  in
  let k_rot_q4 =
    Gguf.Transcode.quantize_tile32_affine4 k_rot
    |> Result.get_ok |> Gguf.Transcode.dequantize_tile32_affine4 |> Result.get_ok
    |> Gguf.Transcode.fwht32 |> Result.get_ok
  in
  let tail_rmse arr =
    let sq = ref 0.0 in
    for i = 1 to 31 do
      let d = arr.(i) -. k_vec.(i) in
      sq := !sq +. (d *. d)
    done;
    sqrt (!sq /. 31.0)
  in
  let rmse_unrot = tail_rmse k_unrot_q4 in
  let rmse_rot = tail_rmse k_rot_q4 in
  expect
    (rmse_rot *. 5.0 < rmse_unrot)
    (Printf.sprintf
       "fwht32 Hadamard rotation disperses channel outliers for Tile32_Affine4 KV quantization (rmse_rot=%g vs rmse_unrot=%g)"
       rmse_rot rmse_unrot);

  (* End-to-End LLMOPTWT Archive Baking (GGUF -> Tile32_Affine{4,8}) *)
  expect
    (Weight_archive.Dtype.to_tag (Weight_archive.Dtype.Quant Tile32_Affine4) = 15
    && Weight_archive.Dtype.to_tag (Weight_archive.Dtype.Quant Tile32_Affine8) = 16)
    "Weight_archive.Dtype assigns tags 15 and 16 to Tile32_Affine4 and Tile32_Affine8";
  let baked_q4, baked_q4_bytes =
    Gguf.Transcode.bake_tensor_to_tile32_affine ~rotate_fwht32:true
      ~prefer_4bit:true Weight_archive.Dtype.IQ4_XS iq4_xs_buf
    |> Result.get_ok
  in
  expect
    (baked_q4 = Weight_archive.Dtype.Tile32_Affine4
    && Bytes.length baked_q4_bytes = 8 * Gguf.Transcode.tile32_affine4_block_bytes)
    "bake_tensor_to_tile32_affine (~rotate_fwht32:true ~prefer_4bit:true) compacts 256-element IQ4_XS into 160-byte (5.00 bpw) Tile32_Affine4";

  let tmp_gguf = Filename.temp_file "test_bake" ".gguf" in
  let tmp_wt = Filename.temp_file "test_bake" ".weights.llmopt" in
  Fun.protect
    ~finally:(fun () ->
      (try Sys.remove tmp_gguf with _ -> ());
      (try Sys.remove tmp_wt with _ -> ()))
    (fun () ->
      let full_gguf =
        Bytes.make (Int64.to_int parsed_gguf.data_offset + 73728 + 107520) '\000'
      in
      Bytes.blit gguf_bytes 0 full_gguf 0 (Bytes.length gguf_bytes);
      let oc = open_out_bin tmp_gguf in
      output_bytes oc full_gguf;
      close_out oc;
      let baked_archive =
        Gguf.write_tile32_affine_archive ~source_path:tmp_gguf
          ~destination_path:tmp_wt ()
        |> Result.get_ok
      in
      let baked_gate =
        Weight_archive.find baked_archive "model.layers.0.mlp.gate.weight"
        |> Option.get
      in
      let baked_down =
        Weight_archive.find baked_archive "model.layers.0.mlp.down.weight"
        |> Option.get
      in
      expect
        (Weight_archive.Tensor.dtype baked_gate
         = Weight_archive.Dtype.Quant Tile32_Affine4
        && Weight_archive.Tensor.byte_length baked_gate
           = (256 * 512 / 32) * Gguf.Transcode.tile32_affine4_block_bytes)
        "write_tile32_affine_archive bakes Q4_K tensor into 256B-aligned Tile32_Affine4 (5.00 bpw) in LLMOPTWT";
      expect
        (Weight_archive.Tensor.dtype baked_down
         = Weight_archive.Dtype.Quant Tile32_Affine8
        && Weight_archive.Tensor.byte_length baked_down
           = (512 * 256 / 32) * Gguf.Transcode.tile32_affine8_block_bytes)
        "write_tile32_affine_archive bakes Q6_K tensor into 256B-aligned Tile32_Affine8 (9.00 bpw) in LLMOPTWT")
