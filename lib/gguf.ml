let ( let* ) = Result.bind

let magic = "GGUF"
let default_alignment = 32

module Value = struct
  type t =
    | Uint8 of int
    | Int8 of int
    | Uint16 of int
    | Int16 of int
    | Uint32 of int32
    | Int32 of int32
    | Float32 of float
    | Bool of bool
    | String of string
    | Array of t list
    | Uint64 of int64
    | Int64 of int64
    | Float64 of float

  let rec to_string = function
    | Uint8 v -> string_of_int v
    | Int8 v -> string_of_int v
    | Uint16 v -> string_of_int v
    | Int16 v -> string_of_int v
    | Uint32 v -> Int32.to_string v
    | Int32 v -> Int32.to_string v
    | Float32 v -> string_of_float v
    | Bool v -> string_of_bool v
    | String v -> Printf.sprintf "%S" v
    | Array vs ->
        "[" ^ String.concat ", " (List.map to_string vs) ^ "]"
    | Uint64 v -> Int64.to_string v
    | Int64 v -> Int64.to_string v
    | Float64 v -> string_of_float v

  let to_int_opt = function
    | Uint8 v | Int8 v | Uint16 v | Int16 v -> Some v
    | Uint32 v | Int32 v -> Some (Int32.to_int v)
    | Uint64 v | Int64 v ->
        if v >= Int64.of_int min_int && v <= Int64.of_int max_int then
          Some (Int64.to_int v)
        else None
    | _ -> None

  let to_string_opt = function
    | String s -> Some s
    | _ -> None

  let to_float_opt = function
    | Float32 f | Float64 f -> Some f
    | _ -> None
end

module Tensor_info = struct
  type t = {
    name : string;
    shape : int list;
    ggml_type : int;
    dtype : Weight_archive.Dtype.t;
    offset : int64;
    byte_length : int;
  }

  let name t = t.name
  let shape t = t.shape
  let ggml_type t = t.ggml_type
  let dtype t = t.dtype
  let offset t = t.offset
  let byte_length t = t.byte_length
end

type t = {
  version : int;
  alignment : int;
  metadata : (string * Value.t) list;
  tensors : Tensor_info.t list;
  data_offset : int64;
  file_size : int64;
}

let dtype_of_ggml_type = function
  | 0 -> Ok Weight_archive.Dtype.F32
  | 1 -> Ok Weight_archive.Dtype.F16
  | 2 -> Ok (Weight_archive.Dtype.Quant Q4_0)
  | 6 -> Ok (Weight_archive.Dtype.Quant Q5_0)
  | 8 -> Ok (Weight_archive.Dtype.Quant Q8_0)
  | 12 -> Ok (Weight_archive.Dtype.Quant Q4_K)
  | 13 -> Ok (Weight_archive.Dtype.Quant Q5_K)
  | 14 -> Ok (Weight_archive.Dtype.Quant Q6_K)
  | 23 -> Ok (Weight_archive.Dtype.Quant IQ4_XS)
  | 28 | 30 -> Ok Weight_archive.Dtype.BF16
  | tag -> Error (Printf.sprintf "unsupported GGML quant type tag: %d" tag)

let checked_product dimensions =
  let rec loop acc = function
    | [] -> Ok acc
    | d :: rest ->
        if d <= 0 then Error "tensor dimension must be positive"
        else if acc > max_int / d then
          Error "tensor dimension product overflows host integer"
        else loop (acc * d) rest
  in
  loop 1 dimensions

let calculate_byte_length dtype elements =
  match dtype with
  | Weight_archive.Dtype.F32 | Weight_archive.Dtype.I32 ->
      if elements > max_int / 4 then Error "tensor byte length overflows"
      else Ok (elements * 4)
  | Weight_archive.Dtype.F16 | Weight_archive.Dtype.BF16 ->
      if elements > max_int / 2 then Error "tensor byte length overflows"
      else Ok (elements * 2)
  | Weight_archive.Dtype.I64 ->
      if elements > max_int / 8 then Error "tensor byte length overflows"
      else Ok (elements * 8)
  | Weight_archive.Dtype.I8 | Weight_archive.Dtype.Bool | Weight_archive.Dtype.U8 ->
      Ok elements
  | Weight_archive.Dtype.Quant q ->
      let blk = Weight_archive.Dtype.block_size q in
      let bpb = Weight_archive.Dtype.bytes_per_block q in
      if elements mod blk <> 0 then
        Error
          (Printf.sprintf
             "element count %d is not a multiple of quant block size %d"
             elements blk)
      else if elements / blk > max_int / bpb then
        Error "tensor byte length overflows"
      else Ok ((elements / blk) * bpb)

let read_string reader =
  let* length = Binary.Reader.u64 reader in
  Binary.Reader.raw_string reader ~length

let rec read_value reader value_type =
  match value_type with
  | 0 -> Binary.Reader.u8 reader |> Result.map (fun v -> Value.Uint8 v)
  | 1 -> Binary.Reader.i8 reader |> Result.map (fun v -> Value.Int8 v)
  | 2 -> Binary.Reader.u16 reader |> Result.map (fun v -> Value.Uint16 v)
  | 3 -> Binary.Reader.i16 reader |> Result.map (fun v -> Value.Int16 v)
  | 4 -> Binary.Reader.i32_raw reader |> Result.map (fun v -> Value.Uint32 v)
  | 5 -> Binary.Reader.i32_raw reader |> Result.map (fun v -> Value.Int32 v)
  | 6 -> Binary.Reader.float32 reader |> Result.map (fun v -> Value.Float32 v)
  | 7 -> Binary.Reader.bool reader |> Result.map (fun v -> Value.Bool v)
  | 8 -> read_string reader |> Result.map (fun v -> Value.String v)
  | 9 ->
      let* item_type = Binary.Reader.u32 reader in
      let* length = Binary.Reader.u64 reader in
      let rec read_items acc remaining =
        if remaining = 0 then Ok (Value.Array (List.rev acc))
        else
          let* item = read_value reader item_type in
          read_items (item :: acc) (remaining - 1)
      in
      read_items [] length
  | 10 -> Binary.Reader.u64_int64 reader |> Result.map (fun v -> Value.Uint64 v)
  | 11 -> Binary.Reader.i64_int64 reader |> Result.map (fun v -> Value.Int64 v)
  | 12 -> Binary.Reader.float64 reader |> Result.map (fun v -> Value.Float64 v)
  | tag -> Error (Printf.sprintf "unknown GGUF metadata value type tag: %d" tag)

let parse_metadata_kv reader =
  let* key = read_string reader in
  let* value_type = Binary.Reader.u32 reader in
  let* value = read_value reader value_type in
  Ok (key, value)

let parse_tensor_info reader =
  let* name = read_string reader in
  let* n_dims = Binary.Reader.u32 reader in
  let rec read_dims acc remaining =
    if remaining = 0 then Ok (List.rev acc)
    else
      let* d = Binary.Reader.u64 reader in
      read_dims (d :: acc) (remaining - 1)
  in
  let* dims = read_dims [] n_dims in
  (* Convert from column-major (GGML) to canonical row-major shape *)
  let shape = List.rev dims in
  let* ggml_type = Binary.Reader.u32 reader in
  let* dtype = dtype_of_ggml_type ggml_type in
  let* offset = Binary.Reader.u64_int64 reader in
  let* elements = checked_product shape in
  let* byte_length = calculate_byte_length dtype elements in
  Ok { Tensor_info.name; shape; ggml_type; dtype; offset; byte_length }

let align_offset (offset : int64) (alignment : int) : int64 =
  let align_i64 = Int64.of_int alignment in
  let remainder = Int64.rem offset align_i64 in
  if remainder = 0L then offset
  else Int64.add offset (Int64.sub align_i64 remainder)

let of_bytes bytes =
  let file_size = Int64.of_int (Bytes.length bytes) in
  let reader = Binary.Reader.create bytes in
  let* actual_magic = Binary.Reader.raw_string reader ~length:4 in
  if actual_magic <> magic then
    Error
      (Printf.sprintf "invalid GGUF magic: expected %S, got %S" magic
         actual_magic)
  else
    let* version = Binary.Reader.u32 reader in
    if version < 2 || version > 3 then
      Error (Printf.sprintf "unsupported GGUF version: %d (expected 2 or 3)" version)
    else
      let* tensor_count = Binary.Reader.u64 reader in
      let* metadata_kv_count = Binary.Reader.u64 reader in
      let rec read_kvs acc remaining =
        if remaining = 0 then Ok (List.rev acc)
        else
          let* kv = parse_metadata_kv reader in
          read_kvs (kv :: acc) (remaining - 1)
      in
      let* metadata = read_kvs [] metadata_kv_count in
      let alignment =
        match List.assoc_opt "general.alignment" metadata with
        | Some (Value.Uint32 a) -> Int32.to_int a
        | Some (Value.Uint64 a) -> Int64.to_int a
        | _ -> default_alignment
      in
      let rec read_tensors acc remaining =
        if remaining = 0 then Ok (List.rev acc)
        else
          let* info = parse_tensor_info reader in
          read_tensors (info :: acc) (remaining - 1)
      in
      let* raw_tensors = read_tensors [] tensor_count in
      let current_offset = Int64.of_int (Binary.Reader.offset reader) in
      let data_offset = align_offset current_offset alignment in
      Ok
        {
          version;
          alignment;
          metadata;
          tensors = raw_tensors;
          data_offset;
          file_size;
        }

let of_file path =
  try
    let stat = Unix.LargeFile.stat path in
    if stat.st_kind <> Unix.S_REG then
      Error ("path is not a regular file: " ^ path)
    else
      let file_size = stat.st_size in
      let fd = Unix.openfile path [ Unix.O_RDONLY ] 0 in
      Fun.protect
        ~finally:(fun () -> Unix.close fd)
        (fun () ->
          (* Read the header + metadata (first 64MB or entire file) *)
          let initial_read = Int64.to_int (Int64.min file_size 67_108_864L) in
          let buffer = Bytes.create initial_read in
          let rec really_read offset remaining =
            if remaining = 0 then offset
            else
              let n = Unix.read fd buffer offset remaining in
              if n = 0 then offset
              else really_read (offset + n) (remaining - n)
          in
          let read_bytes = really_read 0 initial_read in
          let slice =
            if read_bytes = initial_read then buffer
            else Bytes.sub buffer 0 read_bytes
          in
          let* gguf = of_bytes slice in
          Ok { gguf with file_size })
  with
  | Sys_error msg -> Error ("cannot read GGUF file: " ^ msg)
  | Unix.Unix_error (err, _, _) ->
      Error ("cannot access GGUF file: " ^ Unix.error_message err)

let find_metadata t key = List.assoc_opt key t.metadata

let find_string t key =
  match find_metadata t key with
  | Some (Value.String s) -> Some s
  | _ -> None

let find_int t key =
  match find_metadata t key with
  | Some v -> Value.to_int_opt v
  | _ -> None

let find_float t key =
  match find_metadata t key with
  | Some v -> Value.to_float_opt v
  | _ -> None

let find_tensor t name =
  List.find_opt (fun (info : Tensor_info.t) -> info.name = name) t.tensors

let architecture t = find_string t "general.architecture"

let model_param t suffix =
  match architecture t with
  | None -> None
  | Some arch -> find_int t (Printf.sprintf "%s.%s" arch suffix)

let context_length t = model_param t "context_length"
let embedding_length t = model_param t "embedding_length"
let block_count t = model_param t "block_count"
let feed_forward_length t = model_param t "feed_forward_length"
let head_count t = model_param t "attention.head_count"
let head_count_kv t = model_param t "attention.head_count_kv"
let chat_template t = find_string t "tokenizer.chat_template"

module Transcode = struct
  let kvalues_iq4nl =
    [| -127; -104; -83; -65; -49; -35; -22; -10;
        1;   13;  25;  38;  53;  69;  89; 113 |]

  let iq4_xs_superblock_bytes = 136
  let q5_k_superblock_bytes = 176
  let q4_0_block_bytes = 18
  let q5_0_block_bytes = 22
  let q8_0_block_bytes = 34
  let q4_k_superblock_bytes = 144
  let q6_k_superblock_bytes = 210
  let tile32_affine4_block_bytes = 20
  let tile32_affine8_block_bytes = 36

  let decode_f16_u16 u16 =
    let sign = (u16 lsr 15) land 1 in
    let exp = (u16 lsr 10) land 31 in
    let frac = u16 land 1023 in
    if exp = 0 then
      if frac = 0 then (if sign = 1 then -0.0 else 0.0)
      else
        let f = float_of_int frac /. 1024.0 in
        let res = ldexp f (-14) in
        if sign = 1 then -.res else res
    else if exp = 31 then
      if frac = 0 then (if sign = 1 then neg_infinity else infinity)
      else nan
    else
      let f = 1.0 +. (float_of_int frac /. 1024.0) in
      let res = ldexp f (exp - 15) in
      if sign = 1 then -.res else res

  let decode_f16 bytes offset =
    decode_f16_u16 (Bytes.get_uint16_le bytes offset)

  let encode_f16 f =
    if Float.is_nan f then 0x7e00
    else if f = infinity then 0x7c00
    else if f = neg_infinity then 0xfc00
    else if f = 0.0 then
      if Int64.bits_of_float f < 0L then 0x8000 else 0
    else
      let sign = if f < 0.0 then 1 else 0 in
      let abs_f = abs_float f in
      let frac, exp = frexp abs_f in
      let exp16 = exp + 14 in
      if exp16 <= 0 then
        let mant = int_of_float (ldexp frac (10 + exp16) +. 0.5) in
        (sign lsl 15) lor (min 0x7bff mant)
      else if exp16 >= 31 then
        (sign lsl 15) lor 0x7c00
      else
        let mant = int_of_float ((frac *. 2.0 -. 1.0) *. 1024.0 +. 0.5) in
        let combined = (exp16 lsl 10) + mant in
        if combined >= 0x7c00 then (sign lsl 15) lor 0x7c00
        else (sign lsl 15) lor combined

  let dequantize_iq4_xs_superblock (src : bytes) (offset : int) (dst : float array) : unit =
    let d = decode_f16 src offset in
    let scales_h = Bytes.get_uint16_le src (offset + 2) in
    let scales_l = Bytes.sub src (offset + 4) 4 in
    let db = Array.make 8 0.0 in
    for j = 0 to 7 do
      let low = (Bytes.get_uint8 scales_l (j / 2) lsr (4 * (j mod 2))) land 0xF in
      let high = (scales_h lsr (2 * j)) land 3 in
      let ls = (low lor (high lsl 4)) - 32 in
      db.(j) <- d *. (2.0 ** (float_of_int ls /. 8.0))
    done;
    for sb = 0 to 7 do
      let scale = db.(sb) in
      let qs_offset = offset + 8 + (sb / 2) * 32 in
      let use_high_nibble = sb mod 2 = 1 in
      for i = 0 to 31 do
        let byte = Bytes.get_uint8 src (qs_offset + i) in
        let nibble = if use_high_nibble then byte lsr 4 else byte land 0xF in
        let kval = float_of_int kvalues_iq4nl.(nibble) in
        dst.(sb * 32 + i) <- scale *. kval
      done
    done

  let quantize_q5_k_superblock (src : float array) (dst : bytes) (offset : int) : unit =
    let sub_mins = Array.make 8 0.0 in
    let sub_maxs = Array.make 8 0.0 in
    let sub_ranges = Array.make 8 0.0 in
    for sb = 0 to 7 do
      let min_v = ref infinity in
      let max_v = ref neg_infinity in
      for i = 0 to 31 do
        let v = src.(sb * 32 + i) in
        if v < !min_v then min_v := v;
        if v > !max_v then max_v := v
      done;
      sub_mins.(sb) <- !min_v;
      sub_maxs.(sb) <- !max_v;
      sub_ranges.(sb) <- max 0.0 (!max_v -. !min_v)
    done;
    let max_range = ref 1e-7 in
    let max_min = ref 0.0 in
    for sb = 0 to 7 do
      if sub_ranges.(sb) > !max_range then max_range := sub_ranges.(sb);
      let m = -. sub_mins.(sb) in
      if m > !max_min then max_min := m
    done;
    let d = !max_range /. (31.0 *. 63.0) in
    let dmin = if !max_min > 0.0 then !max_min /. 63.0 else 0.0 in
    let sc = Array.make 8 0 in
    let m = Array.make 8 0 in
    for sb = 0 to 7 do
      sc.(sb) <- max 0 (min 63 (int_of_float ((sub_ranges.(sb) /. (31.0 *. d)) +. 0.5)));
      m.(sb) <- (if dmin > 0.0 then max 0 (min 63 (int_of_float (((-. sub_mins.(sb)) /. dmin) +. 0.5))) else 0)
    done;
    Bytes.set_uint16_le dst offset (encode_f16 d);
    Bytes.set_uint16_le dst (offset + 2) (encode_f16 dmin);
    for j = 0 to 3 do
      let sc_low = sc.(j) in
      let m_low = m.(j) in
      let sc_high = sc.(j + 4) in
      let m_high = m.(j + 4) in
      Bytes.set_uint8 dst (offset + 4 + j) ((sc_low land 63) lor ((sc_high lsr 4) lsl 6));
      Bytes.set_uint8 dst (offset + 8 + j) ((m_low land 63) lor ((m_high lsr 4) lsl 6));
      Bytes.set_uint8 dst (offset + 12 + j) ((sc_high land 0xF) lor ((m_high land 0xF) lsl 4))
    done;
    let qh = Bytes.make 32 '\000' in
    for sb = 0 to 7 do
      let dl = d *. float_of_int sc.(sb) in
      let ml = dmin *. float_of_int m.(sb) in
      let qs_base = offset + 16 + 32 + (sb / 2) * 32 in
      let is_odd = sb mod 2 = 1 in
      for i = 0 to 31 do
        let v = src.(sb * 32 + i) in
        let q = if dl > 0.0 then max 0 (min 31 (int_of_float (((v +. ml) /. dl) +. 0.5))) else 0 in
        let low_nib = q land 0xF in
        let high_bit = (q lsr 4) land 1 in
        let prev_qh = Bytes.get_uint8 qh i in
        Bytes.set_uint8 qh i (prev_qh lor (high_bit lsl sb));
        let cur_qs = Bytes.get_uint8 dst (qs_base + i) in
        let new_qs = if is_odd then (cur_qs land 0x0F) lor (low_nib lsl 4) else (cur_qs land 0xF0) lor low_nib in
        Bytes.set_uint8 dst (qs_base + i) new_qs
      done
    done;
    Bytes.blit qh 0 dst (offset + 16) 32

  let iq4_xs_to_q5_k (src : bytes) : (bytes, string) result =
    let len = Bytes.length src in
    if len mod iq4_xs_superblock_bytes <> 0 then
      Error (Printf.sprintf "input length %d is not a multiple of IQ4_XS superblock size %d" len iq4_xs_superblock_bytes)
    else
      let superblocks = len / iq4_xs_superblock_bytes in
      let dst_len = superblocks * q5_k_superblock_bytes in
      let dst = Bytes.make dst_len '\000' in
      let float_buf = Array.make 256 0.0 in
      for sb = 0 to superblocks - 1 do
        let src_offset = sb * iq4_xs_superblock_bytes in
        let dst_offset = sb * q5_k_superblock_bytes in
        dequantize_iq4_xs_superblock src src_offset float_buf;
        quantize_q5_k_superblock float_buf dst dst_offset
      done;
      Ok dst

  let unpack_k_scale_min (src : bytes) (table_offset : int) (sb : int) : int * int =
    let j = sb land 3 in
    let sc =
      if sb < 4 then Bytes.get_uint8 src (table_offset + j) land 63
      else
        (Bytes.get_uint8 src (table_offset + j + 8) land 15)
        lor ((Bytes.get_uint8 src (table_offset + j) lsr 6) lsl 4)
    in
    let m =
      if sb < 4 then Bytes.get_uint8 src (table_offset + j + 4) land 63
      else
        (Bytes.get_uint8 src (table_offset + j + 8) lsr 4)
        lor ((Bytes.get_uint8 src (table_offset + j + 4) lsr 6) lsl 4)
    in
    (sc, m)

  let q4_0_to_tile32_affine4 (src : bytes) : (bytes, string) result =
    let len = Bytes.length src in
    if len mod q4_0_block_bytes <> 0 then
      Error
        (Printf.sprintf
           "input length %d is not a multiple of Q4_0 block size %d" len
           q4_0_block_bytes)
    else
      let blocks = len / q4_0_block_bytes in
      let dst = Bytes.make (blocks * tile32_affine4_block_bytes) '\000' in
      for blk = 0 to blocks - 1 do
        let src_off = blk * q4_0_block_bytes in
        let dst_off = blk * tile32_affine4_block_bytes in
        let d_u16 = Bytes.get_uint16_le src src_off in
        let d = decode_f16_u16 d_u16 in
        let b_u16 = encode_f16 (-8.0 *. d) in
        Bytes.set_uint16_le dst dst_off d_u16;
        Bytes.set_uint16_le dst (dst_off + 2) b_u16;
        Bytes.blit src (src_off + 2) dst (dst_off + 4) 16
      done;
      Ok dst

  let q4_k_to_tile32_affine4 (src : bytes) : (bytes, string) result =
    let len = Bytes.length src in
    if len mod q4_k_superblock_bytes <> 0 then
      Error
        (Printf.sprintf
           "input length %d is not a multiple of Q4_K superblock size %d" len
           q4_k_superblock_bytes)
    else
      let superblocks = len / q4_k_superblock_bytes in
      let dst =
        Bytes.make (superblocks * 8 * tile32_affine4_block_bytes) '\000'
      in
      for sb_idx = 0 to superblocks - 1 do
        let src_off = sb_idx * q4_k_superblock_bytes in
        let d = decode_f16 src src_off in
        let dmin = decode_f16 src (src_off + 2) in
        let table_off = src_off + 4 in
        let qs_off = src_off + 16 in
        for sb = 0 to 7 do
          let sc, m = unpack_k_scale_min src table_off sb in
          let s_u16 = encode_f16 (d *. float_of_int sc) in
          let b_u16 = encode_f16 (-.dmin *. float_of_int m) in
          let dst_off = (sb_idx * 8 + sb) * tile32_affine4_block_bytes in
          Bytes.set_uint16_le dst dst_off s_u16;
          Bytes.set_uint16_le dst (dst_off + 2) b_u16;
          let pair_base = qs_off + (sb lsr 1) * 32 in
          let is_odd = (sb land 1) = 1 in
          for l = 0 to 15 do
            let b0 = Bytes.get_uint8 src (pair_base + l) in
            let b1 = Bytes.get_uint8 src (pair_base + l + 16) in
            let q0 = if is_odd then b0 lsr 4 else b0 land 15 in
            let q1 = if is_odd then b1 lsr 4 else b1 land 15 in
            Bytes.set_uint8 dst (dst_off + 4 + l) (q0 lor (q1 lsl 4))
          done
        done
      done;
      Ok dst

  let q8_0_to_tile32_affine8 (src : bytes) : (bytes, string) result =
    let len = Bytes.length src in
    if len mod q8_0_block_bytes <> 0 then
      Error
        (Printf.sprintf
           "input length %d is not a multiple of Q8_0 block size %d" len
           q8_0_block_bytes)
    else
      let blocks = len / q8_0_block_bytes in
      let dst = Bytes.make (blocks * tile32_affine8_block_bytes) '\000' in
      for blk = 0 to blocks - 1 do
        let src_off = blk * q8_0_block_bytes in
        let dst_off = blk * tile32_affine8_block_bytes in
        let d_u16 = Bytes.get_uint16_le src src_off in
        let d = decode_f16_u16 d_u16 in
        let b_u16 = encode_f16 (-128.0 *. d) in
        Bytes.set_uint16_le dst dst_off d_u16;
        Bytes.set_uint16_le dst (dst_off + 2) b_u16;
        for l = 0 to 31 do
          let q_signed = Bytes.get_int8 src (src_off + 2 + l) in
          Bytes.set_uint8 dst (dst_off + 4 + l) (q_signed + 128)
        done
      done;
      Ok dst

  let q5_0_to_tile32_affine8 (src : bytes) : (bytes, string) result =
    let len = Bytes.length src in
    if len mod q5_0_block_bytes <> 0 then
      Error
        (Printf.sprintf
           "input length %d is not a multiple of Q5_0 block size %d" len
           q5_0_block_bytes)
    else
      let blocks = len / q5_0_block_bytes in
      let dst = Bytes.make (blocks * tile32_affine8_block_bytes) '\000' in
      for blk = 0 to blocks - 1 do
        let src_off = blk * q5_0_block_bytes in
        let dst_off = blk * tile32_affine8_block_bytes in
        let d_u16 = Bytes.get_uint16_le src src_off in
        let d = decode_f16_u16 d_u16 in
        let b_u16 = encode_f16 (-16.0 *. d) in
        Bytes.set_uint16_le dst dst_off d_u16;
        Bytes.set_uint16_le dst (dst_off + 2) b_u16;
        for l = 0 to 31 do
          let high_byte = Bytes.get_uint8 src (src_off + 2 + (l lsr 3)) in
          let high_bit = (high_byte lsr (l land 7)) land 1 in
          let low_byte = Bytes.get_uint8 src (src_off + 6 + (l land 15)) in
          let low_nib = if l < 16 then low_byte land 15 else low_byte lsr 4 in
          let code = low_nib lor (high_bit lsl 4) in
          Bytes.set_uint8 dst (dst_off + 4 + l) code
        done
      done;
      Ok dst

  let q5_k_to_tile32_affine8 (src : bytes) : (bytes, string) result =
    let len = Bytes.length src in
    if len mod q5_k_superblock_bytes <> 0 then
      Error
        (Printf.sprintf
           "input length %d is not a multiple of Q5_K superblock size %d" len
           q5_k_superblock_bytes)
    else
      let superblocks = len / q5_k_superblock_bytes in
      let dst =
        Bytes.make (superblocks * 8 * tile32_affine8_block_bytes) '\000'
      in
      for sb_idx = 0 to superblocks - 1 do
        let src_off = sb_idx * q5_k_superblock_bytes in
        let d = decode_f16 src src_off in
        let dmin = decode_f16 src (src_off + 2) in
        let table_off = src_off + 4 in
        let qh_off = src_off + 16 in
        let qs_off = src_off + 48 in
        for sb = 0 to 7 do
          let sc, m = unpack_k_scale_min src table_off sb in
          let s_u16 = encode_f16 (d *. float_of_int sc) in
          let b_u16 = encode_f16 (-.dmin *. float_of_int m) in
          let dst_off = (sb_idx * 8 + sb) * tile32_affine8_block_bytes in
          Bytes.set_uint16_le dst dst_off s_u16;
          Bytes.set_uint16_le dst (dst_off + 2) b_u16;
          let pair_base = qs_off + (sb lsr 1) * 32 in
          let is_odd = (sb land 1) = 1 in
          for l = 0 to 31 do
            let packed = Bytes.get_uint8 src (pair_base + l) in
            let nibble = if is_odd then packed lsr 4 else packed land 15 in
            let high = (Bytes.get_uint8 src (qh_off + l) lsr sb) land 1 in
            let code = nibble lor (high lsl 4) in
            Bytes.set_uint8 dst (dst_off + 4 + l) code
          done
        done
      done;
      Ok dst

  let rec gcd a b = if b = 0 then abs a else gcd b (a mod b)

  let q6_k_to_tile32_affine8 (src : bytes) : (bytes, string) result =
    let len = Bytes.length src in
    if len mod q6_k_superblock_bytes <> 0 then
      Error
        (Printf.sprintf
           "input length %d is not a multiple of Q6_K superblock size %d" len
           q6_k_superblock_bytes)
    else
      let superblocks = len / q6_k_superblock_bytes in
      let dst =
        Bytes.make (superblocks * 8 * tile32_affine8_block_bytes) '\000'
      in
      let codes = Array.make 32 0 in
      let v_int = Array.make 32 0 in
      for sb_idx = 0 to superblocks - 1 do
        let src_off = sb_idx * q6_k_superblock_bytes in
        let d = decode_f16 src (src_off + 208) in
        for sb = 0 to 7 do
          let half_block = sb lsr 2 in
          let within = sb land 3 in
          let sc0 =
            Bytes.get_int8 src (src_off + 192 + half_block * 8 + within * 2)
          in
          let sc1 =
            Bytes.get_int8 src (src_off + 192 + half_block * 8 + within * 2 + 1)
          in
          for l = 0 to 31 do
            let low_index = half_block * 64 + (within land 1) * 32 + l in
            let low_shift = (within lsr 1) * 4 in
            let high_index = half_block * 32 + l in
            let high_shift = within * 2 in
            let code =
              ((Bytes.get_uint8 src (src_off + low_index) lsr low_shift) land 15)
              lor (((Bytes.get_uint8 src (src_off + 128 + high_index)
                    lsr high_shift)
                   land 3)
                  lsl 4)
            in
            codes.(l) <- code;
            let sc = if l < 16 then sc0 else sc1 in
            v_int.(l) <- sc * (code - 32)
          done;
          let dst_off = (sb_idx * 8 + sb) * tile32_affine8_block_bytes in
          if sc0 = sc1 then (
            let s_u16 = encode_f16 (d *. float_of_int sc0) in
            let s_dec = decode_f16_u16 s_u16 in
            let b_u16 = encode_f16 (-32.0 *. s_dec) in
            Bytes.set_uint16_le dst dst_off s_u16;
            Bytes.set_uint16_le dst (dst_off + 2) b_u16;
            for l = 0 to 31 do
              Bytes.set_uint8 dst (dst_off + 4 + l) codes.(l)
            done)
          else
            let v_min = Array.fold_left min max_int v_int in
            let v_max = Array.fold_left max min_int v_int in
            if v_max = v_min then (
              Bytes.set_uint16_le dst dst_off 0;
              Bytes.set_uint16_le dst (dst_off + 2)
                (encode_f16 (d *. float_of_int v_min));
              for l = 0 to 31 do
                Bytes.set_uint8 dst (dst_off + 4 + l) 0
              done)
            else
              let step_gcd =
                Array.fold_left (fun g v -> gcd g (v - v_min)) 0 v_int
              in
              let step_gcd = max 1 step_gcd in
              if (v_max - v_min) / step_gcd <= 255 then (
                let s_u16 = encode_f16 (d *. float_of_int step_gcd) in
                let b_u16 = encode_f16 (d *. float_of_int v_min) in
                Bytes.set_uint16_le dst dst_off s_u16;
                Bytes.set_uint16_le dst (dst_off + 2) b_u16;
                for l = 0 to 31 do
                  Bytes.set_uint8 dst (dst_off + 4 + l)
                    ((v_int.(l) - v_min) / step_gcd)
                done)
              else
                let w_min = d *. float_of_int v_min in
                let w_max = d *. float_of_int v_max in
                let s_u16 = encode_f16 ((w_max -. w_min) /. 255.0) in
                let b_u16 = encode_f16 w_min in
                let s_dec = decode_f16_u16 s_u16 in
                let b_dec = decode_f16_u16 b_u16 in
                Bytes.set_uint16_le dst dst_off s_u16;
                Bytes.set_uint16_le dst (dst_off + 2) b_u16;
                for l = 0 to 31 do
                  let w = d *. float_of_int v_int.(l) in
                  let q =
                    if abs_float s_dec > 0.0 then
                      max 0
                        (min 255
                           (int_of_float (((w -. b_dec) /. s_dec) +. 0.5)))
                    else 0
                  in
                  Bytes.set_uint8 dst (dst_off + 4 + l) q
                done
        done
      done;
      Ok dst

  let iq4_xs_to_tile32_affine8 (src : bytes) : (bytes, string) result =
    let len = Bytes.length src in
    if len mod iq4_xs_superblock_bytes <> 0 then
      Error
        (Printf.sprintf
           "input length %d is not a multiple of IQ4_XS superblock size %d" len
           iq4_xs_superblock_bytes)
    else
      let superblocks = len / iq4_xs_superblock_bytes in
      let dst =
        Bytes.make (superblocks * 8 * tile32_affine8_block_bytes) '\000'
      in
      for sb_idx = 0 to superblocks - 1 do
        let src_off = sb_idx * iq4_xs_superblock_bytes in
        let d = decode_f16 src src_off in
        let scales_h = Bytes.get_uint16_le src (src_off + 2) in
        for sb = 0 to 7 do
          let low_byte = Bytes.get_uint8 src (src_off + 4 + (sb lsr 1)) in
          let low = (low_byte lsr (4 * (sb land 1))) land 15 in
          let high = (scales_h lsr (2 * sb)) land 3 in
          let ls = (low lor (high lsl 4)) - 32 in
          let scale = d *. (2.0 ** (float_of_int ls /. 8.0)) in
          let s_u16 = encode_f16 scale in
          let s_dec = decode_f16_u16 s_u16 in
          let b_u16 = encode_f16 (-128.0 *. s_dec) in
          let dst_off = (sb_idx * 8 + sb) * tile32_affine8_block_bytes in
          Bytes.set_uint16_le dst dst_off s_u16;
          Bytes.set_uint16_le dst (dst_off + 2) b_u16;
          let qs_off = src_off + 8 + (sb lsr 1) * 32 in
          let is_odd = (sb land 1) = 1 in
          for l = 0 to 31 do
            let packed = Bytes.get_uint8 src (qs_off + l) in
            let nibble = if is_odd then packed lsr 4 else packed land 15 in
            let code = kvalues_iq4nl.(nibble) + 128 in
            Bytes.set_uint8 dst (dst_off + 4 + l) code
          done
        done
      done;
      Ok dst

  let to_tile32_affine (quant : Weight_archive.Dtype.quant_type) (src : bytes) :
      ([ `Affine4 of bytes | `Affine8 of bytes ], string) result =
    match quant with
    | Weight_archive.Dtype.Q4_0 ->
        Result.map (fun b -> `Affine4 b) (q4_0_to_tile32_affine4 src)
    | Weight_archive.Dtype.Q4_K ->
        Result.map (fun b -> `Affine4 b) (q4_k_to_tile32_affine4 src)
    | Weight_archive.Dtype.Q8_0 ->
        Result.map (fun b -> `Affine8 b) (q8_0_to_tile32_affine8 src)
    | Weight_archive.Dtype.Q5_0 ->
        Result.map (fun b -> `Affine8 b) (q5_0_to_tile32_affine8 src)
    | Weight_archive.Dtype.Q5_K ->
        Result.map (fun b -> `Affine8 b) (q5_k_to_tile32_affine8 src)
    | Weight_archive.Dtype.Q6_K ->
        Result.map (fun b -> `Affine8 b) (q6_k_to_tile32_affine8 src)
    | Weight_archive.Dtype.IQ4_XS ->
        Result.map (fun b -> `Affine8 b) (iq4_xs_to_tile32_affine8 src)
    | Weight_archive.Dtype.Tile32_Affine4 ->
        if Bytes.length src mod tile32_affine4_block_bytes <> 0 then
          Error "Tile32_Affine4 byte length is not a multiple of 20"
        else Ok (`Affine4 (Bytes.copy src))
    | Weight_archive.Dtype.Tile32_Affine8 ->
        if Bytes.length src mod tile32_affine8_block_bytes <> 0 then
          Error "Tile32_Affine8 byte length is not a multiple of 36"
        else Ok (`Affine8 (Bytes.copy src))

  let dequantize_tile32_affine4 (src : bytes) : (float array, string) result =
    let len = Bytes.length src in
    if len mod tile32_affine4_block_bytes <> 0 then
      Error
        (Printf.sprintf
           "input length %d is not a multiple of Tile32_Affine4 block size %d"
           len tile32_affine4_block_bytes)
    else
      let blocks = len / tile32_affine4_block_bytes in
      let out = Array.make (blocks * 32) 0.0 in
      for blk = 0 to blocks - 1 do
        let off = blk * tile32_affine4_block_bytes in
        let s = decode_f16 src off in
        let b = decode_f16 src (off + 2) in
        for l = 0 to 15 do
          let packed = Bytes.get_uint8 src (off + 4 + l) in
          let q0 = float_of_int (packed land 15) in
          let q1 = float_of_int (packed lsr 4) in
          out.(blk * 32 + l) <- (s *. q0) +. b;
          out.(blk * 32 + l + 16) <- (s *. q1) +. b
        done
      done;
      Ok out

  let dequantize_tile32_affine8 (src : bytes) : (float array, string) result =
    let len = Bytes.length src in
    if len mod tile32_affine8_block_bytes <> 0 then
      Error
        (Printf.sprintf
           "input length %d is not a multiple of Tile32_Affine8 block size %d"
           len tile32_affine8_block_bytes)
    else
      let blocks = len / tile32_affine8_block_bytes in
      let out = Array.make (blocks * 32) 0.0 in
      for blk = 0 to blocks - 1 do
        let off = blk * tile32_affine8_block_bytes in
        let s = decode_f16 src off in
        let b = decode_f16 src (off + 2) in
        for l = 0 to 31 do
          let q = float_of_int (Bytes.get_uint8 src (off + 4 + l)) in
          out.(blk * 32 + l) <- (s *. q) +. b
        done
      done;
      Ok out

  let dequantize_gguf (quant : Weight_archive.Dtype.quant_type) (src : bytes) :
      (float array, string) result =
    let len = Bytes.length src in
    let block_bytes = Weight_archive.Dtype.bytes_per_block quant in
    let block_elems = Weight_archive.Dtype.block_size quant in
    if len mod block_bytes <> 0 then
      Error
        (Printf.sprintf "input length %d is not a multiple of %s block size %d"
           len
           (Weight_archive.Dtype.quant_to_string quant)
           block_bytes)
    else
      let num_blocks = len / block_bytes in
      let out = Array.make (num_blocks * block_elems) 0.0 in
      (match quant with
      | Weight_archive.Dtype.Q4_0 ->
          for blk = 0 to num_blocks - 1 do
            let off = blk * q4_0_block_bytes in
            let d = decode_f16 src off in
            for l = 0 to 31 do
              let byte = Bytes.get_uint8 src (off + 2 + (l land 15)) in
              let nib = if l < 16 then byte land 15 else byte lsr 4 in
              out.(blk * 32 + l) <- d *. float_of_int (nib - 8)
            done
          done
      | Weight_archive.Dtype.Q5_0 ->
          for blk = 0 to num_blocks - 1 do
            let off = blk * q5_0_block_bytes in
            let d = decode_f16 src off in
            for l = 0 to 31 do
              let high_byte = Bytes.get_uint8 src (off + 2 + (l lsr 3)) in
              let high_bit = (high_byte lsr (l land 7)) land 1 in
              let low_byte = Bytes.get_uint8 src (off + 6 + (l land 15)) in
              let low_nib =
                if l < 16 then low_byte land 15 else low_byte lsr 4
              in
              let q = (low_nib lor (high_bit lsl 4)) - 16 in
              out.(blk * 32 + l) <- d *. float_of_int q
            done
          done
      | Weight_archive.Dtype.Q8_0 ->
          for blk = 0 to num_blocks - 1 do
            let off = blk * q8_0_block_bytes in
            let d = decode_f16 src off in
            for l = 0 to 31 do
              let q = Bytes.get_int8 src (off + 2 + l) in
              out.(blk * 32 + l) <- d *. float_of_int q
            done
          done
      | Weight_archive.Dtype.Q4_K ->
          for sb_idx = 0 to num_blocks - 1 do
            let off = sb_idx * q4_k_superblock_bytes in
            let d = decode_f16 src off in
            let dmin = decode_f16 src (off + 2) in
            for sb = 0 to 7 do
              let sc, m = unpack_k_scale_min src (off + 4) sb in
              let dl = decode_f16_u16 (encode_f16 (d *. float_of_int sc)) in
              let ml = decode_f16_u16 (encode_f16 (dmin *. float_of_int m)) in
              let pair_base = off + 16 + (sb lsr 1) * 32 in
              let is_odd = (sb land 1) = 1 in
              for l = 0 to 31 do
                let packed = Bytes.get_uint8 src (pair_base + l) in
                let code = if is_odd then packed lsr 4 else packed land 15 in
                out.(sb_idx * 256 + sb * 32 + l) <-
                  (dl *. float_of_int code) -. ml
              done
            done
          done
      | Weight_archive.Dtype.Q5_K ->
          for sb_idx = 0 to num_blocks - 1 do
            let off = sb_idx * q5_k_superblock_bytes in
            let d = decode_f16 src off in
            let dmin = decode_f16 src (off + 2) in
            for sb = 0 to 7 do
              let sc, m = unpack_k_scale_min src (off + 4) sb in
              let dl = decode_f16_u16 (encode_f16 (d *. float_of_int sc)) in
              let ml = decode_f16_u16 (encode_f16 (dmin *. float_of_int m)) in
              let pair_base = off + 48 + (sb lsr 1) * 32 in
              let is_odd = (sb land 1) = 1 in
              for l = 0 to 31 do
                let packed = Bytes.get_uint8 src (pair_base + l) in
                let nibble = if is_odd then packed lsr 4 else packed land 15 in
                let high = (Bytes.get_uint8 src (off + 16 + l) lsr sb) land 1 in
                let code = nibble lor (high lsl 4) in
                out.(sb_idx * 256 + sb * 32 + l) <-
                  (dl *. float_of_int code) -. ml
              done
            done
          done
      | Weight_archive.Dtype.Q6_K ->
          for sb_idx = 0 to num_blocks - 1 do
            let off = sb_idx * q6_k_superblock_bytes in
            let d = decode_f16 src (off + 208) in
            for sb = 0 to 7 do
              let half_block = sb lsr 2 in
              let within = sb land 3 in
              for l = 0 to 31 do
                let low_index = half_block * 64 + (within land 1) * 32 + l in
                let low_shift = (within lsr 1) * 4 in
                let high_index = half_block * 32 + l in
                let high_shift = within * 2 in
                let code =
                  ((Bytes.get_uint8 src (off + low_index) lsr low_shift) land 15)
                  lor (((Bytes.get_uint8 src (off + 128 + high_index)
                        lsr high_shift)
                       land 3)
                      lsl 4)
                in
                let scale_index =
                  half_block * 8 + (l lsr 4) + within * 2
                in
                let sc = Bytes.get_int8 src (off + 192 + scale_index) in
                out.(sb_idx * 256 + sb * 32 + l) <-
                  d *. float_of_int sc *. float_of_int (code - 32)
              done
            done
          done
      | Weight_archive.Dtype.IQ4_XS ->
          let tmp = Array.make 256 0.0 in
          for sb_idx = 0 to num_blocks - 1 do
            dequantize_iq4_xs_superblock src
              (sb_idx * iq4_xs_superblock_bytes)
              tmp;
            Array.blit tmp 0 out (sb_idx * 256) 256
          done
      | Weight_archive.Dtype.Tile32_Affine4 ->
          (match dequantize_tile32_affine4 src with
          | Ok arr -> Array.blit arr 0 out 0 (Array.length arr)
          | Error _ -> ())
      | Weight_archive.Dtype.Tile32_Affine8 ->
          (match dequantize_tile32_affine8 src with
          | Ok arr -> Array.blit arr 0 out 0 (Array.length arr)
          | Error _ -> ()));
      Ok out

  let fwht32 (src : float array) : (float array, string) result =
    let len = Array.length src in
    if len mod 32 <> 0 then
      Error
        (Printf.sprintf
           "fwht32 input length %d is not a multiple of 32" len)
    else
      let dst = Array.copy src in
      let norm = 0.17677669529663689 in (* 1 / sqrt(32) *)
      let blocks = len / 32 in
      for blk = 0 to blocks - 1 do
        let base = blk * 32 in
        let stride = ref 1 in
        while !stride < 32 do
          let s = !stride in
          for l = 0 to 31 do
            if (l land s) = 0 then (
              let u = dst.(base + l) in
              let v = dst.(base + (l lxor s)) in
              dst.(base + l) <- u +. v;
              dst.(base + (l lxor s)) <- u -. v)
          done;
          stride := s lsl 1
        done;
        for l = 0 to 31 do
          dst.(base + l) <- dst.(base + l) *. norm
        done
      done;
      Ok dst

  let quantize_tile32_affine8 (src : float array) : (bytes, string) result =
    let len = Array.length src in
    if len mod 32 <> 0 then
      Error
        (Printf.sprintf
           "quantize_tile32_affine8 input length %d is not a multiple of 32" len)
    else
      let blocks = len / 32 in
      let dst = Bytes.make (blocks * tile32_affine8_block_bytes) '\000' in
      for blk = 0 to blocks - 1 do
        let base = blk * 32 in
        let v_min = ref infinity in
        let v_max = ref neg_infinity in
        for l = 0 to 31 do
          let v = src.(base + l) in
          if v < !v_min then v_min := v;
          if v > !v_max then v_max := v
        done;
        let range = max 0.0 (!v_max -. !v_min) in
        let s_u16 = encode_f16 (range /. 255.0) in
        let b_u16 = encode_f16 !v_min in
        let s_dec = decode_f16_u16 s_u16 in
        let b_dec = decode_f16_u16 b_u16 in
        let dst_off = blk * tile32_affine8_block_bytes in
        Bytes.set_uint16_le dst dst_off s_u16;
        Bytes.set_uint16_le dst (dst_off + 2) b_u16;
        for l = 0 to 31 do
          let q =
            if s_dec > 1e-8 then
              max 0
                (min 255
                   (int_of_float (((src.(base + l) -. b_dec) /. s_dec) +. 0.5)))
            else 0
          in
          Bytes.set_uint8 dst (dst_off + 4 + l) q
        done
      done;
      Ok dst

  let quantize_tile32_affine4 (src : float array) : (bytes, string) result =
    let len = Array.length src in
    if len mod 32 <> 0 then
      Error
        (Printf.sprintf
           "quantize_tile32_affine4 input length %d is not a multiple of 32" len)
    else
      let blocks = len / 32 in
      let dst = Bytes.make (blocks * tile32_affine4_block_bytes) '\000' in
      for blk = 0 to blocks - 1 do
        let base = blk * 32 in
        let v_min = ref infinity in
        let v_max = ref neg_infinity in
        for l = 0 to 31 do
          let v = src.(base + l) in
          if v < !v_min then v_min := v;
          if v > !v_max then v_max := v
        done;
        let range = max 0.0 (!v_max -. !v_min) in
        let s_u16 = encode_f16 (range /. 15.0) in
        let b_u16 = encode_f16 !v_min in
        let s_dec = decode_f16_u16 s_u16 in
        let b_dec = decode_f16_u16 b_u16 in
        let dst_off = blk * tile32_affine4_block_bytes in
        Bytes.set_uint16_le dst dst_off s_u16;
        Bytes.set_uint16_le dst (dst_off + 2) b_u16;
        for l = 0 to 15 do
          let q0 =
            if s_dec > 1e-8 then
              max 0
                (min 15
                   (int_of_float (((src.(base + l) -. b_dec) /. s_dec) +. 0.5)))
            else 0
          in
          let q1 =
            if s_dec > 1e-8 then
              max 0
                (min 15
                   (int_of_float
                      (((src.(base + l + 16) -. b_dec) /. s_dec) +. 0.5)))
            else 0
          in
          Bytes.set_uint8 dst (dst_off + 4 + l) (q0 lor (q1 lsl 4))
        done
      done;
      Ok dst

  let bake_tensor_to_tile32_affine ?(rotate_fwht32 = false)
      ?(prefer_4bit = false) (quant : Weight_archive.Dtype.quant_type)
      (src : bytes) :
      (Weight_archive.Dtype.quant_type * bytes, string) result =
    if not rotate_fwht32 && not prefer_4bit then
      match to_tile32_affine quant src with
      | Ok (`Affine4 dst) -> Ok (Weight_archive.Dtype.Tile32_Affine4, dst)
      | Ok (`Affine8 dst) -> Ok (Weight_archive.Dtype.Tile32_Affine8, dst)
      | Error _ as err -> err
    else
      let* floats = dequantize_gguf quant src in
      let* transformed =
        if rotate_fwht32 then fwht32 floats else Ok floats
      in
      let use_4bit =
        match quant with
        | Weight_archive.Dtype.Q4_0 | Weight_archive.Dtype.Q4_K
        | Weight_archive.Dtype.IQ4_XS | Weight_archive.Dtype.Tile32_Affine4 ->
            true
        | Weight_archive.Dtype.Q5_0 | Weight_archive.Dtype.Q5_K ->
            prefer_4bit || rotate_fwht32
        | Weight_archive.Dtype.Q6_K | Weight_archive.Dtype.Q8_0
        | Weight_archive.Dtype.Tile32_Affine8 ->
            prefer_4bit
      in
      if use_4bit then
        let* dst = quantize_tile32_affine4 transformed in
        Ok (Weight_archive.Dtype.Tile32_Affine4, dst)
      else
        let* dst = quantize_tile32_affine8 transformed in
        Ok (Weight_archive.Dtype.Tile32_Affine8, dst)
end

let to_weight_archive (gguf : t) ~path : (Weight_archive.t, string) result =
  let* file_size =
    if gguf.file_size < 0L || gguf.file_size > Int64.of_int max_int then
      Error "gguf file size exceeds host integer range"
    else Ok (Int64.to_int gguf.file_size)
  in
  let* index_bytes =
    if gguf.data_offset < 0L || gguf.data_offset > Int64.of_int max_int then
      Error "gguf data offset exceeds host integer range"
    else Ok (Int64.to_int gguf.data_offset)
  in
  let tensors =
    List.map
      (fun (info : Tensor_info.t) ->
        let abs_offset = Int64.add gguf.data_offset info.offset in
        let offset = Int64.to_int abs_offset in
        Weight_archive.Tensor.create ~name:info.name ~dtype:info.dtype
          ~shape:info.shape ~offset ~byte_length:info.byte_length)
      gguf.tensors
  in
  Ok (Weight_archive.create ~path ~file_size ~index_bytes ~tensors)

let of_file_as_archive path : (Weight_archive.t, string) result =
  let* gguf = of_file path in
  to_weight_archive gguf ~path

let align_256 v = ((v + 255) / 256) * 256

let write_tile32_affine_archive ?(rotate_fwht32 = false) ?(prefer_4bit = false)
    ~source_path ~destination_path () : (Weight_archive.t, string) result =
  let* gguf = of_file source_path in
  let* raw =
    try
      let ic = open_in_bin source_path in
      Fun.protect
        ~finally:(fun () -> close_in_noerr ic)
        (fun () ->
          let len = in_channel_length ic in
          let buf = Bytes.create len in
          really_input ic buf 0 len;
          Ok buf)
    with Sys_error msg -> Error msg
  in
  let sorted_infos =
    List.sort
      (fun (a : Tensor_info.t) (b : Tensor_info.t) ->
        String.compare a.name b.name)
      gguf.tensors
  in
  let rec transcode_all acc = function
    | [] -> Ok (List.rev acc)
    | (info : Tensor_info.t) :: rest ->
        let abs_off =
          Int64.to_int (Int64.add gguf.data_offset info.offset)
        in
        let slice = Bytes.sub raw abs_off info.byte_length in
        let* new_dtype, payload =
          match info.dtype with
          | Weight_archive.Dtype.Quant q ->
              let* q', bytes' =
                Transcode.bake_tensor_to_tile32_affine ~rotate_fwht32
                  ~prefer_4bit q slice
              in
              Ok (Weight_archive.Dtype.Quant q', bytes')
          | other -> Ok (other, slice)
        in
        transcode_all ((info.name, new_dtype, info.shape, payload) :: acc) rest
  in
  let* transcoded = transcode_all [] sorted_infos in
  let raw_index_bytes =
    List.fold_left
      (fun acc (name, _dtype, shape, _payload) ->
        acc + 8 + String.length name + (8 * List.length shape) + 16)
      24 transcoded
  in
  let index_bytes = align_256 raw_index_bytes in
  let cursor = ref index_bytes in
  let laid_out =
    List.map
      (fun (name, dtype, shape, payload) ->
        let off = align_256 !cursor in
        let len = Bytes.length payload in
        cursor := off + len;
        let tensor =
          Weight_archive.Tensor.create ~name ~dtype ~shape ~offset:off
            ~byte_length:len
        in
        (tensor, payload))
      transcoded
  in
  let writer = Binary.Writer.create () in
  Binary.Writer.raw_string writer "LLMOPTWT";
  Binary.Writer.u16 writer 1;
  Binary.Writer.u16 writer 0;
  Binary.Writer.u32 writer (List.length laid_out);
  Binary.Writer.u64 writer index_bytes;
  List.iter
    (fun (tensor, _payload) ->
      let name = Weight_archive.Tensor.name tensor in
      let shape = Weight_archive.Tensor.shape tensor in
      let dtype = Weight_archive.Tensor.dtype tensor in
      Binary.Writer.u32 writer (String.length name);
      Binary.Writer.u8 writer (Weight_archive.Dtype.to_tag dtype);
      Binary.Writer.u8 writer (List.length shape);
      Binary.Writer.u16 writer 0;
      Binary.Writer.raw_string writer name;
      List.iter (Binary.Writer.u64 writer) shape;
      Binary.Writer.u64 writer (Weight_archive.Tensor.offset tensor);
      Binary.Writer.u64 writer (Weight_archive.Tensor.byte_length tensor))
    laid_out;
  let header_prefix = Binary.Writer.contents writer in
  let full_buf = Bytes.make !cursor '\000' in
  Bytes.blit header_prefix 0 full_buf 0 (Bytes.length header_prefix);
  List.iter
    (fun (tensor, payload) ->
      Bytes.blit payload 0 full_buf
        (Weight_archive.Tensor.offset tensor)
        (Bytes.length payload))
    laid_out;
  let* () =
    try
      let oc = open_out_bin destination_path in
      Fun.protect
        ~finally:(fun () -> close_out_noerr oc)
        (fun () ->
          output_bytes oc full_buf;
          Ok ())
    with Sys_error msg -> Error msg
  in
  Weight_archive.of_file destination_path
