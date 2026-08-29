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
      let tensors =
        List.map
          (fun (t : Tensor_info.t) ->
            { t with offset = Int64.add data_offset t.offset })
          raw_tensors
      in
      Ok { version; alignment; metadata; tensors; data_offset; file_size }

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

  let decode_f16 bytes offset =
    let u16 = Bytes.get_uint16_le bytes offset in
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

  let encode_f16 f =
    if Float.is_nan f then 0x7e00
    else if f = infinity then 0x7c00
    else if f = neg_infinity then 0xfc00
    else if f = 0.0 then 0
    else if f = -0.0 then 0x8000
    else
      let sign = if f < 0.0 then 1 else 0 in
      let abs_f = abs_float f in
      let frac, exp = frexp abs_f in
      let exp16 = exp + 14 in
      if exp16 <= 0 then
        let mant = int_of_float (ldexp frac (10 + exp16)) in
        (sign lsl 15) lor mant
      else if exp16 >= 31 then
        (sign lsl 15) lor 0x7c00
      else
        let mant = int_of_float ((frac *. 2.0 -. 1.0) *. 1024.0 +. 0.5) in
        (sign lsl 15) lor (exp16 lsl 10) lor (mant land 1023)

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
end
