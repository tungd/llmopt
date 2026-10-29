let ( let* ) = Result.bind

type t = { bits : int; k : int; n : int }
let bits t = t.bits
let k t = t.k
let n t = t.n
let packed_bytes t = (t.k / 16) * (t.n / 16) * (32 * t.bits)

let create ~bits ~k ~n =
  if bits < 1 || bits > 8 then Error "EXL3 mul1 integer rate must be in 1..8"
  else if k <= 0 || n <= 0 || k mod 128 <> 0 || n mod 128 <> 0 then
    Error "EXL3 dimensions must be positive multiples of 128"
  else if k > 0x7fffffff || n > 0x7fffffff || k > 0x7fffffff / n then
    Error "EXL3 dimensions exceed addressable tensor size"
  else Ok { bits; k; n }

let metadata t =
  let b = Bytes.make 32 '\000' in
  Bytes.blit_string "EXL3MUL1" 0 b 0 8;
  List.iteri (fun i v -> Bytes.set_int32_le b (8 + i * 4) (Int32.of_int v))
    [1; t.bits; t.k; t.n; 128; 0];
  b

let of_metadata b =
  if Bytes.length b <> 32 || Bytes.sub_string b 0 8 <> "EXL3MUL1" then
    Error "invalid EXL3 mul1 metadata"
  else
    let field i = Int32.to_int (Bytes.get_int32_le b (8 + i * 4)) in
    if field 0 <> 1 || field 4 <> 128 || field 5 <> 0 then
      Error "unsupported EXL3 metadata version, transform, or flags"
    else create ~bits:(field 1) ~k:(field 2) ~n:(field 3)

module Archive = struct
  type storage = t
  type t = {
    storage : storage;
    trellis : Weight_archive.Tensor.t;
    suh : Weight_archive.Tensor.t;
    svh : Weight_archive.Tensor.t;
  }
  let storage t = t.storage
  let trellis t = t.trellis
  let suh t = t.suh
  let svh t = t.svh

  let bind archive ~prefix =
    let component suffix dtype shape =
      let name = prefix ^ "." ^ suffix in
      match Weight_archive.find archive name with
      | None -> Error ("missing EXL3 archive component: " ^ name)
      | Some tensor when Weight_archive.Tensor.dtype tensor <> dtype
          || Weight_archive.Tensor.shape tensor <> shape
          || Weight_archive.Tensor.byte_length tensor <>
             List.fold_left ( * ) (Weight_archive.Dtype.byte_width dtype) shape ->
          Error ("EXL3 archive component has wrong dtype or shape: " ^ name)
      | Some tensor -> Ok tensor
    in
    let* meta = component "exl3" Weight_archive.Dtype.U8 [32] in
    let* contents =
      try
        let channel = open_in_bin (Weight_archive.path archive) in
        Fun.protect ~finally:(fun () -> close_in_noerr channel) (fun () ->
          seek_in channel (Weight_archive.Tensor.offset meta);
          Ok (Bytes.of_string (really_input_string channel 32)))
      with Sys_error message -> Error message
         | End_of_file -> Error "truncated EXL3 metadata"
    in
    let* storage = of_metadata contents in
    let* trellis, scale_prefix =
      match Weight_archive.find archive prefix with
      | None ->
          let* tensor = component "trellis" Weight_archive.Dtype.U8
            [storage.k / 16; storage.n / 16; 32 * storage.bits] in
          Ok (tensor, [])
      | Some tensor ->
          let* experts, scale_prefix = match Weight_archive.Tensor.shape tensor with
            | [n; k] when n = storage.n && k = storage.k -> Ok (1, [])
            | [experts; n; k] when experts > 0 && n = storage.n && k = storage.k ->
                Ok (experts, [experts])
            | _ -> Error ("EXL3 logical weight shape mismatch: " ^ prefix) in
          if storage.bits <> 4
             || Weight_archive.Tensor.dtype tensor <> Weight_archive.Dtype.Quant Ir.Dtype.Exl3_4
             || Weight_archive.Tensor.byte_length tensor <> experts * packed_bytes storage
          then Error ("EXL3 packed weight dtype or length mismatch: " ^ prefix)
          else Ok (tensor, scale_prefix)
    in
    let* suh = component "suh" Weight_archive.Dtype.F16 (scale_prefix @ [storage.k]) in
    let* svh = component "svh" Weight_archive.Dtype.F16 (scale_prefix @ [storage.n]) in
    Ok { storage; trellis; suh; svh }
end
