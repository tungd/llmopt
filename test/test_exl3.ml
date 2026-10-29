let get = function Ok x -> x | Error e -> failwith e
let check b message = if not b then failwith message
let rejects value = match value with Error _ -> true | Ok _ -> false
let () =
  List.iter (fun bits ->
    let t = get (Exl3.create ~bits ~k:256 ~n:384) in
    let rt = get (Exl3.of_metadata (Exl3.metadata t)) in
    check (Exl3.bits rt = bits && Exl3.k rt = 256 && Exl3.n rt = 384) "metadata roundtrip";
    check (Exl3.packed_bytes rt = 256 * 384 * bits / 8) "packed byte geometry"
  ) [1; 3; 4; 8];
  List.iter (fun (bits,k,n) -> check (rejects (Exl3.create ~bits ~k ~n)) "invalid storage accepted")
    [0,128,128; 9,128,128; 4,0,128; 4,128,127; 4,max_int,128; 4,128,(-128)];
  let t = get (Exl3.create ~bits:4 ~k:128 ~n:256) in
  List.iter (fun offset ->
    let b = Exl3.metadata t in Bytes.set b offset '\255';
    check (rejects (Exl3.of_metadata b)) "invalid metadata accepted"
  ) [0; 8; 12; 16; 20; 24; 28];
  check (rejects (Exl3.of_metadata Bytes.empty)) "short metadata accepted";
  check (rejects (Metal_exl3_codegen.Schedule.matrix ~rows:7)) "invalid matrix tile accepted";
  check (rejects (Metal_exl3_codegen.Schedule.vector ~rows:3)) "invalid vector tile accepted";
  let vector = get (Metal_exl3_codegen.Schedule.vector ~rows:4) in
  let matrix = get (Metal_exl3_codegen.Schedule.matrix ~rows:16) in
  check (Metal_exl3_codegen.Schedule.grid vector ~m:17 ~n:256 = (64,5,1)) "vector partial row dispatch";
  check (Metal_exl3_codegen.Schedule.grid matrix ~m:17 ~n:256 = (16,2,1)) "matrix partial row dispatch";
  List.iter (fun path ->
    let archive = get (Weight_archive.of_file path) in
    let bound = get (Exl3.Archive.bind archive ~prefix:"weight") in
    check (Exl3.packed_bytes (Exl3.Archive.storage bound) = Weight_archive.Tensor.byte_length (Exl3.Archive.trellis bound)) "archive binding geometry";
    check (rejects (Exl3.Archive.bind archive ~prefix:"missing")) "missing components accepted";
    let renamed = Weight_archive.tensors archive |> List.filter (fun t -> Weight_archive.Tensor.name t <> "weight.suh") in
    let incomplete = Weight_archive.create ~path ~file_size:(Weight_archive.file_size archive)
      ~index_bytes:(Weight_archive.index_bytes archive) ~tensors:renamed in
    check (rejects (Exl3.Archive.bind incomplete ~prefix:"weight")) "incomplete components accepted";
    let wrong_bytes = Weight_archive.tensors archive |> List.map (fun t ->
      if Weight_archive.Tensor.name t = "weight.suh" then
        Weight_archive.Tensor.create ~name:"weight.suh" ~dtype:Weight_archive.Dtype.F16
          ~shape:(Weight_archive.Tensor.shape t) ~offset:(Weight_archive.Tensor.offset t) ~byte_length:1
      else t) in
    let short = Weight_archive.create ~path ~file_size:(Weight_archive.file_size archive)
      ~index_bytes:(Weight_archive.index_bytes archive) ~tensors:wrong_bytes in
    check (rejects (Exl3.Archive.bind short ~prefix:"weight")) "short scale component accepted";
    List.iter (fun (dtype, shape) ->
      let wrong = Weight_archive.tensors archive |> List.map (fun t ->
        if Weight_archive.Tensor.name t = "weight.suh" then
          Weight_archive.Tensor.create ~name:"weight.suh" ~dtype ~shape
            ~offset:(Weight_archive.Tensor.offset t) ~byte_length:(Weight_archive.Tensor.byte_length t)
        else t) in
      let bad = Weight_archive.create ~path ~file_size:(Weight_archive.file_size archive)
        ~index_bytes:(Weight_archive.index_bytes archive) ~tensors:wrong in
      check (rejects (Exl3.Archive.bind bad ~prefix:"weight")) "wrong scale contract accepted"
    ) [Weight_archive.Dtype.F32, [Exl3.k (Exl3.Archive.storage bound)]; Weight_archive.Dtype.F16, [1]]
  ) ["test/fixtures/exl3/mul1-4bit.llmopt"; "test/fixtures/exl3/mul1-3bit.llmopt"];
  print_endline "EXL3 schema/archive tests passed"
