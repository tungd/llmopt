let ( let* ) = Result.bind

type options = {
  output : string;
  warmup : int;
  repeats : int;
  byte_counts : int list;
}

external measure_metal : int -> int -> int -> string * int * float * float
  = "caml_llmopt_metal_measure_hardware"

let defaults =
  {
    output = "target-hardware.json";
    warmup = 5;
    repeats = 21;
    byte_counts = [ 16 * 1024 * 1024; 64 * 1024 * 1024; 256 * 1024 * 1024 ];
  }

let usage () =
  prerr_endline
    "usage: llmopt-hardware-probe [--output path] [--warmup count] \
     [--repeats count] [--bytes count]";
  exit 64

let nonnegative name value =
  match int_of_string_opt value with
  | Some value when value >= 0 -> value
  | _ -> invalid_arg (name ^ " must be non-negative")

let positive name value =
  match int_of_string_opt value with
  | Some value when value > 0 -> value
  | _ -> invalid_arg (name ^ " must be positive")

let arguments () =
  let rec loop options = function
    | [] -> options
    | "--output" :: value :: rest -> loop { options with output = value } rest
    | "--warmup" :: value :: rest ->
        loop { options with warmup = nonnegative "warmup" value } rest
    | "--repeats" :: value :: rest ->
        loop { options with repeats = positive "repeats" value } rest
    | "--bytes" :: value :: rest ->
        let bytes = positive "bytes" value in
        loop { options with byte_counts = [ bytes ] } rest
    | _ -> usage ()
  in
  match Array.to_list Sys.argv with _ :: rest -> loop defaults rest | [] -> usage ()

let write_atomic path contents =
  let directory = Filename.dirname path in
  let temporary = Filename.temp_file ~temp_dir:directory ".hardware-probe-" ".json" in
  let channel = open_out_bin temporary in
  Fun.protect
    ~finally:(fun () -> close_out_noerr channel)
    (fun () -> output_string channel contents);
  Unix.rename temporary path

let () =
  let options = arguments () in
  let measurements =
    List.map
      (fun bytes ->
        let device_name, max_threads, dispatch_seconds, bandwidth_gbps =
          measure_metal options.warmup options.repeats bytes
        in
        (bytes, device_name, max_threads, dispatch_seconds, bandwidth_gbps))
      options.byte_counts
  in
  let _, device_name, max_threads, _, _ = List.hd measurements in
  let dispatch_seconds =
    measurements
    |> List.map (fun (_, _, _, dispatch, _) -> dispatch)
    |> List.sort Float.compare
    |> fun samples -> List.nth samples (List.length samples / 2)
  in
  let bandwidth_bytes, _, _, _, bandwidth_gbps =
    List.fold_left
      (fun ((largest, _, _, _, _) as selected)
           ((bytes, _, _, _, _) as measurement) ->
        if bytes > largest then measurement else selected)
      (List.hd measurements) (List.tl measurements)
  in
  let discovered = Target_hardware.discover ~device_name () in
  let measured =
    Target_hardware.calibrate ~measured_bandwidth_gbps:bandwidth_gbps
      ~measured_dispatch_overhead_seconds:dispatch_seconds discovered
  in
  let target = Target_hardware.to_json measured in
  let report =
    `Assoc
      [
        "schema_version", `Int 1;
        "backend", `String "metal";
        "target", target;
        ( "measurement",
          `Assoc
            [
              "warmup", `Int options.warmup;
              "repeats", `Int options.repeats;
              "max_threads_per_threadgroup", `Int max_threads;
              "dispatch_overhead_seconds", `Float dispatch_seconds;
              "memory_bandwidth_gbps", `Float bandwidth_gbps;
              "memory_bandwidth_stream_bytes", `Int bandwidth_bytes;
              ( "streaming_copy_sweep",
                `List
                  (List.map
                     (fun (bytes, _, _, dispatch, bandwidth) ->
                       `Assoc
                         [
                           "stream_bytes", `Int bytes;
                           "dispatch_overhead_seconds", `Float dispatch;
                           "effective_bandwidth_gbps", `Float bandwidth;
                         ])
                     measurements) );
            ] );
      ]
  in
  write_atomic options.output (Yojson.Basic.pretty_to_string report ^ "\n");
  Printf.printf
    "device=%s backend=metal dispatch_us=%.3f sustained_effective_bandwidth_gbps=%.3f stream_bytes=%d output=%s\n%!"
    device_name (dispatch_seconds *. 1e6) bandwidth_gbps bandwidth_bytes
    options.output;
  List.iter
    (fun (bytes, _, _, _, bandwidth) ->
      Printf.printf "stream_bytes=%d effective_bandwidth_gbps=%.3f\n%!" bytes
        bandwidth)
    measurements
