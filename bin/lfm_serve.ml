let ( let* ) = Result.bind

type options = {
  host : string;
  port : int;
  kv_format : Kv_cache.Format.t;
  token_capacity : int;
  checkpoint_capacity : int;
  max_body_bytes : int;
}

let defaults =
  {
    host = "127.0.0.1";
    port = 8000;
    kv_format = Kv_cache.Format.default;
    token_capacity = 8_192;
    checkpoint_capacity = 1_024;
    max_body_bytes = 64 * 1_024 * 1_024;
  }

let usage () =
  prerr_endline
    "usage: llmopt-serve [--host address] [--port number] [--kv q8|fp16] \
     [--token-capacity count] [--checkpoint-capacity count] \
     [--max-body-bytes count] <tokenizer.llmopt> <prefill-directory> \
     <decode-directory>";
  exit 64

let positive name value =
  match int_of_string_opt value with
  | Some value when value > 0 -> Ok value
  | _ -> Error (name ^ " must be a positive integer")

let kv_format = function
  | "q8" | "q8-group-64" -> Ok Kv_cache.Format.default
  | "fp16" | "f16" -> Ok Kv_cache.Format.f16
  | value -> Error ("unsupported KV format: " ^ value)

let arguments () =
  let rec parse options positional = function
    | [] -> Ok (options, List.rev positional)
    | "--host" :: value :: rest ->
        parse { options with host = value } positional rest
    | "--port" :: value :: rest ->
        let* port = positive "port" value in
        if port > 65_535 then Error "port must not exceed 65535"
        else parse { options with port } positional rest
    | "--kv" :: value :: rest ->
        let* kv_format = kv_format value in
        parse { options with kv_format } positional rest
    | "--token-capacity" :: value :: rest ->
        let* token_capacity = positive "token-capacity" value in
        parse { options with token_capacity } positional rest
    | "--checkpoint-capacity" :: value :: rest ->
        let* checkpoint_capacity = positive "checkpoint-capacity" value in
        parse { options with checkpoint_capacity } positional rest
    | "--max-body-bytes" :: value :: rest ->
        let* max_body_bytes = positive "max-body-bytes" value in
        parse { options with max_body_bytes } positional rest
    | option :: _ when String.starts_with ~prefix:"--" option ->
        Error ("unknown option: " ^ option)
    | value :: rest -> parse options (value :: positional) rest
  in
  match Array.to_list Sys.argv with
  | _ :: arguments -> parse defaults [] arguments
  | [] -> usage ()

let package root =
  Serving_package.of_file (Filename.concat root "package.llmopt")

type service = {
  tokenizer : Tokenizer.t;
  generation : Generation.t;
  mutable request_number : int;
}

let load options tokenizer_path prefill_root decode_root =
  let* tokenizer = Tokenizer.of_file tokenizer_path in
  let* prefill_package = package prefill_root in
  let* decode_package = package decode_root in
  let page_size =
    prefill_package |> Serving_package.cache |> Serving_package.Cache.page_size
  in
  let* cache_config =
    Serving_cache.Config.create ~model:Lfm25.Config.default
      ~kv_format:options.kv_format ~token_capacity:options.token_capacity
      ~checkpoint_capacity:options.checkpoint_capacity ~page_size ()
  in
  let* () =
    Serving_engine.validate_packages ~config:cache_config
      ~prefill:prefill_package ~decode:decode_package
  in
  let* runtimes =
    Metal_runtime.load_packages
      [ prefill_root, prefill_package; decode_root, decode_package ]
  in
  let* prefill, decode =
    match runtimes with
    | [ prefill; decode ] -> Ok (prefill, decode)
    | _ -> Error "serving pair load returned an invalid runtime count"
  in
  let* engine =
    Serving_engine.create ~config:cache_config ~prefill ~decode
  in
  let* generation = Generation.create ~tokenizer ~engine in
  Ok
    ( { tokenizer; generation; request_number = 0 },
      Metal_runtime.device_name prefill )

let trim_cr line =
  let length = String.length line in
  if length > 0 && line.[length - 1] = '\r' then
    String.sub line 0 (length - 1)
  else line

let lowercase = String.lowercase_ascii

let read_headers input =
  let rec read output =
    match input_line input |> trim_cr with
    | "" -> Ok (List.rev output)
    | line ->
        (match String.index_opt line ':' with
        | None -> Error "malformed HTTP header"
        | Some separator ->
            let name = String.sub line 0 separator |> lowercase in
            let value =
              String.sub line (separator + 1)
                (String.length line - separator - 1)
              |> String.trim
            in
            read ((name, value) :: output))
  in
  try read [] with End_of_file -> Error "incomplete HTTP headers"

let content_length headers =
  match List.assoc_opt "content-length" headers with
  | None -> Error "Content-Length is required"
  | Some value ->
      (match int_of_string_opt value with
      | Some length when length >= 0 -> Ok length
      | _ -> Error "Content-Length is invalid")

let write_response output ~status ~content_type body =
  Printf.fprintf output "HTTP/1.1 %s\r\n" status;
  Printf.fprintf output "Content-Type: %s\r\n" content_type;
  Printf.fprintf output "Content-Length: %d\r\n" (String.length body);
  output_string output "Connection: close\r\n\r\n";
  output_string output body;
  flush output

let write_error output ~status message =
  write_response output ~status ~content_type:"application/json"
    (Openai_protocol.error_body message)

let write_stream_headers output =
  output_string output "HTTP/1.1 200 OK\r\n";
  output_string output "Content-Type: text/event-stream\r\n";
  output_string output "Cache-Control: no-cache\r\n";
  output_string output "Connection: close\r\n\r\n";
  flush output

exception Stream_error of string

let stream service output request =
  service.request_number <- service.request_number + 1;
  let id = Printf.sprintf "chatcmpl-llmopt-%d" service.request_number in
  let model = Openai_protocol.Request.model request in
  let created = int_of_float (Unix.time ()) in
  let decoder = Tokenizer.Decoder.create service.tokenizer in
  let emit token =
    match Tokenizer.Decoder.push decoder token with
    | Error message -> raise (Stream_error message)
    | Ok text ->
        output_string output
          (Openai_protocol.Sse.content ~id ~model ~created ~token_id:token text);
        flush output
  in
  let* config =
    Generation_core.Config.create
      ~max_new_tokens:(Openai_protocol.Request.max_tokens request)
  in
  write_stream_headers output;
  let* result =
    Generation.generate ~emit
      ~ignore_eos:(Openai_protocol.Request.ignore_eos request)
      service.generation ~config
      ~messages:(Openai_protocol.Request.messages request)
  in
  let* () = Tokenizer.Decoder.finish decoder in
  let prompt_tokens = Generation.Result.prompt_tokens result |> Array.length in
  let completion_tokens =
    Generation.Result.completion_tokens result |> Array.length
  in
  let reason =
    Generation.Result.finish_reason result
    |> Generation_core.Finish_reason.to_string
  in
  output_string output
    (Openai_protocol.Sse.finish ~id ~model ~created ~reason);
  output_string output
    (Openai_protocol.Sse.usage ~id ~model ~created ~prompt_tokens
       ~cached_prompt_tokens:(Generation.Result.cached_prompt_tokens result)
       ~completion_tokens);
  output_string output Openai_protocol.Sse.done_;
  flush output;
  Ok ()

let handle service options input output =
  let request_line = try Some (input_line input |> trim_cr) with End_of_file -> None in
  match request_line with
  | None -> ()
  | Some request_line ->
      (match String.split_on_char ' ' request_line with
      | [ "GET"; ("/health" | "/healthz"); _ ] ->
          ignore (read_headers input);
          write_response output ~status:"200 OK" ~content_type:"text/plain"
            "ok\n"
      | [ "POST"; "/v1/chat/completions"; _ ] ->
          (match read_headers input with
          | Error message -> write_error output ~status:"400 Bad Request" message
          | Ok headers ->
              (match content_length headers with
              | Error message ->
                  write_error output ~status:"411 Length Required" message
              | Ok length when length > options.max_body_bytes ->
                  write_error output ~status:"413 Content Too Large"
                    "request body exceeds max-body-bytes"
              | Ok length ->
                  let body = really_input_string input length in
                  (match Openai_protocol.Request.of_string body with
                  | Error message ->
                      write_error output ~status:"400 Bad Request" message
                  | Ok request ->
                      (match stream service output request with
                      | Ok () -> ()
                      | Error message -> raise (Stream_error message)))))
      | _ ->
          ignore (read_headers input);
          write_error output ~status:"404 Not Found" "endpoint not found")

let serve service options =
  let address =
    try Ok (Unix.inet_addr_of_string options.host)
    with Failure _ -> Error "host must be a numeric IPv4 or IPv6 address"
  in
  let* address = address in
  let socket = Unix.socket Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.setsockopt socket Unix.SO_REUSEADDR true;
  Unix.bind socket (Unix.ADDR_INET (address, options.port));
  Unix.listen socket 128;
  Printf.eprintf "ready: http://%s:%d\n%!" options.host options.port;
  let rec accept () =
    let client, _ = Unix.accept socket in
    let output_descriptor = Unix.dup client in
    let input = Unix.in_channel_of_descr client in
    let output = Unix.out_channel_of_descr output_descriptor in
    (try handle service options input output with
    | End_of_file -> ()
    | Sys_error _ -> ()
    | Unix.Unix_error _ -> ()
    | Stream_error message ->
        Printf.eprintf "stream error: %s\n%!" message);
    (try flush output with _ -> ());
    close_in_noerr input;
    close_out_noerr output;
    accept ()
  in
  Fun.protect ~finally:(fun () -> Unix.close socket) accept

let run () =
  let* options, positional = arguments () in
  let* tokenizer, prefill, decode =
    match positional with
    | [ tokenizer; prefill; decode ] -> Ok (tokenizer, prefill, decode)
    | _ -> usage ()
  in
  let* service, device = load options tokenizer prefill decode in
  Printf.eprintf "device: %s; kv: %s\n%!" device
    (Kv_cache.Format.to_string options.kv_format);
  Sys.set_signal Sys.sigpipe Sys.Signal_ignore;
  serve service options

let () =
  match run () with
  | Ok () -> ()
  | Error message ->
      prerr_endline message;
      exit 2
