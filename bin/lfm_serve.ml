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
    "usage: llmopt-serve [--host address] [--port number] [--kv q8] \
     [--token-capacity count] [--checkpoint-capacity count] \
     [--max-body-bytes count] <engine-directory | tokenizer.llmopt prefill-dir decode-dir>";
  exit 64

let positive name value =
  match int_of_string_opt value with
  | Some value when value > 0 -> Ok value
  | _ -> Error (name ^ " must be a positive integer")

let kv_format = function
  | "q8" | "q8-group-64" -> Ok Kv_cache.Format.default
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

type active_request = {
  id : Serving_queue.Request_id.t;
  id_str : string;
  model : string;
  created : int;
  prompt_tokens : int array;
  cached_prompt_tokens : int;
  decoder : Tokenizer.Decoder.t;
  out : out_channel;
  in_ch : in_channel;
  mutable driver_state : Generation.Driver.State.t option;
  mutable allocated_tokens : int;
}

let emit_token active_req token =
  match Tokenizer.Decoder.push active_req.decoder token with
  | Error message -> Error message
  | Ok text ->
      output_string active_req.out
        (Openai_protocol.Sse.content ~id:active_req.id_str ~model:active_req.model
           ~created:active_req.created ~token_id:token text);
      flush active_req.out;
      Ok ()

let emit_finish active_req ~finish_reason ~completion_count =
  let prompt_tokens = Array.length active_req.prompt_tokens in
  let reason = Generation_core.Finish_reason.to_string finish_reason in
  output_string active_req.out
    (Openai_protocol.Sse.finish ~id:active_req.id_str ~model:active_req.model
       ~created:active_req.created ~reason);
  output_string active_req.out
    (Openai_protocol.Sse.usage ~id:active_req.id_str ~model:active_req.model
       ~created:active_req.created ~prompt_tokens
       ~cached_prompt_tokens:active_req.cached_prompt_tokens
       ~completion_tokens:completion_count);
  output_string active_req.out Openai_protocol.Sse.done_;
  flush active_req.out;
  close_out_noerr active_req.out;
  close_in_noerr active_req.in_ch

let close_active_request active_req =
  close_out_noerr active_req.out;
  close_in_noerr active_req.in_ch

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
  Unix.set_nonblock socket;
  Printf.eprintf "ready: http://%s:%d\n%!" options.host options.port;

  let queue =
    Serving_queue.create ~token_capacity:options.token_capacity
      ~high_watermark_ratio:0.90 ~low_watermark_ratio:0.75 ()
  in
  let active_requests : (Serving_queue.Request_id.t, active_request) Hashtbl.t =
    Hashtbl.create 32
  in

  let accept_new_connections () =
    let rec accept_loop () =
      match Unix.accept socket with
      | client, _ ->
          Unix.clear_nonblock client;
          let output_descriptor = Unix.dup client in
          let in_ch = Unix.in_channel_of_descr client in
          let out_ch = Unix.out_channel_of_descr output_descriptor in
          let request_line =
            try Some (input_line in_ch |> trim_cr) with _ -> None
          in
          (match request_line with
          | None ->
              close_in_noerr in_ch;
              close_out_noerr out_ch
          | Some line ->
              (match String.split_on_char ' ' line with
              | [ "GET"; ("/health" | "/healthz"); _ ] ->
                  ignore (read_headers in_ch);
                  write_response out_ch ~status:"200 OK" ~content_type:"text/plain" "ok\n";
                  close_in_noerr in_ch;
                  close_out_noerr out_ch
              | [ "POST"; "/v1/chat/completions"; _ ] ->
                  (match read_headers in_ch with
                  | Error msg ->
                      write_error out_ch ~status:"400 Bad Request" msg;
                      close_in_noerr in_ch;
                      close_out_noerr out_ch
                  | Ok headers ->
                      (match content_length headers with
                      | Error msg ->
                          write_error out_ch ~status:"411 Length Required" msg;
                          close_in_noerr in_ch;
                          close_out_noerr out_ch
                      | Ok len when len > options.max_body_bytes ->
                          write_error out_ch ~status:"413 Content Too Large"
                            "request body exceeds max-body-bytes";
                          close_in_noerr in_ch;
                          close_out_noerr out_ch
                      | Ok len ->
                          let body = really_input_string in_ch len in
                          (match Openai_protocol.Request.of_string body with
                          | Error msg ->
                              write_error out_ch ~status:"400 Bad Request" msg;
                              close_in_noerr in_ch;
                              close_out_noerr out_ch
                          | Ok req ->
                              service.request_number <- service.request_number + 1;
                              let id_str =
                                Printf.sprintf "chatcmpl-llmopt-%d" service.request_number
                              in
                              let req_id = Serving_queue.Request_id.create () in
                              let model = Openai_protocol.Request.model req in
                              let created = int_of_float (Unix.time ()) in
                              let decoder = Tokenizer.Decoder.create service.tokenizer in
                              let max_new_tokens = Openai_protocol.Request.max_tokens req in
                              let ignore_eos = Openai_protocol.Request.ignore_eos req in
                              (match Lfm_chat.encode (Generation.chat service.generation)
                                       (Openai_protocol.Request.messages req) with
                              | Error err ->
                                  write_error out_ch ~status:"400 Bad Request" err;
                                  close_in_noerr in_ch;
                                  close_out_noerr out_ch
                              | Ok prompt_tokens ->
                                  write_stream_headers out_ch;
                                  let initial_state =
                                    Serving_queue.Pending_prefill {
                                      prompt_tokens;
                                      cached_tokens = 0;
                                      remaining_prefill = Array.length prompt_tokens;
                                      max_new_tokens;
                                      ignore_eos;
                                    }
                                  in
                                  let now = Unix.gettimeofday () in
                                  let initial_score =
                                    Serving_queue.Score.compute ~prefill_rate:100.0
                                      ~decode_rate:10.0 ~current_time:now
                                      ~arrival_time:now initial_state
                                  in
                                  let queue_req : Serving_queue.request = {
                                    id = req_id;
                                    arrival_time = now;
                                    state = initial_state;
                                    priority_score = initial_score;
                                  } in
                                  let active_req : active_request = {
                                    id = req_id;
                                    id_str;
                                    model;
                                    created;
                                    prompt_tokens;
                                    cached_prompt_tokens = 0;
                                    decoder;
                                    out = out_ch;
                                    in_ch;
                                    driver_state = None;
                                    allocated_tokens = 0;
                                  } in
                                  Hashtbl.add active_requests req_id active_req;
                                  Serving_queue.enqueue queue queue_req))));
              | _ ->
                  ignore (read_headers in_ch);
                  write_error out_ch ~status:"404 Not Found" "endpoint not found";
                  close_in_noerr in_ch;
                  close_out_noerr out_ch));
          accept_loop ()
      | exception Unix.Unix_error ((Unix.EWOULDBLOCK | Unix.EAGAIN), _, _) -> ()
      | exception _ -> ()
    in
    accept_loop ()
  in

  let rec step_server_loop () =
    let is_idle = Serving_queue.is_empty queue in
    let timeout = if is_idle then 0.005 else 0.0 in
    (try
       let r_fds, _, _ = Unix.select [ socket ] [] [] timeout in
       if List.mem socket r_fds then accept_new_connections ()
     with _ -> ());

    if not (Serving_queue.is_empty queue) then (
      let now = Unix.gettimeofday () in
      Serving_queue.update_scores queue ~current_time:now;
      let (batch_decodes, prefill_candidate_opt) =
        Serving_queue.pop_next_batch queue ~max_batch_size:8 ~prefill_chunk_budget:512
      in

      (* Execute decode batch FIRST *)
      List.iter
        (fun (req : Serving_queue.request) ->
          match Hashtbl.find_opt active_requests req.id with
          | None -> ()
          | Some active_req ->
              match active_req.driver_state with
              | None -> ()
              | Some driver_state ->
                  let step_res =
                    Generation.Driver.State.step
                      (Generation.engine service.generation)
                      driver_state
                  in
                  (match step_res with
                  | Error err ->
                      Printf.eprintf "stream step error: %s\n%!" err;
                      Serving_queue.release_tokens queue active_req.allocated_tokens;
                      close_active_request active_req;
                      Hashtbl.remove active_requests req.id
                  | Ok (Some token, finish_opt) ->
                      ignore (Serving_queue.reserve_tokens queue 1);
                      active_req.allocated_tokens <- active_req.allocated_tokens + 1;
                      (match emit_token active_req token with
                      | Error err ->
                          Printf.eprintf "stream emit error: %s\n%!" err;
                          Serving_queue.release_tokens queue active_req.allocated_tokens;
                          close_active_request active_req;
                          Hashtbl.remove active_requests req.id
                      | Ok () ->
                          (match finish_opt with
                          | Some finish_reason ->
                              let comp_tokens =
                                Generation.Driver.State.completion_tokens driver_state
                              in
                              emit_finish active_req ~finish_reason
                                ~completion_count:(List.length comp_tokens);
                              Serving_queue.release_tokens queue active_req.allocated_tokens;
                              Hashtbl.remove active_requests req.id
                          | None ->
                              req.state <- Serving_queue.Active_decode {
                                prompt_length = Array.length active_req.prompt_tokens;
                                generated_tokens =
                                  Generation.Driver.State.completion_tokens driver_state;
                                max_new_tokens =
                                  (match req.state with
                                  | Serving_queue.Active_decode d -> d.max_new_tokens
                                  | _ -> 16);
                                ignore_eos =
                                  (match req.state with
                                  | Serving_queue.Active_decode d -> d.ignore_eos
                                  | _ -> false);
                              };
                              Serving_queue.enqueue queue req))
                  | Ok (None, Some finish_reason) ->
                      let comp_tokens =
                        Generation.Driver.State.completion_tokens driver_state
                      in
                      emit_finish active_req ~finish_reason
                        ~completion_count:(List.length comp_tokens);
                      Serving_queue.release_tokens queue active_req.allocated_tokens;
                      Hashtbl.remove active_requests req.id
                  | Ok (None, None) -> ()))
        batch_decodes;

      (* Execute prefill candidate only when no active decodes are running *)
      (match prefill_candidate_opt with
      | None -> ()
      | Some (req, _slice_budget) ->
          if batch_decodes <> [] then
            (* Defer prefill until active decodes finish to prevent TPOT starvation *)
            Serving_queue.enqueue queue req
          else
            (match Hashtbl.find_opt active_requests req.id with
            | None -> ()
            | Some active_req ->
                (match req.state with
                | Serving_queue.Pending_prefill { prompt_tokens; max_new_tokens; ignore_eos; _ } ->
                    let is_stop =
                      if ignore_eos then Fun.const false
                      else Lfm_chat.is_end_token (Generation.chat service.generation)
                    in
                    let config_res = Generation_core.Config.create ~max_new_tokens in
                    (match config_res with
                    | Error err ->
                        Printf.eprintf "stream error: %s\n%!" err;
                        close_active_request active_req;
                        Hashtbl.remove active_requests req.id
                    | Ok config ->
                        let init_res =
                          Generation.Driver.State.init
                            (Generation.engine service.generation)
                            ~config ~is_stop ~prompt:prompt_tokens
                        in
                        (match init_res with
                        | Error err ->
                            Printf.eprintf "stream error: %s\n%!" err;
                            close_active_request active_req;
                            Hashtbl.remove active_requests req.id
                        | Ok (driver_state, first_token) ->
                            active_req.driver_state <- Some driver_state;
                            let cached_count = Generation.Driver.State.cached_prompt_tokens driver_state in
                            let active_req = { active_req with cached_prompt_tokens = cached_count } in
                            Hashtbl.replace active_requests req.id active_req;
                            let alloc_count = Array.length prompt_tokens + 1 in
                            ignore (Serving_queue.reserve_tokens queue alloc_count);
                            active_req.allocated_tokens <- alloc_count;
                            (match emit_token active_req first_token with
                            | Error err ->
                                Printf.eprintf "stream emit error: %s\n%!" err;
                                Serving_queue.release_tokens queue active_req.allocated_tokens;
                                close_active_request active_req;
                                Hashtbl.remove active_requests req.id
                            | Ok () ->
                                if Generation.Driver.State.is_finished driver_state then (
                                  (match Generation.Driver.State.result driver_state with
                                  | Some r ->
                                      let reason = Generation_core.Result.finish_reason r in
                                      let comp_count = Array.length (Generation_core.Result.completion_tokens r) in
                                      emit_finish active_req ~finish_reason:reason ~completion_count:comp_count
                                  | None -> ());
                                  Serving_queue.release_tokens queue active_req.allocated_tokens;
                                  Hashtbl.remove active_requests req.id
                                ) else (
                                  req.state <- Serving_queue.Active_decode {
                                    prompt_length = Array.length prompt_tokens;
                                    generated_tokens = [ first_token ];
                                    max_new_tokens;
                                    ignore_eos;
                                  };
                                  Serving_queue.enqueue queue req
                                ))))
                | _ -> ())))
    );

    step_server_loop ()
  in
  Fun.protect ~finally:(fun () -> Unix.close socket) step_server_loop

let run () =
  let* options, positional = arguments () in
  let* tokenizer, prefill, decode =
    match positional with
    | [ tokenizer; prefill; decode ] -> Ok (tokenizer, prefill, decode)
    | [ engine_directory ] ->
        let tokenizer = Filename.concat engine_directory "tokenizer.llmopt" in
        let prefill = Filename.concat engine_directory "prefill" in
        let decode = Filename.concat engine_directory "decode" in
        if not (Sys.file_exists tokenizer) then
          Error (Printf.sprintf "engine directory missing tokenizer: %s" tokenizer)
        else if not (Sys.file_exists prefill) then
          Error (Printf.sprintf "engine directory missing prefill: %s" prefill)
        else if not (Sys.file_exists decode) then
          Error (Printf.sprintf "engine directory missing decode: %s" decode)
        else Ok (tokenizer, prefill, decode)
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
