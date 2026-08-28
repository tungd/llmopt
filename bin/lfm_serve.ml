let ( let* ) = Result.bind

type options = {
  host : string;
  port : int;
  token_capacity : int;
  checkpoint_capacity : int;
  max_body_bytes : int;
}

let defaults =
  {
    host = "127.0.0.1";
    port = 8000;
    token_capacity = 8_192;
    checkpoint_capacity = 1_024;
    max_body_bytes = 64 * 1_024 * 1_024;
  }

let usage () =
  prerr_endline
    "usage: llmopt-serve [--host address] [--port number] \
     [--token-capacity count] [--checkpoint-capacity count] \
     [--max-body-bytes count] <engine-directory | tokenizer.llmopt prefill-dir decode-dir>";
  exit 64

let positive name value =
  match int_of_string_opt value with
  | Some value when value > 0 -> Ok value
  | _ -> Error (name ^ " must be a positive integer")

let arguments () =
  let rec parse options positional = function
    | [] -> Ok (options, List.rev positional)
    | "--host" :: value :: rest ->
        parse { options with host = value } positional rest
    | "--port" :: value :: rest ->
        let* port = positive "port" value in
        if port > 65_535 then Error "port must not exceed 65535"
        else parse { options with port } positional rest
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
      ~token_capacity:options.token_capacity
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

type active_request = {
  id : Serving_queue.Request_id.t;
  id_str : string;
  model : string;
  created : int;
  prompt_tokens : int array;
  mutable cached_prompt_tokens : int;
  decoder : Tokenizer.Decoder.t;
  stream : Webs_iomux.Stream.t;
  mutable driver_state : Generation.Driver.State.t option;
  mutable allocated_tokens : int;
}

let emit_token active_req token =
  match Tokenizer.Decoder.push active_req.decoder token with
  | Error message -> Error message
  | Ok text ->
      let chunk =
        Openai_protocol.Sse.content ~id:active_req.id_str ~model:active_req.model
          ~created:active_req.created ~token_id:token text
      in
      ignore (Webs_iomux.Stream.push active_req.stream chunk);
      Ok ()

let emit_finish active_req ~finish_reason ~completion_count =
  let prompt_tokens = Array.length active_req.prompt_tokens in
  let reason = Generation_core.Finish_reason.to_string finish_reason in
  let finish_chunk =
    Openai_protocol.Sse.finish ~id:active_req.id_str ~model:active_req.model
      ~created:active_req.created ~reason
  in
  let usage_chunk =
    Openai_protocol.Sse.usage ~id:active_req.id_str ~model:active_req.model
      ~created:active_req.created ~prompt_tokens
      ~cached_prompt_tokens:active_req.cached_prompt_tokens
      ~completion_tokens:completion_count
  in
  ignore (Webs_iomux.Stream.push active_req.stream finish_chunk);
  ignore (Webs_iomux.Stream.push active_req.stream usage_chunk);
  ignore (Webs_iomux.Stream.push active_req.stream Openai_protocol.Sse.done_);
  Webs_iomux.Stream.close active_req.stream

let close_active_request active_req =
  Webs_iomux.Stream.close active_req.stream

let json_response status body_str =
  let headers =
    Webs.Http.Headers.add_value
      (Webs.Http.Headers.Name.v "content-type")
      "application/json"
      Webs.Http.Headers.empty
  in
  Webs.Http.Response.make ~headers status (Webs.Http.Body.of_string body_str)

let serve service options =
  let queue =
    Serving_queue.create ~token_capacity:options.token_capacity
      ~high_watermark_ratio:0.90 ~low_watermark_ratio:0.75 ()
  in
  let active_requests : (Serving_queue.Request_id.t, active_request) Hashtbl.t =
    Hashtbl.create 32
  in
  let mutex = Mutex.create () in
  let cond = Condition.create () in

  let handler req =
    let path = Webs.Http.Request.path req in
    let meth = Webs.Http.Request.method' req in
    match meth, path with
    | `GET, ([ "health" ] | [ "healthz" ]) ->
        Webs.Http.Response.make Webs.Http.Status.ok_200 (Webs.Http.Body.of_string "ok\n")
    | `POST, [ "v1"; "chat"; "completions" ] ->
        let body_res = Webs.Http.Body.to_string (Webs.Http.Request.body req) in
        (match body_res with
        | Error _ ->
            json_response Webs.Http.Status.bad_request_400
              (Openai_protocol.error_body "invalid body")
        | Ok body ->
            (match Openai_protocol.Request.of_string body with
            | Error msg ->
                json_response Webs.Http.Status.bad_request_400
                  (Openai_protocol.error_body msg)
            | Ok parsed_req ->
                Mutex.lock mutex;
                service.request_number <- service.request_number + 1;
                let id_str =
                  Printf.sprintf "chatcmpl-llmopt-%d" service.request_number
                in
                Mutex.unlock mutex;
                let req_id = Serving_queue.Request_id.create () in
                let model = Openai_protocol.Request.model parsed_req in
                let created = int_of_float (Unix.time ()) in
                let decoder = Tokenizer.Decoder.create service.tokenizer in
                let max_new_tokens = Openai_protocol.Request.max_tokens parsed_req in
                let ignore_eos = Openai_protocol.Request.ignore_eos parsed_req in
                let sampling_params =
                  Openai_protocol.Request.sampling_params parsed_req
                in
                (match Lfm_chat.encode (Generation.chat service.generation)
                         (Openai_protocol.Request.messages parsed_req) with
                | Error err ->
                    json_response Webs.Http.Status.bad_request_400
                      (Openai_protocol.error_body err)
                | Ok prompt_tokens ->
                    let stream = Webs_iomux.Stream.create () in
                    let initial_state =
                      Serving_queue.Pending_prefill {
                        prompt_tokens;
                        cached_tokens = 0;
                        remaining_prefill = Array.length prompt_tokens;
                        max_new_tokens;
                        ignore_eos;
                        sampling_params;
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
                      stream;
                      driver_state = None;
                      allocated_tokens = 0;
                    } in
                    Mutex.lock mutex;
                    Hashtbl.add active_requests req_id active_req;
                    Serving_queue.enqueue queue queue_req;
                    Condition.signal cond;
                    Mutex.unlock mutex;
                    Webs_iomux.Stream.response stream)))
    | _ ->
        json_response Webs.Http.Status.not_found_404
          (Openai_protocol.error_body "endpoint not found")
  in

  let* server =
    Webs_iomux.start ~host:options.host ~port:options.port
      ~max_body_bytes:options.max_body_bytes handler
  in
  let hw = Target_hardware.discover () in
  let cost_model =
    Target_hardware.Prefill_cost_model.analyze ~target:hw.execution
      ~weight_bytes:175_000_000 ~active_params:350_000_000 ()
  in
  let prefill_chunk_budget = cost_model.optimal_chunk_size in
  Printf.eprintf "prefill cost model: knee=%d sat=%d chunk_budget=%d buckets=[%s] est_latency=%.2fms\n%!"
    cost_model.roofline_knee_tokens cost_model.core_saturation_tokens
    prefill_chunk_budget
    (String.concat ";" (List.map string_of_int cost_model.template_buckets))
    cost_model.predicted_chunk_latency_ms;

  let rec step_engine_loop () =
    Mutex.lock mutex;
    while Serving_queue.is_empty queue do
      Condition.wait cond mutex
    done;
    let now = Unix.gettimeofday () in
    Serving_queue.update_scores queue ~current_time:now;
    let (batch_decodes, prefill_candidate_opt) =
      Serving_queue.pop_next_batch queue ~max_batch_size:8 ~prefill_chunk_budget
    in
    Mutex.unlock mutex;

    (* Execute decode batch FIRST *)
    List.iter
      (fun (req : Serving_queue.request) ->
        Mutex.lock mutex;
        let active_opt = Hashtbl.find_opt active_requests req.id in
        Mutex.unlock mutex;
        match active_opt with
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
                    close_active_request active_req;
                    Mutex.lock mutex;
                    Serving_queue.release_tokens queue active_req.allocated_tokens;
                    Hashtbl.remove active_requests req.id;
                    Mutex.unlock mutex
                | Ok (Some token, finish_opt) ->
                    Mutex.lock mutex;
                    ignore (Serving_queue.reserve_tokens queue 1);
                    active_req.allocated_tokens <- active_req.allocated_tokens + 1;
                    Mutex.unlock mutex;
                    (match emit_token active_req token with
                    | Error err ->
                        Printf.eprintf "stream emit error: %s\n%!" err;
                        close_active_request active_req;
                        Mutex.lock mutex;
                        Serving_queue.release_tokens queue active_req.allocated_tokens;
                        Hashtbl.remove active_requests req.id;
                        Mutex.unlock mutex
                    | Ok () ->
                        (match finish_opt with
                        | Some finish_reason ->
                            let comp_tokens =
                              Generation.Driver.State.completion_tokens driver_state
                            in
                            emit_finish active_req ~finish_reason
                              ~completion_count:(List.length comp_tokens);
                            Mutex.lock mutex;
                            Serving_queue.release_tokens queue active_req.allocated_tokens;
                            Hashtbl.remove active_requests req.id;
                            Mutex.unlock mutex
                        | None ->
                            let sampling_params =
                              match req.state with
                              | Serving_queue.Active_decode d -> d.sampling_params
                              | Serving_queue.Pending_prefill p -> p.sampling_params
                            in
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
                              sampling_params;
                            };
                            Mutex.lock mutex;
                            Serving_queue.enqueue queue req;
                            Mutex.unlock mutex))
                | Ok (None, Some finish_reason) ->
                    let comp_tokens =
                      Generation.Driver.State.completion_tokens driver_state
                    in
                    emit_finish active_req ~finish_reason
                      ~completion_count:(List.length comp_tokens);
                    Mutex.lock mutex;
                    Serving_queue.release_tokens queue active_req.allocated_tokens;
                    Hashtbl.remove active_requests req.id;
                    Mutex.unlock mutex
                | Ok (None, None) -> ()))
      batch_decodes;

    (* Execute prefill candidate *)
    (match prefill_candidate_opt with
    | None -> ()
    | Some (req, _slice_budget) ->
        if batch_decodes <> [] then (
          Mutex.lock mutex;
          Serving_queue.enqueue queue req;
          Mutex.unlock mutex
        ) else (
          Mutex.lock mutex;
          let active_opt = Hashtbl.find_opt active_requests req.id in
          Mutex.unlock mutex;
          match active_opt with
          | None -> ()
          | Some active_req ->
              match req.state with
              | Serving_queue.Pending_prefill { prompt_tokens; max_new_tokens; ignore_eos; sampling_params; _ } ->
                  let is_stop =
                    if ignore_eos then Fun.const false
                    else Lfm_chat.is_end_token (Generation.chat service.generation)
                  in
                  let config_res = Generation_core.Config.create ~max_new_tokens in
                  (match config_res with
                  | Error err ->
                      Printf.eprintf "stream error: %s\n%!" err;
                      close_active_request active_req;
                      Mutex.lock mutex;
                      Serving_queue.release_tokens queue active_req.allocated_tokens;
                      Hashtbl.remove active_requests req.id;
                      Mutex.unlock mutex
                  | Ok config ->
                      let init_res =
                        Generation.Driver.State.init
                          (Generation.engine service.generation)
                          ~config ~sampling_params ~is_stop ~prompt:prompt_tokens
                          ()
                      in
                      (match init_res with
                      | Error err ->
                          Printf.eprintf "stream error: %s\n%!" err;
                          close_active_request active_req;
                          Mutex.lock mutex;
                          Serving_queue.release_tokens queue active_req.allocated_tokens;
                          Hashtbl.remove active_requests req.id;
                          Mutex.unlock mutex
                      | Ok (driver_state, first_token) ->
                          active_req.driver_state <- Some driver_state;
                          let cached_count =
                            Generation.Driver.State.cached_prompt_tokens driver_state
                          in
                          active_req.cached_prompt_tokens <- cached_count;
                          let alloc_count = Array.length prompt_tokens + 1 in
                          Mutex.lock mutex;
                          ignore (Serving_queue.reserve_tokens queue alloc_count);
                          active_req.allocated_tokens <- alloc_count;
                          Mutex.unlock mutex;
                          (match emit_token active_req first_token with
                          | Error err ->
                              Printf.eprintf "stream emit error: %s\n%!" err;
                              close_active_request active_req;
                              Mutex.lock mutex;
                              Serving_queue.release_tokens queue active_req.allocated_tokens;
                              Hashtbl.remove active_requests req.id;
                              Mutex.unlock mutex
                          | Ok () ->
                              if Generation.Driver.State.is_finished driver_state then (
                                (match Generation.Driver.State.result driver_state with
                                | Some r ->
                                    let reason = Generation_core.Result.finish_reason r in
                                    let comp_count =
                                      Array.length (Generation_core.Result.completion_tokens r)
                                    in
                                    emit_finish active_req ~finish_reason:reason
                                      ~completion_count:comp_count
                                | None -> ());
                                Mutex.lock mutex;
                                Serving_queue.release_tokens queue active_req.allocated_tokens;
                                Hashtbl.remove active_requests req.id;
                                Mutex.unlock mutex
                              ) else (
                                req.state <- Serving_queue.Active_decode {
                                  prompt_length = Array.length prompt_tokens;
                                  generated_tokens = [ first_token ];
                                  max_new_tokens;
                                  ignore_eos;
                                  sampling_params;
                                };
                                Mutex.lock mutex;
                                Serving_queue.enqueue queue req;
                                Mutex.unlock mutex
                              ))))
              | _ -> ()));
    step_engine_loop ()
  in
  step_engine_loop ()

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
    (Kv_cache.Format.to_string Kv_cache.Format.default);
  Sys.set_signal Sys.sigpipe Sys.Signal_ignore;
  serve service options

let () =
  match run () with
  | Ok () -> ()
  | Error message ->
      prerr_endline message;
      exit 2
