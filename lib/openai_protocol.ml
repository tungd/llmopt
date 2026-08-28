let ( let* ) = Result.bind

let field name fields = List.assoc_opt name fields

let object_ context = function
  | `Assoc fields -> Ok fields
  | _ -> Error (context ^ " must be an object")

let string context = function
  | `String value -> Ok value
  | _ -> Error (context ^ " must be a string")

let bool context = function
  | `Bool value -> Ok value
  | _ -> Error (context ^ " must be a boolean")

let integer context = function
  | `Int value -> Ok value
  | `Intlit value ->
      (match int_of_string_opt value with
      | Some value -> Ok value
      | None -> Error (context ^ " is outside the supported integer range"))
  | _ -> Error (context ^ " must be an integer")

let float_ context = function
  | `Float value -> Ok value
  | `Int value -> Ok (float_of_int value)
  | `Intlit value ->
      (match float_of_string_opt value with
      | Some value -> Ok value
      | None -> Error (context ^ " is outside the supported float range"))
  | _ -> Error (context ^ " must be a number")

let required context name fields decode =
  match field name fields with
  | None -> Error (context ^ " is missing " ^ name)
  | Some value -> decode (context ^ "." ^ name) value

let optional context name fields decode ~default =
  match field name fields with
  | None | Some `Null -> Ok default
  | Some value -> decode (context ^ "." ^ name) value

let role = function
  | "system" -> Ok Lfm_chat.Role.System
  | "user" -> Ok Lfm_chat.Role.User
  | "assistant" -> Ok Lfm_chat.Role.Assistant
  | "tool" -> Ok Lfm_chat.Role.Tool
  | value -> Error (Printf.sprintf "unsupported chat role: %S" value)

module Request = struct
  type t = {
    model : string;
    messages : Lfm_chat.Message.t list;
    max_tokens : int;
    ignore_eos : bool;
    sampling_params : Sampling.Params.t;
  }

  let message index value =
    let context = Printf.sprintf "request.messages[%d]" index in
    let* fields = object_ context value in
    let* role_name = required context "role" fields string in
    let* role = role role_name in
    let* content = required context "content" fields string in
    Ok (Lfm_chat.Message.create ~role ~content)

  let messages context = function
    | `List [] -> Error (context ^ " must not be empty")
    | `List values ->
        let rec decode output index = function
          | [] -> Ok (List.rev output)
          | value :: rest ->
              let* message = message index value in
              decode (message :: output) (index + 1) rest
        in
        decode [] 0 values
    | _ -> Error (context ^ " must be an array")

  let of_json value =
    let* fields = object_ "request" value in
    let* model = required "request" "model" fields string in
    let* messages = required "request" "messages" fields messages in
    let* stream = required "request" "stream" fields bool in
    if not stream then Error "request.stream must be true"
    else
      let* max_tokens = required "request" "max_tokens" fields integer in
      if max_tokens <= 0 then Error "request.max_tokens must be positive"
      else
        let* minimum =
          optional "request" "min_tokens" fields integer ~default:0
        in
        if minimum < 0 || minimum > max_tokens then
          Error "request.min_tokens must be between 0 and max_tokens"
        else
          let* ignore_eos =
            optional "request" "ignore_eos" fields bool ~default:false
          in
          let* temperature =
            optional "request" "temperature" fields float_ ~default:0.0
          in
          let* top_p =
            optional "request" "top_p" fields float_ ~default:1.0
          in
          let* top_k =
            optional "request" "top_k" fields integer ~default:0
          in
          let* min_p =
            optional "request" "min_p" fields float_ ~default:0.0
          in
          let* seed =
            optional "request" "seed" fields integer ~default:0
          in
          let sampling_params =
            Sampling.Params.create ~temperature ~top_k ~top_p ~min_p ~seed ()
          in
          Ok { model; messages; max_tokens; ignore_eos; sampling_params }

  let of_string source =
    let ( let* ) = Result.bind in
    let* json =
      try Ok (Yojson.Safe.from_string source)
      with Yojson.Json_error message ->
        Error ("malformed json request: " ^ message)
    in
    of_json json

  let model request = request.model
  let messages request = request.messages
  let max_tokens request = request.max_tokens
  let ignore_eos request = request.ignore_eos
  let sampling_params request = request.sampling_params
end

module Sse = struct
  let event payload = "data: " ^ Yojson.Safe.to_string payload ^ "\n\n"

  let choice ~delta ~finish_reason =
    `Assoc
      [ ("index", `Int 0); ("delta", delta); ("finish_reason", finish_reason) ]

  let chunk ~id ~model ~created ~choices ?usage () =
    let fields =
      [ ("id", `String id);
        ("object", `String "chat.completion.chunk");
        ("created", `Int created);
        ("model", `String model);
        ("choices", `List choices) ]
    in
    let fields = match usage with None -> fields | Some value -> fields @ [ "usage", value ] in
    event (`Assoc fields)

  let content ~id ~model ~created ~token_id text =
    chunk ~id ~model ~created
      ~choices:
        [ choice
            ~delta:
              (`Assoc
                [ ("content", `String text);
                  ("x_llmopt_token_id", `Int token_id) ])
            ~finish_reason:`Null ]
      ()

  let finish ~id ~model ~created ~reason =
    chunk ~id ~model ~created
      ~choices:
        [ choice ~delta:(`Assoc []) ~finish_reason:(`String reason) ]
      ()

  let usage ~id ~model ~created ~prompt_tokens ~cached_prompt_tokens
      ~completion_tokens =
    let usage =
      `Assoc
        [ ("prompt_tokens", `Int prompt_tokens);
          ("completion_tokens", `Int completion_tokens);
          ("total_tokens", `Int (prompt_tokens + completion_tokens));
          ( "prompt_tokens_details",
            `Assoc [ "cached_tokens", `Int cached_prompt_tokens ] ) ]
    in
    chunk ~id ~model ~created ~choices:[] ~usage ()

  let done_ = "data: [DONE]\n\n"
end

let error_body message =
  `Assoc
    [ ( "error",
        `Assoc
          [ ("message", `String message);
            ("type", `String "invalid_request_error") ] ) ]
  |> Yojson.Safe.to_string
