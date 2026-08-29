let ( let* ) = Result.bind

let usage () =
  prerr_endline
    "usage: llmopt-tokenize encode [--bos-token-id <id>] <tokenizer.llmopt>\n\
     \       llmopt-tokenize decode [--keep-special] <tokenizer.llmopt>\n\
     \       llmopt-tokenize chat --bos-token-id <id> \
     --message-start-token-id <id> --message-end-token-id <id> \
     <tokenizer.llmopt> \
     <role> <content> [<role> <content> ...]";
  exit 64

let read_stdin () =
  let buffer = Buffer.create 256 in
  let bytes = Bytes.create 4096 in
  let rec read () =
    match input stdin bytes 0 (Bytes.length bytes) with
    | 0 -> Buffer.contents buffer
    | count ->
        Buffer.add_subbytes buffer bytes 0 count;
        read ()
  in
  read ()

let print_ids ids =
  Array.iteri
    (fun index id ->
      if index > 0 then output_char stdout ',';
      output_string stdout (string_of_int id))
    ids;
  output_char stdout '\n'

let ids_of_string input =
  let input = String.trim input in
  if String.equal input "" then Ok [||]
  else
    let values = String.split_on_char ',' input in
    let rec parse output = function
      | [] -> Ok (Array.of_list (List.rev output))
      | value :: rest ->
          (match int_of_string_opt (String.trim value) with
          | Some id when id >= 0 -> parse (id :: output) rest
          | _ -> Error (Printf.sprintf "invalid token id: %S" value))
  in
  parse [] values

let nonnegative name value =
  match int_of_string_opt value with
  | Some value when value >= 0 -> Ok value
  | _ -> Error (name ^ " must be a nonnegative integer")

let role = function
  | "system" -> Ok Chat_template.Role.System
  | "user" -> Ok Chat_template.Role.User
  | "assistant" -> Ok Chat_template.Role.Assistant
  | "tool" -> Ok Chat_template.Role.Tool
  | value -> Error (Printf.sprintf "unsupported chat role: %S" value)

let messages arguments =
  let rec parse output = function
    | [] -> Ok (List.rev output)
    | role_name :: content :: rest ->
        Result.bind (role role_name) (fun role ->
            parse (Chat_template.Message.create ~role ~content :: output) rest)
    | [ _ ] -> Error "chat messages require role/content pairs"
  in
  parse [] arguments

let run () =
  match Array.to_list Sys.argv with
  | [ _; "encode"; archive ] ->
      let* tokenizer = Tokenizer.of_file archive in
      let* ids = Tokenizer.encode tokenizer (read_stdin ()) in
      print_ids ids;
      Ok ()
  | [ _; "encode"; "--bos-token-id"; raw_bos; archive ] ->
      let* bos_token_id = nonnegative "bos-token-id" raw_bos in
      let* tokenizer = Tokenizer.of_file archive in
      let* ids = Tokenizer.encode ~bos_token_id tokenizer (read_stdin ()) in
      print_ids ids;
      Ok ()
  | [ _; "decode"; archive ] ->
      let* tokenizer = Tokenizer.of_file archive in
      let* ids = ids_of_string (read_stdin ()) in
      let* text = Tokenizer.decode tokenizer ids in
      print_string text;
      Ok ()
  | [ _; "decode"; "--keep-special"; archive ] ->
      let* tokenizer = Tokenizer.of_file archive in
      let* ids = ids_of_string (read_stdin ()) in
      let* text = Tokenizer.decode ~skip_special:false tokenizer ids in
      print_string text;
      Ok ()
  | _ :: "chat" :: "--bos-token-id" :: raw_bos
    :: "--message-start-token-id" :: raw_start
    :: "--message-end-token-id" :: raw_end :: archive :: arguments ->
      let* bos = nonnegative "bos-token-id" raw_bos in
      let* message_start =
        nonnegative "message-start-token-id" raw_start
      in
      let* message_end = nonnegative "message-end-token-id" raw_end in
      let* tokenizer = Tokenizer.of_file archive in
      let* template =
        Chat_template.create ~bos ~message_start ~message_end tokenizer
      in
      let* messages = messages arguments in
      let* ids = Chat_template.encode template messages in
      print_ids ids;
      Ok ()
  | _ -> usage ()

let () =
  match run () with
  | Ok () -> ()
  | Error message ->
      prerr_endline message;
      exit 2
