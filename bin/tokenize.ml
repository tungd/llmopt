let usage () =
  prerr_endline
    "usage: llmopt-tokenize encode [--no-bos] <tokenizer.llmopt>\n\
     \       llmopt-tokenize decode [--keep-special] <tokenizer.llmopt>\n\
     \       llmopt-tokenize chat <tokenizer.llmopt> \
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

let role = function
  | "system" -> Ok Lfm_chat.Role.System
  | "user" -> Ok Lfm_chat.Role.User
  | "assistant" -> Ok Lfm_chat.Role.Assistant
  | "tool" -> Ok Lfm_chat.Role.Tool
  | value -> Error (Printf.sprintf "unsupported chat role: %S" value)

let messages arguments =
  let rec parse output = function
    | [] -> Ok (List.rev output)
    | role_name :: content :: rest ->
        Result.bind (role role_name) (fun role ->
            parse (Lfm_chat.Message.create ~role ~content :: output) rest)
    | [ _ ] -> Error "chat messages require role/content pairs"
  in
  parse [] arguments

let run () =
  match Array.to_list Sys.argv with
  | [ _; "encode"; archive ] ->
      Result.bind (Tokenizer.of_file archive) (fun tokenizer ->
          Result.map print_ids (Tokenizer.encode tokenizer (read_stdin ())))
  | [ _; "encode"; "--no-bos"; archive ] ->
      Result.bind (Tokenizer.of_file archive) (fun tokenizer ->
          Result.map print_ids
            (Tokenizer.encode ~add_bos:false tokenizer (read_stdin ())))
  | [ _; "decode"; archive ] ->
      Result.bind (Tokenizer.of_file archive) (fun tokenizer ->
          Result.bind (ids_of_string (read_stdin ())) (fun ids ->
              Result.map print_string (Tokenizer.decode tokenizer ids)))
  | [ _; "decode"; "--keep-special"; archive ] ->
      Result.bind (Tokenizer.of_file archive) (fun tokenizer ->
          Result.bind (ids_of_string (read_stdin ())) (fun ids ->
              Result.map print_string
                (Tokenizer.decode ~skip_special:false tokenizer ids)))
  | _ :: "chat" :: archive :: arguments ->
      Result.bind (Tokenizer.of_file archive) (fun tokenizer ->
          Result.bind (Lfm_chat.create tokenizer) (fun template ->
              Result.bind (messages arguments) (fun messages ->
                  Result.map print_ids (Lfm_chat.encode template messages))))
  | _ -> usage ()

let () =
  match run () with
  | Ok () -> ()
  | Error message ->
      prerr_endline message;
      exit 2
