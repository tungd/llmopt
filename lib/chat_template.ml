let ( let* ) = Result.bind

module Role = struct
  type t = System | User | Assistant | Tool

  let to_string = function
    | System -> "system"
    | User -> "user"
    | Assistant -> "assistant"
    | Tool -> "tool"
end

module Message = struct
  type t = {
    role : Role.t;
    content : string;
  }

  let create ~role ~content = { role; content }
  let role message = message.role
  let content message = message.content
end

module Int_buffer = struct
  type t = {
    mutable values : int array;
    mutable length : int;
  }

  let create () = { values = Array.make 64 0; length = 0 }

  let reserve buffer additional =
    let required = buffer.length + additional in
    if required > Array.length buffer.values then (
      let capacity = ref (Array.length buffer.values * 2) in
      while !capacity < required do
        capacity := !capacity * 2
      done;
      let values = Array.make !capacity 0 in
      Array.blit buffer.values 0 values 0 buffer.length;
      buffer.values <- values)

  let add buffer value =
    reserve buffer 1;
    buffer.values.(buffer.length) <- value;
    buffer.length <- buffer.length + 1

  let add_array buffer values =
    reserve buffer (Array.length values);
    Array.blit values 0 buffer.values buffer.length (Array.length values);
    buffer.length <- buffer.length + Array.length values

  let contents buffer = Array.sub buffer.values 0 buffer.length
end

type t = {
  tokenizer : Tokenizer.t;
  bos : int;
  message_start : int;
  message_end : int;
}

let create ~bos ~message_start ~message_end tokenizer =
  let validate label id =
    if Tokenizer.has_token_id tokenizer id then Ok ()
    else Error (Printf.sprintf "chat template %s token id %d is absent" label id)
  in
  let* () = validate "BOS" bos in
  let* () = validate "message-start" message_start in
  let* () = validate "message-end" message_end in
  Ok { tokenizer; bos; message_start; message_end }

let end_token template = template.message_end
let is_end_token template token = token = template.message_end

let last_assistant messages =
  let rec find index found = function
    | [] -> found
    | message :: rest ->
        let found =
          match Message.role message with
          | Role.Assistant -> Some index
          | Role.System | Role.User | Role.Tool -> found
        in
        find (index + 1) found rest
  in
  find 0 None messages

let after_last marker text =
  let marker_length = String.length marker in
  let rec search found index =
    if index + marker_length > String.length text then found
    else if String.sub text index marker_length = marker then
      search (Some (index + marker_length)) (index + 1)
    else search found (index + 1)
  in
  match search None 0 with
  | None -> text
  | Some start ->
      String.sub text start (String.length text - start) |> String.trim

let encode ?(add_generation_prompt = true) ?(preserve_thinking = false)
    template messages =
  let output = Int_buffer.create () in
  let append_ordinary text =
    let* ids = Tokenizer.encode template.tokenizer text in
    Int_buffer.add_array output ids;
    Ok ()
  in
  let assistant = last_assistant messages in
  let rec append_messages index = function
    | [] -> Ok ()
    | message :: rest ->
        let skip_empty_initial_system =
          index = 0
          && Message.role message = Role.System
          && String.equal (Message.content message) ""
        in
        if skip_empty_initial_system then append_messages (index + 1) rest
        else
          let content =
            match Message.role message with
            | Role.Assistant
              when not preserve_thinking && assistant <> Some index ->
                after_last "</think>" (Message.content message)
            | Role.System | Role.User | Role.Assistant | Role.Tool ->
                Message.content message
          in
          Int_buffer.add output template.message_start;
          let* () =
            append_ordinary
              (Role.to_string (Message.role message) ^ "\n" ^ content)
          in
          Int_buffer.add output template.message_end;
          let* () = append_ordinary "\n" in
          append_messages (index + 1) rest
  in
  Int_buffer.add output template.bos;
  let* () = append_messages 0 messages in
  if add_generation_prompt then (
    Int_buffer.add output template.message_start;
    let* () = append_ordinary "assistant\n" in
    Ok (Int_buffer.contents output))
  else Ok (Int_buffer.contents output)
