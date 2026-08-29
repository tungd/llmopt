let ( let* ) = Result.bind

let binary_magic = "LLMOPTTK"

module Pair = struct
  type t = string * string

  let equal (left_a, right_a) (left_b, right_b) =
    String.equal left_a left_b && String.equal right_a right_b

  let hash = Hashtbl.hash
end

module Pair_table = Hashtbl.Make (Pair)

type token = {
  text : string;
  added : bool;
  special : bool;
}

type trie = {
  mutable terminal : int option;
  children : (char, trie) Hashtbl.t;
}

type t = {
  by_text : (string, int) Hashtbl.t;
  by_id : token option array;
  merges : int Pair_table.t;
  added : trie;
}

let trie () = { terminal = None; children = Hashtbl.create 4 }

let insert_added root text id =
  let rec insert node index =
    if index = String.length text then node.terminal <- Some id
    else
      let byte = text.[index] in
      let child =
        match Hashtbl.find_opt node.children byte with
        | Some child -> child
        | None ->
            let child = trie () in
            Hashtbl.add node.children byte child;
            child
      in
      insert child (index + 1)
  in
  insert root 0

let longest_added root text start =
  let rec walk node index best =
    let best =
      match node.terminal with None -> best | Some id -> Some (index, id)
    in
    if index = String.length text then best
    else
      match Hashtbl.find_opt node.children text.[index] with
      | None -> best
      | Some child -> walk child (index + 1) best
  in
  walk root start None

let of_bytes bytes =
  let reader = Binary.Reader.create bytes in
  let* magic = Binary.Reader.raw_string reader ~length:8 in
  if magic <> binary_magic then Error "invalid tokenizer archive magic"
  else
    let* version = Binary.Reader.u16 reader in
    if version <> 1 then
      Error (Printf.sprintf "unsupported tokenizer archive version: %d" version)
    else
      let* profile = Binary.Reader.u16 reader in
      if profile <> 1 then
        Error (Printf.sprintf "unsupported tokenizer profile: %d" profile)
      else
        let* token_count = Binary.Reader.u32 reader in
        let* merge_count = Binary.Reader.u32 reader in
        let* maximum_id = Binary.Reader.u32 reader in
        if token_count = 0 then Error "tokenizer archive has no tokens"
        else if maximum_id = max_int then
          Error "tokenizer maximum id exceeds array capacity"
        else
          let by_text = Hashtbl.create token_count in
          let by_id = Array.make (maximum_id + 1) None in
          let added = trie () in
          let rec tokens remaining =
            if remaining = 0 then Ok ()
            else
              let* id = Binary.Reader.u32 reader in
              let* flags = Binary.Reader.u8 reader in
              let* padding = Binary.Reader.raw_bytes reader ~length:3 in
              let* length = Binary.Reader.u32 reader in
              let* text = Binary.Reader.raw_string reader ~length in
              if id > maximum_id then Error "tokenizer token id exceeds maximum"
              else if flags land lnot 0x03 <> 0 then
                Error "tokenizer token has unknown flags"
              else if padding <> Bytes.make 3 '\000' then
                Error "tokenizer token padding is nonzero"
              else if text = "" then Error "tokenizer token text is empty"
              else if Option.is_some by_id.(id) then
                Error (Printf.sprintf "duplicate tokenizer token id: %d" id)
              else if Hashtbl.mem by_text text then
                Error ("duplicate tokenizer token text: " ^ text)
              else
                let token =
                  {
                    text;
                    added = flags land 0x01 <> 0;
                    special = flags land 0x02 <> 0;
                  }
                in
                by_id.(id) <- Some token;
                Hashtbl.add by_text text id;
                if token.added then insert_added added text id;
                tokens (remaining - 1)
          in
          let* () = tokens token_count in
          let merges = Pair_table.create merge_count in
          let rec merge rank =
            if rank = merge_count then Ok ()
            else
              let* left_length = Binary.Reader.u32 reader in
              let* right_length = Binary.Reader.u32 reader in
              let* left = Binary.Reader.raw_string reader ~length:left_length in
              let* right = Binary.Reader.raw_string reader ~length:right_length in
              if left = "" || right = "" then
                Error "tokenizer merge contains an empty symbol"
              else if Pair_table.mem merges (left, right) then
                Error "tokenizer archive contains a duplicate merge"
              else if not (Hashtbl.mem by_text (left ^ right)) then
                Error "tokenizer merge result is absent from vocabulary"
              else (
                Pair_table.add merges (left, right) rank;
                merge (rank + 1))
          in
          let* () = merge 0 in
          let* () = Binary.Reader.finish reader in
          Ok { by_text; by_id; merges; added }

let of_file path =
  try
    let channel = open_in_bin path in
    Fun.protect
      ~finally:(fun () -> close_in_noerr channel)
      (fun () ->
        let length = in_channel_length channel in
        let bytes = Bytes.create length in
        really_input channel bytes 0 length;
        of_bytes bytes)
  with Sys_error message -> Error message

let token_to_id tokenizer token = Hashtbl.find_opt tokenizer.by_text token

let token tokenizer id =
  if id < 0 || id >= Array.length tokenizer.by_id then None
  else tokenizer.by_id.(id)

let is_special tokenizer id =
  match token tokenizer id with Some token -> token.special | None -> false

let has_token_id tokenizer id = Option.is_some (token tokenizer id)

type rune = {
  value : Uchar.t;
  first : int;
  last : int;
}

let runes text =
  let decoder = Uutf.decoder ~encoding:`UTF_8 (`String text) in
  let rec decode acc =
    let first = Uutf.decoder_byte_count decoder in
    match Uutf.decode decoder with
    | `Uchar value ->
        let last = Uutf.decoder_byte_count decoder in
        decode ({ value; first; last } :: acc)
    | `End -> Ok (Array.of_list (List.rev acc))
    | `Malformed bytes ->
        Error
          (Printf.sprintf "tokenizer input has malformed UTF-8 at byte %d: %S"
             first bytes)
    | `Await -> assert false
  in
  decode []

let code rune = Uchar.to_int rune.value
let is_crlf rune = code rune = 0x0a || code rune = 0x0d

let is_letter rune =
  match Uucp.Gc.general_category rune.value with
  | `Ll | `Lm | `Lo | `Lt | `Lu -> true
  | _ -> false

let is_number rune =
  match Uucp.Gc.general_category rune.value with
  | `Nd | `Nl | `No -> true
  | _ -> false

let is_space rune = Uucp.White.is_white_space rune.value

let ascii_lower rune =
  let value = code rune in
  if value >= Char.code 'A' && value <= Char.code 'Z' then value + 32 else value

let contraction runes start =
  if code runes.(start) <> Char.code '\'' then None
  else
    let suffixes = [ "s"; "t"; "re"; "ve"; "m"; "ll"; "d" ] in
    List.find_map
      (fun suffix ->
        let length = String.length suffix in
        if start + 1 + length > Array.length runes then None
        else
          let rec equal index =
            index = length
            ||
            (ascii_lower runes.(start + 1 + index) = Char.code suffix.[index]
            && equal (index + 1))
          in
          if equal 0 then Some (start + 1 + length) else None)
      suffixes

let pretokenize text =
  let* runes = runes text in
  let count = Array.length runes in
  let slice first last =
    let byte_first = runes.(first).first in
    let byte_last = runes.(last - 1).last in
    String.sub text byte_first (byte_last - byte_first)
  in
  let rec while_predicate predicate index =
    if index < count && predicate runes.(index) then
      while_predicate predicate (index + 1)
    else index
  in
  let rec scan pieces index =
    if index = count then Ok (List.rev pieces)
    else
      match contraction runes index with
      | Some next -> scan (slice index next :: pieces) next
      | None ->
          let letter_start =
            if is_letter runes.(index) then Some index
            else if
              not (is_crlf runes.(index)) && not (is_letter runes.(index))
              && not (is_number runes.(index)) && index + 1 < count
              && is_letter runes.(index + 1)
            then Some (index + 1)
            else None
          in
          (match letter_start with
          | Some start ->
              let next = while_predicate is_letter (start + 1) in
              scan (slice index next :: pieces) next
          | None when is_number runes.(index) ->
              let rec digits next remaining =
                if
                  remaining > 0 && next < count && is_number runes.(next)
                then digits (next + 1) (remaining - 1)
                else next
              in
              let next = digits (index + 1) 2 in
              scan (slice index next :: pieces) next
          | None ->
              let punctuation_start =
                if
                  code runes.(index) = Char.code ' ' && index + 1 < count
                  && not (is_space runes.(index + 1))
                  && not (is_letter runes.(index + 1))
                  && not (is_number runes.(index + 1))
                then Some (index + 1)
                else if
                  not (is_space runes.(index))
                  && not (is_letter runes.(index))
                  && not (is_number runes.(index))
                then Some index
                else None
              in
              (match punctuation_start with
              | Some start ->
                  let next =
                    while_predicate
                      (fun rune ->
                        not (is_space rune) && not (is_letter rune)
                        && not (is_number rune))
                      (start + 1)
                  in
                  let next = while_predicate is_crlf next in
                  scan (slice index next :: pieces) next
              | None when is_space runes.(index) ->
                  let run_end = while_predicate is_space (index + 1) in
                  let rec last_newline candidate cursor =
                    if cursor = run_end then candidate
                    else
                      last_newline
                        (if is_crlf runes.(cursor) then Some cursor else candidate)
                        (cursor + 1)
                  in
                  (match last_newline None index with
                  | Some newline ->
                      let next = newline + 1 in
                      scan (slice index next :: pieces) next
                  | None ->
                      let next =
                        if run_end < count && run_end - index > 1 then run_end - 1
                        else run_end
                      in
                      scan (slice index next :: pieces) next)
              | None -> Error "tokenizer pre-tokenizer made no progress"))
  in
  if count = 0 then Ok [] else scan [] 0

let byte_symbols =
  let direct byte =
    (byte >= Char.code '!' && byte <= Char.code '~')
    || (byte >= 0xa1 && byte <= 0xac)
    || (byte >= 0xae && byte <= 0xff)
  in
  let next = ref 0 in
  Array.init 256 (fun byte ->
      let scalar =
        if direct byte then byte
        else
          let scalar = 256 + !next in
          incr next;
          scalar
      in
      let buffer = Buffer.create 2 in
      Uutf.Buffer.add_utf_8 buffer (Uchar.of_int scalar);
      Buffer.contents buffer)

let inverse_bytes =
  let table = Hashtbl.create 256 in
  Array.iteri
    (fun byte symbol ->
      let decoder = Uutf.decoder ~encoding:`UTF_8 (`String symbol) in
      match Uutf.decode decoder with
      | `Uchar value -> Hashtbl.add table (Uchar.to_int value) byte
      | _ -> assert false)
    byte_symbols;
  table

let symbols piece =
  List.init (String.length piece) (fun index -> byte_symbols.(Char.code piece.[index]))

let best_pair merges = function
  | [] | [ _ ] -> None
  | first :: rest ->
      let rec find best left = function
        | [] -> best
        | right :: tail ->
            let best =
              match Pair_table.find_opt merges (left, right), best with
              | None, _ -> best
              | Some rank, None -> Some (rank, left, right)
              | Some rank, Some (best_rank, _, _) when rank < best_rank ->
                  Some (rank, left, right)
              | Some _, Some _ -> best
            in
            find best right tail
      in
      find None first rest

let merge_pair left right symbols =
  let rec merge output = function
    | first :: second :: rest
      when String.equal first left && String.equal second right ->
        merge ((first ^ second) :: output) rest
    | first :: rest -> merge (first :: output) rest
    | [] -> List.rev output
  in
  merge [] symbols

let rec bpe merges symbols =
  match best_pair merges symbols with
  | None -> symbols
  | Some (_, left, right) -> bpe merges (merge_pair left right symbols)

let encode_piece tokenizer piece =
  let rec ids output = function
    | [] -> Ok (List.rev output)
    | symbol :: rest ->
        (match Hashtbl.find_opt tokenizer.by_text symbol with
        | None -> Error ("tokenizer vocabulary has no symbol: " ^ symbol)
        | Some id -> ids (id :: output) rest)
  in
  ids [] (bpe tokenizer.merges (symbols piece))

let encode_ordinary tokenizer text =
  let* pieces = pretokenize text in
  let rec encode output = function
    | [] -> Ok (List.rev output |> List.concat)
    | piece :: rest ->
        let* ids = encode_piece tokenizer piece in
        encode (ids :: output) rest
  in
  encode [] pieces

let encode ?bos_token_id tokenizer text =
  let* bos =
    match bos_token_id with
    | None -> Ok []
    | Some id when has_token_id tokenizer id -> Ok [ id ]
    | Some id -> Error (Printf.sprintf "tokenizer has no BOS token id %d" id)
  in
  let rec split output ordinary_start cursor =
    if cursor = String.length text then
      let ordinary = String.sub text ordinary_start (cursor - ordinary_start) in
      let* ids = encode_ordinary tokenizer ordinary in
      Ok (List.rev_append ids output |> List.rev)
    else
      match longest_added tokenizer.added text cursor with
      | None -> split output ordinary_start (cursor + 1)
      | Some (next, id) ->
          let ordinary = String.sub text ordinary_start (cursor - ordinary_start) in
          let* ids = encode_ordinary tokenizer ordinary in
          let output = id :: List.rev_append ids output in
          split output next next
  in
  let* ids = split [] 0 0 in
  Ok (Array.of_list (bos @ ids))

let decode ?(skip_special = true) tokenizer ids =
  let buffer = Buffer.create (Array.length ids * 4) in
  let rec token_text index =
    if index = Array.length ids then Ok ()
    else
      match token tokenizer ids.(index) with
      | None -> Error (Printf.sprintf "unknown tokenizer id: %d" ids.(index))
      | Some token when skip_special && token.special -> token_text (index + 1)
      | Some token ->
          let decoder = Uutf.decoder ~encoding:`UTF_8 (`String token.text) in
          let rec bytes () =
            match Uutf.decode decoder with
            | `Uchar value ->
                (match Hashtbl.find_opt inverse_bytes (Uchar.to_int value) with
                | None ->
                    Error
                      (Printf.sprintf
                         "token %d contains a non-byte-level scalar U+%04X"
                         ids.(index) (Uchar.to_int value))
                | Some byte ->
                    Buffer.add_char buffer (Char.chr byte);
                    bytes ())
            | `End -> token_text (index + 1)
            | `Malformed _ ->
                Error (Printf.sprintf "token %d contains malformed UTF-8" ids.(index))
            | `Await -> assert false
          in
          bytes ()
  in
  let* () = token_text 0 in
  let result = Buffer.contents buffer in
  let* _ = runes result in
  Ok result

module Decoder = struct
  type tokenizer = t

  type t = {
    tokenizer : tokenizer;
    skip_special : bool;
    mutable pending : string;
    mutable finished : bool;
  }

  let create ?(skip_special = true) tokenizer =
    { tokenizer; skip_special; pending = ""; finished = false }

  let token_bytes decoder id =
    match token decoder.tokenizer id with
    | None -> Error (Printf.sprintf "unknown tokenizer id: %d" id)
    | Some token when decoder.skip_special && token.special -> Ok ""
    | Some token ->
        let output = Buffer.create (String.length token.text) in
        let source = Uutf.decoder ~encoding:`UTF_8 (`String token.text) in
        let rec bytes () =
          match Uutf.decode source with
          | `Uchar value ->
              (match Hashtbl.find_opt inverse_bytes (Uchar.to_int value) with
              | None ->
                  Error
                    (Printf.sprintf
                       "token %d contains a non-byte-level scalar U+%04X" id
                       (Uchar.to_int value))
              | Some byte ->
                  Buffer.add_char output (Char.chr byte);
                  bytes ())
          | `End -> Ok (Buffer.contents output)
          | `Malformed _ ->
              Error (Printf.sprintf "token %d contains malformed UTF-8" id)
          | `Await -> assert false
        in
        bytes ()

  let continuation byte = byte land 0xc0 = 0x80

  let complete_prefix text =
    let length = String.length text in
    let byte index = Char.code text.[index] in
    let invalid index =
      Error (Printf.sprintf "token stream contains malformed UTF-8 at byte %d" index)
    in
    let rec scan index =
      if index = length then Ok length
      else
        let first = byte index in
        if first <= 0x7f then scan (index + 1)
        else if first >= 0xc2 && first <= 0xdf then
          if index + 2 > length then Ok index
          else if continuation (byte (index + 1)) then scan (index + 2)
          else invalid index
        else if first >= 0xe0 && first <= 0xef then
          if index + 3 > length then Ok index
          else
            let second = byte (index + 1) in
            let second_valid =
              if first = 0xe0 then second >= 0xa0 && second <= 0xbf
              else if first = 0xed then second >= 0x80 && second <= 0x9f
              else continuation second
            in
            if second_valid && continuation (byte (index + 2)) then
              scan (index + 3)
            else invalid index
        else if first >= 0xf0 && first <= 0xf4 then
          if index + 4 > length then Ok index
          else
            let second = byte (index + 1) in
            let second_valid =
              if first = 0xf0 then second >= 0x90 && second <= 0xbf
              else if first = 0xf4 then second >= 0x80 && second <= 0x8f
              else continuation second
            in
            if
              second_valid && continuation (byte (index + 2))
              && continuation (byte (index + 3))
            then scan (index + 4)
            else invalid index
        else invalid index
    in
    scan 0

  let push decoder id =
    if decoder.finished then Error "token decoder is already finished"
    else
      let* bytes = token_bytes decoder id in
      let pending = decoder.pending ^ bytes in
      let* complete = complete_prefix pending in
      let output = String.sub pending 0 complete in
      decoder.pending <-
        String.sub pending complete (String.length pending - complete);
      Ok output

  let finish decoder =
    if decoder.finished then Error "token decoder is already finished"
    else (
      decoder.finished <- true;
      if decoder.pending = "" then Ok ()
      else Error "token stream ends with incomplete UTF-8")
end
