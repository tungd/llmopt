module Format = struct
  type t = F16 | Q8 of { group_size : int }

  let default = Q8 { group_size = 64 }
  let f16 = F16

  let q8 ~group_size =
    if group_size <= 0 then Error "Q8 KV group_size must be positive"
    else Ok (Q8 { group_size })

  let to_string = function
    | F16 -> "f16"
    | Q8 { group_size } -> Printf.sprintf "q8-group-%d" group_size

  let bytes_for_elements format ~elements =
    if elements < 0 then invalid_arg "KV element count cannot be negative";
    match format with
    | F16 -> 2 * elements
    | Q8 { group_size } ->
        let groups = (elements + group_size - 1) / group_size in
        elements + (2 * groups)
end

module Layout = struct
  type t = {
    format : Format.t;
    bytes_per_token : int;
    bytes_per_checkpoint : int;
  }

  let create ~format ~attention_layers ~kv_heads ~head_dim ~recurrent_layers
      ~recurrent_width ~recurrent_window =
    let dimensions =
      [ ("attention_layers", attention_layers); ("kv_heads", kv_heads);
        ("head_dim", head_dim); ("recurrent_layers", recurrent_layers);
        ("recurrent_width", recurrent_width);
        ("recurrent_window", recurrent_window) ]
    in
    match List.find_opt (fun (_, value) -> value < 0) dimensions with
    | Some (name, _) -> Error (name ^ " cannot be negative")
    | None when attention_layers > 0 && (kv_heads = 0 || head_dim = 0) ->
        Error "attention KV layout requires positive kv_heads and head_dim"
    | None when recurrent_layers > 0 &&
        (recurrent_width = 0 || recurrent_window = 0) ->
        Error
          "recurrent checkpoint layout requires positive width and window"
    | None ->
        let token_elements =
          2 * attention_layers * kv_heads * head_dim
        in
        let checkpoint_elements =
          recurrent_layers * recurrent_width * recurrent_window
        in
        Ok
          {
            format;
            bytes_per_token =
              Format.bytes_for_elements format ~elements:token_elements;
            bytes_per_checkpoint =
              Format.bytes_for_elements format ~elements:checkpoint_elements;
          }

  let format layout = layout.format
  let bytes_per_token layout = layout.bytes_per_token
  let bytes_per_checkpoint layout = layout.bytes_per_checkpoint
end

module Slot = struct
  type t = int
  let to_int slot = slot
end

module Checkpoint = struct
  type t = int
  let to_int checkpoint = checkpoint
end

module Config = struct
  type t = {
    layout : Layout.t;
    token_capacity : int;
    checkpoint_capacity : int;
  }

  let create ~layout ~token_capacity ~checkpoint_capacity =
    if token_capacity <= 0 then Error "KV token_capacity must be positive"
    else if checkpoint_capacity <= 0 then
      Error "KV checkpoint_capacity must be positive"
    else Ok { layout; token_capacity; checkpoint_capacity }

  let layout config = config.layout
  let token_capacity config = config.token_capacity
  let checkpoint_capacity config = config.checkpoint_capacity
end

module Stats = struct
  type t = {
    token_capacity : int;
    used_tokens : int;
    checkpoint_capacity : int;
    used_checkpoints : int;
    allocated_bytes : int;
  }
end

type pool = {
  mutable free : int list;
  in_use : bool array;
  mutable used : int;
}

type t = {
  config : Config.t;
  tokens : pool;
  checkpoints : pool;
}

type error =
  | Token_capacity_exhausted of { requested : int; available : int }
  | Checkpoint_capacity_exhausted
  | Invalid_release of string

let pool capacity =
  {
    free = List.init capacity (fun index -> capacity - index - 1);
    in_use = Array.make capacity false;
    used = 0;
  }

let create config =
  {
    config;
    tokens = pool (Config.token_capacity config);
    checkpoints = pool (Config.checkpoint_capacity config);
  }

let reserve pool count =
  let rec take remaining free selected =
    if remaining = 0 then Some (List.rev selected, free)
    else
      match free with
      | [] -> None
      | slot :: rest -> take (remaining - 1) rest (slot :: selected)
  in
  match take count pool.free [] with
  | None -> None
  | Some (selected, free) ->
      pool.free <- free;
      List.iter (fun slot -> pool.in_use.(slot) <- true) selected;
      pool.used <- pool.used + count;
      Some (Array.of_list selected)

let reserve_tokens cache count =
  if count < 0 then Error (Invalid_release "cannot reserve a negative token count")
  else
    match reserve cache.tokens count with
    | Some slots -> Ok slots
    | None ->
        Error
          (Token_capacity_exhausted
             {
               requested = count;
               available = Array.length cache.tokens.in_use - cache.tokens.used;
             })

let release pool kind slots =
  let error = ref None in
  let seen = Hashtbl.create (Array.length slots) in
  Array.iter
    (fun slot ->
      if Hashtbl.mem seen slot then
        error := Some (Printf.sprintf "%s slot %d is duplicated" kind slot)
      else if slot < 0 || slot >= Array.length pool.in_use then
        error := Some (Printf.sprintf "%s slot %d is out of range" kind slot)
      else if not pool.in_use.(slot) then
        error := Some (Printf.sprintf "%s slot %d is not reserved" kind slot)
      else Hashtbl.add seen slot ())
    slots;
  match !error with
  | Some message -> Error (Invalid_release message)
  | None ->
      Array.iter
        (fun slot ->
          pool.in_use.(slot) <- false;
          pool.free <- slot :: pool.free)
        slots;
      pool.used <- pool.used - Array.length slots;
      Ok ()

let release_tokens cache slots = release cache.tokens "token" slots

let reserve_checkpoint cache =
  match reserve cache.checkpoints 1 with
  | Some slots -> Ok slots.(0)
  | None -> Error Checkpoint_capacity_exhausted

let release_checkpoint cache checkpoint =
  release cache.checkpoints "checkpoint" [| checkpoint |]

let stats cache =
  let layout = Config.layout cache.config in
  let used_tokens = cache.tokens.used in
  let used_checkpoints = cache.checkpoints.used in
  {
    Stats.token_capacity = Array.length cache.tokens.in_use;
    used_tokens;
    checkpoint_capacity = Array.length cache.checkpoints.in_use;
    used_checkpoints;
    allocated_bytes =
      (used_tokens * Layout.bytes_per_token layout)
      + (used_checkpoints * Layout.bytes_per_checkpoint layout);
  }

let error_to_string = function
  | Token_capacity_exhausted { requested; available } ->
      Printf.sprintf "KV token capacity exhausted: requested=%d available=%d"
        requested available
  | Checkpoint_capacity_exhausted -> "KV checkpoint capacity exhausted"
  | Invalid_release message -> "invalid KV release: " ^ message
