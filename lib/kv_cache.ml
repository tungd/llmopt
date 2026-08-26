module Format = struct
  type t = F16 | Q8 of { group_size : int }

  let q8_group_size = 64
  let default = Q8 { group_size = q8_group_size }
  let f16 = F16

  let validate = function
    | F16 -> Ok ()
    | Q8 { group_size } when group_size = q8_group_size -> Ok ()
    | Q8 _ -> Error "Q8 KV group_size must be 64"

  let q8 ~group_size =
    if group_size <= 0 then Error "Q8 KV group_size must be positive"
    else if group_size <> q8_group_size then
      Error "Q8 KV group_size must be 64"
    else Ok (Q8 { group_size })

  let to_string = function
    | F16 -> "f16"
    | Q8 { group_size } -> Printf.sprintf "q8-group-%d" group_size

  let group_size = function
    | F16 -> None
    | Q8 { group_size } -> Some group_size

  let groups_for_elements format ~elements =
    if elements < 0 then invalid_arg "KV element count cannot be negative";
    match format with
    | F16 -> 0
    | Q8 { group_size } ->
        (elements / group_size)
        + if elements mod group_size = 0 then 0 else 1

  let bytes_for_elements format ~elements =
    if elements < 0 then invalid_arg "KV element count cannot be negative";
    match format with
    | F16 ->
        if elements > max_int / 2 then
          invalid_arg "FP16 KV byte length overflows"
        else 2 * elements
    | Q8 _ ->
        let groups = groups_for_elements format ~elements in
        if groups > (max_int - elements) / 2 then
          invalid_arg "Q8 KV byte length overflows"
        else elements + (2 * groups)
end

module Layout = struct
  let q8_head_dim = 64

  type t = {
    format : Format.t;
    attention_layers : int;
    kv_heads : int;
    head_dim : int;
    recurrent_layers : int;
    recurrent_width : int;
    recurrent_window : int;
    bytes_per_token : int;
    bytes_per_checkpoint : int;
  }

  let checked_product label factors =
    let rec multiply product = function
      | [] -> Ok product
      | factor :: rest ->
          if factor <> 0 && product > max_int / factor then
            Error (label ^ " overflows")
          else multiply (product * factor) rest
    in
    multiply 1 factors

  let checked_bytes format ~elements label =
    try Ok (Format.bytes_for_elements format ~elements)
    with Invalid_argument _ -> Error (label ^ " byte length overflows")

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
        let open Result.Syntax in
        let* () = Format.validate format in
        let* token_elements =
          checked_product "attention KV element count"
            [ 2; attention_layers; kv_heads; head_dim ]
        in
        let* recurrent_layer_elements =
          checked_product "recurrent layer checkpoint element count"
            [ recurrent_width; recurrent_window ]
        in
        let* checkpoint_elements =
          checked_product "recurrent checkpoint element count"
            [ recurrent_layers; recurrent_layer_elements ]
        in
        let incompatible_segment =
          match Format.group_size format with
          | None -> None
          | Some _
            when attention_layers > 0 && head_dim <> q8_head_dim ->
              Some "Q8 KV attention head_dim must be 64"
          | Some group_size
            when attention_layers > 0 && head_dim mod group_size <> 0 ->
              Some "Q8 KV group_size must divide attention head_dim"
          | Some group_size
            when recurrent_layers > 0
                 && recurrent_layer_elements mod group_size <> 0 ->
              Some
                "Q8 KV group_size must divide each recurrent layer checkpoint"
          | Some _ -> None
        in
        (match incompatible_segment with
        | Some message -> Error message
        | None ->
            let* bytes_per_token =
              checked_bytes format ~elements:token_elements "KV token"
            in
            let* bytes_per_checkpoint =
              checked_bytes format ~elements:checkpoint_elements
                "KV checkpoint"
            in
            Ok
              {
                format;
                attention_layers;
                kv_heads;
                head_dim;
                recurrent_layers;
                recurrent_width;
                recurrent_window;
                bytes_per_token;
                bytes_per_checkpoint;
              })

  let format layout = layout.format
  let attention_layers layout = layout.attention_layers
  let kv_heads layout = layout.kv_heads
  let head_dim layout = layout.head_dim
  let recurrent_layers layout = layout.recurrent_layers
  let recurrent_width layout = layout.recurrent_width
  let recurrent_window layout = layout.recurrent_window
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
    token_pool_bytes : int;
    checkpoint_pool_bytes : int;
  }

  let create ~layout ~token_capacity ~checkpoint_capacity =
    let token_bytes = Layout.bytes_per_token layout in
    let checkpoint_bytes = Layout.bytes_per_checkpoint layout in
    if token_capacity <= 0 then Error "KV token_capacity must be positive"
    else if checkpoint_capacity <= 0 then
      Error "KV checkpoint_capacity must be positive"
    else if token_bytes > 0 && token_capacity > max_int / token_bytes then
      Error "KV token pool byte length overflows"
    else if
      checkpoint_bytes > 0
      && checkpoint_capacity > max_int / checkpoint_bytes
    then Error "KV checkpoint pool byte length overflows"
    else
      Ok
        {
          layout;
          token_capacity;
          checkpoint_capacity;
          token_pool_bytes = token_capacity * token_bytes;
          checkpoint_pool_bytes =
            checkpoint_capacity * checkpoint_bytes;
        }

  let layout config = config.layout
  let token_capacity config = config.token_capacity
  let checkpoint_capacity config = config.checkpoint_capacity
  let token_pool_bytes config = config.token_pool_bytes
  let checkpoint_pool_bytes config = config.checkpoint_pool_bytes
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
