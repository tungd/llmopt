module Format = struct
  type t = Q8

  let q8_group_size = 64
  let default = Q8
  let validate Q8 = Ok ()
  let to_string Q8 = "q8-group-64"
  let group_size Q8 = Some q8_group_size

  let groups_for_elements format ~elements =
    if elements < 0 then invalid_arg "KV element count cannot be negative";
    let group_size = q8_group_size in
    (elements / group_size)
    + if elements mod group_size = 0 then 0 else 1

  let bytes_for_elements format ~elements =
    if elements < 0 then invalid_arg "KV element count cannot be negative";
    let groups = groups_for_elements format ~elements in
    if groups > (max_int - elements) / 2 then
      invalid_arg "Q8 KV byte length overflows"
    else elements + (2 * groups)
end

module Layout = struct
  let q8_head_dim = 64

  module Attention_layer = struct
    type t = {
      kv_heads : int;
      head_dim : int;
    }

    let create ~kv_heads ~head_dim =
      if kv_heads <= 0 then
        Error "attention KV layer requires positive kv_heads"
      else if head_dim <= 0 then
        Error "attention KV layer requires positive head_dim"
      else Ok { kv_heads; head_dim }

    let kv_heads layer = layer.kv_heads
    let head_dim layer = layer.head_dim
  end

  type attention_region = {
    geometry : Attention_layer.t;
    key_offset : int;
    value_offset : int;
    segment_elements : int;
    segment_bytes : int;
  }

  type t = {
    format : Format.t;
    attentions : attention_region array;
    recurrent_layers : int;
    recurrent_width : int;
    recurrent_window : int;
    token_elements : int;
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

  let checked_sum label left right =
    if right > max_int - left then Error (label ^ " overflows")
    else Ok (left + right)

  let checked_bytes format ~elements label =
    try Ok (Format.bytes_for_elements format ~elements)
    with Invalid_argument _ -> Error (label ^ " byte length overflows")

  let create_heterogeneous ~format ~attentions ~recurrent_layers
      ~recurrent_width ~recurrent_window =
    let dimensions =
      [ ("recurrent_layers", recurrent_layers);
        ("recurrent_width", recurrent_width);
        ("recurrent_window", recurrent_window) ]
    in
    match List.find_opt (fun (_, value) -> value < 0) dimensions with
    | Some (name, _) -> Error (name ^ " cannot be negative")
    | None when recurrent_layers > 0 &&
        (recurrent_width = 0 || recurrent_window = 0) ->
        Error
          "recurrent checkpoint layout requires positive width and window"
    | None ->
        let open Result.Syntax in
        let* () = Format.validate format in
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
          | Some group_size
            when
              List.exists
                (fun layer ->
                  Attention_layer.head_dim layer mod group_size <> 0)
                attentions ->
              Some
                "Q8 KV group_size must divide each attention head dimension"
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
            let add_region (element_offset, regions) geometry =
              let* elements =
                checked_product "attention KV layer element count"
                  [ Attention_layer.kv_heads geometry;
                    Attention_layer.head_dim geometry ]
              in
              let* segment_bytes =
                checked_bytes format ~elements "attention KV segment"
              in
              let key_offset = element_offset in
              let* value_offset =
                checked_sum "KV token element count" key_offset elements
              in
              let* next_element_offset =
                checked_sum "KV token element count" value_offset elements
              in
              Ok
                ( next_element_offset,
                  {
                    geometry;
                    key_offset;
                    value_offset;
                    segment_elements = elements;
                    segment_bytes;
                  }
                  :: regions )
            in
            let* token_elements, reverse_regions =
              List.fold_left
                (fun result geometry ->
                  let* state = result in
                  add_region state geometry)
                (Ok (0, [])) attentions
            in
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
                attentions = Array.of_list (List.rev reverse_regions);
                recurrent_layers;
                recurrent_width;
                recurrent_window;
                token_elements;
                bytes_per_token;
                bytes_per_checkpoint;
              })

  let create ~format ~attention_layers ~kv_heads ~head_dim ~recurrent_layers
      ~recurrent_width ~recurrent_window =
    let open Result.Syntax in
    if attention_layers < 0 then Error "attention_layers cannot be negative"
    else if kv_heads < 0 then Error "kv_heads cannot be negative"
    else if head_dim < 0 then Error "head_dim cannot be negative"
    else
      let* attentions =
        if attention_layers = 0 then
          if kv_heads = 0 && head_dim = 0 then Ok []
          else
            Error
              "attention KV layout with zero layers requires zero kv_heads and head_dim"
        else
          let* layer = Attention_layer.create ~kv_heads ~head_dim in
          Ok (List.init attention_layers (Fun.const layer))
      in
      create_heterogeneous ~format ~attentions ~recurrent_layers
        ~recurrent_width ~recurrent_window

  let format layout = layout.format
  let attention_layers layout = Array.length layout.attentions

  let attentions layout =
    Array.to_list layout.attentions |> List.map (fun region -> region.geometry)

  let region layout layer =
    if layer < 0 || layer >= Array.length layout.attentions then None
    else Some layout.attentions.(layer)

  let attention layout ~layer =
    region layout layer |> Option.map (fun region -> region.geometry)

  let attention_key_offset layout ~layer =
    region layout layer |> Option.map (fun region -> region.key_offset)

  let attention_value_offset layout ~layer =
    region layout layer |> Option.map (fun region -> region.value_offset)

  let attention_segment_elements layout ~layer =
    region layout layer |> Option.map (fun region -> region.segment_elements)

  let attention_segment_bytes layout ~layer =
    region layout layer |> Option.map (fun region -> region.segment_bytes)

  let uniform_attention layout =
    match attentions layout with
    | [] -> Some (0, 0)
    | first :: rest ->
        let geometry layer =
          (Attention_layer.kv_heads layer, Attention_layer.head_dim layer)
        in
        let expected = geometry first in
        if List.for_all (fun layer -> geometry layer = expected) rest then
          Some expected
        else None

  let kv_heads layout =
    match attention layout ~layer:0 with
    | None -> 0
    | Some layer -> Attention_layer.kv_heads layer

  let head_dim layout =
    match attention layout ~layer:0 with
    | None -> 0
    | Some layer -> Attention_layer.head_dim layer

  let recurrent_layers layout = layout.recurrent_layers
  let recurrent_width layout = layout.recurrent_width
  let recurrent_window layout = layout.recurrent_window
  let token_elements layout = layout.token_elements
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
