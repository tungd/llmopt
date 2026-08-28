let ( let* ) = Result.bind

module Config = struct

  type t = {
    model : Lfm25.Config.t option;
    state_plan : Model_program.State.t option;
    kv : Kv_cache.Config.t;
    page_size : int;
  }

  let of_state_plan ~state ~token_capacity ~checkpoint_capacity ~page_size () =
    if page_size <= 0 then Error "serving radix page_size must be positive"
    else
      let layout = Model_program.State.layout state in
      let attention_layers =
        Model_program.State.Cache_layout.attention_layers layout
      in
      let kv_heads = Model_program.State.Cache_layout.kv_heads layout in
      let head_dim = Model_program.State.Cache_layout.head_dim layout in
      let recurrent_layers =
        Model_program.State.Cache_layout.recurrent_layers layout
      in
      let recurrent_dim =
        Model_program.State.Cache_layout.recurrent_dim layout
      in
      if attention_layers > 0 && head_dim <> Kv_cache.Layout.q8_head_dim then
        Error "serving Q8 KV requires attention head_dim=64"
      else
        let recurrent_width =
          if recurrent_layers = 0 then 0 else recurrent_dim
        in
        let recurrent_window = if recurrent_layers = 0 then 0 else 3 in
        match
          Kv_cache.Layout.create ~format:Kv_cache.Format.default
            ~attention_layers ~kv_heads ~head_dim ~recurrent_layers
            ~recurrent_width ~recurrent_window
        with
        | Error message -> Error message
        | Ok kv_layout ->
            (match
               Kv_cache.Config.create ~layout:kv_layout ~token_capacity
                 ~checkpoint_capacity
             with
            | Error message -> Error message
            | Ok kv ->
                Ok
                  {
                    model = None;
                    state_plan = Some state;
                    kv;
                    page_size;
                  })

  let create ~model ~token_capacity ~checkpoint_capacity ~page_size () =
    if model.Lfm25.Config.hidden_size <= 0 then
      Error "serving model hidden_size must be positive"
    else if model.num_attention_heads <= 0 then
      Error "serving model num_attention_heads must be positive"
    else if model.num_key_value_heads <= 0 then
      Error "serving model num_key_value_heads must be positive"
    else if model.hidden_size mod model.num_attention_heads <> 0 then
      Error "serving model hidden_size must divide evenly into attention heads"
    else
      match Lfm25.Config.validate model with
      | Error message -> Error ("invalid serving model config: " ^ message)
      | Ok () when page_size <= 0 ->
          Error "serving radix page_size must be positive"
      | Ok () ->
          let attention_layers =
            Lfm25.Config.count_layers Lfm25.Config.Full_attention model
          in
          let recurrent_layers =
            Lfm25.Config.count_layers Lfm25.Config.Conv model
          in
          let head_dim = model.hidden_size / model.num_attention_heads in
          if head_dim <> Kv_cache.Layout.q8_head_dim then
            Error "serving Q8 KV requires attention head_dim=64"
          else
            match
              Kv_cache.Layout.create ~format:Kv_cache.Format.default
                ~attention_layers ~kv_heads:model.num_key_value_heads ~head_dim
                ~recurrent_layers ~recurrent_width:model.hidden_size
                ~recurrent_window:model.conv_l_cache
            with
            | Error message -> Error message
            | Ok layout ->
                (match
                   Kv_cache.Config.create ~layout ~token_capacity
                     ~checkpoint_capacity
                 with
                | Error message -> Error message
                | Ok kv ->
                    Ok
                      {
                        model = Some model;
                        state_plan = None;
                        kv;
                        page_size;
                      })

  let kv config = config.kv
  let page_size config = config.page_size
  let model config =
    match config.model with
    | Some m -> m
    | None -> Lfm25.Config.default
  let state_plan config = config.state_plan
end

module Match = struct
  type t = {
    lease : (Kv_cache.Slot.t, Kv_cache.Checkpoint.t) Radix_cache.lease;
  }

  let tokens match_ = Radix_cache.matched_tokens match_.lease
  let slots match_ = Radix_cache.matched_values match_.lease
  let checkpoint match_ = Radix_cache.checkpoint match_.lease
end

module Stats = struct
  type t = {
    radix : Radix_cache.Stats.t;
    kv : Kv_cache.Stats.t;
  }
end

type t = {
  kv : Kv_cache.t;
  radix : (Kv_cache.Slot.t, Kv_cache.Checkpoint.t) Radix_cache.t;
}

let create config =
  let radix =
    match Radix_cache.create ~page_size:(Config.page_size config) with
    | Ok radix -> radix
    | Error message -> invalid_arg message
  in
  { kv = Kv_cache.create (Config.kv config); radix }

let reserve_tokens cache = Kv_cache.reserve_tokens cache.kv
let reserve_checkpoint cache = Kv_cache.reserve_checkpoint cache.kv
let release_tokens cache slots =
  Kv_cache.release_tokens cache.kv slots
  |> Result.map_error Kv_cache.error_to_string

let release_checkpoint cache checkpoint =
  Kv_cache.release_checkpoint cache.kv checkpoint
  |> Result.map_error Kv_cache.error_to_string

let match_prefix cache ?namespace ~reserve_tail tokens =
  let retained = max 0 (Array.length tokens - max 0 reserve_tail) in
  let key = Radix_cache.Key.create ?namespace (Array.sub tokens 0 retained) in
  { Match.lease = Radix_cache.match_prefix cache.radix key }

let release_match cache match_ = Radix_cache.release cache.radix match_.Match.lease

let same_slot left right =
  Kv_cache.Slot.to_int left = Kv_cache.Slot.to_int right

let same_checkpoint left right =
  Kv_cache.Checkpoint.to_int left = Kv_cache.Checkpoint.to_int right

let release_redundant cache result =
  let redundant_values = result.Radix_cache.redundant_values in
  let canonical_values = result.Radix_cache.canonical_values in
  if Array.length redundant_values <> Array.length canonical_values then
    Error "radix insertion returned inconsistent canonical ownership"
  else
  let releasable =
    Array.to_list
      (Array.mapi
         (fun index value ->
           if same_slot value canonical_values.(index) then None else Some value)
         redundant_values)
    |> List.filter_map Fun.id |> Array.of_list
  in
  match Kv_cache.release_tokens cache.kv releasable with
  | Error error -> Error (Kv_cache.error_to_string error)
  | Ok () ->
      (match result.redundant_checkpoint with
      | None -> Ok ()
      | Some checkpoint
        when Option.exists (same_checkpoint checkpoint)
               result.retained_checkpoint ->
          Ok ()
      | Some checkpoint ->
          release_checkpoint cache checkpoint)

let insert cache ?namespace ~tokens ~slots ~checkpoint () =
  let key = Radix_cache.Key.create ?namespace tokens in
  match Radix_cache.insert cache.radix ~key ~values:slots ~checkpoint with
  | Error message -> Error message
  | Ok result ->
      (match release_redundant cache result with
      | Error message -> Error message
      | Ok () -> Ok result.prefix_tokens)

let evict cache ~target_tokens =
  let eviction = Radix_cache.evict cache.radix ~target_tokens in
  match Kv_cache.release_tokens cache.kv eviction.values with
  | Error error -> Error (Kv_cache.error_to_string error)
  | Ok () ->
      let rec release_checkpoints = function
        | [] -> Ok ()
        | checkpoint :: rest ->
            (match Kv_cache.release_checkpoint cache.kv checkpoint with
            | Error error -> Error (Kv_cache.error_to_string error)
            | Ok () -> release_checkpoints rest)
      in
      (match release_checkpoints eviction.checkpoints with
      | Error message -> Error message
      | Ok () -> Ok (Array.length eviction.values))

let stats cache =
  { Stats.radix = Radix_cache.stats cache.radix; kv = Kv_cache.stats cache.kv }

let validate cache =
  let radix = Radix_cache.stats cache.radix in
  let kv = Kv_cache.stats cache.kv in
  match Radix_cache.validate cache.radix with
  | Error message -> Error message
  | Ok () when radix.cached_tokens <> kv.used_tokens ->
      Error "serving cache radix/KV token ownership mismatch"
  | Ok () when radix.checkpoints <> kv.used_checkpoints ->
      Error "serving cache radix/KV checkpoint ownership mismatch"
  | Ok () -> Ok ()
