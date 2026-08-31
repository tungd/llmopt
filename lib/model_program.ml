let ( let* ) = Result.bind

module Artifact = struct
  type t = string

  let create path =
    let path = String.trim path in
    let components = String.split_on_char '/' path in
    if path = "" then Error "model-program artifact path cannot be empty"
    else if not (Filename.is_relative path) then
      Error ("model-program artifact path must be relative: " ^ path)
    else if String.contains path '\000' then
      Error "model-program artifact path contains a NUL byte"
    else if
      List.exists
        (fun component -> component = "" || component = "." || component = "..")
        components
    then Error ("model-program artifact path is not canonical: " ^ path)
    else Ok path

  let path artifact = artifact
end

module Identity = struct
  type t = {
    model : string;
    architecture : string option;
    family : string option;
  }

  let validate_non_empty name value =
    let trimmed = String.trim value in
    if trimmed = "" then
      Error (Printf.sprintf "model-program %s cannot be empty" name)
    else Ok trimmed

  let create ~model ?architecture ?family () =
    let* model = validate_non_empty "model identifier" model in
    let* architecture =
      match architecture with
      | None -> Ok None
      | Some arch ->
          let* arch = validate_non_empty "architecture" arch in
          Ok (Some arch)
    in
    let* family =
      match family with
      | None -> Ok None
      | Some fam ->
          let* fam = validate_non_empty "family" fam in
          Ok (Some fam)
    in
    Ok { model; architecture; family }

  let model identity = identity.model
  let architecture identity = identity.architecture
  let family identity = identity.family
end

module Processor = struct
  module Chat = struct
    type format = Chatml

    type t = {
      format : format;
      bos_token_id : int;
      message_start_token_id : int;
      message_end_token_id : int;
    }

    let format_to_string = function Chatml -> "chatml"

    let create ~format ~bos_token_id ~message_start_token_id
        ~message_end_token_id =
      if
        bos_token_id < 0 || message_start_token_id < 0
        || message_end_token_id < 0
      then Error "model-program chat token ids must be nonnegative"
      else
        Ok { format; bos_token_id; message_start_token_id; message_end_token_id }

    let format chat = chat.format
    let bos_token_id chat = chat.bos_token_id
    let message_start_token_id chat = chat.message_start_token_id
    let message_end_token_id chat = chat.message_end_token_id
  end

  type t = {
    tokenizer : Artifact.t;
    chat : Chat.t option;
  }

  let create ~tokenizer ?chat () = { tokenizer; chat }
  let tokenizer processor = processor.tokenizer
  let chat processor = processor.chat
end

module Entrypoint = struct
  module Head = struct
    type t = {
      logits : string option;
      token_id : string option;
    }

    let create ?logits ?token_id () =
      let clean_opt name = function
        | None -> Ok None
        | Some value ->
            let trimmed = String.trim value in
            if trimmed = "" then
              Error (Printf.sprintf "model-program entrypoint %s cannot be empty" name)
            else Ok (Some trimmed)
      in
      let* logits = clean_opt "logits output" logits in
      let* token_id = clean_opt "token_id output" token_id in
      if Option.is_none logits && Option.is_none token_id then
        Error "model-program entrypoint must declare at least one output (logits or token_id)"
      else Ok { logits; token_id }

    let logits head = head.logits
    let token_id head = head.token_id
  end

  type kind = Prefill | Decode

  let kind_to_string = function
    | Prefill -> "prefill"
    | Decode -> "decode"

  type t = {
    kind : kind;
    package : Artifact.t;
    input_ids : string;
    head : Head.t;
  }

  let create ~kind ~package ~input_ids ~head =
    let trimmed_input = String.trim input_ids in
    if trimmed_input = "" then
      Error "model-program entrypoint input_ids cannot be empty"
    else Ok { kind; package; input_ids = trimmed_input; head }

  let kind entrypoint = entrypoint.kind
  let package entrypoint = entrypoint.package
  let input_ids entrypoint = entrypoint.input_ids
  let head entrypoint = entrypoint.head
end

module Generation = struct
  type t = {
    vocab_size : int;
    max_positions : int;
    eos_token_id : int option;
    bos_token_id : int option;
  }

  let create ~vocab_size ~max_positions ?eos_token_id ?bos_token_id () =
    if vocab_size <= 0 then
      Error "model-program vocab_size must be positive"
    else if max_positions <= 0 then
      Error "model-program max_positions must be positive"
    else
      let* eos_token_id =
        match eos_token_id with
        | None -> Ok None
        | Some id when id < 0 ->
            Error "model-program eos_token_id cannot be negative"
        | Some id -> Ok (Some id)
      in
      let* bos_token_id =
        match bos_token_id with
        | None -> Ok None
        | Some id when id < 0 ->
            Error "model-program bos_token_id cannot be negative"
        | Some id -> Ok (Some id)
      in
      Ok { vocab_size; max_positions; eos_token_id; bos_token_id }

  let vocab_size generation = generation.vocab_size
  let max_positions generation = generation.max_positions
  let eos_token_id generation = generation.eos_token_id
  let bos_token_id generation = generation.bos_token_id
end

module State = struct
  module Attention_binding = struct
    type t = {
      cache_layer : int;
      key_input : string;
      value_input : string;
      key_output : string;
      value_output : string;
    }

    let create ~cache_layer ~key_input ~value_input ~key_output ~value_output =
      if cache_layer < 0 then
        Error "attention binding cache_layer cannot be negative"
      else
        let check_name field name =
          let trimmed = String.trim name in
          if trimmed = "" then
            Error (Printf.sprintf "attention binding %s cannot be empty" field)
          else Ok trimmed
        in
        let* key_input = check_name "key_input" key_input in
        let* value_input = check_name "value_input" value_input in
        let* key_output = check_name "key_output" key_output in
        let* value_output = check_name "value_output" value_output in
        Ok { cache_layer; key_input; value_input; key_output; value_output }

    let cache_layer binding = binding.cache_layer
    let key_input binding = binding.key_input
    let value_input binding = binding.value_input
    let key_output binding = binding.key_output
    let value_output binding = binding.value_output
  end

  module Recurrent_binding = struct
    type t = {
      cache_layer : int;
      state_input : string;
      state_output : string;
    }

    let create ~cache_layer ~state_input ~state_output =
      if cache_layer < 0 then
        Error "recurrent binding cache_layer cannot be negative"
      else
        let check_name field name =
          let trimmed = String.trim name in
          if trimmed = "" then
            Error (Printf.sprintf "recurrent binding %s cannot be empty" field)
          else Ok trimmed
        in
        let* state_input = check_name "state_input" state_input in
        let* state_output = check_name "state_output" state_output in
        Ok { cache_layer; state_input; state_output }

    let cache_layer binding = binding.cache_layer
    let state_input binding = binding.state_input
    let state_output binding = binding.state_output
  end

  module Cache_layout = struct
    module Attention_layer = struct
      type storage = Q8_group_64

      type t = {
        storage : storage;
        kv_heads : int;
        head_dim : int;
      }

      let create ?(storage = Q8_group_64) ~kv_heads ~head_dim () =
        if kv_heads <= 0 then
          Error "cache attention layer requires positive kv_heads"
        else if head_dim <= 0 then
          Error "cache attention layer requires positive head_dim"
        else Ok { storage; kv_heads; head_dim }

      let storage layer = layer.storage
      let storage_to_string Q8_group_64 = "q8-group-64"
      let kv_heads layer = layer.kv_heads
      let head_dim layer = layer.head_dim
    end

    type t = {
      attentions : Attention_layer.t array;
      recurrent_layers : int;
      recurrent_dim : int;
      recurrent_window : int;
    }

    let create_heterogeneous ~attentions ~recurrent_layers ~recurrent_dim
        ~recurrent_window =
      if recurrent_layers < 0 then
        Error "cache layout recurrent_layers cannot be negative"
      else if recurrent_dim < 0 then
        Error "cache layout recurrent_dim cannot be negative"
      else if recurrent_window < 0 then
        Error "cache layout recurrent_window cannot be negative"
      else if
        recurrent_layers > 0
        && (recurrent_dim <= 0 || recurrent_window <= 0)
      then
        Error
          "cache layout with recurrent layers requires positive recurrent dimensions"
      else if
        recurrent_layers = 0
        && (recurrent_dim <> 0 || recurrent_window <> 0)
      then
        Error
          "cache layout with zero recurrent layers must have zero recurrent dimensions"
      else
        Ok
          {
            attentions = Array.of_list attentions;
            recurrent_layers;
            recurrent_dim;
            recurrent_window;
          }

    let create ~attention_layers ~kv_heads ~head_dim ~recurrent_layers
        ~recurrent_dim ~recurrent_window =
      if attention_layers < 0 then
        Error "cache layout attention_layers cannot be negative"
      else if kv_heads < 0 then
        Error "cache layout kv_heads cannot be negative"
      else if head_dim < 0 then
        Error "cache layout head_dim cannot be negative"
      else if attention_layers = 0 && (kv_heads <> 0 || head_dim <> 0) then
        Error
          "cache layout with zero attention layers must have zero kv_heads and head_dim"
      else
        let* attentions =
          if attention_layers = 0 then Ok []
          else
            let* layer = Attention_layer.create ~kv_heads ~head_dim () in
            Ok (List.init attention_layers (Fun.const layer))
        in
        create_heterogeneous ~attentions ~recurrent_layers ~recurrent_dim
          ~recurrent_window

    let attention_layers layout = Array.length layout.attentions
    let attentions layout = Array.to_list layout.attentions

    let attention layout ~cache_layer =
      if cache_layer < 0 || cache_layer >= Array.length layout.attentions then
        None
      else Some layout.attentions.(cache_layer)

    let uniform_attention layout =
      match Array.to_list layout.attentions with
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
      match attention layout ~cache_layer:0 with
      | None -> 0
      | Some layer -> Attention_layer.kv_heads layer

    let head_dim layout =
      match attention layout ~cache_layer:0 with
      | None -> 0
      | Some layer -> Attention_layer.head_dim layer

    let recurrent_layers layout = layout.recurrent_layers
    let recurrent_dim layout = layout.recurrent_dim
    let recurrent_window layout = layout.recurrent_window
  end

  type t = {
    layout : Cache_layout.t;
    attentions : Attention_binding.t list;
    recurrents : Recurrent_binding.t list;
  }

  let create ~layout ~attentions ~recurrents =
    let num_att = List.length attentions in
    let num_rec = List.length recurrents in
    if num_att <> Cache_layout.attention_layers layout then
      Error
        (Printf.sprintf
           "state attention binding count (%d) disagrees with cache layout attention layers (%d)"
           num_att (Cache_layout.attention_layers layout))
    else if num_rec <> layout.Cache_layout.recurrent_layers then
      Error
        (Printf.sprintf
           "state recurrent binding count (%d) disagrees with cache layout recurrent layers (%d)"
           num_rec layout.Cache_layout.recurrent_layers)
    else
      let rec check_unique_indices seen = function
        | [] -> Ok ()
        | idx :: rest ->
            if List.mem idx seen then
              Error (Printf.sprintf "duplicate state layer index: %d" idx)
            else check_unique_indices (idx :: seen) rest
      in
      let att_indices = List.map Attention_binding.cache_layer attentions in
      let rec_indices = List.map Recurrent_binding.cache_layer recurrents in
      let* () = check_unique_indices [] att_indices in
      let* () = check_unique_indices [] rec_indices in
      let* () =
        match
          List.find_opt
            (fun idx -> idx < 0 || idx >= Cache_layout.attention_layers layout)
            att_indices
        with
        | Some bad ->
            Error
              (Printf.sprintf
                 "attention binding layer %d is out of range [0, %d)" bad
                 (Cache_layout.attention_layers layout))
        | None -> Ok ()
      in
      let* () =
        match List.find_opt (fun idx -> idx < 0 || idx >= layout.Cache_layout.recurrent_layers) rec_indices with
        | Some bad ->
            Error (Printf.sprintf "recurrent binding layer %d is out of range [0, %d)" bad layout.Cache_layout.recurrent_layers)
        | None -> Ok ()
      in
      let rec check_unique_names seen = function
        | [] -> Ok ()
        | name :: rest ->
            if List.mem name seen then
              Error (Printf.sprintf "duplicate state binding tensor name: %s" name)
            else check_unique_names (name :: seen) rest
      in
      let att_inputs =
        List.concat_map
          (fun b -> [ Attention_binding.key_input b; Attention_binding.value_input b ])
          attentions
      in
      let att_outputs =
        List.concat_map
          (fun b -> [ Attention_binding.key_output b; Attention_binding.value_output b ])
          attentions
      in
      let rec_inputs = List.map Recurrent_binding.state_input recurrents in
      let rec_outputs = List.map Recurrent_binding.state_output recurrents in
      let* () = check_unique_names [] (att_inputs @ rec_inputs) in
      let* () = check_unique_names [] (att_outputs @ rec_outputs) in
      Ok { layout; attentions; recurrents }

  let layout state = state.layout
  let attentions state = state.attentions
  let recurrents state = state.recurrents
end

module Specialization = struct
  type t = {
    min_prefill_tokens : int;
    rope_cosine_input : string option;
    rope_sine_input : string option;
    paged_slots_input : string option;
  }

  let create ?(min_prefill_tokens = 1) ?rope_cosine_input ?rope_sine_input ?paged_slots_input () =
    if min_prefill_tokens < 1 then
      Error "model-program min_prefill_tokens must be at least 1"
    else
      let clean_opt name = function
        | None -> Ok None
        | Some value ->
            let trimmed = String.trim value in
            if trimmed = "" then
              Error (Printf.sprintf "model-program specialization %s cannot be empty" name)
            else Ok (Some trimmed)
      in
      let* rope_cosine_input = clean_opt "rope_cosine_input" rope_cosine_input in
      let* rope_sine_input = clean_opt "rope_sine_input" rope_sine_input in
      let* paged_slots_input = clean_opt "paged_slots_input" paged_slots_input in
      Ok { min_prefill_tokens; rope_cosine_input; rope_sine_input; paged_slots_input }

  let min_prefill_tokens specialization = specialization.min_prefill_tokens
  let rope_cosine_input specialization = specialization.rope_cosine_input
  let rope_sine_input specialization = specialization.rope_sine_input
  let paged_slots_input specialization = specialization.paged_slots_input
end

type t = {
  identity : Identity.t;
  processor : Processor.t;
  prefill : Entrypoint.t;
  decode : Entrypoint.t;
  generation : Generation.t;
  state : State.t;
  specialization : Specialization.t;
}

let current_abi_version = 3
let legacy_uniform_cache_abi_version = 2

let validate program =
  if program.prefill.Entrypoint.kind <> Entrypoint.Prefill then
    Error "model-program prefill entrypoint must have kind Prefill"
  else if program.decode.Entrypoint.kind <> Entrypoint.Decode then
    Error "model-program decode entrypoint must have kind Decode"
  else if Option.is_none (Processor.chat program.processor) then
    Error "model-program requires an explicit chat contract"
  else
    match Processor.chat program.processor with
    | Some chat
      when
        List.exists
          (fun id -> id >= Generation.vocab_size program.generation)
          [ Processor.Chat.bos_token_id chat;
            Processor.Chat.message_start_token_id chat;
            Processor.Chat.message_end_token_id chat ] ->
        Error "model-program chat token id exceeds vocabulary"
    | Some _ | None -> Ok ()

let make ~identity ~processor ~prefill ~decode ~generation ~state ~specialization =
  let program =
    {
      identity;
      processor;
      prefill;
      decode;
      generation;
      state;
      specialization;
    }
  in
  let* () = validate program in
  Ok program

let create ~identity ~processor ~prefill ~decode ~generation ~state
    ~specialization =
  make ~identity ~processor ~prefill ~decode ~generation ~state ~specialization

let abi_version _program = current_abi_version
let identity program = program.identity
let processor program = program.processor
let prefill program = program.prefill
let decode program = program.decode
let generation program = program.generation
let state program = program.state
let specialization program = program.specialization

let magic = "LLMOPT-MODEL\000"

let write_option writer write = function
  | None -> Binary.Writer.u8 writer 0
  | Some value ->
      Binary.Writer.u8 writer 1;
      write writer value

let read_option reader read =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 -> Ok None
  | 1 -> read reader |> Result.map Option.some
  | _ -> Error (Printf.sprintf "unknown model-program option tag: %d" tag)

let write_artifact writer artifact =
  Binary.Writer.string writer (Artifact.path artifact)

let read_artifact reader =
  let* path = Binary.Reader.string reader in
  Artifact.create path

let write_entrypoint writer (entry : Entrypoint.t) =
  write_artifact writer entry.package;
  Binary.Writer.string writer entry.input_ids;
  write_option writer Binary.Writer.string (Entrypoint.Head.logits entry.head);
  write_option writer Binary.Writer.string (Entrypoint.Head.token_id entry.head)

let read_entrypoint reader kind =
  let* package = read_artifact reader in
  let* input_ids = Binary.Reader.string reader in
  let* logits = read_option reader Binary.Reader.string in
  let* token_id = read_option reader Binary.Reader.string in
  let* head = Entrypoint.Head.create ?logits ?token_id () in
  Entrypoint.create ~kind ~package ~input_ids ~head

let to_bytes program =
  let writer = Binary.Writer.create () in
  Binary.Writer.raw_string writer magic;
  Binary.Writer.u16 writer current_abi_version;

  (* Identity *)
  Binary.Writer.string writer (Identity.model program.identity);
  write_option writer Binary.Writer.string (Identity.architecture program.identity);
  write_option writer Binary.Writer.string (Identity.family program.identity);

  (* Processor *)
  write_artifact writer (Processor.tokenizer program.processor);
  write_option writer
    (fun writer chat ->
      Binary.Writer.u8 writer
        (match Processor.Chat.format chat with Processor.Chat.Chatml -> 1);
      Binary.Writer.u32 writer (Processor.Chat.bos_token_id chat);
      Binary.Writer.u32 writer (Processor.Chat.message_start_token_id chat);
      Binary.Writer.u32 writer (Processor.Chat.message_end_token_id chat))
    (Processor.chat program.processor);

  (* Prefill entrypoint *)
  write_entrypoint writer program.prefill;

  (* Decode entrypoint *)
  write_entrypoint writer program.decode;

  (* Generation *)
  Binary.Writer.u64 writer (Generation.vocab_size program.generation);
  Binary.Writer.u64 writer (Generation.max_positions program.generation);
  write_option writer Binary.Writer.i64 (Generation.eos_token_id program.generation);
  write_option writer Binary.Writer.i64 (Generation.bos_token_id program.generation);

  (* State *)
  let layout = State.layout program.state in
  let attention_layouts = State.Cache_layout.attentions layout in
  Binary.Writer.u16 writer (List.length attention_layouts);
  List.iter
    (fun layer ->
      Binary.Writer.u8 writer
        (match State.Cache_layout.Attention_layer.storage layer with
        | State.Cache_layout.Attention_layer.Q8_group_64 -> 1);
      Binary.Writer.u16 writer
        (State.Cache_layout.Attention_layer.kv_heads layer);
      Binary.Writer.u16 writer
        (State.Cache_layout.Attention_layer.head_dim layer))
    attention_layouts;
  Binary.Writer.u16 writer (State.Cache_layout.recurrent_layers layout);
  Binary.Writer.u16 writer (State.Cache_layout.recurrent_dim layout);
  Binary.Writer.u16 writer (State.Cache_layout.recurrent_window layout);

  let attentions = State.attentions program.state in
  Binary.Writer.u16 writer (List.length attentions);
  List.iter
    (fun (b : State.Attention_binding.t) ->
      Binary.Writer.u16 writer (State.Attention_binding.cache_layer b);
      Binary.Writer.string writer (State.Attention_binding.key_input b);
      Binary.Writer.string writer (State.Attention_binding.value_input b);
      Binary.Writer.string writer (State.Attention_binding.key_output b);
      Binary.Writer.string writer (State.Attention_binding.value_output b))
    attentions;

  let recurrents = State.recurrents program.state in
  Binary.Writer.u16 writer (List.length recurrents);
  List.iter
    (fun (b : State.Recurrent_binding.t) ->
      Binary.Writer.u16 writer (State.Recurrent_binding.cache_layer b);
      Binary.Writer.string writer (State.Recurrent_binding.state_input b);
      Binary.Writer.string writer (State.Recurrent_binding.state_output b))
    recurrents;

  (* Specialization *)
  Binary.Writer.u32 writer (Specialization.min_prefill_tokens program.specialization);
  write_option writer Binary.Writer.string (Specialization.rope_cosine_input program.specialization);
  write_option writer Binary.Writer.string (Specialization.rope_sine_input program.specialization);
  write_option writer Binary.Writer.string (Specialization.paged_slots_input program.specialization);

  Binary.Writer.contents writer

let of_bytes bytes =
  let reader = Binary.Reader.create bytes in
  let* actual_magic =
    Binary.Reader.raw_string reader ~length:(String.length magic)
  in
  if actual_magic <> magic then Error "invalid model-program magic"
  else
    let* version = Binary.Reader.u16 reader in
    if
      version <> current_abi_version
      && version <> legacy_uniform_cache_abi_version
    then
      Error (Printf.sprintf "unsupported model-program version: %d" version)
    else
      let* model = Binary.Reader.string reader in
      let* architecture = read_option reader Binary.Reader.string in
      let* family = read_option reader Binary.Reader.string in
      let* identity = Identity.create ~model ?architecture ?family () in

      let* tokenizer = read_artifact reader in
      let* chat =
        read_option reader (fun reader ->
            let* format_tag = Binary.Reader.u8 reader in
            let* format =
              match format_tag with
              | 1 -> Ok Processor.Chat.Chatml
              | tag ->
                  Error
                    (Printf.sprintf "unsupported chat template format: %d" tag)
            in
            let* bos_token_id = Binary.Reader.u32 reader in
            let* message_start_token_id = Binary.Reader.u32 reader in
            let* message_end_token_id = Binary.Reader.u32 reader in
            Processor.Chat.create ~format ~bos_token_id ~message_start_token_id
              ~message_end_token_id)
      in
      let processor = Processor.create ~tokenizer ?chat () in

      let* prefill = read_entrypoint reader Entrypoint.Prefill in
      let* decode = read_entrypoint reader Entrypoint.Decode in

      let* vocab_size = Binary.Reader.u64 reader in
      let* max_positions = Binary.Reader.u64 reader in
      let* eos_token_id = read_option reader Binary.Reader.i64 in
      let* bos_token_id = read_option reader Binary.Reader.i64 in
      let* generation = Generation.create ~vocab_size ~max_positions ?eos_token_id ?bos_token_id () in

      let* attentions_layout =
        if version = legacy_uniform_cache_abi_version then
          let* attention_layers = Binary.Reader.u16 reader in
          let* kv_heads = Binary.Reader.u16 reader in
          let* head_dim = Binary.Reader.u16 reader in
          if attention_layers = 0 then
            if kv_heads = 0 && head_dim = 0 then Ok []
            else
              Error
                "cache layout with zero attention layers must have zero kv_heads and head_dim"
          else
            let* layer =
              State.Cache_layout.Attention_layer.create ~kv_heads ~head_dim ()
            in
            Ok (List.init attention_layers (Fun.const layer))
        else
          let* attention_layers = Binary.Reader.u16 reader in
          let rec read_layers acc remaining =
            if remaining = 0 then Ok (List.rev acc)
            else
              let* storage_tag = Binary.Reader.u8 reader in
              let* storage =
                match storage_tag with
                | 1 ->
                    Ok State.Cache_layout.Attention_layer.Q8_group_64
                | tag ->
                    Error
                      (Printf.sprintf
                         "unsupported cache attention storage tag: %d" tag)
              in
              let* kv_heads = Binary.Reader.u16 reader in
              let* head_dim = Binary.Reader.u16 reader in
              let* layer =
                State.Cache_layout.Attention_layer.create ~storage ~kv_heads
                  ~head_dim ()
              in
              read_layers (layer :: acc) (remaining - 1)
          in
          read_layers [] attention_layers
      in
      let* recurrent_layers = Binary.Reader.u16 reader in
      let* recurrent_dim = Binary.Reader.u16 reader in
      let* recurrent_window = Binary.Reader.u16 reader in
      let* layout =
        State.Cache_layout.create_heterogeneous ~attentions:attentions_layout
          ~recurrent_layers ~recurrent_dim ~recurrent_window
      in

      let* att_count = Binary.Reader.u16 reader in
      let rec read_attentions acc remaining =
        if remaining = 0 then Ok (List.rev acc)
        else
          let* cache_layer = Binary.Reader.u16 reader in
          let* key_input = Binary.Reader.string reader in
          let* value_input = Binary.Reader.string reader in
          let* key_output = Binary.Reader.string reader in
          let* value_output = Binary.Reader.string reader in
          let* binding = State.Attention_binding.create ~cache_layer ~key_input ~value_input ~key_output ~value_output in
          read_attentions (binding :: acc) (remaining - 1)
      in
      let* attentions = read_attentions [] att_count in

      let* rec_count = Binary.Reader.u16 reader in
      let rec read_recurrents acc remaining =
        if remaining = 0 then Ok (List.rev acc)
        else
          let* cache_layer = Binary.Reader.u16 reader in
          let* state_input = Binary.Reader.string reader in
          let* state_output = Binary.Reader.string reader in
          let* binding = State.Recurrent_binding.create ~cache_layer ~state_input ~state_output in
          read_recurrents (binding :: acc) (remaining - 1)
      in
      let* recurrents = read_recurrents [] rec_count in
      let* state = State.create ~layout ~attentions ~recurrents in

      let* min_prefill_tokens = Binary.Reader.u32 reader in
      let* rope_cosine_input = read_option reader Binary.Reader.string in
      let* rope_sine_input = read_option reader Binary.Reader.string in
      let* paged_slots_input = read_option reader Binary.Reader.string in
      let* specialization = Specialization.create ~min_prefill_tokens ?rope_cosine_input ?rope_sine_input ?paged_slots_input () in

      let* () = Binary.Reader.finish reader in
      make ~identity ~processor ~prefill ~decode ~generation ~state
        ~specialization

let write_file path program =
  try
    let channel = open_out_bin path in
    Fun.protect
      ~finally:(fun () -> close_out_noerr channel)
      (fun () ->
        output_bytes channel (to_bytes program);
        flush channel;
        Ok ())
  with Sys_error message -> Error ("cannot write model program: " ^ message)

let of_file path =
  try
    let channel = open_in_bin path in
    Fun.protect
      ~finally:(fun () -> close_in_noerr channel)
      (fun () ->
        let bytes = really_input_string channel (in_channel_length channel) in
        of_bytes (Bytes.of_string bytes))
  with
  | Sys_error message -> Error ("cannot read model program: " ^ message)
  | End_of_file -> Error "model program ended unexpectedly"

let validate_files ~root program =
  let artifacts =
    [
      Processor.tokenizer program.processor;
      Entrypoint.package program.prefill;
      Entrypoint.package program.decode;
    ]
  in
  let rec check = function
    | [] -> Ok ()
    | artifact :: rest ->
        let relative = Artifact.path artifact in
        let path = Filename.concat root relative in
        (try
           let stats = Unix.stat path in
           if stats.st_kind <> Unix.S_REG && stats.st_kind <> Unix.S_LNK then
             Error ("model-program artifact is not a regular file: " ^ relative)
           else check rest
         with Unix.Unix_error (error, _, _) ->
           Error
             (Printf.sprintf "cannot access model-program artifact %s: %s"
                relative (Unix.error_message error)))
  in
  check artifacts
