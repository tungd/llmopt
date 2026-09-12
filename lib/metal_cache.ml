let ( let* ) = Result.bind

open Metal_native
module Parameters = Metal_parameters

module Attention = struct
  type t = Key | Value
end

module Pack_layout = struct
  type t = Scalar | Simdgroup
end

module Unpack_layout = struct
  type t = Scalar | Vec4
end

type pack_kernel = {
  entry : Kernel_abi.Entry.t;
  dispatch_layout : Pack_layout.t;
}

type unpack_kernel = {
  entry : Kernel_abi.Entry.t;
  unpack_layout : Unpack_layout.t;
}

type kernels = {
  pack_attention : pack_kernel;
  pack_attention_pair : pack_kernel option;
  unpack_attention : unpack_kernel;
  pack_checkpoint : pack_kernel;
  unpack_checkpoint : unpack_kernel;
}

type t = {
  runtime : runtime;
  config : Kv_cache.Config.t;
  layout : Kv_cache.Layout.t;
  token_pool : Buffer.t;
  checkpoint_pool : Buffer.t;
  slots_scratch : Buffer.t;
  pack_slots_scratch : Buffer.t;
  kernels : kernels;
}

type batch = {
  cache : t;
  commands : Batch.t;
  mutable resources : Buffer.t list;
  mutable dispatches : int;
}

let release_batch_resources batch = batch.resources <- []

let with_batch cache operation =
  let* commands = Batch.create cache.runtime in
  let batch = { cache; commands; resources = []; dispatches = 0 } in
  match operation batch with
  | Error message ->
      ignore (Batch.abort commands);
      release_batch_resources batch;
      Error message
  | Ok value ->
      let completion =
        if batch.dispatches = 0 then Batch.abort commands
        else Batch.commit commands
      in
      release_batch_resources batch;
      let* () = completion in
      Ok value
  | exception exception_value ->
      ignore (Batch.abort commands);
      release_batch_resources batch;
      raise exception_value

let with_batch_async cache operation =
  let* commands = Batch.create cache.runtime in
  let batch = { cache; commands; resources = []; dispatches = 0 } in
  match operation batch with
  | Error message ->
      ignore (Batch.abort commands);
      release_batch_resources batch;
      Error message
  | Ok value ->
      let completion =
        if batch.dispatches = 0 then Batch.abort commands
        else Batch.commit_async commands
      in
      release_batch_resources batch;
      let* () = completion in
      Ok value
  | exception exception_value ->
      ignore (Batch.abort commands);
      release_batch_resources batch;
      raise exception_value

let format cache = Kv_cache.Layout.format cache.layout
let token_pool_bytes cache = Kv_cache.Config.token_pool_bytes cache.config

let checkpoint_pool_bytes cache =
  Kv_cache.Config.checkpoint_pool_bytes cache.config

let kernel_names _format =
  ( "llmopt_cache_pack_attention_q8",
    "llmopt_cache_unpack_attention_q8",
    "llmopt_cache_pack_checkpoint_q8",
    "llmopt_cache_unpack_checkpoint_q8" )

let kernel_dtypes _format =
  ( (Ir.Dtype.Float16, Ir.Dtype.Int8),
    (Ir.Dtype.Int8, Ir.Dtype.Float16) )

let cache_entry runtime name (input_dtype, output_dtype) =
  kernel_entry ~name runtime ~operation:Kernel_abi.Operation.Cache
    ~input_dtype ~output_dtype

let cache_entry_opt runtime name (input_dtype, output_dtype) =
  Serving_package.kernels runtime.package
  |> List.find_opt (fun entry ->
         Kernel_abi.Entry.name entry = name
         && Kernel_abi.Entry.operation entry = Kernel_abi.Operation.Cache
         && Kernel_abi.Entry.input_dtype entry = input_dtype
         && Kernel_abi.Entry.output_dtype entry = output_dtype)

let select_pack_kernel runtime _format scalar_name dtypes =
  let simd_name = scalar_name ^ "_simd" in
  match cache_entry_opt runtime simd_name dtypes with
  | Some entry -> Ok { entry; dispatch_layout = Pack_layout.Simdgroup }
  | None ->
      let* entry = cache_entry runtime scalar_name dtypes in
      Ok { entry; dispatch_layout = Pack_layout.Scalar }

let select_unpack_kernel runtime _format scalar_name dtypes =
  let vector_name = scalar_name ^ "_vec4" in
  match cache_entry_opt runtime vector_name dtypes with
  | Some entry -> Ok { entry; unpack_layout = Unpack_layout.Vec4 }
  | None ->
      let* entry = cache_entry runtime scalar_name dtypes in
      Ok { entry; unpack_layout = Unpack_layout.Scalar }

let create ~runtime ~config =
  if Serving_package.stage runtime.package <> Serving_package.Stage.Serving then
    Error "physical Metal cache requires a serving-stage package"
  else
    let layout = Kv_cache.Config.layout config in
    let format = Kv_cache.Layout.format layout in
    let pack_attention, unpack_attention, pack_checkpoint,
          unpack_checkpoint =
        kernel_names format
      in
      let pack_dtype, unpack_dtype = kernel_dtypes format in
      let* pack_attention =
        select_pack_kernel runtime format pack_attention pack_dtype
      in
      let pack_attention_pair =
        cache_entry_opt runtime "llmopt_cache_pack_attention_q8_pair_simd"
          pack_dtype
        |> Option.map (fun entry ->
               { entry; dispatch_layout = Pack_layout.Simdgroup })
      in
      let* unpack_attention =
        select_unpack_kernel runtime format unpack_attention unpack_dtype
      in
      let* pack_checkpoint =
        select_pack_kernel runtime format pack_checkpoint pack_dtype
      in
      let* unpack_checkpoint =
        select_unpack_kernel runtime format unpack_checkpoint unpack_dtype
      in
      let* token_pool =
        Buffer.create ~runtime ~bytes:(Kv_cache.Config.token_pool_bytes config)
      in
      let* checkpoint_pool =
        Buffer.create ~runtime
          ~bytes:(Kv_cache.Config.checkpoint_pool_bytes config)
      in
      let* slots_scratch =
        Buffer.create ~runtime
          ~bytes:(max 32768 (Kv_cache.Config.token_capacity config * 4))
      in
      let* pack_slots_scratch =
        Buffer.create ~runtime
          ~bytes:(max 32768 (Kv_cache.Config.token_capacity config * 4))
      in
    Ok
      {
          runtime;
          config;
          layout;
          token_pool;
          checkpoint_pool;
          slots_scratch;
          pack_slots_scratch;
          kernels =
            {
              pack_attention;
              pack_attention_pair;
              unpack_attention;
              pack_checkpoint;
              unpack_checkpoint;
            };
      }

let checked_product left right label =
  if left < 0 || right < 0 || (right <> 0 && left > max_int / right) then
    Error ("physical cache " ^ label ^ " overflows")
  else Ok (left * right)

let pack_grid kernel logical_groups =
  match kernel.dispatch_layout with
  | Pack_layout.Scalar -> Ok logical_groups
  | Pack_layout.Simdgroup ->
      let simdgroups_per_threadgroup = 8 in
      if logical_groups > max_int - (simdgroups_per_threadgroup - 1) then
        Error "physical cache SIMD pack grid overflows"
      else
        let threadgroups =
          (logical_groups + simdgroups_per_threadgroup - 1)
          / simdgroups_per_threadgroup
        in
        checked_product threadgroups 256 "SIMD pack grid"

let unpack_grid kernel elements =
  match kernel.unpack_layout with
  | Unpack_layout.Scalar -> elements
  | Unpack_layout.Vec4 -> elements / 4

let exact_buffer_length buffer expected label =
  let actual = Buffer.byte_length buffer in
  if actual = expected then Ok ()
  else
    Error
      (Printf.sprintf "physical cache %s requires %d bytes but received %d"
         label expected actual)

let validate_layer layer count label =
  if layer < 0 || layer >= count then
    Error
      (Printf.sprintf "physical cache %s layer %d is outside [0,%d)" label
         layer count)
  else Ok ()

let group_size format =
  Kv_cache.Format.group_size format |> Option.value ~default:1

let slots_buffer ?(pack = false) cache slots =
  let count = Array.length slots in
  if count = 0 then Error "physical attention cache requires at least one slot"
  else
    let capacity = Kv_cache.Config.token_capacity cache.config in
    if count > capacity then
      Error
        (Printf.sprintf "physical cache slots count %d exceeds capacity %d"
           count capacity)
    else
      let scratch =
        if pack then cache.pack_slots_scratch else cache.slots_scratch
      in
      let int_slots = Array.map Kv_cache.Slot.to_int slots in
      let* () = Buffer.set_u32_array scratch ~offset:0 int_slots in
      if pack then Buffer.view ~parent:scratch ~offset:0 ~bytes:(count * 4)
      else Ok scratch

let attention_parameters cache ~layer ~kind ~items ~source_items
    ~source_offset =
  let layout = cache.layout in
  let* geometry =
    match Kv_cache.Layout.attention layout ~layer with
    | Some geometry -> Ok geometry
    | None -> Error "physical cache attention layer is outside the layout"
  in
  let heads = Kv_cache.Layout.Attention_layer.kv_heads geometry in
  let head_dim = Kv_cache.Layout.Attention_layer.head_dim geometry in
  let group_size = group_size (format cache) in
  let token_elements = Kv_cache.Layout.token_elements layout in
  let segment_offset =
    match kind with
    | Attention.Key -> Kv_cache.Layout.attention_key_offset layout ~layer
    | Attention.Value -> Kv_cache.Layout.attention_value_offset layout ~layer
  in
  let* segment_offset =
    match segment_offset with
    | Some offset -> Ok offset
    | None -> Error "physical cache attention region is missing"
  in
  let token_groups =
    Kv_cache.Format.groups_for_elements (format cache)
      ~elements:token_elements
  in
  Parameters.u32s
    [ items; segment_offset; heads; head_dim; group_size; token_elements;
      token_groups;
      Kv_cache.Layout.bytes_per_token layout; source_items; source_offset ]

let attention_pair_parameters cache ~layer ~items ~source_items
    ~source_offset =
  let layout = cache.layout in
  let* geometry =
    match Kv_cache.Layout.attention layout ~layer with
    | Some geometry -> Ok geometry
    | None -> Error "physical cache attention layer is outside the layout"
  in
  let* key_offset, value_offset =
    match
      ( Kv_cache.Layout.attention_key_offset layout ~layer,
        Kv_cache.Layout.attention_value_offset layout ~layer )
    with
    | Some key_offset, Some value_offset -> Ok (key_offset, value_offset)
    | _ -> Error "physical cache attention region is missing"
  in
  let heads = Kv_cache.Layout.Attention_layer.kv_heads geometry in
  let head_dim = Kv_cache.Layout.Attention_layer.head_dim geometry in
  let group_size = group_size (format cache) in
  let token_elements = Kv_cache.Layout.token_elements layout in
  let token_groups =
    Kv_cache.Format.groups_for_elements (format cache)
      ~elements:token_elements
  in
  Parameters.u32s
    [ items; key_offset; value_offset; heads; head_dim; group_size;
      token_elements; token_groups; Kv_cache.Layout.bytes_per_token layout;
      source_items; source_offset ]

let attention_counts cache ~layer items =
  let layout = cache.layout in
  let* segment_elements =
    match Kv_cache.Layout.attention_segment_elements layout ~layer with
    | Some elements -> Ok elements
    | None -> Error "physical cache attention layer is outside the layout"
  in
  let* elements =
    checked_product items segment_elements "attention transfer size"
  in
  let groups = elements / Kv_cache.Format.q8_group_size in
  Ok (elements, groups)

let attention ?batch cache ~pack ~layer ~kind ~slots ~source_items
    ~source_offset buffer =
  let* () =
    validate_layer layer (Kv_cache.Layout.attention_layers cache.layout)
      "attention"
  in
  let items = Array.length slots in
  let* elements, groups = attention_counts cache ~layer items in
  let* expected_elements =
    if pack then
      if source_items <= 0 then
        Error "physical attention cache source_items must be positive"
      else if source_offset < 0 || source_offset > source_items - items then
        Error "physical attention cache source slice is out of range"
      else
        let* segment_elements =
          match
            Kv_cache.Layout.attention_segment_elements cache.layout ~layer
          with
          | Some elements -> Ok elements
          | None ->
              Error "physical cache attention layer is outside the layout"
        in
        let* source_elements =
          checked_product source_items segment_elements
            "attention source size"
        in
        Ok source_elements
    else Ok elements
  in
  let* expected_bytes =
    checked_product expected_elements 2 "attention byte length"
  in
  let* () =
    exact_buffer_length buffer expected_bytes
      (if pack then "attention source" else "attention destination")
  in
  let* slots_buffer = slots_buffer ~pack cache slots in
  let* parameters =
    attention_parameters cache ~layer ~kind ~items ~source_items
      ~source_offset
  in
  let entry =
    if pack then cache.kernels.pack_attention.entry
    else cache.kernels.unpack_attention.entry
  in
  let buffers =
    if pack then [ buffer; slots_buffer; cache.token_pool ]
    else [ cache.token_pool; slots_buffer; buffer ]
  in
  let* grid =
    if pack then pack_grid cache.kernels.pack_attention groups
    else Ok (unpack_grid cache.kernels.unpack_attention elements)
  in
  Option.iter
    (fun batch -> batch.resources <- slots_buffer :: batch.resources)
    batch;
  let* kernel =
    dispatch ?batch:(Option.map (fun batch -> batch.commands) batch)
      cache.runtime entry ~buffers ~parameters ~grid:(grid, 1, 1)
  in
  Option.iter
    (fun batch -> batch.dispatches <- batch.dispatches + 1)
    batch;
  Ok kernel

let pack_attention cache ~layer ~kind ~slots ~source =
  attention cache ~pack:true ~layer ~kind ~slots
    ~source_items:(Array.length slots) ~source_offset:0 source

let pack_attention_slice cache ~layer ~kind ~slots ~source_items
    ~source_offset ~source =
  if Serving_package.abi_version cache.runtime.package < 8 then
    Error "attention cache source slicing requires serving-package ABI v8"
  else
    attention cache ~pack:true ~layer ~kind ~slots ~source_items ~source_offset
      source

let unpack_attention cache ~layer ~kind ~slots ~destination =
  attention cache ~pack:false ~layer ~kind ~slots
    ~source_items:(Array.length slots) ~source_offset:0 destination

let batch_pack_attention batch ~layer ~kind ~slots ~source =
  attention ~batch batch.cache ~pack:true ~layer ~kind ~slots
    ~source_items:(Array.length slots) ~source_offset:0 source

let batch_pack_attention_slice batch ~layer ~kind ~slots ~source_items
    ~source_offset ~source =
  if Serving_package.abi_version batch.cache.runtime.package < 8 then
    Error "attention cache source slicing requires serving-package ABI v8"
  else
    attention ~batch batch.cache ~pack:true ~layer ~kind ~slots ~source_items
      ~source_offset source

let batch_pack_attention_pair_slice batch ~layer ~slots ~source_items
    ~source_offset ~key_source ~value_source =
  match batch.cache.kernels.pack_attention_pair with
  | None ->
      let* _ =
        batch_pack_attention_slice batch ~layer ~kind:Attention.Key ~slots
          ~source_items ~source_offset ~source:key_source
      in
      batch_pack_attention_slice batch ~layer ~kind:Attention.Value ~slots
        ~source_items ~source_offset ~source:value_source
  | Some kernel ->
      let cache = batch.cache in
      let* () =
        validate_layer layer (Kv_cache.Layout.attention_layers cache.layout)
          "attention"
      in
      let items = Array.length slots in
      let* _, groups = attention_counts cache ~layer items in
      let* segment_elements =
        match Kv_cache.Layout.attention_segment_elements cache.layout ~layer with
        | Some elements -> Ok elements
        | None -> Error "physical cache attention layer is outside the layout"
      in
      let* () =
        if source_items <= 0 then
          Error "physical attention cache source_items must be positive"
        else if source_offset < 0 || source_offset > source_items - items then
          Error "physical attention cache source slice is out of range"
        else Ok ()
      in
      let* source_elements =
        checked_product source_items segment_elements "attention source size"
      in
      let* expected_bytes =
        checked_product source_elements 2 "attention byte length"
      in
      let* () =
        exact_buffer_length key_source expected_bytes "attention key source"
      in
      let* () =
        exact_buffer_length value_source expected_bytes "attention value source"
      in
      let* slots_buffer = slots_buffer ~pack:true cache slots in
      let* parameters =
        attention_pair_parameters cache ~layer ~items ~source_items
          ~source_offset
      in
      let* grid = pack_grid kernel groups in
      batch.resources <- slots_buffer :: batch.resources;
      let* dispatched =
        dispatch ~batch:batch.commands cache.runtime kernel.entry
          ~buffers:[ key_source; value_source; slots_buffer; cache.token_pool ]
          ~parameters ~grid:(grid, 1, 1)
      in
      batch.dispatches <- batch.dispatches + 1;
      Ok dispatched

let batch_pack_attention_pair batch ~layer ~slots ~key_source ~value_source =
  match batch.cache.kernels.pack_attention_pair with
  | None ->
      let* _ =
        batch_pack_attention batch ~layer ~kind:Attention.Key ~slots
          ~source:key_source
      in
      batch_pack_attention batch ~layer ~kind:Attention.Value ~slots
        ~source:value_source
  | Some _ ->
      batch_pack_attention_pair_slice batch ~layer ~slots
        ~source_items:(Array.length slots) ~source_offset:0 ~key_source
        ~value_source

let batch_unpack_attention batch ~layer ~kind ~slots ~destination =
  attention ~batch batch.cache ~pack:false ~layer ~kind ~slots
    ~source_items:(Array.length slots) ~source_offset:0 destination

let checkpoint_parameters cache ~layer ~checkpoint =
  let layout = cache.layout in
  let layer_elements =
    Kv_cache.Layout.recurrent_width layout
    * Kv_cache.Layout.recurrent_window layout
  in
  let checkpoint_elements =
    Kv_cache.Layout.recurrent_layers layout * layer_elements
  in
  let checkpoint_groups =
    Kv_cache.Format.groups_for_elements (format cache)
      ~elements:checkpoint_elements
  in
  Parameters.u32s
    [ Kv_cache.Checkpoint.to_int checkpoint; layer; layer_elements;
      group_size (format cache); checkpoint_elements; checkpoint_groups;
      Kv_cache.Layout.bytes_per_checkpoint layout ]

let transfer_checkpoint ?batch cache ~pack ~layer ~checkpoint buffer =
  let* () =
    validate_layer layer (Kv_cache.Layout.recurrent_layers cache.layout)
      "checkpoint"
  in
  let checkpoint_id = Kv_cache.Checkpoint.to_int checkpoint in
  let capacity = Kv_cache.Config.checkpoint_capacity cache.config in
  if checkpoint_id < 0 || checkpoint_id >= capacity then
    Error
      (Printf.sprintf "physical cache checkpoint %d is outside [0,%d)"
         checkpoint_id capacity)
  else
    let* layer_elements =
      checked_product (Kv_cache.Layout.recurrent_width cache.layout)
        (Kv_cache.Layout.recurrent_window cache.layout)
        "checkpoint layer size"
    in
    let* expected_bytes =
      checked_product layer_elements 2 "checkpoint byte length"
    in
    let* () =
      exact_buffer_length buffer expected_bytes
        (if pack then "checkpoint source" else "checkpoint destination")
    in
    let* parameters = checkpoint_parameters cache ~layer ~checkpoint in
    let entry =
      if pack then cache.kernels.pack_checkpoint.entry
      else cache.kernels.unpack_checkpoint.entry
    in
    let buffers =
      if pack then [ buffer; cache.checkpoint_pool ]
      else [ cache.checkpoint_pool; buffer ]
    in
    let* grid =
      if pack then
        pack_grid cache.kernels.pack_checkpoint
          (layer_elements / Kv_cache.Format.q8_group_size)
      else Ok (unpack_grid cache.kernels.unpack_checkpoint layer_elements)
    in
    let* kernel =
      dispatch ?batch:(Option.map (fun batch -> batch.commands) batch)
        cache.runtime entry ~buffers ~parameters ~grid:(grid, 1, 1)
    in
    Option.iter
      (fun batch -> batch.dispatches <- batch.dispatches + 1)
      batch;
    Ok kernel

let pack_checkpoint cache ~layer ~checkpoint ~source =
  transfer_checkpoint cache ~pack:true ~layer ~checkpoint source

let unpack_checkpoint cache ~layer ~checkpoint ~destination =
  transfer_checkpoint cache ~pack:false ~layer ~checkpoint destination

let batch_pack_checkpoint batch ~layer ~checkpoint ~source =
  transfer_checkpoint ~batch batch.cache ~pack:true ~layer ~checkpoint source

let batch_unpack_checkpoint batch ~layer ~checkpoint ~destination =
  transfer_checkpoint ~batch batch.cache ~pack:false ~layer ~checkpoint
    destination

let q8_attention_inputs cache ~slots =
  let* slots = slots_buffer cache slots in
  Ok
    [ Serving_schedule.Sequence.q8_attention_pool_input, cache.token_pool;
      Serving_schedule.Sequence.q8_attention_slots_input, slots ]

let update_pack_slot cache slot =
  Buffer.set_u32_array cache.pack_slots_scratch ~offset:0
    [| Kv_cache.Slot.to_int slot |]
