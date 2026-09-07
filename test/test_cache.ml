open Test_support

(* Test_cache: decode-buffer replay and serving-cache state-plan tests
   (LFM2.5 / Gemma heterogeneous layouts, prefix caches, speculative
   slot reservation), serving specialization, and decode uniforms. *)

let run () =

  let replay_buffers =
    Serving_replay.Decode_buffers.create
      ~attention:[ "attention-0", ("key", 1), ("value", 2) ]
      ~recurrent:[ "recurrent-0", ("state", 3) ]
  in
  let replay_buffers =
    Serving_replay.Decode_buffers.update_attention replay_buffers ~f:(function
      | "attention-0" -> Ok (4, 5)
      | binding -> Error ("unexpected attention binding: " ^ binding))
    |> expect_ok
  in
  expect
    (Serving_replay.Decode_buffers.inputs replay_buffers
    = [ "key", 4; "value", 5; "state", 3 ])
    "decode replay preserves recurrent state while attention advances";

  (* Serving cache from state plan tests *)
  let lfm_att, lfm_rec = Lfm25_probe.cache_bindings () in
  let lfm_layout =
    Model_program.State.Cache_layout.create ~attention_layers:6 ~kv_heads:8
      ~head_dim:64 ~recurrent_layers:10 ~recurrent_dim:1024
      ~recurrent_window:3
    |> Result.get_ok
  in
  let lfm_state =
    Model_program.State.create ~layout:lfm_layout ~attentions:lfm_att
      ~recurrents:lfm_rec
    |> Result.get_ok
  in
  let cache_cfg =
    Serving_cache.Config.of_state_plan ~state:lfm_state ~token_capacity:1024
      ~checkpoint_capacity:16 ~page_size:16 ()
    |> Result.get_ok
  in
  let kv_cfg = Serving_cache.Config.kv cache_cfg in
  let kv_layout = Kv_cache.Config.layout kv_cfg in
  expect (Kv_cache.Layout.attention_layers kv_layout = 6) "cache layout has 6 attention layers";
  expect (Kv_cache.Layout.recurrent_layers kv_layout = 10) "cache layout has 10 recurrent layers";
  expect (Kv_cache.Layout.bytes_per_checkpoint kv_layout > 0) "cache layout checkpoint bytes positive for hybrid";

  (* Transformer-only state plan (zero recurrent layers) *)
  let tf_att0 =
    Model_program.State.Attention_binding.create ~cache_layer:0
      ~key_input:"k0" ~value_input:"v0" ~key_output:"kout0" ~value_output:"vout0"
    |> Result.get_ok
  in
  let tf_att1 =
    Model_program.State.Attention_binding.create ~cache_layer:1
      ~key_input:"k1" ~value_input:"v1" ~key_output:"kout1" ~value_output:"vout1"
    |> Result.get_ok
  in
  let tf_layout =
    Model_program.State.Cache_layout.create ~attention_layers:2 ~kv_heads:8
      ~head_dim:64 ~recurrent_layers:0 ~recurrent_dim:0
      ~recurrent_window:0
    |> Result.get_ok
  in
  let tf_state =
    Model_program.State.create ~layout:tf_layout
      ~attentions:[ tf_att0; tf_att1 ] ~recurrents:[]
    |> Result.get_ok
  in
  let tf_cache_cfg =
    Serving_cache.Config.of_state_plan ~state:tf_state ~token_capacity:1024
      ~checkpoint_capacity:16 ~page_size:16 ()
    |> Result.get_ok
  in
  let tf_kv_layout = Kv_cache.Config.layout (Serving_cache.Config.kv tf_cache_cfg) in
  expect (Kv_cache.Layout.attention_layers tf_kv_layout = 2) "transformer cache layout has 2 attention layers";
  expect (Kv_cache.Layout.recurrent_layers tf_kv_layout = 0) "transformer cache layout has 0 recurrent layers";
  expect (Kv_cache.Layout.bytes_per_checkpoint tf_kv_layout = 0) "transformer cache layout has 0 checkpoint bytes";

  let tf_prefix_cfg =
    Serving_cache.Config.of_state_plan ~state:tf_state ~token_capacity:5
      ~checkpoint_capacity:1 ~page_size:1 ()
    |> Result.get_ok
  in
  let tf_prefix_cache = Serving_cache.create tf_prefix_cfg in
  let tf_prefix_slots = ref [||] in
  Unix.putenv "LLMOPT_EXPERIMENTAL_RADIX_OFFSET_INSERT" "1";
  for token = 1 to 3 do
    let slot = Serving_cache.reserve_tokens tf_prefix_cache 1 |> Result.get_ok in
    tf_prefix_slots := Array.append !tf_prefix_slots slot;
    Serving_cache.insert tf_prefix_cache
      ~tokens:(Array.init token (fun index -> index + 1))
      ~slots:!tf_prefix_slots ()
    |> Result.get_ok |> ignore
  done;
  Unix.putenv "LLMOPT_EXPERIMENTAL_RADIX_OFFSET_INSERT" "";
  let tf_prefix_match =
    Serving_cache.match_prefix tf_prefix_cache ~reserve_tail:0 [| 1; 2; 3 |]
  in
  expect
    (Serving_cache.Match.tokens tf_prefix_match = 3
    && Serving_cache.Match.checkpoint tf_prefix_match = None
    && Array.for_all2
         (fun actual expected ->
           Kv_cache.Slot.to_int actual = Kv_cache.Slot.to_int expected)
         (Serving_cache.Match.slots tf_prefix_match) !tf_prefix_slots)
    "transformer prefix remains resumable without recurrent checkpoints";
  Serving_cache.release_match tf_prefix_cache tf_prefix_match |> Result.get_ok;
  Unix.putenv "LLMOPT_EXPERIMENTAL_RADIX_OFFSET_WALK" "1";
  let tf_offset_prefix_match =
    Serving_cache.match_prefix tf_prefix_cache ~reserve_tail:0 [| 1; 2; 3 |]
  in
  expect
    (Serving_cache.Match.tokens tf_offset_prefix_match = 3
    && Serving_cache.Match.checkpoint tf_offset_prefix_match = None)
    "offset radix walk matches the same transformer prefix";
  Serving_cache.release_match tf_prefix_cache tf_offset_prefix_match
  |> Result.get_ok;
  Unix.putenv "LLMOPT_EXPERIMENTAL_RADIX_OFFSET_WALK" "";
  Unix.putenv "LLMOPT_EXPERIMENTAL_RADIX_DIRECT_VALUES" "1";
  let tf_direct_values_match =
    Serving_cache.match_prefix tf_prefix_cache ~reserve_tail:0 [| 1; 2; 3 |]
  in
  expect
    (Serving_cache.Match.tokens tf_direct_values_match = 3
    && Array.for_all2
         (fun actual expected ->
           Kv_cache.Slot.to_int actual = Kv_cache.Slot.to_int expected)
         (Serving_cache.Match.slots tf_direct_values_match) !tf_prefix_slots)
    "direct radix value assembly preserves canonical slot order";
  expect
    (Array.for_all2
       (fun actual expected ->
         Kv_cache.Slot.to_int actual = Kv_cache.Slot.to_int expected)
       (Serving_cache.Match.slots_view tf_direct_values_match) !tf_prefix_slots)
    "borrowed radix values preserve canonical slot order";
  Serving_cache.release_match tf_prefix_cache tf_direct_values_match
  |> Result.get_ok;
  Unix.putenv "LLMOPT_EXPERIMENTAL_RADIX_DIRECT_VALUES" "";
  let fourth_slot =
    Serving_cache.reserve_tokens tf_prefix_cache 1 |> Result.get_ok
  in
  let four_slots = Array.append !tf_prefix_slots fourth_slot in
  let inserted_tokens, inserted_match =
    Serving_cache.insert_and_match tf_prefix_cache ~tokens:[| 1; 2; 3; 4 |]
      ~slots:four_slots ()
    |> Result.get_ok
  in
  expect (inserted_tokens = 3)
    "insert-and-match reports the existing canonical prefix length";
  expect (Serving_cache.Match.tokens inserted_match = 4)
    "insert-and-match leases the complete inserted prefix";
  expect
    (Array.for_all2
       (fun actual expected ->
         Kv_cache.Slot.to_int actual = Kv_cache.Slot.to_int expected)
       (Serving_cache.Match.slots_view inserted_match) four_slots)
    "insert-and-match preserves canonical slot order";
  expect
    ((Serving_cache.stats tf_prefix_cache).radix.protected_tokens = 4)
    "insert-and-match protects the complete active prefix";
  let fifth_slot =
    Serving_cache.reserve_tokens tf_prefix_cache 1 |> Result.get_ok
  in
  let extended_match =
    Serving_cache.extend_match tf_prefix_cache inserted_match ~token:5
      ~slot:fifth_slot.(0) ()
    |> Result.get_ok
  in
  Serving_cache.release_match tf_prefix_cache inserted_match |> Result.get_ok;
  expect
    (Serving_cache.Match.tokens extended_match = 5
    && (Serving_cache.stats tf_prefix_cache).radix.protected_tokens = 5)
    "lease extension advances and protects the active prefix without a root walk";
  Serving_cache.release_match tf_prefix_cache extended_match |> Result.get_ok;
  expect
    ((Serving_cache.stats tf_prefix_cache).kv.used_checkpoints = 0)
    "transformer prefixes do not consume checkpoint identifiers";
  expect (Result.is_ok (Serving_cache.validate tf_prefix_cache))
    "transformer prefix cache remains valid beyond checkpoint capacity";

  (* Gemma 4 target cache: five sliding layers followed by one global layer,
     repeated eight times. Geometry and Q8 physical regions are per-layer. *)
  let gemma_attention_layouts =
    List.init 48 (fun layer ->
        if (layer + 1) mod 6 = 0 then
          Model_program.State.Cache_layout.Attention_layer.create ~kv_heads:1
            ~head_dim:512 ()
          |> Result.get_ok
        else
          Model_program.State.Cache_layout.Attention_layer.create ~kv_heads:8
            ~head_dim:256 ()
          |> Result.get_ok)
  in
  let gemma_bindings =
    List.init 48 (fun layer ->
        Model_program.State.Attention_binding.create ~cache_layer:layer
          ~key_input:(Printf.sprintf "gemma_k_%d" layer)
          ~value_input:(Printf.sprintf "gemma_v_%d" layer)
          ~key_output:(Printf.sprintf "gemma_k_out_%d" layer)
          ~value_output:(Printf.sprintf "gemma_v_out_%d" layer)
        |> Result.get_ok)
  in
  let gemma_layout =
    Model_program.State.Cache_layout.create_heterogeneous
      ~attentions:gemma_attention_layouts ~recurrent_layers:0
      ~recurrent_dim:0 ~recurrent_window:0
    |> Result.get_ok
  in
  expect
    (Model_program.State.Cache_layout.uniform_attention gemma_layout = None)
    "Gemma mixed sliding/global layout is heterogeneous";
  let gemma_state =
    Model_program.State.create ~layout:gemma_layout
      ~attentions:gemma_bindings ~recurrents:[]
    |> Result.get_ok
  in
  let identity =
    expect_ok
      (Model_program.Identity.create ~model:"test-model"
         ~architecture:"transformer" ~family:"llama" ())
  in
  let tok_art = expect_ok (Model_program.Artifact.create "tokenizer.llmopt") in
  let chat =
    expect_ok
      (Model_program.Processor.Chat.create ~bos_token_id:1
         ~format:Model_program.Processor.Chat.Chatml
         ~message_start_token_id:6 ~message_end_token_id:7)
  in
  let processor = Model_program.Processor.create ~tokenizer:tok_art ~chat () in
  let prefill_pkg =
    expect_ok (Model_program.Artifact.create "prefill/package.llmopt")
  in
  let decode_pkg =
    expect_ok (Model_program.Artifact.create "decode/package.llmopt")
  in
  let head =
    expect_ok
      (Model_program.Entrypoint.Head.create ~logits:"logits"
         ~token_id:"token_id" ())
  in
  let prefill_entry =
    expect_ok
      (Model_program.Entrypoint.create ~kind:Model_program.Entrypoint.Prefill
         ~package:prefill_pkg ~input_ids:"input_ids" ~head)
  in
  let decode_entry =
    expect_ok
      (Model_program.Entrypoint.create ~kind:Model_program.Entrypoint.Decode
         ~package:decode_pkg ~input_ids:"input_ids" ~head)
  in
  let generation =
    expect_ok
      (Model_program.Generation.create ~vocab_size:32000 ~max_positions:4096
         ~eos_token_id:2 ~bos_token_id:1 ~suppress_token_ids:[ 883; 882 ] ())
  in
  let specialization =
    expect_ok
      (Model_program.Specialization.create ~min_prefill_tokens:3
         ~rope_cosine_input:"cos" ~rope_sine_input:"sin"
         ~paged_slots_input:"slots" ())
  in
  let gemma_program =
    Model_program.create ~identity ~processor ~prefill:prefill_entry
      ~decode:decode_entry ~generation ~state:gemma_state ~specialization
    |> Result.get_ok
    |> Model_program.to_bytes |> Model_program.of_bytes |> Result.get_ok
  in
  let restored_gemma_layout =
    Model_program.state gemma_program |> Model_program.State.layout
  in
  let restored_global =
    Model_program.State.Cache_layout.attention restored_gemma_layout
      ~cache_layer:5
    |> Option.get
  in
  expect
    (Model_program.State.Cache_layout.Attention_layer.kv_heads restored_global
       = 1
    && Model_program.State.Cache_layout.Attention_layer.head_dim
         restored_global
       = 512)
    "ABI-v3 round-trip preserves the first Gemma global layer geometry";
  let restored_sliding =
    Model_program.State.Cache_layout.attention restored_gemma_layout
      ~cache_layer:6
    |> Option.get
  in
  expect
    (Model_program.State.Cache_layout.Attention_layer.kv_heads restored_sliding
       = 8
    && Model_program.State.Cache_layout.Attention_layer.head_dim
         restored_sliding
       = 256)
    "ABI-v3 round-trip preserves Gemma sliding geometry after a global layer";
  let gemma_cache_cfg =
    Serving_cache.Config.of_state_plan ~state:gemma_state ~token_capacity:8
      ~checkpoint_capacity:1 ~page_size:16 ()
    |> Result.get_ok
  in
  let gemma_kv_layout =
    Serving_cache.Config.kv gemma_cache_cfg |> Kv_cache.Config.layout
  in
  expect (Kv_cache.Layout.token_elements gemma_kv_layout = 172_032)
    "Gemma heterogeneous cache counts exact logical token elements";
  expect (Kv_cache.Layout.bytes_per_token gemma_kv_layout = 177_408)
    "Gemma heterogeneous cache counts exact Q8 token bytes";
  expect
    (Kv_cache.Layout.attention_key_offset gemma_kv_layout ~layer:5
       = Some 20_480
    && Kv_cache.Layout.attention_value_offset gemma_kv_layout ~layer:5
       = Some 20_992
    && Kv_cache.Layout.attention_key_offset gemma_kv_layout ~layer:6
       = Some 21_504)
    "Gemma global layer occupies its exact physical value offsets";
  expect
    (Kv_cache.Config.token_pool_bytes
       (Serving_cache.Config.kv gemma_cache_cfg)
    = 1_419_264)
    "Gemma cache allocates exact heterogeneous token-pool bytes";
  let gemma_cache = Serving_cache.create gemma_cache_cfg in
  let gemma_rollback_slots =
    Serving_cache.reserve_tokens gemma_cache 4 |> Result.get_ok
  in
  let gemma_rollback =
    Serving_cache.Speculative.create ~tokens:[| 1; 2; 3; 4 |]
      ~slots:gemma_rollback_slots ()
  in
  expect
    (Serving_cache.rollback_speculative gemma_cache
       ~reservation:gemma_rollback
    = Ok ())
    "Gemma heterogeneous speculative reservation rolls back";
  let gemma_commit_slots =
    Serving_cache.reserve_tokens gemma_cache 4 |> Result.get_ok
  in
  let gemma_commit =
    Serving_cache.Speculative.create ~tokens:[| 5; 6; 7; 8 |]
      ~slots:gemma_commit_slots ()
  in
  expect
    (Serving_cache.commit_speculative gemma_cache ~reservation:gemma_commit
       ~accepted_count:3
    = Ok 3)
    "Gemma heterogeneous speculative reservation commits its accepted prefix";
  expect
    ((Serving_cache.stats gemma_cache).kv.allocated_bytes = 3 * 177_408)
    "Gemma partial commit accounts for accepted physical cache bytes";
  Serving_cache.release_tokens gemma_cache (Array.sub gemma_commit_slots 0 3)
  |> Result.get_ok;
  expect (Result.is_ok (Serving_cache.validate gemma_cache))
    "Gemma heterogeneous cache remains valid after commit and rollback";

  (* Speculative slot reservation and rollback test *)
  let spec_cache = Serving_cache.create tf_cache_cfg in
  let initial_stats = Serving_cache.stats spec_cache in
  expect (initial_stats.kv.used_tokens = 0) "initial cache has 0 used tokens";

  (* 1. Full rollback test *)
  let tentative_slots = Serving_cache.reserve_tokens spec_cache 4 |> Result.get_ok in
  expect (Array.length tentative_slots = 4) "reserved 4 speculative slots";
  let after_reserve_stats = Serving_cache.stats spec_cache in
  expect (after_reserve_stats.kv.used_tokens = 4) "4 slots used after reservation";

  let reservation =
    Serving_cache.Speculative.create
      ~tokens:[| 101; 102; 103; 104 |]
      ~slots:tentative_slots ()
  in
  let rollback_res = Serving_cache.rollback_speculative spec_cache ~reservation in
  expect (Result.is_ok rollback_res) "speculative rollback succeeds";
  let after_rollback_stats = Serving_cache.stats spec_cache in
  expect (after_rollback_stats.kv.used_tokens = 0) "0 slots used after full rollback";

  (* 2. Partial acceptance test (accept 2 of 4) *)
  let tentative_slots2 = Serving_cache.reserve_tokens spec_cache 4 |> Result.get_ok in
  let reservation2 =
    Serving_cache.Speculative.create
      ~tokens:[| 201; 202; 203; 204 |]
      ~slots:tentative_slots2 ()
  in
  let commit_res = Serving_cache.commit_speculative spec_cache ~reservation:reservation2 ~accepted_count:2 in
  expect (commit_res = Ok 2) "committed 2 accepted speculative tokens";
  let after_commit_stats = Serving_cache.stats spec_cache in
  expect (after_commit_stats.kv.used_tokens = 2) "2 slots used and 2 released after partial acceptance";
  (* Clean up the 2 accepted slots *)
  let _ = Serving_cache.release_tokens spec_cache (Array.sub tentative_slots2 0 2) in

  (* 3. 100 random speculative draft/verify cycles *)
  for cycle = 1 to 100 do
    let draft_count = 1 + (cycle mod 5) in
    let slots = Serving_cache.reserve_tokens spec_cache draft_count |> Result.get_ok in
    let tokens = Array.init draft_count (fun i -> 1000 * cycle + i) in
    let res = Serving_cache.Speculative.create ~tokens ~slots () in
    let accepted = cycle mod (draft_count + 1) in
    let commit_res = Serving_cache.commit_speculative spec_cache ~reservation:res ~accepted_count:accepted in
    expect (commit_res = Ok accepted) "random speculative cycle matches accepted count";
    if accepted > 0 then
      let _ = Serving_cache.release_tokens spec_cache (Array.sub slots 0 accepted) in ()
  done;
  let final_stats = Serving_cache.stats spec_cache in
  expect (final_stats.kv.used_tokens = 0) "zero leaked slots after 100 speculative cycles";
  expect (Result.is_ok (Serving_cache.validate spec_cache)) "serving cache remains valid after 100 speculative cycles";

  (* Serving specialization tests *)
  let spec =
    Model_program.Specialization.create ~min_prefill_tokens:3
      ~rope_cosine_input:"custom_cos" ~rope_sine_input:"custom_sin"
      ~paged_slots_input:"custom_slots" ()
    |> Result.get_ok
  in
  let empty_sched =
    Serving_schedule.of_graph (Ir.Graph.create ())
    |> Result.get_ok
  in
  let pref_err =
    Serving_specialization.prefill ~specialization:spec ~captured_tokens:3
      ~tokens:2 empty_sched
  in
  expect (Result.is_error pref_err)
    "specialization rejects prefill tokens < min_prefill_tokens";
  let suf_err =
    Serving_specialization.suffix_prefill_paged_q8 ~specialization:spec
      ~captured_tokens:3 ~tokens:1 ~past_tokens:5
      ~cache:(Serving_cache.Config.kv cache_cfg) empty_sched
  in
  expect (Result.is_error suf_err)
    "specialization rejects suffix prefill tokens < min_prefill_tokens";
  expect
    (Serving_specialization.recurrent_in_input 0 = "__llmopt_recurrent_in_0")
    "recurrent_in_input 0 produces __llmopt_recurrent_in_0";
  expect
    (Serving_specialization.recurrent_in_input 5 = "__llmopt_recurrent_in_5")
    "recurrent_in_input 5 produces __llmopt_recurrent_in_5";
  expect
    (Result.is_error (Metal_runtime.Decode_uniforms.create ~past_tokens:0))
    "decode uniforms reject an empty prefix";
  let decode_uniforms =
    Metal_runtime.Decode_uniforms.create ~past_tokens:37 |> Result.get_ok
  in
  expect
    (Metal_runtime.Decode_uniforms.past_tokens decode_uniforms = 37)
    "decode uniforms own the dispatch-time past-token count"
