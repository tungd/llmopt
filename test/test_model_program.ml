open Test_support

(* Test_model_program: Model Program contract/ABI tests, proposal bounds,
   and LFM2.5 probe fixture bindings. *)

let run () =

  (* Model Program contract and ABI tests *)
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
  expect
    (Model_program.Entrypoint.Head.token_id_shape_matches ~tokens:2 [ 2 ])
    "token_id contract accepts the fused flat token shape";
  expect
    (Model_program.Entrypoint.Head.token_id_shape_matches ~tokens:2 [ 1; 2 ])
    "token_id contract accepts the generic batch-preserving token shape";
  expect
    (not
       (Model_program.Entrypoint.Head.token_id_shape_matches ~tokens:2
          [ 2; 1 ]))
    "token_id contract rejects a non-semantic equal-element shape";
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
  expect
    (Result.is_error
       (Model_program.Generation.create ~vocab_size:8 ~max_positions:16
          ~suppress_token_ids:[ 3; 3 ] ()))
    "generation rejects duplicate suppressed token IDs";
  expect
    (Result.is_error
       (Model_program.Generation.create ~vocab_size:8 ~max_positions:16
          ~suppress_token_ids:[ 8 ] ()))
    "generation rejects suppressed token IDs outside the vocabulary";
  let layout =
    expect_ok
      (Model_program.State.Cache_layout.create ~attention_layers:2 ~kv_heads:4
         ~head_dim:64 ~recurrent_layers:1 ~recurrent_dim:128
         ~recurrent_window:3)
  in
  let att0 =
    expect_ok
      (Model_program.State.Attention_binding.create ~cache_layer:0
         ~key_input:"k0" ~value_input:"v0" ~key_output:"out_k0"
         ~value_output:"out_v0")
  in
  let att1 =
    expect_ok
      (Model_program.State.Attention_binding.create ~cache_layer:1
         ~key_input:"k1" ~value_input:"v1" ~key_output:"out_k1"
         ~value_output:"out_v1")
  in
  let rec0 =
    expect_ok
      (Model_program.State.Recurrent_binding.create ~cache_layer:0
         ~state_input:"s0" ~state_output:"out_s0")
  in
  let state =
    expect_ok
      (Model_program.State.create ~layout ~attentions:[ att0; att1 ]
         ~recurrents:[ rec0 ])
  in
  let specialization =
    expect_ok
      (Model_program.Specialization.create ~min_prefill_tokens:3
         ~rope_cosine_input:"cos" ~rope_sine_input:"sin"
         ~paged_slots_input:"slots" ())
  in
  let program =
    expect_ok
      (Model_program.create ~identity ~processor ~prefill:prefill_entry
         ~decode:decode_entry ~generation ~state ~specialization)
  in
  expect
    (Model_program.abi_version program = Model_program.current_abi_version)
    "model program uses the current ABI version";
  expect
    (Model_program.Identity.model (Model_program.identity program)
    = "test-model")
    "model program identity model matches";

  (* Round-trip serialization *)
  let program_bytes = Model_program.to_bytes program in
  let restored = expect_ok (Model_program.of_bytes program_bytes) in
  let obsolete_program = Bytes.copy program_bytes in
  let version_offset = String.length "LLMOPT-MODEL\000" in
  Bytes.set obsolete_program version_offset '\001';
  Bytes.set obsolete_program (version_offset + 1) '\000';
  expect
    (Result.is_error (Model_program.of_bytes obsolete_program))
    "rejects obsolete model-program ABI versions";
  expect
    (Model_program.Identity.model (Model_program.identity restored)
    = "test-model")
    "restored identity model matches";
  expect
    (Model_program.Identity.architecture (Model_program.identity restored)
    = Some "transformer")
    "restored identity architecture matches";
  expect
    (Model_program.Identity.family (Model_program.identity restored)
    = Some "llama")
    "restored identity family matches";
  expect
    (Model_program.Artifact.path
       (Model_program.Processor.tokenizer (Model_program.processor restored))
    = "tokenizer.llmopt")
    "restored tokenizer path matches";
  expect
    (Model_program.Processor.chat (Model_program.processor restored)
    |> Option.map Model_program.Processor.Chat.message_end_token_id
    = Some 7)
    "restored explicit chat token contract matches";
  expect
    (Model_program.Generation.vocab_size (Model_program.generation restored)
    = 32000)
    "restored vocab size matches";
  expect
    (Model_program.Generation.max_positions (Model_program.generation restored)
    = 4096)
    "restored max positions matches";
  expect
    (Model_program.Generation.eos_token_id (Model_program.generation restored)
    = Some 2)
    "restored eos matches";
  expect
    (Model_program.Generation.bos_token_id (Model_program.generation restored)
    = Some 1)
    "restored bos matches";
  expect
    (Model_program.Generation.suppress_token_ids
       (Model_program.generation restored)
    = [ 882; 883 ])
    "restored generation suppression policy matches";
  expect
    (Model_program.State.Cache_layout.attention_layers
       (Model_program.State.layout (Model_program.state restored))
    = 2)
    "restored state layout attention layers matches";
  expect
    (Model_program.State.Cache_layout.recurrent_layers
       (Model_program.State.layout (Model_program.state restored))
    = 1)
    "restored state layout recurrent layers matches";
  expect
    (Model_program.State.Cache_layout.recurrent_window
       (Model_program.State.layout (Model_program.state restored))
    = 3)
    "restored state layout recurrent window matches";
  expect
    (Model_program.Specialization.min_prefill_tokens
       (Model_program.specialization restored)
    = 3)
    "restored specialization min prefill tokens matches";
  expect
    (Model_program.Specialization.rope_cosine_input
       (Model_program.specialization restored)
    = Some "cos")
    "restored specialization rope cosine matches";

  let assistant_pkg =
    Model_program.Artifact.create "assistant/package.llmopt" |> expect_ok
  in
  let mtp =
    Model_program.Mtp.create ~assistant_package:assistant_pkg
      ~target_hidden_output:"target_hidden_state"
      ~coupled_target_state_input:"coupled_target_embedding_and_hidden"
      ~position_ids_input:"position_ids"
      ~full_attention_mask_input:"full_attention_mask"
      ~sliding_attention_mask_input:"sliding_attention_mask"
      ~full_attention_key_input:"shared.full_attention.key"
      ~full_attention_value_input:"shared.full_attention.value"
      ~sliding_attention_key_input:"shared.sliding_attention.key"
      ~sliding_attention_value_input:"shared.sliding_attention.value"
      ~logits_output:"logits"
      ~projected_target_hidden_output:"projected_target_hidden_state"
      ~max_draft_tokens:4
    |> expect_ok
  in
  let mtp_program =
    Model_program.create ~identity ~processor ~prefill:prefill_entry
      ~decode:decode_entry ~generation ~state ~specialization
    |> expect_ok |> fun program -> Model_program.with_mtp program mtp
    |> expect_ok |> Model_program.to_bytes |> Model_program.of_bytes |> expect_ok
  in
  let restored_mtp = Model_program.mtp mtp_program |> Option.get in
  expect
    (Model_program.Artifact.path
       (Model_program.Mtp.assistant_package restored_mtp)
    = "assistant/package.llmopt"
    && Model_program.Mtp.target_hidden_output restored_mtp
       = "target_hidden_state"
    && Model_program.Mtp.coupled_target_state_input restored_mtp
       = "coupled_target_embedding_and_hidden"
    && Model_program.Mtp.projected_target_hidden_output restored_mtp
       = "projected_target_hidden_state"
    && Model_program.Mtp.max_draft_tokens restored_mtp = 4)
    "ABI-v4 round-trip preserves the target-coupled MTP assistant boundary";

  (* Validation rejection tests *)
  expect
    (Result.is_error (Model_program.Artifact.create "/absolute/path"))
    "rejects absolute artifact path";
  expect
    (Result.is_error (Model_program.Artifact.create "path/../escape"))
    "rejects non-canonical artifact path";
  expect
    (Result.is_error (Model_program.Identity.create ~model:"" ()))
    "rejects empty model identifier";
  let processor_without_chat =
    Model_program.Processor.create ~tokenizer:tok_art ()
  in
  expect
    (Result.is_error
       (Model_program.create ~identity ~processor:processor_without_chat
          ~prefill:prefill_entry ~decode:decode_entry ~generation ~state
          ~specialization))
    "rejects a model program without an explicit chat contract";
  expect
    (Result.is_error (Model_program.Entrypoint.Head.create ()))
    "rejects entrypoint head with no outputs";
  expect
    (Result.is_error
       (Model_program.State.Cache_layout.create ~attention_layers:1 ~kv_heads:0
          ~head_dim:64 ~recurrent_layers:0 ~recurrent_dim:0
          ~recurrent_window:0))
    "rejects cache layout with zero kv heads when attention layers > 0";
  expect
    (Result.is_error
       (Model_program.State.create ~layout ~attentions:[ att0; att0 ]
          ~recurrents:[ rec0 ]))
    "rejects duplicate attention cache layer index";
  expect
    (Result.is_error
       (Model_program.Specialization.create ~min_prefill_tokens:0 ()))
    "rejects min prefill tokens < 1";
  expect
    (Result.is_error
       (Model_program.Mtp.create ~assistant_package:assistant_pkg
          ~target_hidden_output:"target_hidden_state"
          ~coupled_target_state_input:"coupled"
          ~position_ids_input:"position_ids"
          ~full_attention_mask_input:"full_attention_mask"
          ~sliding_attention_mask_input:"sliding_attention_mask"
          ~full_attention_key_input:"full_key"
          ~full_attention_value_input:"full_value"
          ~sliding_attention_key_input:"sliding_key"
          ~sliding_attention_value_input:"sliding_value" ~logits_output:"logits"
          ~projected_target_hidden_output:"projected" ~max_draft_tokens:0))
    "rejects a non-positive target-coupled MTP proposal bound";
  expect
    (Result.is_error
       (Model_program.Mtp.create ~assistant_package:assistant_pkg
          ~target_hidden_output:"target_hidden_state"
          ~coupled_target_state_input:"coupled"
          ~position_ids_input:"position_ids"
          ~full_attention_mask_input:"full_attention_mask"
          ~sliding_attention_mask_input:"sliding_attention_mask"
          ~full_attention_key_input:"full_key"
          ~full_attention_value_input:"full_value"
          ~sliding_attention_key_input:"sliding_key"
          ~sliding_attention_value_input:"sliding_value" ~logits_output:"logits"
          ~projected_target_hidden_output:"projected" ~max_draft_tokens:65_536))
    "rejects an MTP proposal bound that cannot round-trip through ABI-v4";

  (* LFM2.5 probe fixture tests *)
  let lfm_att, lfm_rec = Lfm25_probe.cache_bindings () in
  expect (List.length lfm_att = 6) "LFM2.5 has 6 attention bindings";
  expect (List.length lfm_rec = 10) "LFM2.5 has 10 recurrent bindings";
  expect
    (Model_program.State.Attention_binding.key_input (List.hd lfm_att)
    = "l_kwargs_past_key_values_layers_2_keys")
    "first attention layer input key matches";
  expect
    (Model_program.State.Attention_binding.key_output (List.hd lfm_att)
    = "keys")
    "first attention layer output key matches";
  expect
    (Model_program.State.Attention_binding.key_output (List.nth lfm_att 1)
    = "keys_1")
    "second attention layer output key is indexed keys_1";
  expect
    (Model_program.State.Recurrent_binding.state_input (List.hd lfm_rec)
    = "l_kwargs_past_key_values_layers_0_conv_states")
    "first recurrent layer input state matches";
  expect
    (Model_program.State.Recurrent_binding.state_output (List.hd lfm_rec)
    = "conv_states")
    "first recurrent layer output state matches";
  expect
    (Model_program.State.Recurrent_binding.state_output (List.nth lfm_rec 1)
    = "conv_states_1")
    "second recurrent layer output state is indexed conv_states_1";


  (* Serving engine create_from_program validation tests *)
  let engine_err =
    Serving_engine.create_from_program ~program:restored
      ~model_dir:"/nonexistent/dir" ()
  in
  expect (Result.is_error engine_err)
    "create_from_program fails on nonexistent package files";

  (* Generation create_from_program and create_from_dir validation tests *)
  let gen_prog_err =
    Generation.create_from_program ~program:restored
      ~model_dir:"/nonexistent/dir" ()
  in
  expect (Result.is_error gen_prog_err)
    "generation create_from_program fails on nonexistent dir";
  let gen_dir_err =
    Generation.create_from_dir ~model_dir:"/nonexistent/dir" ()
  in
  expect (Result.is_error gen_dir_err)
    "generation create_from_dir fails on nonexistent model.llmopt";
