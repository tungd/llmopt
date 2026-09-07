open Test_support

module Generation_test_engine = struct
  type t = {
    mutable outputs : int list;
    mutable prompt_calls : int array list;
    mutable decode_calls : (int array * int) list;
    mutable release_calls : int;
    cached_prompt_tokens : int;
  }

  type step = { engine : t; tokens : int array }

  let create ~outputs ~cached_prompt_tokens =
    {
      outputs;
      prompt_calls = [];
      decode_calls = [];
      release_calls = 0;
      cached_prompt_tokens;
    }

  let prompt engine ~tokens =
    engine.prompt_calls <- Array.copy tokens :: engine.prompt_calls;
    Ok ({ engine; tokens = Array.copy tokens }, engine.cached_prompt_tokens)

  let decode engine ~step ~token =
    let prefix = step.tokens in
    engine.decode_calls <- (Array.copy prefix, token) :: engine.decode_calls;
    Ok { engine; tokens = Array.append prefix [| token |] }

  let release step =
    step.engine.release_calls <- step.engine.release_calls + 1;
    Ok ()
  let tokens step = Array.copy step.tokens

  let next_token ?params:_ step =
    match step.engine.outputs with
    | token :: rest ->
        step.engine.outputs <- rest;
        Ok token
    | [] -> Error "synthetic generation engine exhausted"
end

module Generation_test = Generation_core.Make (Generation_test_engine)

let tokenizer_fixture () =
  let writer = Binary.Writer.create () in
  let token id flags text =
    Binary.Writer.u32 writer id;
    Binary.Writer.u8 writer flags;
    Binary.Writer.raw_bytes writer (Bytes.make 3 '\000');
    Binary.Writer.u32 writer (String.length text);
    Binary.Writer.raw_string writer text
  in
  let merge left right =
    Binary.Writer.u32 writer (String.length left);
    Binary.Writer.u32 writer (String.length right);
    Binary.Writer.raw_string writer left;
    Binary.Writer.raw_string writer right
  in
  let space = "\196\160" in
  Binary.Writer.raw_string writer Tokenizer.binary_magic;
  Binary.Writer.u16 writer 1;
  Binary.Writer.u16 writer 1;
  Binary.Writer.u32 writer 7;
  Binary.Writer.u32 writer 2;
  Binary.Writer.u32 writer 6;
  token 0 3 "<|startoftext|>";
  token 1 0 "a";
  token 2 0 "b";
  token 3 0 "ab";
  token 4 0 space;
  token 5 0 (space ^ "a");
  token 6 1 "python";
  merge "a" "b";
  merge space "a";
  Binary.Writer.contents writer

let run () =
    let generation_config =
      Generation_core.Config.create ~max_new_tokens:5 |> expect_ok
    in
    let generation_engine =
      Generation_test_engine.create ~outputs:[ 11; 12 ] ~cached_prompt_tokens:2
    in
    let emitted = ref [] in
    let generated =
      Generation_test.run
        ~emit:(fun token -> emitted := token :: !emitted)
        generation_engine ~config:generation_config
        ~is_stop:(fun token -> token = 12) ~prompt:[| 1; 2; 3 |]
      |> expect_ok
    in
    expect_int_array (Generation_core.Result.completion_tokens generated)
      [| 11; 12 |] "generation emits through the stop token";
    expect_int_array (Array.of_list (List.rev !emitted)) [| 11; 12 |]
      "generation callback preserves token order";
    expect
      (Generation_core.Result.finish_reason generated
      = Generation_core.Finish_reason.End_token)
      "generation records the end-token finish reason";
    expect (generation_engine.release_calls = 1)
      "generation releases its active engine step exactly once";
    let step_engine =
      Generation_test_engine.create ~outputs:[ 101; 102; 103 ]
        ~cached_prompt_tokens:1
    in
    let state, first =
      Generation_test.State.init step_engine
        ~config:(Generation_core.Config.create ~max_new_tokens:3 |> expect_ok)
        ~is_stop:(fun token -> token = 103) ~prompt:[| 10; 20 |] ()
      |> expect_ok
    in
    expect (first = 101) "stateful generation returns its first token";
    ignore (Generation_test.State.step step_engine state |> expect_ok);
    ignore (Generation_test.State.step step_engine state |> expect_ok);
    expect
      (Generation_test.State.completion_tokens state = [ 101; 102; 103 ])
      "stateful generation preserves the interleaved token sequence";
    expect (step_engine.release_calls = 1)
      "stateful generation releases its terminal engine step";
    let exhausted_engine =
      Generation_test_engine.create ~outputs:[] ~cached_prompt_tokens:0
    in
    expect
      (Result.is_error
         (Generation_test.State.init exhausted_engine
            ~config:(Generation_core.Config.create ~max_new_tokens:2 |> expect_ok)
            ~is_stop:(Fun.const false) ~prompt:[| 1 |] ()))
      "generation reports an initial token failure";
    expect (exhausted_engine.release_calls = 1)
      "generation releases its engine step when initial sampling fails";
    let close_engine =
      Generation_test_engine.create ~outputs:[ 201; 202 ] ~cached_prompt_tokens:0
    in
    let close_state, _ =
      Generation_test.State.init close_engine
        ~config:(Generation_core.Config.create ~max_new_tokens:2 |> expect_ok)
        ~is_stop:(Fun.const false) ~prompt:[| 1 |] ()
      |> expect_ok
    in
    Generation_test.State.close close_state |> expect_ok;
    expect
      (close_engine.release_calls = 1
      && Generation_test.State.is_finished close_state)
      "closing an active generation releases its engine step exactly once";

    let tokenizer_bytes = tokenizer_fixture () in
    let tokenizer = Tokenizer.of_bytes tokenizer_bytes |> expect_ok in
    expect_int_array (Tokenizer.encode tokenizer "ab" |> expect_ok) [| 3 |]
      "binary tokenizer performs ranked BPE without inventing BOS";
    expect_int_array
      (Tokenizer.encode ~bos_token_id:0 tokenizer "python ab" |> expect_ok)
      [| 0; 6; 4; 3 |] "explicit BOS composes with added-token BPE";
    expect_int_array
      (Tokenizer.encode tokenizer "python ab" |> expect_ok)
      [| 6; 4; 3 |] "added-token matching composes with byte-level BPE";
    expect
      (Tokenizer.decode tokenizer [| 0; 5; 2 |] |> expect_ok = " ab")
      "binary tokenizer decodes and skips special tokens";
    (match
       Tokenizer.of_bytes
         (Bytes.sub tokenizer_bytes 0 (Bytes.length tokenizer_bytes - 1))
     with
    | Error _ -> ()
    | Ok _ -> fail "tokenizer accepted a truncated archive");

    let queue =
      Serving_queue.create ~alpha_age:0.01 ~prefill_rate:100.0
        ~decode_rate:10.0 ()
    in
    let short_id = Serving_queue.Request_id.create () in
    let long_id = Serving_queue.Request_id.create () in
    let decode_id = Serving_queue.Request_id.create () in
    let short_request : Serving_queue.request =
      {
        id = short_id;
        arrival_time = 0.0;
        state =
          Serving_queue.Pending_prefill
            {
              prompt_tokens = [| 1; 2; 3; 4 |];
              cached_tokens = 0;
              remaining_prefill = 4;
              max_new_tokens = 2;
              ignore_eos = false;
              sampling_params = Sampling.Params.greedy;
            };
        priority_score = 0.0;
      }
    in
    let long_request : Serving_queue.request =
      {
        id = long_id;
        arrival_time = 0.0;
        state =
          Serving_queue.Pending_prefill
            {
              prompt_tokens = Array.make 2048 1;
              cached_tokens = 0;
              remaining_prefill = 2048;
              max_new_tokens = 100;
              ignore_eos = false;
              sampling_params = Sampling.Params.greedy;
            };
        priority_score = 0.0;
      }
    in
    let decode_request : Serving_queue.request =
      {
        id = decode_id;
        arrival_time = 0.0;
        state =
          Serving_queue.Active_decode
            {
              prompt_length = 100;
              generated_tokens = [ 1; 2 ];
              max_new_tokens = 3;
              ignore_eos = false;
              sampling_params = Sampling.Params.greedy;
            };
        priority_score = 0.0;
      }
    in
    List.iter
      (fun (request : Serving_queue.request) ->
        request.priority_score <-
          Serving_queue.Score.compute ~prefill_rate:100.0 ~decode_rate:10.0
            ~current_time:0.0 ~arrival_time:request.arrival_time request.state)
      [ short_request; long_request; decode_request ];
    expect (short_request.priority_score > long_request.priority_score)
      "SRPT prioritizes short prefill over long prefill";
    expect (decode_request.priority_score > short_request.priority_score)
      "SRPT prioritizes near-completion decode";
    List.iter (Serving_queue.enqueue queue)
      [ long_request; short_request; decode_request ];
    let popped_ids =
      List.init 3 (fun _ ->
          Serving_queue.pop_next queue
          |> Option.map (fun (request : Serving_queue.request) -> request.id))
    in
    expect
      (match popped_ids with
      | [ Some first; Some second; Some third ] ->
          Serving_queue.Request_id.equal first decode_id
          && Serving_queue.Request_id.equal second short_id
          && Serving_queue.Request_id.equal third long_id
      | _ -> false)
      "serving queue pops decode, short prefill, then long prefill";

    (* Greedy speculative acceptance tests *)
    let verify draft target =
      Serving_engine.Speculative_acceptance.greedy ~draft_tokens:draft
        ~target_predictions:target
      |> Result.get_ok
    in
    let mismatch = verify [| 11; 12; 13 |] [| 99; 12; 13; 14 |] in
    expect
      (Serving_engine.Speculative_acceptance.accepted_draft_tokens mismatch = 0
      && Serving_engine.Speculative_acceptance.emitted_tokens mismatch = [| 99 |])
      "speculative acceptance rejects a first-token mismatch";
    let partial = verify [| 11; 12; 13 |] [| 11; 99; 13; 14 |] in
    expect
      (Serving_engine.Speculative_acceptance.accepted_draft_tokens partial = 1
      && Serving_engine.Speculative_acceptance.emitted_tokens partial = [| 11; 99 |])
      "speculative acceptance emits the matched prefix and target correction";
    let complete = verify [| 11; 12; 13 |] [| 11; 12; 13; 14 |] in
    expect
      (Serving_engine.Speculative_acceptance.accepted_draft_tokens complete = 3
      && Serving_engine.Speculative_acceptance.emitted_tokens complete
         = [| 11; 12; 13; 14 |])
      "speculative acceptance emits the bonus target token after a full match";
    expect
      (Result.is_error
         (Serving_engine.Speculative_acceptance.greedy
            ~draft_tokens:[| 11; 12 |] ~target_predictions:[| 11; 12 |]))
      "speculative acceptance requires one target prediction beyond the draft";
    let mtp_commit_counts = ref [] in
    let mtp_result, mtp_state =
      Serving_engine.Target_coupled_mtp.run ~max_draft_tokens:4
        ~state:"target-before"
        ~propose:(fun state ->
          if state = "target-before" then Ok [| 11; 12; 13 |]
          else Error "assistant did not receive the target state")
        ~verify:(fun state draft_tokens ->
          if state = "target-before" && draft_tokens = [| 11; 12; 13 |]
          then Ok ([| 11; 99; 13; 14 |], `Verified)
          else Error "target verification did not receive the full proposal")
        ~commit:(fun verified ~accepted_draft_tokens ->
          mtp_commit_counts := accepted_draft_tokens :: !mtp_commit_counts;
          match verified with
          | `Verified -> Ok "target-after"
          | _ -> Error "unexpected verified target state")
      |> Result.get_ok
    in
    expect
      (Serving_engine.Target_coupled_mtp.proposed_tokens mtp_result
         = [| 11; 12; 13 |]
      && Serving_engine.Target_coupled_mtp.emitted_tokens mtp_result
         = [| 11; 99 |]
      && Serving_engine.Target_coupled_mtp.accepted_draft_tokens mtp_result = 1
      && mtp_state = "target-after" && !mtp_commit_counts = [ 1 ])
      "target-coupled MTP verifies one full proposal then commits only its accepted prefix";
    expect
      (Result.is_error
         (Serving_engine.Target_coupled_mtp.run ~max_draft_tokens:4 ~state:()
            ~propose:(fun () -> Ok [| 1; 2; 3; 4; 5 |])
            ~verify:(fun () _ -> failwith "verification must not run")
            ~commit:(fun _ ~accepted_draft_tokens:_ ->
              failwith "commit must not run")))
      "target-coupled MTP rejects an assistant proposal beyond the declared K";
    let medusa_tree =
      Serving_engine.Medusa.create_linear_tree ~head_predictions:[| 101; 102; 103 |]
    in
    let masks = Serving_engine.Medusa.tree_masks medusa_tree in
    expect (masks.(0) = 1 && masks.(1) = 3 && masks.(2) = 7)
      "Medusa linear tree generates causal prefix masks";
    let medusa_commit_depths = ref [] in
    let medusa_result, medusa_state =
      Serving_engine.Medusa.run ~tree:medusa_tree ~state:"medusa-before"
        ~verify:(fun state tree ->
          if state = "medusa-before"
             && Serving_engine.Medusa.tree_tokens_array tree = [| 101; 102; 103 |]
          then Ok ([| 101; 102; 999; 104 |], `Verified)
          else Error "Medusa verification did not receive the candidate tree")
        ~commit:(fun verified ~accepted_depth ->
          medusa_commit_depths := accepted_depth :: !medusa_commit_depths;
          match verified with
          | `Verified -> Ok "medusa-after"
          | _ -> Error "unexpected verified state")
      |> Result.get_ok
    in
    expect
      (Serving_engine.Medusa.accepted_depth medusa_result = 2
      && Serving_engine.Medusa.emitted_tokens medusa_result = [| 101; 102; 999 |]
      && medusa_state = "medusa-after"
      && !medusa_commit_depths = [ 2 ])
      "Medusa tree accepts matching candidate prefix and bonus target token";

    (* Prompt-lookup weightless speculative decoding tests *)
    let context_tokens = [| 10; 20; 30; 40; 50; 60; 70; 10; 20; 30 |] in
    let prompt_candidates =
      Serving_engine.Prompt_lookup.find_candidate_ngram ~min_ngram:2 ~max_ngram:3
        ~max_draft_tokens:3 ~tokens:context_tokens ()
    in
    expect (prompt_candidates = [| 40; 50; 60 |])
      "Prompt lookup matches exact 3-gram suffix [10; 20; 30] and proposes [40; 50; 60]";

    let prompt_tree_opt =
      Serving_engine.Prompt_lookup.propose_tree ~min_ngram:2 ~max_ngram:3
        ~max_draft_tokens:3 ~tokens:context_tokens ()
    in
    expect (Option.is_some prompt_tree_opt)
      "Prompt lookup builds a valid Medusa candidate tree from matched tokens";

    let prompt_tree = Option.get prompt_tree_opt in
    expect (Serving_engine.Medusa.tree_tokens_array prompt_tree = [| 40; 50; 60 |])
      "Prompt lookup tree tokens match candidate array";

    let no_match_tokens = [| 1; 2; 3; 4; 5; 6; 7; 8; 9 |] in
    let no_candidates =
      Serving_engine.Prompt_lookup.find_candidate_ngram ~min_ngram:2 ~max_ngram:3
        ~max_draft_tokens:3 ~tokens:no_match_tokens ()
    in
    expect (no_candidates = [||])
      "Prompt lookup returns empty array when no n-gram match exists in history";

    let repeated_candidates =
      Serving_engine.Prompt_lookup.find_candidate_ngram ~min_ngram:2 ~max_ngram:4
        ~max_draft_tokens:4 ~tokens:[| 7; 7; 7; 7; 7; 7; 7; 7 |] ()
    in
    expect (repeated_candidates = [| 7; 7; 7; 7 |])
      "Prompt lookup scans beyond an overlapping nearest match for the longest continuation";

    let no_tree_opt =
      Serving_engine.Prompt_lookup.propose_tree ~min_ngram:2 ~max_ngram:3
        ~max_draft_tokens:3 ~tokens:no_match_tokens ()
    in
    expect (Option.is_none no_tree_opt)
      "Prompt lookup returns None when no candidate tree can be formed";

    let end_to_end_result, end_to_end_state =
      Serving_engine.Medusa.run ~tree:prompt_tree ~state:"lookup-state"
        ~verify:(fun state _tree ->
          if state = "lookup-state" then Ok ([| 40; 50; 999; 0 |], `Verified)
          else Error "invalid state")
        ~commit:(fun `Verified ~accepted_depth ->
          if accepted_depth = 2 then Ok "lookup-committed"
          else Error "unexpected depth")
      |> Result.get_ok
    in
    expect
      (Serving_engine.Medusa.accepted_depth end_to_end_result = 2
      && Serving_engine.Medusa.emitted_tokens end_to_end_result = [| 40; 50; 999 |]
      && end_to_end_state = "lookup-committed")
      "Prompt-lookup tree executes and commits through Medusa tree engine";
    let capacity_queue =
      Serving_queue.create ~token_capacity:1000 ~high_watermark_ratio:0.90
        ~low_watermark_ratio:0.75 ()
    in
    Serving_queue.reserve_tokens capacity_queue 950 |> expect_ok;
    expect (Serving_queue.is_congested capacity_queue)
      "capacity accounting enters congestion at the configured high watermark";
    Serving_queue.release_tokens capacity_queue 150;
    expect (Serving_queue.is_congested capacity_queue)
      "capacity accounting retains congestion above the low watermark";
    Serving_queue.release_tokens capacity_queue 100;
    expect (not (Serving_queue.is_congested capacity_queue))
      "capacity accounting clears congestion below the low watermark";

    expect_ok (Lfm25.Config.validate Lfm25.Config.probe_350m);
    expect (Lfm25.Config.probe_350m.dtype = Ir.Dtype.Float16)
      "canonical model uses float16 activations"
