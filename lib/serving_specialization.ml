let prefill ~specialization ~captured_tokens ~tokens schedule =
  let min_tokens = Model_program.Specialization.min_prefill_tokens specialization in
  if tokens < min_tokens then
    Error
      (Printf.sprintf "prefill request for %d tokens is below minimum %d"
         tokens min_tokens)
  else
    Serving_schedule.Sequence.specialize_prefill ~minimum_tokens:min_tokens
      ~captured_tokens ~tokens schedule

let recurrent_in_input layer =
  Serving_schedule.Sequence.recurrent_in_input layer

let suffix_prefill_paged_q8 ?(all_token_outputs = false) ~specialization
    ~captured_tokens ~tokens ~past_tokens ~cache schedule =
  let min_tokens = Model_program.Specialization.min_prefill_tokens specialization in
  if tokens < min_tokens then
    Error
      (Printf.sprintf "suffix prefill request for %d tokens is below minimum %d"
         tokens min_tokens)
  else
    Serving_schedule.Sequence.specialize_suffix_prefill_paged_q8
      ~all_token_outputs ~minimum_tokens:min_tokens ~captured_tokens ~tokens
      ~past_tokens ~cache schedule

let decode_paged_q8 ~specialization:_ ~captured_past ~past_tokens ~cache schedule =
  Serving_schedule.Sequence.specialize_decode_paged_q8 ~captured_past ~past_tokens
    ~cache schedule

let solidify_decode_paged_q8 ~specialization ~captured_past ~cache schedule =
  Serving_schedule.Sequence.solidify_decode_paged_q8
    ?sliding_mask_input:
      (Model_program.Specialization.sliding_attention_mask_input specialization)
    ?sliding_window:(Model_program.Specialization.sliding_window specialization)
    ~captured_past ~cache schedule

let decode ~specialization:_ ~captured_past ~past_tokens schedule =
  Serving_schedule.Sequence.specialize_decode ~captured_past ~past_tokens schedule

let prompt_lookup_paged_q8 ~specialization:_ ~captured_query ~query_tokens
    ~captured_past ~past_tokens ~cache schedule =
  Serving_schedule.Sequence.specialize_prompt_lookup_paged_q8
    ~captured_query ~query_tokens ~captured_past ~past_tokens ~cache schedule
