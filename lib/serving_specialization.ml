let prefill ~specialization ~captured_tokens ~tokens schedule =
  let min_tokens = Model_program.Specialization.min_prefill_tokens specialization in
  if tokens < min_tokens then
    Error
      (Printf.sprintf "prefill request for %d tokens is below minimum %d"
         tokens min_tokens)
  else
    Serving_schedule.Lfm25.specialize_prefill ~captured_tokens ~tokens schedule

let suffix_prefill_paged_q8 ~specialization ~captured_tokens ~tokens ~past_tokens
    ~cache schedule =
  let min_tokens = Model_program.Specialization.min_prefill_tokens specialization in
  if tokens < min_tokens then
    Error
      (Printf.sprintf "suffix prefill request for %d tokens is below minimum %d"
         tokens min_tokens)
  else
    Serving_schedule.Lfm25.specialize_suffix_prefill_paged_q8 ~captured_tokens
      ~tokens ~past_tokens ~cache schedule

let decode_paged_q8 ~specialization:_ ~captured_past ~past_tokens ~cache schedule =
  Serving_schedule.Lfm25.specialize_decode_paged_q8 ~captured_past ~past_tokens
    ~cache schedule

let decode ~specialization:_ ~captured_past ~past_tokens schedule =
  Serving_schedule.Lfm25.specialize_decode ~captured_past ~past_tokens schedule
