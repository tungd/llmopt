let ( let* ) = Result.bind

type t = {
  identity : Model_program.Identity.t;
  generation : Model_program.Generation.t;
  chat : Model_program.Processor.Chat.t;
  minimum_prefill_tokens : int;
}

let create ~model ?architecture ?family ~vocab_size ~max_positions
    ~chat_format ~bos_token_id ~message_start_token_id ~message_end_token_id
    ?(minimum_prefill_tokens = 1) () =
  if minimum_prefill_tokens <= 0 then
    Error "model profile minimum_prefill_tokens must be positive"
  else
    let* identity =
      Model_program.Identity.create ~model ?architecture ?family ()
    in
    let* generation =
      Model_program.Generation.create ~vocab_size ~max_positions
        ~bos_token_id ~eos_token_id:message_end_token_id ()
    in
    let* chat =
      Model_program.Processor.Chat.create ~format:chat_format ~bos_token_id
        ~message_start_token_id ~message_end_token_id
    in
    Ok { identity; generation; chat; minimum_prefill_tokens }

let identity profile = profile.identity
let generation profile = profile.generation
let chat profile = profile.chat
let minimum_prefill_tokens profile = profile.minimum_prefill_tokens
