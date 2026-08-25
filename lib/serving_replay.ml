module Decode_buffers = struct
  type 'buffer named = string * 'buffer

  type ('attention, 'buffer) attention = {
    binding : 'attention;
    key : 'buffer named;
    value : 'buffer named;
  }

  type ('recurrent, 'buffer) recurrent = {
    binding : 'recurrent;
    state : 'buffer named;
  }

  type ('attention, 'recurrent, 'buffer) t = {
    attention : ('attention, 'buffer) attention list;
    recurrent : ('recurrent, 'buffer) recurrent list;
  }

  let create ~attention ~recurrent =
    {
      attention =
        List.map
          (fun (binding, key, value) -> { binding; key; value })
          attention;
      recurrent =
        List.map (fun (binding, state) -> { binding; state }) recurrent;
    }

  let inputs buffers =
    List.concat_map
      (fun attention -> [ attention.key; attention.value ])
      buffers.attention
    @ List.map (fun recurrent -> recurrent.state) buffers.recurrent

  let update_attention buffers ~f =
    let ( let* ) = Result.bind in
    let rec update output = function
      | [] -> Ok (List.rev output)
      | (attention : ('attention, 'buffer) attention) :: rest ->
          let* key, value = f attention.binding in
          let key_name, _ = attention.key in
          let value_name, _ = attention.value in
          update
            ({ attention with key = (key_name, key); value = (value_name, value) }
            :: output)
            rest
    in
    let* attention = update [] buffers.attention in
    Ok { buffers with attention }

  let recurrent buffers =
    List.map
      (fun recurrent ->
        let _, state = recurrent.state in
        recurrent.binding, state)
      buffers.recurrent
end
