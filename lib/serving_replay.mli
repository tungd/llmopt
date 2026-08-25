module Decode_buffers : sig
  type ('attention, 'recurrent, 'buffer) t

  val create :
    attention:
      ('attention * (string * 'buffer) * (string * 'buffer)) list ->
    recurrent:('recurrent * (string * 'buffer)) list ->
    ('attention, 'recurrent, 'buffer) t

  val inputs :
    ('attention, 'recurrent, 'buffer) t -> (string * 'buffer) list

  val update_attention :
    ('attention, 'recurrent, 'buffer) t ->
    f:('attention -> ('buffer * 'buffer, 'error) result) ->
    (('attention, 'recurrent, 'buffer) t, 'error) result

  val recurrent :
    ('attention, 'recurrent, 'buffer) t -> ('recurrent * 'buffer) list
end
