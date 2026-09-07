let fail message = raise (Failure message)
let expect condition message = if not condition then fail message
let expect_ok = function Ok value -> value | Error message -> fail message

let expect_int_array actual expected message =
  expect
    (Array.length actual = Array.length expected
    && Array.for_all2 ( = ) actual expected)
    message

let contains_substring source needle =
  let source_length = String.length source in
  let needle_length = String.length needle in
  let rec search offset =
    if offset + needle_length > source_length then false
    else if String.sub source offset needle_length = needle then true
    else search (offset + 1)
  in
  needle_length = 0 || search 0

let tensor_input graph ~name ~source ~shape ~dtype =
  Ir.Graph.tensor_input graph ~name ~source
    ~shape:(Tensor_shape.of_ints_exn shape) ~dtype
