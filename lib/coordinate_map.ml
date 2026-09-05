type t =
  | Identity
  | Permute of int list
  | Slice of { offsets : int list; strides : int list }
  | Reshape of { in_shape : int list; out_shape : int list }
  | Broadcast of { in_shape : int list; out_shape : int list }
  | Compose of t * t

let identity = Identity

let rec compose a b =
  match a, b with
  | Identity, other | other, Identity -> other
  | Compose (a1, a2), b -> Compose (a1, compose a2 b)
  | a, b -> Compose (a, b)

let rec simplify = function
  | Identity -> Identity
  | Permute p -> Permute p
  | Slice s -> Slice s
  | Reshape r ->
      if r.in_shape = r.out_shape then Identity else Reshape r
  | Broadcast b ->
      if b.in_shape = b.out_shape then Identity else Broadcast b
  | Compose (a, b) -> (
      let sa = simplify a in
      let sb = simplify b in
      match sa, sb with
      | Identity, other | other, Identity -> other
      | Permute p1, Permute p2 ->
          if List.length p1 = List.length p2 then
            try
              let composed = List.map (fun i -> List.nth p1 i) p2 in
              let is_id = List.mapi (fun i x -> i = x) composed |> List.for_all (fun x -> x) in
              if is_id then Identity else Permute composed
            with _ -> Compose (sa, sb)
          else Compose (sa, sb)
      | _ -> Compose (sa, sb))

let is_identity m =
  match simplify m with
  | Identity -> true
  | _ -> false

let rec equal a b =
  match simplify a, simplify b with
  | Identity, Identity -> true
  | Permute p1, Permute p2 -> p1 = p2
  | Slice s1, Slice s2 -> s1.offsets = s2.offsets && s1.strides = s2.strides
  | Reshape r1, Reshape r2 -> r1.in_shape = r2.in_shape && r1.out_shape = r2.out_shape
  | Broadcast b1, Broadcast b2 -> b1.in_shape = b2.in_shape && b1.out_shape = b2.out_shape
  | Compose (a1, a2), Compose (b1, b2) -> equal a1 b1 && equal a2 b2
  | _ -> false

let rec to_string = function
  | Identity -> "id"
  | Permute dims ->
      Printf.sprintf "permute[%s]" (String.concat "," (List.map string_of_int dims))
  | Slice { offsets; strides } ->
      Printf.sprintf "slice[offset=%s,stride=%s]"
        (String.concat "," (List.map string_of_int offsets))
        (String.concat "," (List.map string_of_int strides))
  | Reshape { in_shape; out_shape } ->
      Printf.sprintf "reshape[%s -> %s]"
        (String.concat "x" (List.map string_of_int in_shape))
        (String.concat "x" (List.map string_of_int out_shape))
  | Broadcast { in_shape; out_shape } ->
      Printf.sprintf "bcast[%s -> %s]"
        (String.concat "x" (List.map string_of_int in_shape))
        (String.concat "x" (List.map string_of_int out_shape))
  | Compose (a, b) ->
      Printf.sprintf "(%s ∘ %s)" (to_string a) (to_string b)

let emit_msl_index m ~index_var ~shape =
  match simplify m with
  | Identity -> index_var
  | Permute dims -> (
      match shape with
      | [ rows; cols ] when dims = [ 1; 0 ] ->
          (* 2D matrix transpose: index is (r * cols + c). Transposed is (c * rows + r) *)
          Printf.sprintf "((%s %% %du) * %du + (%s / %du))" index_var cols rows index_var cols
      | _ ->
          Printf.sprintf "permute_index<%s>(%s)"
            (String.concat "," (List.map string_of_int dims))
            index_var)
  | Slice { offsets = [ offset ]; strides = [ stride ] } ->
      if stride = 1 then
        if offset = 0 then index_var else Printf.sprintf "(%s + %du)" index_var offset
      else
        Printf.sprintf "((%s * %du) + %du)" index_var stride offset
  | _ -> Printf.sprintf "(%s)" index_var
