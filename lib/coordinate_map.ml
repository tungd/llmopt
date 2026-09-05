type t =
  | Identity
  | Permute of int list
  | Slice of { offsets : int list; strides : int list }
  | Reshape of { in_shape : int list; out_shape : int list }
  | Broadcast of { in_shape : int list; out_shape : int list }
  | Compose of t * t

let identity = Identity

let invalid msg = invalid_arg ("Coordinate_map." ^ msg)

let is_identity_permutation dims =
  List.mapi (fun i d -> i = d) dims |> List.for_all (fun b -> b)

let permute dims =
  let n = List.length dims in
  let seen = Array.make n false in
  List.iter
    (fun d ->
      if d < 0 || d >= n then
        invalid
          (Printf.sprintf "permute: index %d is out of range for rank %d" d n);
      if seen.(d) then
        invalid
          (Printf.sprintf "permute: duplicate index %d in a permutation" d);
      seen.(d) <- true)
    dims;
  if is_identity_permutation dims then Identity else Permute dims

let slice ~offsets ~strides =
  let n = List.length offsets in
  if n = 0 then invalid "slice: must describe at least one axis";
  if List.length strides <> n then
    invalid "slice: offsets and strides must have equal length";
  List.iter2
    (fun offset stride ->
      if offset < 0 then
        invalid (Printf.sprintf "slice: negative offset %d" offset);
      if stride <= 0 then
        invalid (Printf.sprintf "slice: non-positive stride %d" stride))
    offsets strides;
  match offsets, strides with
  | [ 0 ], [ 1 ] -> Identity
  | _ -> Slice { offsets; strides }

let shape_positive name shape =
  List.iter
    (fun d ->
      if d <= 0 then
        invalid
          (Printf.sprintf "shape: %s dimension %d is not positive" name d))
    shape

let reshape ~in_shape ~out_shape =
  if in_shape = [] || out_shape = [] then
    invalid "reshape: shapes must be non-empty";
  shape_positive "input" in_shape;
  shape_positive "output" out_shape;
  let product shape = List.fold_left ( * ) 1 shape in
  let in_elems = product in_shape and out_elems = product out_shape in
  if in_elems <> out_elems then
    invalid
      (Printf.sprintf "reshape: element counts differ (%d input vs %d output)"
         in_elems out_elems);
  if in_shape = out_shape then Identity else Reshape { in_shape; out_shape }

let broadcast ~in_shape ~out_shape =
  if in_shape = [] || out_shape = [] then
    invalid "broadcast: shapes must be non-empty";
  if List.length in_shape <> List.length out_shape then
    invalid
      (Printf.sprintf "broadcast: rank mismatch (%d vs %d)"
         (List.length in_shape) (List.length out_shape));
  shape_positive "input" in_shape;
  shape_positive "output" out_shape;
  List.iter2
    (fun in_dim out_dim ->
      if out_dim < in_dim then
        invalid
          (Printf.sprintf "broadcast: output axis %d shrinks input axis %d"
             out_dim in_dim);
      if in_dim <> 1 && in_dim <> out_dim then
        invalid
          (Printf.sprintf
             "broadcast: axis cannot expand from %d to %d (only 1 expands)"
             in_dim out_dim))
    in_shape out_shape;
  if in_shape = out_shape then Identity else Broadcast { in_shape; out_shape }

(* Composition of two validated permutations, normalized to [Identity] when the
   permutation product is the identity.  Both permutations are validated at
   construction, so every [List.nth] below is in range and total. *)
let compose_permutations p1 p2 =
  let composed = List.map (fun i -> List.nth p1 i) p2 in
  if is_identity_permutation composed then Identity else Permute composed

let rec compose a b =
  match a, b with
  | Identity, other | other, Identity -> other
  | Compose (a1, a2), b -> Compose (a1, compose a2 b)
  | a, b -> Compose (a, b)

let rec simplify = function
  | Identity -> Identity
  | Permute p -> Permute p
  | Slice { offsets = [ 0 ]; strides = [ 1 ] } -> Identity
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
          if List.length p1 = List.length p2 then compose_permutations p1 p2
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
  | Reshape r1, Reshape r2 ->
      r1.in_shape = r2.in_shape && r1.out_shape = r2.out_shape
  | Broadcast b1, Broadcast b2 ->
      b1.in_shape = b2.in_shape && b1.out_shape = b2.out_shape
  | Compose (a1, a2), Compose (b1, b2) -> equal a1 b1 && equal a2 b2
  | _ -> false

let rec to_string = function
  | Identity -> "id"
  | Permute dims ->
      Printf.sprintf "permute[%s]"
        (String.concat "," (List.map string_of_int dims))
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
          Printf.sprintf "((%s %% %du) * %du + (%s / %du))" index_var cols rows
            index_var cols
      | _ ->
          invalid
            (Printf.sprintf
               "emit_msl_index: permutation of rank %d is not resolvable to a \
                linear index expression (only the 2-D transpose [1; 0] is \
                supported)"
               (List.length dims)))
  | Slice { offsets = [ offset ]; strides = [ stride ] } ->
      if stride = 1 then
        if offset = 0 then index_var
        else Printf.sprintf "(%s + %du)" index_var offset
      else Printf.sprintf "((%s * %du) + %du)" index_var stride offset
  | m ->
      invalid
        (Printf.sprintf
           "emit_msl_index: coordinate morphism %s is not resolvable to a \
            linear index expression (only identity, the 2-D transpose, and \
            single-axis slices are supported)"
           (to_string m))
