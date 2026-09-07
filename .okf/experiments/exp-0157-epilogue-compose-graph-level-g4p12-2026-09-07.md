---
type: Experiment
title: 'Typed contraction groups G4P12-03: Epilogue.compose reduced to graph-level composition'
description: 'Commit 29451b0 strips the typed compatibility cases from Epilogue.compose so composition is identity cases plus Monoid.append over scalar trees (substitution — no producer duplication, no FP reassociation), with legacy composite tags produced only by tag_of_composition normal-form reconstruction over exactly the four producer×consumer pairings that used to have typed cases (Bias×Residual_add / Residual_add×Bias → Bias_residual_add, Silu_mul×Residual_add → Silu_mul_residual_add, Gelu_mul×Residual_add → Gelu_mul_residual_add) with explicit roles; every other pairing yields a pure Expr composition. New pins in test_epilogue_monoid.ml §4 prove tag/role/MSL-text identity for the four pairings and the pure-Expr two-step Add structure of Residual_add∘Residual_add; all Fold_catalog and epilogue-suite composition pins stay byte-identical.'
tags: [experiment, compiler, gemma4, contraction-group, epilogue, monoid, compose, normal-form, kernel-ir]
status: stable
generated: { by: poolside/1.0, at: '2026-09-07T10:10:00+07:00' }
sources:
  - id: loop
    resource: /LOOP-typed-contraction-groups.md
    title: 'Typed Contraction Groups G4P12 executor checklist (slice 3: compose reduced to the graph)'
  - id: epilogue
    resource: /lib/epilogue.ml
    title: Monoid above compose; tag_of_composition (:122-131); compose = identity + Monoid.append + tag reconstruction (:133-146)
  - id: epilogue-mli
    resource: /lib/epilogue.mli
    title: compose doc — composition step is Monoid.append over scalar trees; only the four legacy composite pairings return composite tags
  - id: kernel-ir
    resource: /lib/kernel_ir.ml
    title: sole lib compose call sites (chain_of Silu/Gelu step :2300 and Add/Mul step :2316)
  - id: test-epilogue-monoid
    resource: /test/test_epilogue_monoid.ml
    title: §4 G4P12-03 composite-pairing pins (:214-277) and x8 digest section (:299+)
  - id: exp-g4p1202
    resource: /.okf/experiments/exp-0156-typed-contraction-groups-g4p12-2026-09-07.md
    title: G4P12-01/02 record that this slice continues
---
# Result

G4P12-03 (commit `29451b0`) makes good on the critique reduction adopted at the top of the G4P12 LOOP: composition is defined over the assignment graph, with the epilogue monoid retained as its scalar sublanguage, and tags are recognized normal forms used for legacy buffer binding only — never computed by single-slot typed compatibility cases.

1. **`Monoid` moves above `compose`** (lib/epilogue.ml, now :87-111) so compose can call the substitution append directly.
2. **`compose` loses its typed cases** (:133-146). It is now identity cases plus `Monoid.append (to_expr producer) (to_expr consumer)`: the `Expr`-branch append substitutes the producer's scalar tree at the consumer's accumulator slot. Because the append operates on trees and never duplicates shared producers, SSA fan-out semantics are preserved; because it does not rewrite FP structure, byte-identity of every folded text is preserved.
3. **Legacy composite tags survive only as normal forms.** `tag_of_composition` (:122-131) recognizes exactly the four producer×consumer pairings that used to have typed cases and returns their legacy tags with explicit roles: `Bias × Residual_add` and `Residual_add × Bias` → `Bias_residual_add {bias; residual}`; `Silu_mul{up} × Residual_add{residual}` → `Silu_mul_residual_add`; `Gelu_mul{up} × Residual_add{residual}` → `Gelu_mul_residual_add`. Every other pairing returns `None` and the composition stays a pure `Expr` (the old fallback path, now the only path).

New pins in `test/test_epilogue_monoid.ml` §4 (fresh `v_up`, existing `v_bias`/`v_res`/`v_scale`):

- compose(`silu_mul`, `residual_add`) → `Silu_mul_residual_add`; compose(`gelu_mul`, `residual_add`) → `Gelu_mul_residual_add`.
- compose(`residual_add`, `bias`) **and** compose(`bias`, `residual_add`) → `Bias_residual_add` in both orders, with roles checked and the MSL text pinned to `= "((acc + (float)bias[i]) + (float)residual[i])"`.
- compose(`residual_add`, `residual_add(v_scale)`) stays a pure `Expr` structurally equal to `Binop (Add, Binop (Add, Slot Acc, Read v_res), Read v_scale)` — the associative stacked case remains a graph-level composition with no composite tag, exactly as before.

`lib/epilogue.mli`'s compose doc now states that the composition step is `Monoid.append` over scalar trees and only the four legacy composite pairings return composite tags. No other lib file changed: the sole lib compose callers are `kernel_ir.ml` :2300/:2316, whose chain-building semantics are preserved (all Fold_catalog and §-suite pins unchanged), and no contraction_group/metal/serving_schedule/metal_runtime/constructive_lowering caller composes.

## Status of the slice

- **Done.** Compose reduced to the graph; typed cases deleted; tags = normal-form reconstruction with explicit roles; every pre-existing composition pin byte-identical; new §4 pins green.
- Gate tails at `29451b0`: `ninja -f ninja.build ocaml-test` prints the new pins `[PASS] compose reduces to substitution: non-template composites stay pure Expr` and `[PASS] compose reconstructs legacy composite tags with explicit roles` plus `All Epilogue Monoid & Coordinate Category tests passed!`, `All typed contraction-group algebra tests passed!`, `llmopt fused Linear_epilogue fold catalog passed`; `ninja -f ninja.build test` ends `Ran 79 tests in 2.248s` / `OK`; `git diff --check` clean.
- **Next**: G4P12-04 emission widening (Q4_0-family emitters gain an `epilogue : Epilogue.t` tail parameter; retire the `q4_0_x8_digest_*` text splice + `quant_linear_add_source`; widen `lower` digest admission to all Q4_0 tiles; golden-capture digest kernel text before deleting the splice), then G4P12-05 remaining quant families, G4P12-06 multi-slot groups through the stack, G4P12-07 fan-out/export scheduling, G4P12-08 reduction phases + numerics + round close-out.
- Device/bundle benchmarks remain user-run (standing constraint).

## Supersedes

exp-0156 recorded G4P12-01/02 and listed G4P12-03 as the next slice; this experiment records its landing. No earlier experiment is superseded.

# Static Validation

`ninja -f ninja.build ocaml-test` and `ninja -f ninja.build test` both passed at `29451b0` (79 Python tests OK), `git diff --check` clean, working tree committed with no residual changes before this docs slice.
