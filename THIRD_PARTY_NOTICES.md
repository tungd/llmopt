# Third-party notices

## SGLang radix-cache design

`lib/radix_cache.ml` adapts the compressed-edge insertion, prefix matching,
node splitting, reference locking, leaf eviction, and hybrid-state checkpoint
semantics of SGLang's `RadixCache` and `MambaRadixCache` at revision
`d1af3c89233c475fc1bf11939d86787e6cddd58c`.

SGLang is Copyright 2023-2026 the SGLang team and is distributed under the
[Apache License 2.0](https://github.com/sgl-project/sglang/blob/d1af3c89233c475fc1bf11939d86787e6cddd58c/LICENSE).

The llmopt implementation is an OCaml adaptation over abstract KV slots and
hybrid recurrent checkpoints; it does not include SGLang's PyTorch memory-pool
or scheduler code.
