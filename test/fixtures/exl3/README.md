# EXL3 reference fixtures

These archives contain valid synthetic packed EXL3 mul1 trellises, signed
nonuniform scales, F16 inputs, all 65,536 codebook values, decoded F16 weights,
and F32 linear expectations. They contain no downloaded model weights.

The oracle is `tests/testlib/trellis.py` at ExLlamaV3 revision
`cdc785db89e263e65e5849cb53e1ada156411414`. Each adjacent JSON receipt records the
oracle and archive SHA-256, dimensions, seed, and input-staging numerical error.
The upstream MIT license is retained in `UPSTREAM_LICENSE`.
The generator imports only that CPU reference (NumPy/PyTorch); it does not build
or execute the upstream CUDA extension. Regular tests use the checked-in files
and need neither a clone nor upstream dependencies.

Regenerate from a checkout at that revision:

```sh
python/.venv/bin/python3 bench/make_exl3_fixture.py --upstream /path/to/exllamav3 --output test/fixtures/exl3/mul1-4bit.llmopt
python/.venv/bin/python3 bench/make_exl3_fixture.py --upstream /path/to/exllamav3 --output test/fixtures/exl3/mul1-3bit.llmopt --bits 3 --n 128 --rows 9
```

Run the native schema tests with `ninja -f ninja.build -j1 exl3-test`.
Run the sequential Metal correctness/timing probe with
`python3 bench/run_exl3_probe.py`. Results and full logs default to
`_artifacts/exl3-slice/`; each run should use its own `--output` directory when
preserving multiple measurements.

The Metal probe validates the decoder bit-exactly and the complete linear
operation against a reference that includes the F16 input-transform boundary.
It exercises vector/matrix schedule parameters, partial M tiles, and a SiLU
epilogue after the output transform. Compile time and GPU execution time are
reported separately. These small repeated buffers are not a full-model benchmark.
