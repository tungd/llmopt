"""Generate deterministic EXL3 fixtures using a pinned, independent upstream CPU oracle.

No CUDA extension, encoder, model download, or GPU execution is used. Every packed
bit pattern is a valid tail-biting trellis. This tests the execution representation,
not the quality of conversion from a trained model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from llmopt_backend.tensor_archive import TensorSpec, write_payloads

REVISION = "cdc785db89e263e65e5849cb53e1ada156411414"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bits", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--rows", type=int, default=17)
    args = parser.parse_args()
    if args.k <= 0 or args.n <= 0 or args.k % 128 or args.n % 128 or args.rows <= 0:
        parser.error("positive dimensions divisible by 128 and positive rows are required")
    actual = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise ValueError(f"reference revision is {actual}; expected {REVISION}")
    reference = args.upstream / "tests/testlib/trellis.py"
    clean_reference = subprocess.check_output(["git", "-C", str(args.upstream), "show", f"{REVISION}:tests/testlib/trellis.py"])
    if reference.read_bytes() != clean_reference:
        raise ValueError("reference file differs from pinned revision")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("exl3_upstream_reference", reference)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    rng = np.random.default_rng(20261030)
    trellis = ref.random_packed(args.k // 16 * (args.n // 16), args.bits, rng)
    trellis = trellis.reshape(args.k // 16, args.n // 16, 16 * args.bits)
    # Signed, nonuniform scales detect missing/reversed transforms or scale order.
    suh = ((rng.integers(0, 2, args.k) * 2 - 1) * rng.uniform(.5, 1.5, args.k)).astype(np.float16)
    svh = ((rng.integers(0, 2, args.n) * 2 - 1) * rng.uniform(.03, .1, args.n)).astype(np.float16)
    x = rng.uniform(-.5, .5, (args.rows, args.k)).astype(np.float16)
    decoded = ref.dequant_rotated(trellis, args.bits, "mul1")
    h = ref.sylvester(128).numpy() / 128 ** .5
    rotated = ((x.astype(np.float64) * suh).reshape(-1, 128) @ h).reshape(x.shape).astype(np.float16)
    # Stage-aware oracle: half input transform, float contraction, float output
    # transform. Reference operations use float64 and round only at these boundaries.
    product = (rotated.astype(np.float64) @ decoded.astype(np.float64)).astype(np.float32)
    expected = ((product.astype(np.float64).reshape(-1, 128) @ h).reshape(args.rows, args.n) * svh).astype(np.float32)
    full_precision = ref.linear(torch.from_numpy(x), torch.from_numpy(trellis.copy()),
                                torch.from_numpy(suh), torch.from_numpy(svh), args.bits, "mul1").numpy()
    payloads = {
        "weight.exl3": (7, (32,), struct.pack("<8s6I", b"EXL3MUL1", 1, args.bits, args.k, args.n, 128, 0)),
        "weight.trellis": (7, (args.k // 16, args.n // 16, 32 * args.bits), trellis.astype("<i2").tobytes()),
        "weight.suh": (1, (args.k,), suh.astype("<f2").tobytes()),
        "weight.svh": (1, (args.n,), svh.astype("<f2").tobytes()),
        "input": (1, x.shape, x.astype("<f2").tobytes()),
        "expected": (0, expected.shape, expected.astype("<f4").tobytes()),
        "decoded": (1, decoded.shape, decoded.astype("<f2").tobytes()),
        "codebook": (1, (65536,), ref.decode(np.arange(65536), "mul1").astype("<f2").tobytes()),
    }
    write_payloads([TensorSpec(name, tag, tuple(shape), len(data)) for name, (tag, shape, data) in payloads.items()],
                   args.output, lambda name: (payloads[name][2],))
    receipt = {
        "reference_repository": "https://github.com/turboderp-org/exllamav3",
        "reference_revision": REVISION,
        "reference_file": "tests/testlib/trellis.py",
        "reference_sha256": hashlib.sha256(clean_reference).hexdigest(),
        "archive_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "bits": args.bits, "k": args.k, "n": args.n, "rows": args.rows,
        "seed": 20261030, "codebook": "mul1",
        "fixture_kind": "synthetic valid packed trellis; not quantized model weights",
        "staging_max_abs_error_vs_full_precision": float(np.max(np.abs(expected - full_precision))),
    }
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
