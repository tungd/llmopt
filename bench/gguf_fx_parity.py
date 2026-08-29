"""Run one graph-captured GGUF linear through the native llmopt Metal runtime."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gguf import GGUFReader, dequantize
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parents[1] / "python"))

from llmopt_backend import ExternalTensor, bind_external_tensors, capture_from_fx
from llmopt_backend.fx_graph import write_graph


CASES = (
    {
        "slug": "smollm",
        "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "gguf_repo": "unsloth/SmolLM2-135M-Instruct-GGUF",
        "gguf_file": "SmolLM2-135M-Instruct-Q4_K_M.gguf",
        "tensor": "blk.0.attn_q.weight",
    },
    {
        "slug": "qwen",
        "model": "unsloth/Qwen3.5-0.8B",
        "gguf_repo": "unsloth/Qwen3.5-0.8B-GGUF",
        "gguf_file": "Qwen3.5-0.8B-UD-Q4_K_XL.gguf",
        "tensor": "blk.0.ffn_gate.weight",
    },
    {
        "slug": "gemma",
        "model": "unsloth/gemma-4-E2B-it",
        "gguf_repo": "unsloth/gemma-4-E2B-it-GGUF",
        "gguf_file": "gemma-4-E2B-it-UD-Q4_K_XL.gguf",
        "tensor": "blk.0.attn_q.weight",
    },
)


class LinearProbe(nn.Module):
    def __init__(self, output_features: int, input_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros((output_features, input_features), dtype=torch.float16),
            requires_grad=False,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight)


def sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def metadata_string(reader: GGUFReader, key: str) -> str | None:
    field = reader.get_field(key)
    if field is None:
        return None
    value = field.contents()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def cached_gguf(repository: str, filename: str) -> Path:
    cache = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
            / "hub",
        )
    )
    repository_directory = "models--" + repository.replace("/", "--")
    matches = list((cache / repository_directory / "snapshots").glob(f"*/{filename}"))
    if not matches:
        raise FileNotFoundError(
            f"{filename} is not cached under {cache / repository_directory}"
        )
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def capture_linear(output_features: int, input_features: int):
    model = LinearProbe(output_features, input_features).eval()
    input = torch.zeros((1, input_features), dtype=torch.float16)
    captured: dict[str, Any] = {}

    def backend(graph_module: Any, example_inputs: list[Any]):
        captured["graph"] = capture_from_fx(graph_module, example_inputs)
        return graph_module.forward

    torch._dynamo.reset()
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        compiled(input)
    return captured["graph"]


def bind_weight(captured, *, tensor_name: str, quant_type: str):
    manifest = captured.manifest
    static = [
        node
        for node in manifest["nodes"]
        if node.get("binding", {}).get("kind") == "tensor-store"
    ]
    runtime = [
        node
        for node in manifest["nodes"]
        if node.get("binding", {}).get("kind") == "runtime"
    ]
    if len(static) != 1 or len(runtime) != 1 or len(manifest["outputs"]) != 1:
        raise RuntimeError(
            "linear probe must capture one static weight, one runtime input, and one output"
        )
    source_key = str(static[0]["binding"]["key"])
    shape = tuple(int(value) for value in static[0]["shape"])
    rebound = bind_external_tensors(
        captured,
        {source_key: ExternalTensor(key=tensor_name, dtype=quant_type, shape=shape)},
    )
    return rebound.manifest, str(runtime[0]["name"]), str(manifest["outputs"][0])


def run_case(
    case: dict[str, Any], *, root: Path, artifact_root: Path
) -> dict[str, Any]:
    gguf_path = Path(case["gguf"])
    if not gguf_path.exists():
        raise FileNotFoundError(gguf_path)
    reader = GGUFReader(str(gguf_path), mode="r")
    tensor = next(
        (candidate for candidate in reader.tensors if candidate.name == case["tensor"]),
        None,
    )
    if tensor is None:
        raise KeyError(f"{gguf_path.name} has no tensor {case['tensor']}")
    quant_type = tensor.tensor_type.name
    if quant_type not in {"Q8_0", "Q4_K", "Q5_K", "Q6_K", "Q5_0", "Q4_0"}:
        raise ValueError(f"native linear probe does not support {quant_type}")
    output_features, input_features = (int(value) for value in reversed(tensor.shape))
    captured = capture_linear(output_features, input_features)
    manifest, input_name, output_name = bind_weight(
        captured, tensor_name=str(case["tensor"]), quant_type=quant_type
    )

    slug = str(case["model"]).split("/")[-1].lower()
    directory = artifact_root / slug
    directory.mkdir(parents=True)
    graph_path = directory / "graph.llmopt"
    write_graph(manifest, graph_path)
    (directory / "fx.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.symlink(gguf_path, directory / "weights.gguf")

    compiler = root / "_build/bin/llmopt-fx"
    checker = root / "_build/bin/llmopt-package-check"
    runner = root / "_build/bin/llmopt-package-run"
    subprocess.run(
        [str(compiler), "--weights", "weights.gguf", str(graph_path), str(directory)],
        check=True,
        cwd=root,
    )
    subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(directory / "kernel.metal"),
            "-o",
            str(directory / "kernel.air"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metallib",
            str(directory / "kernel.air"),
            "-o",
            str(directory / "kernel.metallib"),
        ],
        check=True,
    )
    package_check = subprocess.check_output([str(checker), str(directory)], text=True)

    values = np.linspace(-0.5, 0.5, input_features, dtype=np.float32).astype(np.float16)
    input_path = directory / "input.f16"
    output_path = directory / "output.f16"
    input_path.write_bytes(values.tobytes())
    execution = subprocess.check_output(
        [
            str(runner),
            str(directory),
            input_name,
            str(input_path),
            output_name,
            str(output_path),
        ],
        text=True,
    ).strip()
    actual = np.fromfile(output_path, dtype=np.float16)
    weights = dequantize(tensor.data, tensor.tensor_type)
    reference = (
        torch.from_numpy(values.copy()).float()
        @ torch.from_numpy(np.asarray(weights).copy()).float().T
    ).to(torch.float16).numpy()
    if actual.shape != reference.shape:
        raise RuntimeError(
            f"native output shape {actual.shape} differs from reference {reference.shape}"
        )
    delta = np.abs(actual.astype(np.float32) - reference.astype(np.float32))
    result = {
        "model": case["model"],
        "gguf": str(gguf_path),
        "gguf_architecture_provenance": metadata_string(reader, "general.architecture"),
        "tensor": case["tensor"],
        "tensor_shape": [output_features, input_features],
        "quant_type": quant_type,
        "capture": {
            "entry": "torch.compile",
            "fx_nodes": len(manifest["nodes"]),
            "runtime_input": input_name,
            "output": output_name,
        },
        "package_check": package_check.strip(),
        "execution": execution,
        "comparison": {
            "elements": int(actual.size),
            "exact_elements": int(np.count_nonzero(actual == reference)),
            "max_abs": float(delta.max(initial=0.0)),
            "mean_abs": float(delta.mean() if delta.size else 0.0),
            "argmax_equal": bool(np.argmax(actual) == np.argmax(reference)),
            "actual_sha256": sha256(actual),
            "reference_sha256": sha256(reference),
        },
    }
    del weights, reference, actual, tensor, reader
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="_artifacts/gguf-fx-parity")
    parser.add_argument("--output", default="_artifacts/gguf-fx-parity/result.json")
    parser.add_argument("--replace-artifacts", action="store_true")
    for case in CASES:
        parser.add_argument(f"--{case['slug']}-gguf", type=Path)
    args = parser.parse_args()
    root = Path(__file__).parents[1].resolve()
    artifact_root = (root / args.artifact_dir).resolve()
    output = (root / args.output).resolve()
    if artifact_root.exists():
        if not args.replace_artifacts:
            raise FileExistsError(artifact_root)
        shutil.rmtree(artifact_root)
    artifact_root.mkdir(parents=True)
    resolved_cases = []
    for case in CASES:
        supplied = getattr(args, f"{case['slug']}_gguf")
        gguf = supplied or cached_gguf(case["gguf_repo"], case["gguf_file"])
        resolved_cases.append({**case, "gguf": gguf.resolve()})
    results = [
        run_case(case, root=root, artifact_root=artifact_root)
        for case in resolved_cases
    ]
    receipt = {
        "contract": "torch.compile -> FX capture -> llmopt native Metal -> GGUF tensor",
        "torch": torch.__version__,
        "gguf_python": importlib.metadata.version("gguf"),
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
