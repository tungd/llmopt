#!/usr/bin/env python3
"""Run one bounded lowering probe for the captured Gemma 4 MTP graphs.

This probe deliberately stops at the first failed stage.  Its packages contain
meta-device graph metadata only: it neither writes model weights nor reports a
Gemma execution or benchmark result.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "bench"))

from gemma4_mtp_capture import (
    _assistant_arguments,
    _assistant_wrapper_type,
    _load_target_config,
    _target_arguments,
    _target_wrapper_type,
    build_assistant_text_config,
    expected_target_geometry,
    target_layer_geometry,
)
from llmopt_backend import capture_from_fx
from llmopt_backend.fx_graph import write_graph


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def capture(module: Any, arguments: tuple[Any, ...]) -> Any:
    import torch

    captured: dict[str, Any] = {}

    def backend(graph_module: Any, example_inputs: Sequence[Any]) -> Any:
        captured["graph"] = capture_from_fx(graph_module, example_inputs)
        return graph_module.forward

    torch._dynamo.reset()
    torch.compile(module, backend=backend, fullgraph=True, dynamic=False)(*arguments)
    try:
        return captured["graph"]
    except KeyError as error:
        raise RuntimeError("Dynamo completed without yielding an FX graph") from error


def models() -> tuple[Any, Any, tuple[Any, ...]]:
    import torch
    from transformers import (
        Gemma4AssistantConfig,
        Gemma4AssistantForCausalLM,
        Gemma4ForCausalLM,
    )

    target_config, _ = _load_target_config(local_only=True)
    target_config._attn_implementation = "sdpa"
    geometry = target_layer_geometry(target_config)
    if geometry != expected_target_geometry():
        raise RuntimeError("pinned Gemma target geometry differs from the receipt")
    assistant_text_config = build_assistant_text_config(target_config)
    assistant_text_config._attn_implementation = "sdpa"
    assistant_config = Gemma4AssistantConfig(
        text_config=assistant_text_config,
        backbone_hidden_size=target_config.hidden_size,
        use_ordered_embeddings=False,
    )
    with torch.device("meta"):
        target = Gemma4ForCausalLM(target_config).to(dtype=torch.float16)
        assistant = Gemma4AssistantForCausalLM(assistant_config).to(
            dtype=torch.float16
        )
    target.eval()
    assistant.eval()
    return target, assistant, geometry


def stages(target: Any, assistant: Any, geometry: tuple[Any, ...]):
    return (
        (
            "target-prefill",
            _target_wrapper_type()(target),
            _target_arguments(geometry, query_tokens=2, past_tokens=0),
        ),
        (
            "target-decode",
            _target_wrapper_type()(target),
            _target_arguments(geometry, query_tokens=1, past_tokens=2),
        ),
        (
            "assistant-step",
            _assistant_wrapper_type()(assistant),
            _assistant_arguments(shared_tokens=3),
        ),
    )


def compile_metal_and_validate(directory: Path) -> dict[str, Any]:
    metal = directory / "kernel.metal"
    air = directory / "kernel.air"
    library = directory / "kernel.metallib"
    metal_compile = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(metal), "-o", str(air)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if metal_compile.returncode != 0:
        return {
            "metal_compile_exit_code": metal_compile.returncode,
            "metal_compile_stdout": metal_compile.stdout,
            "metal_compile_stderr": metal_compile.stderr,
        }
    metallib = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(library)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if metallib.returncode != 0:
        return {
            "metal_compile_exit_code": 0,
            "metallib_exit_code": metallib.returncode,
            "metallib_stdout": metallib.stdout,
            "metallib_stderr": metallib.stderr,
        }
    package_check = subprocess.run(
        [str(REPO_ROOT / "_build/bin/llmopt-package-check"), str(directory)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    opaque_by_target: dict[str, int] = {}
    for line in (directory / "plan.txt").read_text(encoding="utf-8").splitlines():
        match = re.search(r"opaque\([^,]+,([^)]*)\)", line)
        if match is not None:
            target = match.group(1)
            opaque_by_target[target] = opaque_by_target.get(target, 0) + 1
    return {
        "metal_compile_exit_code": 0,
        "metallib_exit_code": 0,
        "package_check_exit_code": package_check.returncode,
        "package_check_stdout": package_check.stdout,
        "package_check_stderr": package_check.stderr,
        "opaque_commands": sum(opaque_by_target.values()),
        "opaque_by_target": opaque_by_target,
    }


def lower(stage: str, module: Any, arguments: tuple[Any, ...], output: Path) -> dict[str, Any]:
    graph = capture(module, arguments)
    directory = output / stage
    directory.mkdir(parents=True, exist_ok=False)
    graph_path = directory / "graph.llmopt"
    write_graph(graph.manifest, graph_path)
    (directory / "fx.json").write_text(
        json.dumps(graph.manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    compiler = REPO_ROOT / "_build/bin/llmopt-fx"
    completed = subprocess.run(
        [str(compiler), str(graph_path), str(directory)],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    result = {
        "stage": stage,
        "fx_nodes": len(graph.manifest["nodes"]),
        "static_tensors": len(graph.tensors),
        "runtime_inputs": sum(
            node["op"] == "placeholder"
            and node["binding"]["kind"] == "runtime"
            for node in graph.manifest["nodes"]
        ),
        "compiler_exit_code": completed.returncode,
        "compiler_stdout": completed.stdout,
        "compiler_stderr": completed.stderr,
        "package_written": (directory / "package.llmopt").is_file(),
    }
    if completed.returncode == 0:
        result.update(compile_metal_and_validate(directory))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lower the pinned meta-device Gemma MTP graphs once, stopping at the first failure"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    target, assistant, geometry = models()
    report: dict[str, Any] = {
        "kind": "gemma4-mtp-lowering-probe",
        "mode": "meta-graph-only",
        "stages": [],
    }
    for stage, module, arguments in stages(target, assistant, geometry):
        result = lower(stage, module, arguments, output)
        report["stages"].append(result)
        write_json_atomic(output / "report.json", report)
        if result["compiler_exit_code"] != 0:
            return result["compiler_exit_code"]
        if result.get("metal_compile_exit_code", 0) != 0:
            return int(result["metal_compile_exit_code"])
        if result.get("metallib_exit_code", 0) != 0:
            return int(result["metallib_exit_code"])
        if result.get("package_check_exit_code", 0) != 0:
            return int(result["package_check_exit_code"])
        if result.get("opaque_commands", 0) != 0:
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
