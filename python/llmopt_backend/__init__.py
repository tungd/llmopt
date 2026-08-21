"""PyTorch Dynamo/FX entry point for the llmopt research compiler.

The backend keeps graph acquisition in Python, invokes the Ninja-built OCaml
planner, and returns an MPS executable for the captured graph.
"""

from __future__ import annotations

import json
import itertools
import operator
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import metal_runtime


_compile_counter = itertools.count()


class NaiveMpsExecutable:
    """Execute the captured FX graph through PyTorch's normal device dispatch.

    This runtime intentionally performs no fusion or scheduling. When the
    module parameters and inputs live on MPS, each FX operation therefore uses
    PyTorch MPS directly while the OCaml planner records and validates the
    graph.
    """

    def __init__(self, graph_module: Any):
        self.graph_module = graph_module

    def __call__(self, *args: Any, **kwargs: Any):
        try:
            import torch.fx
        except ImportError:
            return self.graph_module.forward(*args, **kwargs)
        if kwargs:
            return self.graph_module.forward(*args, **kwargs)
        return torch.fx.Interpreter(self.graph_module).run(*args)


class DirectMpsExecutable:
    """Run the generated FX forward directly, with optional generated Metal Q8.

    This is the first executable optimization pass: graph capture and OCaml
    planning stay unchanged, while the runtime removes the Python
    ``torch.fx.Interpreter`` loop from the hot path.  When a generated Q8
    ``metallib`` is available, the Q8 operator dispatches through it while the
    GraphModule remains the graph semantics authority.
    """

    def __init__(
        self,
        graph_module: Any,
        *,
        metal_library: Path | None = None,
        temporary_directory: tempfile.TemporaryDirectory[str] | None = None,
    ):
        self.graph_module = graph_module
        self.metal_library = metal_library
        self._temporary_directory = temporary_directory

    def __call__(self, *args: Any, **kwargs: Any):
        context = metal_runtime.activate(self.metal_library)
        with context:
            return self.graph_module(*args, **kwargs)


def _target_name(target: Any) -> str:
    if isinstance(target, str):
        return target
    if target is operator.add:
        return "aten.add.Tensor"
    if target is operator.mul:
        return "aten.mul.Tensor"
    if target is operator.matmul:
        return "aten.matmul.default"

    schema = getattr(target, "_schema", None)
    if schema is not None:
        name = str(getattr(schema, "name", target)).replace("::", ".")
        overload = str(getattr(schema, "overload_name", ""))
        if name.startswith("aten."):
            return f"{name}.{overload}" if overload else name
        return name

    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return str(target)


def _node_refs(value: Any) -> Iterable[str]:
    # Importing torch is intentionally avoided here so manifest tests can run
    # in the small Python environment used by the OCaml build.
    if hasattr(value, "op") and hasattr(value, "name"):
        yield str(value.name)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _node_refs(item)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _node_refs(item)


def _first_tensor(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _shape(value: Any) -> list[int] | None:
    tensor = _first_tensor(value)
    if tensor is None:
        return None
    dims: list[int] = []
    for dimension in tuple(tensor.shape):
        # SymInts are deliberately not coerced: the first backend slice is
        # static-shape and must leave symbolic dimensions visible as absent.
        if type(dimension) is not int:
            return None
        dims.append(dimension)
    return dims


def _dtype(value: Any) -> str:
    tensor = _first_tensor(value)
    if tensor is None:
        return "float32"
    return str(tensor.dtype).removeprefix("torch.")


def _metadata(node: Any, fallback: Any = None) -> tuple[list[int] | None, str]:
    meta = getattr(node, "meta", {})
    candidate = meta.get("val")
    if candidate is None:
        candidate = meta.get("tensor_meta")
    if candidate is None:
        candidate = fallback
    return _shape(candidate), _dtype(candidate)


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def manifest_from_fx(gm: Any, example_inputs: Sequence[Any]) -> dict[str, Any]:
    """Serialize the stable subset of an FX graph consumed by the OCaml side."""

    nodes = list(gm.graph.nodes)
    placeholders = [node for node in nodes if node.op == "placeholder"]
    fallback_by_name = {
        node.name: example_inputs[index]
        for index, node in enumerate(placeholders)
        if index < len(example_inputs)
    }

    serialized: list[dict[str, Any]] = []
    output_names: list[str] = []
    for node in nodes:
        shape, dtype = _metadata(node, fallback_by_name.get(node.name))
        refs = _unique(
            ref
            for value in (getattr(node, "args", ()), getattr(node, "kwargs", {}))
            for ref in _node_refs(value)
        )
        if node.op == "output":
            output_names = refs
        serialized.append(
            {
                "name": str(node.name),
                "op": str(node.op),
                "target": _target_name(node.target),
                "inputs": refs,
                "shape": shape,
                "dtype": dtype,
            }
        )

    return {"version": 1, "nodes": serialized, "outputs": output_names}


def write_fx_manifest(gm: Any, example_inputs: Sequence[Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest_from_fx(gm, example_inputs), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _compiler_path() -> Path | None:
    configured = os.environ.get("LLMOPT_FX_COMPILER")
    if configured:
        return Path(configured)
    repository = Path(__file__).resolve().parents[2]
    candidate = repository / "_build" / "bin" / "llmopt-fx"
    return candidate if candidate.exists() else None


def compile_fx(gm: Any, example_inputs: Sequence[Any]):
    """Plan an FX graph with OCaml and return its direct MPS executable."""

    compiler = _compiler_path()
    if compiler is None:
        if os.environ.get("LLMOPT_FX_FALLBACK", "1") == "1":
            return NaiveMpsExecutable(gm)
        raise RuntimeError(
            "llmopt-fx is not available; run `ninja -f ninja.build all` or set "
            "LLMOPT_FX_COMPILER"
        )

    artifact_root = os.environ.get("LLMOPT_ARTIFACT_DIR")
    if artifact_root:
        output_directory = Path(artifact_root) / f"graph-{next(_compile_counter):04d}"
        output_directory.mkdir(parents=True, exist_ok=True)
        temporary_directory = None
    else:
        temporary_directory = tempfile.TemporaryDirectory(prefix="llmopt-fx-")
        output_directory = Path(temporary_directory.name)

    manifest = output_directory / "fx.json"
    write_fx_manifest(gm, example_inputs, manifest)
    quantization = os.environ.get("LLMOPT_QUANTIZATION", "q8")
    metal_library: Path | None = None
    try:
        subprocess.run(
            [str(compiler), str(manifest), str(output_directory)],
            check=True,
            text=True,
        )
        metal_library = metal_runtime.compile_library(output_directory / "kernel.metal")
        (output_directory / "runtime.json").write_text(
            json.dumps(
                {
                    "target": "pytorch-mps",
                    "mode": "fx-graphmodule",
                    "optimization": f"fx-direct-execution+{quantization}",
                    "quantization": quantization,
                    "runtime": (
                        "generated-metal-q8"
                        if metal_library is not None
                        else "pytorch-mps-fallback"
                    ),
                    "metal_library": (
                        None if metal_library is None else str(metal_library)
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        if temporary_directory is not None and metal_library is None:
            # The compiler output is useful for debugging only when an artifact
            # directory was explicitly requested.
            temporary_directory.cleanup()
            temporary_directory = None
    return DirectMpsExecutable(
        gm,
        metal_library=metal_library,
        temporary_directory=temporary_directory,
    )


def llmopt(gm: Any, example_inputs: Sequence[Any]):
    return compile_fx(gm, example_inputs)


try:  # pragma: no cover - exercised in the Torch environment
    from torch._dynamo import register_backend

    register_backend(llmopt)
except ImportError:
    pass


__all__ = [
    "DirectMpsExecutable",
    "NaiveMpsExecutable",
    "compile_fx",
    "llmopt",
    "manifest_from_fx",
    "write_fx_manifest",
]
