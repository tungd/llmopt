"""PyTorch Dynamo/FX entry point for the llmopt research compiler.

The backend keeps graph acquisition in Python, invokes the Ninja-built OCaml
planner, and returns an MPS executable for the captured graph.
"""

from __future__ import annotations

import enum
import json
import itertools
import math
import operator
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import metal_runtime
from .tensor_archive import ArchiveSummary, write_archive


_compile_counter = itertools.count()
_capture_sessions: dict[Path, "CaptureSession"] = {}
_capture_sessions_lock = threading.Lock()


@dataclass(frozen=True)
class CapturedFx:
    manifest: dict[str, Any]
    tensors: dict[str, Any]


@dataclass(frozen=True)
class SessionCapture:
    captured: CapturedFx
    tensor_archive: ArchiveSummary | None


class CaptureSession:
    """Own one static tensor archive shared by every graph in a capture run.

    Dynamo specializes prefill and decode into separate graphs, but their
    lifted model tensors refer to the same storage.  The first graph writes the
    archive; subsequent graphs hard-link it and rebind aliases to the first
    graph's canonical tensor keys.  A later graph may use a subset of those
    tensors, but cannot introduce new static storage after the archive has been
    sealed.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.archive_path = self.root / "weights.llmopt"
        self._archive_summary: ArchiveSummary | None = None
        self._key_by_identity: dict[tuple[Any, ...], str] = {}
        self._identity_by_key: dict[str, tuple[Any, ...]] = {}
        self._lock = threading.Lock()

    def _seal(self, tensors: Mapping[str, Any]) -> None:
        if self.archive_path.exists():
            raise FileExistsError(
                f"capture-session archive already exists: {self.archive_path}"
            )
        self._archive_summary = write_archive(tensors, self.archive_path)
        for key, tensor in tensors.items():
            identity = _tensor_identity(tensor)
            self._key_by_identity[identity] = key
            self._identity_by_key[key] = identity

    def _canonical_keys(self, tensors: Mapping[str, Any]) -> dict[str, str]:
        canonical: dict[str, str] = {}
        for key, tensor in tensors.items():
            identity = _tensor_identity(tensor)
            canonical_key = self._key_by_identity.get(identity)
            if canonical_key is None:
                previous_identity = self._identity_by_key.get(key)
                if previous_identity is not None:
                    raise ValueError(
                        f"static tensor {key} changed storage within capture session"
                    )
                raise ValueError(
                    f"static tensor {key} was not present when the capture-session "
                    "archive was sealed"
                )
            canonical[key] = canonical_key
        return canonical

    @staticmethod
    def _rebind(captured: CapturedFx, canonical: Mapping[str, str]) -> CapturedFx:
        nodes: list[dict[str, Any]] = []
        for node in captured.manifest["nodes"]:
            binding = node.get("binding")
            if isinstance(binding, Mapping) and binding.get("kind") == "tensor-store":
                source_key = str(binding["key"])
                canonical_key = canonical.get(source_key)
                if canonical_key is None:
                    raise ValueError(
                        f"FX tensor binding {source_key} has no captured static tensor"
                    )
                node = {**node, "binding": {"kind": "tensor-store", "key": canonical_key}}
            nodes.append(node)
        manifest = {**captured.manifest, "nodes": nodes}
        tensors = {canonical[key]: tensor for key, tensor in captured.tensors.items()}
        return CapturedFx(manifest=manifest, tensors=tensors)

    def bind(self, captured: CapturedFx, output_directory: str | Path) -> SessionCapture:
        if not captured.tensors:
            return SessionCapture(captured=captured, tensor_archive=None)
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self._archive_summary is None:
                self._seal(captured.tensors)
            canonical = self._canonical_keys(captured.tensors)
            rebound = self._rebind(captured, canonical)
            graph_archive = output / "weights.llmopt"
            if graph_archive.exists():
                raise FileExistsError(
                    f"graph tensor archive already exists: {graph_archive}"
                )
            os.link(self.archive_path, graph_archive)
            return SessionCapture(
                captured=rebound,
                tensor_archive=self._archive_summary,
            )


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


def _argument(value: Any) -> dict[str, Any]:
    """Encode an FX argument without collapsing constants into display text."""

    if hasattr(value, "op") and hasattr(value, "name"):
        return {"kind": "node", "name": str(value.name)}
    if value is None:
        return {"kind": "null"}
    if value is Ellipsis:
        return {"kind": "ellipsis"}
    if type(value) is bool:
        return {"kind": "bool", "value": value}
    if type(value) is int:
        return {"kind": "int", "value": value}
    if type(value) is float:
        if not math.isfinite(value):
            raise TypeError(f"FX argument contains a non-finite float: {value!r}")
        return {"kind": "float", "value": value}
    if isinstance(value, str):
        return {"kind": "string", "value": value}
    if isinstance(value, slice):
        return {
            "kind": "slice",
            "start": _argument(value.start),
            "stop": _argument(value.stop),
            "step": _argument(value.step),
        }
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_argument(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_argument(item) for item in value]}
    if isinstance(value, Mapping):
        items: list[dict[str, Any]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"FX argument mapping key is not a string: {key!r}")
            items.append({"name": key, "value": _argument(item)})
        return {"kind": "mapping", "items": items}
    if isinstance(value, enum.Enum):
        return {
            "kind": "symbol",
            "value": f"{type(value).__module__}.{type(value).__qualname__}.{value.name}",
        }
    module = type(value).__module__
    if module == "torch" or module.startswith("torch."):
        return {"kind": "symbol", "value": str(value)}
    raise TypeError(
        "unsupported FX argument type "
        f"{type(value).__module__}.{type(value).__qualname__}: {value!r}"
    )


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
        # Dynamo's compiled GraphModule commonly records FakeTensor results
        # under ``example_value`` rather than ``val`` or ``tensor_meta``.
        candidate = meta.get("example_value")
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


def _resolve_get_attr(gm: Any, target: Any) -> Any:
    value = gm
    for component in str(target).split("."):
        value = getattr(value, component)
    return value


def _is_static_placeholder(node: Any) -> bool:
    tensor_dict = getattr(node, "meta", {}).get("tensor_dict")
    return isinstance(tensor_dict, Mapping) and bool(
        tensor_dict.get("_dynamo_static_input_type")
    )


def _tensor_identity(tensor: Any) -> tuple[Any, ...]:
    try:
        return (
            str(tensor.device),
            int(tensor.data_ptr()),
            int(tensor.storage_offset()),
            tuple(int(dimension) for dimension in tensor.shape),
            tuple(int(stride) for stride in tensor.stride()),
            str(tensor.dtype),
        )
    except (AttributeError, RuntimeError, TypeError):
        return ("object", id(tensor))


def capture_from_fx(gm: Any, example_inputs: Sequence[Any]) -> CapturedFx:
    """Capture the manifest and its canonical static tensor bindings together."""

    nodes = list(gm.graph.nodes)
    placeholders = [node for node in nodes if node.op == "placeholder"]
    fallback_by_name = {
        node.name: example_inputs[index]
        for index, node in enumerate(placeholders)
        if index < len(example_inputs)
    }
    canonical_by_identity: dict[tuple[Any, ...], str] = {}
    tensor_key_by_node: dict[str, str] = {}
    tensors: dict[str, Any] = {}
    for node in nodes:
        tensor = None
        if node.op == "get_attr":
            tensor = _resolve_get_attr(gm, node.target)
        elif node.op == "placeholder" and _is_static_placeholder(node):
            tensor = fallback_by_name.get(node.name)
        if tensor is None:
            continue
        if not hasattr(tensor, "shape") or not hasattr(tensor, "dtype"):
            raise TypeError(f"static FX node {node.name} does not contain a tensor")
        identity = _tensor_identity(tensor)
        key = canonical_by_identity.get(identity)
        if key is None:
            key = str(node.name)
            canonical_by_identity[identity] = key
            tensors[key] = tensor
        tensor_key_by_node[str(node.name)] = key

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
        tensor_key = tensor_key_by_node.get(str(node.name))
        if tensor_key is not None:
            binding = {"kind": "tensor-store", "key": tensor_key}
        elif node.op == "placeholder":
            binding = {"kind": "runtime"}
        else:
            binding = {"kind": "computed"}
        serialized.append(
            {
                "name": str(node.name),
                "op": str(node.op),
                "target": _target_name(node.target),
                "inputs": refs,
                "shape": shape,
                "dtype": dtype,
                "binding": binding,
                "arguments": {
                    "args": [
                        _argument(value) for value in getattr(node, "args", ())
                    ],
                    "kwargs": [
                        {"name": str(key), "value": _argument(value)}
                        for key, value in getattr(node, "kwargs", {}).items()
                    ],
                },
            }
        )
    return CapturedFx(
        manifest={"version": 2, "nodes": serialized, "outputs": output_names},
        tensors=tensors,
    )


def manifest_from_fx(gm: Any, example_inputs: Sequence[Any]) -> dict[str, Any]:
    """Serialize the stable subset of an FX graph consumed by the OCaml side."""
    return capture_from_fx(gm, example_inputs).manifest


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


def _next_graph_directory(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    while True:
        candidate = root / f"graph-{next(_compile_counter):04d}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue


def _capture_session(root: Path) -> CaptureSession:
    canonical_root = root.resolve()
    with _capture_sessions_lock:
        session = _capture_sessions.get(canonical_root)
        if session is None:
            session = CaptureSession(canonical_root)
            _capture_sessions[canonical_root] = session
        return session


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
        root = Path(artifact_root)
        output_directory = _next_graph_directory(root)
        capture_session = _capture_session(root)
        temporary_directory = None
    else:
        temporary_directory = tempfile.TemporaryDirectory(prefix="llmopt-fx-")
        output_directory = Path(temporary_directory.name)
        capture_session = None

    captured = capture_from_fx(gm, example_inputs)
    tensor_archive: ArchiveSummary | None = None
    tensor_store = output_directory / "weights.llmopt"
    if capture_session is not None:
        session_capture = capture_session.bind(captured, output_directory)
        captured = session_capture.captured
        tensor_archive = session_capture.tensor_archive
    elif captured.tensors:
        tensor_archive = write_archive(captured.tensors, tensor_store)
    manifest = output_directory / "fx.json"
    manifest.write_text(
        json.dumps(captured.manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    quantization = os.environ.get("LLMOPT_QUANTIZATION", "q8")
    metal_library: Path | None = None
    try:
        compiler_command = [str(compiler)]
        if tensor_archive is not None:
            compiler_command.extend(["--weights", tensor_store.name])
        compiler_command.extend([str(manifest), str(output_directory)])
        subprocess.run(
            compiler_command,
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
                        metal_runtime.runtime_kind()
                        if metal_library is not None
                        else "pytorch-mps-fallback"
                    ),
                    "metal_library": (
                        None if metal_library is None else str(metal_library)
                    ),
                    "tensor_store": (
                        None
                        if tensor_archive is None
                        else {
                            "file": tensor_store.name,
                            "tensors": tensor_archive.tensor_count,
                            "index_bytes": tensor_archive.index_bytes,
                            "data_bytes": tensor_archive.data_bytes,
                            "padding_bytes": tensor_archive.padding_bytes,
                            "file_bytes": tensor_archive.file_bytes,
                        }
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
    "CaptureSession",
    "CapturedFx",
    "DirectMpsExecutable",
    "NaiveMpsExecutable",
    "SessionCapture",
    "capture_from_fx",
    "compile_fx",
    "llmopt",
    "manifest_from_fx",
    "write_fx_manifest",
]
