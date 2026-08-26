"""Native loading and dispatch for OCaml-generated Metal libraries.

The Python layer owns library lifetime and fallback selection; the small
PyTorch C++ bridge binds MPS tensor storage to the generated kernel.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import importlib.util
import os
import subprocess
from typing import Any, Iterator


_active_library: ContextVar[Path | None] = ContextVar(
    "llmopt_active_metal_library", default=None
)
_native_module: Any | None = None
_dispatches = 0
_compiled_runtime_kind = "generated-metal-native"

# Keep generated W4A16 arithmetic deterministic across compiler revisions.
_METAL_FLAGS = (
    "-fmetal-math-mode=safe",
    "-fmetal-math-fp32-functions=precise",
    "-ffp-contract=on",
)


def _mode() -> str:
    return os.environ.get("LLMOPT_METAL_RUNTIME", "auto").lower()


def _native_candidates() -> list[Path]:
    configured = os.environ.get("LLMOPT_METAL_RUNTIME_MODULE")
    if configured:
        return [Path(configured)]
    repository = Path(__file__).resolve().parents[2]
    return sorted((repository / "_build" / "native").glob("_llmopt_metal_runtime*.so"))


def _native() -> Any | None:
    global _native_module
    if _native_module is not None:
        return _native_module
    for candidate in _native_candidates():
        if not candidate.exists():
            continue
        spec = importlib.util.spec_from_file_location(
            "_llmopt_metal_runtime", candidate
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _native_module = module
        return module
    if _mode() in {"native", "exact"}:
        raise RuntimeError(
            "LLMOPT_METAL_RUNTIME requires the Ninja-built MPS bridge, but it is unavailable"
        )
    return None


def native_available() -> bool:
    return _native() is not None


def reset_dispatch_count() -> None:
    global _dispatches
    _dispatches = 0


def dispatch_count() -> int:
    return _dispatches


def runtime_kind() -> str:
    if _mode() == "off":
        return "pytorch-mps-fallback"
    return _compiled_runtime_kind


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, text=True)


def compile_library(source: Path) -> Path | None:
    """Compile a generated W4A16 MSL source into a metallib."""

    if _mode() == "off" or not source.exists():
        return None
    generated_source = source.read_text(encoding="utf-8")
    if "llmopt_w4a16_linear_f16_g64" not in generated_source:
        return None
    if _native() is None:
        return None

    global _compiled_runtime_kind
    _compiled_runtime_kind = "generated-metal-w4a16-f16-g64"

    air = source.with_suffix(".air")
    library = source.with_suffix(".metallib")
    flags_stamp = source.with_suffix(".metal-flags")
    flags_text = " ".join(_METAL_FLAGS) + "\n"
    flags_changed = not flags_stamp.exists() or flags_stamp.read_text(encoding="utf-8") != flags_text
    if flags_changed or not library.exists() or library.stat().st_mtime < source.stat().st_mtime:
        _run(["xcrun", "metal", *_METAL_FLAGS, "-c", str(source), "-o", str(air)])
        _run(["xcrun", "metallib", str(air), "-o", str(library)])
        flags_stamp.write_text(flags_text, encoding="utf-8")
    return library


@contextmanager
def activate(library: Path | None) -> Iterator[None]:
    token = _active_library.set(library)
    try:
        yield
    finally:
        _active_library.reset(token)


def dispatch_w4a16_linear(
    input: Any,
    packed_weight: Any,
    scale: Any,
    bias: Any | None,
) -> Any | None:
    """Dispatch the fixed packed-W4/group-64, FP16-activation kernel."""

    library = _active_library.get()
    if library is None or _mode() == "off":
        return None
    tensors = (input, packed_weight, scale)
    if any(getattr(tensor, "device", None) is None for tensor in tensors):
        return None
    if any(tensor.device.type != "mps" for tensor in tensors):
        return None
    if str(input.dtype) != "torch.float16":
        return None
    if str(packed_weight.dtype) != "torch.uint8":
        return None
    if str(scale.dtype) != "torch.float16":
        return None
    if bias is not None:
        if getattr(bias, "device", None) is None or bias.device.type != "mps":
            return None
        if str(bias.dtype) != "torch.float16":
            return None
    module = _native()
    if module is None:
        return None
    global _dispatches
    output = module.w4a16_linear(
        input, packed_weight, scale, bias, str(library)
    )
    _dispatches += 1
    return output
