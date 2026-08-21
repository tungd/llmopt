"""Native loading and dispatch for OCaml-generated Q8 Metal libraries.

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

# The Xcode Metal compiler defaults to aggressive fast-math.  The generated
# Q8 reduction is deliberately kept in source order so that its numerical
# behavior is comparable with the MPS fallback; use safe FP32 math while the
# kernel is being used as the model reference implementation.
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
    if _mode() == "native":
        raise RuntimeError(
            "LLMOPT_METAL_RUNTIME=native but the Ninja-built MPS bridge is unavailable"
        )
    return None


def native_available() -> bool:
    return _native() is not None


def reset_dispatch_count() -> None:
    global _dispatches
    _dispatches = 0


def dispatch_count() -> int:
    return _dispatches


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, text=True)


def compile_library(source: Path) -> Path | None:
    """Compile one generated MSL source into a cached metallib when Q8 is present."""

    if _mode() == "off" or not source.exists() or "llmopt_q8_linear" not in source.read_text(
        encoding="utf-8"
    ):
        return None
    if _native() is None:
        return None

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


def dispatch_q8_linear(
    input: Any,
    weight: Any,
    scale: Any,
    bias: Any | None,
) -> Any | None:
    library = _active_library.get()
    if library is None or _mode() == "off":
        return None
    if getattr(input, "device", None) is None or input.device.type != "mps":
        return None
    if getattr(input, "dtype", None) is None or str(input.dtype) not in {
        "torch.float16",
        "torch.float32",
    }:
        return None
    if getattr(weight, "device", None) is None or weight.device.type != "mps":
        return None
    if getattr(scale, "device", None) is None or scale.device.type != "mps":
        return None
    if str(weight.dtype) != "torch.int8" or str(scale.dtype) != "torch.float16":
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
    output = module.q8_linear(input, weight, scale, bias, str(library))
    _dispatches += 1
    return output
