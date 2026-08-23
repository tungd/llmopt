"""Streaming safetensors export for captured static FX inputs.

The official torch serializer keeps every non-CPU conversion alive until the
whole file is written.  A serving compile can receive all model tensors on MPS,
so this writer emits one tensor at a time and releases its CPU staging copy
before moving to the next tensor.
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_DTYPES = {
    "torch.float32": "F32",
    "torch.float16": "F16",
    "torch.bfloat16": "BF16",
    "torch.int64": "I64",
    "torch.int32": "I32",
    "torch.int8": "I8",
    "torch.bool": "BOOL",
}


@dataclass(frozen=True)
class ArchiveSummary:
    tensor_count: int
    data_bytes: int
    file_bytes: int


def _tensor_spec(name: str, tensor: Any, begin: int) -> tuple[dict[str, Any], int]:
    dtype = _DTYPES.get(str(tensor.dtype))
    if dtype is None:
        raise TypeError(f"tensor {name} has unsupported safetensors dtype {tensor.dtype}")
    shape = [int(dimension) for dimension in tensor.shape]
    byte_length = int(tensor.numel()) * int(tensor.element_size())
    return {
        "dtype": dtype,
        "shape": shape,
        "data_offsets": [begin, begin + byte_length],
    }, byte_length


def _header(tensors: Mapping[str, Any], metadata: Mapping[str, str]) -> tuple[bytes, int]:
    entries: dict[str, Any] = {"__metadata__": dict(sorted(metadata.items()))}
    cursor = 0
    for name in sorted(tensors):
        if not name or name == "__metadata__":
            raise ValueError(f"invalid safetensors key: {name!r}")
        spec, byte_length = _tensor_spec(name, tensors[name], cursor)
        entries[name] = spec
        cursor += byte_length
    encoded = json.dumps(entries, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    encoded += b" " * (-len(encoded) % 8)
    return encoded, cursor


def _cpu_bytes(tensor: Any) -> memoryview:
    import torch

    cpu = tensor.detach().contiguous().to(device="cpu")
    if cpu.dtype == torch.bfloat16:
        cpu = cpu.view(torch.uint16)
    return memoryview(cpu.numpy()).cast("B")


def write_safetensors(
    tensors: Mapping[str, Any],
    destination: str | Path,
    *,
    metadata: Mapping[str, str] | None = None,
) -> ArchiveSummary:
    """Write one deterministic archive with at most one tensor staged on CPU."""

    if not tensors:
        raise ValueError("cannot write an empty serving tensor archive")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    header, data_bytes = _header(
        tensors,
        {"format": "pt", **({} if metadata is None else dict(metadata))},
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(struct.pack("<Q", len(header)))
            output.write(header)
            for name in sorted(tensors):
                data = _cpu_bytes(tensors[name])
                expected = int(tensors[name].numel()) * int(
                    tensors[name].element_size()
                )
                if len(data) != expected:
                    raise ValueError(
                        f"tensor {name} yielded {len(data)} bytes; expected {expected}"
                    )
                output.write(data)
                del data
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return ArchiveSummary(
        tensor_count=len(tensors),
        data_bytes=data_bytes,
        file_bytes=8 + len(header) + data_bytes,
    )


__all__ = ["ArchiveSummary", "write_safetensors"]
