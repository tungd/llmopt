"""Streaming binary weight export for captured static FX inputs.

The serving archive is intentionally owned by llmopt instead of inheriting a
JSON-indexed interchange format. Its index and tensor payload are both binary,
and every payload begins at a 256-byte boundary suitable for direct Metal
buffer views. At most one non-CPU tensor is staged on the CPU at a time.
"""

from __future__ import annotations

import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


MAGIC = b"LLMOPTWT"
VERSION = 1
ALIGNMENT = 256
_PREFIX = struct.Struct("<8sHHIQ")
_ENTRY_PREFIX = struct.Struct("<IBBH")
_U64 = struct.Struct("<Q")

_DTYPES = {
    "torch.float32": 0,
    "torch.float16": 1,
    "torch.bfloat16": 2,
    "torch.int64": 3,
    "torch.int32": 4,
    "torch.int8": 5,
    "torch.bool": 6,
    "torch.uint8": 7,
}


@dataclass(frozen=True)
class ArchiveSummary:
    tensor_count: int
    index_bytes: int
    data_bytes: int
    padding_bytes: int
    file_bytes: int


@dataclass(frozen=True)
class _TensorSpec:
    name: str
    dtype: int
    shape: tuple[int, ...]
    offset: int
    byte_length: int


def _align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def _tensor_metadata(name: str, tensor: Any) -> tuple[int, tuple[int, ...], int]:
    dtype = _DTYPES.get(str(tensor.dtype))
    if dtype is None:
        raise TypeError(f"tensor {name} has unsupported llmopt dtype {tensor.dtype}")
    if not name or "\x00" in name:
        raise ValueError(f"invalid llmopt tensor key: {name!r}")
    encoded_name = name.encode("utf-8")
    if len(encoded_name) > 0xFFFF_FFFF:
        raise ValueError(f"llmopt tensor key is too long: {name!r}")
    shape = tuple(int(dimension) for dimension in tensor.shape)
    if len(shape) > 0xFF:
        raise ValueError(f"tensor {name} rank exceeds 255")
    if any(dimension < 0 for dimension in shape):
        raise ValueError(f"tensor {name} has a negative dimension")
    byte_length = int(tensor.numel()) * int(tensor.element_size())
    if byte_length <= 0:
        raise ValueError(f"tensor {name} must contain at least one byte")
    return dtype, shape, byte_length


def _layout(tensors: Mapping[str, Any]) -> tuple[bytes, tuple[_TensorSpec, ...]]:
    if not tensors:
        raise ValueError("cannot write an empty llmopt weight archive")
    metadata: list[tuple[str, bytes, int, tuple[int, ...], int]] = []
    index_bytes = _PREFIX.size
    for name in sorted(tensors):
        dtype, shape, byte_length = _tensor_metadata(name, tensors[name])
        encoded_name = name.encode("utf-8")
        index_bytes += _ENTRY_PREFIX.size + len(encoded_name) + 8 * len(shape) + 16
        metadata.append((name, encoded_name, dtype, shape, byte_length))

    data_start = _align(index_bytes)
    cursor = data_start
    specs: list[_TensorSpec] = []
    for name, _encoded_name, dtype, shape, byte_length in metadata:
        cursor = _align(cursor)
        specs.append(_TensorSpec(name, dtype, shape, cursor, byte_length))
        cursor += byte_length

    header = bytearray()
    header.extend(_PREFIX.pack(MAGIC, VERSION, 0, len(specs), data_start))
    for spec, (_, encoded_name, _dtype, _shape, _byte_length) in zip(
        specs, metadata, strict=True
    ):
        header.extend(
            _ENTRY_PREFIX.pack(len(encoded_name), spec.dtype, len(spec.shape), 0)
        )
        header.extend(encoded_name)
        for dimension in spec.shape:
            header.extend(_U64.pack(dimension))
        header.extend(_U64.pack(spec.offset))
        header.extend(_U64.pack(spec.byte_length))
    if len(header) > data_start:
        raise AssertionError("llmopt archive index layout overflowed")
    header.extend(b"\x00" * (data_start - len(header)))
    return bytes(header), tuple(specs)


def _cpu_bytes(tensor: Any) -> memoryview:
    import torch

    cpu = tensor.detach().contiguous().to(device="cpu")
    if cpu.dtype == torch.bfloat16:
        cpu = cpu.view(torch.uint16)
    return memoryview(cpu.numpy()).cast("B")


def write_archive(
    tensors: Mapping[str, Any],
    destination: str | Path,
) -> ArchiveSummary:
    """Atomically write one deterministic, JSON-free serving archive."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    header, specs = _layout(tensors)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(header)
            for spec in specs:
                position = output.tell()
                if position > spec.offset:
                    raise AssertionError("llmopt archive payload layout overflowed")
                output.write(b"\x00" * (spec.offset - position))
                data = _cpu_bytes(tensors[spec.name])
                if len(data) != spec.byte_length:
                    raise ValueError(
                        f"tensor {spec.name} yielded {len(data)} bytes; "
                        f"expected {spec.byte_length}"
                    )
                output.write(data)
                del data
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    data_bytes = sum(spec.byte_length for spec in specs)
    file_bytes = specs[-1].offset + specs[-1].byte_length
    return ArchiveSummary(
        tensor_count=len(specs),
        index_bytes=len(header),
        data_bytes=data_bytes,
        padding_bytes=file_bytes - len(header) - data_bytes,
        file_bytes=file_bytes,
    )


__all__ = [
    "ALIGNMENT",
    "MAGIC",
    "VERSION",
    "ArchiveSummary",
    "write_archive",
]
