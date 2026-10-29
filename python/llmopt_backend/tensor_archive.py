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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


MAGIC = b"LLMOPTWT"
VERSION = 1
ALIGNMENT = 256
_PREFIX = struct.Struct("<8sHHIQ")
_ENTRY_PREFIX = struct.Struct("<IBBH")
_U64 = struct.Struct("<Q")

DTYPE_NAMES = (
    "float32", "float16", "bfloat16", "int64", "int32", "int8", "bool", "uint8",
    "Q8_0", "Q4_K", "Q5_K", "Q6_K", "Q5_0", "Q4_0", "IQ4_XS",
    "Tile32_Affine4", "Tile32_Affine8",
)
DTYPE_TAGS = {name: tag for tag, name in enumerate(DTYPE_NAMES)}
DTYPE_TAGS.update(F32=0, F16=1, BF16=2)


@dataclass(frozen=True)
class ArchiveSummary:
    tensor_count: int
    index_bytes: int
    data_bytes: int
    padding_bytes: int
    file_bytes: int


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype_tag: int
    shape: tuple[int, ...]
    byte_length: int
    offset: int = 0

    @property
    def dtype_name(self) -> str:
        return DTYPE_NAMES[self.dtype_tag]


def _align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def _validate(spec: TensorSpec) -> None:
    name, shape = spec.name, spec.shape
    if not 0 <= spec.dtype_tag < len(DTYPE_NAMES):
        raise TypeError(f"tensor {name} has unsupported llmopt dtype tag {spec.dtype_tag}")
    if not name or "\x00" in name:
        raise ValueError(f"invalid llmopt tensor key: {name!r}")
    encoded_name = name.encode("utf-8")
    if len(encoded_name) > 0xFFFF_FFFF:
        raise ValueError(f"llmopt tensor key is too long: {name!r}")
    if len(shape) > 0xFF:
        raise ValueError(f"tensor {name} rank exceeds 255")
    if any(dimension < 0 for dimension in shape):
        raise ValueError(f"tensor {name} has a negative dimension")
    if spec.byte_length <= 0:
        raise ValueError(f"tensor {name} must contain at least one byte")


def _layout(tensors: Iterable[TensorSpec]) -> tuple[bytes, tuple[TensorSpec, ...]]:
    tensors = sorted(tensors, key=lambda spec: spec.name)
    if not tensors:
        raise ValueError("cannot write an empty llmopt weight archive")
    names: set[str] = set()
    index_bytes = _PREFIX.size
    for spec in tensors:
        _validate(spec)
        if spec.name in names:
            raise ValueError(f"duplicate llmopt tensor key: {spec.name!r}")
        names.add(spec.name)
        index_bytes += _ENTRY_PREFIX.size + len(spec.name.encode("utf-8")) + 8 * len(spec.shape) + 16

    data_start = _align(index_bytes)
    cursor = data_start
    specs: list[TensorSpec] = []
    for spec in tensors:
        cursor = _align(cursor)
        specs.append(replace(spec, offset=cursor))
        cursor += spec.byte_length

    header = bytearray()
    header.extend(_PREFIX.pack(MAGIC, VERSION, 0, len(specs), data_start))
    for spec in specs:
        encoded_name = spec.name.encode("utf-8")
        header.extend(
            _ENTRY_PREFIX.pack(len(encoded_name), spec.dtype_tag, len(spec.shape), 0)
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


def write_payloads(
    tensors: Iterable[TensorSpec],
    destination: str | Path,
    chunks: Callable[[str], Iterable[bytes | memoryview]],
) -> ArchiveSummary:
    """Atomically write sorted metadata and bounded payload chunks in ABI-v1 format."""

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
                for data in chunks(spec.name):
                    if output.tell() + len(data) > spec.offset + spec.byte_length:
                        raise ValueError(f"tensor {spec.name} payload exceeds {spec.byte_length} bytes")
                    output.write(data)
                    del data
                if output.tell() != spec.offset + spec.byte_length:
                    raise ValueError(f"tensor {spec.name} payload is shorter than {spec.byte_length} bytes")
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


def write_archive(tensors: Mapping[str, Any], destination: str | Path) -> ArchiveSummary:
    """Export PyTorch tensors, staging at most one tensor on the CPU at a time."""
    specs = []
    for name, tensor in tensors.items():
        dtype = DTYPE_TAGS.get(str(tensor.dtype).removeprefix("torch."))
        if dtype is None:
            raise TypeError(f"tensor {name} has unsupported llmopt dtype {tensor.dtype}")
        specs.append(TensorSpec(name, dtype, tuple(int(d) for d in tensor.shape),
                                int(tensor.numel()) * int(tensor.element_size())))
    return write_payloads(specs, destination, lambda name: (_cpu_bytes(tensors[name]),))


def read_inventory(path: str | Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Read the binary index without loading weight payloads."""
    with Path(path).open("rb") as source:
        magic, version, flags, count, data_start = _PREFIX.unpack(source.read(_PREFIX.size))
        if (magic, version, flags) != (MAGIC, VERSION, 0):
            raise ValueError("unsupported llmopt weight archive header")
        inventory = {}
        for _ in range(count):
            name_length, dtype, rank, reserved = _ENTRY_PREFIX.unpack(source.read(_ENTRY_PREFIX.size))
            name = source.read(name_length).decode("utf-8")
            shape = struct.unpack(f"<{rank}Q", source.read(8 * rank))
            offset, byte_length = struct.unpack("<QQ", source.read(16))
            spec = TensorSpec(name, dtype, shape, byte_length, offset)
            _validate(spec)
            if reserved or name in inventory or offset < data_start or offset % ALIGNMENT:
                raise ValueError(f"invalid llmopt archive entry: {name}")
            inventory[name] = (spec.dtype_name, shape)
        if source.tell() > data_start:
            raise ValueError("llmopt archive index exceeds data start")
        return inventory


__all__ = [
    "ALIGNMENT", "MAGIC", "VERSION", "DTYPE_TAGS", "ArchiveSummary", "TensorSpec",
    "write_archive", "write_payloads", "read_inventory",
]
