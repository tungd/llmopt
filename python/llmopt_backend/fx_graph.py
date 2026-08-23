"""Versioned binary transport for captured Dynamo/FX graph metadata.

The Python backend and OCaml compiler exchange this format once per Dynamo
specialization. JSON remains available as an explicit diagnostic export, but
is not part of the default compiler subprocess path.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any, Mapping


MAGIC = b"LLMOPTFX"
VERSION = 1
MANIFEST_VERSION = 2
_HEADER = struct.Struct("<8sHHII")
_U8 = struct.Struct("<B")
_U16 = struct.Struct("<H")
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
_I64 = struct.Struct("<q")
_F64 = struct.Struct("<d")

_DTYPE_TO_TAG = {
    "float32": 0,
    "float16": 1,
    "bfloat16": 2,
    "int64": 3,
    "int32": 4,
    "int8": 5,
    "bool": 6,
}
_TAG_TO_DTYPE = {tag: dtype for dtype, tag in _DTYPE_TO_TAG.items()}


class _Writer:
    def __init__(self) -> None:
        self.parts: list[bytes] = []

    def raw(self, value: bytes) -> None:
        self.parts.append(value)

    def u8(self, value: int) -> None:
        self.raw(_U8.pack(value))

    def u16(self, value: int) -> None:
        self.raw(_U16.pack(value))

    def u32(self, value: int) -> None:
        self.raw(_U32.pack(value))

    def u64(self, value: int) -> None:
        self.raw(_U64.pack(value))

    def i64(self, value: int) -> None:
        self.raw(_I64.pack(value))

    def f64(self, value: float) -> None:
        self.raw(_F64.pack(value))

    def string(self, value: str) -> None:
        encoded = value.encode("utf-8")
        self.u32(len(encoded))
        self.raw(encoded)

    def finish(self) -> bytes:
        return b"".join(self.parts)


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    def raw(self, size: int) -> bytes:
        end = self.offset + size
        if size < 0 or end > len(self.data):
            raise ValueError(
                f"truncated binary FX graph at byte {self.offset}: "
                f"need {size} bytes, have {len(self.data) - self.offset}"
            )
        value = self.data[self.offset : end]
        self.offset = end
        return value

    def unpack(self, layout: struct.Struct) -> Any:
        return layout.unpack(self.raw(layout.size))[0]

    def u8(self) -> int:
        return int(self.unpack(_U8))

    def u16(self) -> int:
        return int(self.unpack(_U16))

    def u32(self) -> int:
        return int(self.unpack(_U32))

    def u64(self) -> int:
        return int(self.unpack(_U64))

    def i64(self) -> int:
        return int(self.unpack(_I64))

    def f64(self) -> float:
        return float(self.unpack(_F64))

    def string(self) -> str:
        try:
            return self.raw(self.u32()).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"binary FX graph contains invalid UTF-8: {error}") from error

    def finish(self) -> None:
        if self.offset != len(self.data):
            raise ValueError(
                f"binary FX graph has {len(self.data) - self.offset} trailing bytes"
            )


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"FX {label} must be a list")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"FX {label} must be a mapping")
    return value


def _write_argument(writer: _Writer, argument: Any, depth: int = 0) -> None:
    if depth > 64:
        raise ValueError("FX argument nesting exceeds 64 levels")
    argument = _mapping(argument, "argument")
    kind = argument.get("kind")
    if kind == "node":
        writer.u8(0)
        writer.string(str(argument["name"]))
    elif kind == "null":
        writer.u8(1)
    elif kind == "ellipsis":
        writer.u8(2)
    elif kind == "bool":
        writer.u8(3)
        value = argument["value"]
        if type(value) is not bool:
            raise TypeError("FX bool argument does not contain a bool")
        writer.u8(1 if value else 0)
    elif kind == "int":
        writer.u8(4)
        value = argument["value"]
        if type(value) is not int:
            raise TypeError("FX int argument does not contain an int")
        writer.i64(value)
    elif kind == "float":
        writer.u8(5)
        value = float(argument["value"])
        if not math.isfinite(value):
            raise ValueError("FX float argument must be finite")
        writer.f64(value)
    elif kind == "string":
        writer.u8(6)
        writer.string(str(argument["value"]))
    elif kind == "symbol":
        writer.u8(7)
        writer.string(str(argument["value"]))
    elif kind in ("list", "tuple"):
        writer.u8(8 if kind == "list" else 9)
        items = _sequence(argument.get("items"), f"{kind} argument items")
        writer.u32(len(items))
        for item in items:
            _write_argument(writer, item, depth + 1)
    elif kind == "mapping":
        writer.u8(10)
        items = _sequence(argument.get("items"), "mapping argument items")
        writer.u32(len(items))
        for field in items:
            field = _mapping(field, "mapping argument field")
            writer.string(str(field["name"]))
            _write_argument(writer, field["value"], depth + 1)
    elif kind == "slice":
        writer.u8(11)
        _write_argument(writer, argument["start"], depth + 1)
        _write_argument(writer, argument["stop"], depth + 1)
        _write_argument(writer, argument["step"], depth + 1)
    else:
        raise ValueError(f"unsupported FX argument kind: {kind!r}")


def _read_argument(reader: _Reader, depth: int = 0) -> dict[str, Any]:
    if depth > 64:
        raise ValueError("FX argument nesting exceeds 64 levels")
    tag = reader.u8()
    if tag == 0:
        return {"kind": "node", "name": reader.string()}
    if tag == 1:
        return {"kind": "null"}
    if tag == 2:
        return {"kind": "ellipsis"}
    if tag == 3:
        value = reader.u8()
        if value not in (0, 1):
            raise ValueError(f"invalid binary FX bool value: {value}")
        return {"kind": "bool", "value": value == 1}
    if tag == 4:
        return {"kind": "int", "value": reader.i64()}
    if tag == 5:
        value = reader.f64()
        if not math.isfinite(value):
            raise ValueError("binary FX graph contains a non-finite float")
        return {"kind": "float", "value": value}
    if tag == 6:
        return {"kind": "string", "value": reader.string()}
    if tag == 7:
        return {"kind": "symbol", "value": reader.string()}
    if tag in (8, 9):
        items = [_read_argument(reader, depth + 1) for _ in range(reader.u32())]
        return {"kind": "list" if tag == 8 else "tuple", "items": items}
    if tag == 10:
        items = [
            {"name": reader.string(), "value": _read_argument(reader, depth + 1)}
            for _ in range(reader.u32())
        ]
        return {"kind": "mapping", "items": items}
    if tag == 11:
        return {
            "kind": "slice",
            "start": _read_argument(reader, depth + 1),
            "stop": _read_argument(reader, depth + 1),
            "step": _read_argument(reader, depth + 1),
        }
    raise ValueError(f"unknown binary FX argument tag: {tag}")


def _dtype_tag(value: Any) -> int:
    dtype = str(value).removeprefix("torch.")
    try:
        return _DTYPE_TO_TAG[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported FX dtype: {value}") from error


def _write_node(writer: _Writer, node: Any) -> None:
    node = _mapping(node, "node")
    name = str(node["name"])
    op = str(node["op"])
    writer.string(name)
    writer.string(op)
    writer.string(str(node.get("target", "")))
    writer.u8(_dtype_tag(node.get("dtype", "float32")))

    binding = _mapping(node.get("binding"), f"node {name} binding")
    binding_kind = binding.get("kind")
    if binding_kind == "computed":
        writer.u8(0)
    elif binding_kind == "runtime":
        writer.u8(1)
    elif binding_kind == "tensor-store":
        writer.u8(2)
        writer.string(str(binding["key"]))
    else:
        raise ValueError(f"unsupported FX binding kind: {binding_kind!r}")

    shape = node.get("shape")
    if shape is None:
        writer.u8(0)
    else:
        dimensions = _sequence(shape, f"node {name} shape")
        writer.u8(1)
        writer.u16(len(dimensions))
        for dimension in dimensions:
            if type(dimension) is not int or dimension < 0:
                raise ValueError(f"FX node {name} has invalid dimension {dimension!r}")
            writer.u64(dimension)

    inputs = _sequence(node.get("inputs", []), f"node {name} inputs")
    writer.u32(len(inputs))
    for input_name in inputs:
        writer.string(str(input_name))

    arguments = _mapping(node.get("arguments"), f"node {name} arguments")
    positional = _sequence(arguments.get("args"), f"node {name} arguments.args")
    writer.u32(len(positional))
    for argument in positional:
        _write_argument(writer, argument)
    keywords = _sequence(arguments.get("kwargs"), f"node {name} arguments.kwargs")
    writer.u32(len(keywords))
    for field in keywords:
        field = _mapping(field, f"node {name} keyword argument")
        writer.string(str(field["name"]))
        _write_argument(writer, field["value"])


def _read_node(reader: _Reader) -> dict[str, Any]:
    name = reader.string()
    op = reader.string()
    target = reader.string()
    dtype_tag = reader.u8()
    try:
        dtype = _TAG_TO_DTYPE[dtype_tag]
    except KeyError as error:
        raise ValueError(f"unknown binary FX dtype tag: {dtype_tag}") from error

    binding_tag = reader.u8()
    if binding_tag == 0:
        binding = {"kind": "computed"}
    elif binding_tag == 1:
        binding = {"kind": "runtime"}
    elif binding_tag == 2:
        binding = {"kind": "tensor-store", "key": reader.string()}
    else:
        raise ValueError(f"unknown binary FX binding tag: {binding_tag}")

    shape_tag = reader.u8()
    if shape_tag == 0:
        shape = None
    elif shape_tag == 1:
        shape = [reader.u64() for _ in range(reader.u16())]
    else:
        raise ValueError(f"unknown binary FX shape tag: {shape_tag}")

    inputs = [reader.string() for _ in range(reader.u32())]
    positional = [_read_argument(reader) for _ in range(reader.u32())]
    keywords = [
        {"name": reader.string(), "value": _read_argument(reader)}
        for _ in range(reader.u32())
    ]
    return {
        "name": name,
        "op": op,
        "target": target,
        "inputs": inputs,
        "shape": shape,
        "dtype": dtype,
        "binding": binding,
        "arguments": {"args": positional, "kwargs": keywords},
    }


def encode_manifest(manifest: Mapping[str, Any]) -> bytes:
    version = manifest.get("version")
    if version != MANIFEST_VERSION:
        raise ValueError(
            f"binary FX transport requires manifest version {MANIFEST_VERSION}, "
            f"got {version!r}"
        )
    nodes = _sequence(manifest.get("nodes"), "nodes")
    outputs = _sequence(manifest.get("outputs"), "outputs")
    writer = _Writer()
    writer.raw(_HEADER.pack(MAGIC, VERSION, MANIFEST_VERSION, len(nodes), len(outputs)))
    for node in nodes:
        _write_node(writer, node)
    for output in outputs:
        writer.string(str(output))
    return writer.finish()


def decode_manifest(data: bytes) -> dict[str, Any]:
    reader = _Reader(data)
    magic, version, manifest_version, node_count, output_count = _HEADER.unpack(
        reader.raw(_HEADER.size)
    )
    if magic != MAGIC:
        raise ValueError("invalid binary FX graph magic")
    if version != VERSION:
        raise ValueError(f"unsupported binary FX graph version: {version}")
    if manifest_version != MANIFEST_VERSION:
        raise ValueError(f"unsupported FX manifest version: {manifest_version}")
    nodes = [_read_node(reader) for _ in range(node_count)]
    outputs = [reader.string() for _ in range(output_count)]
    reader.finish()
    return {"version": manifest_version, "nodes": nodes, "outputs": outputs}


def write_graph(manifest: Mapping[str, Any], destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_manifest(manifest))
    return path


def read_graph(source: str | Path) -> dict[str, Any]:
    return decode_manifest(Path(source).read_bytes())


__all__ = [
    "MAGIC",
    "MANIFEST_VERSION",
    "VERSION",
    "decode_manifest",
    "encode_manifest",
    "read_graph",
    "write_graph",
]
