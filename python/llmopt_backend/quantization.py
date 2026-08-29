"""Experimental static W4A16 probe utilities for the MPS target.

Weights are stored as symmetric signed int4 values in two's-complement
nibbles.  The low nibble represents the lower K index, and each output row is
split into groups of 64 K values with one FP16 scale per group.  The custom
operator keeps that physical representation visible to FX; its Python
implementation is the FP32-accumulating reference fallback.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


GROUP_SIZE = 64
_QMIN = -8
_QMAX = 7
_W4_LIBRARY: Any | None = None


def _validate_weight_shape(
    weight: torch.Tensor, label: str = "W4 weight"
) -> tuple[int, int, int]:
    if weight.ndim != 2:
        raise ValueError(f"{label} must be 2-D, got shape {tuple(weight.shape)}")
    output_features, input_features = (int(dimension) for dimension in weight.shape)
    if input_features <= 0 or output_features <= 0:
        raise ValueError(f"{label} dimensions must be positive")
    if input_features % GROUP_SIZE != 0:
        raise ValueError(
            f"{label} K dimension must be a multiple of {GROUP_SIZE}, "
            f"got {input_features}"
        )
    return output_features, input_features, input_features // GROUP_SIZE


def _validate_packed_weight_shape(
    packed_weight: torch.Tensor,
) -> tuple[int, int, int]:
    if packed_weight.ndim != 2:
        raise ValueError(
            "packed W4 weight must be 2-D, "
            f"got shape {tuple(packed_weight.shape)}"
        )
    output_features, packed_features = (
        int(dimension) for dimension in packed_weight.shape
    )
    logical_features = packed_features * 2
    if output_features <= 0 or packed_features <= 0:
        raise ValueError("packed W4 weight dimensions must be positive")
    if logical_features % GROUP_SIZE != 0:
        raise ValueError(
            f"packed W4 weight logical K dimension must be a multiple of "
            f"{GROUP_SIZE}, got {logical_features}"
        )
    return output_features, packed_features, logical_features // GROUP_SIZE


def pack_int4_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a floating ``[N, K]`` weight into W4 bytes and FP16 scales.

    Quantization is symmetric per output row and per contiguous group of 64 K
    values.  A packed byte contains the two's-complement nibble for K=2*i in
    its low half and K=2*i+1 in its high half.
    """

    output_features, input_features, groups = _validate_weight_shape(weight)
    source = weight.detach().float().reshape(output_features, groups, GROUP_SIZE)
    scale = (
        source.abs()
        .amax(dim=-1)
        .clamp_min(torch.finfo(torch.float32).eps)
        / float(_QMAX)
    )
    quantized = torch.round(source / scale.unsqueeze(-1)).clamp(_QMIN, _QMAX).to(
        torch.int16
    )

    # remainder gives the unsigned two's-complement nibble for negative q.
    nibbles = torch.remainder(quantized, 16).to(torch.uint8)
    low = nibbles[..., 0::2]
    high = nibbles[..., 1::2]
    packed = (low | (high << 4)).reshape(output_features, input_features // 2)
    return packed.contiguous(), scale.to(dtype=torch.float16).contiguous()


def unpack_int4_weight(
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Return the FP32 logical ``[N, K]`` weight represented by W4 storage."""

    output_features, packed_k, groups = _validate_packed_weight_shape(packed_weight)
    if packed_weight.dtype != torch.uint8:
        raise TypeError(
            f"packed W4 weight storage must be torch.uint8, got {packed_weight.dtype}"
        )
    if scale.dtype != torch.float16:
        raise TypeError(f"W4 scales must be torch.float16, got {scale.dtype}")
    if tuple(scale.shape) != (output_features, groups):
        raise ValueError(
            "W4 scales must have shape "
            f"[{output_features}, {groups}], got {tuple(scale.shape)}"
        )

    packed = packed_weight.reshape(output_features, groups, GROUP_SIZE // 2)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibbles = torch.stack((low, high), dim=-1).reshape(
        output_features, groups, GROUP_SIZE
    )
    # Values 8..15 are the negative two's-complement values -8..-1.
    signed = torch.where(nibbles < 8, nibbles, nibbles.to(torch.int16) - 16)
    return (
        signed.to(dtype=torch.float32)
        * scale.to(dtype=torch.float32).unsqueeze(-1)
    ).reshape(output_features, packed_k * 2)


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a floating 2-D weight using this probe's W4A16 contract."""

    return pack_int4_weight(weight)


def _w4a16_linear_impl(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    from . import metal_runtime

    dispatch = getattr(metal_runtime, "dispatch_w4a16_linear", None)
    if dispatch is not None:
        native_output = dispatch(input, packed_weight, scale, bias)
        if native_output is not None:
            return native_output

    if input.dtype != torch.float16:
        raise TypeError(f"W4A16 activations must be torch.float16, got {input.dtype}")
    if bias is not None and bias.dtype != torch.float16:
        raise TypeError(f"W4A16 bias must be torch.float16, got {bias.dtype}")
    dequantized_weight = unpack_int4_weight(packed_weight, scale)
    # The fallback deliberately promotes both operands and the bias before
    # linear, so its accumulation is FP32 even though the result is FP16.
    output = F.linear(
        input.to(dtype=torch.float32),
        dequantized_weight,
        None if bias is None else bias.to(dtype=torch.float32),
    )
    return output.to(dtype=input.dtype)


def _w4a16_linear_meta(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    del scale, bias
    return torch.empty(
        (*input.shape[:-1], packed_weight.shape[0]),
        dtype=torch.float16,
        device="meta",
    )


def _ensure_w4_operator() -> None:
    global _W4_LIBRARY
    if _W4_LIBRARY is not None:
        return
    library = torch.library.Library("llmopt", "DEF")
    library.define(
        "w4a16_linear(Tensor input, Tensor packed_weight, Tensor scale, Tensor? bias) -> Tensor"
    )
    library.impl("w4a16_linear", _w4a16_linear_impl, "CPU")
    library.impl("w4a16_linear", _w4a16_linear_impl, "MPS")
    library.impl("w4a16_linear", _w4a16_linear_meta, "Meta")
    _W4_LIBRARY = library


class W4A16Linear(nn.Module):
    """An explicit group-64 W4A16 probe module."""

    def __init__(
        self,
        packed_weight: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        output_features, _packed_k, groups = _validate_packed_weight_shape(
            packed_weight
        )
        if packed_weight.dtype != torch.uint8:
            raise TypeError(
                "W4A16 weight storage must be torch.uint8, "
                f"got {packed_weight.dtype}"
            )
        if scale.dtype != torch.float16:
            raise TypeError(f"W4A16 scales must be torch.float16, got {scale.dtype}")
        if tuple(scale.shape) != (output_features, groups):
            raise ValueError(
                "W4A16Linear expects packed weight [out_features, in_features/2] "
                f"and scale [{output_features}, {groups}], got {tuple(scale.shape)}"
            )
        if bias is not None:
            if bias.dtype != torch.float16:
                raise TypeError(f"W4A16 bias must be torch.float16, got {bias.dtype}")
            if tuple(bias.shape) != (output_features,):
                raise ValueError(
                    f"W4A16Linear bias must have shape [{output_features}], "
                    f"got {tuple(bias.shape)}"
                )
        _ensure_w4_operator()
        self.register_buffer("packed_weight", packed_weight.detach().contiguous())
        self.register_buffer("scale", scale.detach().contiguous())
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "W4A16Linear":
        packed_weight, scale = quantize_weight(linear.weight)
        bias = None if linear.bias is None else linear.bias.detach().to(torch.float16)
        return cls(packed_weight, scale, bias)

    @property
    def in_features(self) -> int:
        return int(self.packed_weight.shape[1]) * 2

    @property
    def out_features(self) -> int:
        return int(self.packed_weight.shape[0])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.ops.llmopt.w4a16_linear.default(
            input, self.packed_weight, self.scale, self.bias
        )


def quantize_model_(model: nn.Module) -> dict[str, Any]:
    """Replace every linear child in place and return an audit summary."""

    converted: list[str] = []

    def visit(module: nn.Module, prefix: str) -> None:
        for name, child in list(module.named_children()):
            qualified_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, W4A16Linear):
                continue
            if isinstance(child, nn.Linear):
                setattr(module, name, W4A16Linear.from_linear(child))
                converted.append(qualified_name)
                continue
            visit(child, qualified_name)

    visit(model, "")
    return {
        "scheme": "w4a16-weight-only-group64",
        "activation_dtype": "float16",
        "scale_dtype": "float16",
        "group_size": GROUP_SIZE,
        "converted_linear_modules": len(converted),
        "converted_modules": converted,
    }


__all__ = [
    "GROUP_SIZE",
    "W4A16Linear",
    "pack_int4_weight",
    "quantize_model_",
    "quantize_weight",
    "unpack_int4_weight",
]
