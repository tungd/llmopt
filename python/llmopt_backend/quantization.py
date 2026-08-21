"""Small, explicit Q8 weight-only runtime boundary for the MPS target.

The representation is deliberately boring: each output row owns one
float16/float32 scale and the corresponding weight row is stored as signed
int8.  The generated MPS runtime consumes the packed weights directly when
the graph has an active Q8 metallib; the dequantizing PyTorch operator remains
the explicit fallback for unsupported shapes, dtypes, or unavailable builds.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


_Q8_LIBRARY: Any | None = None


def _q8_linear_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    from . import metal_runtime

    native_output = metal_runtime.dispatch_q8_linear(input, weight, scale, bias)
    if native_output is not None:
        return native_output
    dequantized_weight = weight.to(dtype=input.dtype) * scale.reshape(-1, 1).to(
        dtype=input.dtype
    )
    return F.linear(input, dequantized_weight, bias)


def _q8_linear_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    del scale, bias
    return torch.empty(
        (*input.shape[:-1], weight.shape[0]),
        dtype=input.dtype,
        device="meta",
    )


def _ensure_q8_operator() -> None:
    global _Q8_LIBRARY
    if _Q8_LIBRARY is not None:
        return
    library = torch.library.Library("llmopt", "DEF")
    library.define(
        "q8_linear(Tensor input, Tensor weight, Tensor scale, Tensor? bias) -> Tensor"
    )
    library.impl("q8_linear", _q8_linear_impl, "CPU")
    library.impl("q8_linear", _q8_linear_impl, "MPS")
    library.impl("q8_linear", _q8_linear_meta, "Meta")
    _Q8_LIBRARY = library


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetrically quantize a 2-D weight per output channel."""

    if weight.ndim != 2:
        raise ValueError(f"Q8 weight must be 2-D, got shape {tuple(weight.shape)}")
    source = weight.detach().float()
    scale = source.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).eps) / 127.0
    quantized = torch.round(source / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    scale_dtype = torch.float16 if weight.dtype in (torch.float16, torch.bfloat16) else torch.float32
    return quantized, scale.to(dtype=scale_dtype)


class Q8Linear(nn.Module):
    """A weight-only Q8 linear module with an explicit custom-op boundary."""

    def __init__(
        self,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        if qweight.dtype != torch.int8:
            raise TypeError(f"Q8 weight storage must be torch.int8, got {qweight.dtype}")
        if qweight.ndim != 2 or scale.shape != (qweight.shape[0],):
            raise ValueError(
                "Q8Linear expects weight [out_features, in_features] and scale [out_features]"
            )
        _ensure_q8_operator()
        self.register_buffer("qweight", qweight.detach().contiguous())
        self.register_buffer("scale", scale.detach().contiguous())
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "Q8Linear":
        qweight, scale = quantize_weight(linear.weight)
        bias = None if linear.bias is None else linear.bias.detach()
        return cls(qweight, scale, bias)

    @property
    def in_features(self) -> int:
        return int(self.qweight.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.qweight.shape[0])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.ops.llmopt.q8_linear.default(
            input, self.qweight, self.scale, self.bias
        )


def quantize_model_(
    model: nn.Module,
    *,
    skip_suffixes: Iterable[str] = ("lm_head",),
) -> dict[str, Any]:
    """Replace eligible linear children in place and return an audit summary."""

    suffixes = tuple(skip_suffixes)
    converted: list[str] = []
    skipped: list[str] = []

    def visit(module: nn.Module, prefix: str) -> None:
        for name, child in list(module.named_children()):
            qualified_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, Q8Linear):
                continue
            if isinstance(child, nn.Linear):
                if qualified_name.endswith(suffixes):
                    skipped.append(qualified_name)
                else:
                    setattr(module, name, Q8Linear.from_linear(child))
                    converted.append(qualified_name)
                continue
            visit(child, qualified_name)

    visit(model, "")
    return {
        "scheme": "q8-weight-only-per-output-channel",
        "converted_linear_modules": len(converted),
        "converted_modules": converted,
        "skipped_modules": skipped,
    }


__all__ = ["Q8Linear", "quantize_model_", "quantize_weight"]
