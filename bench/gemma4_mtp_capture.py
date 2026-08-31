#!/usr/bin/env python3
"""Capture the pinned Gemma 4 12B target/MTP functional graph contract.

The capture is metadata-only: models and inputs live on the PyTorch ``meta``
device and no model weights are materialized.  Every persistent KV tensor is an
explicit graph input and output; no Transformers cache object crosses the
Model Program boundary.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from download_gemma12b_mtp import FILES, REPO_ID, REVISION

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_FILENAME = "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"
ASSISTANT_FILENAME = "mtp-gemma-4-12B-it.gguf"
TARGET_CONFIG_REPO = "google/gemma-4-12b-it"
TARGET_CONFIG_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
DEFAULT_OUTPUT = (
    REPO_ROOT / "bench/results/gemma4-mtp-functional-capture-2026-08-31.json"
)


@dataclass(frozen=True)
class LayerGeometry:
    index: int
    attention: str
    kv_heads: int
    head_dim: int

    def state_shape(self, *, batch: int, tokens: int) -> tuple[int, int, int, int]:
        return (batch, self.kv_heads, tokens, self.head_dim)


@dataclass(frozen=True)
class TensorBinding:
    captured_name: str
    gguf_name: str
    shape: tuple[int, ...]
    gguf_type: str

    def as_json(self) -> dict[str, Any]:
        return {
            "captured_name": self.captured_name,
            "gguf_name": self.gguf_name,
            "shape": list(self.shape),
            "gguf_type": self.gguf_type,
        }


_LAYER_SUFFIXES = {
    "layer_scalar": "layer_output_scale.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "post_attention_norm.weight",
    "pre_feedforward_layernorm.weight": "ffn_norm.weight",
    "post_feedforward_layernorm.weight": "post_ffw_norm.weight",
}


def captured_to_gguf(name: str, *, assistant: bool) -> str:
    """Map one unique Transformers parameter/buffer to its GGUF tensor."""

    roots = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
    }
    if assistant:
        roots.update(
            {
                "pre_projection.weight": "nextn.pre_projection.weight",
                "post_projection.weight": "nextn.post_projection.weight",
            }
        )
    if name in roots:
        return roots[name]
    match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name)
    if match is None or match.group(2) not in _LAYER_SUFFIXES:
        raise KeyError(f"no Gemma GGUF binding for captured tensor {name}")
    return f"blk.{match.group(1)}.{_LAYER_SUFFIXES[match.group(2)]}"


def target_layer_geometry(text_config: Any) -> tuple[LayerGeometry, ...]:
    layers = []
    for index, attention in enumerate(text_config.layer_types):
        is_global = attention == "full_attention"
        layers.append(
            LayerGeometry(
                index=index,
                attention=attention,
                kv_heads=(
                    text_config.num_global_key_value_heads
                    if is_global
                    else text_config.num_key_value_heads
                ),
                head_dim=(
                    text_config.global_head_dim if is_global else text_config.head_dim
                ),
            )
        )
    return tuple(layers)


def expected_target_geometry() -> tuple[LayerGeometry, ...]:
    layer_types = tuple(
        "full_attention" if (index + 1) % 6 == 0 else "sliding_attention"
        for index in range(48)
    )
    return tuple(
        LayerGeometry(
            index=index,
            attention=attention,
            kv_heads=1 if attention == "full_attention" else 8,
            head_dim=512 if attention == "full_attention" else 256,
        )
        for index, attention in enumerate(layer_types)
    )


def build_assistant_text_config(target_text_config: Any) -> Any:
    from transformers import Gemma4TextConfig

    return Gemma4TextConfig(
        vocab_size=target_text_config.vocab_size,
        hidden_size=1024,
        intermediate_size=8192,
        num_hidden_layers=4,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_global_key_value_heads=1,
        head_dim=256,
        global_head_dim=512,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=0,
        num_kv_shared_layers=4,
        sliding_window=target_text_config.sliding_window,
        max_position_embeddings=target_text_config.max_position_embeddings,
        rope_parameters=copy.deepcopy(target_text_config.rope_parameters),
        final_logit_softcapping=None,
        attention_k_eq_v=True,
        use_cache=False,
    )


class FunctionalCache:
    """A traceable cache facade whose entire state is supplied as tensors."""

    def __init__(self, states: Sequence[Any]):
        if len(states) % 2 != 0:
            raise ValueError("functional cache requires key/value pairs")
        self.keys = list(states[0::2])
        self.values = list(states[1::2])

    def update(
        self, key_states: Any, value_states: Any, layer_idx: int, *_: Any, **__: Any
    ) -> tuple[Any, Any]:
        import torch

        keys = torch.cat((self.keys[layer_idx], key_states), dim=-2)
        values = torch.cat((self.values[layer_idx], value_states), dim=-2)
        self.keys[layer_idx] = keys
        self.values[layer_idx] = values
        return keys, values

    def state(self) -> tuple[Any, ...]:
        return tuple(item for pair in zip(self.keys, self.values) for item in pair)


def _target_wrapper_type() -> type[Any]:
    import torch

    class FunctionalTarget(torch.nn.Module):
        def __init__(self, model: Any):
            super().__init__()
            self.model = model

        def forward(
            self,
            input_ids: Any,
            position_ids: Any,
            full_attention_mask: Any,
            sliding_attention_mask: Any,
            *cache_states: Any,
        ) -> tuple[Any, ...]:
            cache = FunctionalCache(cache_states)
            outputs = self.model.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask={
                    "full_attention": full_attention_mask,
                    "sliding_attention": sliding_attention_mask,
                },
                past_key_values=cache,
                use_cache=True,
                return_shared_kv_states=True,
            )
            hidden = outputs.last_hidden_state
            logits = self.model.lm_head(hidden)
            cap = self.model.config.final_logit_softcapping
            if cap is not None:
                logits = torch.tanh(logits / cap) * cap
            shared = outputs.shared_kv_states
            return (
                logits,
                hidden,
                shared["full_attention"][0],
                shared["full_attention"][1],
                shared["sliding_attention"][0],
                shared["sliding_attention"][1],
                *cache.state(),
            )

    return FunctionalTarget


def _assistant_wrapper_type() -> type[Any]:
    import torch

    class FunctionalAssistant(torch.nn.Module):
        def __init__(self, model: Any):
            super().__init__()
            self.model = model

        def forward(
            self,
            coupled_target_state: Any,
            position_ids: Any,
            full_attention_mask: Any,
            sliding_attention_mask: Any,
            full_key: Any,
            full_value: Any,
            sliding_key: Any,
            sliding_value: Any,
        ) -> tuple[Any, Any]:
            hidden = self.model.pre_projection(coupled_target_state)
            outputs = self.model.model(
                inputs_embeds=hidden,
                position_ids=position_ids,
                attention_mask={
                    "full_attention": full_attention_mask,
                    "sliding_attention": sliding_attention_mask,
                },
                shared_kv_states={
                    "full_attention": (full_key, full_value),
                    "sliding_attention": (sliding_key, sliding_value),
                },
                use_cache=False,
            )
            last_hidden = outputs.last_hidden_state
            return (
                self.model.lm_head(last_hidden),
                self.model.post_projection(last_hidden),
            )

    return FunctionalAssistant


def _tensor_spec(name: str, tensor: Any) -> dict[str, Any]:
    return {
        "name": name,
        "shape": [int(dimension) for dimension in tensor.shape],
        "dtype": str(tensor.dtype).removeprefix("torch."),
    }


def _graph_digest(manifest: dict[str, Any]) -> str:
    stable = [
        {
            "op": node["op"],
            "target": node["target"],
            "shape": node["shape"],
            "dtype": node["dtype"],
            "binding": node["binding"]["kind"],
        }
        for node in manifest["nodes"]
    ]
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def capture_graph(module: Any, arguments: tuple[Any, ...], names: dict[str, list[str]]) -> dict[str, Any]:
    import torch
    from llmopt_backend import capture_from_fx

    captured: dict[str, Any] = {}

    def backend(graph_module: Any, example_inputs: Sequence[Any]) -> Any:
        captured["fx"] = capture_from_fx(graph_module, example_inputs)
        return graph_module.forward

    torch._dynamo.reset()
    outputs = torch.compile(
        module, backend=backend, fullgraph=True, dynamic=False
    )(*arguments)
    fx = captured["fx"]
    manifest = fx.manifest
    runtime_nodes = [
        node
        for node in manifest["nodes"]
        if node["op"] == "placeholder" and node["binding"]["kind"] == "runtime"
    ]
    if len(runtime_nodes) != len(names["inputs"]):
        raise ValueError(
            f"capture exposed {len(runtime_nodes)} runtime inputs; expected {len(names['inputs'])}"
        )
    if len(outputs) != len(names["outputs"]):
        raise ValueError(
            f"capture returned {len(outputs)} tensors; expected {len(names['outputs'])}"
        )
    return {
        "node_count": len(manifest["nodes"]),
        "graph_sha256": _graph_digest(manifest),
        "static_tensor_count": len(fx.tensors),
        "inputs": [
            _tensor_spec(name, tensor)
            for name, tensor in zip(names["inputs"], arguments)
        ],
        "outputs": [
            _tensor_spec(name, tensor)
            for name, tensor in zip(names["outputs"], outputs)
        ],
    }


def _target_names(geometry: Sequence[LayerGeometry]) -> dict[str, list[str]]:
    inputs = [
        "input_ids",
        "position_ids",
        "full_attention_mask",
        "sliding_attention_mask",
    ]
    outputs = [
        "logits",
        "target_hidden_state",
        "shared.full_attention.key",
        "shared.full_attention.value",
        "shared.sliding_attention.key",
        "shared.sliding_attention.value",
    ]
    for layer in geometry:
        inputs.extend((f"cache.layer.{layer.index}.key", f"cache.layer.{layer.index}.value"))
        outputs.extend((f"cache.layer.{layer.index}.key", f"cache.layer.{layer.index}.value"))
    return {"inputs": inputs, "outputs": outputs}


def _assistant_names() -> dict[str, list[str]]:
    return {
        "inputs": [
            "coupled_target_embedding_and_hidden",
            "position_ids",
            "full_attention_mask",
            "sliding_attention_mask",
            "shared.full_attention.key",
            "shared.full_attention.value",
            "shared.sliding_attention.key",
            "shared.sliding_attention.value",
        ],
        "outputs": ["logits", "projected_target_hidden_state"],
    }


def _target_arguments(
    geometry: Sequence[LayerGeometry], *, query_tokens: int, past_tokens: int
) -> tuple[Any, ...]:
    import torch

    total_tokens = query_tokens + past_tokens
    values: list[Any] = [
        torch.zeros((1, query_tokens), dtype=torch.long, device="meta"),
        torch.arange(past_tokens, total_tokens, dtype=torch.long, device="meta").unsqueeze(0),
        torch.zeros((1, 1, query_tokens, total_tokens), dtype=torch.float16, device="meta"),
        torch.zeros((1, 1, query_tokens, total_tokens), dtype=torch.float16, device="meta"),
    ]
    for layer in geometry:
        shape = layer.state_shape(batch=1, tokens=past_tokens)
        values.extend(
            (
                torch.zeros(shape, dtype=torch.float16, device="meta"),
                torch.zeros(shape, dtype=torch.float16, device="meta"),
            )
        )
    return tuple(values)


def _assistant_arguments(*, shared_tokens: int) -> tuple[Any, ...]:
    import torch

    return (
        torch.zeros((1, 1, 7680), dtype=torch.float16, device="meta"),
        torch.zeros((1, 1), dtype=torch.long, device="meta"),
        torch.zeros((1, 1, 1, shared_tokens), dtype=torch.float16, device="meta"),
        torch.zeros((1, 1, 1, shared_tokens), dtype=torch.float16, device="meta"),
        torch.zeros((1, 1, shared_tokens, 512), dtype=torch.float16, device="meta"),
        torch.zeros((1, 1, shared_tokens, 512), dtype=torch.float16, device="meta"),
        torch.zeros((1, 8, shared_tokens, 256), dtype=torch.float16, device="meta"),
        torch.zeros((1, 8, shared_tokens, 256), dtype=torch.float16, device="meta"),
    )


def _model_binding_tensors(model: Any) -> dict[str, Any]:
    tensors = dict(model.named_parameters())
    tensors.update(
        (name, tensor)
        for name, tensor in model.named_buffers()
        if name.endswith(".layer_scalar")
    )
    return tensors


def validate_gguf_bindings(model: Any, gguf_path: Path, *, assistant: bool) -> list[TensorBinding]:
    from gguf import GGUFReader

    reader = GGUFReader(gguf_path, "r")
    inventory = {tensor.name: tensor for tensor in reader.tensors}
    expected_gguf = set(inventory) - {"rope_freqs.weight"}
    bindings = []
    for captured_name, tensor in sorted(_model_binding_tensors(model).items()):
        gguf_name = captured_to_gguf(captured_name, assistant=assistant)
        if gguf_name not in inventory:
            raise KeyError(f"{gguf_path.name} has no tensor {gguf_name}")
        gguf_tensor = inventory[gguf_name]
        gguf_shape = tuple(int(value) for value in reversed(gguf_tensor.shape))
        captured_shape = tuple(int(value) for value in tensor.shape)
        if gguf_shape != captured_shape:
            raise ValueError(
                f"{captured_name} shape {captured_shape} does not match {gguf_name} {gguf_shape}"
            )
        bindings.append(
            TensorBinding(
                captured_name=captured_name,
                gguf_name=gguf_name,
                shape=captured_shape,
                gguf_type=str(gguf_tensor.tensor_type),
            )
        )
    actual_gguf = {binding.gguf_name for binding in bindings}
    if actual_gguf != expected_gguf:
        missing = sorted(expected_gguf - actual_gguf)
        extra = sorted(actual_gguf - expected_gguf)
        raise ValueError(f"GGUF binding coverage mismatch: missing={missing}, extra={extra}")
    return bindings


def binding_digest(bindings: Iterable[TensorBinding]) -> str:
    payload = json.dumps(
        [binding.as_json() for binding in bindings],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _resolve_file(filename: str, *, local_only: bool) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            revision=REVISION,
            local_files_only=local_only,
        )
    )


def _load_target_config(*, local_only: bool) -> tuple[Any, Path]:
    from huggingface_hub import hf_hub_download
    from transformers import Gemma4Config

    config_path = Path(
        hf_hub_download(
            repo_id=TARGET_CONFIG_REPO,
            filename="config.json",
            revision=TARGET_CONFIG_REVISION,
            local_files_only=local_only,
        )
    )
    config = Gemma4Config.from_pretrained(config_path.parent, local_files_only=True)
    return config.text_config, config_path


def build_receipt(*, local_only: bool) -> dict[str, Any]:
    import torch
    import transformers
    from transformers import (
        Gemma4AssistantConfig,
        Gemma4AssistantForCausalLM,
        Gemma4ForCausalLM,
    )

    target_path = _resolve_file(TARGET_FILENAME, local_only=local_only)
    assistant_path = _resolve_file(ASSISTANT_FILENAME, local_only=local_only)
    target_config, config_path = _load_target_config(local_only=local_only)
    target_config._attn_implementation = "eager"
    geometry = target_layer_geometry(target_config)
    if geometry != expected_target_geometry():
        raise ValueError("pinned Gemma target geometry differs from the audited 40 sliding/8 global layout")

    assistant_text_config = build_assistant_text_config(target_config)
    assistant_text_config._attn_implementation = "eager"
    assistant_config = Gemma4AssistantConfig(
        text_config=assistant_text_config,
        backbone_hidden_size=target_config.hidden_size,
        use_ordered_embeddings=False,
    )

    with torch.device("meta"):
        target_model = Gemma4ForCausalLM(target_config).to(dtype=torch.float16)
        assistant_model = Gemma4AssistantForCausalLM(assistant_config).to(
            dtype=torch.float16
        )
    target_model.eval()
    assistant_model.eval()

    target_bindings = validate_gguf_bindings(
        target_model, target_path, assistant=False
    )
    assistant_bindings = validate_gguf_bindings(
        assistant_model, assistant_path, assistant=True
    )

    target_wrapper = _target_wrapper_type()(target_model)
    target_names = _target_names(geometry)
    prefill = capture_graph(
        target_wrapper,
        _target_arguments(geometry, query_tokens=2, past_tokens=0),
        target_names,
    )
    decode = capture_graph(
        target_wrapper,
        _target_arguments(geometry, query_tokens=1, past_tokens=2),
        target_names,
    )
    assistant = capture_graph(
        _assistant_wrapper_type()(assistant_model),
        _assistant_arguments(shared_tokens=3),
        _assistant_names(),
    )

    return {
        "schema_version": 1,
        "kind": "gemma4-target-coupled-mtp-functional-capture",
        "generated_at": "2026-08-31",
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": "meta",
            "activation_dtype": "float16",
        },
        "sources": {
            "target_config": {
                "repo": TARGET_CONFIG_REPO,
                "revision": TARGET_CONFIG_REVISION,
                "path": str(config_path),
            },
            "target_gguf": {
                "repo": REPO_ID,
                "revision": REVISION,
                "filename": TARGET_FILENAME,
                "path": str(target_path),
                **FILES[TARGET_FILENAME],
            },
            "assistant_gguf": {
                "repo": REPO_ID,
                "revision": REVISION,
                "filename": ASSISTANT_FILENAME,
                "path": str(assistant_path),
                **FILES[ASSISTANT_FILENAME],
            },
        },
        "contract": {
            "target_hidden_size": target_config.hidden_size,
            "assistant_hidden_size": assistant_text_config.hidden_size,
            "coupled_input_size": 2 * target_config.hidden_size,
            "max_draft_tokens": 4,
            "target_layers": [
                {
                    "index": layer.index,
                    "attention": layer.attention,
                    "kv_heads": layer.kv_heads,
                    "head_dim": layer.head_dim,
                }
                for layer in geometry
            ],
            "assistant_layers": [
                {
                    "index": layer.index,
                    "attention": layer.attention,
                    "kv_heads": layer.kv_heads,
                    "head_dim": layer.head_dim,
                    "kv_source": "target_shared_state",
                }
                for layer in target_layer_geometry(assistant_text_config)
            ],
            "acceptance": "K draft tokens require K+1 target predictions",
        },
        "graphs": {
            "target_prefill": prefill,
            "target_decode": decode,
            "assistant_step": assistant,
        },
        "tensor_bindings": {
            "target": {
                "count": len(target_bindings),
                "sha256": binding_digest(target_bindings),
                "bindings": [binding.as_json() for binding in target_bindings],
                "derived_capture_constants": [
                    "model.embed_tokens.embed_scale",
                    "model.rotary_emb.full_attention_inv_freq",
                    "model.rotary_emb.sliding_attention_inv_freq",
                ],
                "gguf_rope_tensor_policy": "rope_freqs.weight is replaced by config-derived full/sliding inverse frequencies",
            },
            "assistant": {
                "count": len(assistant_bindings),
                "sha256": binding_digest(assistant_bindings),
                "bindings": [binding.as_json() for binding in assistant_bindings],
                "derived_capture_constants": [
                    "model.embed_tokens.embed_scale",
                    "model.rotary_emb.full_attention_inv_freq",
                    "model.rotary_emb.sliding_attention_inv_freq",
                ],
                "gguf_rope_tensor_policy": "rope_freqs.weight is replaced by config-derived full/sliding inverse frequencies",
            },
        },
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture and verify the functional Gemma 4 target/MTP graph contract"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="require the pinned config and GGUFs to exist in the local Hugging Face cache",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    receipt = build_receipt(local_only=args.verify)
    write_json_atomic(args.output, receipt)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "target_prefill_nodes": receipt["graphs"]["target_prefill"]["node_count"],
                "target_decode_nodes": receipt["graphs"]["target_decode"]["node_count"],
                "assistant_nodes": receipt["graphs"]["assistant_step"]["node_count"],
                "target_bindings": receipt["tensor_bindings"]["target"]["count"],
                "assistant_bindings": receipt["tensor_bindings"]["assistant"]["count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
