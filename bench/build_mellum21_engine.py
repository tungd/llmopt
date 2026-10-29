#!/usr/bin/env python3
"""Build Mellum 2.1 from a typed archive, including canonical EXL3 weights."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoConfig, MellumForCausalLM
from transformers.models.mellum.modeling_mellum import MellumRotaryEmbedding

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "python"), str(REPO_ROOT / "bench")]

from llmopt_backend import CapturedFx, ExternalTensor, bind_external_tensors, capture_from_fx
from llmopt_backend.fx_graph import write_graph

from llmopt_backend.tensor_archive import read_inventory

# 1. Custom MoE operation registration
@torch.library.custom_op("llmopt::moe_swiglu_ffn", mutates_args=())
def moe_swiglu_ffn(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    gate_exps: torch.Tensor,
    up_exps: torch.Tensor,
    down_exps: torch.Tensor,
    top_k: int = 8,
    norm_topk_prob: bool = True,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)

@moe_swiglu_ffn.register_fake
def _(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    gate_exps: torch.Tensor,
    up_exps: torch.Tensor,
    down_exps: torch.Tensor,
    top_k: int = 8,
    norm_topk_prob: bool = True,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)

class MellumLlmoptMoE(torch.nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate_inp = torch.nn.Parameter(torch.empty(config.num_experts, config.hidden_size))
        self.gate_exps = torch.nn.Parameter(torch.empty(config.num_experts, config.moe_intermediate_size, config.hidden_size))
        self.up_exps = torch.nn.Parameter(torch.empty(config.num_experts, config.moe_intermediate_size, config.hidden_size))
        self.down_exps = torch.nn.Parameter(torch.empty(config.num_experts, config.hidden_size, config.moe_intermediate_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        b, s, d = hidden_states.shape
        hs_flat = hidden_states.view(-1, d)
        router_logits = F.linear(hs_flat, self.gate_inp)
        out = torch.ops.llmopt.moe_swiglu_ffn(
            hs_flat,
            router_logits,
            self.gate_exps,
            self.up_exps,
            self.down_exps,
            top_k=self.top_k,
            norm_topk_prob=self.norm_topk_prob,
        )
        return out.view(b, s, d)

# 2. Traceable functional caches
class EmptyFunctionalCache:
    def __init__(self, layers: int):
        self.keys = [None] * layers
        self.values = [None] * layers

    def update(self, key_states: Any, value_states: Any, layer_idx: int, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        self.keys[layer_idx] = key_states
        self.values[layer_idx] = value_states
        return key_states, value_states

    def state(self) -> tuple[Any, ...]:
        if any(item is None for item in self.keys + self.values):
            raise RuntimeError("prefill did not populate every cache layer")
        return tuple(item for pair in zip(self.keys, self.values) for item in pair)

class FunctionalCache:
    def __init__(self, states: Sequence[Any]):
        self.keys = list(states[0::2])
        self.values = list(states[1::2])

    def update(self, key_states: Any, value_states: Any, layer_idx: int, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        keys = torch.cat((self.keys[layer_idx], key_states), dim=-2)
        values = torch.cat((self.values[layer_idx], value_states), dim=-2)
        self.keys[layer_idx] = keys
        self.values[layer_idx] = values
        return keys, values

    def state(self) -> tuple[Any, ...]:
        return tuple(item for pair in zip(self.keys, self.values) for item in pair)

# 3. Functional target wrappers
class FunctionalMellumPrefill(torch.nn.Module):
    def __init__(self, model: Any):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        full_attention_mask: torch.Tensor,
        sliding_attention_mask: torch.Tensor,
    ) -> tuple[Any, ...]:
        cache = EmptyFunctionalCache(self.model.config.num_hidden_layers)
        outputs = self.model.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask={
                "full_attention": full_attention_mask,
                "sliding_attention": sliding_attention_mask,
            },
            past_key_values=cache,
            use_cache=True,
        )
        logits = self.model.lm_head(outputs.last_hidden_state)
        return (logits, *cache.state())

class FunctionalMellumDecode(torch.nn.Module):
    def __init__(self, model: Any):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        full_attention_mask: torch.Tensor,
        sliding_attention_mask: torch.Tensor,
        *cache_states: torch.Tensor,
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
        )
        logits = self.model.lm_head(outputs.last_hidden_state)
        return (logits, *cache.state())

def capture(module: Any, arguments: tuple[Any, ...]) -> CapturedFx:
    captured: dict[str, Any] = {}

    def backend(graph_module: Any, example_inputs: Sequence[Any]) -> Any:
        captured["graph"] = capture_from_fx(graph_module, example_inputs)
        return graph_module.forward

    torch._dynamo.reset()
    torch.compile(module, backend=backend, fullgraph=True, dynamic=False)(*arguments)
    return captured["graph"]

def captured_to_gguf(name: str) -> str:
    cleaned = name
    while cleaned.startswith(("model.", "self.", "m.")):
        for prefix in ("model.", "self.", "m."):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
    if cleaned == "embed_tokens.weight":
        return "token_embd.weight"
    if cleaned == "norm.weight":
        return "output_norm.weight"
    if cleaned in ("lm_head.weight", "output.weight"):
        return "output.weight"
    m = re.match(r"layers\.(\d+)\.(.+)", cleaned)
    if not m:
        raise KeyError(f"unrecognized tensor name: {name} (cleaned: {cleaned})")
    layer, sub = m.group(1), m.group(2)
    mapping = {
        "input_layernorm.weight": "attn_norm.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "mlp.gate_inp": "ffn_gate_inp.weight",
        "mlp.gate_inp.weight": "ffn_gate_inp.weight",
        "mlp.gate_exps": "ffn_gate_exps.weight",
        "mlp.gate_exps.weight": "ffn_gate_exps.weight",
        "mlp.up_exps": "ffn_up_exps.weight",
        "mlp.up_exps.weight": "ffn_up_exps.weight",
        "mlp.down_exps": "ffn_down_exps.weight",
        "mlp.down_exps.weight": "ffn_down_exps.weight",
    }
    if sub not in mapping:
        raise KeyError(f"unrecognized layer tensor: {sub} in {name}")
    return f"blk.{layer}.{mapping[sub]}"

def external_binding(key: str, inventory: dict[str, tuple[str, tuple[int, ...]]]) -> ExternalTensor:
    if key.endswith("sliding_attention_inv_freq"):
        gguf_name = "z_mellum_sliding_attention_inv_freq"
    elif key.endswith("full_attention_inv_freq"):
        gguf_name = "z_mellum_full_attention_inv_freq"
    else:
        gguf_name = captured_to_gguf(key)
    if gguf_name not in inventory:
        raise KeyError(f"tensor {gguf_name} (from {key}) not in weight archive inventory")
    dtype_name, shape = inventory[gguf_name]
    return ExternalTensor(gguf_name, dtype_name, shape)

def semantic_runtime_name(name: str) -> str:
    cleaned = name.strip("l_").rstrip("_")
    if "input_ids" in cleaned:
        return "input_ids"
    if "position_ids" in cleaned:
        return "position_ids"
    if "full_attention_mask" in cleaned or "full_mask" in cleaned:
        return "full_attention_mask"
    if "sliding_attention_mask" in cleaned or "sliding_mask" in cleaned:
        return "sliding_attention_mask"
    if "cache_states" in name:
        m = re.search(r"cache_states_(\d+)", name)
        if m:
            index = int(m.group(1))
            return f"cache.layer.{index // 2}.{'key' if index % 2 == 0 else 'value'}"
    raise KeyError(f"unrecognized runtime input: {name}")

def replace_names(value: Any, names: dict[str, str]) -> Any:
    if isinstance(value, str):
        return names.get(value, value)
    if isinstance(value, list):
        return [replace_names(item, names) for item in value]
    if isinstance(value, dict):
        return {key: replace_names(item, names) for key, item in value.items()}
    return value

def target_graph(
    captured: CapturedFx,
    inventory: dict[str, tuple[str, tuple[int, ...]]],
    num_layers: int = 28,
    retain_diagnostics: bool = False,
) -> dict[str, Any]:
    bindings = {key: external_binding(key, inventory) for key in captured.tensors}
    rebound = bind_external_tensors(captured, bindings)
    manifest = rebound.manifest

    runtime_names = {
        node["name"]: semantic_runtime_name(node["name"])
        for node in manifest["nodes"]
        if node.get("binding", {}).get("kind") == "runtime"
    }
    raw_outputs = list(manifest["outputs"])
    expected_output_count = 1 + 2 * num_layers
    if len(raw_outputs) != expected_output_count:
        raise ValueError(f"expected {expected_output_count} outputs, got {len(raw_outputs)}")

    output_names = {raw_outputs[0]: "logits"}
    cache_outputs = []
    for layer in range(num_layers):
        k_out = f"cache.layer.{layer}.key"
        v_out = f"cache.layer.{layer}.value"
        output_names[raw_outputs[1 + 2 * layer]] = k_out
        output_names[raw_outputs[2 + 2 * layer]] = v_out
        cache_outputs.extend([k_out, v_out])

    manifest = replace_names(manifest, {**runtime_names, **output_names})
    retained = ["logits", *cache_outputs]
    if retain_diagnostics:
        head = next(node for node in manifest["nodes"] if node["name"] == "logits")
        hidden_name = head["inputs"][0]
        manifest = replace_names(manifest, {hidden_name: "diagnostic.hidden"})
        retained.append("diagnostic.hidden")
    manifest["outputs"] = retained
    output_node = next(node for node in manifest["nodes"] if node["op"] == "output")
    output_node["inputs"] = retained
    output_node["arguments"] = {
        "args": [{"kind": "tuple", "items": [{"kind": "node", "name": name} for name in retained]}],
        "kwargs": [],
    }
    return manifest

def preserve_reference_constant_precision(model: Any) -> None:
    rotary = model.model.rotary_emb
    for layer_type in ("sliding_attention", "full_attention"):
        for suffix in ("inv_freq", "original_inv_freq"):
            name = f"{layer_type}_{suffix}"
            if hasattr(rotary, name):
                setattr(rotary, name, getattr(rotary, name).float())

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-config", type=Path, required=True, help="Path to hf_config directory")
    parser.add_argument("--weights", type=Path, required=True, help="Path to weights.llmopt archive")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Path to GGUF or tokenizer file")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for compiled engine")
    parser.add_argument("--retain-diagnostics", action="store_true", help="Retain final normalized hidden values for numerical diagnosis")
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_dir = output / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build_mellum21_engine] Loading config from {args.hf_config}...")
    config = AutoConfig.from_pretrained(args.hf_config)
    config._attn_implementation = "sdpa"
    num_layers = config.num_hidden_layers

    print(f"[build_mellum21_engine] Loading weight inventory from {args.weights}...")
    inventory = read_inventory(args.weights)
    print(f"[build_mellum21_engine] Loaded {len(inventory)} tensors from archive.")

    print("[build_mellum21_engine] Initializing Mellum meta model with llmopt MoE blocks...")
    with torch.device("meta"):
        model = MellumForCausalLM(config).to(dtype=torch.float16)
        preserve_reference_constant_precision(model)
        for layer in model.model.layers:
            layer.mlp = MellumLlmoptMoE(config).to(dtype=torch.float16)
    model.eval()

    print("[build_mellum21_engine] Tracing prefill graph (tokens=2)...")
    prefill_wrapper = FunctionalMellumPrefill(model)
    prefill_args = (
        torch.zeros(1, 2, dtype=torch.long, device="meta"),
        torch.zeros(1, 2, dtype=torch.long, device="meta"),
        torch.zeros(1, 1, 2, 2, dtype=torch.float16, device="meta"),
        torch.zeros(1, 1, 2, 2, dtype=torch.float16, device="meta"),
    )
    prefill_captured = capture(prefill_wrapper, prefill_args)
    prefill_manifest = target_graph(prefill_captured, inventory, num_layers=num_layers, retain_diagnostics=args.retain_diagnostics)
    prefill_graph_path = source_dir / "prefill.graph.llmopt"
    write_graph(prefill_manifest, prefill_graph_path)
    print(f"[build_mellum21_engine] Prefill graph written to {prefill_graph_path}")

    print("[build_mellum21_engine] Tracing decode graph (query=1, past=2)...")
    decode_wrapper = FunctionalMellumDecode(model)
    cache_inputs = []
    for _ in range(num_layers):
        cache_inputs.append(torch.zeros(1, config.num_key_value_heads, 2, config.head_dim, dtype=torch.float16, device="meta"))
        cache_inputs.append(torch.zeros(1, config.num_key_value_heads, 2, config.head_dim, dtype=torch.float16, device="meta"))

    decode_args = (
        torch.zeros(1, 1, dtype=torch.long, device="meta"),
        torch.zeros(1, 1, dtype=torch.long, device="meta"),
        torch.zeros(1, 1, 1, 3, dtype=torch.float16, device="meta"),
        torch.zeros(1, 1, 1, 3, dtype=torch.float16, device="meta"),
        *cache_inputs,
    )
    decode_captured = capture(decode_wrapper, decode_args)
    decode_manifest = target_graph(decode_captured, inventory, num_layers=num_layers, retain_diagnostics=args.retain_diagnostics)
    decode_graph_path = source_dir / "decode.graph.llmopt"
    write_graph(decode_manifest, decode_graph_path)
    print(f"[build_mellum21_engine] Decode graph written to {decode_graph_path}")

    print("[build_mellum21_engine] Compiling unified engine with llmopt-pipeline...")
    cmd = [
        str(REPO_ROOT / "_build/bin/llmopt-pipeline"),
        "--weights", str(args.weights.resolve()),
        "--tokenizer", str(args.tokenizer.resolve()),
        "--prefill", str(prefill_graph_path.resolve()),
        "--decode", str(decode_graph_path.resolve()),
        "--output", str(output),
        "--model", "JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF",
        "--architecture", "mellum",
        "--family", "mellum",
        "--vocab-size", str(config.vocab_size),
        "--max-positions", str(config.max_position_embeddings),
        "--chat-format", "chatml",
        "--bos-token-id", str(config.bos_token_id),
        "--message-start-token-id", str(config.bos_token_id),
        "--message-end-token-id", str(config.eos_token_id),
        "--input-ids", "input_ids",
        "--position-ids-input", "position_ids",
        "--full-attention-mask-input", "full_attention_mask",
        "--sliding-attention-mask-input", "sliding_attention_mask",
        "--sliding-window", str(config.sliding_window),
        "--no-greedy",
    ]
    for layer in range(num_layers):
        cmd.extend([
            "--attention-state",
            f"cache.layer.{layer}.key:cache.layer.{layer}.value:cache.layer.{layer}.key:cache.layer.{layer}.value",
        ])

    print("[build_mellum21_engine] Executing pipeline compilation:")
    print(" ".join(cmd[:12]), "...")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    print(f"\n[build_mellum21_engine] Engine successfully compiled at: {output}")

if __name__ == "__main__":
    main()
