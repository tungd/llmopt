#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def main():
    output_dir = REPO_ROOT / "_artifacts/gemma4-12b-qat-uoc-g4p12-2026-09-14"
    source_dir = REPO_ROOT / "_artifacts/gemma4-12b-qat-primitive-schedule-2026-09-03/source"
    weights = source_dir / "target.weights.llmopt"
    tokenizer = REPO_ROOT / "_artifacts/w4-engine-clean/tokenizer.llmopt"
    prefill = source_dir / "prefill.graph.llmopt"
    decode = source_dir / "decode.graph.llmopt"
    target_profile = REPO_ROOT / "_artifacts/gemma4-12b-qat-round7-2026-09-05/target.json"

    assert weights.exists(), f"Missing weights: {weights}"
    assert tokenizer.exists(), f"Missing tokenizer: {tokenizer}"
    assert prefill.exists(), f"Missing prefill: {prefill}"
    assert decode.exists(), f"Missing decode: {decode}"
    assert target_profile.exists(), f"Missing target profile: {target_profile}"

    cmd = [
        str(REPO_ROOT / "_build/bin/llmopt-pipeline"),
        "--weights", str(weights),
        "--tokenizer", str(tokenizer),
        "--prefill", str(prefill),
        "--decode", str(decode),
        "--target-profile", str(target_profile),
        "--output", str(output_dir),
        "--model", "unsloth/gemma-4-12B-it-qat-GGUF",
        "--architecture", "gemma4",
        "--family", "gemma4",
        "--vocab-size", "262144",
        "--max-positions", "262144",
        "--chat-format", "chatml",
        "--bos-token-id", "2",
        "--message-start-token-id", "2",
        "--message-end-token-id", "1",
        "--input-ids", "input_ids",
        "--position-ids-input", "position_ids",
        "--full-attention-mask-input", "full_attention_mask",
        "--sliding-attention-mask-input", "sliding_attention_mask",
        "--sliding-window", "1024",
        "--suppress-token-id", "258882",
        "--suppress-token-id", "258883",
    ]

    for layer in range(48):
        cmd.extend([
            "--attention-state",
            f"cache.layer.{layer}.key:cache.layer.{layer}.value:cache.layer.{layer}.key:cache.layer.{layer}.value"
        ])

    print("Executing pipeline command...")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    print(f"Pipeline finished successfully. Output: {output_dir}")

if __name__ == "__main__":
    main()
