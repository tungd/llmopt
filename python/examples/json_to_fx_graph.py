#!/usr/bin/env python3
"""Convert a diagnostic FX JSON manifest into the binary compiler transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmopt_backend.fx_graph import write_graph


def upgrade_v1(manifest: dict) -> dict:
    if manifest.get("version", 1) != 1:
        return manifest
    nodes = []
    for node in manifest["nodes"]:
        op = str(node["op"])
        binding = node.get("binding")
        if binding is None:
            binding = {
                "kind": "runtime" if op in ("placeholder", "get_attr") else "computed"
            }
        inputs = list(node.get("inputs", []))
        nodes.append(
            {
                **node,
                "binding": binding,
                "arguments": {
                    "args": [{"kind": "node", "name": name} for name in inputs],
                    "kwargs": [],
                },
            }
        )
    return {**manifest, "version": 2, "nodes": nodes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = upgrade_v1(json.loads(args.input.read_text(encoding="utf-8")))
    write_graph(manifest, args.output)


if __name__ == "__main__":
    main()
