"""Transpile an XGBoost JSON tree ensemble into standalone OCaml.

The generated module only uses ``Float.is_nan``, comparisons, arithmetic, and
function calls.  It does not import Python, XGBoost, ctypes, or a runtime model
library, so it can be linked into the OCaml planner as ordinary source.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


MAX_OCAML_SOURCE_BYTES = 50 * 1024
MAX_TREE_DEPTH = 128


@dataclass(frozen=True)
class Leaf:
    value: float

    def evaluate(self, features: Sequence[float]) -> float:
        del features
        return self.value


@dataclass(frozen=True)
class Split:
    feature_index: int
    threshold: float
    yes: "Tree"
    no: "Tree"
    missing: "Tree"

    def evaluate(self, features: Sequence[float]) -> float:
        value = features[self.feature_index]
        if math.isnan(value):
            return self.missing.evaluate(features)
        if value < self.threshold:
            return self.yes.evaluate(features)
        return self.no.evaluate(features)


Tree = Leaf | Split


@dataclass(frozen=True)
class ModelBundle:
    feature_names: tuple[str, ...]
    base_score: float
    learning_rate: float
    trees: tuple[Tree, ...]


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"expected one numeric value, got {value!r}")
        return _as_float(value[0])
    if isinstance(value, str):
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value)
        if match is not None:
            return float(match.group(0))
    raise ValueError(f"expected numeric value, got {value!r}")


def _feature_index(split: object, feature_names: Sequence[str]) -> int:
    if isinstance(split, int):
        return split
    if not isinstance(split, str):
        raise ValueError(f"unsupported XGBoost split name: {split!r}")
    match = re.fullmatch(r"f(\d+)", split)
    if match is not None:
        return int(match.group(1))
    try:
        return feature_names.index(split)
    except ValueError as exc:
        raise ValueError(f"split {split!r} is absent from feature names") from exc


def _tree_from_dump(
    node: Mapping[str, Any],
    feature_names: Sequence[str],
    *,
    depth: int = 0,
) -> Tree:
    if depth > MAX_TREE_DEPTH:
        raise ValueError(f"XGBoost tree depth exceeds {MAX_TREE_DEPTH}")
    if "leaf" in node:
        return Leaf(_as_float(node["leaf"]))

    raw_children = node.get("children")
    if not isinstance(raw_children, list) or len(raw_children) < 2:
        raise ValueError(f"split node has fewer than two children: {node!r}")
    children = {
        int(child["nodeid"]): child
        for child in raw_children
        if isinstance(child, Mapping) and "nodeid" in child
    }
    if len(children) != len(raw_children):
        raise ValueError(f"split node has malformed child node IDs: {node!r}")

    child_ids = tuple(children)
    yes_id = int(node.get("yes", child_ids[0]))
    no_id = int(node.get("no", child_ids[1]))
    missing_id = int(node.get("missing", yes_id))
    try:
        yes = _tree_from_dump(children[yes_id], feature_names, depth=depth + 1)
        no = _tree_from_dump(children[no_id], feature_names, depth=depth + 1)
        missing = _tree_from_dump(children[missing_id], feature_names, depth=depth + 1)
    except KeyError as exc:
        raise ValueError(f"split node references unknown child {exc.args[0]}") from exc
    return Split(
        feature_index=_feature_index(node.get("split"), feature_names),
        threshold=_as_float(node.get("split_condition")),
        yes=yes,
        no=no,
        missing=missing,
    )


def _native_tree_from_arrays(
    tree: Mapping[str, Any],
    feature_names: Sequence[str],
) -> Tree:
    """Read the array form used by XGBoost's saved-model JSON format."""

    left = [int(value) for value in tree["left_children"]]
    right = [int(value) for value in tree["right_children"]]
    split_indices = [int(value) for value in tree["split_indices"]]
    split_conditions = [_as_float(value) for value in tree["split_conditions"]]
    base_weights = [_as_float(value) for value in tree["base_weights"]]
    default_left = [int(value) for value in tree.get("default_left", [1] * len(left))]

    def visit(node_id: int, depth: int) -> Tree:
        if depth > MAX_TREE_DEPTH:
            raise ValueError(f"XGBoost tree depth exceeds {MAX_TREE_DEPTH}")
        if left[node_id] == -1 and right[node_id] == -1:
            return Leaf(base_weights[node_id])
        yes = visit(left[node_id], depth + 1)
        no = visit(right[node_id], depth + 1)
        missing = yes if default_left[node_id] else no
        return Split(
            feature_index=split_indices[node_id],
            threshold=split_conditions[node_id],
            yes=yes,
            no=no,
            missing=missing,
        )

    return visit(0, 0)


def _native_model_payload(payload: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], float, float, list[str]]:
    learner = payload["learner"]
    gradient_booster = learner["gradient_booster"]
    model = gradient_booster["model"]
    raw_trees = model["trees"]
    if not isinstance(raw_trees, list):
        raise ValueError("saved XGBoost model trees must be a list")
    tree_dumps = [tree for tree in raw_trees if isinstance(tree, Mapping)]
    if len(tree_dumps) != len(raw_trees):
        raise ValueError("saved XGBoost model contains a malformed tree")

    learner_params = learner.get("learner_model_param", {})
    objective_params = learner.get("objective", {}).get("reg_loss_param", {})
    base_score = _as_float(learner_params.get("base_score", 0.0))
    train_params = gradient_booster.get("train_param", {})
    learning_rate_value = train_params.get(
        "learning_rate", gradient_booster.get("learning_rate")
    )
    if learning_rate_value is not None:
        learning_rate = _as_float(learning_rate_value)
    else:
        ratios: list[float] = []
        for tree in tree_dumps:
            left = [int(value) for value in tree.get("left_children", [])]
            right = [int(value) for value in tree.get("right_children", [])]
            base_weights = [_as_float(value) for value in tree.get("base_weights", [])]
            split_conditions = [
                _as_float(value) for value in tree.get("split_conditions", [])
            ]
            if not (len(left) == len(right) == len(base_weights) == len(split_conditions)):
                raise ValueError("saved XGBoost tree arrays have inconsistent lengths")
            for index, (left_child, right_child) in enumerate(zip(left, right)):
                if left_child == -1 and right_child == -1:
                    raw_value = base_weights[index]
                    scaled_value = split_conditions[index]
                    if raw_value != 0.0:
                        ratios.append(scaled_value / raw_value)
        if ratios:
            learning_rate = ratios[0]
            if learning_rate <= 0.0 or not math.isfinite(learning_rate):
                raise ValueError("saved XGBoost model has an invalid learning rate")
            if not all(math.isclose(value, learning_rate, rel_tol=1e-5, abs_tol=1e-7) for value in ratios[1:]):
                raise ValueError("saved XGBoost tree leaves use inconsistent learning rates")
        else:
            learning_rate = 1.0
    if not learning_rate:
        learning_rate = 1.0
    feature_names = [str(value) for value in learner.get("feature_names", [])]
    if objective_params.get("reg_loss_param"):
        base_score = _as_float(objective_params["reg_loss_param"])
    return tree_dumps, base_score, learning_rate, feature_names


def _load_payload(model: Path | str | Mapping[str, Any] | Any) -> Mapping[str, Any] | list[Any]:
    if isinstance(model, Mapping) or isinstance(model, list):
        return model
    if hasattr(model, "get_dump"):
        raw_trees = model.get_dump(dump_format="json")
        return {
            "format": "llmopt-xgboost-v1",
            "feature_names": list(getattr(model, "feature_names", None) or []),
            "base_score": 0.0,
            "learning_rate": 1.0,
            "trees": [json.loads(tree) for tree in raw_trees],
        }
    path = Path(model)
    return json.loads(path.read_text(encoding="utf-8"))


def load_model_bundle(model: Path | str | Mapping[str, Any] | Any) -> ModelBundle:
    payload = _load_payload(model)
    if isinstance(payload, list):
        raw_trees = payload
        feature_names: list[str] = []
        base_score = 0.0
        learning_rate = 1.0
    elif payload.get("format") == "llmopt-xgboost-v1" or "trees" in payload:
        raw_trees = payload.get("trees", [])
        feature_names = [str(value) for value in payload.get("feature_names", [])]
        base_score = _as_float(payload.get("base_score", 0.0))
        learning_rate = _as_float(payload.get("learning_rate", 1.0))
    elif "learner" in payload:
        raw_trees, base_score, learning_rate, feature_names = _native_model_payload(payload)
    else:
        raise ValueError("unsupported XGBoost JSON format")

    if not isinstance(raw_trees, list):
        raise ValueError("XGBoost JSON trees must be a list")
    if not feature_names:
        max_feature = -1
        for raw_tree in raw_trees:
            raw_text = json.dumps(raw_tree)
            for match in re.finditer(r'"split"\s*:\s*"f(\d+)"', raw_text):
                max_feature = max(max_feature, int(match.group(1)))
        feature_names = [f"f{index}" for index in range(max_feature + 1)]
    if not feature_names:
        raise ValueError("model has no features")

    trees: list[Tree] = []
    for raw_tree in raw_trees:
        if not isinstance(raw_tree, Mapping):
            raise ValueError("each XGBoost tree must be an object")
        if "leaf" in raw_tree or "children" in raw_tree:
            trees.append(_tree_from_dump(raw_tree, feature_names))
        elif "left_children" in raw_tree:
            trees.append(_native_tree_from_arrays(raw_tree, feature_names))
        else:
            raise ValueError("tree is neither dump-format nor saved-model format")

    return ModelBundle(
        feature_names=tuple(feature_names),
        base_score=base_score,
        learning_rate=learning_rate,
        trees=tuple(trees),
    )


def evaluate_bundle(bundle: ModelBundle, features: Sequence[float]) -> float:
    if len(features) != len(bundle.feature_names):
        raise ValueError(
            f"expected {len(bundle.feature_names)} features, got {len(features)}"
        )
    return bundle.base_score + bundle.learning_rate * sum(
        tree.evaluate(features) for tree in bundle.trees
    )


def _ocaml_float(value: float) -> str:
    if math.isnan(value):
        return "Float.nan"
    if math.isinf(value):
        return "Float.infinity" if value > 0 else "Float.neg_infinity"
    if value == 0.0 and math.copysign(1.0, value) < 0:
        return "-0.0"
    literal = format(value, ".17g")
    if "." not in literal and "e" not in literal.lower():
        literal += ".0"
    return literal


def _tree_ocaml(tree: Tree, arguments: Sequence[str], indent: int = 2) -> list[str]:
    prefix = " " * indent
    if isinstance(tree, Leaf):
        return [prefix + _ocaml_float(tree.value)]
    feature = arguments[tree.feature_index]
    lines = [prefix + f"if Float.is_nan {feature} then"]
    lines.extend(_tree_ocaml(tree.missing, arguments, indent + 2))
    lines.append(prefix + f"else if {feature} < {_ocaml_float(tree.threshold)} then")
    lines.extend(_tree_ocaml(tree.yes, arguments, indent + 2))
    lines.append(prefix + "else")
    lines.extend(_tree_ocaml(tree.no, arguments, indent + 2))
    return lines


def emit_ocaml(bundle: ModelBundle, *, source_name: str = "XGBoost JSON") -> str:
    arguments = tuple(f"f{index}" for index in range(len(bundle.feature_names)))
    argument_text = " ".join(arguments)
    lines = [
        "(* Generated by transpile_ocaml.py from " + source_name + ". *)",
        "(* The planner calls predict_latency with explicit, unboxed float arguments. *)",
        "",
    ]
    for index, tree in enumerate(bundle.trees):
        lines.append(f"let tree_{index} {argument_text} =")
        lines.extend(_tree_ocaml(tree, arguments))
        lines.append("")

    if bundle.trees:
        prediction = " +. ".join(
            f"tree_{index} {argument_text}" for index in range(len(bundle.trees))
        )
        prediction = f"{_ocaml_float(bundle.base_score)} +. {_ocaml_float(bundle.learning_rate)} *. ({prediction})"
    else:
        prediction = _ocaml_float(bundle.base_score)
    lines.extend([f"let predict_latency {argument_text} =", f"  {prediction}", ""])
    source = "\n".join(lines)
    encoded_size = len(source.encode("utf-8"))
    if encoded_size > MAX_OCAML_SOURCE_BYTES:
        raise ValueError(
            f"generated OCaml source is {encoded_size} bytes; "
            f"the {MAX_OCAML_SOURCE_BYTES}-byte item limit was exceeded"
        )
    return source


def transpile_xgboost_to_ocaml(
    model: Path | str | Mapping[str, Any] | Any,
    output_path: Path | str | None = None,
) -> str:
    """Return generated OCaml and optionally write it to ``output_path``."""

    bundle = load_model_bundle(model)
    source = emit_ocaml(bundle, source_name=str(model))
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        source = transpile_xgboost_to_ocaml(args.model, args.output)
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bytes": len(source.encode("utf-8")),
                "features": len(load_model_bundle(args.model).feature_names),
            },
            sort_keys=True,
        )
    )


__all__ = [
    "Leaf",
    "Split",
    "ModelBundle",
    "emit_ocaml",
    "evaluate_bundle",
    "load_model_bundle",
    "transpile_xgboost_to_ocaml",
]


if __name__ == "__main__":
    main()
