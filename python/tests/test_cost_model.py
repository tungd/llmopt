from __future__ import annotations

import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "python"))

from llmopt_backend.cost_model.train import load_dataset, train_cost_model  # noqa: E402
from llmopt_backend.cost_model.transpile_ocaml import (  # noqa: E402
    evaluate_bundle,
    load_model_bundle,
    transpile_xgboost_to_ocaml,
)


MODEL = {
    "format": "llmopt-xgboost-v1",
    "feature_names": ["f0", "f1", "f2"],
    "base_score": 1.25,
    "learning_rate": 0.2,
    "trees": [
        {
            "nodeid": 0,
            "depth": 0,
            "split": "f0",
            "split_condition": 0.5,
            "yes": 1,
            "no": 2,
            "missing": 1,
            "children": [
                {"nodeid": 1, "leaf": -0.75},
                {
                    "nodeid": 2,
                    "depth": 1,
                    "split": "f1",
                    "split_condition": -1.0,
                    "yes": 3,
                    "no": 4,
                    "missing": 4,
                    "children": [
                        {"nodeid": 3, "leaf": 1.5},
                        {"nodeid": 4, "leaf": 0.25},
                    ],
                },
            ],
        },
        {"nodeid": 0, "leaf": 0.125},
    ],
}


def _compile_and_predict(source: str, rows: list[tuple[float, float, float]], tmp_path: Path) -> list[float]:
    generated = tmp_path / "kernel_cost_model.ml"
    runner = tmp_path / "runner.ml"
    executable = tmp_path / "runner"
    generated.write_text(source, encoding="utf-8")
    runner.write_text(
        """let () =
  try
    while true do
      match String.split_on_char ' ' (input_line stdin) with
      | [f0; f1; f2] ->
          Printf.printf \"%.17g\\n\"
            (Kernel_cost_model.predict_latency (float_of_string f0)
               (float_of_string f1) (float_of_string f2));
          flush stdout
      | _ -> exit 2
    done
  with End_of_file -> ()
""",
        encoding="utf-8",
    )
    subprocess.run(
        ["ocamlopt", "-c", generated.name],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    subprocess.run(
        ["ocamlopt", "-o", str(executable), generated.with_suffix(".cmx").name, runner.name],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    input_data = "".join(f"{f0} {f1} {f2}\n" for f0, f1, f2 in rows)
    completed = subprocess.run(
        [str(executable)],
        input=input_data,
        check=True,
        capture_output=True,
        text=True,
    )
    return [float(line) for line in completed.stdout.splitlines()]


def test_transpiled_ocaml_matches_python_tree_evaluation_for_1000_points(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    output_path = tmp_path / "kernel_cost_model.ml"
    model_path.write_text(json.dumps(MODEL), encoding="utf-8")
    source = transpile_xgboost_to_ocaml(model_path, output_path)
    bundle = load_model_bundle(model_path)
    assert output_path.read_text(encoding="utf-8") == source
    assert "Float.is_nan f0" in source
    assert "let predict_latency f0 f1 f2" in source

    rng = random.Random(23)
    rows = [(rng.uniform(-3.0, 3.0), rng.uniform(-3.0, 3.0), rng.uniform(-3.0, 3.0)) for _ in range(1000)]
    expected = [evaluate_bundle(bundle, row) for row in rows]
    if shutil.which("ocamlopt") is None:
        pytest.skip("ocamlopt is unavailable")
    actual = _compile_and_predict(source, rows, tmp_path)
    assert len(actual) == len(expected) == 1000
    for left, right in zip(actual, expected):
        assert math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def test_missing_values_follow_xgboost_default_child(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(MODEL), encoding="utf-8")
    bundle = load_model_bundle(model_path)
    assert evaluate_bundle(bundle, (float("nan"), 0.0, 0.0)) == pytest.approx(1.25 + 0.2 * (-0.75 + 0.125))


def test_training_writes_portable_tree_bundle(tmp_path: Path) -> None:
    xgboost = pytest.importorskip("xgboost")
    np = pytest.importorskip("numpy")
    dataset = tmp_path / "dataset.jsonl"
    rows = []
    for index in range(32):
        rows.append(
            {
                "m": 1 + index % 8,
                "n": 512 + (index % 4) * 512,
                "k": 512 + (index % 2) * 512,
                "tile_m": 16,
                "tile_n": 16,
                "tile_k": 64,
                "latency_us": 2.0 + index / 32.0,
                "hardware": {"gpu_core_count": 16, "memory_bandwidth_gbps": 273},
                "mode": "metal",
            }
        )
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "model.json"
    bundle = train_cost_model(dataset, output, n_estimators=4, max_depth=2)
    assert output.exists()
    assert bundle["format"] == "llmopt-xgboost-v1"
    assert bundle["feature_names"]
    assert len(bundle["trees"]) == 4
    assert load_model_bundle(output).feature_names == tuple(bundle["feature_names"])

    dataset_rows = load_dataset(dataset)
    matrix = np.asarray(
        [row.features for row in dataset_rows],
        dtype=np.float32,
    )
    # The native model check uses the same XGBoost parameters as train.py and
    # ensures dump-format and saved-model JSON have identical semantics.
    targets = np.asarray([row.target for row in dataset_rows], dtype=np.float32)
    booster = xgboost.train(
        {
            "objective": "reg:squarederror",
            "max_depth": 2,
            "eta": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "seed": 23,
            "nthread": 1,
            "tree_method": "hist",
            "base_score": 0.0,
        },
        xgboost.DMatrix(matrix, label=targets, feature_names=list(bundle["feature_names"])),
        num_boost_round=4,
    )
    native_output = tmp_path / "native-model.json"
    booster.save_model(native_output)
    native_bundle = load_model_bundle(native_output)
    rng = random.Random(23)
    points = [
        tuple(float(rng.uniform(-1.0, 20000.0)) for _ in bundle["feature_names"])
        for _ in range(1000)
    ]
    native_predictions = booster.predict(
        xgboost.DMatrix(np.asarray(points, dtype=np.float32), feature_names=list(bundle["feature_names"]))
    )
    portable_predictions = np.asarray(
        [evaluate_bundle(native_bundle, point) for point in points]
    )
    assert np.max(np.abs(native_predictions - portable_predictions)) < 1e-5
