"""Build and run the bounded EXL3 fixture probes strictly sequentially.

Every subprocess has a timeout and a full log. On timeout its owned process group
is terminated and reaped before the controller exits; no automatic retry occurs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "_artifacts/exl3-slice")
    parser.add_argument("--fixture", type=Path, action="append")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Inspect executable identities, not a substring match on our own command.
    active = []
    for line in subprocess.check_output(["ps", "-axo", "pid=,comm=,args="], text=True).splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) < 3 or int(fields[0]) == os.getpid():
            continue
        executable, command = Path(fields[1]).name, fields[2]
        if executable.startswith(("llmopt-", "metal-exl3-probe")) or executable in ("ninja", "ocamlopt", "swiftc", "metal", "metallib"):
            active.append(line.strip())
        elif executable.startswith("python") and any(Path(a).name in ("build_mellum21_engine.py", "build_gemma4_engine.py") for a in command.split()):
            active.append(line.strip())
    if active:
        (out / "run.json").write_text(json.dumps({"status": "not_started", "pid": os.getpid(), "active_processes": active}, indent=2) + "\n")
        print("Another compiler/engine/probe process is active:\n" + "\n".join(active), file=sys.stderr)
        return 2
    receipt = {"job": f"exl3-{os.getpid()}", "pid": os.getpid(), "steps": [], "status": "running"}
    def save():
        (out / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    save()
    def run(label, command, timeout=180):
        path = out / (label + ".log")
        start = time.monotonic()
        with path.open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            timed_out = False
            try:
                code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
        receipt["steps"].append({"name": label, "command": list(map(str, command)), "pid": child.pid,
                                  "exit_code": code, "timeout": timed_out, "seconds": time.monotonic() - start, "log": str(path)})
        save()
        if code or timed_out:
            receipt["status"] = "timed_out" if timed_out else "failed"
            save()
            print(json.dumps(receipt))
            raise RuntimeError(f"{label}: exit={code} timeout={timed_out}; log={path}")
        print(f"{label}: passed; log={path}", flush=True)
    run("build", ["ninja", "-f", "ninja.build", "-j1", "_build/bin/llmopt-exl3-probe-source", "_build/bin/llmopt-exl3-test"], 300)
    run("schema-tests", ["_build/bin/llmopt-exl3-test"])
    runner = out / "metal-exl3-probe"
    run("swift-build", ["swiftc", "-O", "bench/metal_exl3_probe.swift", "-o", str(runner)], 180)
    fixtures = args.fixture or [ROOT / "test/fixtures/exl3/mul1-4bit.llmopt", ROOT / "test/fixtures/exl3/mul1-3bit.llmopt"]
    for i, fixture in enumerate(fixtures):
        fixture = fixture.resolve()
        directory = out / f"fixture-{i}"
        directory.mkdir(exist_ok=True)
        run(f"emit-{i}", ["_build/bin/llmopt-exl3-probe-source", str(fixture), str(directory)])
        run(f"device-{i}", [str(runner), str(directory / "probe.json"), str(directory / "result.json")], 300)
        receipt.setdefault("artifacts", []).append({"fixture": str(fixture), "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
            "result": str(directory / "result.json"), "generated_sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.glob("*.metal"))}})
    receipt["status"] = "passed"
    save()
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
