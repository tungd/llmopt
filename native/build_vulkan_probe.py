#!/usr/bin/env python3
"""Build the portable Vulkan hardware probe, bootstrapping MoltenVK on macOS."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import urllib.request


MOLTENVK_VERSION = "1.4.2"
MOLTENVK_URL = (
    "https://github.com/KhronosGroup/MoltenVK/releases/download/"
    f"v{MOLTENVK_VERSION}/MoltenVK-macos.tar"
)
MOLTENVK_SHA256 = "f95765a6229cb7b915990a2890ce12ebe36a730b021545d3d52ae69ce4c4024e"
GLSLANG_VERSION = "16.5.0"
GLSLANG_URL = (
    "https://github.com/KhronosGroup/glslang/releases/download/"
    f"{GLSLANG_VERSION}/glslang-{GLSLANG_VERSION}-macos-universal-release.tar.gz"
)
GLSLANG_SHA256 = "0f0b8ae3873d1f182ecd1806cd1a10fee3f9965896a660b7a0bbe6bdd9e1f46d"


def checked_download(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if digest != expected_sha256:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"checksum mismatch for {url}: expected {expected_sha256}, got {digest}"
        )


def extract_once(archive: Path, destination: Path, marker: Path) -> None:
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        bundle.extractall(destination, filter="data")
    marker.touch()


def executable(name: str, sdk: Path | None) -> str:
    if sdk is not None:
        candidate = sdk / "bin" / name
        if candidate.exists():
            return str(candidate)
    found = shutil.which(name)
    if found is None:
        raise RuntimeError(f"required tool not found: {name}")
    return found


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=Path("_build/vulkan"))
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    build = args.build_dir.resolve()
    dependencies = build / "deps"
    build.mkdir(parents=True, exist_ok=True)

    system = platform.system()
    vulkan_sdk_env = os.environ.get("VULKAN_SDK") or os.environ.get("LLMOPT_VULKAN_SDK")
    sdk = Path(vulkan_sdk_env).resolve() if vulkan_sdk_env else None
    include_flags: list[str]
    link_flags: list[str]

    if system == "Darwin" and sdk is None:
        molten_archive = dependencies / "downloads" / "MoltenVK-macos.tar"
        glslang_archive = dependencies / "downloads" / "glslang-macos.tar.gz"
        checked_download(MOLTENVK_URL, molten_archive, MOLTENVK_SHA256)
        checked_download(GLSLANG_URL, glslang_archive, GLSLANG_SHA256)
        molten_root = dependencies / "moltenvk"
        glslang_root = dependencies / "glslang"
        extract_once(molten_archive, molten_root, molten_root / ".complete")
        extract_once(glslang_archive, glslang_root, glslang_root / ".complete")
        sdk = glslang_root
        molten = molten_root / "MoltenVK" / "MoltenVK"
        library = molten / "dynamic" / "dylib" / "macOS"
        include_flags = [f"-I{molten / 'include'}"]
        link_flags = [
            f"-L{library}",
            "-lMoltenVK",
            f"-Wl,-rpath,{library}",
        ]
    else:
        include_flags = [f"-I{sdk / 'include'}"] if sdk is not None else []
        library = sdk / "lib" if sdk is not None else None
        link_flags = ([f"-L{library}", f"-Wl,-rpath,{library}"] if library else []) + [
            "-lvulkan"
        ]

    validator = executable("glslangValidator", sdk)
    for name in ("copy", "empty"):
        run(
            [
                validator,
                "-V",
                "--spirv-val",
                "--target-env",
                "vulkan1.1",
                str(repository / "native" / f"vulkan_{name}.comp"),
                "-o",
                str(build / f"vulkan_{name}.spv"),
            ]
        )

    compiler = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        raise RuntimeError("C++ compiler not found")
    run(
        [
            compiler,
            "-std=c++17",
            "-O3",
            "-Wall",
            "-Wextra",
            "-Wno-missing-field-initializers",
            *include_flags,
            str(repository / "native" / "vulkan_probe.cpp"),
            *link_flags,
            "-o",
            str(build / "llmopt-vulkan-probe"),
        ]
    )


if __name__ == "__main__":
    main()
