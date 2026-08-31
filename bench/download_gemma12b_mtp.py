#!/usr/bin/env python3
"""Download and verify the pinned Gemma 4 12B QAT and MTP GGUF files."""

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "unsloth/gemma-4-12B-it-qat-GGUF"
REVISION = "980b060c40a8539ac159e0501a3e0f66a6365af3"
FILES = {
    "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf": {
        "size": 6_716_356_800,
        "sha256": "90fd44e29e0d7cffeb0fd00dc73cfdab9ed0b0e95306ecf7821ea634c940c370",
    },
    "mtp-gemma-4-12B-it.gguf": {
        "size": 253_708_800,
        "sha256": "fcb35dea42c71333db904cee11baac525c9ef872818ee3753f6cb156f3c6f4f6",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_size: int, expected_sha256: str) -> dict:
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{path.name}: expected {expected_size} bytes, found {actual_size}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{path.name}: expected SHA256 {expected_sha256}, found {actual_sha256}"
        )
    return {
        "path": str(path),
        "size": actual_size,
        "sha256": actual_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and verify pinned Gemma 4 12B QAT and MTP GGUFs"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="require both files to exist in the local Hugging Face cache",
    )
    args = parser.parse_args()

    print(f"Resolving {REPO_ID} at {REVISION}...")
    paths = {}
    for filename, expected in FILES.items():
        print(f"\n--- Resolving {filename} ---")
        resolved = Path(hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="model",
            revision=REVISION,
            local_files_only=args.verify_only,
        ))
        receipt = verify_file(
            resolved,
            expected_size=expected["size"],
            expected_sha256=expected["sha256"],
        )
        print(f"Path:   {receipt['path']}")
        print(f"Size:   {receipt['size']} bytes")
        print(f"SHA256: {receipt['sha256']}")
        paths[filename] = receipt

    print("\nBoth pinned GGUF files passed size and SHA256 verification:")
    for filename, receipt in paths.items():
        print(f"  {filename}: {receipt['sha256']}")

if __name__ == "__main__":
    main()
