from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from examples.write_q8_weight_fixture import write_fixture
from llmopt_backend.tensor_archive import ALIGNMENT, MAGIC, VERSION


def read_index(path: Path) -> dict[str, tuple[int, tuple[int, ...], int, int]]:
    contents = path.read_bytes()
    magic, version, flags, count, data_start = struct.unpack_from(
        "<8sHHIQ", contents, 0
    )
    if magic != MAGIC or version != VERSION or flags != 0:
        raise AssertionError("invalid fixture archive prefix")
    if data_start % ALIGNMENT != 0:
        raise AssertionError("fixture archive data is not aligned")
    cursor = struct.calcsize("<8sHHIQ")
    result: dict[str, tuple[int, tuple[int, ...], int, int]] = {}
    for _ in range(count):
        name_length, dtype, rank, reserved = struct.unpack_from(
            "<IBBH", contents, cursor
        )
        cursor += struct.calcsize("<IBBH")
        if reserved != 0:
            raise AssertionError("invalid fixture archive entry")
        name = contents[cursor : cursor + name_length].decode("utf-8")
        cursor += name_length
        shape = struct.unpack_from(f"<{rank}Q", contents, cursor)
        cursor += 8 * rank
        offset, byte_length = struct.unpack_from("<QQ", contents, cursor)
        cursor += 16
        result[name] = dtype, shape, offset, byte_length
    if any(contents[cursor:data_start]):
        raise AssertionError("fixture archive has non-zero index padding")
    return result


class WeightArchiveFixtureTests(unittest.TestCase):
    def test_q8_fixture_is_one_binary_tensor_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "weights.llmopt"
            write_fixture(output)
            tensors = read_index(output)

            self.assertEqual(set(tensors), {"weight_q8", "weight_scale", "bias"})
            self.assertEqual(tensors["weight_q8"][:2], (5, (3, 4)))
            self.assertEqual(tensors["weight_scale"][:2], (1, (1, 3)))
            self.assertEqual(tensors["bias"][:2], (1, (1, 3)))
            for _dtype, _shape, offset, byte_length in tensors.values():
                self.assertEqual(offset % ALIGNMENT, 0)
                self.assertGreater(byte_length, 0)
            self.assertEqual(list(Path(directory).iterdir()), [output])


if __name__ == "__main__":
    unittest.main()
