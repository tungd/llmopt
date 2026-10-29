from pathlib import Path
import tempfile
import unittest
import numpy as np
from convert_gguf_exl3 import fwht128
from llmopt_backend.tensor_archive import TensorSpec, read_inventory, write_payloads

class Exl3ConversionTests(unittest.TestCase):
    def test_block_rotation_inverse_and_energy(self):
        rng=np.random.default_rng(18)
        x=rng.normal(size=(256,384)).astype(np.float32)
        y=fwht128(fwht128(x.copy(),0),1)
        restored=fwht128(fwht128(y.copy(),1),0)
        np.testing.assert_allclose(restored,x,atol=1e-6,rtol=1e-6)
        self.assertAlmostEqual(float(np.sum(x*x)/np.sum(y*y)),1.0,places=6)

    def test_canonical_dtype_inventory_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'weights.llmopt'
            payload=bytes(128*256//2)
            write_payloads([TensorSpec('weight',17,(256,128),len(payload))],path,lambda _:(payload,))
            self.assertEqual(read_inventory(path),{'weight':('EXL3_4',(256,128))})
