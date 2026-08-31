import unittest

from vulkan_metal_compare import merge_receipts


class VulkanMetalCompareTest(unittest.TestCase):
    def test_merges_matching_working_sets_and_preserves_exact_deltas(self) -> None:
        def receipt(backend: str, dispatch: float, bandwidth: float):
            return {
                "backend": backend,
                "measurement": {
                    "warmup": 5,
                    "repeats": 21,
                    "dispatch_overhead_seconds": dispatch,
                    "streaming_copy_sweep": [
                        {
                            "stream_bytes": 16 * 1024 * 1024,
                            "effective_bandwidth_gbps": bandwidth,
                        }
                    ],
                },
            }

        result = merge_receipts(receipt("metal", 2e-6, 200.0), receipt("vulkan", 3e-6, 150.0))
        self.assertAlmostEqual(result["comparison"]["dispatch"]["vulkan_over_metal"], 1.5)
        sweep = result["comparison"]["streaming_copy_sweep"]
        self.assertEqual(sweep[0]["vulkan_minus_metal_gbps"], -50.0)
        self.assertEqual(sweep[0]["vulkan_over_metal"], 0.75)


if __name__ == "__main__":
    unittest.main()
