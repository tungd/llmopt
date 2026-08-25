import json
import tempfile
import unittest
from pathlib import Path

from racebench.needle import (
    DEFAULT_LENGTHS,
    DEFAULT_POSITIONS,
    EXPECTED,
    evaluate_response,
)
from racebench.benchmark import RequestResult, engine_pass
from racebench.profiles import official_shape_70x6, semantic_5x3
from racebench.score import (
    accuracy_factor,
    effective_request_score,
    final_score,
    request_score,
    tpot_score,
    ttft_score,
)
from racebench.trace import WorkloadTrace, validate_warmup_policy


class RacebenchContractTest(unittest.TestCase):
    def test_reference_latency_formula(self):
        self.assertAlmostEqual(ttft_score(10.0), 1.0)
        self.assertAlmostEqual(ttft_score(400.0), 0.0)
        self.assertAlmostEqual(tpot_score(1.0), 1.0)
        self.assertAlmostEqual(tpot_score(10.0), 0.0)
        self.assertAlmostEqual(request_score(205.0, 5.5, completion_tokens=8), 0.25)

    def test_request_mean_and_accuracy_factor(self):
        self.assertAlmostEqual(effective_request_score([1.0, 0.0, 0.5]), 0.5)
        self.assertAlmostEqual(accuracy_factor(0.13), 0.5)
        self.assertAlmostEqual(final_score(0.8, 0.13), 40.0)

    def test_trace_and_warmup_contract(self):
        raw = {
            "name": "test",
            "model": "model",
            "shared_system_prefix": "system",
            "conversation_prefixes": ["a", "b"],
            "user_messages": [["q1", "q2"], ["r1", "r2"]],
            "output_tokens_per_turn": 8,
            "num_conversations": 2,
            "user_turns_per_conversation": 2,
            "total_requests": 4,
        }
        trace = WorkloadTrace.from_dict(raw)
        self.assertEqual(trace.output_tokens_per_turn, ((8, 8), (8, 8)))
        self.assertEqual(trace.total_requests, 4)
        with tempfile.TemporaryDirectory() as directory:
            scored = Path(directory) / "scored.json"
            warmup = Path(directory) / "warmup.json"
            scored.write_text(json.dumps(raw), encoding="utf-8")
            warmup.write_text(json.dumps({**raw, "name": "warmup"}), encoding="utf-8")
            validate_warmup_policy(scored, warmup, require_shape_matched=True)
            with self.assertRaisesRegex(ValueError, "byte-identical"):
                validate_warmup_policy(scored, scored)

    def test_needle_defaults_are_reference_values(self):
        self.assertEqual(DEFAULT_LENGTHS, (7500, 9000, 16000, 30000))
        self.assertEqual(DEFAULT_POSITIONS, (10, 50, 90))
        self.assertEqual(EXPECTED, "RAVEN-4271")

    def test_needle_retrieval_is_separate_from_exact_response_format(self):
        exact = evaluate_response("RAVEN-4271")
        continued = evaluate_response("RAVEN-4271Lottery")
        prose = evaluate_response("The code is `RAVEN-4271`.")
        wrong = evaluate_response("RAVEN-42710")

        self.assertTrue(exact.retrieved)
        self.assertTrue(exact.exact_response)
        self.assertTrue(continued.retrieved)
        self.assertFalse(continued.exact_response)
        self.assertTrue(prose.retrieved)
        self.assertFalse(prose.exact_response)
        self.assertFalse(wrong.retrieved)
        self.assertFalse(wrong.exact_response)

    def test_engine_pass_separates_needle_validation(self):
        summary = {
            "requests": 4,
            "successful_requests": 4,
            "failed_requests": 0,
            "output_token_mismatches": 0,
            "ers": 0.4,
        }
        candidates = {
            "eager": {"warmup": summary, "scored": summary},
            "llmopt": {"warmup": summary, "scored": summary},
        }
        self.assertTrue(engine_pass(candidates, {"exact": True}))
        self.assertFalse(engine_pass(candidates, {"exact": False}))
        self.assertFalse(
            engine_pass(
                {"eager": {"warmup": summary, "scored": {**summary, "failed_requests": 1}}},
                {"exact": True},
            )
        )

    def test_exact_token_output_parity(self):
        from lfm25_benchsuite import token_output_parity

        def result(request_id, token_ids):
            return RequestResult(
                request_id=request_id,
                conversation=0,
                turn=0,
                scheduled_offset_s=0.0,
                queue_delay_ms=0.0,
                http_status=200,
                ttft_ms=1.0,
                tpot_ms=1.0,
                latency_ms=2.0,
                completion_tokens=len(token_ids),
                expected_completion_tokens=len(token_ids),
                output_text="",
                succeeded=True,
                error=None,
                score=1.0,
                output_token_ids=token_ids,
            )

        self.assertTrue(
            token_output_parity([result("a", [1, 2])], [result("a", [1, 2])])["equal"]
        )
        self.assertFalse(
            token_output_parity([result("a", [1, 2])], [result("a", [1, 3])])["equal"]
        )

    def test_native_needle_token_parity_is_explicit_and_optional(self):
        from lfm25_http_needle import _token_ids, token_parity

        expected = _token_ids("8832,563,2880,522,31429,526,7,2,1,553,849,18149")
        self.assertTrue(
            token_parity(
                [8832, 563, 2880, 522, 31429, 526, 7, 2, 1, 553, 849, 18149],
                expected,
            )
        )
        self.assertFalse(token_parity([8832, 563], expected))
        self.assertIsNone(token_parity([8832, 563], None))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _token_ids("8832,-1")

    def test_semantic_profile_matches_reference_shape(self):
        scored = semantic_5x3()
        warmup = semantic_5x3(warmup=True)
        self.assertEqual(scored.total_requests, 15)
        self.assertEqual(scored.output_tokens_per_turn, ((300, 300, 300),) * 5)
        self.assertEqual(scored.arrival.initial_offsets_s, warmup.arrival.initial_offsets_s)
        self.assertNotEqual(scored.shared_system_prefix, warmup.shared_system_prefix)
        self.assertNotEqual(scored.user_messages, warmup.user_messages)

    def test_official_profile_matches_reference_shape(self):
        scored = official_shape_70x6()
        warmup = official_shape_70x6(warmup=True)
        self.assertEqual(scored.num_conversations, 70)
        self.assertEqual(scored.user_turns_per_conversation, 6)
        self.assertEqual(scored.total_requests, 420)
        self.assertEqual(scored.output_tokens_per_turn, ((300,) * 6,) * 70)
        self.assertEqual(len(scored.arrival.initial_offsets_s), 70)
        self.assertEqual(scored.arrival.initial_offsets_s[-1], 0.8653318005326972)
        self.assertEqual(scored.arrival, warmup.arrival)
        self.assertEqual(scored.request, {"temperature": 0.0, "seed": 123})
        self.assertNotEqual(scored.shared_system_prefix, warmup.shared_system_prefix)


if __name__ == "__main__":
    unittest.main()
