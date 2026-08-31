from __future__ import annotations

import unittest
from bench_prompt_lookup_speculative import find_candidate_ngram, parse_llmopt_output


class PromptLookupSpeculativeTests(unittest.TestCase):
    def test_find_candidate_ngram_matches_exact_3gram(self) -> None:
        tokens = [1, 2, 3, 4, 5, 6, 7, 1, 2, 3]
        candidates = find_candidate_ngram(tokens, min_ngram=2, max_ngram=3, max_draft_tokens=3)
        self.assertEqual(candidates, [4, 5, 6])

    def test_find_candidate_ngram_returns_empty_when_no_match(self) -> None:
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        candidates = find_candidate_ngram(tokens, min_ngram=2, max_ngram=3, max_draft_tokens=3)
        self.assertEqual(candidates, [])

    def test_find_candidate_ngram_respects_max_draft_tokens(self) -> None:
        tokens = [10, 20, 30, 40, 50, 60, 70, 80, 10, 20]
        candidates = find_candidate_ngram(tokens, min_ngram=2, max_ngram=3, max_draft_tokens=2)
        self.assertEqual(candidates, [30, 40])

    def test_find_candidate_ngram_handles_empty_or_short_tokens(self) -> None:
        self.assertEqual(find_candidate_ngram([], min_ngram=2), [])
        self.assertEqual(find_candidate_ngram([1], min_ngram=2), [])

    def test_parse_real_llmopt_metrics(self) -> None:
        output = """completion-token-count: 3
completion-tokens: 4,5,6
elapsed-seconds: 0.250000
tokens-per-second: 12.000000
decode-mode: prompt-lookup
target-forward-passes: 2
verification-passes: 1
proposed-tokens: 3
accepted-draft-tokens: 2
"""
        parsed = parse_llmopt_output(output, ["llmopt-generate"])
        self.assertEqual(parsed["completion_tokens"], [4, 5, 6])
        self.assertEqual(parsed["accepted_draft_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
