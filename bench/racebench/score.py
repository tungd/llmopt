"""The ERS request and final-score functions used by the reference runner."""

from __future__ import annotations

from dataclasses import dataclass

TTFT_FLOOR_MS = 10.0
TTFT_CEILING_MS = 400.0
TPOT_FLOOR_MS = 1.0
TPOT_CEILING_MS = 10.0
GAMMA = 2.0
TTFT_WEIGHT = 0.5


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def latency_component(
    observed_ms: float,
    *,
    floor_ms: float,
    ceiling_ms: float,
    gamma: float = GAMMA,
) -> float:
    if observed_ms < 0:
        raise ValueError("latency cannot be negative")
    normalized = clamp((ceiling_ms - observed_ms) / (ceiling_ms - floor_ms))
    return normalized**gamma


def ttft_score(ttft_ms: float) -> float:
    return latency_component(
        ttft_ms, floor_ms=TTFT_FLOOR_MS, ceiling_ms=TTFT_CEILING_MS
    )


def tpot_score(tpot_ms: float) -> float:
    return latency_component(
        tpot_ms, floor_ms=TPOT_FLOOR_MS, ceiling_ms=TPOT_CEILING_MS
    )


def request_score(
    ttft_ms: float | None,
    tpot_ms: float | None,
    *,
    completion_tokens: int,
    succeeded: bool = True,
    ttft_weight: float = TTFT_WEIGHT,
) -> float:
    if not succeeded or completion_tokens <= 0 or ttft_ms is None or tpot_ms is None:
        return 0.0
    if not 0.0 <= ttft_weight <= 1.0:
        raise ValueError("ttft_weight must be in [0, 1]")
    return ttft_weight * ttft_score(ttft_ms) + (1.0 - ttft_weight) * tpot_score(tpot_ms)


def effective_request_score(scores: list[float]) -> float:
    if not scores:
        raise ValueError("at least one request score is required")
    if any(not 0.0 <= score <= 1.0 for score in scores):
        raise ValueError("request scores must be in [0, 1]")
    return sum(scores) / len(scores)


def accuracy_factor(delta: float) -> float:
    if delta <= 0.10:
        return 1.0
    if delta >= 0.16:
        return 0.0
    return 1.0 - (delta - 0.10) / 0.06


def final_score(ers: float, delta: float) -> float:
    if not 0.0 <= ers <= 1.0:
        raise ValueError("ERS must be in [0, 1]")
    return 100.0 * ers * accuracy_factor(delta)


@dataclass(frozen=True)
class RequestMetrics:
    ttft_ms: float | None
    tpot_ms: float | None
    completion_tokens: int
    succeeded: bool = True

    @property
    def score(self) -> float:
        return request_score(
            self.ttft_ms,
            self.tpot_ms,
            completion_tokens=self.completion_tokens,
            succeeded=self.succeeded,
        )
