"""How precise is one pass rate, and how many examples make it precise enough."""

from __future__ import annotations

from dataclasses import dataclass

from ._stats import check_level, check_rate, wilson


@dataclass(frozen=True)
class PassRate:
    passes: int
    n: int
    rate: float
    low: float
    high: float
    confidence: float

    def summary(self) -> str:
        return (
            f"Pass rate {self.rate:.1%} ({self.passes} of {self.n}). "
            f"{self.confidence:.0%} confidence interval: {self.low:.1%} to {self.high:.1%}."
        )

    def __str__(self) -> str:
        return self.summary()


def pass_rate_interval(passes: int, n: int, confidence: float = 0.95) -> PassRate:
    """Confidence interval for a pass rate, using the Wilson score method.

    >>> pass_rate_interval(160, 200).summary()
    'Pass rate 80.0% (160 of 200). 95% confidence interval: 73.9% to 85.0%.'
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= passes <= n:
        raise ValueError("passes must be between 0 and n")
    check_level(confidence, "confidence")
    low, high = wilson(passes, n, confidence)
    return PassRate(passes, n, passes / n, low, high, confidence)


def margin_for_examples(n: int, pass_rate: float = 0.5, confidence: float = 0.95) -> float:
    """Half width of the confidence interval you can expect with ``n`` examples.

    ``pass_rate`` is your best guess of the score. 0.5 gives the widest
    (most cautious) answer.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    check_rate(pass_rate, "pass_rate", allow_edges=True)
    check_level(confidence, "confidence")
    low, high = wilson(pass_rate * n, n, confidence)
    return (high - low) / 2


def examples_for_margin(margin: float, pass_rate: float = 0.5, confidence: float = 0.95) -> int:
    """Smallest number of examples that gives a ``margin`` of plus or minus around the score.

    >>> examples_for_margin(0.05, pass_rate=0.8)
    245
    """
    check_level(margin, "margin")
    check_rate(pass_rate, "pass_rate", allow_edges=True)
    check_level(confidence, "confidence")
    low, high = 1, 1
    while margin_for_examples(high, pass_rate, confidence) > margin:
        high *= 2
        if high > 10**9:
            raise ValueError("margin is too small to reach")
    while low < high:
        mid = (low + high) // 2
        if margin_for_examples(mid, pass_rate, confidence) <= margin:
            high = mid
        else:
            low = mid + 1
    return low
