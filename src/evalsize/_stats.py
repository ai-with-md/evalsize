"""Small statistics helpers built on the standard library only."""

from __future__ import annotations

import math
from statistics import NormalDist

_NORMAL = NormalDist()


def check_rate(value: float, name: str, allow_edges: bool = False) -> None:
    low_ok = value >= 0 if allow_edges else value > 0
    high_ok = value <= 1 if allow_edges else value < 1
    if not (low_ok and high_ok):
        bounds = "between 0 and 1" if allow_edges else "strictly between 0 and 1"
        raise ValueError(f"{name} must be {bounds}, got {value!r}")


def check_level(value: float, name: str) -> None:
    if not 0 < value < 1:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {value!r}")


def z_two_sided(alpha: float) -> float:
    """Critical value for a two-sided test or interval at level alpha."""
    return _NORMAL.inv_cdf(1 - alpha / 2)


def z_power(power: float) -> float:
    return _NORMAL.inv_cdf(power)


def normal_cdf(x: float) -> float:
    return _NORMAL.cdf(x)


def wilson(successes: float, n: float, confidence: float) -> tuple:
    """Wilson score interval for a proportion. Works well near 0 and 1."""
    z = z_two_sided(1 - confidence)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def binom_two_sided_half(k: int, n: int) -> float:
    """Exact two-sided p-value for k successes out of n when p = 0.5."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)
