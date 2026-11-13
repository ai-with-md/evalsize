"""Comparing two models (or prompts) on pass/fail evals.

"Paired" means both models answer the same questions, which is how most evals
are run. Pairing usually needs far fewer questions than comparing two separate
samples, because both models get the easy questions right and the hard ones
wrong, and only the questions where they disagree carry information.

The key planning input for a paired comparison is the **disagreement rate**:
the share of questions where exactly one of the two models is right. Measure
it on a small pilot if you can. If you leave it out, the functions assume the
two models make mistakes independently, which is cautious: real models
usually agree more than that, so you will often need fewer questions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

from ._stats import (
    binom_two_sided_half,
    check_level,
    check_rate,
    normal_cdf,
    z_power,
    z_two_sided,
)


def _independent_disagreement(p_a: float, p_b: float) -> float:
    return p_a * (1 - p_b) + p_b * (1 - p_a)


def _check_disagreement(p_a: float, p_b: float, disagreement: float) -> None:
    d = p_b - p_a
    only_b = (disagreement + d) / 2
    only_a = (disagreement - d) / 2
    if only_a < -1e-12 or only_b < -1e-12:
        raise ValueError(
            f"disagreement {disagreement:g} is too low: going from {p_a:g} to {p_b:g} "
            f"means the models must disagree on at least {abs(d):g} of questions"
        )
    if only_b > min(p_b, 1 - p_a) + 1e-12 or only_a > min(p_a, 1 - p_b) + 1e-12:
        raise ValueError(
            f"disagreement {disagreement:g} is too high for pass rates {p_a:g} and {p_b:g}"
        )


def _setup(baseline: float, improvement: float, disagreement: Optional[float]):
    check_rate(baseline, "baseline")
    if improvement == 0:
        raise ValueError("improvement must not be zero")
    target = baseline + improvement
    check_rate(target, "baseline + improvement")
    if disagreement is None:
        disagreement = _independent_disagreement(baseline, target)
    else:
        check_rate(disagreement, "disagreement")
        _check_disagreement(baseline, target, disagreement)
    return target, disagreement


def _split_alpha(alpha: float, comparisons: int) -> float:
    """Bonferroni: share the false alarm rate across all comparisons."""
    check_level(alpha, "alpha")
    if int(comparisons) != comparisons or comparisons < 1:
        raise ValueError("comparisons must be a whole number, 1 or more")
    return alpha / comparisons


def examples_to_compare(
    baseline: float,
    improvement: float,
    paired: bool = True,
    disagreement: Optional[float] = None,
    alpha: float = 0.05,
    power: float = 0.8,
    comparisons: int = 1,
) -> int:
    """Questions needed to reliably detect that model B beats model A.

    Args:
        baseline: model A's expected pass rate, for example 0.70.
        improvement: the smallest gain worth detecting, for example 0.05 for
            five percentage points.
        paired: True if both models answer the same questions. If False, the
            answer is the number of questions for each model.
        disagreement: share of questions where exactly one model is right.
            Only used when paired.
        alpha: false alarm rate (two-sided), for all comparisons together.
        power: chance of detecting the improvement if it is real.
        comparisons: how many comparisons the eval will make, for example 4
            when testing 4 new prompts against the current one. The false
            alarm rate is shared across them (Bonferroni), so more
            comparisons need more questions.

    Paired uses the McNemar sample size formula (Connor, 1987). Unpaired uses
    the standard two-proportion formula.

    >>> examples_to_compare(0.70, 0.05, disagreement=0.15)
    469
    >>> examples_to_compare(0.70, 0.05, disagreement=0.15, comparisons=4)
    667
    """
    alpha = _split_alpha(alpha, comparisons)
    check_level(power, "power")
    target, psi = _setup(baseline, improvement, disagreement)
    d = abs(improvement)
    za, zb = z_two_sided(alpha), z_power(power)
    if paired:
        n = (za * math.sqrt(psi) + zb * math.sqrt(psi - d * d)) ** 2 / (d * d)
    else:
        p_bar = (baseline + target) / 2
        spread = baseline * (1 - baseline) + target * (1 - target)
        n = (za * math.sqrt(2 * p_bar * (1 - p_bar)) + zb * math.sqrt(spread)) ** 2 / (d * d)
    return math.ceil(n - 1e-9)


def power_to_compare(
    n: int,
    baseline: float,
    improvement: float,
    paired: bool = True,
    disagreement: Optional[float] = None,
    alpha: float = 0.05,
    comparisons: int = 1,
) -> float:
    """Chance that an eval with ``n`` questions detects the improvement."""
    if n <= 0:
        raise ValueError("n must be positive")
    alpha = _split_alpha(alpha, comparisons)
    target, psi = _setup(baseline, improvement, disagreement)
    d = abs(improvement)
    za = z_two_sided(alpha)
    if paired:
        spread = math.sqrt(max(psi - d * d, 1e-15))
        return normal_cdf((d * math.sqrt(n) - za * math.sqrt(psi)) / spread)
    p_bar = (baseline + target) / 2
    spread = math.sqrt(baseline * (1 - baseline) + target * (1 - target))
    return normal_cdf((d * math.sqrt(n) - za * math.sqrt(2 * p_bar * (1 - p_bar))) / spread)


def smallest_detectable_improvement(
    n: int,
    baseline: float,
    paired: bool = True,
    disagreement: Optional[float] = None,
    alpha: float = 0.05,
    power: float = 0.8,
    comparisons: int = 1,
) -> Optional[float]:
    """Smallest gain over ``baseline`` that ``n`` questions can reliably detect.

    Returns None if no improvement up to a perfect score is detectable.

    If you give a fixed ``disagreement``, the improvement can never exceed it,
    so the answer is capped there.
    """
    check_level(power, "power")
    check_rate(baseline, "baseline")
    _split_alpha(alpha, comparisons)
    if n <= 0:
        raise ValueError("n must be positive")
    high = 1 - baseline - 1e-9
    if disagreement is not None and paired:
        high = min(high, disagreement)

    def detectable(gain: float) -> bool:
        try:
            return power_to_compare(n, baseline, gain, paired, disagreement, alpha, comparisons) >= power
        except ValueError:
            return False

    if not detectable(high):
        return None
    low = 1e-6
    for _ in range(60):
        mid = (low + high) / 2
        if detectable(mid):
            high = mid
        else:
            low = mid
    return high


@dataclass(frozen=True)
class Comparison:
    n: int
    only_a_right: int
    only_b_right: int
    difference: float
    low: float
    high: float
    p_value: float
    confidence: float

    def summary(self) -> str:
        direction = "better" if self.difference > 0 else "worse"
        verdict = (
            f"B looks {direction} than A"
            if self.p_value < 1 - self.confidence
            else "The difference could easily be noise"
        )
        return (
            f"B minus A: {self.difference:+.1%} over {self.n} shared questions "
            f"({self.confidence:.0%} interval {self.low:+.1%} to {self.high:+.1%}, p = {self.p_value:.3g}). "
            f"{verdict}. The models disagreed on {self.only_a_right + self.only_b_right} questions."
        )

    def __str__(self) -> str:
        return self.summary()


def compare_counts(n: int, only_a_right: int, only_b_right: int, confidence: float = 0.95) -> Comparison:
    """Is B really better than A? Uses only the questions where they disagree.

    Args:
        n: questions both models answered.
        only_a_right: questions A got right and B got wrong.
        only_b_right: questions B got right and A got wrong.

    The p-value is the exact McNemar test. The interval for the difference
    uses the Agresti and Min (2005) adjustment, which behaves well with small
    counts.

    >>> compare_counts(300, only_a_right=12, only_b_right=28).p_value < 0.05
    True
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if only_a_right < 0 or only_b_right < 0 or only_a_right + only_b_right > n:
        raise ValueError("disagreement counts must be non-negative and add up to at most n")
    check_level(confidence, "confidence")
    diff = (only_b_right - only_a_right) / n
    a_adj, b_adj, n_adj = only_a_right + 0.5, only_b_right + 0.5, n + 2
    d_adj = (b_adj - a_adj) / n_adj
    var = ((a_adj + b_adj) / n_adj - d_adj ** 2) / n_adj
    half = z_two_sided(1 - confidence) * math.sqrt(max(var, 0.0))
    p_value = binom_two_sided_half(only_b_right, only_a_right + only_b_right)
    return Comparison(
        n, only_a_right, only_b_right, diff,
        max(-1.0, d_adj - half), min(1.0, d_adj + half), p_value, confidence,
    )


def compare_results(a: Iterable, b: Iterable, confidence: float = 0.95) -> Comparison:
    """Same as ``compare_counts`` but from two lists of pass/fail results, in question order."""
    a, b = [bool(x) for x in a], [bool(x) for x in b]
    if len(a) != len(b):
        raise ValueError("a and b must have one result per question, in the same order")
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    return compare_counts(len(a), only_a, only_b, confidence)
