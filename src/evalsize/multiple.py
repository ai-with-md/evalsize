"""Comparing several prompts or models at once.

Test five prompts that are all exactly as good as the current one, and there
is still a fair chance that one of them looks significantly better by luck.
Pick that one, and you ship a change that does nothing. The more candidates
you try, the more likely a lucky winner.

The fix is to share the 5% false alarm rate across all the comparisons. This
module does that with the Holm method for p-values, which is never less
powerful than the simpler Bonferroni method, and Bonferroni adjusted
intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ._stats import check_level
from .comparison import compare_counts


def false_win_chance(comparisons: int, alpha: float = 0.05) -> float:
    """Chance, at most, that at least one comparison is a false alarm.

    This is the risk of a lucky winner when none of the candidates is really
    better and you check each one at ``alpha`` without any correction. It is
    an upper bound: comparisons against the same baseline are correlated, so
    the real chance is a little lower.

    >>> round(false_win_chance(4), 3)
    0.185
    """
    if int(comparisons) != comparisons or comparisons < 1:
        raise ValueError("comparisons must be a whole number, 1 or more")
    check_level(alpha, "alpha")
    return 1 - (1 - alpha) ** comparisons


def holm(p_values: Sequence[float]) -> List[float]:
    """Holm adjusted p-values, in the same order as given.

    Compare each adjusted p-value with your usual threshold (0.05) and the
    chance of any false alarm across the whole set stays at that threshold.

    >>> [round(p, 3) for p in holm([0.01, 0.04, 0.03, 0.005])]
    [0.03, 0.06, 0.06, 0.02]
    """
    p = [float(x) for x in p_values]
    if any(not 0 <= x <= 1 for x in p):
        raise ValueError("p-values must be between 0 and 1")
    m = len(p)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(sorted(range(m), key=p.__getitem__)):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted


@dataclass(frozen=True)
class PairResult:
    name: str
    versus: str
    difference: float
    low: float
    high: float
    p_value: float
    adjusted_p: float
    significant: bool
    significant_uncorrected: bool


@dataclass(frozen=True)
class ManyComparison:
    n: int
    pass_rates: Dict[str, float]
    results: Tuple[PairResult, ...]
    baseline: Optional[str]
    confidence: float

    @property
    def comparisons(self) -> int:
        return len(self.results)

    @property
    def winners(self) -> List[str]:
        """Candidates reliably better than the baseline, after the correction."""
        return [r.name for r in self.results if r.significant and r.difference > 0]

    def summary(self) -> str:
        alpha = 1 - self.confidence
        lines = []
        if self.baseline is not None:
            lines.append(
                f"{self.comparisons} candidates against '{self.baseline}' "
                f"({self.pass_rates[self.baseline]:.1%}) on {self.n} shared questions, "
                f"corrected for {self.comparisons} comparisons."
            )
        else:
            lines.append(
                f"All {self.comparisons} pairs among {len(self.pass_rates)} candidates "
                f"on {self.n} shared questions, corrected for {self.comparisons} comparisons."
            )
        width = max(len(r.name) for r in self.results)
        vwidth = max(len(r.versus) for r in self.results)
        for r in self.results:
            label = "reliably different" if r.significant else "could be noise"
            if r.significant and self.baseline is not None:
                label = "reliably better" if r.difference > 0 else "reliably worse"
            lines.append(
                f"  {r.name:<{width}}  vs {r.versus:<{vwidth}}  {r.difference:+6.1%}  "
                f"({r.low:+.1%} to {r.high:+.1%})  adjusted p = {r.adjusted_p:.3g}  {label}"
            )
        lucky = [r.name for r in self.results if r.significant_uncorrected and not r.significant]
        if self.baseline is not None:
            if self.winners:
                lines.append(f"Reliably better than '{self.baseline}': {', '.join(self.winners)}.")
            else:
                lines.append(f"None of the candidates is reliably better than '{self.baseline}'.")
        if lucky:
            lines.append(
                f"Without the correction, {', '.join(lucky)} would have looked significant "
                f"at {alpha:g}. Treat {'it' if len(lucky) == 1 else 'them'} as a possible lucky win."
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def compare_many(
    results: Mapping[str, Iterable],
    baseline: Optional[str] = None,
    confidence: float = 0.95,
) -> ManyComparison:
    """Compare several prompts or models that answered the same questions.

    Args:
        results: candidate name to a list of pass/fail results, one per
            question, in the same question order for every candidate.
        baseline: name of the current prompt or model. Each other candidate
            is compared with it. Leave it out to compare every pair.
        confidence: 0.95 keeps the chance of any false alarm across all the
            comparisons at 5%.

    Each comparison uses the exact McNemar test. P-values are adjusted with
    the Holm method; intervals use the Bonferroni level
    ``1 - (1 - confidence) / comparisons`` so they agree with the tests.
    """
    check_level(confidence, "confidence")
    data = {name: [bool(x) for x in values] for name, values in results.items()}
    if len(data) < 2:
        raise ValueError("need at least two candidates")
    lengths = {len(v) for v in data.values()}
    if len(lengths) != 1:
        raise ValueError("every candidate needs one result per question, in the same order")
    n = lengths.pop()
    if n == 0:
        raise ValueError("no questions")
    if baseline is not None and baseline not in data:
        raise ValueError(f"baseline {baseline!r} is not one of the candidates")
    rates = {name: sum(v) / n for name, v in data.items()}

    if baseline is not None:
        pairs = [(name, baseline) for name in data if name != baseline]
    else:
        # Name the better scorer first so differences read as "how much better".
        pairs = [
            (a, b) if rates[a] >= rates[b] else (b, a) for a, b in combinations(list(data), 2)
        ]
    m = len(pairs)
    alpha = 1 - confidence
    pair_level = 1 - alpha / m
    raw = []
    for name, versus in pairs:
        only_name = sum(1 for x, y in zip(data[name], data[versus]) if x and not y)
        only_versus = sum(1 for x, y in zip(data[name], data[versus]) if y and not x)
        raw.append(compare_counts(n, only_versus, only_name, pair_level))
    adjusted = holm([r.p_value for r in raw])
    out = tuple(
        PairResult(
            name, versus, r.difference, r.low, r.high, r.p_value, adj,
            adj < alpha, r.p_value < alpha,
        )
        for (name, versus), r, adj in zip(pairs, raw, adjusted)
    )
    return ManyComparison(n, rates, out, baseline, confidence)
