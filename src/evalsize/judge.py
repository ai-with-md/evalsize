"""When an LLM judge grades the eval, its mistakes shift the score.

A judge that passes 10% of bad answers and fails 5% of good ones does not
just add noise: it pushes the score in one direction. If you check the judge
against human labels on a sample, you can correct the score and widen the
interval to account for what you learned from only a sample.

Terms used here, with humans as the reference:
    sensitivity: share of answers humans pass that the judge also passes.
    specificity: share of answers humans fail that the judge also fails.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from ._stats import check_level, check_rate, z_two_sided
from .precision import examples_for_margin


@dataclass(frozen=True)
class JudgeAgreement:
    sensitivity: float
    specificity: float
    human_passes: int
    human_fails: int
    agreement: float

    def summary(self) -> str:
        return (
            f"The judge agreed with humans on {self.agreement:.1%} of {self.human_passes + self.human_fails} labels. "
            f"It passed {self.sensitivity:.1%} of answers humans passed "
            f"and failed {self.specificity:.1%} of answers humans failed."
        )

    def __str__(self) -> str:
        return self.summary()


def judge_agreement(human: Iterable, judge: Iterable) -> JudgeAgreement:
    """Measure the judge against human labels for the same answers (True means pass)."""
    human, judge = [bool(x) for x in human], [bool(x) for x in judge]
    if len(human) != len(judge):
        raise ValueError("human and judge need one label per answer, in the same order")
    passes = [j for h, j in zip(human, judge) if h]
    fails = [j for h, j in zip(human, judge) if not h]
    if not passes or not fails:
        raise ValueError("the labelled sample needs at least one human pass and one human fail")
    sensitivity = sum(passes) / len(passes)
    specificity = sum(1 for j in fails if not j) / len(fails)
    agreement = sum(1 for h, j in zip(human, judge) if h == j) / len(human)
    return JudgeAgreement(sensitivity, specificity, len(passes), len(fails), agreement)


@dataclass(frozen=True)
class CorrectedRate:
    judge_rate: float
    corrected: float
    low: float
    high: float
    confidence: float

    def summary(self) -> str:
        return (
            f"The judge says {self.judge_rate:.1%}. Corrected for the judge's mistakes: "
            f"{self.corrected:.1%} ({self.confidence:.0%} interval {self.low:.1%} to {self.high:.1%})."
        )

    def __str__(self) -> str:
        return self.summary()


def correct_judge_rate(
    judge_rate: float,
    n_judged: int,
    sensitivity: float,
    specificity: float,
    human_passes: int,
    human_fails: int,
    confidence: float = 0.95,
) -> CorrectedRate:
    """Correct a judge-graded pass rate using the judge's measured error rates.

    Args:
        judge_rate: pass rate the judge gave over the whole eval.
        n_judged: answers the judge graded.
        sensitivity, specificity: from ``judge_agreement``.
        human_passes, human_fails: how many human labels each of those was
            measured on. Fewer labels means a wider interval.

    Uses the Rogan and Gladen (1978) estimator with a delta method interval
    that includes the uncertainty in all three measured rates.

    >>> r = correct_judge_rate(0.82, 1000, 0.95, 0.70, 140, 60)
    >>> round(r.corrected, 3)
    0.8
    """
    check_rate(judge_rate, "judge_rate", allow_edges=True)
    check_rate(sensitivity, "sensitivity", allow_edges=True)
    check_rate(specificity, "specificity", allow_edges=True)
    check_level(confidence, "confidence")
    if n_judged <= 0 or human_passes <= 0 or human_fails <= 0:
        raise ValueError("n_judged, human_passes and human_fails must be positive")
    youden = sensitivity + specificity - 1
    if youden <= 0.05:
        raise ValueError(
            "the judge is barely better than a coin flip against human labels "
            "(sensitivity + specificity must be well above 1), so its score cannot be corrected"
        )
    raw = (judge_rate + specificity - 1) / youden
    p = min(1.0, max(0.0, raw))
    var = (
        judge_rate * (1 - judge_rate) / n_judged
        + p ** 2 * sensitivity * (1 - sensitivity) / human_passes
        + (1 - p) ** 2 * specificity * (1 - specificity) / human_fails
    ) / youden ** 2
    half = z_two_sided(1 - confidence) * math.sqrt(var)
    return CorrectedRate(judge_rate, p, max(0.0, p - half), min(1.0, p + half), confidence)


def labels_to_check_judge(
    margin: float = 0.05,
    expected_sensitivity: float = 0.9,
    expected_specificity: float = 0.9,
    pass_rate: float = 0.7,
    confidence: float = 0.95,
) -> int:
    """Human labels needed to measure the judge's error rates to within ``margin``.

    Both error rates must be measured, and the rarer outcome (usually fails)
    decides the total: if 30% of answers fail, you need to label about three
    answers to find one fail.
    """
    check_rate(pass_rate, "pass_rate")
    need_passes = examples_for_margin(margin, expected_sensitivity, confidence)
    need_fails = examples_for_margin(margin, expected_specificity, confidence)
    return math.ceil(max(need_passes / pass_rate, need_fails / (1 - pass_rate)) - 1e-9)
