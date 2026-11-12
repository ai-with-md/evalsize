"""More questions, or more runs per question?

When a model is not deterministic, the same question can pass on one run and
fail on the next. Running each question several times averages out that
randomness, but it cannot fix the other source of noise: which questions you
happened to pick. This module measures how much reruns are worth.

The planning input is the **question share**: the part of the score's
variance that comes from questions being easy or hard, as opposed to run to
run randomness. A deterministic model has a question share of 1, and reruns
add nothing. Estimate it from a pilot with ``question_share``.
"""

from __future__ import annotations

import math
from typing import Sequence

from ._stats import check_rate


def effective_examples(n_questions: int, runs_per_question: int, question_share: float) -> float:
    """How many single-run questions your design is worth.

    With ``n`` questions and ``k`` runs each, the score is as precise as
    ``n / (s + (1 - s) / k)`` single-run questions, where ``s`` is the question
    share. No number of reruns can do better than ``n / s``.

    >>> round(effective_examples(200, 5, 0.6))
    294
    """
    if n_questions <= 0 or runs_per_question <= 0:
        raise ValueError("n_questions and runs_per_question must be positive")
    check_rate(question_share, "question_share", allow_edges=True)
    return n_questions / (question_share + (1 - question_share) / runs_per_question)


def questions_needed(single_run_examples: int, runs_per_question: int, question_share: float) -> int:
    """Questions needed, at ``runs_per_question`` runs each, to match a single-run plan.

    Use it with ``examples_for_margin`` or ``examples_to_compare``, which assume
    one run per question.
    """
    if single_run_examples <= 0 or runs_per_question <= 0:
        raise ValueError("single_run_examples and runs_per_question must be positive")
    check_rate(question_share, "question_share", allow_edges=True)
    factor = question_share + (1 - question_share) / runs_per_question
    return max(1, math.ceil(single_run_examples * factor - 1e-9))


def question_share(results: Sequence[Sequence[float]]) -> float:
    """Estimate the question share from a pilot where each question was run several times.

    ``results`` has one row per question and one value per run (1 for pass,
    0 for fail, or any score). Every question needs the same number of runs,
    at least two. Uses the one-way analysis of variance estimator of the
    intraclass correlation.
    """
    rows = [list(map(float, row)) for row in results]
    if len(rows) < 2:
        raise ValueError("need at least two questions")
    k = len(rows[0])
    if k < 2 or any(len(r) != k for r in rows):
        raise ValueError("every question needs the same number of runs, at least two")
    n = len(rows)
    means = [sum(r) / k for r in rows]
    grand = sum(means) / n
    ms_between = k * sum((m - grand) ** 2 for m in means) / (n - 1)
    ms_within = sum((x - m) ** 2 for r, m in zip(rows, means) for x in r) / (n * (k - 1))
    between = max(0.0, (ms_between - ms_within) / k)
    total = between + ms_within
    if total == 0:
        return 1.0
    return between / total
