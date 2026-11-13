"""How many eval examples do I need? Sample size planning for LLM evals."""

from .comparison import (
    Comparison,
    compare_counts,
    compare_results,
    examples_to_compare,
    power_to_compare,
    smallest_detectable_improvement,
)
from .multiple import ManyComparison, PairResult, compare_many, false_win_chance, holm
from .judge import CorrectedRate, JudgeAgreement, correct_judge_rate, judge_agreement, labels_to_check_judge
from .precision import PassRate, examples_for_margin, margin_for_examples, pass_rate_interval
from .reruns import effective_examples, question_share, questions_needed

__all__ = [
    "examples_for_margin",
    "margin_for_examples",
    "pass_rate_interval",
    "examples_to_compare",
    "power_to_compare",
    "smallest_detectable_improvement",
    "compare_counts",
    "compare_results",
    "effective_examples",
    "questions_needed",
    "question_share",
    "compare_many",
    "false_win_chance",
    "holm",
    "judge_agreement",
    "correct_judge_rate",
    "labels_to_check_judge",
    "PassRate",
    "Comparison",
    "JudgeAgreement",
    "CorrectedRate",
    "ManyComparison",
    "PairResult",
]

__version__ = "0.2.0"
