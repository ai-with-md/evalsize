"""Command line: ``evalsize compare``, ``evalsize margin``, ``evalsize detect`` and more."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import (
    compare_counts,
    correct_judge_rate,
    effective_examples,
    examples_for_margin,
    examples_to_compare,
    labels_to_check_judge,
    smallest_detectable_improvement,
)


def _pct(x: float) -> str:
    return f"{x:.1%}"


def _cmd_margin(args: argparse.Namespace) -> str:
    n = examples_for_margin(args.margin, args.rate, args.confidence)
    return (
        f"{n} examples give a score of about {_pct(args.rate)} "
        f"plus or minus {_pct(args.margin)} ({args.confidence:.0%} confidence)."
    )


def _cmd_compare(args: argparse.Namespace) -> str:
    paired = not args.unpaired
    n = examples_to_compare(args.baseline, args.improvement, paired, args.disagreement, args.alpha, args.power)
    target = args.baseline + args.improvement
    lines = [
        f"To detect {_pct(args.baseline)} -> {_pct(target)} with {args.power:.0%} power "
        f"(false alarm rate {args.alpha:g}):",
        f"  {n} questions" + (" per model" if not paired else ", both models on the same questions"),
    ]
    if paired and args.disagreement is None:
        lines.append(
            "  This assumes the models make mistakes independently. Most model pairs agree more,\n"
            "  so measure the disagreement rate on a pilot and pass --disagreement to get a smaller number."
        )
    return "\n".join(lines)


def _cmd_detect(args: argparse.Namespace) -> str:
    gain = smallest_detectable_improvement(
        args.n, args.baseline, not args.unpaired, args.disagreement, args.alpha, args.power
    )
    if gain is None:
        return f"With {args.n} questions no improvement over {_pct(args.baseline)} can be reliably detected."
    return (
        f"With {args.n} questions and a baseline of {_pct(args.baseline)}, the smallest improvement "
        f"you can reliably detect is {gain * 100:.1f} percentage points."
    )


def _cmd_diff(args: argparse.Namespace) -> str:
    return compare_counts(args.n, args.only_a, args.only_b, args.confidence).summary()


def _cmd_reruns(args: argparse.Namespace) -> str:
    lines = [f"{args.n} questions, question share {args.share:g}:"]
    for k in (1, 2, 3, 5, 10):
        lines.append(f"  {k:>2} run(s) each = worth {effective_examples(args.n, k, args.share):,.0f} single-run questions")
    lines.append(f"  No number of reruns can beat {args.n / args.share:,.0f}." if args.share > 0 else "")
    return "\n".join(line for line in lines if line)


def _cmd_judge(args: argparse.Namespace) -> str:
    if args.judge_rate is None:
        n = labels_to_check_judge(args.margin, args.sensitivity, args.specificity, args.pass_rate)
        return (
            f"Label about {n} answers by hand to measure the judge's error rates "
            f"to within {_pct(args.margin)}."
        )
    return correct_judge_rate(
        args.judge_rate, args.n, args.sensitivity, args.specificity, args.human_passes, args.human_fails
    ).summary()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalsize",
        description="How many eval examples do I need? Sample size planning for LLM evals.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("margin", help="examples needed to measure one score to within +/- margin")
    p.add_argument("--rate", type=float, default=0.5, help="expected pass rate (default 0.5, the cautious case)")
    p.add_argument("--margin", type=float, required=True, help="for example 0.03 for +/- 3 points")
    p.add_argument("--confidence", type=float, default=0.95)
    p.set_defaults(run=_cmd_margin)

    def comparison_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--baseline", type=float, required=True, help="model A's expected pass rate")
        p.add_argument("--disagreement", type=float, help="share of questions where exactly one model is right")
        p.add_argument("--unpaired", action="store_true", help="the two models answer different questions")
        p.add_argument("--alpha", type=float, default=0.05)
        p.add_argument("--power", type=float, default=0.8)

    p = sub.add_parser("compare", help="questions needed to detect that B beats A")
    comparison_options(p)
    p.add_argument("--improvement", type=float, required=True, help="smallest gain worth detecting, e.g. 0.05")
    p.set_defaults(run=_cmd_compare)

    p = sub.add_parser("detect", help="smallest improvement a given number of questions can detect")
    comparison_options(p)
    p.add_argument("--n", type=int, required=True)
    p.set_defaults(run=_cmd_detect)

    p = sub.add_parser("diff", help="is B really better than A? (from disagreement counts)")
    p.add_argument("--n", type=int, required=True, help="questions both models answered")
    p.add_argument("--only-a", type=int, required=True, help="questions only A got right")
    p.add_argument("--only-b", type=int, required=True, help="questions only B got right")
    p.add_argument("--confidence", type=float, default=0.95)
    p.set_defaults(run=_cmd_diff)

    p = sub.add_parser("reruns", help="what running each question several times is worth")
    p.add_argument("--n", type=int, required=True, help="number of questions")
    p.add_argument("--share", type=float, required=True, help="question share of the variance (0 to 1)")
    p.set_defaults(run=_cmd_reruns)

    p = sub.add_parser("judge", help="human labels needed for an LLM judge, or correct its score")
    p.add_argument("--sensitivity", type=float, default=0.9)
    p.add_argument("--specificity", type=float, default=0.9)
    p.add_argument("--pass-rate", type=float, default=0.7, help="expected pass rate, for planning labels")
    p.add_argument("--margin", type=float, default=0.05, help="precision wanted for the error rates")
    p.add_argument("--judge-rate", type=float, help="the judge's pass rate, to correct it")
    p.add_argument("--n", type=int, default=1000, help="answers the judge graded")
    p.add_argument("--human-passes", type=int, default=100)
    p.add_argument("--human-fails", type=int, default=50)
    p.set_defaults(run=_cmd_judge)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        print(args.run(args))
    except ValueError as exc:
        print(f"evalsize: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
