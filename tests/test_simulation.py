"""Check the formulas against simulated evals.

Each test runs a few thousand fake evals with known true pass rates and checks
that the numbers evalsize promises (power, interval coverage, precision) show
up in practice. Seeds are fixed so the tests are repeatable.
"""

import math
import random

import pytest

import evalsize as es

SIMS = 2000


def binomial(rng, n, p):
    return sum(1 for _ in range(n) if rng.random() < p)


def test_wilson_interval_covers_the_truth_95_percent_of_the_time():
    rng = random.Random(1)
    truth, n = 0.85, 120
    hits = 0
    for _ in range(SIMS):
        r = es.pass_rate_interval(binomial(rng, n, truth), n)
        hits += r.low <= truth <= r.high
    assert 0.93 <= hits / SIMS <= 0.97


def simulate_paired(rng, n, baseline, improvement, disagreement):
    """One eval of two models on the same n questions with a set disagreement rate."""
    only_b = (disagreement + improvement) / 2
    only_a = (disagreement - improvement) / 2
    a_count = b_count = 0
    for _ in range(n):
        u = rng.random()
        if u < only_a:
            a_count += 1
        elif u < only_a + only_b:
            b_count += 1
    return a_count, b_count


def test_paired_sample_size_delivers_the_promised_power():
    rng = random.Random(2)
    n = es.examples_to_compare(0.7, 0.05, disagreement=0.15)
    detected = 0
    for _ in range(SIMS):
        only_a, only_b = simulate_paired(rng, n, 0.7, 0.05, 0.15)
        z = (only_b - only_a) / math.sqrt(only_a + only_b)
        detected += abs(z) > 1.96
    assert 0.77 <= detected / SIMS <= 0.83


def test_unpaired_sample_size_delivers_the_promised_power():
    rng = random.Random(3)
    n = es.examples_to_compare(0.6, 0.08, paired=False)
    detected = 0
    for _ in range(SIMS):
        a, b = binomial(rng, n, 0.6), binomial(rng, n, 0.68)
        pooled = (a + b) / (2 * n)
        z = (b - a) / n / math.sqrt(2 * pooled * (1 - pooled) / n)
        detected += abs(z) > 1.96
    assert 0.77 <= detected / SIMS <= 0.83


def test_compare_counts_keeps_false_alarms_under_five_percent():
    rng = random.Random(4)
    alarms = 0
    for _ in range(SIMS):
        only_a, only_b = simulate_paired(rng, 300, 0.7, 0.0, 0.12)
        alarms += es.compare_counts(300, only_a, only_b).p_value < 0.05
    assert alarms / SIMS <= 0.06


def test_compare_counts_interval_covers_the_true_difference():
    rng = random.Random(5)
    hits = 0
    for _ in range(SIMS):
        only_a, only_b = simulate_paired(rng, 400, 0.7, 0.04, 0.14)
        r = es.compare_counts(400, only_a, only_b)
        hits += r.low <= 0.04 <= r.high
    assert 0.93 <= hits / SIMS <= 0.97


def simulate_reruns(rng, n_questions, runs, a, b):
    """Question difficulty from Beta(a, b), then independent pass/fail runs."""
    rows = []
    for _ in range(n_questions):
        q = rng.betavariate(a, b)
        rows.append([1.0 if rng.random() < q else 0.0 for _ in range(runs)])
    return rows


def test_effective_examples_matches_simulated_precision():
    rng = random.Random(6)
    a, b = 1.4, 0.6  # mean 0.7, question share 1 / (a + b + 1) = 1 / 3
    share, p = 1 / (a + b + 1), a / (a + b)
    n, k = 100, 4
    scores = []
    for _ in range(SIMS):
        rows = simulate_reruns(rng, n, k, a, b)
        scores.append(sum(map(sum, rows)) / (n * k))
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)
    predicted = p * (1 - p) / es.effective_examples(n, k, share)
    assert variance == pytest.approx(predicted, rel=0.1)


def test_question_share_recovers_the_truth():
    rng = random.Random(7)
    rows = simulate_reruns(rng, 3000, 3, 1.4, 0.6)
    assert es.question_share(rows) == pytest.approx(1 / 3, abs=0.04)


def test_judge_correction_interval_covers_the_true_rate():
    rng = random.Random(8)
    truth, sens, spec = 0.75, 0.9, 0.8
    n_judged, human_passes, human_fails = 1000, 150, 50
    hits = 0
    for _ in range(SIMS):
        judged = 0
        for _ in range(n_judged):
            good = rng.random() < truth
            judged += rng.random() < (sens if good else 1 - spec)
        est_sens = binomial(rng, human_passes, sens) / human_passes
        est_spec = binomial(rng, human_fails, spec) / human_fails
        r = es.correct_judge_rate(judged / n_judged, n_judged, est_sens, est_spec, human_passes, human_fails)
        hits += r.low <= truth <= r.high
    assert 0.92 <= hits / SIMS <= 0.975
