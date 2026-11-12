"""Checks against known textbook answers and basic properties."""

import pytest

import evalsize as es


def test_classic_two_proportion_answer():
    # 50% vs 60%, alpha 0.05, power 0.8: the standard table answer is 388 per group.
    assert es.examples_to_compare(0.5, 0.1, paired=False) == 388


def test_wilson_interval_known_values():
    # 8 of 10 has a 95% Wilson interval of about 0.490 to 0.943.
    r = es.pass_rate_interval(8, 10)
    assert r.low == pytest.approx(0.4902, abs=1e-3)
    assert r.high == pytest.approx(0.9433, abs=1e-3)


def test_wilson_interval_handles_perfect_scores():
    r = es.pass_rate_interval(50, 50)
    assert r.high == 1.0
    assert 0.9 < r.low < 1.0


def test_margin_and_examples_agree():
    n = es.examples_for_margin(0.03, pass_rate=0.75)
    assert es.margin_for_examples(n, 0.75) <= 0.03
    assert es.margin_for_examples(n - 1, 0.75) > 0.03


def test_half_a_coin_is_the_cautious_case():
    assert es.examples_for_margin(0.05, 0.5) > es.examples_for_margin(0.05, 0.9)


def test_pairing_helps_only_when_models_agree():
    independent = es.examples_to_compare(0.7, 0.05)
    unpaired = es.examples_to_compare(0.7, 0.05, paired=False)
    agreeing = es.examples_to_compare(0.7, 0.05, disagreement=0.15)
    assert abs(independent - unpaired) / unpaired < 0.01
    assert agreeing < independent / 2


def test_smaller_improvements_need_more_questions():
    sizes = [es.examples_to_compare(0.7, gain, disagreement=0.2) for gain in (0.02, 0.05, 0.1)]
    assert sizes[0] > sizes[1] > sizes[2]


def test_improvement_can_be_negative():
    assert es.examples_to_compare(0.7, -0.05, disagreement=0.15) == es.examples_to_compare(
        0.65, 0.05, disagreement=0.15
    )


def test_power_matches_sample_size():
    n = es.examples_to_compare(0.7, 0.05, disagreement=0.15)
    assert es.power_to_compare(n, 0.7, 0.05, disagreement=0.15) == pytest.approx(0.8, abs=0.005)


def test_smallest_detectable_round_trip():
    gain = es.smallest_detectable_improvement(500, 0.7, disagreement=0.15)
    assert es.power_to_compare(500, 0.7, gain, disagreement=0.15) == pytest.approx(0.8, abs=0.001)


def test_smallest_detectable_can_be_out_of_reach():
    assert es.smallest_detectable_improvement(5, 0.9) is None


@pytest.mark.parametrize("disagreement", [0.01, 0.99])
def test_impossible_disagreement_is_explained(disagreement):
    with pytest.raises(ValueError, match="disagreement"):
        es.examples_to_compare(0.7, 0.05, disagreement=disagreement)


@pytest.mark.parametrize(
    "call",
    [
        lambda: es.examples_to_compare(0.7, 0),
        lambda: es.examples_to_compare(0.98, 0.05),
        lambda: es.examples_for_margin(0),
        lambda: es.pass_rate_interval(11, 10),
        lambda: es.compare_counts(10, 6, 6),
        lambda: es.effective_examples(0, 1, 0.5),
    ],
)
def test_bad_inputs_raise(call):
    with pytest.raises(ValueError):
        call()


def test_compare_counts_exact_p_value():
    # 0 of 5 disagreements going one way: exact two-sided p is 2 / 32.
    assert es.compare_counts(100, 5, 0).p_value == pytest.approx(0.0625)
    assert es.compare_counts(100, 3, 3).p_value == 1.0


def test_compare_results_counts_disagreements():
    a = [1, 1, 0, 0, 1, 0]
    b = [1, 0, 1, 1, 1, 0]
    result = es.compare_results(a, b)
    assert (result.n, result.only_a_right, result.only_b_right) == (6, 1, 2)
    assert "B minus A" in result.summary()


def test_effective_examples_limits():
    assert es.effective_examples(100, 10, 1.0) == 100  # deterministic model: reruns add nothing
    assert es.effective_examples(100, 4, 0.0) == 400  # pure run noise: every run counts
    assert es.effective_examples(100, 1000, 0.5) < 200  # never better than n / share


def test_questions_needed_inverts_effective_examples():
    q = es.questions_needed(500, 4, 0.5)
    assert es.effective_examples(q, 4, 0.5) >= 500


def test_question_share_on_obvious_cases():
    assert es.question_share([[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]]) == pytest.approx(1.0)
    assert es.question_share([[1, 0], [0, 1], [1, 0], [0, 1]]) == 0.0


def test_judge_agreement():
    human = [1, 1, 1, 1, 0, 0]
    judge = [1, 1, 1, 0, 1, 0]
    a = es.judge_agreement(human, judge)
    assert a.sensitivity == 0.75
    assert a.specificity == 0.5
    assert a.agreement == pytest.approx(4 / 6)


def test_perfect_judge_needs_no_correction():
    r = es.correct_judge_rate(0.8, 1000, 1.0, 1.0, 100, 100)
    assert r.corrected == pytest.approx(0.8)


def test_lenient_judge_is_corrected_down():
    r = es.correct_judge_rate(0.82, 1000, 0.95, 0.70, 140, 60)
    assert r.corrected == pytest.approx(0.80)
    assert r.low < r.corrected < r.high


def test_useless_judge_is_refused():
    with pytest.raises(ValueError, match="coin flip"):
        es.correct_judge_rate(0.6, 1000, 0.55, 0.48, 100, 100)


def test_fewer_human_labels_means_wider_interval():
    few = es.correct_judge_rate(0.8, 1000, 0.9, 0.8, 30, 10)
    many = es.correct_judge_rate(0.8, 1000, 0.9, 0.8, 300, 100)
    assert (few.high - few.low) > (many.high - many.low)


def test_labels_to_check_judge_is_driven_by_the_rarer_outcome():
    assert es.labels_to_check_judge(pass_rate=0.9) > es.labels_to_check_judge(pass_rate=0.5)
