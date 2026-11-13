import random

import pytest

import evalsize as es
from evalsize.cli import main


def test_holm_known_values():
    assert es.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert es.holm([]) == []
    assert es.holm([0.9, 0.8]) == [1.0, 1.0]


def test_holm_rejects_bad_p_values():
    with pytest.raises(ValueError):
        es.holm([0.5, 1.2])


def test_false_win_chance():
    assert es.false_win_chance(1) == pytest.approx(0.05)
    assert es.false_win_chance(10) == pytest.approx(1 - 0.95 ** 10)
    with pytest.raises(ValueError):
        es.false_win_chance(0)


def test_more_comparisons_need_more_questions():
    one = es.examples_to_compare(0.7, 0.05, disagreement=0.15)
    assert es.examples_to_compare(0.7, 0.05, disagreement=0.15, comparisons=1) == one
    four = es.examples_to_compare(0.7, 0.05, disagreement=0.15, comparisons=4)
    ten = es.examples_to_compare(0.7, 0.05, disagreement=0.15, comparisons=10)
    assert one < four < ten


def test_comparisons_round_trip_through_power():
    n = es.examples_to_compare(0.7, 0.05, disagreement=0.15, comparisons=4)
    assert es.power_to_compare(n, 0.7, 0.05, disagreement=0.15, comparisons=4) == pytest.approx(0.8, abs=0.005)
    gain = es.smallest_detectable_improvement(n, 0.7, disagreement=0.15, comparisons=4)
    assert gain == pytest.approx(0.05, abs=0.002)


@pytest.mark.parametrize("comparisons", [0, -1, 2.5])
def test_bad_comparisons(comparisons):
    with pytest.raises(ValueError):
        es.examples_to_compare(0.7, 0.05, comparisons=comparisons)
    with pytest.raises(ValueError):
        es.smallest_detectable_improvement(300, 0.7, comparisons=comparisons)


def test_compare_many_against_baseline():
    results = {
        "current": [1, 1, 0, 0, 1, 0, 1, 0],
        "new": [1, 1, 1, 1, 1, 0, 1, 1],
        "other": [1, 0, 0, 0, 1, 0, 1, 0],
    }
    out = es.compare_many(results, baseline="current")
    assert out.comparisons == 2
    by_name = {r.name: r for r in out.results}
    assert by_name["new"].versus == "current"
    assert by_name["new"].difference == pytest.approx(3 / 8)
    assert by_name["other"].difference == pytest.approx(-1 / 8)
    assert out.pass_rates["current"] == 0.5
    assert all(r.adjusted_p >= r.p_value for r in out.results)


def test_compare_many_all_pairs_names_the_better_one_first():
    results = {"a": [0, 0, 1, 0], "b": [1, 1, 1, 0], "c": [1, 0, 1, 0]}
    out = es.compare_many(results)
    assert out.comparisons == 3
    assert all(r.difference >= 0 for r in out.results)
    assert ("b", "a") in [(r.name, r.versus) for r in out.results]


def test_compare_many_summary_flags_lucky_winners():
    rng = random.Random(0)
    base = [rng.random() < 0.7 for _ in range(400)]
    # 14 questions gained, 4 lost: p = 0.03 alone, but not after correcting for 5 comparisons.
    lucky = list(base)
    flips = [i for i, x in enumerate(base) if not x][:14]
    for i in flips:
        lucky[i] = True
    for i in [i for i, x in enumerate(base) if x][:4]:
        lucky[i] = False
    results = {"current": base, "lucky": lucky}
    for k in range(4):
        results[f"same{k}"] = list(base)
    out = es.compare_many(results, baseline="current")
    text = out.summary()
    lucky_row = next(r for r in out.results if r.name == "lucky")
    assert lucky_row.significant_uncorrected and not lucky_row.significant
    assert "possible lucky win" in text
    assert "None of the candidates is reliably better" in text


@pytest.mark.parametrize(
    "results, baseline",
    [
        ({"a": [1, 0]}, None),
        ({"a": [1, 0], "b": [1]}, None),
        ({"a": [1, 0], "b": [0, 1]}, "c"),
        ({"a": [], "b": []}, None),
    ],
)
def test_compare_many_bad_input(results, baseline):
    with pytest.raises(ValueError):
        es.compare_many(results, baseline=baseline)


# Simulation: does the correction keep its promise?

SIMS = 2000


def simulate_candidates(rng, n, baseline_rate, gains, disagreement):
    """Baseline results plus one candidate per gain, each disagreeing with the baseline on about
    ``disagreement`` of questions and better by ``gain`` on average."""
    base = [rng.random() < baseline_rate for _ in range(n)]
    data = {"base": base}
    for k, gain in enumerate(gains):
        only_new = (disagreement + gain) / 2
        only_base = (disagreement - gain) / 2
        # Conditional flip rates that give the requested disagreement pattern.
        p_up = only_new / (1 - baseline_rate)
        p_down = only_base / baseline_rate
        data[f"c{k}"] = [
            (not b and rng.random() < p_up) or (b and rng.random() >= p_down) for b in base
        ]
    return data


def test_correction_keeps_false_wins_at_five_percent():
    rng = random.Random(21)
    corrected = uncorrected = 0
    for _ in range(SIMS):
        data = simulate_candidates(rng, 200, 0.7, [0, 0, 0, 0], 0.12)
        out = es.compare_many(data, baseline="base")
        corrected += any(r.significant for r in out.results)
        uncorrected += any(r.significant_uncorrected for r in out.results)
    assert corrected / SIMS <= 0.055
    assert uncorrected / SIMS > 0.08
    assert uncorrected / SIMS <= es.false_win_chance(4) + 0.01


def test_planned_size_finds_the_real_winner():
    rng = random.Random(22)
    n = es.examples_to_compare(0.7, 0.05, disagreement=0.15, comparisons=4)
    found = 0
    for _ in range(SIMS):
        data = simulate_candidates(rng, n, 0.7, [0.05, 0, 0, 0], 0.15)
        out = es.compare_many(data, baseline="base")
        found += "c0" in out.winners
    assert found / SIMS >= 0.75


# Command line


def test_many_command(tmp_path, capsys):
    path = tmp_path / "results.csv"
    path.write_text("question,current,new\n1,1,1\n2,0,1\n3,pass,fail\n4,no,yes\n")
    assert main(["many", str(path), "--baseline", "current"]) == 0
    out = capsys.readouterr().out
    assert "new" in out and "current" in out


def test_many_command_reports_bad_cells(tmp_path, capsys):
    path = tmp_path / "results.csv"
    path.write_text("current,new\n1,maybe\n")
    assert main(["many", str(path)]) == 2
    assert "maybe" in capsys.readouterr().err


def test_compare_command_with_comparisons(capsys):
    assert main(["compare", "--baseline", "0.7", "--improvement", "0.05", "--disagreement", "0.15", "--comparisons", "4"]) == 0
    out = capsys.readouterr().out
    assert "667 questions" in out and "469" in out
