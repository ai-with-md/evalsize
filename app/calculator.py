import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="evalsize: how many eval examples do I need?",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
async def _(mo):
    import sys

    if sys.platform == "emscripten":
        # In the browser, install the library from the wheel published with this page.
        # wheel.txt holds the wheel's file name, so a new version needs no change here.
        import micropip
        from pyodide.http import pyfetch

        _public = mo.notebook_location() / "public"
        _wheel = (await (await pyfetch(str(_public / "wheel.txt"))).string()).strip()
        await micropip.install(str(_public / _wheel))
    import evalsize as es
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (7, 3.2),
            "figure.dpi": 110,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )
    INK, ACCENT, MUTED = "#1f2937", "#6d28d9", "#9ca3af"
    return ACCENT, INK, MUTED, es, plt


@app.cell
def _(mo):
    mo.md("""
    # How many eval examples do I need?

    Planning numbers for LLM evals. Move the sliders to match
    your situation. Everything runs in your browser, so nothing is sent anywhere. The first
    load takes a few seconds while Python starts up.

    1. [How many questions to compare two models?](#compare)
    2. [How precise is one score?](#precision)
    3. [Is B really better than A?](#diff)
    4. [More questions, or more runs per question?](#reruns)
    5. [Can I trust my LLM judge?](#judge)
    6. [Testing several prompts at once?](#many)

    The same calculations are in the Python library: `pip install evalsize`.
    """)
    return


@app.cell
def _(mo):
    cmp_baseline = mo.ui.slider(0.30, 0.95, step=0.01, value=0.70, label="Model A's pass rate", show_value=True)
    cmp_gain = mo.ui.slider(0.01, 0.20, step=0.01, value=0.05, label="Smallest gain worth detecting", show_value=True)
    cmp_shared = mo.ui.switch(value=True, label="Both models answer the same questions")
    cmp_disagree = mo.ui.slider(
        0.02, 0.60, step=0.01, value=0.15, label="Share of questions where they disagree", show_value=True
    )
    cmp_power = mo.ui.dropdown({"80%": 0.8, "90%": 0.9}, value="80%", label="Power")
    return cmp_baseline, cmp_disagree, cmp_gain, cmp_power, cmp_shared


@app.cell
def _(
    ACCENT,
    INK,
    MUTED,
    cmp_baseline,
    cmp_disagree,
    cmp_gain,
    cmp_power,
    cmp_shared,
    es,
    mo,
    plt,
):
    _base, _gain, _psi, _power = cmp_baseline.value, cmp_gain.value, cmp_disagree.value, cmp_power.value
    _controls = mo.vstack(
        [mo.hstack([cmp_baseline, cmp_gain], wrap=True), mo.hstack([cmp_shared, cmp_disagree, cmp_power], wrap=True)]
    )
    _header = mo.md(
        """
        <h2 id="compare">1. How many questions to compare two models?</h2>

        You want to know whether model B (or a new prompt) beats model A. When both answer the
        same questions, only the questions where they **disagree** carry information, so the
        disagreement rate matters more than anything else. Measure it on a small pilot.
        """
    )
    if _base + _gain >= 1:
        _body = mo.callout(mo.md("Model A's pass rate plus the gain must stay below 100%."), kind="warn")
    else:
        _independent = es.examples_to_compare(_base, _gain, power=_power)
        _separate = es.examples_to_compare(_base, _gain, paired=False, power=_power)
        try:
            _yours = es.examples_to_compare(_base, _gain, disagreement=_psi, power=_power)
            _note = None
        except ValueError as _exc:
            _yours, _note = None, str(_exc)
        if not cmp_shared.value:
            _stats = mo.hstack(
                [mo.stat(f"{_separate:,}", label="Questions per model", caption="separate question sets", bordered=True)],
                justify="start", wrap=True)
        else:
            _stats = mo.hstack(
                [
                    mo.stat(
                        f"{_yours:,}" if _yours else "n/a",
                        label="Questions needed",
                        caption=f"with {_psi:.0%} disagreement",
                        bordered=True,
                    ),
                    mo.stat(
                        f"{_independent:,}",
                        label="If mistakes were independent",
                        caption="the cautious default",
                        bordered=True,
                    ),
                    mo.stat(f"{_separate:,}", label="Per model, unshared", caption="different questions", bordered=True),
                ],
                justify="start", wrap=True)
        _gains = [g / 100 for g in range(1, 16) if _base + g / 100 < 1]
        _fig, _ax = plt.subplots()
        _ax.plot([g * 100 for g in _gains], [es.examples_to_compare(_base, g, paired=False, power=_power) for g in _gains], color=MUTED, label="different questions")
        _ax.plot([g * 100 for g in _gains], [es.examples_to_compare(_base, g, power=_power) for g in _gains], color=INK, linestyle="--", label="same questions, independent mistakes")
        _fit = [g for g in _gains if g <= _psi]
        if _fit:
            _ax.plot([g * 100 for g in _fit], [es.examples_to_compare(_base, g, disagreement=_psi, power=_power) for g in _fit], color=ACCENT, linewidth=2.2, label=f"same questions, {_psi:.0%} disagreement")
        _ax.set_yscale("log")
        _ax.set_xlabel("Smallest gain worth detecting (percentage points)")
        _ax.set_ylabel("Questions needed")
        _ax.legend(frameon=False, fontsize=8)
        _fig.tight_layout()
        _parts = [_stats]
        if _note and cmp_shared.value:
            _parts.append(mo.callout(mo.md(_note), kind="warn"))
        _parts.append(_fig)
        _body = mo.vstack(_parts)
    mo.vstack([_header, _controls, _body])
    return


@app.cell
def _(mo):
    prec_rate = mo.ui.slider(0.50, 0.99, step=0.01, value=0.80, label="Expected pass rate", show_value=True)
    prec_margin = mo.ui.slider(0.01, 0.10, step=0.005, value=0.03, label="Plus or minus", show_value=True)
    prec_n = mo.ui.number(10, 100000, step=10, value=200, label="Questions you have now")
    return prec_margin, prec_n, prec_rate


@app.cell
def _(ACCENT, es, mo, plt, prec_margin, prec_n, prec_rate):
    _need = es.examples_for_margin(prec_margin.value, prec_rate.value)
    _have = es.margin_for_examples(int(prec_n.value), prec_rate.value)
    _ns = [25 * 2 ** (i / 4) for i in range(0, 29)]
    _fig, _ax = plt.subplots()
    _ax.plot(_ns, [100 * es.margin_for_examples(int(n), prec_rate.value) for n in _ns], color=ACCENT, linewidth=2)
    _ax.scatter([prec_n.value], [100 * _have], color=ACCENT, zorder=3)
    _ax.set_xscale("log")
    _ax.set_xlabel("Number of questions")
    _ax.set_ylabel("Plus or minus (points)")
    _fig.tight_layout()
    mo.vstack(
        [
            mo.md(
                """
                <h2 id="precision">2. How precise is one score?</h2>

                A score of 80% on 200 questions could easily be 74% or 85%. Here is how many
                questions you need for the precision you want, and how precise your current set is.
                """
            ),
            mo.hstack([prec_rate, prec_margin, prec_n], wrap=True),
            mo.hstack(
                [
                    mo.stat(f"{_need:,}", label="Questions needed", caption=f"for plus or minus {prec_margin.value:.1%}", bordered=True),
                    mo.stat(f"±{_have:.1%}", label="Your current precision", caption=f"with {int(prec_n.value):,} questions", bordered=True),
                ],
                justify="start", wrap=True),
            _fig,
        ]
    )
    return


@app.cell
def _(mo):
    diff_n = mo.ui.number(1, 1000000, value=300, label="Questions both models answered")
    diff_a = mo.ui.number(0, 1000000, value=12, label="Only A got right")
    diff_b = mo.ui.number(0, 1000000, value=28, label="Only B got right")
    return diff_a, diff_b, diff_n


@app.cell
def _(diff_a, diff_b, diff_n, es, mo):
    try:
        _r = es.compare_counts(int(diff_n.value), int(diff_a.value), int(diff_b.value))
        _significant = _r.p_value < 0.05
        _result = mo.vstack(
            [
                mo.hstack(
                    [
                        mo.stat(f"{_r.difference:+.1%}", label="B minus A", bordered=True),
                        mo.stat(f"{_r.low:+.1%} to {_r.high:+.1%}", label="95% interval", bordered=True),
                        mo.stat(f"{_r.p_value:.3g}", label="p-value (exact McNemar)", bordered=True),
                    ],
                    justify="start", wrap=True),
                mo.callout(mo.md(_r.summary()), kind="success" if _significant else "neutral"),
            ]
        )
    except ValueError as _exc:
        _result = mo.callout(mo.md(str(_exc)), kind="warn")
    mo.vstack(
        [
            mo.md(
                """
                <h2 id="diff">3. Is B really better than A?</h2>

                You ran both models on the same questions. Count the questions where exactly one of
                them was right. Questions both got right, or both got wrong, say nothing about which
                is better.
                """
            ),
            mo.hstack([diff_n, diff_a, diff_b], justify="start", wrap=True),
            _result,
        ]
    )
    return


@app.cell
def _(mo):
    rr_n = mo.ui.slider(50, 2000, step=50, value=200, label="Questions", show_value=True)
    rr_share = mo.ui.slider(0.05, 1.0, step=0.05, value=0.6, label="Question share", show_value=True)
    rr_runs = mo.ui.slider(1, 10, step=1, value=3, label="Runs per question", show_value=True)
    return rr_n, rr_runs, rr_share


@app.cell
def _(ACCENT, MUTED, es, mo, plt, rr_n, rr_runs, rr_share):
    _n, _s, _k = rr_n.value, rr_share.value, rr_runs.value
    _worth = es.effective_examples(_n, _k, _s)
    _ks = list(range(1, 11))
    _fig, _ax = plt.subplots()
    _ax.plot(_ks, [es.effective_examples(_n, k, _s) for k in _ks], color=ACCENT, linewidth=2, marker="o")
    _ax.axhline(_n / _s, color=MUTED, linestyle="--")
    _ax.text(10, _n / _s, " ceiling", va="center", color=MUTED, fontsize=8)
    _ax.set_xlabel("Runs per question")
    _ax.set_ylabel("Worth this many single-run questions")
    _fig.tight_layout()
    mo.vstack(
        [
            mo.md(
                """
                <h2 id="reruns">4. More questions, or more runs per question?</h2>

                When a model gives different answers on different runs, rerunning each question
                averages out that randomness. It cannot fix the other noise: which questions you
                picked. The **question share** is how much of the noise comes from the questions.
                A model that always gives the same answer has a share of 1, and reruns add nothing.
                Estimate it from a pilot with `evalsize.question_share`.
                """
            ),
            mo.hstack([rr_n, rr_share, rr_runs], wrap=True),
            mo.hstack(
                [
                    mo.stat(f"{_worth:,.0f}", label="Worth this many single-run questions", caption=f"{_n} questions x {_k} runs = {_n * _k:,} calls", bordered=True),
                    mo.stat(f"{_n / _s:,.0f}", label="Best possible with reruns", caption="adding questions is the only way past this", bordered=True),
                ],
                justify="start", wrap=True),
            _fig,
        ]
    )
    return


@app.cell
def _(mo):
    jd_rate = mo.ui.slider(0.30, 0.98, step=0.01, value=0.82, label="Judge's pass rate", show_value=True)
    jd_n = mo.ui.number(10, 1000000, value=1000, label="Answers the judge graded")
    jd_sens = mo.ui.slider(0.50, 1.0, step=0.01, value=0.95, label="Judge passes what humans pass", show_value=True)
    jd_spec = mo.ui.slider(0.50, 1.0, step=0.01, value=0.70, label="Judge fails what humans fail", show_value=True)
    jd_hp = mo.ui.number(1, 100000, value=140, label="Human-labelled passes")
    jd_hf = mo.ui.number(1, 100000, value=60, label="Human-labelled fails")
    return jd_hf, jd_hp, jd_n, jd_rate, jd_sens, jd_spec


@app.cell
def _(es, jd_hf, jd_hp, jd_n, jd_rate, jd_sens, jd_spec, mo):
    try:
        _r = es.correct_judge_rate(jd_rate.value, int(jd_n.value), jd_sens.value, jd_spec.value, int(jd_hp.value), int(jd_hf.value))
        _result = mo.vstack(
            [
                mo.hstack(
                    [
                        mo.stat(f"{_r.judge_rate:.1%}", label="What the judge says", bordered=True),
                        mo.stat(f"{_r.corrected:.1%}", label="Corrected pass rate", caption=f"95% interval {_r.low:.1%} to {_r.high:.1%}", bordered=True),
                    ],
                    justify="start", wrap=True),
                mo.callout(mo.md(_r.summary()), kind="info"),
            ]
        )
    except ValueError as _exc:
        _result = mo.callout(mo.md(str(_exc)), kind="warn")
    _labels = es.labels_to_check_judge(0.05, jd_sens.value if jd_sens.value < 1 else 0.99, jd_spec.value if jd_spec.value < 1 else 0.99, min(max(jd_rate.value, 0.05), 0.95))
    mo.vstack(
        [
            mo.md(
                """
                <h2 id="judge">5. Can I trust my LLM judge?</h2>

                A judge that is too lenient or too strict does not just add noise: it moves the score.
                Label a random sample of answers by hand, compare with the judge
                (`evalsize.judge_agreement`), and correct the score.
                """
            ),
            mo.hstack([jd_rate, jd_n], justify="start", wrap=True),
            mo.hstack([jd_sens, jd_spec], wrap=True),
            mo.hstack([jd_hp, jd_hf], justify="start", wrap=True),
            _result,
            mo.md(
                f"To measure both judge error rates to within 5 points, label about **{_labels:,}** answers at random."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mp_k = mo.ui.slider(1, 10, step=1, value=4, label="New prompts to test", show_value=True)
    mp_baseline = mo.ui.slider(0.30, 0.95, step=0.01, value=0.70, label="Current prompt's pass rate", show_value=True)
    mp_gain = mo.ui.slider(0.01, 0.20, step=0.01, value=0.05, label="Smallest gain worth detecting", show_value=True)
    mp_disagree = mo.ui.slider(
        0.02, 0.60, step=0.01, value=0.15, label="Share of questions where they disagree", show_value=True
    )
    return mp_baseline, mp_disagree, mp_gain, mp_k


@app.cell
def _(ACCENT, MUTED, es, mo, mp_baseline, mp_disagree, mp_gain, mp_k, plt):
    _k, _base, _gain, _psi = mp_k.value, mp_baseline.value, mp_gain.value, mp_disagree.value
    _header = mo.md(
        """
        <h2 id="many">6. Testing several prompts at once?</h2>

        Try five prompts that are all exactly as good as the current one, and there is still a fair
        chance one of them looks significantly better by luck. Pick it and you ship a change that
        does nothing. Correcting for the number of comparisons keeps the false alarm rate at 5%
        across all of them, and needs a few more questions. To analyze results, run
        `evalsize many results.csv --baseline current`.
        """
    )
    _controls = mo.vstack([mo.hstack([mp_k, mp_baseline], wrap=True), mo.hstack([mp_gain, mp_disagree], wrap=True)])
    if _base + _gain >= 1:
        _body = mo.callout(mo.md("The current pass rate plus the gain must stay below 100%."), kind="warn")
    else:
        try:
            _single = es.examples_to_compare(_base, _gain, disagreement=_psi)
            _many = es.examples_to_compare(_base, _gain, disagreement=_psi, comparisons=_k)
            _ks = list(range(1, 11))
            _fig, _ax = plt.subplots()
            _ax.plot(_ks, [es.examples_to_compare(_base, _gain, disagreement=_psi, comparisons=k) for k in _ks], color=ACCENT, linewidth=2, marker="o")
            _ax.axhline(_single, color=MUTED, linestyle="--")
            _ax.text(10, _single, " one prompt", va="bottom", ha="right", color=MUTED, fontsize=8)
            _ax.set_xlabel("New prompts tested against the current one")
            _ax.set_ylabel("Questions needed")
            _fig.tight_layout()
            _body = mo.vstack(
                [
                    mo.hstack(
                        [
                            mo.stat(f"{_many:,}", label="Questions needed", caption=f"testing {_k} prompt{'s' if _k > 1 else ''}, corrected", bordered=True),
                            mo.stat(f"{_single:,}", label="For a single prompt", caption="no correction needed", bordered=True),
                            mo.stat(f"{es.false_win_chance(_k):.0%}", label="Lucky winner risk", caption="at most, if you skip the correction", bordered=True),
                        ],
                        justify="start",
                        wrap=True,
                    ),
                    _fig,
                ]
            )
        except ValueError as _exc:
            _body = mo.callout(mo.md(str(_exc)), kind="warn")
    mo.vstack([_header, _controls, _body])
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    Built with [evalsize](https://github.com/ai-with-md/evalsize), MIT licensed. The methods:
    Wilson intervals, the McNemar sample size formula (Connor, 1987), the exact McNemar test with
    the Agresti and Min (2005) interval, the intraclass correlation for reruns, the Rogan and
    Gladen (1978) correction for imperfect judges, and the Holm and Bonferroni corrections for
    testing several prompts. Every formula is checked against simulation in the test suite.
    """)
    return


if __name__ == "__main__":
    app.run()
