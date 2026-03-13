"""
significance_test.py
───────────────────────────────────────────────────────────────────────────────
Computes descriptive statistics and two-sample hypothesis tests comparing
active-day counts between the Counter and Journal conditions.

Statistical approach
--------------------
Two complementary tests are run to provide converging evidence — standard
practice when n is small and normality cannot be assumed:

  1. Welch two-sample t-test
       Assumes interval-level data; does NOT assume equal variances
       (Welch–Satterthwaite correction applied to degrees of freedom).
       Reports: t(df), p, mean difference, 95% CI of the mean difference.

  2. Mann–Whitney U test (two-sided)
       Non-parametric alternative; makes no distributional assumptions.
       Exact method requested where scipy version supports it (≥ 1.7);
       falls back to asymptotic approximation for older installs.
       Reports: U, p.

Effect sizes
------------
  Hedges' g   — standardised mean difference with small-sample correction (J),
                 appropriate when group sizes differ.
                 Rule of thumb: |g| < 0.5 small, 0.5–0.8 medium, > 0.8 large.

  CLES         — Common-Language Effect Size P(Counter > Journal):
                 probability that a randomly selected Counter participant
                 has more active days than a randomly selected Journal
                 participant.  Ties counted as 0.5.

Usage
-----
    python significance_test.py

Edit the ``counter`` and ``journal`` arrays in ``main()`` to match your data.

Reproducibility note
--------------------
No randomness; all statistics are deterministic.
scipy's exact Mann–Whitney uses a complete enumeration, not permutation.

Requirements
------------
    python  >= 3.9
    numpy   >= 1.21
    scipy   >= 1.7
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np
from scipy import stats


# ── Descriptive statistics ────────────────────────────────────────────────────

def describe(x: Iterable[float]) -> dict:
    """
    Compute descriptive statistics for a 1-D array of values.

    Returns the statistics typically reported alongside inferential tests
    in CHI papers: n, mean, median, SD (sample), min, Q1, Q3, IQR, max.

    Parameters
    ----------
    x : iterable of float
        Observed values for one group.

    Returns
    -------
    dict with keys:
        n, values (np.ndarray), mean, median, sd, min, q1, q3, iqr, max
    """
    a      = np.asarray(list(x), dtype=float)
    n      = a.size
    mean   = float(np.mean(a))
    median = float(np.median(a))
    sd     = float(np.std(a, ddof=1)) if n > 1 else float("nan")   # sample SD
    mn     = float(np.min(a))
    q1     = float(np.quantile(a, 0.25, method="linear"))
    q3     = float(np.quantile(a, 0.75, method="linear"))
    iqr    = q3 - q1
    mx     = float(np.max(a))

    return {
        "n": n, "values": a, "mean": mean, "median": median,
        "sd": sd, "min": mn, "q1": q1, "q3": q3, "iqr": iqr, "max": mx,
    }


# ── Inferential helpers ───────────────────────────────────────────────────────

def welch_df(s1: float, n1: int, s2: float, n2: int) -> float:
    """
    Welch–Satterthwaite effective degrees of freedom.

    Used to correct for unequal variances and unequal group sizes in the
    two-sample t-test. Formula follows Welch (1947).

    Parameters
    ----------
    s1, s2 : float
        Sample standard deviations for groups 1 and 2.
    n1, n2 : int
        Sample sizes for groups 1 and 2.

    Returns
    -------
    float
        Effective degrees of freedom (may be non-integer).
    """
    v1  = (s1**2) / n1
    v2  = (s2**2) / n2
    num = (v1 + v2) ** 2
    den = (v1**2) / (n1 - 1) + (v2**2) / (n2 - 1)
    return num / den


def mean_diff_ci(
    m1: float, s1: float, n1: int,
    m2: float, s2: float, n2: int,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Compute the (1 − alpha) confidence interval for the mean difference.

    Uses Welch–Satterthwaite df so the interval is valid under unequal
    variances. Returns a (lower, upper) tuple for ``m1 − m2``.

    Parameters
    ----------
    m1, m2 : float    Group means.
    s1, s2 : float    Sample SDs.
    n1, n2 : int      Sample sizes.
    alpha  : float    Significance level (default 0.05 → 95% CI).

    Returns
    -------
    (float, float)  Lower and upper bounds of the CI.
    """
    diff  = m1 - m2
    se    = math.sqrt((s1**2) / n1 + (s2**2) / n2)
    df    = welch_df(s1, n1, s2, n2)
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    return diff - tcrit * se, diff + tcrit * se


# ── Effect sizes ──────────────────────────────────────────────────────────────

def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Hedges' g: a bias-corrected standardised mean difference.

    The small-sample correction factor J is applied (Hedges, 1981), making
    this preferable to Cohen's d when group sizes are small or unequal — both
    common in CHI studies.

    Interpretation (Cohen's benchmarks, commonly adopted for g):
        |g| < 0.5  small,  0.5–0.8  medium,  > 0.8  large.

    Parameters
    ----------
    x, y : np.ndarray
        Observed values for the two groups (Counter and Journal).

    Returns
    -------
    float
        Hedges' g (positive when mean(x) > mean(y)), or NaN if pooled SD = 0.
    """
    n1, n2 = len(x), len(y)
    s1 = np.std(x, ddof=1)
    s2 = np.std(y, ddof=1)

    # Pooled SD (assumes approximately equal population variances for effect size)
    sp = math.sqrt(((n1 - 1) * (s1**2) + (n2 - 1) * (s2**2)) / (n1 + n2 - 2))
    if sp == 0:
        return float("nan")

    d  = (np.mean(x) - np.mean(y)) / sp
    df = n1 + n2 - 2
    J  = 1 - (3 / (4 * df - 1))    # small-sample correction factor (Hedges, 1981)
    return float(J * d)


def common_language_effect_size(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the Common-Language Effect Size (CLES): P(X > Y).

    CLES is the probability that a randomly chosen participant from group X
    has a higher score than a randomly chosen participant from group Y.
    Ties are counted as 0.5 (continuity-corrected). This is a scale-free
    measure easily interpretable by non-statistician readers.

    For active-day counts: CLES = 0.80 means there is an 80% chance that a
    randomly selected Counter participant has more active days than a
    randomly selected Journal participant.

    Parameters
    ----------
    x, y : np.ndarray
        Observed values for groups X and Y respectively.

    Returns
    -------
    float in [0, 1].
    """
    wins  = 0.0
    total = len(x) * len(y)
    for xi in x:
        for yj in y:
            if xi > yj:
                wins += 1.0
            elif xi == yj:
                wins += 0.5      # split credit for ties
    return float(wins / total)


# ── Reporting helpers ─────────────────────────────────────────────────────────

def print_group(name: str, desc: dict) -> None:
    """
    Print full descriptive statistics for one group (CHI supplemental style).

    Parameters
    ----------
    name : str    Condition label.
    desc : dict   Output of ``describe()``.
    """
    vals = ", ".join(
        str(int(v)) if float(v).is_integer() else str(v)
        for v in desc["values"]
    )
    print(f"{name} (active days per participant)")
    print(f"n = {desc['n']}")
    print(f"values = {vals}")       # raw values for manual verification
    print(f"mean = {desc['mean']:.2f}")
    print(f"median = {desc['median']:.2f}")
    print(f"sd = {desc['sd']:.2f}")
    print(f"min = {desc['min']:.2f}")
    print(f"q1 (25%) = {desc['q1']:.2f}")
    print(f"q3 (75%) = {desc['q3']:.2f}")
    print(f"iqr = {desc['iqr']:.2f}")
    print(f"max = {desc['max']:.2f}")
    print()


# ── Main routine ──────────────────────────────────────────────────────────────

def main() -> None:
    """
    Run all analyses and print results to stdout.

    Data entry
    ----------
    Edit the ``counter`` and ``journal`` arrays below with the active-day
    counts produced by ``make_boxplot_active_days.py`` (one value per
    participant, same order as the printed 'values' line).

    Output order
    ------------
    1. Descriptive statistics (both groups)
    2. Welch t-test + 95% CI of mean difference
    3. Mann–Whitney U test
    4. Effect sizes (Hedges' g, CLES)
    """
    # ── Data — replace with values from your study ────────────────────────────
    counter = np.array([7, 6, 7, 6, 6, 7, 6, 7, 3, 5], dtype=float)
    journal = np.array([4, 3, 3, 1, 6, 2, 4, 4],       dtype=float)

    # ── 1. Descriptives ───────────────────────────────────────────────────────
    d_counter = describe(counter)
    d_journal = describe(journal)

    print_group("Counter group", d_counter)
    print_group("Journal group", d_journal)

    # ── 2. Welch two-sample t-test ────────────────────────────────────────────
    t_res  = stats.ttest_ind(counter, journal, equal_var=False)
    s1, s2 = d_counter["sd"], d_journal["sd"]
    n1, n2 = d_counter["n"],  d_journal["n"]
    df     = welch_df(s1, n1, s2, n2)
    diff   = d_counter["mean"] - d_journal["mean"]
    ci_lo, ci_hi = mean_diff_ci(d_counter["mean"], s1, n1,
                                 d_journal["mean"], s2, n2)

    print("Welch two-sample t-test")
    print(f"mean difference (Counter − Journal) = {diff:.2f} days")
    print(f"t({df:.2f}) = {t_res.statistic:.2f}, p = {t_res.pvalue:.4f}")
    print(f"95% CI for mean difference = [{ci_lo:.2f}, {ci_hi:.2f}]")
    print()

    # ── 3. Mann–Whitney U test ────────────────────────────────────────────────
    # Two-sided: H1 is that the distributions differ in location.
    # Exact method used when scipy ≥ 1.7; falls back to asymptotic otherwise.
    try:
        mw = stats.mannwhitneyu(counter, journal,
                                 alternative="two-sided", method="exact")
    except TypeError:
        mw = stats.mannwhitneyu(counter, journal, alternative="two-sided")

    print("Mann–Whitney U test (two-sided)")
    print(f"U = {mw.statistic:.0f}, p = {mw.pvalue:.4f}")
    print()

    # ── 4. Effect sizes ───────────────────────────────────────────────────────
    g    = hedges_g(counter, journal)
    cles = common_language_effect_size(counter, journal)

    print("Effect sizes")
    print(f"Hedges' g = {g:.2f}")
    print(f"Common-language effect size P(Counter > Journal) = {cles:.2f}")
    print()


if __name__ == "__main__":
    main()
