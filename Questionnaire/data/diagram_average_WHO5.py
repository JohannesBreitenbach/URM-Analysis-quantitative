"""
diagram_average_WHO5.py
───────────────────────────────────────────────────────────────────────────────
Plots mean WHO-5 Wellbeing scores across three study timepoints for both
conditions (Counter and Journal) with 95% confidence interval error bars.

Research question addressed
---------------------------
"Does subjective wellbeing (WHO-5) change over the study week, and does the
trajectory differ between the Counter and Journal conditions?"

WHO-5 operationalisation
------------------------
The WHO-5 Well-Being Index (Topp et al., 2015) is scored as the sum of five
Likert items (0–5 each), yielding a total range of 0–25.  Higher scores
indicate better wellbeing.  The three measurement points correspond to the
beginning, middle, and end of the study week:

    Sum_START   → "Day 1"
    Sum_MIDDLE  → "Day 4"
    Sum_END     → "Day 8"

Confidence interval note
------------------------
95% CIs are computed using a normal approximation (z = 1.96 × SEM).  This
is appropriate for visual display but should be interpreted cautiously given
the small group sizes typical in CHI studies (n < 30).  For inferential
testing, use t-distribution CIs; see ``compare_total_entries.py``.

Reference
---------
Topp, C. W., Østergaard, S. D., Søndergaard, S., & Bech, P. (2015). The
WHO-5 Well-Being Index: a systematic review of the literature. Psychotherapy
and Psychosomatics, 84(3), 167–176. https://doi.org/10.1159/000376585

Usage
-----
    python diagram_average_WHO5.py
    (or run interactively; the script calls plt.show() at the end)

Data format
-----------
counter.csv / journal.csv
    Semicolon-delimited, decimal comma (European locale).
    Required columns: Sum_START, Sum_MIDDLE, Sum_END.

Output
------
Interactive matplotlib window (and/or inline display in Jupyter).
summary DataFrame printed to the last cell (if run interactively).

Reproducibility note
--------------------
No randomness; output is fully deterministic given the input CSVs.
Displayed error bars match the printed ``summary`` table exactly.

Requirements
------------
    python     >= 3.9
    pandas     >= 1.3
    numpy      >= 1.21
    matplotlib >= 3.5
───────────────────────────────────────────────────────────────────────────────
"""

from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Configuration ─────────────────────────────────────────────────────────────

COUNTER_PATH = "counter.csv"   # Counter questionnaire (same directory as script)
JOURNAL_PATH = "journal.csv"   # Journal questionnaire (same directory as script)

# Column names in the questionnaire CSVs that hold the WHO-5 sum scores
# mapped to the display labels used on the x-axis.
TIME_COLS = {
    "Day 1": "Sum_START",
    "Day 4": "Sum_MIDDLE",
    "Day 8": "Sum_END",
}


# ── Data loading ──────────────────────────────────────────────────────────────

# Load both conditions; decimal="," handles European locale exports
# where numbers are stored as e.g. "14,0" instead of "14.0".
counter = pd.read_csv(COUNTER_PATH, sep=";", decimal=",", encoding="utf-8")
journal = pd.read_csv(JOURNAL_PATH, sep=";", decimal=",", encoding="utf-8-sig")

# Add explicit condition labels as a new column.
# This is safer than relying on any APP_TYPE or similar string column that
# might contain inconsistent values across export batches.
counter["Condition"] = "Counter"
journal["Condition"] = "Journal"

df = pd.concat([counter, journal], ignore_index=True)


# ── Data validation ───────────────────────────────────────────────────────────

# Verify that all expected WHO-5 sum columns are present before proceeding.
# Missing columns typically indicate a renamed or differently exported file.
missing = [c for c in TIME_COLS.values() if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing expected wellbeing columns: {missing}\n"
        f"Columns found: {list(df.columns)}"
    )


# ── Reshape to long format ────────────────────────────────────────────────────

# Melt from wide (one column per timepoint) to long (one row per observation)
# so that groupby aggregation can be applied cleanly across conditions and times.
long_df = df[["Condition"] + list(TIME_COLS.values())].copy()
long_df = long_df.melt(
    id_vars=["Condition"],
    value_vars=list(TIME_COLS.values()),
    var_name="TimeVar",
    value_name="WellbeingSum",
)

# Map raw column names back to the ordered display labels ("Day 1", "Day 4", "Day 8")
inv_map     = {v: k for k, v in TIME_COLS.items()}
long_df["Time"] = long_df["TimeVar"].map(inv_map)

# Make Time an ordered Categorical so groupby / sort operations respect
# the intended temporal order rather than alphabetical order.
long_df["Time"] = pd.Categorical(
    long_df["Time"],
    categories=["Day 1", "Day 4", "Day 8"],
    ordered=True,
)

# Coerce to numeric; non-parseable entries become NaN and are dropped below.
long_df["WellbeingSum"] = pd.to_numeric(long_df["WellbeingSum"], errors="coerce")
long_df = long_df.dropna(subset=["WellbeingSum"])


# ── Descriptive statistics ────────────────────────────────────────────────────

summary = (
    long_df.groupby(["Condition", "Time"], observed=True)
    .agg(
        n   = ("WellbeingSum", "count"),
        mean= ("WellbeingSum", "mean"),
        sd  = ("WellbeingSum", "std"),    # sample SD (ddof=1 by default in pandas)
    )
    .reset_index()
)

# Standard error of the mean (SEM = SD / sqrt(n))
summary["sem"] = summary["sd"] / np.sqrt(summary["n"])

# 95% CI half-width using normal approximation (z = 1.96).
# Note: for small n (< 30) the t-distribution is more appropriate for
# inferential tests, but the normal approximation is acceptable for
# error-bar display in a figure caption context.
summary["ci95"] = 1.96 * summary["sem"]


# ── Figure ────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4.5))

for cond in ["Counter", "Journal"]:
    s = summary[summary["Condition"] == cond].sort_values("Time")
    ax.errorbar(
        s["Time"].astype(str),
        s["mean"],
        yerr=s["ci95"],       # error bars = ±95% CI
        marker="o",
        linewidth=2,
        capsize=4,
        label=f"{cond} (n={int(s['n'].max())})",
    )

ax.set_title("Wellbeing (WHO-5 sum) across the week by condition")
ax.set_xlabel("Timepoint")
ax.set_ylabel("WHO-5 Sum (0–25)")
ax.set_ylim(0, 25)   # fix y-axis to full WHO-5 scale for honest visual comparison
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
ax.legend()

plt.tight_layout()
plt.show()

# Print the aggregated table used to draw the figure.
# Retaining this as the final expression makes it visible in Jupyter output
# and also serves as a machine-readable record of the plotted values.
summary.sort_values(["Condition", "Time"]).reset_index(drop=True)
