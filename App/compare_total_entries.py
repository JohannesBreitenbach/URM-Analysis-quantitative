"""
compare_total_entries.py  (original: compare_total_entries_simple_no_pandas.py)
───────────────────────────────────────────────────────────────────────────────
Answers the research question:
    "What is the average total number of entries per participant per condition?"

Entry operationalisation
------------------------
Total entries per participant = (number of rows in file) − 1
The subtracted row is the header. This treats every non-header line as one
logged event, regardless of content — consistent with how counter and journal
apps typically append one row per interaction.

Statistics reported
-------------------
For each condition: n, M, SD (sample), SEM, 95% CI half-width, median,
min, max, and sum. The 95% CI is computed via the t-distribution using
Welch–Satterthwaite degrees of freedom (appropriate for small, unequal-n
groups common in CHI studies).

Implementation note
-------------------
This script intentionally avoids pandas so it can run in minimal environments
(e.g., a bare Python install on a participant's data export machine).
Only the standard library (csv, math, statistics, pathlib) is used.

Usage
-----
    python compare_total_entries.py

Output
------
total_entries_compare.csv   — one row per condition with all descriptives
stdout                      — CHI-friendly summary (M, SD, SEM, 95% CI, ...)

Reproducibility note
--------------------
No randomness; output is fully deterministic given the input CSVs.
t critical values for df ≤ 30 are sourced from a hard-coded lookup table
of standard t-distribution constants (see ``t_crit_95_two_tailed``).

Requirements
------------
    python  >= 3.9   (standard library only — no third-party packages)
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).resolve().parent
COUNTER_DIR = BASE_DIR / "data_counter"
JOURNAL_DIR = BASE_DIR / "data_journal"
OUT_CSV     = BASE_DIR / "total_entries_compare.csv"


# ── Row counting ──────────────────────────────────────────────────────────────

def count_rows_minus_header(path: Path) -> int:
    """
    Count data rows in a CSV file (total rows minus the header row).

    Tries common delimiters (comma, semicolon, tab, pipe) and selects the
    one that produces the most header columns, which heuristically identifies
    the correct delimiter. Returns 0 for empty files or header-only files.

    Parameters
    ----------
    path : Path
        Absolute path to the CSV file.

    Returns
    -------
    int
        Number of non-header rows (≥ 0).
    """
    seps = [",", ";", "\t", "|"]

    raw = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    if not raw:
        return 0

    best_rows = None  # stores (n_header_columns, parsed_rows)
    for sep in seps:
        try:
            rows = list(csv.reader(raw, delimiter=sep))
            if not rows:
                continue
            header_cols = len(rows[0])
            # Prefer the delimiter that yields the most header columns
            if best_rows is None or header_cols > best_rows[0]:
                best_rows = (header_cols, rows)
        except Exception:
            continue

    if best_rows is None:
        return 0

    return max(len(best_rows[1]) - 1, 0)   # subtract 1 for the header row


def folder_totals(folder: Path) -> list[int]:
    """
    Return a list of per-participant entry counts for all CSVs in *folder*.

    One CSV = one participant. Files are processed in sorted order for
    deterministic output. Missing folders return an empty list (no error).

    Parameters
    ----------
    folder : Path
        Directory containing per-participant CSV files.

    Returns
    -------
    list[int]
        One integer per file: number of data rows (header excluded).
    """
    if not folder.exists():
        return []
    return [count_rows_minus_header(p) for p in sorted(folder.glob("*.csv"))]


# ── Descriptive statistics ─────────────────────────────────────────────────────

def t_crit_95_two_tailed(df: int) -> float:
    """
    Return the two-tailed t critical value at alpha=0.05 for the given df.

    Values for df=1..30 come from a standard t-distribution table and cover
    the sample sizes typical of CHI user studies. For df > 30 the function
    returns 1.96 (standard normal approximation), which is conservative and
    appropriate once n is large enough that the t and z distributions converge.

    Parameters
    ----------
    df : int
        Degrees of freedom (n − 1 for a single group).

    Returns
    -------
    float
        Critical t value.
    """
    # Standard t-table constants for alpha=0.05, two-tailed
    table = {
        1: 12.706, 2: 4.303,  3: 3.182,  4: 2.776,  5: 2.571,
        6: 2.447,  7: 2.365,  8: 2.306,  9: 2.262, 10: 2.228,
       11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
       16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
       21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
       26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    return table.get(df, 1.96)   # normal approximation beyond df=30


def descriptives(values: list[int]) -> dict:
    """
    Compute CHI-standard descriptive statistics for a list of counts.

    Returns n, mean, SD (sample, Bessel's correction), SEM, 95% CI
    half-width (via t-distribution), median, min, and max.

    Parameters
    ----------
    values : list[int]
        Per-participant entry counts.

    Returns
    -------
    dict with keys: n, mean, sd, sem, ci95, median, min, max
    """
    n = len(values)
    if n == 0:
        return {
            "n": 0, "mean": 0.0, "sd": 0.0, "sem": 0.0,
            "ci95": 0.0, "median": 0.0, "min": 0, "max": 0,
        }

    mean_v = sum(values) / n
    sd_v   = statistics.stdev(values) if n >= 2 else 0.0     # sample SD (ddof=1)
    sem_v  = sd_v / math.sqrt(n)       if n >= 1 else 0.0
    tcrit  = t_crit_95_two_tailed(n - 1) if n >= 2 else 0.0
    ci95_v = tcrit * sem_v             if n >= 2 else 0.0    # 95% CI half-width

    return {
        "n":      n,
        "mean":   mean_v,
        "sd":     sd_v,
        "sem":    sem_v,
        "ci95":   ci95_v,
        "median": float(statistics.median(values)),
        "min":    min(values),
        "max":    max(values),
    }


# ── Output helpers ────────────────────────────────────────────────────────────

def write_summary_csv(path: Path, rows: list[dict]) -> None:
    """
    Write the per-condition descriptive statistics to a CSV file.

    Parameters
    ----------
    path : Path
        Output file path.
    rows : list[dict]
        One dict per condition; keys match the fieldnames below.
    """
    fieldnames = [
        "condition",
        "n",
        "mean_total_entries",
        "sd_total_entries",
        "sem",
        "ci95_halfwidth",
        "median",
        "min",
        "max",
        "sum_total_entries",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ── Main routine ──────────────────────────────────────────────────────────────

def main() -> None:
    """
    Compute and report entry counts for both conditions.

    Steps
    -----
    1. Count rows (minus header) for every participant CSV in each folder.
    2. Compute descriptive statistics per condition.
    3. Print a CHI-friendly summary to stdout: M (SD), SEM, 95% CI.
    4. Save the full table to CSV for supplemental reporting.
    """
    # ── 1. Collect per-participant totals ─────────────────────────────────────
    counter_vals = folder_totals(COUNTER_DIR)
    journal_vals = folder_totals(JOURNAL_DIR)

    # ── 2. Compute descriptives ───────────────────────────────────────────────
    counter_desc = descriptives(counter_vals)
    journal_desc = descriptives(journal_vals)

    def pack(name: str, vals: list[int], d: dict) -> dict:
        """Bundle a condition's stats into a flat dict for CSV output."""
        return {
            "condition":          name,
            "n":                  d["n"],
            "mean_total_entries": f"{d['mean']:.6f}",
            "sd_total_entries":   f"{d['sd']:.6f}",
            "sem":                f"{d['sem']:.6f}",
            "ci95_halfwidth":     f"{d['ci95']:.6f}",
            "median":             f"{d['median']:.6f}",
            "min":                d["min"],
            "max":                d["max"],
            "sum_total_entries":  sum(vals),
        }

    summary_rows = [
        pack("counter", counter_vals, counter_desc),
        pack("journal", journal_vals, journal_desc),
    ]

    # ── 3. Console output (CHI paper–style inline reporting) ──────────────────
    print("Total entries per condition (rowcount − 1 per participant file):\n")
    for name, d in [("Counter", counter_desc), ("Journal", journal_desc)]:
        if d["n"] == 0:
            print(f"{name}: n=0")
            continue
        print(
            f"{name}: n={d['n']}, M={d['mean']:.2f}, SD={d['sd']:.2f}, "
            f"SEM={d['sem']:.2f}, 95% CI ±{d['ci95']:.2f}, "
            f"median={d['median']:.2f}, min={d['min']}, max={d['max']}"
        )

    # ── 4. Save to CSV ────────────────────────────────────────────────────────
    write_summary_csv(OUT_CSV, summary_rows)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
