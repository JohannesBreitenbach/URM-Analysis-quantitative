"""
entries_gender.py  (original: entries_by_gender_and_condition.py)
───────────────────────────────────────────────────────────────────────────────
Counts total app entries per participant and compares them across four
subgroups: Counter × male, Counter × female, Journal × male, Journal × female.

Research question addressed
---------------------------
"Does total app usage (entry count) differ across condition and gender?"

Entry operationalisation
------------------------
entries = (number of non-empty lines in the participant's app CSV) − 1
The subtracted row is the header. This treats every non-header, non-blank
line as one logged event, regardless of content.

Implementation note
-------------------
This script deliberately avoids pandas so it can run in minimal environments
(e.g., bare Python on a participant's export machine or a restricted server).
Only the standard library (csv, statistics, pathlib, re, math) is used.

Data sources
------------
counter.csv / journal.csv
    Questionnaire exports; semicolon-delimited; one row per participant.
    Required columns: MOTHER_CODE, GENDER (1=male, 2=female).

./data_counter/<MOTHER_CODE>.csv
    Counter-condition app exports.

./data_journal/<MOTHER_CODE>.csv
    Journal-condition app exports.

Plot style
----------
Matches the rest of the analysis suite:
    Boxes/whiskers/caps: black outline, linewidth 2.
    Median: tab:orange.
    Points: tab:blue (Counter), tab:orange (Journal), no jitter.

Output
------
entries_per_participant.csv
    mother_code, condition, gender, entries, missing_app_file (0/1).

entries_summary.csv
    Means per condition, per gender, and per condition × gender cell.

entries_boxplot_4groups.png
    4-group boxplot (120 dpi).

stdout
    Group sizes and any exclusion/missing-file warnings.

Reproducibility note
--------------------
No randomness (``random`` import retained for API compatibility but unused).
Output is fully deterministic given the input files.

Requirements
------------
    python     >= 3.9    (standard library only — no third-party packages)
    matplotlib >= 3.5    (for the boxplot figure only)
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import random   # currently unused; retained for potential future use
import matplotlib.pyplot as plt


# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent

Q_COUNTER = BASE_DIR / "counter.csv"
Q_JOURNAL = BASE_DIR / "journal.csv"

DIR_COUNTER = BASE_DIR / "data_counter"
DIR_JOURNAL = BASE_DIR / "data_journal"

OUT_PARTICIPANTS = BASE_DIR / "entries_per_participant.csv"
OUT_SUMMARY      = BASE_DIR / "entries_summary.csv"
OUT_BOXPLOT      = BASE_DIR / "entries_boxplot_4groups.png"

# Mother codes follow the pattern: one letter, three digits, one letter (e.g., A016S)
MOTHER_CODE_RE = re.compile(r"^[A-Z]\d{3}[A-Z]$")


# ── Shared helpers ────────────────────────────────────────────────────────────

def norm(s: str) -> str:
    """Normalise a string for case-insensitive, whitespace-tolerant comparison."""
    return (s or "").strip().lower().replace(" ", "_")


def find_col_idx(header: List[str], target: str) -> int:
    """
    Return the index of *target* in *header*, case-insensitively.

    Accepts minor naming variations (spaces ↔ underscores) that can arise
    from different questionnaire export settings.

    Parameters
    ----------
    header : list[str]   CSV header row.
    target : str         Column name to locate.

    Returns
    -------
    int   Zero-based column index.

    Raises
    ------
    ValueError   If no match is found.
    """
    t  = norm(target)
    hn = [norm(h) for h in header]
    if t in hn:
        return hn.index(t)
    raise ValueError(f"Column '{target}' not found. Found: {header}")


def normalize_gender(g: str) -> str:
    """
    Map questionnaire gender codes to a canonical string.

    Numeric coding assumed: 1 = male, 2 = female.
    String labels are also accepted to handle manual entries.

    Parameters
    ----------
    g : str   Raw gender cell from the questionnaire CSV.

    Returns
    -------
    str   'male', 'female', 'unknown', or the original value.
    """
    s = (g or "").strip().lower()
    if s in {"1", "m", "male", "man"}:
        return "male"
    if s in {"2", "f", "female", "woman"}:
        return "female"
    if not s:
        return "unknown"
    return s


# ── Questionnaire loading ─────────────────────────────────────────────────────

def read_questionnaire(path: Path, condition: str) -> Dict[str, Dict[str, str]]:
    """
    Parse a questionnaire CSV and return participant metadata.

    Rows whose MOTHER_CODE does not match the expected regex (e.g., summary
    rows, blank rows, test entries) are silently skipped.

    Parameters
    ----------
    path      : Path   Absolute path to the semicolon-delimited questionnaire.
    condition : str    'counter' or 'journal'.

    Returns
    -------
    dict[str, dict]
        MOTHER_CODE → {'condition': str, 'gender': str}.
    """
    participants: Dict[str, Dict[str, str]] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r      = csv.reader(f, delimiter=";")
        header = next(r, None)
        if not header:
            return participants

        i_gender = find_col_idx(header, "GENDER")
        i_code   = find_col_idx(header, "MOTHER_CODE")

        for row in r:
            if not row:
                continue
            if max(i_gender, i_code) >= len(row):
                continue

            code = row[i_code].strip().upper()
            if not MOTHER_CODE_RE.match(code):
                continue   # skip non-participant rows

            participants[code] = {
                "condition": condition,
                "gender":    normalize_gender(row[i_gender]),
            }

    return participants


# ── Entry counting ────────────────────────────────────────────────────────────

def count_entries(app_file: Path) -> Optional[int]:
    """
    Count the number of data rows in a participant's app CSV.

    entries = non-empty lines − 1 (header excluded).
    Returns None if the file does not exist, so callers can flag it as
    missing without raising an exception.

    Parameters
    ----------
    app_file : Path   Path to the participant's app export CSV.

    Returns
    -------
    int or None   Row count (≥ 0), or None if the file is absent.
    """
    if not app_file.exists():
        return None
    lines     = app_file.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    non_empty = sum(1 for ln in lines if ln.strip())
    return max(non_empty - 1, 0)   # subtract 1 for the header row


# ── Statistics helpers ────────────────────────────────────────────────────────

def stats(values: List[int]) -> Tuple[int, float, float]:
    """
    Compute n, mean, and sample SD for a list of counts.

    Parameters
    ----------
    values : list[int]

    Returns
    -------
    (n, mean, sd)   SD is 0.0 when n < 2.
    """
    n = len(values)
    if n == 0:
        return 0, 0.0, 0.0
    mean = sum(values) / n
    sd   = statistics.stdev(values) if n >= 2 else 0.0   # sample SD (Bessel's)
    return n, mean, sd


# ── Figure ────────────────────────────────────────────────────────────────────

def make_boxplot_4groups(groups: Dict[str, List[int]], out_path: Path) -> None:
    """
    Draw and save the 4-group entry-count boxplot.

    Design choices
    --------------
    - ``showfliers=False``: outlier suppression because individual points are
      plotted explicitly, making separate flier markers redundant.
    - No jitter: aligned scatter gives a cleaner read at small n.
    - Orange median line: visually distinct from the black box outline and
      prints as mid-grey in greyscale — consistent with the rest of the suite.

    Parameters
    ----------
    groups   : dict[str, list[int]]
        Keys: 'counter_male', 'counter_female', 'journal_male', 'journal_female'.
    out_path : Path
        Destination for the saved PNG.
    """
    order        = ["counter_male", "counter_female", "journal_male", "journal_female"]
    labels       = ["Counter (male)", "Counter (female)", "Journal (male)", "Journal (female)"]
    positions    = [1, 2, 3, 4]
    data         = [groups.get(k, []) for k in order]

    c_counter    = "tab:blue"
    c_journal    = "tab:orange"
    point_colors = [c_counter, c_counter, c_journal, c_journal]

    fig, ax = plt.subplots(figsize=(14, 8), dpi=120)

    ax.boxplot(
        data,
        positions=positions,
        widths=0.3,
        showfliers=False,
        patch_artist=False,         # outline-only boxes (no fill)
        boxprops=dict(color="black",    linewidth=2),
        whiskerprops=dict(color="black", linewidth=2),
        capprops=dict(color="black",    linewidth=2),
        medianprops=dict(color=c_journal, linewidth=2),
    )

    # Overlay individual participant data points (centred, no jitter)
    for x, vals, col in zip(positions, data, point_colors):
        ax.scatter([x] * len(vals), vals, s=150, color=col, alpha=0.85, zorder=3)

    ax.set_title("Entries by Gender", fontsize=26, pad=18)
    ax.set_ylabel("Total Entry Count", fontsize=22)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=20, rotation=15, ha="right")
    ax.tick_params(axis="y", labelsize=18)

    ax.grid(axis="y", linestyle="--", linewidth=1.2, alpha=0.45)

    # Manual margins prevent long x-tick labels from being clipped
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.22, top=0.88)
    fig.savefig(out_path)
    plt.close(fig)


# ── Main routine ──────────────────────────────────────────────────────────────

def main() -> None:
    """
    Orchestrate questionnaire loading, entry counting, CSV export, and figure.

    Steps
    -----
    1. Parse questionnaire CSVs for MOTHER_CODE → {condition, gender}.
    2. For each participant, count non-header rows in their app CSV.
       Participants whose app file is absent are flagged (missing_app_file=1)
       and excluded from summaries and the plot.
    3. Write per-participant CSV (for inspection and supplemental reporting).
    4. Aggregate by condition, gender, and condition × gender; write summary CSV.
    5. Assign participants to the 4 plot groups; exclude non-binary gender codes
       and report them to stdout for transparency.
    6. Render and save the 4-group boxplot.
    """
    if not Q_COUNTER.exists() or not Q_JOURNAL.exists():
        raise FileNotFoundError("Missing counter.csv or journal.csv next to the script.")
    if not DIR_COUNTER.exists() or not DIR_JOURNAL.exists():
        raise FileNotFoundError("Missing data_counter/ or data_journal/ folders.")

    # ── 1. Load questionnaire metadata ────────────────────────────────────────
    participants: Dict[str, Dict[str, str]] = {}
    participants.update(read_questionnaire(Q_COUNTER, "counter"))
    participants.update(read_questionnaire(Q_JOURNAL, "journal"))

    # ── 2. Count entries per participant ──────────────────────────────────────
    rows    = []
    missing = []

    for code in sorted(participants.keys()):
        cond   = participants[code]["condition"]
        gender = participants[code]["gender"]

        folder   = DIR_COUNTER if cond == "counter" else DIR_JOURNAL
        app_path = folder / f"{code}.csv"

        entries      = count_entries(app_path)
        missing_flag = 0 if entries is not None else 1
        if entries is None:
            entries = 0
            missing.append(f"{cond}:{code}")

        rows.append({
            "mother_code":     code,
            "condition":       cond,
            "gender":          gender,
            "entries":         entries,
            "missing_app_file": missing_flag,
        })

    # ── 3. Write per-participant CSV ──────────────────────────────────────────
    with OUT_PARTICIPANTS.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["mother_code", "condition", "gender",
                        "entries", "missing_app_file"],
        )
        w.writeheader()
        w.writerows(rows)

    # ── 4. Aggregate and write summary CSV ────────────────────────────────────
    by_condition:   Dict[str, List[int]] = {}
    by_gender:      Dict[str, List[int]] = {}
    by_cond_gender: Dict[str, List[int]] = {}

    # 4-group plot containers (male/female only, as per analysis plan)
    plot_groups: Dict[str, List[int]] = {
        "counter_male": [], "counter_female": [],
        "journal_male": [], "journal_female": [],
    }
    excluded_for_plot = []

    for r in rows:
        if int(r["missing_app_file"]) == 1:
            continue    # exclude participants with no app data

        cond   = str(r["condition"])
        gender = str(r["gender"])
        e      = int(r["entries"])

        by_condition.setdefault(cond, []).append(e)
        by_gender.setdefault(gender, []).append(e)
        by_cond_gender.setdefault(f"{cond}×{gender}", []).append(e)

        if gender in {"male", "female"}:
            plot_groups[f"{cond}_{gender}"].append(e)
        else:
            excluded_for_plot.append(f"{cond}:{r['mother_code']} ({gender})")

    with OUT_SUMMARY.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_type", "group", "n", "mean_entries", "sd_entries"])

        for cond in sorted(by_condition):
            n, m, sd = stats(by_condition[cond])
            w.writerow(["condition", cond, n, f"{m:.6f}", f"{sd:.6f}"])

        for g in sorted(by_gender):
            n, m, sd = stats(by_gender[g])
            w.writerow(["gender", g, n, f"{m:.6f}", f"{sd:.6f}"])

        for cg in sorted(by_cond_gender):
            n, m, sd = stats(by_cond_gender[cg])
            w.writerow(["condition_x_gender", cg, n, f"{m:.6f}", f"{sd:.6f}"])

    # ── 5 & 6. Plot and console output ────────────────────────────────────────
    make_boxplot_4groups(plot_groups, OUT_BOXPLOT)

    print(f"Saved: {OUT_PARTICIPANTS.name}")
    print(f"Saved: {OUT_SUMMARY.name}")
    print(f"Saved: {OUT_BOXPLOT.name}")

    print("\nBoxplot group sizes:")
    for k in ["counter_male", "counter_female", "journal_male", "journal_female"]:
        print(f"  {k}: n={len(plot_groups[k])}")

    if excluded_for_plot:
        print("\nExcluded from 4-group plot (gender not coded male/female):")
        for x in excluded_for_plot:
            print(f"  - {x}")

    if missing:
        print("\nMissing app files (excluded from summaries and plot):")
        for x in missing:
            print(f"  - {x}")


if __name__ == "__main__":
    main()
