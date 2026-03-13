"""
active_days_gender.py  (original: active_days_boxplot_4groups.py)
───────────────────────────────────────────────────────────────────────────────
Produces a 4-group boxplot comparing unique active days per participant,
stratified by both condition (Counter vs. Journal) and gender (male vs. female).

Research question addressed
---------------------------
"Does engagement (measured by unique active days) differ across condition and
gender subgroups?"

Groups plotted (left to right)
-------------------------------
    1. Counter — male
    2. Counter — female
    3. Journal — male
    4. Journal — female

Participants whose gender is neither "male" nor "female" are excluded from
the plot but reported to stdout for transparency.

Operationalisations
-------------------
Active day
    A calendar date on which a participant has at least one parseable log entry.
    Multiple entries on the same date for the same participant count once.

Participant identity
    One CSV file per participant; the filename stem must match the participant's
    MOTHER_CODE (e.g., ``A016S.csv``).

Data sources
------------
counter.csv / journal.csv
    Questionnaire exports; semicolon-delimited; one row per participant.
    Required columns: MOTHER_CODE, GENDER (1=male, 2=female).

./data_counter/<MOTHER_CODE>.csv
    Counter-condition app exports. Date parsed from year/month/day columns,
    a "date" column, or a generic timestamp column (see ``unique_dates_from_counter_file``).

./data_journal/<MOTHER_CODE>.csv
    Journal-condition app exports. Date expected in the second column
    (schema: participant_id, date, ...).

Plot style
----------
Boxes/whiskers/caps: black outline, linewidth 2.
Median line: tab:orange.
Individual data points: tab:blue (Counter), tab:orange (Journal), no jitter.

Output
------
./out/active_days_boxplot_4groups.png   — 120 dpi figure
stdout                                  — per-participant active days,
                                          group descriptives, Tukey box stats

Reproducibility note
--------------------
No randomness (the ``random`` import is present but unused in this version).
Output is fully deterministic given the input files.

Requirements
------------
    python     >= 3.9
    matplotlib >= 3.5
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import csv
import math
import random   # imported but currently unused; retained for potential future use
import re
import statistics
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent

Q_COUNTER = BASE_DIR / "counter.csv"          # Counter questionnaire data
Q_JOURNAL = BASE_DIR / "journal.csv"          # Journal questionnaire data

DIR_COUNTER = BASE_DIR / "data_counter"        # Counter app exports (one CSV per participant)
DIR_JOURNAL = BASE_DIR / "data_journal"        # Journal app exports (one CSV per participant)

OUT_DIR = BASE_DIR / "out"
OUT_DIR.mkdir(exist_ok=True)
OUT_PNG = OUT_DIR / "active_days_boxplot_4groups.png"

# Mother codes follow the pattern: one letter, three digits, one letter (e.g., A016S)
MOTHER_CODE_RE = re.compile(r"^[A-Z]\d{3}[A-Z]$")


# ── Questionnaire helpers ─────────────────────────────────────────────────────

def norm(s: str) -> str:
    """Normalise a string for case-insensitive, whitespace-tolerant comparison."""
    return (s or "").strip().lower().replace(" ", "_")


def find_col_idx(header: List[str], target: str) -> int:
    """
    Return the index of *target* in *header* using case-insensitive matching.

    Parameters
    ----------
    header : list[str]   Column names from the CSV header row.
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

    Numeric coding assumed: 1 = male, 2 = female (consistent with the
    questionnaire export format observed in this study).
    String labels ("m", "f", "male", "female") are also accepted to
    handle potential manual entries or future export changes.

    Parameters
    ----------
    g : str   Raw gender cell value from the questionnaire CSV.

    Returns
    -------
    str   One of 'male', 'female', or the original value if unrecognised.
    """
    s = (g or "").strip().lower()
    if s in {"1", "m", "male", "man"}:
        return "male"
    if s in {"2", "f", "female", "woman"}:
        return "female"
    return s or "unknown"


def read_questionnaire(path: Path, condition: str) -> Dict[str, Dict[str, str]]:
    """
    Parse a questionnaire CSV and return participant metadata.

    Lines whose MOTHER_CODE does not match the expected regex pattern
    (e.g., summary rows, blank rows, or test entries) are silently skipped.

    Parameters
    ----------
    path      : Path    Absolute path to the semicolon-delimited questionnaire CSV.
    condition : str     'counter' or 'journal' — attached to each record.

    Returns
    -------
    dict[str, dict]
        Mapping of MOTHER_CODE → {'condition': str, 'gender': str}.
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
            if not row or max(i_gender, i_code) >= len(row):
                continue

            code = row[i_code].strip().upper()
            if not MOTHER_CODE_RE.match(code):
                continue   # skip non-participant rows (summaries, blanks, etc.)

            gender = normalize_gender(row[i_gender])
            participants[code] = {"condition": condition, "gender": gender}

    return participants


# ── Date parsing ──────────────────────────────────────────────────────────────

# Format strings tried in order; covers ISO 8601, European DD.MM.YYYY,
# and common timestamp variants seen in mobile app exports.
_DT_FORMATS = [
    "%Y-%m-%d",
    "%d.%m.%Y",
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f%z",
]


def detect_delimiter(sample: str) -> str:
    """
    Heuristically identify the CSV delimiter from the first few lines.

    Parameters
    ----------
    sample : str   Raw text snippet (e.g., first 50 lines joined).

    Returns
    -------
    str   The first delimiter found among ',', ';', tab, '|'; defaults to ','.
    """
    for d in [",", ";", "\t", "|"]:
        if d in sample:
            return d
    return ","


def read_rows(path: Path) -> Tuple[List[str], List[List[str]]]:
    """
    Read a CSV into (header, data_rows), auto-detecting the delimiter.

    Parameters
    ----------
    path : Path   Absolute path to the CSV file.

    Returns
    -------
    (list[str], list[list[str]])   Header row and remaining data rows.
    """
    raw   = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    if not raw:
        return [], []
    delim = detect_delimiter("\n".join(raw[:50]))
    rows  = list(csv.reader(raw, delimiter=delim))
    if not rows:
        return [], []
    return rows[0], rows[1:] if len(rows) > 1 else []


def try_parse_datetime(s: str) -> Optional[datetime]:
    """
    Attempt to parse a string as a datetime using the format list above.

    Falls back to ``datetime.fromisoformat`` for ISO-8601 variants not
    covered by strptime (e.g., milliseconds with UTC offset).

    Parameters
    ----------
    s : str   Raw date/time string from an app export.

    Returns
    -------
    datetime or None   None if parsing fails with all formats.
    """
    if s is None:
        return None
    txt = str(s).strip()
    if not txt:
        return None

    for fmt in _DT_FORMATS:
        try:
            return datetime.strptime(txt, fmt)
        except Exception:
            pass

    # Final fallback: Python's built-in ISO parser (handles fractional seconds, offsets)
    try:
        return datetime.fromisoformat(txt.replace("Z", "+00:00"))
    except Exception:
        return None


def find_col(header: List[str], candidates: List[str]) -> Optional[int]:
    """
    Find the first column whose name matches any of *candidates*.

    Tries exact (normalised) matches first, then substring matches.
    Returns None if no candidate matches any header.

    Parameters
    ----------
    header     : list[str]   CSV header row.
    candidates : list[str]   Preferred column name variants, in priority order.

    Returns
    -------
    int or None
    """
    hn = [norm(h) for h in header]

    for cand in candidates:
        c = norm(cand)
        if c in hn:
            return hn.index(c)

    for cand in candidates:
        c = norm(cand)
        for i, h in enumerate(hn):
            if c in h:
                return i

    return None


# ── App data: active-day counting ────────────────────────────────────────────

def unique_dates_from_counter_file(path: Path) -> List[date]:
    """
    Extract the unique calendar dates from a Counter-condition app export.

    Resolution priority:
      1. Separate year/month/day integer columns (most unambiguous).
      2. A recognised date/timestamp column (see *candidates* list).
      3. Column index 0 as a last resort.

    Parameters
    ----------
    path : Path   Absolute path to the participant's Counter CSV.

    Returns
    -------
    list[date]   Sorted list of unique active calendar dates.
    """
    header, data = read_rows(path)
    if not header:
        return []

    hn = [norm(h) for h in header]

    # Priority 1: explicit integer columns — avoids all date-format ambiguity
    if all(k in hn for k in ["year", "month", "day"]):
        iy, im, iday = hn.index("year"), hn.index("month"), hn.index("day")
        out = set()
        for row in data:
            if max(iy, im, iday) >= len(row):
                continue
            try:
                out.add(date(int(row[iy]), int(row[im]), int(row[iday])))
            except Exception:
                continue
        return sorted(out)

    # Priority 2: a named date/timestamp column
    idx = find_col(
        header,
        ["date", "timestamp", "datetime", "created_at", "time", "created", "ts"],
    )
    if idx is None:
        idx = 0   # fallback: column 0

    out = set()
    for row in data:
        if idx >= len(row):
            continue
        dt = try_parse_datetime(row[idx])
        if dt is not None:
            out.add(dt.date())
    return sorted(out)


def unique_dates_from_journal_file(path: Path) -> List[date]:
    """
    Extract the unique calendar dates from a Journal-condition app export.

    Per the agreed schema for this study, the date is always in the
    **second column** (index 1), following the participant-ID column.

    Parameters
    ----------
    path : Path   Absolute path to the participant's Journal CSV.

    Returns
    -------
    list[date]   Sorted list of unique active calendar dates.
    """
    header, data = read_rows(path)
    if not header or len(header) < 2:
        return []

    out = set()
    for row in data:
        if 1 >= len(row):
            continue
        dt = try_parse_datetime(row[1])
        if dt is not None:
            out.add(dt.date())
    return sorted(out)


# ── Descriptive statistics ────────────────────────────────────────────────────

def percentile(sorted_vals: List[float], p: float) -> float:
    """
    Compute a percentile using linear interpolation on already-sorted values.

    Equivalent to numpy's default 'linear' method, which is what matplotlib's
    boxplot uses internally — ensuring the printed stats match the figure.

    Parameters
    ----------
    sorted_vals : list[float]   Values sorted in ascending order.
    p           : float         Percentile in [0, 100].

    Returns
    -------
    float
    """
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])

    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return float(sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f))


def tukey_box_stats(vals: List[int]) -> Dict[str, float]:
    """
    Compute Tukey boxplot statistics: Q1, median, Q3, whiskers, min, max.

    Whiskers extend to the most extreme observed value within 1.5 × IQR of
    the box edges (Tukey, 1977). Points outside the whiskers are outliers
    (not plotted here since ``showfliers=False``).

    Parameters
    ----------
    vals : list[int]   Per-participant active-day counts.

    Returns
    -------
    dict with keys: q1, median, q3, wl (whisker low), wh (whisker high),
                    vmin, vmax.
    """
    if not vals:
        return dict(q1=0, median=0, q3=0, wl=0, wh=0, vmin=0, vmax=0)

    s           = sorted(float(v) for v in vals)
    q1          = percentile(s, 25)
    med         = percentile(s, 50)
    q3          = percentile(s, 75)
    iqr         = q3 - q1
    low_fence   = q1 - 1.5 * iqr
    high_fence  = q3 + 1.5 * iqr

    wl = min(v for v in s if v >= low_fence)    # lowest non-outlier value
    wh = max(v for v in s if v <= high_fence)   # highest non-outlier value

    return dict(q1=q1, median=med, q3=q3, wl=wl, wh=wh, vmin=min(s), vmax=max(s))


# ── Figure ────────────────────────────────────────────────────────────────────

def make_boxplot_4groups_active_days(
    groups: Dict[str, List[int]], out_path: Path
) -> None:
    """
    Draw and save the 4-group active-days boxplot.

    Design choices
    --------------
    - ``showfliers=False``: outlier suppression because individual points are
      plotted explicitly, making separate flier markers redundant and
      confusing at small n.
    - No jitter on points: with the small group sizes typical in CHI studies,
      aligned scatter provides a cleaner read than stochastic jitter.
    - Median rendered in tab:orange so it is visually distinct from the black
      box outline even in greyscale prints (orange prints mid-grey).

    Parameters
    ----------
    groups   : dict[str, list[int]]
        Keys: 'counter_male', 'counter_female', 'journal_male', 'journal_female'.
        Values: per-participant active-day counts.
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
        showfliers=False,           # individual points shown instead
        patch_artist=False,         # outline-only boxes (no fill)
        boxprops=dict(color="black",   linewidth=2),
        whiskerprops=dict(color="black", linewidth=2),
        capprops=dict(color="black",   linewidth=2),
        medianprops=dict(color=c_journal, linewidth=2),   # orange median
    )

    # Overlay individual participant data points (centred, no jitter)
    for x, vals, col in zip(positions, data, point_colors):
        ax.scatter([x] * len(vals), vals, s=150, color=col, alpha=0.85, zorder=3)

    ax.set_title("Engagement by condition (unique active days)", fontsize=26, pad=18)
    ax.set_ylabel("Active days per participant", fontsize=22)

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
    Orchestrate data loading, active-day computation, stats reporting, and
    figure export.

    Steps
    -----
    1. Parse questionnaire CSVs to obtain MOTHER_CODE → {condition, gender}.
    2. For each participant, locate their app CSV and count unique active days.
    3. Assign participants to one of four groups (condition × gender).
    4. Print per-participant counts and Tukey box statistics to stdout.
    5. Report any participants excluded due to non-binary gender codes or
       missing app files (for transparency in the methods section).
    6. Render and save the 4-group boxplot.
    """
    if not Q_COUNTER.exists() or not Q_JOURNAL.exists():
        raise FileNotFoundError("Missing counter.csv or journal.csv next to the script.")
    if not DIR_COUNTER.exists() or not DIR_JOURNAL.exists():
        raise FileNotFoundError("Missing data_counter/ or data_journal/ folders.")

    # ── 1. Load participant metadata ──────────────────────────────────────────
    participants: Dict[str, Dict[str, str]] = {}
    participants.update(read_questionnaire(Q_COUNTER, "counter"))
    participants.update(read_questionnaire(Q_JOURNAL, "journal"))

    # ── 2. Count active days per participant ──────────────────────────────────
    per_participant = []    # (condition, gender, mother_code, active_days)
    missing_files   = []

    for code in sorted(participants.keys()):
        cond   = participants[code]["condition"]
        gender = participants[code]["gender"]

        folder   = DIR_COUNTER if cond == "counter" else DIR_JOURNAL
        app_path = folder / f"{code}.csv"

        if not app_path.exists():
            missing_files.append(f"{cond}:{code}")
            continue

        if cond == "counter":
            uniq_dates = unique_dates_from_counter_file(app_path)
        else:
            uniq_dates = unique_dates_from_journal_file(app_path)

        per_participant.append((cond, gender, code, len(uniq_dates)))

    # ── 3. Assign to 4 groups ─────────────────────────────────────────────────
    groups: Dict[str, List[int]] = {
        "counter_male": [], "counter_female": [],
        "journal_male": [], "journal_female": [],
    }
    excluded_gender = []    # tracks participants with non-binary gender codes

    for cond, gender, code, days in per_participant:
        if gender not in {"male", "female"}:
            excluded_gender.append(f"{cond}:{code} ({gender})")
            continue
        groups[f"{cond}_{gender}"].append(days)

    # ── 4. Console output ─────────────────────────────────────────────────────
    print("\nActive days per participant (unique dates):")
    for cond, gender, code, days in per_participant:
        print(f"  {code}: {cond}, {gender}, active_days={days}")

    print("\nGroup stats (descriptives + Tukey box stats):")
    for key in ["counter_male", "counter_female", "journal_male", "journal_female"]:
        vals = groups[key]
        n    = len(vals)
        m    = sum(vals) / n if n else 0.0
        sd   = statistics.stdev(vals) if n >= 2 else 0.0
        b    = tukey_box_stats(vals)
        print(
            f"  {key}: n={n}, M={m:.2f}, SD={sd:.2f}, "
            f"q1={b['q1']:.2f}, median={b['median']:.2f}, q3={b['q3']:.2f}, "
            f"whiskers=[{b['wl']:.2f}, {b['wh']:.2f}], "
            f"min={b['vmin']:.2f}, max={b['vmax']:.2f}, values={sorted(vals)}"
        )

    # ── 5. Exclusion reporting ────────────────────────────────────────────────
    if excluded_gender:
        print("\nExcluded from 4-group plot (gender not coded male/female):")
        for x in excluded_gender:
            print(f"  - {x}")

    if missing_files:
        print("\nMissing app files (excluded from all analyses):")
        for x in missing_files:
            print(f"  - {x}")

    # ── 6. Render and save figure ─────────────────────────────────────────────
    make_boxplot_4groups_active_days(groups, OUT_PNG)
    print(f"\nSaved plot: {OUT_PNG}")


if __name__ == "__main__":
    main()
