"""
FTI_active_days.py  (original: active_days_vs_future_intention.py)
───────────────────────────────────────────────────────────────────────────────
Examines whether engagement (unique active days) correlates with self-reported
Future Intention to Use (FTI) the app.

Research question addressed
---------------------------
"Does spending more active days in the app predict higher future intention
to continue using it, overall and within each condition?"

Variable operationalisations
-----------------------------
Active days
    Unique calendar dates on which a participant logged at least one entry.
    Multiple entries on the same date count once.

Future Intention to Use (FTI)
    Mean of the 3 questionnaire items whose headers begin with
    "Future Intention to use:" (7-point Likert scale assumed).
    Mean is used rather than sum so the score is comparable to the
    item scale and interpretable regardless of the exact item wording.

Statistical approach
--------------------
Pearson r is computed as a measure of linear association.  Because n is
small (< 30 per group), the conventional t-based p-value can be unreliable.
A permutation test (10,000 shuffles, seed=42) is therefore used as the
primary significance test.  The permutation p-value is two-sided
(test statistic: |r|).

Add-one smoothing is applied to the permutation p-value to avoid p = 0:
    p = (count of |r_perm| ≥ |r_obs| + 1) / (n_perm + 1)
This ensures the reported p is never exactly 0 while being nearly identical
to the raw p for any n_perm ≥ 1000.

Reference
---------
Phipson, B. & Smyth, G. K. (2010). Permutation p-values should never be
zero. Statistical Applications in Genetics and Molecular Biology, 9(1).
https://doi.org/10.2202/1544-6115.1585

Usage
-----
    python FTI_active_days.py

Data sources
------------
counter.csv / journal.csv
    Questionnaire exports; semicolon-delimited.
    Required columns: MOTHER_CODE, and the 3 FTI items (headers starting
    with "Future Intention to use:").

./data_counter/<MOTHER_CODE>.csv
    Counter-condition app exports.

./data_journal/<MOTHER_CODE>.csv
    Journal-condition app exports.

Output
------
./out/active_days_vs_fti.csv   — merged per-participant table
stdout                         — per-participant values + correlation results

Reproducibility note
--------------------
The permutation test uses a fixed seed (seed=42) so results are fully
reproducible. Changing n_perm or seed will alter p-values slightly.

Requirements
------------
    python >= 3.9   (standard library only — no third-party packages)
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import csv
import math
import random
import re
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Configuration ─────────────────────────────────────────────────────────────

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()   # fallback for interactive / notebook use

Q_COUNTER   = BASE_DIR / "counter.csv"
Q_JOURNAL   = BASE_DIR / "journal.csv"
DIR_COUNTER = BASE_DIR / "data_counter"
DIR_JOURNAL = BASE_DIR / "data_journal"

OUT_DIR = BASE_DIR / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "active_days_vs_fti.csv"

# Mother codes follow the pattern: one letter, three digits, one letter (e.g., A016S)
MOTHER_CODE_RE = re.compile(r"^[A-Z]\d{3}[A-Z]$")


# ── Shared helpers ────────────────────────────────────────────────────────────

def norm(s: str) -> str:
    """Normalise a string for case-insensitive comparison (strip + lower)."""
    return (s or "").strip().lower()


def detect_delimiter(sample: str) -> str:
    """
    Identify the CSV delimiter from a short text sample.

    Parameters
    ----------
    sample : str   First ~2000 characters of the file.

    Returns
    -------
    str   First delimiter found among ',', ';', tab, '|'; defaults to ';'.
    """
    for d in [",", ";", "\t", "|"]:
        if d in sample:
            return d
    return ";"


def read_csv_rows(
    path: Path, delimiter: str = ";"
) -> Tuple[List[str], List[List[str]]]:
    """
    Read a CSV and return (header_row, data_rows).

    Parameters
    ----------
    path      : Path   Absolute path to the file.
    delimiter : str    Field separator.

    Returns
    -------
    (list[str], list[list[str]])   Empty lists if the file is empty.
    """
    raw  = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    if not raw:
        return [], []
    r    = csv.reader(raw, delimiter=delimiter)
    rows = list(r)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def find_col_idx(header: List[str], exact_name: str) -> int:
    """
    Return the index of *exact_name* in *header* (case-insensitive).

    Parameters
    ----------
    header     : list[str]   CSV header row.
    exact_name : str         Column name to locate.

    Returns
    -------
    int   Zero-based column index.

    Raises
    ------
    ValueError   If not found.
    """
    target = norm(exact_name)
    for i, h in enumerate(header):
        if norm(h) == target:
            return i
    raise ValueError(f"Column '{exact_name}' not found in {header}")


def parse_number(x: str) -> Optional[float]:
    """
    Parse a numeric string, accepting both period and comma as decimal separators.

    Comma-decimal format (e.g., "2,666666667") is common in European-locale
    questionnaire exports.

    Parameters
    ----------
    x : str   Raw cell value.

    Returns
    -------
    float or None   None if the string is empty or unparseable.
    """
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    s = s.replace(",", ".")   # normalise decimal separator
    try:
        return float(s)
    except Exception:
        return None


# ── Date parsing ──────────────────────────────────────────────────────────────

# Format strings tried in order; covers ISO 8601, European DD.MM.YYYY,
# and common timestamp variants from mobile app exports.
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
]


def try_parse_date(s: str) -> Optional[date]:
    """
    Attempt to parse a string as a calendar date.

    Tries Python's built-in ISO parser first (handles fractional seconds
    and UTC offsets), then falls through the format list above.

    Parameters
    ----------
    s : str   Raw date/time string from an app CSV.

    Returns
    -------
    datetime.date or None   None if all formats fail.
    """
    if s is None:
        return None
    txt = str(s).strip()
    if not txt:
        return None

    try:
        return datetime.fromisoformat(txt.replace("Z", "+00:00")).date()
    except Exception:
        pass

    for fmt in _DT_FORMATS:
        try:
            return datetime.strptime(txt, fmt).date()
        except Exception:
            pass

    return None


def find_participant_file(folder: Path, mother_code: str) -> Optional[Path]:
    """
    Locate a participant's app CSV, with a case-insensitive fallback.

    The primary lookup is case-sensitive (expected convention).  If it fails,
    the folder is scanned for a case-insensitive stem match to handle systems
    where filenames were lowercased during transfer.

    Parameters
    ----------
    folder      : Path   Data directory.
    mother_code : str    Participant ID (e.g., 'A016S').

    Returns
    -------
    Path or None   None if the file cannot be found.
    """
    p = folder / f"{mother_code}.csv"
    if p.exists():
        return p

    mc = mother_code.strip().lower()
    for f in folder.glob("*.csv"):
        if f.stem.strip().lower() == mc:
            return f
    return None


# ── Questionnaire loading ─────────────────────────────────────────────────────

def read_fti_from_questionnaire(
    path: Path, condition: str
) -> Dict[str, Dict[str, object]]:
    """
    Parse a questionnaire CSV and return FTI mean per participant.

    The three FTI items are identified by prefix matching on the header:
    any column whose header starts with "future intention to use:" is
    included.  The mean of the first three such columns is used.

    Parameters
    ----------
    path      : Path   Absolute path to the semicolon-delimited questionnaire.
    condition : str    'counter' or 'journal'.

    Returns
    -------
    dict[str, dict]
        MOTHER_CODE → {'condition': str, 'fti': float}.

    Raises
    ------
    ValueError
        If fewer than 3 FTI columns are found (indicates a malformed export).
    """
    header, data = read_csv_rows(path, delimiter=";")
    if not header:
        return {}

    i_code = find_col_idx(header, "MOTHER_CODE")

    # Identify the three FTI columns by prefix (prefix-matching is robust to
    # minor wording differences across questionnaire versions)
    fti_cols = [
        i for i, h in enumerate(header)
        if norm(h).startswith("future intention to use:")
    ]
    if len(fti_cols) < 3:
        raise ValueError(
            f"Expected ≥3 FTI columns starting with 'Future Intention to use:' "
            f"in {path.name}, but found {len(fti_cols)}."
        )
    fti_cols = fti_cols[:3]   # use the first three if more are present

    out: Dict[str, Dict[str, object]] = {}
    for row in data:
        if not row or i_code >= len(row):
            continue

        code = row[i_code].strip().upper()
        if not MOTHER_CODE_RE.match(code):
            continue

        # Parse all three FTI items; skip the participant if any is missing
        vals = []
        ok   = True
        for idx in fti_cols:
            if idx >= len(row):
                ok = False
                break
            v = parse_number(row[idx])
            if v is None:
                ok = False
                break
            vals.append(v)

        if not ok:
            continue

        fti = sum(vals) / 3.0   # mean across the three items
        out[code] = {"condition": condition, "fti": fti}

    return out


# ── App data: active-day counting ────────────────────────────────────────────

def unique_dates_counter(app_file: Path) -> List[date]:
    """
    Count unique active days from a Counter-condition app export.

    Resolution priority:
      1. Separate year/month/day integer columns (most unambiguous).
      2. A "date" column, or the first column as fallback.

    Parameters
    ----------
    app_file : Path   Participant's Counter CSV.

    Returns
    -------
    list[date]   Sorted unique active calendar dates.
    """
    delimiter = detect_delimiter(
        app_file.read_text(encoding="utf-8-sig", errors="replace")[:2000]
    )
    header, data = read_csv_rows(app_file, delimiter=delimiter)
    if not header:
        return []

    hn         = [norm(h) for h in header]
    dates_set  = set()

    # Priority 1: explicit integer columns
    if "year" in hn and "month" in hn and "day" in hn:
        iy, im, iday = hn.index("year"), hn.index("month"), hn.index("day")
        for row in data:
            if max(iy, im, iday) >= len(row):
                continue
            try:
                dates_set.add(
                    date(int(str(row[iy]).strip()),
                         int(str(row[im]).strip()),
                         int(str(row[iday]).strip()))
                )
            except Exception:
                continue
        return sorted(dates_set)

    # Priority 2: a column named "date", else column 0
    idx = hn.index("date") if "date" in hn else 0
    for row in data:
        if idx >= len(row):
            continue
        d = try_parse_date(row[idx])
        if d is not None:
            dates_set.add(d)
    return sorted(dates_set)


def unique_dates_journal(app_file: Path) -> List[date]:
    """
    Count unique active days from a Journal-condition app export.

    Per the agreed schema for this study, the date is always in the
    **second column** (index 1), following the participant-ID column.

    Parameters
    ----------
    app_file : Path   Participant's Journal CSV.

    Returns
    -------
    list[date]   Sorted unique active calendar dates.
    """
    delimiter = detect_delimiter(
        app_file.read_text(encoding="utf-8-sig", errors="replace")[:2000]
    )
    header, data = read_csv_rows(app_file, delimiter=delimiter)
    if not header or len(header) < 2:
        return []

    dates_set = set()
    for row in data:
        if 1 >= len(row):
            continue
        d = try_parse_date(row[1])
        if d is not None:
            dates_set.add(d)
    return sorted(dates_set)


# ── Correlation and permutation test ─────────────────────────────────────────

def pearson_r(xs: List[float], ys: List[float]) -> Optional[float]:
    """
    Compute Pearson r for two equal-length lists.

    Returns None if n < 2 or if either variable has zero variance
    (correlation is undefined in both cases).

    Parameters
    ----------
    xs, ys : list[float]   Paired observations.

    Returns
    -------
    float in [−1, 1] or None.
    """
    n = len(xs)
    if n != len(ys) or n < 2:
        return None
    mx   = sum(xs) / n
    my   = sum(ys) / n
    num  = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def permutation_p_value(
    xs: List[float],
    ys: List[float],
    n_perm: int = 10_000,
    seed:   int = 42,
) -> Optional[float]:
    """
    Estimate a two-sided permutation p-value for Pearson r.

    The null hypothesis is that xs and ys are independent.  Under H0,
    the labels of ys are exchangeable.  The test statistic is |r|;
    the p-value is the proportion of permutations where |r_perm| ≥ |r_obs|.

    Add-one (Laplace) smoothing is applied so the p-value is never exactly 0:
        p = (count + 1) / (n_perm + 1)
    This is appropriate for small n where the resolution of the permutation
    distribution is limited (Phipson & Smyth, 2010).

    Parameters
    ----------
    xs, ys  : list[float]   Paired observations.
    n_perm  : int           Number of permutations (default 10,000).
    seed    : int           RNG seed for reproducibility (default 42).

    Returns
    -------
    float in (0, 1] or None if r_obs is undefined.
    """
    r_obs = pearson_r(xs, ys)
    if r_obs is None:
        return None
    r_obs = abs(r_obs)

    rng     = random.Random(seed)
    ys_copy = ys[:]
    count   = 0

    for _ in range(n_perm):
        rng.shuffle(ys_copy)
        r_perm = pearson_r(xs, ys_copy)
        if r_perm is None:
            continue
        if abs(r_perm) >= r_obs:
            count += 1

    return (count + 1) / (n_perm + 1)   # add-one smoothing


def summarize_relation(label: str, xs: List[float], ys: List[float]) -> None:
    """
    Print Pearson r and permutation p-value for one group to stdout.

    Reports 'insufficient variance/data' when r cannot be computed (n < 2
    or zero-variance variable) so the caller never raises silently.

    Parameters
    ----------
    label : str           Condition label for the printed line.
    xs, ys : list[float]  Active-day counts (x) and FTI means (y).
    """
    r = pearson_r(xs, ys)
    p = permutation_p_value(xs, ys, n_perm=10_000)
    n = len(xs)
    if r is None or p is None:
        print(f"{label}: n={n} (insufficient variance/data for correlation)")
        return
    print(f"{label}: n={n}, Pearson r={r:.3f}, permutation p={p:.4f}")


# ── Main routine ──────────────────────────────────────────────────────────────

def main() -> None:
    """
    Load data, build the merged table, and run correlations.

    Steps
    -----
    1. Parse questionnaire CSVs for MOTHER_CODE → FTI mean.
    2. For each participant, find their app CSV and count unique active days.
    3. Print the merged per-participant table (for manual verification).
    4. Save the merged table to CSV.
    5. Run Pearson r + permutation test: overall, Counter only, Journal only.
    6. Report any missing app files.
    """
    if not Q_COUNTER.exists() or not Q_JOURNAL.exists():
        raise FileNotFoundError("Missing counter.csv or journal.csv next to the script.")
    if not DIR_COUNTER.exists() or not DIR_JOURNAL.exists():
        raise FileNotFoundError("Missing data_counter/ or data_journal/ folders.")

    # ── 1. Load FTI metadata ──────────────────────────────────────────────────
    meta: Dict[str, Dict[str, object]] = {}
    meta.update(read_fti_from_questionnaire(Q_COUNTER, "counter"))
    meta.update(read_fti_from_questionnaire(Q_JOURNAL, "journal"))

    # ── 2. Compute active days ────────────────────────────────────────────────
    merged_rows   = []
    missing_files = []

    for code in sorted(meta.keys()):
        cond     = str(meta[code]["condition"])
        fti      = float(meta[code]["fti"])
        folder   = DIR_COUNTER if cond == "counter" else DIR_JOURNAL
        app_file = find_participant_file(folder, code)

        if app_file is None:
            missing_files.append(f"{cond}:{code}")
            continue

        days = len(
            unique_dates_counter(app_file)
            if cond == "counter"
            else unique_dates_journal(app_file)
        )
        merged_rows.append({
            "mother_code": code,
            "condition":   cond,
            "active_days": days,
            "fti_mean":    fti,
        })

    # ── 3. Print per-participant table ────────────────────────────────────────
    print("\nMerged table (active days + FTI mean):")
    for r in merged_rows:
        print(
            f"  {r['mother_code']}: {r['condition']}, "
            f"active_days={r['active_days']}, fti_mean={r['fti_mean']:.3f}"
        )

    # ── 4. Save merged CSV ────────────────────────────────────────────────────
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["mother_code", "condition", "active_days", "fti_mean"]
        )
        w.writeheader()
        for r in merged_rows:
            w.writerow(r)
    print(f"\nSaved: {OUT_CSV.resolve()}")

    # ── 5. Correlation analyses ───────────────────────────────────────────────
    xs_all = [float(r["active_days"]) for r in merged_rows]
    ys_all = [float(r["fti_mean"])    for r in merged_rows]

    print("\nRelationship: active days vs Future Intention to Use (FTI mean of 3 items)")
    summarize_relation("Overall",             xs_all, ys_all)

    for cond in ["counter", "journal"]:
        xs = [float(r["active_days"]) for r in merged_rows if r["condition"] == cond]
        ys = [float(r["fti_mean"])    for r in merged_rows if r["condition"] == cond]
        summarize_relation(f"{cond.capitalize()} only", xs, ys)

    # ── 6. Missing-file report ────────────────────────────────────────────────
    if missing_files:
        print("\nMissing app files (excluded from all analyses):")
        for x in missing_files:
            print(f"  - {x}")


if __name__ == "__main__":
    main()
