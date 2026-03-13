"""
analysis_motivation.py  (original: compare_daily_participants.py)
───────────────────────────────────────────────────────────────────────────────
Answers the research question:
    "How many participants were active on a given calendar date, per condition?"

For each calendar date observed in either data folder, the script reports:
    - counter_participants   : unique Counter participants active that day
    - journal_participants   : unique Journal participants active that day
    - total_participants     : unique participants across *both* groups (global)
    - overlap_same_filename  : participants whose ID appears in *both* groups
                               and who were active on the same date
                               (useful for detecting cross-contamination or
                               shared IDs across conditions)

Usage
-----
    python analysis_motivation.py

Data format
-----------
Counter CSVs  (./data_counter/<participant_id>.csv)
    Date parsed from year/month/day columns, a "date" column, or column 0.

Journal CSVs  (./data_journal/<participant_id>.csv)
    Date parsed from the second column (schema: participant_id, date, ...).

Operationalisation
------------------
One file = one participant (filename stem = participant ID).
Multiple entries on the same calendar date for the same participant count
as a single active day (de-duplication applied before aggregation).

Output
------
daily_participants_compare.csv   — date-indexed summary table
stdout                           — the same table printed to the console

Reproducibility note
--------------------
No randomness; output is fully deterministic given the input CSVs.

Requirements
------------
    python  >= 3.9
    pandas  >= 1.3
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

BASE_DIR    = Path(__file__).resolve().parent
COUNTER_DIR = BASE_DIR / "data_counter"
JOURNAL_DIR = BASE_DIR / "data_journal"
OUT_CSV     = BASE_DIR / "daily_participants_compare.csv"


# ── I/O helpers ───────────────────────────────────────────────────────────────

def read_csv_flexible(path: Path) -> pd.DataFrame:
    """
    Load a CSV while being tolerant of non-standard delimiters.

    Tries comma, semicolon, tab, and pipe in order; falls back to pandas'
    built-in csv.Sniffer (``sep=None``) as a last resort.

    Parameters
    ----------
    path : Path
        Absolute path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with no type coercion beyond pandas' defaults.
    """
    for sep in [",", ";", "\t", "|"]:
        try:
            return pd.read_csv(path, sep=sep)
        except Exception:
            pass
    return pd.read_csv(path, sep=None, engine="python")


# ── Date extraction ───────────────────────────────────────────────────────────

def extract_dates_counter(df: pd.DataFrame) -> pd.Series:
    """
    Extract a datetime Series from a Counter-condition participant file.

    Resolution priority (most- to least-preferred):
      1. Separate "year", "month", "day" integer columns.
      2. A single "date" string column.
      3. The first column (fallback for unlabelled exports).

    All parsing failures are coerced to NaT and dropped downstream.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from a Counter CSV.

    Returns
    -------
    pd.Series of dtype datetime64[ns]
    """
    # Build a case-insensitive column lookup to handle varied export styles
    cols = {c.strip().lower(): c for c in df.columns}

    # Priority 1: explicit numeric year/month/day — least ambiguous format
    if {"year", "month", "day"}.issubset(cols.keys()):
        dt = pd.to_datetime(
            dict(
                year  = pd.to_numeric(df[cols["year"]],  errors="coerce"),
                month = pd.to_numeric(df[cols["month"]], errors="coerce"),
                day   = pd.to_numeric(df[cols["day"]],   errors="coerce"),
            ),
            errors="coerce",
        )
        return dt

    # Priority 2: a column literally named "date"
    if "date" in cols:
        return pd.to_datetime(df[cols["date"]], errors="coerce", dayfirst=True)

    # Priority 3: assume the first column holds dates (rare but observed in wild)
    return pd.to_datetime(df.iloc[:, 0], errors="coerce", dayfirst=True)


def extract_dates_journal(df: pd.DataFrame) -> pd.Series:
    """
    Extract a datetime Series from a Journal-condition participant file.

    Expected schema: ``(participant_id, date, ...)``.
    The date is in the **second column** (index 1).

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from a Journal CSV.

    Returns
    -------
    pd.Series of dtype datetime64[ns]

    Raises
    ------
    ValueError
        If the DataFrame has fewer than 2 columns, indicating a malformed file.
    """
    if df.shape[1] < 2:
        raise ValueError(
            "Journal CSV has < 2 columns; cannot read date from 2nd column."
        )
    return pd.to_datetime(df.iloc[:, 1], errors="coerce", dayfirst=True)


# ── Per-participant date extraction ───────────────────────────────────────────

def participant_dates_from_folder(folder: Path, kind: str) -> pd.DataFrame:
    """
    Build a long-format table of unique (participant, date) pairs from a folder.

    Each CSV file is treated as one participant (filename stem = ID). Rows with
    unparseable dates are silently dropped. Multiple entries on the same
    calendar date for the same participant are de-duplicated to a single row.

    Parameters
    ----------
    folder : Path
        Directory containing per-participant CSV files.
    kind : {'counter', 'journal'}
        Selects the appropriate date-extraction function.

    Returns
    -------
    pd.DataFrame with columns ['participant', 'date', 'kind']
        One row per (participant × unique calendar date) observation.

    Raises
    ------
    FileNotFoundError
        If *folder* does not exist.
    ValueError
        If *kind* is not 'counter' or 'journal'.
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    rows = []
    for csv_path in sorted(folder.glob("*.csv")):
        participant = csv_path.stem       # filename stem = participant ID
        df = read_csv_flexible(csv_path)

        if kind == "counter":
            dt = extract_dates_counter(df)
        elif kind == "journal":
            dt = extract_dates_journal(df)
        else:
            raise ValueError("kind must be 'counter' or 'journal'")

        dt = dt.dropna()
        if dt.empty:
            continue

        # De-duplicate: one active-day entry per participant per calendar date
        for d in pd.Series(dt.dt.date).dropna().unique():
            rows.append((participant, d, kind))

    return pd.DataFrame(rows, columns=["participant", "date", "kind"])


# ── Main routine ──────────────────────────────────────────────────────────────

def main() -> None:
    """
    Aggregate daily participant counts and save the summary table.

    Steps
    -----
    1. Extract unique (participant, date) pairs for each condition.
    2. Count active participants per calendar date per condition.
    3. Count globally unique participants per date (across both conditions).
    4. Count participants whose ID appears in both conditions on the same date
       (overlap column — useful for flagging shared IDs or crossover effects).
    5. Write the merged summary to CSV and print it to stdout.
    """
    # ── 1. Extract per-condition (participant, date) pairs ────────────────────
    counter_pd = participant_dates_from_folder(COUNTER_DIR, "counter")
    journal_pd = participant_dates_from_folder(JOURNAL_DIR, "journal")

    # ── 2. Daily counts per condition ─────────────────────────────────────────
    counter_daily = (
        counter_pd.groupby("date")["participant"].nunique().rename("counter_participants")
        if not counter_pd.empty
        else pd.Series(dtype="int64", name="counter_participants")
    )
    journal_daily = (
        journal_pd.groupby("date")["participant"].nunique().rename("journal_participants")
        if not journal_pd.empty
        else pd.Series(dtype="int64", name="journal_participants")
    )

    # ── 3. Global unique participants per date (across both conditions) ────────
    both = pd.concat([counter_pd, journal_pd], ignore_index=True)
    if not both.empty:
        # Prefix IDs with condition to avoid false matches across groups
        both["participant_global"] = both["kind"] + "::" + both["participant"]
        total_daily = (
            both.groupby("date")["participant_global"]
            .nunique()
            .rename("total_participants")
        )
    else:
        total_daily = pd.Series(dtype="int64", name="total_participants")

    # ── 4. Same-ID overlap per date ───────────────────────────────────────────
    # Participants whose filename stem matches across conditions and who were
    # active on the same day — non-zero values may warrant investigation.
    if not counter_pd.empty and not journal_pd.empty:
        overlap = (
            counter_pd.merge(journal_pd, on=["date", "participant"], how="inner")
            .groupby("date")["participant"]
            .nunique()
            .rename("overlap_same_filename")
        )
    else:
        overlap = pd.Series(dtype="int64", name="overlap_same_filename")

    # ── 5. Merge, clean, and export ───────────────────────────────────────────
    summary = (
        pd.concat([counter_daily, journal_daily, total_daily, overlap], axis=1)
        .fillna(0)
        .astype(int)
        .sort_index()
    )
    summary.index = summary.index.astype("datetime64[ns]")   # consistent datetime index

    print(summary)
    summary.to_csv(OUT_CSV, index=True)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
