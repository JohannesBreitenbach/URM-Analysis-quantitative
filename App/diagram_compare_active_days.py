"""
diagram_compare_active_days.py  (original: plot_daily_active_rate_by_day_combined.py)
───────────────────────────────────────────────────────────────────────────────
Plots daily active rates for both conditions (Counter and Journal) on a single
time-series figure with a study-day x-axis (Day 1, Day 2, …).

Research question addressed
---------------------------
"How does day-by-day engagement evolve over the study period for each app type?"

Operationalisations
-------------------
Active participant
    A participant is counted as active on a given day if their file contains
    at least one row with a parseable date on that calendar date.
    Multiple entries on the same date for the same participant count once.

Daily active rate
    active_participants_on_day  ÷  valid_participants_in_group
    where "valid" = having at least one parseable date across the study.
    Range: [0, 1].  A rate of 1.0 means every participant logged that day.

Day numbering
    Day 1 is the earliest calendar date observed across *both* groups.
    This ensures both lines share a common temporal reference, allowing
    direct visual comparison even if the two groups started on different dates.
    Days with zero active participants are included as 0.0 (not omitted),
    so the x-axis is contiguous.

Usage
-----
    python diagram_compare_active_days.py

Output
------
daily_active_rate_by_day_combined.png   — 300 dpi raster (ACM camera-ready)
daily_active_rate_by_day_combined.pdf   — vector for submission or editing

Reproducibility note
--------------------
No randomness; output is fully deterministic given the input CSVs.
Tick spacing adapts to study length: every day (≤14 days), every 2 days
(15–30 days), or every 5 days (> 30 days).

Requirements
------------
    python     >= 3.9
    pandas     >= 1.3
    matplotlib >= 3.5
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).resolve().parent
COUNTER_DIR = BASE_DIR / "data_counter"
JOURNAL_DIR = BASE_DIR / "data_journal"

OUT_PNG = BASE_DIR / "daily_active_rate_by_day_combined.png"
OUT_PDF = BASE_DIR / "daily_active_rate_by_day_combined.pdf"


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

    Resolution priority:
      1. Separate "year", "month", "day" integer columns.
      2. A single "date" string column.
      3. The first column (fallback for unlabelled exports).

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from a Counter CSV.

    Returns
    -------
    pd.Series of dtype datetime64[ns]; parse failures become NaT.
    """
    cols = {c.strip().lower(): c for c in df.columns}

    # Priority 1: explicit numeric columns — most unambiguous
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

    if "date" in cols:
        return pd.to_datetime(df[cols["date"]], errors="coerce", dayfirst=True)

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
    pd.Series of dtype datetime64[ns]; parse failures become NaT.

    Raises
    ------
    ValueError
        If the DataFrame has fewer than 2 columns.
    """
    if df.shape[1] < 2:
        raise ValueError(
            "Journal CSV has < 2 columns; expected date in 2nd column."
        )
    return pd.to_datetime(df.iloc[:, 1], errors="coerce", dayfirst=True)


# ── Per-participant date extraction ───────────────────────────────────────────

def participant_dates(folder: Path, kind: str) -> pd.DataFrame:
    """
    Build a long-format table of unique (participant, date) pairs.

    Dates are normalised to midnight Timestamps (datetime64[ns]) so they
    can be used directly as a DatetimeIndex for reindexing onto a full
    calendar range.

    Parameters
    ----------
    folder : Path
        Directory containing per-participant CSV files.
    kind : {'counter', 'journal'}
        Selects the appropriate date-extraction function.

    Returns
    -------
    pd.DataFrame with columns ['participant', 'date']
        De-duplicated: one row per (participant × unique calendar date).

    Raises
    ------
    FileNotFoundError
        If *folder* does not exist.
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    rows: list[tuple[str, pd.Timestamp]] = []
    for csv_path in sorted(folder.glob("*.csv")):
        participant = csv_path.stem
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

        # Normalise to midnight so dates align when used as an index
        unique_days = pd.Series(dt.dt.normalize()).dropna().unique()
        for d in unique_days:
            rows.append((participant, pd.Timestamp(d)))

    out = pd.DataFrame(rows, columns=["participant", "date"]).drop_duplicates()
    return out


# ── Rate computation ──────────────────────────────────────────────────────────

def daily_active_rate(pdays: pd.DataFrame) -> pd.Series:
    """
    Compute the daily active rate (proportion of valid participants active).

    active_rate(date) = active_participants(date) / total_valid_participants

    A participant is "valid" if they appear at least once in *pdays*.
    The denominator is constant across dates (group size), not just active days.

    Parameters
    ----------
    pdays : pd.DataFrame
        Unique (participant, date) pairs for one condition.

    Returns
    -------
    pd.Series
        Index: pd.Timestamp (one per observed date), values: float in [0, 1].
        Returns an empty Series if *pdays* is empty.
    """
    if pdays.empty:
        return pd.Series(dtype="float64")

    valid_participants = pdays["participant"].nunique()   # constant denominator
    daily_active = pdays.groupby("date")["participant"].nunique().sort_index()
    return daily_active / float(valid_participants)


def to_day_index(rate: pd.Series, global_start: pd.Timestamp) -> pd.Series:
    """
    Convert a date-indexed Series to a study-day-indexed Series (Day 1 = start).

    Day numbers are computed relative to *global_start* (shared across both
    groups) so the two lines are plotted on the same scale.

    Parameters
    ----------
    rate : pd.Series
        Date-indexed active-rate values.
    global_start : pd.Timestamp
        The first calendar date of the study (earliest date across all groups).

    Returns
    -------
    pd.Series indexed by int (day number, 1-based).
    """
    if rate.empty:
        return rate
    day_numbers = (rate.index.normalize() - global_start.normalize()).days + 1
    out = rate.copy()
    out.index = day_numbers.astype(int)
    return out


# ── Main routine ──────────────────────────────────────────────────────────────

def main() -> None:
    """
    Build and save the combined daily-active-rate time-series figure.

    Steps
    -----
    1. Extract unique (participant, date) pairs for each condition.
    2. Compute daily active rates (active ÷ group size).
    3. Reindex both series to a full calendar range so missing days appear
       as 0 rather than being omitted, preserving a contiguous x-axis.
    4. Convert date indices to study-day numbers (Day 1 = global start).
    5. Plot both lines on a single axis and export to PNG and PDF.
    """
    # ── 1. Extract (participant, date) pairs ──────────────────────────────────
    counter_pdays = participant_dates(COUNTER_DIR, "counter")
    journal_pdays = participant_dates(JOURNAL_DIR, "journal")

    # ── 2. Compute daily active rates ─────────────────────────────────────────
    counter_rate = daily_active_rate(counter_pdays)
    journal_rate = daily_active_rate(journal_pdays)

    if counter_rate.empty and journal_rate.empty:
        print("No usable data found in either folder.")
        return

    # ── 3. Establish global date range ────────────────────────────────────────
    # The x-axis spans the earliest to latest date observed across BOTH groups
    # so both lines share a common temporal origin (Day 1).
    all_dates = []
    if not counter_rate.empty:
        all_dates += [counter_rate.index.min(), counter_rate.index.max()]
    if not journal_rate.empty:
        all_dates += [journal_rate.index.min(), journal_rate.index.max()]

    global_start = min(all_dates)
    global_end   = max(all_dates)

    # Fill missing calendar days with 0.0 (no active participants that day)
    full_range = pd.date_range(
        global_start.normalize(), global_end.normalize(), freq="D"
    )
    counter_rate_full = counter_rate.reindex(full_range, fill_value=0.0)
    journal_rate_full = journal_rate.reindex(full_range, fill_value=0.0)

    # ── 4. Convert to study-day index ─────────────────────────────────────────
    counter_day = to_day_index(counter_rate_full, global_start)
    journal_day = to_day_index(journal_rate_full, global_start)

    # ── 5. Build and export figure ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(counter_day.index, counter_day.values, label="Counter (daily active rate)")
    ax.plot(journal_day.index, journal_day.values, label="Journal (daily active rate)")

    ax.set_xlabel("Study day")
    ax.set_ylabel("Daily active rate (active ÷ group size)")
    ax.set_ylim(0, 1)   # rate is bounded [0, 1]

    # Adaptive x-tick spacing — avoids over-crowding on long studies
    max_day = int(max(counter_day.index.max(), journal_day.index.max()))
    if max_day <= 14:
        ticks = list(range(1, max_day + 1))           # every day
    else:
        step  = 2 if max_day <= 30 else 5             # every 2 or 5 days
        ticks = list(range(1, max_day + 1, step))
        if ticks[-1] != max_day:
            ticks.append(max_day)                     # always label the last day

    ax.set_xticks(ticks)
    ax.set_xticklabels([f"Day {t}" for t in ticks], rotation=30, ha="right")

    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300)   # 300 dpi meets ACM camera-ready requirements
    plt.close(fig)

    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
