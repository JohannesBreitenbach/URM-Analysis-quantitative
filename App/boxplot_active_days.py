"""
make_boxplot_active_days.py
───────────────────────────────────────────────────────────────────────────────
Compares participant-level "active day" counts between two app conditions
(Counter vs. Journal) and produces a publication-ready boxplot.

Usage
-----
Place participant CSV files in the two data directories (see Configuration),
then run:

    python make_boxplot_active_days.py

Output
------
active_days_boxplot.png   — 300 dpi figure (saved next to this script)
stdout                    — Descriptive statistics for each group

Data format assumptions
-----------------------
Counter CSVs  (./data_counter/<participant_id>.csv)
    One row per logged event. Date is parsed from:
      1. Separate "year", "month", "day" columns (preferred), OR
      2. A single "date" column, OR
      3. The first column (fallback).

Journal CSVs  (./data_journal/<participant_id>.csv)
    One row per journal entry. Date is expected in the **second column**
    (column index 1), following a participant-ID column.

    Both formats accept dayfirst dates (DD/MM/YYYY) and ISO 8601.

Active-day operationalisation
------------------------------
A calendar date is counted as "active" if at least one data row with a
parseable date exists for that date. Unique active days are counted per
participant file; one file = one participant.

Reproducibility note
--------------------
Randomness: none. Output is fully deterministic given the input CSVs.
Dependencies: see requirements below.

Requirements
------------
    python  >= 3.9
    pandas  >= 1.3
    matplotlib >= 3.5

───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ── Configuration ────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).resolve().parent   # script's own directory
COUNTER_DIR = BASE_DIR / "data_counter"         # one CSV per Counter participant
JOURNAL_DIR = BASE_DIR / "data_journal"         # one CSV per Journal participant
OUT_PNG     = BASE_DIR / "active_days_boxplot.png"


# ── I/O helpers ──────────────────────────────────────────────────────────────

def read_csv_flexible(path: Path) -> pd.DataFrame:
    """
    Load a CSV file while being tolerant of non-standard delimiters.

    Tries common separators (comma, semicolon, tab, pipe) in order before
    falling back to Python's csv.Sniffer via ``sep=None``. This handles the
    variety of export formats produced by different diary/logging apps.

    Parameters
    ----------
    path : Path
        Absolute path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame; no type coercion beyond what pandas applies by default.
    """
    for sep in [",", ";", "\t", "|"]:
        try:
            return pd.read_csv(path, sep=sep)
        except Exception:
            pass
    # Last resort: let pandas sniff the delimiter
    return pd.read_csv(path, sep=None, engine="python")


# ── Date extraction ───────────────────────────────────────────────────────────

def extract_dates_counter(df: pd.DataFrame) -> pd.Series:
    """
    Extract a Series of timestamps from a Counter-condition CSV.

    Resolution priority:
      1. Separate "year", "month", "day" integer columns  ← most reliable
      2. A single "date" string column
      3. The first column (fallback for unlabelled exports)

    All parsing errors are coerced to NaT and handled downstream.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame loaded from a Counter participant file.

    Returns
    -------
    pd.Series of dtype datetime64[ns]
    """
    # Build a case-insensitive lookup so column naming is not brittle
    cols = {c.strip().lower(): c for c in df.columns}

    # Priority 1: numeric year/month/day columns (most unambiguous)
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

    # Priority 3: assume first column contains dates
    return pd.to_datetime(df.iloc[:, 0], errors="coerce", dayfirst=True)


def extract_dates_journal(df: pd.DataFrame) -> pd.Series:
    """
    Extract a Series of timestamps from a Journal-condition CSV.

    The expected schema is ``(participant_id, date, ...)``, so the date lives
    in the second column (index 1). Raises if the file has fewer than two
    columns, which would indicate a malformed export.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame loaded from a Journal participant file.

    Returns
    -------
    pd.Series of dtype datetime64[ns]

    Raises
    ------
    ValueError
        If the DataFrame has fewer than 2 columns.
    """
    if df.shape[1] < 2:
        raise ValueError(
            "Journal CSV has < 2 columns; expected date in the 2nd column "
            "(schema: participant_id, date, ...)."
        )
    return pd.to_datetime(df.iloc[:, 1], errors="coerce", dayfirst=True)


# ── Per-participant aggregation ───────────────────────────────────────────────

def participant_day_counts(folder: Path, kind: str) -> pd.DataFrame:
    """
    Count unique active calendar days for every participant in *folder*.

    Each CSV file is treated as one participant (filename stem = participant ID).
    Rows with unparseable dates are silently dropped; only valid calendar dates
    contribute to the count.

    Parameters
    ----------
    folder : Path
        Directory containing per-participant CSV files.
    kind : {'counter', 'journal'}
        Selects the appropriate date-extraction function.

    Returns
    -------
    pd.DataFrame with columns ['participant', 'active_days']

    Raises
    ------
    FileNotFoundError
        If *folder* does not exist.
    ValueError
        If *kind* is not 'counter' or 'journal'.
    """
    if not folder.exists():
        raise FileNotFoundError(f"Data folder not found: {folder}")

    rows: list[tuple[str, int]] = []

    for csv_path in sorted(folder.glob("*.csv")):
        participant = csv_path.stem                     # filename = participant ID
        df          = read_csv_flexible(csv_path)

        if kind == "counter":
            dt = extract_dates_counter(df)
        elif kind == "journal":
            dt = extract_dates_journal(df)
        else:
            raise ValueError("kind must be 'counter' or 'journal'")

        dt          = dt.dropna()                       # discard unparseable rows
        active_days = pd.Series(dt.dt.date).dropna().nunique()   # unique calendar dates
        rows.append((participant, int(active_days)))

    return pd.DataFrame(rows, columns=["participant", "active_days"])


# ── Descriptive statistics ────────────────────────────────────────────────────

def print_stats(label: str, vals: pd.Series) -> None:
    """
    Print the descriptive statistics that underlie the boxplot for one group.

    Reports n, individual values, mean, median, SD (sample), min, Q1, Q3,
    IQR, and max. This mirrors the statistics typically reported in CHI
    papers alongside figures showing small-n distributions.

    Parameters
    ----------
    label : str
        Human-readable group label (printed as a header).
    vals  : pd.Series
        Numeric active-day counts for this group.
    """
    vals = vals.dropna().astype(float)
    n    = int(vals.shape[0])

    if n == 0:
        print(f"\n{label}: no valid values.")
        return

    mean   = vals.mean()
    median = vals.median()
    sd     = vals.std(ddof=1) if n > 1 else 0.0   # sample SD (Bessel's correction)
    q1     = vals.quantile(0.25)
    q3     = vals.quantile(0.75)
    iqr    = q3 - q1
    vmin   = vals.min()
    vmax   = vals.max()

    print(f"\n{label}")
    print(f"n = {n}")
    # Print raw values so readers can verify the figure by hand
    print(
        "values = "
        + ", ".join(
            str(int(v)) if float(v).is_integer() else str(v)
            for v in vals.tolist()
        )
    )
    print(f"mean = {mean:.2f}")
    print(f"median = {median:.2f}")
    print(f"sd = {sd:.2f}")
    print(f"min = {vmin:.2f}")
    print(f"q1 (25%) = {q1:.2f}")
    print(f"q3 (75%) = {q3:.2f}")
    print(f"iqr = {iqr:.2f}")
    print(f"max = {vmax:.2f}")


# ── Main routine ──────────────────────────────────────────────────────────────

def main() -> None:
    """
    Orchestrate data loading, statistics printing, and figure export.

    Steps
    -----
    1. Aggregate active days per participant for each condition.
    2. Print descriptive statistics to stdout (for supplemental reporting).
    3. Draw a grouped boxplot with overlaid individual data points.
       Individual points are shown because n is typically small in CHI
       studies (< 30 per group), where a boxplot alone can obscure the
       distribution [1].
    4. Save the figure as a 300 dpi PNG suitable for camera-ready submission.

    References
    ----------
    [1] Weissgerber, T. L. et al. (2015). Beyond bar and line graphs: time
        for a new data presentation paradigm. PLOS Biology, 13(4), e1002128.
        https://doi.org/10.1371/journal.pbio.1002128
    """
    # ── 1. Load and label both conditions ────────────────────────────────────
    counter = participant_day_counts(COUNTER_DIR, "counter").assign(group="Counter")
    journal = participant_day_counts(JOURNAL_DIR, "journal").assign(group="Journal")
    data    = pd.concat([counter, journal], ignore_index=True)

    if data.empty:
        print(
            "No data found. Ensure data_counter/ and data_journal/ "
            "each contain at least one CSV file."
        )
        return

    # ── 2. Print descriptive statistics ──────────────────────────────────────
    print_stats(
        "Counter group (active days per participant)",
        data.loc[data["group"] == "Counter", "active_days"],
    )
    print_stats(
        "Journal group (active days per participant)",
        data.loc[data["group"] == "Journal", "active_days"],
    )

    # ── 3. Build figure ───────────────────────────────────────────────────────
    groups = ["Counter", "Journal"]
    values = [
        data.loc[data["group"] == g, "active_days"].astype(float).tolist()
        for g in groups
    ]

    fig, ax = plt.subplots(figsize=(6, 4))

    # Standard boxplot: box = IQR, whiskers = 1.5×IQR, fliers shown
    ax.boxplot(values, labels=groups, showfliers=True)

    # Overlay individual participant points to make small-n structure visible
    for i, g in enumerate(groups, start=1):
        ys = data.loc[data["group"] == g, "active_days"].astype(float).values
        xs = [i] * len(ys)
        ax.scatter(xs, ys, alpha=0.7)

    ax.set_ylabel("Active days per participant")
    ax.set_title("Engagement by app type (unique active days)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    # ── 4. Export ─────────────────────────────────────────────────────────────
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300)   # 300 dpi meets ACM camera-ready requirements
    plt.close(fig)

    print(f"\nSaved: {OUT_PNG}")


if __name__ == "__main__":
    main()