"""
clean_counter_data.py  (original: clean_all_counter_csvs.py)
───────────────────────────────────────────────────────────────────────────────
Normalises all raw Counter-condition CSVs to a canonical two-column format
suitable for downstream analysis and archival.

Transformation applied
----------------------
Input  : raw export with varied column layouts (see Data format below)
Output : two columns only — ``type`` and ``date`` — sorted by date descending.

Date normalisation
    Dates are standardised to ``DD.MM.YYYY`` regardless of the input format.
    This removes ambiguity that arises when files from different participants
    were exported with different locale settings.

Sort order
    Rows are sorted by date descending using a stable merge-sort so that the
    relative order of same-day entries is preserved from the original file.

Usage
-----
    python clean_counter_data.py

Data format (input)
-------------------
./data_counter/<participant_id>.csv
    Must contain either:
      (a) columns "year", "month", "day" (integers) AND "type", OR
      (b) a "date" column AND "type".
    Files missing "type" or a parseable date column raise ValueError and
    are reported as FAIL in the console summary.

Output
------
./data_counter_clean/<participant_id>_clean.csv
    UTF-8, LF line endings, no BOM.  One file per successfully cleaned input.

Reproducibility note
--------------------
No randomness; sort is deterministic (stable merge-sort on parsed datetime).
Output is fully reproducible given identical input files.

Requirements
------------
    python  >= 3.9
    pandas  >= 1.3
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


# ── Configuration ─────────────────────────────────────────────────────────────

IN_DIR  = Path(__file__).resolve().parent / "data_counter"        # raw exports
OUT_DIR = Path(__file__).resolve().parent / "data_counter_clean"  # cleaned output


# ── Per-file cleaning ─────────────────────────────────────────────────────────

def _clean_one_file(in_path: Path, out_dir: Path) -> Path:
    """
    Clean a single Counter-condition CSV and write it to *out_dir*.

    Steps
    -----
    1. Read the raw file with pandas (comma-separated, header assumed).
    2. Strip leading/trailing whitespace from column names.
    3. Construct a datetime Series from year/month/day columns or a "date"
       column; normalise to ``DD.MM.YYYY`` string format.
    4. Retain only ``type`` and ``date``; drop rows where either is NaN.
    5. Sort descending by date (stable, preserves same-day row order).
    6. Write cleaned file as UTF-8 CSV with Unix line endings.

    Parameters
    ----------
    in_path : Path
        Absolute path to the raw input CSV.
    out_dir : Path
        Directory where the cleaned output file will be written.

    Returns
    -------
    Path
        Absolute path to the written output file.

    Raises
    ------
    ValueError
        If the file lacks parseable date columns or the required ``type``
        column (both are reported as FAIL in ``main``).
    """
    df = pd.read_csv(in_path)

    # Normalise column names to remove accidental whitespace from exports
    df.columns = [c.strip() for c in df.columns]

    # ── Date construction ──────────────────────────────────────────────────────
    if {"year", "month", "day"}.issubset(df.columns):
        # Preferred path: explicit numeric columns are unambiguous
        dt = pd.to_datetime(
            dict(
                year  = pd.to_numeric(df["year"],  errors="coerce"),
                month = pd.to_numeric(df["month"], errors="coerce"),
                day   = pd.to_numeric(df["day"],   errors="coerce"),
            ),
            errors="coerce",
        )
        date_str = dt.dt.strftime("%d.%m.%Y")   # canonical DD.MM.YYYY
    elif "date" in df.columns:
        # Fallback: re-parse an existing date string and reformat for consistency
        dt = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        date_str = dt.dt.strftime("%d.%m.%Y")
    else:
        raise ValueError(
            f"{in_path.name}: expected columns (year, month, day) or (date). "
            f"Found: {list(df.columns)}"
        )

    if "type" not in df.columns:
        raise ValueError(f"{in_path.name}: missing required column 'type'.")

    # ── Build output DataFrame ────────────────────────────────────────────────
    out = pd.DataFrame({"type": df["type"], "date": date_str})
    out = out.dropna(subset=["type", "date"])   # drop rows with missing values

    # ── Sort descending by date (stable = preserves within-day order) ─────────
    sort_dt = pd.to_datetime(out["date"], format="%d.%m.%Y", errors="coerce")
    out = (
        out.assign(_sort_dt=sort_dt)
           .sort_values("_sort_dt", ascending=False, kind="mergesort")
           .drop(columns=["_sort_dt"])
    )

    # ── Write output ──────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{in_path.stem}_clean.csv"
    out.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    return out_path


# ── Main routine ──────────────────────────────────────────────────────────────

def main() -> None:
    """
    Iterate over all CSVs in IN_DIR, clean each, and report a summary.

    Files that raise exceptions (e.g. missing required columns) are logged
    as FAIL with the error message; all other files are processed
    independently so one malformed file does not abort the batch.
    """
    if not IN_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {IN_DIR}")

    csv_files = sorted(IN_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in: {IN_DIR}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ok, failed = 0, 0
    for p in csv_files:
        try:
            out_path = _clean_one_file(p, OUT_DIR)
            print(f"OK   {p.name} -> {out_path.name}")
            ok += 1
        except Exception as e:
            print(f"FAIL {p.name}: {e}")
            failed += 1

    print(f"\nDone. Success: {ok}, Failed: {failed}")


if __name__ == "__main__":
    main()
