# Analysis Repository

Quantitative analysis scripts for a two-condition within-week study comparing a Counter app and a Journal app on engagement and wellbeing outcomes.

---

Participant files are named `<MOTHER_CODE>.csv` (e.g., `A016S.csv`). One file equals one participant.

---

## File Structure
```
URM-Analysis-quantitative/
│
├── App/
│   ├── data_counter/                    # Raw Counter app CSV exports (one file per participant)
│   │   └── <MOTHER_CODE>.csv
│   ├── data_counter_clean/              # Output of clean_counter_data.py
│   │   └── <MOTHER_CODE>_clean.csv
│   ├── data_journal/                    # Journal app CSV exports (one file per participant)
│   │   └── <MOTHER_CODE>.csv
│   │
│   ├── analysis_motivation.py           # Active participants per date per condition → daily_participants_compare.csv
│   ├── boxplot_active_days.py           # Boxplot: unique active days → active_days_boxplot.png
│   ├── clean_counter_data.py            # Normalise raw Counter exports → data_counter_clean/
│   ├── compare_total_entries.py         # Average total entries per participant → total_entries_compare.csv
│   ├── daily_active_rate_diagram.py     # Daily active rate line plot → daily_active_rate_by_day_combined.png/.pdf
│   ├── significance_test.py             # Welch t-test, Mann-Whitney U, Hedges' g, CLES
│   └── stats_active_days_alpha05.py     # Same tests with explicit alpha = 0.05 decisions
│
├── Questionnaire/                       # Questionnaire response data
│   └── <questionnaire_data>.csv         # Semicolon-delimited, decimal comma
│
└── README.md
```
``` *(close fence)*

---

## Scripts

| Script | Question answered |
|---|---|
| `clean_counter_data.py` | Normalise raw Counter exports to a canonical two-column format (type, date) |
| `analysis_motivation.py` | How many participants were active on each calendar date, per condition? |
| `compare_total_entries.py` | What is the average total number of entries per participant per condition? |
| `make_boxplot_active_days.py` | Boxplot of unique active days: Counter vs. Journal |
| `diagram_compare_active_days.py` | Daily active rate over the study period (line plot, both conditions) |
| `active_days_gender.py` | Active days stratified by condition and gender (4-group boxplot) |
| `entries_gender.py` | Total entries stratified by condition and gender (4-group boxplot) |
| `diagram_average_WHO5.py` | WHO-5 wellbeing trajectories across three timepoints |
| `significance_test.py` | Welch t-test, Mann-Whitney U, Hedges' g, and CLES for active days |
| `stats_active_days_alpha05.py` | Same tests with explicit reject/fail-to-reject decisions at alpha = 0.05 |
| `FTI_active_days.py` | Pearson r and permutation test: active days vs. Future Intention to Use |

---

## Key operationalisations

**Active day.** A calendar date on which a participant logged at least one entry. Multiple entries on the same date count once.

**Total entries.** Number of non-header rows in the participant's app CSV.

**FTI (Future Intention to Use).** Mean of three Likert items from the end-of-study questionnaire (columns starting with "Future Intention to use:").

**WHO-5 sum.** Sum of five wellbeing items (range 0-25), measured at Day 1 (Sum_START), Day 4 (Sum_MIDDLE), and Day 8 (Sum_END).

---

## Requirements

```
python     >= 3.9
pandas     >= 1.3
numpy      >= 1.21
matplotlib >= 3.5
scipy      >= 1.7
```

`compare_total_entries.py` and `FTI_active_days.py` use the standard library only and do not require third-party packages. Install all other dependencies with:

```bash
pip install pandas numpy matplotlib scipy
```

---

## Running the scripts

Each script is self-contained and reads from the paths relative to its own location. Run from the repository root:

```bash
python clean_counter_data.py        # run first if using raw Counter exports
python make_boxplot_active_days.py
python significance_test.py
# etc.
```

All scripts print results to stdout and save outputs (figures, CSVs) to the same directory or `./out/`.

---

## Data format reference

**Counter CSVs.** Date parsed from `year`/`month`/`day` integer columns (preferred), a `date` column, or the first column as a fallback.

**Journal CSVs.** Date expected in the second column (schema: `participant_id, date, ...`).

**Questionnaire CSVs.** Semicolon-delimited, decimal comma. Required columns: `MOTHER_CODE`, `GENDER` (1 = male, 2 = female). FTI and WHO-5 columns identified by header prefix.
