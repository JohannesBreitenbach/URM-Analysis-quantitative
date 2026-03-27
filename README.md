# Analysis Repository

Quantitative analysis scripts for a two-condition within-week study comparing a Counter app and a Journal app on engagement and wellbeing outcomes.

---

Participant files are named `<MOTHER_CODE>.csv` (e.g., `A016S.csv`). One file equals one participant.

---

## File Structure
```
URM-Analysis-quantitative/
│
├── App/                            # Participant data exported from the apps
│   ├── Counter/                    # Counter app CSV exports (one file per participant)
│   │   └── <MOTHER_CODE>.csv
│   └── Journal/                    # Journal app CSV exports (one file per participant)
│       └── <MOTHER_CODE>.csv
│
├── Questionnaire/                  # Questionnaire response data
│   └── questionnaire_data.csv      # Semicolon-delimited, decimal comma
│
├── clean_counter_data.py           # Normalise raw Counter exports
├── analysis_motivation.py          # Active participants per date per condition
├── compare_total_entries.py        # Average total entries per participant per condition
├── make_boxplot_active_days.py     # Boxplot: unique active days
├── diagram_compare_active_days.py  # Line plot: daily active rate over study period
├── active_days_gender.py           # Active days by condition and gender
├── entries_gender.py               # Total entries by condition and gender
├── diagram_average_WHO5.py         # WHO-5 wellbeing trajectories
├── significance_test.py            # Welch t-test, Mann-Whitney U, Hedges' g, CLES
├── stats_active_days_alpha05.py    # Same tests with alpha = 0.05 decisions
├── FTI_active_days.py              # Pearson r: active days vs. Future Intention to Use
│
└── README.md
```
``` *(closing fence for the code block above)*

---

## Scripts

| Script | Question answered |
| --- | --- |
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

... *(rest of README unchanged)*
```

---

**To apply this via terminal:**
```bash
# Open README in your editor
nano README.md   # or code README.md / vim README.md

# After saving, commit and push
git add README.md
git commit -m "Add file structure to README"
git push origin main
```

> **Note:** I based the file structure on what's visible in the repo (`App/`, `Questionnaire/` folders and the scripts in the README table). If the internal layout of `App/` or `Questionnaire/` differs (e.g. the CSVs live at the root of `App/` rather than in subfolders), just adjust the tree accordingly before pushing.