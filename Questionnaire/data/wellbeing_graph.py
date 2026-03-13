import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# -----------------------------
# Paths (same folder as notebook)
# -----------------------------
COUNTER_PATH = "data/counter.csv"
JOURNAL_PATH = "data/journal.csv"

# -----------------------------
# Load data (your files are semicolon-separated; decimals may use comma)
# -----------------------------
counter = pd.read_csv(COUNTER_PATH, sep=";", decimal=",", encoding="utf-8")
journal = pd.read_csv(JOURNAL_PATH, sep=";", decimal=",", encoding="utf-8-sig")

# Add explicit condition label (safer than relying on APP_TYPE strings)
counter["Condition"] = "Counter"
journal["Condition"] = "Journal"

df = pd.concat([counter, journal], ignore_index=True)

# -----------------------------
# Select wellbeing sum columns (WHO-5 sum 0–25)
# -----------------------------
time_cols = {
    "START": "Sum_START",
    "MIDDLE": "Sum_MIDDLE",
    "END": "Sum_END",
}

missing = [c for c in time_cols.values() if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected wellbeing columns: {missing}\nColumns found: {list(df.columns)}")

# Long format
long_df = df[["Condition"] + list(time_cols.values())].copy()
long_df = long_df.melt(
    id_vars=["Condition"],
    value_vars=list(time_cols.values()),
    var_name="TimeVar",
    value_name="WellbeingSum"
)

# Map to ordered time labels
inv_map = {v: k for k, v in time_cols.items()}
long_df["Time"] = long_df["TimeVar"].map(inv_map)
long_df["Time"] = pd.Categorical(long_df["Time"], categories=["START", "MIDDLE", "END"], ordered=True)

# Ensure numeric
long_df["WellbeingSum"] = pd.to_numeric(long_df["WellbeingSum"], errors="coerce")

# Drop missing values (if any)
long_df = long_df.dropna(subset=["WellbeingSum"])

# -----------------------------
# Summary stats per condition/time
# -----------------------------
summary = (
    long_df.groupby(["Condition", "Time"], observed=True)
    .agg(
        n=("WellbeingSum", "count"),
        mean=("WellbeingSum", "mean"),
        sd=("WellbeingSum", "std")
    )
    .reset_index()
)

# 95% CI using normal approximation (fine for plotting; small n -> interpret cautiously)
summary["sem"] = summary["sd"] / np.sqrt(summary["n"])
summary["ci95"] = 1.96 * summary["sem"]

# -----------------------------
# Plot: two lines (Counter vs Journal) + error bars (95% CI)
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))

for cond in ["Counter", "Journal"]:
    s = summary[summary["Condition"] == cond].sort_values("Time")
    ax.errorbar(
        s["Time"].astype(str),
        s["mean"],
        yerr=s["ci95"],
        marker="o",
        linewidth=2,
        capsize=4,
        label=f"{cond} (n={int(s['n'].max())})"
    )

ax.set_title("Wellbeing (WHO-5 sum) across the week by condition")
ax.set_xlabel("Timepoint")
ax.set_ylabel("WHO-5 Sum (0–25)")
ax.set_ylim(0, 25)
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
ax.legend()

plt.tight_layout()
plt.show()

# Optional: print the table used for the plot
summary.sort_values(["Condition", "Time"]).reset_index(drop=True)
