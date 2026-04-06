import numpy as np

def holm_bonferroni(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)          # indices that sort p ascending
    sorted_p = np.array(p_values)[order]

    adjusted = np.minimum(1.0, sorted_p * np.arange(m, 0, -1))  # multiply p_(i) by (m - i + 1)
    adjusted = np.maximum.accumulate(adjusted)             

    result = np.empty(m)
    result[order] = adjusted
    return result.tolist()


# Between-condition inferential tests reported in Study I.
# Welch t (p=.0015) is excluded, it tests the same hypothesis as the Mann-Whitney
labels = [
    "Mann-Whitney U  (active days)",
    "WHO-5 trajectory (Counter vs Journal)",   # replace with actual p if formally tested
    "Future intention (Counter vs Journal)",   # replace with actual p if formally tested
]
raw_p = [0.0031, 0.05, 0.60]  # <-- substitute real p-values for the last two

corrected = holm_bonferroni(raw_p)

print(f"{'Test':<42} {'p (raw)':>10} {'p_Holm':>10}")
print("-" * 64)
for label, p, pc in zip(labels, raw_p, corrected):
    print(f"{label:<42} {p:>10.4f} {pc:>10.4f}")
