"""Clip physiologically impossible values before analysis."""
import pandas as pd, numpy as np

df = pd.read_csv("cohort.csv")

# clinically plausible ranges (values outside -> NaN, then median-imputed later)
LIMITS = {
    "age": (18, 100), "sofa": (0, 24), "sapsii": (0, 163), "charlson": (0, 20),
    "heart_rate_mean": (20, 220), "map": (20, 150), "resp_rate_mean": (4, 60),
    "temp": (30, 43), "spo2_mean": (50, 100),
    "creatinine": (0.1, 20), "wbc": (0.1, 200),
    "platelets": (1, 1500), "bilirubin": (0.1, 60),
}
before = df.notna().sum().sum()
for c,(lo,hi) in LIMITS.items():
    n_bad = ((df[c] < lo) | (df[c] > hi)).sum()
    if n_bad: print(f"  {c}: {n_bad} impossible values -> NaN")
    df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan

df.to_csv("cohort_clean.csv", index=False)
print(f"\nsaved cohort_clean.csv ({len(df)} rows)")