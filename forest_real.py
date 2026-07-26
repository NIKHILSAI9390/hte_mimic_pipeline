"""B — Causal forest HTE on real MIMIC sepsis cohort. Median-impute to keep all rows."""
import numpy as np, pandas as pd
import pickle
from econml.grf import CausalForest

df = pd.read_csv("cohort_clean.csv")
COVS = ["age","female","sofa","sapsii","charlson","heart_rate_mean","map",
        "resp_rate_mean","temp","spo2_mean","creatinine","wbc","platelets","bilirubin"]

# median-impute missing covariates (keep all 22,747)
d = df.dropna(subset=["treated","mortality_28d"]).copy()
for c in COVS:
    d[c] = d[c].fillna(d[c].median())
print(f"n = {len(d)}, treated = {d['treated'].sum()} ({d['treated'].mean():.1%})")

X = d[COVS].values
A = d["treated"].values.astype(float)
Y = d["mortality_28d"].values.astype(float)

# tuned forest (same settings as synthetic study)
forest = CausalForest(n_estimators=2000, min_samples_leaf=8,
                      max_features=0.9, max_samples=0.5, random_state=0, inference=True)
forest.fit(X, A, Y)
tau = forest.predict(X).ravel()

print(f"\nHTE estimates (effect of early vasopressor on 28d mortality):")
print(f"  mean tau (= ATE): {tau.mean():+.4f}")
print(f"  tau range: [{tau.min():+.3f}, {tau.max():+.3f}]")
print(f"  % patients with harmful effect (tau>0): {(tau>0).mean():.1%}")
print(f"  % patients with beneficial effect (tau<0): {(tau<0).mean():.1%}")

# save for downstream deferral/interpolation
d["tau_hat"] = tau
d.to_csv("cohort_with_tau.csv", index=False)
pickle.dump(forest, open("forest.pkl", "wb"))
print("saved forest.pkl")
print("\nsaved cohort_with_tau.csv")