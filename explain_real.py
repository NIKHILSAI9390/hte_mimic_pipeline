"""C3 on real data: which subgroups benefit from early vasopressor?"""
import numpy as np, pandas as pd
from econml.grf import CausalForest
from sklearn.tree import DecisionTreeRegressor, export_text

d = pd.read_csv("cohort_clean.csv")
COVS = ["age","female","sofa","sapsii","charlson","heart_rate_mean","map",
        "resp_rate_mean","temp","spo2_mean","creatinine","wbc","platelets","bilirubin"]
d = d.dropna(subset=["treated","mortality_28d"]).copy()
for c in COVS: d[c] = d[c].fillna(d[c].median())

X = d[COVS].values; A = d["treated"].values.astype(float); Y = d["mortality_28d"].values.astype(float)
f = CausalForest(n_estimators=2000, min_samples_leaf=8, max_features=0.9,
                 max_samples=0.5, random_state=0)
f.fit(X, A, Y)
tau = f.predict(X).ravel()

# 1. Variable importance — what drives the effect?
imp = pd.Series(f.feature_importances_, index=COVS).sort_values(ascending=False)
print("Effect-driver ranking (what modifies the treatment effect):")
for c,v in imp.items(): print(f"  {c:18s} {v:.3f}")

# 2. Surrogate rules — readable subgroups
surr = DecisionTreeRegressor(max_depth=3, min_samples_leaf=500, random_state=0)
surr.fit(X, tau)
print("\nSubgroup rules (who benefits vs is harmed):")
print(export_text(surr, feature_names=COVS))

# 3. Profile the benefiters vs harmed
d["tau"] = tau
ben = d[d.tau < 0]; harm = d[d.tau > 0]
print(f"\nBENEFIT group (n={len(ben)}, {len(ben)/len(d):.1%}) vs HARMED (n={len(harm)}):")
for c in ["age","sofa","sapsii","charlson","map","creatinine","wbc"]:
    print(f"  {c:12s} benefit {ben[c].mean():7.1f} | harmed {harm[c].mean():7.1f}")
print(f"  mortality    benefit {ben.mortality_28d.mean():7.1%} | harmed {harm.mortality_28d.mean():7.1%}")

# Surrogate restricted to patients with tau < median (where benefit lives)
low = tau < np.median(tau)
surr2 = DecisionTreeRegressor(max_depth=3, min_samples_leaf=200, random_state=0)
surr2.fit(X[low], tau[low])
print("\nSubgroup rules WITHIN lower-tau half (where benefit lives):")
print(export_text(surr2, feature_names=COVS))

# And: what distinguishes strong benefiters (bottom 10%)?
strong = d[d.tau <= d.tau.quantile(0.10)]
print(f"\nSTRONG benefiters (bottom 10%, n={len(strong)}, mean tau {strong.tau.mean():+.3f}):")
for c in ["age","sofa","sapsii","charlson","resp_rate_mean","creatinine","map"]:
    print(f"  {c:16s} {strong[c].mean():7.1f}  (cohort {d[c].mean():7.1f})")