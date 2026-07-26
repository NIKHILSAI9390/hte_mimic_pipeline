"""M1 baseline: tabular-only propensity score matching. Reproduce paper's ATE ~0.055."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("cohort.csv")
COVS = ["age","female","sofa","sapsii","charlson","heart_rate_mean","map",
        "resp_rate_mean","temp","spo2_mean","creatinine","wbc","platelets","bilirubin"]

# drop rows missing any covariate or outcome (simple complete-case for M1)
d = df.dropna(subset=COVS+["treated","mortality_28d"]).copy()
print(f"cohort: {len(df)}, complete-case: {len(d)}")

X = d[COVS].values; A = d["treated"].values; Y = d["mortality_28d"].values

# propensity score via logistic regression
ps = LogisticRegression(max_iter=1000).fit(X, A).predict_proba(X)[:,1]

# 1:1 nearest-neighbor matching on PS (treated -> nearest control)
treated_idx = np.where(A==1)[0]; control_idx = np.where(A==0)[0]
nn = NearestNeighbors(n_neighbors=1).fit(ps[control_idx].reshape(-1,1))
_, m = nn.kneighbors(ps[treated_idx].reshape(-1,1))
matched_control = control_idx[m.ravel()]

# ATE = mean(Y_treated - Y_matched_control)
ate = Y[treated_idx].mean() - Y[matched_control].mean()
print(f"M1 PSM ATE (early vasopressor -> 28d mortality): {ate:.4f}")
print(f"  (paper M1: ~0.055)")
print(f"  treated mortality: {Y[treated_idx].mean():.3f}")
print(f"  matched control  : {Y[matched_control].mean():.3f}")