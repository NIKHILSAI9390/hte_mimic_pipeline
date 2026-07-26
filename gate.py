"""Three-signal deferral gate. All signals computed over the SAMPLED covariate
draws (not the interval midpoint), with ABSOLUTE thresholds so a single note can
be gated alone. Reads only extracted covariates — never tier/support labels.

  distance  : median k-NN distance of the sampled points to the training cohort
              (out-of-support)
  variance  : median forest tree-variance over the sampled points
              (estimator-uncertain)
  sign_flip : fraction of the tau distribution on the minority side
              (extraction-fragile; self-contained)
"""
import json, pickle, sys, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

COVS = ["age","female","sofa","sapsii","charlson","heart_rate_mean","map",
        "resp_rate_mean","temp","spo2_mean","creatinine","wbc","platelets","bilirubin"]

INTEGER_COVS = {"age", "female", "sofa", "sapsii", "charlson"}
forest = pickle.load(open("forest.pkl","rb"))

_d = pd.read_csv("cohort_clean.csv").dropna(subset=["treated","mortality_28d"]).copy()
for c in COVS: _d[c] = _d[c].fillna(_d[c].median())
_scaler = StandardScaler().fit(_d[COVS].values)
_nn = NearestNeighbors(n_neighbors=15).fit(_scaler.transform(_d[COVS].values))

def _sample_box(iv, n, rng):
    """Draw n realistic points inside the interval box (same as the predictor)."""
    X = np.empty((n, len(COVS)))
    for j, c in enumerate(COVS):
        lo, hi = iv[c]
        col = np.full(n, lo, float) if lo == hi else rng.uniform(lo, hi, n)
        if c in INTEGER_COVS:
            col = np.round(col)
        X[:, j] = col
    return X

def compute_signals(intervals, n_draws=200, seed=0):
    """intervals: {cov:[lo,hi]}. Returns the three signals over sampled points."""
    rng = np.random.default_rng(seed)
    X = _sample_box(intervals, n_draws, rng)

    # distance: each sampled point's k-NN distance into the cohort; take the median
    dist,_ = _nn.kneighbors(_scaler.transform(X))
    dscore = 0.5*dist.mean(1) + 0.5*dist[:,0]
    distance = float(np.median(dscore))

    # variance: forest tree-variance at each sampled point; take the median SE
    _, var = forest.predict_and_var(X)
    se = np.sqrt(np.maximum(var.ravel(), 0))
    variance = float(np.median(se))

    # sign-flip: minority-side mass of the tau distribution
    taus = forest.predict(X).ravel()
    frac_harm = float((taus > 0).mean())
    sign_minority = min(frac_harm, 1 - frac_harm)

    return {"distance": distance, "variance": variance,
            "sign_minority": sign_minority, "frac_harm": frac_harm,
            "tau_mean": float(taus.mean())}

# ---- ABSOLUTE thresholds, calibrated on realistic served-to-cohort geometry ----
# Reference = the cohort's own patients scored the SAME served-to-cohort way
# (leave-one-out distance / self variance), so the cutoff means "further/less
# certain than a normal patient", not "further than the internal packing".
def calibrate_thresholds(dist_pct=95, var_pct=95, sign_thr=0.15):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(_d), size=min(3000, len(_d)), replace=False)
    Xr = _d[COVS].values[idx]
    Xs = _scaler.transform(Xr)
    nn16 = NearestNeighbors(n_neighbors=16).fit(_scaler.transform(_d[COVS].values))
    dist,_ = nn16.kneighbors(Xs); dist = dist[:,1:]
    dscore = 0.5*dist.mean(1) + 0.5*dist[:,0]

    # served-style variance: median SE over a small sampled box around each patient
    def boxed(x, n=50, w=0.05):
        XX = rng.uniform(x*(1-w), x*(1+w), (n, len(x)))
        _, v = forest.predict_and_var(XX)
        return np.median(np.sqrt(np.maximum(v.ravel(),0)))
    se = np.array([boxed(Xr[i]) for i in range(len(Xr))])

    return {"DIST_THR": float(np.percentile(dscore, dist_pct)),
            "VAR_THR":  float(np.percentile(se, var_pct)),
            "SIGN_THR": sign_thr}

import os
if os.path.exists("thresholds.json"):
    THR = json.load(open("thresholds.json"))
else:
    THR = calibrate_thresholds()

def gate_one(intervals):
    s = compute_signals(intervals)
    reasons = []
    if s["distance"]      >= THR["DIST_THR"]: reasons.append("out-of-support")
    if s["variance"]      >= THR["VAR_THR"]:  reasons.append("estimator-uncertain")
    if s["sign_minority"] >= THR["SIGN_THR"]: reasons.append("extraction-fragile")
    return {"decision": "defer" if reasons else "answer",
            "reasons": reasons, **s}

if __name__ == "__main__":
    print("absolute thresholds (calibrated on held-out cohort geometry):")
    for k, v in THR.items(): print(f"  {k}: {v:.4f}")