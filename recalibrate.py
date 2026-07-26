"""Recalibrate DIST_THR / VAR_THR so the threshold's reference matches the serving
measurement procedure.

The bug: thresholds are the 95th percentile of REAL COHORT POINTS measured to the
cohort. But serving measures UNIFORMLY-SAMPLED INTERVAL POINTS. Different
quantities — the same category error as before, one level up.

Fix: wrap real cohort patients in synthetic intervals drawn from the SAME
provenance mix the real notes produce, sample them the SAME way, and take the
95th percentile of that. Then "defer" means "further out / less certain than 95%
of normal patients served the same way".
"""
import json, pickle, numpy as np, pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

COVS = ["age","female","sofa","sapsii","charlson","heart_rate_mean","map",
        "resp_rate_mean","temp","spo2_mean","creatinine","wbc","platelets","bilirubin"]

INTEGER_COVS = {"age", "female", "sofa", "sapsii", "charlson"}

forest = pickle.load(open("forest.pkl","rb"))
CAL = json.load(open("calibration.json"))

d = pd.read_csv("cohort_clean.csv").dropna(subset=["treated","mortality_28d"]).copy()
for c in COVS: d[c] = d[c].fillna(d[c].median())
X_all = d[COVS].values
scaler = StandardScaler().fit(X_all)
nn = NearestNeighbors(n_neighbors=15).fit(scaler.transform(X_all))

# same cutoffs as calibrate.py's DEFS — value -> category
BUCKET = {
 "creatinine":[(1.3,"renal_normal"),(3.0,"renal_impairment"),(1e12,"renal_failure")],
 "wbc":[(4,"wbc_low"),(11.0001,"wbc_normal"),(1e12,"wbc_high")],
 "platelets":[(100,"plt_low"),(400.0001,"plt_normal"),(1e12,"plt_high")],
 "bilirubin":[(1.2,"bili_normal"),(3,"bili_mild"),(1e12,"bili_marked")],
 "temp":[(36,"temp_low"),(38,"temp_normal"),(1e12,"temp_high")],
 "map":[(65,"bp_shock"),(100.0001,"bp_stable"),(1e12,"bp_high")],
 "heart_rate_mean":[(60,"hr_low"),(100.0001,"hr_normal"),(1e12,"hr_high")],
 "resp_rate_mean":[(12,"rr_low"),(22.0001,"rr_normal"),(1e12,"rr_high")],
 "spo2_mean":[(90,"o2_severe"),(95,"o2_mild"),(1e12,"o2_normal")],
 "age":[(50,"age_young"),(70,"age_middle"),(1e12,"age_old")],
}
def bucket_of(cov, v):
    if cov not in BUCKET: return None
    for cut, name in BUCKET[cov]:
        if v < cut: return name
    return BUCKET[cov][-1][1]

# provenance mix measured from the ACTUAL extraction output
def provenance_mix(path="extraction_openai_gpt_5.json"):
    res = json.load(open(path, encoding="utf-8"))
    mix = {}
    for c in COVS:
        if c == "female": continue
        cnt = Counter(r["pred"][c]["mode"] for r in res)
        tot = sum(cnt.values())
        mix[c] = {k: v/tot for k, v in cnt.items()}
    return mix

MIX = provenance_mix()
print("provenance mix from extraction (fraction stated / described / absent):")
for c in ["creatinine","map","temp","age"]:
    print(f"  {c:16s} " + "  ".join(f"{k}={v:.0%}" for k,v in MIX[c].items()))

def simulate_intervals(x, rng, regime="note_only"):
    """Wrap a real cohort patient in the intervals a served note would produce."""
    iv = {}
    for j, c in enumerate(COVS):
        if c == "female":
            iv[c] = [x[j], x[j]]; continue
        if c in ("sofa","sapsii","charlson"):
            iv[c] = ([x[j], x[j]] if regime == "composites"
                     else CAL["_marginal"][c])
            continue
        r, cum, mode = rng.random(), 0.0, "absent"
        for k, p in MIX[c].items():
            cum += p
            if r <= cum: mode = k; break
        if mode == "stated":
            iv[c] = [float(x[j]), float(x[j])]
        elif mode == "described":
            iv[c] = CAL.get(c, {}).get(bucket_of(c, x[j]), CAL["_marginal"][c])
        else:
            iv[c] = CAL["_marginal"][c]
    return iv

def sample_box(iv, n, rng):
    X = np.empty((n, len(COVS)))
    for j, c in enumerate(COVS):
        lo, hi = iv[c]
        col = np.full(n, lo, float) if lo == hi else rng.uniform(lo, hi, n)
        if c in INTEGER_COVS:
            col = np.round(col)
        X[:, j] = col
    return X

def calibrate(n_patients=400, n_draws=100, pct=95, regime="note_only", seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_all), n_patients, replace=False)
    dists, vars_ = [], []
    for i in idx:
        iv = simulate_intervals(X_all[i], rng, regime)
        X = sample_box(iv, n_draws, rng)
        dd, _ = nn.kneighbors(scaler.transform(X))
        dists.append(np.median(0.5*dd.mean(1) + 0.5*dd[:, 0]))
        _, v = forest.predict_and_var(X)
        vars_.append(np.median(np.sqrt(np.maximum(v.ravel(), 0))))
    return float(np.percentile(dists, pct)), float(np.percentile(vars_, pct)), \
           np.array(dists), np.array(vars_)

for regime in ["note_only", "composites"]:
    dt, vt, dd, vv = calibrate(regime=regime)
    print(f"\n[{regime}]  n=400 simulated served patients")
    print(f"  distance  median {np.median(dd):.3f}  p95 {dt:.3f}  max {dd.max():.3f}")
    print(f"  variance  median {np.median(vv):.4f}  p95 {vt:.4f}  max {vv.max():.4f}")
    if regime == "note_only":
        json.dump({"DIST_THR": dt, "VAR_THR": vt, "SIGN_THR": 0.15},
                  open("thresholds.json","w"), indent=2)
        print("  -> wrote thresholds.json (note-only used as the deployment default)")