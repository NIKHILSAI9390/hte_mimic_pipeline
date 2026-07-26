"""Leaf-weight evidence base, corrected for subsampling + honest splitting."""
import pickle, numpy as np, pandas as pd

COVS = ["age","female","sofa","sapsii","charlson","heart_rate_mean","map",
        "resp_rate_mean","temp","spo2_mean","creatinine","wbc","platelets","bilirubin"]
INTEGER_COVS = {"age","female","sofa","sapsii","charlson"}

forest = pickle.load(open("forest.pkl","rb"))
_d = pd.read_csv("cohort_clean.csv").dropna(subset=["treated","mortality_28d"]).copy()
for c in COVS: _d[c] = _d[c].fillna(_d[c].median())
X_TRAIN = _d[COVS].values
A_TRAIN = _d["treated"].values.astype(float)
Y_TRAIN = _d["mortality_28d"].values.astype(float)
N, B = len(X_TRAIN), forest.n_estimators

print("leaf assignments...")
LEAVES = forest.apply(X_TRAIN)                       # (N, B)

print("building eligibility mask (subsample + honest estimation half)...")
sub = forest.get_subsample_inds()                    # (B, n_subsample)
sub = np.asarray(sub)
print(f"  subsample_inds shape {sub.shape}  (of {N} training patients)")

ELIGIBLE = np.zeros((N, B), dtype=bool)
for b in range(B):
    tree_inds = sub[b]
    est = forest.estimators_[b]
    if getattr(forest, "honest", False):
        tr_i, val_i = est.get_train_test_split_inds()
        eligible_ids = tree_inds[np.asarray(val_i)]   # honest ESTIMATION half
    else:
        eligible_ids = tree_inds
    ELIGIBLE[eligible_ids, b] = True
print(f"  mean eligible patients per tree: {ELIGIBLE.sum(0).mean():.0f}")

def leaf_weights(x):
    q = forest.apply(np.asarray(x, float).reshape(1, -1))[0]      # (B,)
    match = (LEAVES == q[None, :]) & ELIGIBLE
    sizes = np.maximum(match.sum(axis=0), 1)
    w = (match / sizes).sum(axis=1) / B
    s = w.sum()
    return w / s if s > 0 else w

def local_estimate(w):
    Abar, Ybar = w @ A_TRAIN, w @ Y_TRAIN
    num = w @ ((A_TRAIN - Abar) * (Y_TRAIN - Ybar))
    den = w @ ((A_TRAIN - Abar) ** 2)
    return float(num / den) if den > 1e-12 else float("nan")

def evidence(intervals, n_query=20, seed=0):
    """Weighted evidence base over ALL contributing training patients."""
    rng = np.random.default_rng(seed)
    W = np.zeros(N)
    for _ in range(n_query):
        x = np.empty(len(COVS))
        for j, c in enumerate(COVS):
            lo, hi = intervals[c]
            v = lo if lo == hi else rng.uniform(lo, hi)
            x[j] = round(v) if c in INTEGER_COVS else v
        W += leaf_weights(x)
    W /= n_query

    tr, ct = A_TRAIN == 1, A_TRAIN == 0
    wt, wc = W[tr], W[ct]
    def wmean(w, y): return float(w @ y / w.sum()) if w.sum() > 0 else float("nan")
    ess = lambda w: float(w.sum()**2 / (w**2).sum()) if (w**2).sum() > 0 else 0.0

    return {
        "n_contributing": int((W > 1e-9).sum()),
        "effective_n": ess(W),
        "effective_n_treated": ess(wt),
        "effective_n_control": ess(wc),
        "wmort_treated": wmean(wt, Y_TRAIN[tr]),
        "wmort_control": wmean(wc, Y_TRAIN[ct]),
        "naive_contrast": wmean(wt, Y_TRAIN[tr]) - wmean(wc, Y_TRAIN[ct]),
        "profile": {c: float(W @ _d[c].values / W.sum())
                    for c in ["age","sofa","sapsii","charlson","creatinine","map"]},
        "weights": W,
    }

if __name__ == "__main__":
    print("\n=== verification ===")
    rng = np.random.default_rng(0)
    idx = rng.choice(N, 8, replace=False)
    print(f"{'patient':>8s} {'forest.predict':>15s} {'leaf-weighted':>15s} {'diff':>9s}")
    diffs = []
    for i in idx:
        w = leaf_weights(X_TRAIN[i])
        fp = float(forest.predict(X_TRAIN[i].reshape(1,-1)).ravel()[0])
        le = local_estimate(w)
        diffs.append(abs(fp-le))
        print(f"{i:8d} {fp:+15.4f} {le:+15.4f} {abs(fp-le):9.4f}")
    print(f"  mean diff {np.mean(diffs):.5f}   max {np.max(diffs):.5f}")

    # concentration check — did the mass sharpen?
    w = leaf_weights(X_TRAIN[idx[0]])
    order = np.argsort(-w)
    for k in (50, 200, 1000):
        print(f"  top {k:5d} carry {w[order[:k]].sum():.1%} of weight")
    print(f"  patients with any weight: {(w > 1e-9).sum()}")
    
# --- worked example on a served patient ---
    import json, sys
    served = json.load(open(sys.argv[1] if len(sys.argv) > 1
                            else "served_note_only.json", encoding="utf-8"))
    r = served[0]
    e = evidence(r["pred_interval"]["_intervals"])
    print(f"\n=== evidence for {r['id']} ===")
    print(f"  estimate (sampled median): {r['pred_interval']['tau_median']:+.4f}")
    print(f"  contributing training patients: {e['n_contributing']}  "
          f"(effective n = {e['effective_n']:.0f})")
    print(f"  weighted 28-day mortality among similar patients:")
    print(f"    early vasopressor    {e['wmort_treated']:.1%}  "
          f"(effective n {e['effective_n_treated']:.0f})")
    print(f"    no early vasopressor {e['wmort_control']:.1%}  "
          f"(effective n {e['effective_n_control']:.0f})")
    print(f"    unadjusted contrast  {e['naive_contrast']:+.3f}")
    print(f"  weighted profile: " + ", ".join(f"{k} {v:.1f}" for k,v in e["profile"].items()))