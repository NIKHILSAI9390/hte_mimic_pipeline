"""Interval predictor: extracted note -> per-covariate intervals -> forest sampled
across the joint box -> tau distribution. This is the serving core.

Covariate intervals by provenance:
  stated    -> [x, x]          (regex-verified point, zero width)
  described -> calibration[cov][category]   (cohort quantile band)
  redacted  -> calibration._marginal[cov]   (know it exists, not the value)
  absent    -> calibration._marginal[cov]   (know nothing)
Composites (sofa, sapsii, charlson) come from the structured channel if supplied,
else marginal.
"""
import json, numpy as np, pandas as pd
from econml.grf import CausalForest
import pickle

COVS = ["age","female","sofa","sapsii","charlson","heart_rate_mean","map",
        "resp_rate_mean","temp","spo2_mean","creatinine","wbc","platelets","bilirubin"]

INTEGER_COVS = {"age", "female", "sofa", "sapsii", "charlson"}
# ---- fit (or load) the frozen forest on the real cohort ----
def load_forest():
    return pickle.load(open("forest.pkl","rb"))
forest = load_forest()

CAL = json.load(open("calibration.json"))

def field_interval(cov, pred_field, structured=None):
    """Return [lo, hi] for one covariate from its extracted representation."""
    structured = structured or {}
    if cov in ("sofa","sapsii","charlson"):
        if cov in structured and structured[cov] is not None:
            v = float(structured[cov]); return [v, v]
        return CAL["_marginal"].get(cov, None) or _cohort_marginal(cov)
    mode = pred_field["mode"]
    if mode == "stated":
        v = float(pred_field["value"]); return [v, v]
    if mode == "described":
        cat = pred_field["category"]
        return CAL.get(cov, {}).get(cat, CAL["_marginal"][cov])
    return CAL["_marginal"][cov]        # redacted / absent

def _cohort_marginal(cov):
    return CAL["_marginal"][cov]

def predict_interval(pred, forest, structured=None, n_draws=200, seed=0):
    """Monte-Carlo the forest across the covariate box. Returns tau summary + widths."""
    rng = np.random.default_rng(seed)
    lo, hi = {}, {}
    for c in COVS:
        if c == "female":
            v = pred["female"]["value"]
            v = v if v is not None else round(rng.random())    # rare: sex missing
            lo[c] = hi[c] = float(v); continue
        iv = field_interval(c, pred.get(c, {"mode":"absent"}), structured)
        lo[c], hi[c] = float(iv[0]), float(iv[1])

    # sample uniformly within each covariate's interval
    INTEGER_COVS = {"age", "female", "sofa", "sapsii", "charlson"}

# inside predict_interval, replacing the sampling loop:
    X = np.empty((n_draws, len(COVS)))
    for j, c in enumerate(COVS):
        col = (np.full(n_draws, lo[c], float) if lo[c] == hi[c]
               else rng.uniform(lo[c], hi[c], n_draws))
        if c in INTEGER_COVS:
            col = np.round(col)
        X[:, j] = col
    taus = forest.predict(X).ravel()

    # midpoint prediction — round integer covariates too
    mid = np.array([[round((lo[c]+hi[c])/2) if c in INTEGER_COVS else (lo[c]+hi[c])/2
                     for c in COVS]])
    tau_mid = float(forest.predict(mid).ravel()[0])

    return {
        "tau_mid": tau_mid,
        "tau_mean": float(taus.mean()),
        "tau_median": float(np.percentile(taus, 50)),
        "tau_lo": float(np.percentile(taus,5)),
        "tau_hi": float(np.percentile(taus,95)),
        "tau_spread": float(np.percentile(taus,95)-np.percentile(taus,5)),
        "sign_flips": float(((taus>0).mean())*( (taus<0).mean()>0 )),  # >0 if both signs occur
        "frac_harm": float((taus>0).mean()),
        "_mid": {c:(lo[c]+hi[c])/2 for c in COVS},
        "widths": {c:(hi[c]-lo[c]) for c in COVS},
        "_intervals": {c:[lo[c], hi[c]] for c in COVS},
    }

if __name__ == "__main__":
    import sys
    res = json.load(open(sys.argv[1] if len(sys.argv)>1
                         else "extraction_openai_gpt_5.json", encoding="utf-8"))

    # --- regime 1: note-only (composites fall back to marginal) ---
    print(f"{'id':22s} {'tau_med':>8s} {'5-95 band':>16s} {'spread':>7s} {'%harm':>6s}")
    rows = []
    for r in res:
        out = predict_interval(r["pred"], forest, structured=None)
        rows.append({**r, "pred_interval": out})
        band = f"[{out['tau_lo']:+.3f},{out['tau_hi']:+.3f}]"
        print(f"{r['id']:22s} {out['tau_median']:+8.3f} {band:>16s} "
              f"{out['tau_spread']:7.3f} {out['frac_harm']:6.0%}")
    json.dump(rows, open("served_note_only.json","w"), indent=2)

    # --- regime 2: note + composites (supplied from ground truth) ---
    rows_c = []
    for r in res:
        comp = {k: r["truth"][k] for k in ("sofa","sapsii","charlson")}
        out = predict_interval(r["pred"], forest, structured=comp)
        rows_c.append({**r, "pred_interval": out})
    json.dump(rows_c, open("served_with_composites.json","w"), indent=2)

    # --- ablation summary: mean spread per regime ---
    import numpy as np
    s_note = np.mean([x["pred_interval"]["tau_spread"] for x in rows])
    s_comp = np.mean([x["pred_interval"]["tau_spread"] for x in rows_c])
    print(f"\nmean tau spread  note-only: {s_note:.4f}   note+composites: {s_comp:.4f}")
    print("wrote served_note_only.json and served_with_composites.json")