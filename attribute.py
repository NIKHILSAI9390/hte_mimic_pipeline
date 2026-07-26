"""Attribution: for each covariate the note did NOT pin down, measure how much
of the prediction's instability it accounts for, and what the estimate becomes
at each end of its interval. This is what makes a deferral actionable."""
import numpy as np
from gate import gate_one, forest, COVS, THR, INTEGER_COVS

def _sample(iv, n, rng):
    X = np.empty((n, len(COVS)))
    for j, c in enumerate(COVS):
        lo, hi = iv[c]
        col = np.full(n, lo, float) if lo == hi else rng.uniform(lo, hi, n)
        if c in INTEGER_COVS:
            col = np.round(col)
        X[:, j] = col
    return X

def _spread(iv, n=150, seed=0):
    taus = forest.predict(_sample(iv, n, np.random.default_rng(seed))).ravel()
    return float(np.percentile(taus,95) - np.percentile(taus,5)), taus

def attribute(intervals, n=150, seed=0):
    """Rank the unknown covariates by how much pinning them collapses the spread."""
    base_spread, _ = _spread(intervals, n, seed)
    free = [c for c in COVS if intervals[c][0] != intervals[c][1]]
    rows = []
    for c in free:
        lo, hi = intervals[c]
        mid = round((lo+hi)/2) if c in INTEGER_COVS else (lo+hi)/2
        iv_p = {**intervals, c: [mid, mid]}
        sp, _ = _spread(iv_p, n, seed)

        # what the estimate becomes at each end of this covariate's range
        tau_lo = float(np.median(forest.predict(
            _sample({**intervals, c: [lo, lo]}, n, np.random.default_rng(seed))).ravel()))
        tau_hi = float(np.median(forest.predict(
            _sample({**intervals, c: [hi, hi]}, n, np.random.default_rng(seed))).ravel()))

        rows.append({
            "covariate": c,
            "interval": [lo, hi],
            "spread_if_known": sp,
            "spread_reduction": base_spread - sp,
            "pct_of_instability": (base_spread - sp) / base_spread if base_spread else 0.0,
            "tau_at_low": tau_lo,
            "tau_at_high": tau_hi,
            "flips_sign": (tau_lo > 0) != (tau_hi > 0),
            "gate_if_known": gate_one(iv_p)["decision"],
        })
    rows.sort(key=lambda r: -r["spread_reduction"])
    return {"base_spread": base_spread, "drivers": rows}

if __name__ == "__main__":
    import json, sys
    from collections import Counter
    served = json.load(open(sys.argv[1] if len(sys.argv)>1 else "served_note_only.json"))
    fixable, top_driver = [], Counter()
    for r in served:
        iv = r["pred_interval"]["_intervals"]
        g = gate_one(iv)
        if g["decision"] != "defer": continue
        a = attribute(iv, n=400)
        if not a["drivers"]: continue
        top_driver[a["drivers"][0]["covariate"]] += 1
        for row in a["drivers"]:
            if row["gate_if_known"] == "answer":
                fixable.append((r["id"], r["tier"], r["support"], row))
                break

    print("most common top driver of instability among deferred patients:")
    for k, v in top_driver.most_common(): print(f"  {k:18s} {v}")

    print(f"\ndeferrals resolvable by supplying ONE covariate: {len(fixable)}")
    for pid, tier, sup, row in fixable[:8]:
        print(f"  {pid:24s} ({tier}/{sup})  supply {row['covariate']} "
              f"[{row['interval'][0]:.1f},{row['interval'][1]:.1f}]  "
              f"-> tau {row['tau_at_low']:+.3f}..{row['tau_at_high']:+.3f}")