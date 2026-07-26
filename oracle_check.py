"""ORACLE CHECK — serve every patient with their TRUE covariate vector.

Two purposes:
 1. Diagnostic: with zero-width intervals there is NO extraction uncertainty and
    NO sampling artifact. If the gate still defers most typical patients, the
    threshold/logic is wrong. If it defers ~5% (matching DIST_THR's design), the
    uniform-box sampling is confirmed as the cause of the inflated defer rate.
 2. Validation: forest.predict(true_vector) is the answer the forest WOULD give
    with perfect information. Serving error = |tau_served - tau_true|. Then ask
    the question the method exists for: do DEFERRED patients have higher serving
    error than ANSWERED ones?
"""
import json, sys, numpy as np
from gate import gate_one, forest, COVS

served_file = sys.argv[1] if len(sys.argv) > 1 else "served_note_only.json"
served = json.load(open(served_file, encoding="utf-8"))

# ---------- 1. oracle gate: true covariates, zero-width intervals ----------
oracle = []
for r in served:
    iv = {c: [float(r["truth"][c]), float(r["truth"][c])] for c in COVS}
    g = gate_one(iv)
    tau_true = float(forest.predict(np.array([[r["truth"][c] for c in COVS]])).ravel()[0])
    oracle.append({"id": r["id"], "support": r["support"], "tier": r["tier"],
                   "gate": g, "tau_true": tau_true,
                   "tau_served_mid": r["pred_interval"]["tau_mid"],
                   "tau_served_mean": r["pred_interval"]["tau_mean"],
                   "served_gate": None})

d = sum(o["gate"]["decision"] == "defer" for o in oracle)
print(f"=== ORACLE GATE (true covariates, no uncertainty) ===")
print(f"deferred {d}/{len(oracle)}\n")
print("defer rate by support:")
for s in ["typical", "out_of_cohort", "implausible"]:
    grp = [o for o in oracle if o["support"] == s]
    dd = sum(o["gate"]["decision"] == "defer" for o in grp)
    print(f"  {s:14s} {dd}/{len(grp)}  ({dd/len(grp):.0%})")

from collections import Counter
c = Counter(rn for o in oracle if o["gate"]["decision"]=="defer" for rn in o["gate"]["reasons"])
print("\noracle deferral reasons:")
for k, v in c.most_common(): print(f"  {k:22s} {v}")

# ---------- 2. serving error vs the served gate's decision ----------
print(f"\n=== SERVING ERROR ({served_file}) ===")
for r, o in zip(served, oracle):
    r["_tau_true"] = o["tau_true"]
    r["_err_mid"]  = abs(r["pred_interval"]["tau_mid"]  - o["tau_true"])
    r["_err_mean"] = abs(r["pred_interval"]["tau_mean"] - o["tau_true"])
    r["_err_median"] = abs(r["pred_interval"]["tau_median"] - o["tau_true"])
    r["_gate"] = gate_one(r["pred_interval"]["_intervals"])

ans = [r for r in served if r["_gate"]["decision"] == "answer"]
dfr = [r for r in served if r["_gate"]["decision"] == "defer"]

def mstat(rows, k): return np.mean([r[k] for r in rows]) if rows else float("nan")
print(f"answered {len(ans)}  deferred {len(dfr)}")
print(f"  mean |error| (midpoint est)  answered {mstat(ans,'_err_mid'):.4f} "
      f"| deferred {mstat(dfr,'_err_mid'):.4f}")
print(f"  mean |error| (box-mean est)  answered {mstat(ans,'_err_mean'):.4f} "
      f"| deferred {mstat(dfr,'_err_mean'):.4f}")
print(f"  mean |error| (sampled median) answered {mstat(ans,'_err_median'):.4f} "
      f"| deferred {mstat(dfr,'_err_median'):.4f}")
# same, restricted to physiologically meaningful patients
print("\n  (excluding 'implausible' — true tau is not meaningful for impossible vectors)")
ans2 = [r for r in ans if r["support"] != "implausible"]
dfr2 = [r for r in dfr if r["support"] != "implausible"]
print(f"  answered {len(ans2)}  deferred {len(dfr2)}")
print(f"  mean |error| (midpoint est)  answered {mstat(ans2,'_err_mid'):.4f} "
      f"| deferred {mstat(dfr2,'_err_mid'):.4f}")
print(f"  mean |error| (sampled median) answered {mstat(ans2,'_err_median'):.4f} "
      f"| deferred {mstat(dfr2,'_err_median'):.4f}")

# error by support level
print("\nmean |error| by support (sampled median):")
for s in ["typical", "out_of_cohort", "implausible"]:
    grp = [r for r in served if r["support"] == s]
    print(f"  {s:14s} {mstat(grp,'_err_median'):.4f}")