# gate_eval.py
import json, sys, numpy as np
from gate import gate_one

served = json.load(open(sys.argv[1] if len(sys.argv)>1 else "served_note_only.json"))
out = [{**r, "gate": gate_one(r["pred_interval"]["_intervals"])} for r in served]

ans = [r for r in out if r["gate"]["decision"]=="answer"]
dfr = [r for r in out if r["gate"]["decision"]=="defer"]
print(f"answered {len(ans)}/{len(out)}  deferred {len(dfr)}/{len(out)}\n")

print("defer rate by support (post-hoc validation — gate never saw these):")
for s in ["typical","out_of_cohort","implausible"]:
    grp = [r for r in out if r["support"]==s]
    d = sum(r["gate"]["decision"]=="defer" for r in grp)
    print(f"  {s:14s} {d}/{len(grp)}  ({d/len(grp):.0%})")

from collections import Counter
c = Counter(reason for r in dfr for reason in r["gate"]["reasons"])
print("\ndeferral reasons:")
for k,v in c.most_common(): print(f"  {k:22s} {v}")

sa = np.mean([r["pred_interval"]["tau_spread"] for r in ans]) if ans else 0
sd = np.mean([r["pred_interval"]["tau_spread"] for r in dfr]) if dfr else 0
print(f"\nmean tau spread  answered {sa:.4f}  |  deferred {sd:.4f}")