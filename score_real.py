"""Score extraction on REAL notes. We know true covariate VALUES from the tables but
not whether the note stated/described/omitted each field, so we score:
  1. value accuracy where the extractor reports 'stated'
  2. INTERVAL COVERAGE where it reports 'described' — does the true value fall inside
     the calibration band? This validates extraction AND the calibration table at once.
  3. the provenance mix, comparable to the synthetic distribution
"""
import json, sys, numpy as np
from collections import Counter, defaultdict

CAL = json.load(open("calibration.json"))
res = json.load(open(sys.argv[1] if len(sys.argv)>1
                     else "extraction_qwen2_5_7b_instruct.json", encoding="utf-8"))
COVS = [c for c in res[0]["truth"] if c != "female"]

mix = defaultdict(Counter)
val = defaultdict(lambda: [0,0])       # [correct, n]
cov = defaultdict(lambda: [0,0])       # [inside interval, n]

for r in res:
    for c in COVS:
        p = r["pred"].get(c, {})
        m = p.get("mode", "absent")
        mix[c][m] += 1
        t = r["truth"][c]
        if m == "stated" and p.get("value") is not None:
            val[c][1] += 1
            if abs(p["value"] - t) <= max(0.1, 0.15*abs(t)):   # 15% tolerance
                val[c][0] += 1
        elif m == "described" and p.get("category"):
            band = CAL.get(c, {}).get(p["category"])
            if band:
                cov[c][1] += 1
                if band[0] <= t <= band[1]:
                    cov[c][0] += 1

print(f"patients: {len(res)}\n")
print(f"{'covariate':18s} {'stated':>8s} {'descr':>7s} {'absent':>7s} "
      f"{'val acc':>9s} {'interval cover':>15s}")
for c in COVS:
    tot = sum(mix[c].values())
    va = f"{val[c][0]}/{val[c][1]}" if val[c][1] else "-"
    cv = (f"{cov[c][0]}/{cov[c][1]} ({cov[c][0]/cov[c][1]:.0%})"
          if cov[c][1] else "-")
    print(f"{c:18s} {mix[c]['stated']/tot:8.0%} {mix[c]['described']/tot:7.0%} "
          f"{mix[c]['absent']/tot:7.0%} {va:>9s} {cv:>15s}")

tv = [sum(val[c][i] for c in COVS) for i in (0,1)]
tc = [sum(cov[c][i] for c in COVS) for i in (0,1)]
print(f"\noverall value accuracy   {tv[0]}/{tv[1]}" +
      (f" ({tv[0]/tv[1]:.0%})" if tv[1] else ""))
print(f"overall interval coverage {tc[0]}/{tc[1]}" +
      (f" ({tc[0]/tc[1]:.0%})" if tc[1] else "") + "   [target ~80% for 10-90 bands]")