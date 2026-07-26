"""Score extraction. Metrics: provenance accuracy, value accuracy, category
accuracy, hallucination rate, abstention correctness."""
import json, sys
from collections import defaultdict

sys.path.insert(0, ".")
from gen_synthetic_notes import bucket_of
       # reuse the generator's buckets

f = sys.argv[1]
res = json.load(open(f, encoding="utf-8"))
COVS = [c for c in res[0]["truth_mode"] if c != "female"]

def bucket(c, v):
    return bucket_of(c, v)  

M = defaultdict(lambda: defaultdict(int))
for r in res:
    for c in COVS:
        tm, pm = r["truth_mode"][c], r["pred"][c]["mode"]
        tv, pv = r["truth"][c], r["pred"][c]["value"]
        tier = r["tier"]

        # provenance (treat generator 'redacted' as 'not stated')
        tm_eff = "absent" if tm == "redacted" else tm
        M[tier]["prov_n"] += 1
        M[tier]["prov_ok"] += (pm == tm_eff) or (tm_eff=="absent" and pm in ("absent","described"))

        # hallucination: claimed stated when truth wasn't stated
        if tm_eff != "stated" and pm == "stated":
            M[tier]["halluc"] += 1
        M[tier]["halluc_n"] += 1

        # value accuracy on truly-stated fields
        if tm == "stated":
            M[tier]["val_n"] += 1
            if pv is not None and abs(pv - tv) <= max(0.05, 0.02*abs(tv)):
                M[tier]["val_ok"] += 1

        # category accuracy on described fields
        if tm == "described":
            M[tier]["cat_n"] += 1
            if r["pred"][c]["category"] == bucket(c, tv):
                M[tier]["cat_ok"] += 1

        # abstention: truly absent -> should not claim stated
        if tm == "absent":
            M[tier]["abs_n"] += 1
            M[tier]["abs_ok"] += (pm != "stated")

print(f"{'tier':6s} {'provenance':>11s} {'value':>10s} {'category':>10s} "
      f"{'halluc':>9s} {'abstain':>9s}")
for tier in ["easy","med","hard"]:
    m = M[tier]
    pct = lambda a,b: f"{m[a]/m[b]:.0%}" if m[b] else "  -"
    print(f"{tier:6s} {pct('prov_ok','prov_n'):>11s} {pct('val_ok','val_n'):>10s} "
          f"{pct('cat_ok','cat_n'):>10s} {pct('halluc','halluc_n'):>9s} "
          f"{pct('abs_ok','abs_n'):>9s}")