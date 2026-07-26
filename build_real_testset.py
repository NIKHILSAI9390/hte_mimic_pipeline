"""Real MIMIC test set: N patients with true structured covariates + admission-section
text. Output stays LOCAL — it contains credentialed note text."""
import gzip, json, pandas as pd
from sections import admission_text

N = 100
COVS = ["age","female","sofa","sapsii","charlson","heart_rate_mean","map",
        "resp_rate_mean","temp","spo2_mean","creatinine","wbc","platelets","bilirubin"]

d = pd.read_csv("cohort_clean.csv").dropna(subset=["treated","mortality_28d"])
d = d.dropna(subset=COVS)                       # complete ground truth only
picks = d.sample(N, random_state=0).set_index("hadm_id")
want = set(picks.index.astype(int))

out = []
with gzip.open("notes/notes_unique", "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line); h = int(r["hadm_id"])
        if h not in want: continue
        adm = admission_text(r["text"])
        if len(adm) < 500: continue             # skip degenerate parses
        out.append({
            "id": f"real_{h}", "tier": "real", "support": "real",
            "text": adm,
            "truth": {c: float(picks.loc[h, c]) for c in COVS},
            "truth_mode": {c: "unknown" for c in COVS},   # provenance unknown for real notes
        })

json.dump(out, open("real_testset.json", "w"), indent=2)
print(f"wrote real_testset.json: {len(out)} patients")
import numpy as np
print(f"  admission text: median {int(np.median([len(o['text']) for o in out]))} chars")