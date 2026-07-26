"""Calibration table: category -> empirical value interval from the real cohort.
PLACEHOLDER version (threshold-defined buckets). Replace with GPT-5-classified
real-note quantiles in the real-data phase.
"""
import pandas as pd, numpy as np, json

d = pd.read_csv("cohort_clean.csv")

# category -> boolean mask over the cohort, using clinical value ranges
DEFS = {
 "creatinine": {"renal_normal":   d.creatinine < 1.3,
                "renal_impairment":(d.creatinine>=1.3)&(d.creatinine<3.0),
                "renal_failure":   d.creatinine>=3.0},
 "wbc": {"wbc_low": d.wbc<4, "wbc_normal":(d.wbc>=4)&(d.wbc<=11), "wbc_high": d.wbc>11},
 "platelets": {"plt_low": d.platelets<100, "plt_normal":(d.platelets>=100)&(d.platelets<=400),
               "plt_high": d.platelets>400},
 "bilirubin": {"bili_normal": d.bilirubin<1.2, "bili_mild":(d.bilirubin>=1.2)&(d.bilirubin<3),
               "bili_marked": d.bilirubin>=3},
 "temp": {"temp_low": d.temp<36, "temp_normal":(d.temp>=36)&(d.temp<38), "temp_high": d.temp>=38},
 "map": {"bp_shock": d["map"]<65, "bp_stable":(d["map"]>=65)&(d["map"]<=100), "bp_high": d["map"]>100},
 "heart_rate_mean": {"hr_low": d.heart_rate_mean<60,
                     "hr_normal":(d.heart_rate_mean>=60)&(d.heart_rate_mean<=100),
                     "hr_high": d.heart_rate_mean>100},
 "resp_rate_mean": {"rr_low": d.resp_rate_mean<12,
                    "rr_normal":(d.resp_rate_mean>=12)&(d.resp_rate_mean<=22),
                    "rr_high": d.resp_rate_mean>22},
 "spo2_mean": {"o2_severe": d.spo2_mean<90, "o2_mild":(d.spo2_mean>=90)&(d.spo2_mean<95),
               "o2_normal": d.spo2_mean>=95},
 "age": {"age_young": d.age<50, "age_middle":(d.age>=50)&(d.age<70), "age_old": d.age>=70},
}

# interval = 10th-90th percentile of TRUE values among cohort patients in that bucket
LO, HI = 10, 90
table = {}
print(f"{'category':22s} {'n':>6s} {'p10':>8s} {'p50':>8s} {'p90':>8s}")
for cov, cats in DEFS.items():
    table[cov] = {}
    for cat, mask in cats.items():
        vals = d.loc[mask, cov].dropna()
        if len(vals) < 20:
            lo, hi = float(d[cov].min()), float(d[cov].max())   # too few -> full marginal
            flag = " (SPARSE-> marginal)"
        else:
            lo, hi = float(np.percentile(vals, LO)), float(np.percentile(vals, HI))
            flag = ""
        table[cov][cat] = [round(lo,2), round(hi,2)]
        med = float(vals.median()) if len(vals) else float("nan")
        print(f"{cat:22s} {len(vals):6d} {lo:8.2f} {med:8.2f} {hi:8.2f}{flag}")


# marginal fallback for absent/redacted fields (full cohort 10-90)
table["_marginal"] = {c: [round(float(np.percentile(d[c].dropna(),LO)),2),
                          round(float(np.percentile(d[c].dropna(),HI)),2)] for c in DEFS}

# composites: marginal from cohort (never described in notes; supplied or marginal)
for comp in ("sofa","sapsii","charlson"):
    s = d[comp].dropna()
    table["_marginal"][comp] = [round(float(np.percentile(s,LO)),2),
                                round(float(np.percentile(s,HI)),2)]
json.dump(table, open("calibration.json","w"), indent=2)
print("\nwrote calibration.json")