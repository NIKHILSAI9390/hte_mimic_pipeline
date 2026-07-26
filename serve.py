"""Output layer: assemble one structured record per patient, then verbalize it.

Every number is computed deterministically and lives in the record. The LLM only
phrases it — a validation pass rejects any number in the generated text that does
not appear in the record, so it cannot invent figures (same discipline as extraction).
"""
import json, os, re, pickle, sys, numpy as np
import requests

from gate import gate_one, COVS
from attribute import attribute
from evidence import evidence
from explain_save import rule_path

SURROGATE_R2 = 0.291          # disclose: the subgroup rule is approximate

PRETTY = {"age":"age","female":"sex","sofa":"SOFA","sapsii":"SAPS-II",
          "charlson":"Charlson index","heart_rate_mean":"heart rate","map":"MAP",
          "resp_rate_mean":"respiratory rate","temp":"temperature",
          "spo2_mean":"SpO2","creatinine":"creatinine","wbc":"WBC",
          "platelets":"platelets","bilirubin":"bilirubin"}

def build_record(r, structured_supplied=False):
    iv   = r["pred_interval"]["_intervals"]
    pi   = r["pred_interval"]
    gate = gate_one(iv)
    ev   = evidence(iv)
    att  = attribute(iv, n=400)

    # provenance: what the note gave us vs what we assumed
    read, assumed = {}, {}
    for c in COVS:
        p = r["pred"].get(c, {})
        if p.get("mode") == "stated":
            read[PRETTY[c]] = round(float(p["value"]), 2)
        elif p.get("mode") == "described":
            read[PRETTY[c]] = f"{p.get('category')} -> [{iv[c][0]}, {iv[c][1]}]"
        else:
            assumed[PRETTY[c]] = [iv[c][0], iv[c][1]]
    if structured_supplied:
        for c in ("sofa","sapsii","charlson"):
            assumed.pop(PRETTY[c], None)
            read[PRETTY[c]] = iv[c][0]

    # CHANGE 1: sex as a word, not a 0/1 code
    if PRETTY["female"] in read:
        read["sex"] = "female" if float(read.pop(PRETTY["female"])) == 1 else "male"
    elif PRETTY["female"] in assumed:
        assumed.pop(PRETTY["female"])
        read["sex"] = "not stated"

    mid = [round((iv[c][0]+iv[c][1])/2) if c in {"age","female","sofa","sapsii","charlson"}
           else (iv[c][0]+iv[c][1])/2 for c in COVS]
    # CHANGE 2: pretty names in the subgroup rule
    conds, leaf_val = rule_path(np.array(mid), names=[PRETTY[c] for c in COVS])

    rec = {
        "patient_id": r["id"],
        "decision": gate["decision"],
        "estimate": round(pi["tau_median"], 4),
        "band_low": round(pi["tau_lo"], 4),
        "band_high": round(pi["tau_hi"], 4),
        # CHANGE 3: percentage-point versions for clinical readability
        "estimate_pct_points": round(pi["tau_median"]*100, 1),
        "band_low_pct_points": round(pi["tau_lo"]*100, 1),
        "band_high_pct_points": round(pi["tau_hi"]*100, 1),
        "band_is_degenerate": bool(pi["tau_lo"] == pi["tau_hi"]),
        "pct_of_range_harmful": round(pi["frac_harm"] * 100, 1),
        "deferral_reasons": gate["reasons"],
        "evidence": {
            "contributing_patients": ev["n_contributing"],
            "effective_n": round(ev["effective_n"]),
            "effective_n_treated": round(ev["effective_n_treated"]),
            "effective_n_control": round(ev["effective_n_control"]),
            # CHANGE 4: units baked in as strings so they can't be dropped
            "observed_mortality_treated": f"{round(ev['wmort_treated']*100,1)}%",
            "observed_mortality_control": f"{round(ev['wmort_control']*100,1)}%",
            "unadjusted_contrast_pct_points": round(ev["naive_contrast"]*100, 1),
        },
        "read_from_note": read,
        "assumed_from_cohort": assumed,
        "subgroup_rule": " AND ".join(conds),
        "subgroup_rule_avg_effect": round(leaf_val, 4),
        "subgroup_rule_r2": SURROGATE_R2,
        "top_drivers": [
            {"covariate": PRETTY[d["covariate"]],
             "range": [round(d["interval"][0],2), round(d["interval"][1],2)],
             "pct_of_uncertainty": round(d["pct_of_instability"]*100, 1),
             "estimate_at_low_pct_points": round(d["tau_at_low"]*100, 1),
             "estimate_at_high_pct_points": round(d["tau_at_high"]*100, 1),
             "changes_direction": bool(d["flips_sign"]),
             "would_resolve": d["gate_if_known"] == "answer"}
            for d in att["drivers"][:3]],
    }
    return rec

# ---------------- verbalizer ----------------
SYSTEM = """You explain the output of a causal treatment-effect model to an ICU clinician.

You are given a JSON record. Write 3-5 short sentences in plain clinical prose.

STRICT RULES
- Use ONLY numbers that appear in the record. Never compute, round differently,
  or introduce any figure not present.
- The estimate is an absolute change in 28-day mortality risk from starting
  vasopressors early. Positive = higher mortality (harm). Negative = lower (benefit).
- If decision is "defer", do NOT give the estimate as an answer. State that no
  reliable estimate can be given, name the covariate driving the uncertainty and
  what the estimate becomes at each end of its range, and say that supplying it
  would help IF would_resolve is true.
- If decision is "answer", give the estimate and band, then the observed mortality
  among similar patients, then note which values were read from the note versus
  assumed from cohort distributions.
- The subgroup_rule is an approximate population descriptor (R2 given). If you
  mention it, say it is approximate.
- No treatment recommendation. No advice. Describe only.
- Express the estimate and band in percentage points using estimate_pct_points,
  band_low_pct_points and band_high_pct_points. Never quote the raw decimals.
- Report observed mortality using the string fields, which already carry the % sign.
- When answering, state the unadjusted contrast alongside the estimate and note
  that the difference reflects adjustment for which patients receive treatment.
- If band_is_degenerate is true, do NOT write "band X to X". Say the estimate has
  no interval uncertainty because all covariates were read exactly, and give the
  single value only.

 
Output plain prose. No headers, no bullets, no JSON."""

def verbalize(rec, model="openai/gpt-5"):
    body = {"model": model, "temperature": 0,
            "messages": [{"role":"system","content":SYSTEM},
                         {"role":"user","content":json.dumps(rec, indent=2)}]}
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                         headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                                  "Content-Type": "application/json"},
                         json=body, timeout=300)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

def _numbers_in(obj, acc=None):
    """Every numeric value in the record, plus its percentage form."""
    acc = acc if acc is not None else set()
    if isinstance(obj, dict):
        for v in obj.values(): _numbers_in(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj: _numbers_in(v, acc)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        acc.add(round(float(obj), 4))
        acc.add(round(float(obj) * 100, 4))      # e.g. 0.069 -> 6.9
    elif isinstance(obj, str):
        for m in re.findall(r"-?\d+\.?\d*", obj):
            acc.add(round(float(m), 4))
    return acc

WHITELIST = {28.0, 0.0, 1.0, 2.0, 3.0, 100.0}    # "28-day", counts, ordinals

def validate_numbers(text, rec, tol=0.02):
    allowed = _numbers_in(rec) | WHITELIST
    allowed |= {abs(a) for a in allowed}          # "2.1 lower" for a stored -2.1
    clean = re.sub(r"(?<=\d),(?=\d)", "", text)   # strip thousands separators
    bad = []
    for tok in re.findall(r"-?\d+\.?\d*", clean):
        v = round(float(tok), 4)
        if not any(abs(v - a) <= max(tol, 0.01*abs(a)) for a in allowed):
            bad.append(v)
    return bad

if __name__ == "__main__":
    fn = sys.argv[1] if len(sys.argv) > 1 else "served_note_only.json"
    supplied = "composites" in fn
    served = json.load(open(fn, encoding="utf-8"))

    # pick one answered and one deferred patient as worked examples
    picks, seen = [], set()
    for r in served:
        d = gate_one(r["pred_interval"]["_intervals"])["decision"]
        if d not in seen:
            picks.append(r); seen.add(d)
        if len(seen) == 2: break

    out = []
    for r in picks:
        rec = build_record(r, structured_supplied=supplied)
        text = verbalize(rec)
        bad = validate_numbers(text, rec)
        out.append({"record": rec, "text": text, "unverified_numbers": bad})
        print("=" * 70)
        print(f"{rec['patient_id']}  [{rec['decision'].upper()}]")
        print("=" * 70)
        print(text)
        print(f"\n  unverified numbers: {bad if bad else 'none'}")
    json.dump(out, open("served_output.json","w"), indent=2, default=str)
    print("\nwrote served_output.json")