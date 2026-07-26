"""Interactive layer: clinician asks a question, deterministic code computes,
LLM verbalizes. Same discipline as everything else — every number in the reply
must exist in the computed record.

Intents:
  why_defer        - which signals fired and which covariate drives the uncertainty
  what_if          - pin a covariate to a supplied value, re-run, report the change
  what_would_help  - ranked list of what to supply, and whether it would resolve
  similar_patients - the weighted evidence base: who backs this estimate, what happened
  explain          - restate the current estimate and its basis
"""
import json, os, re, sys, numpy as np, requests

from gate import gate_one, COVS, forest
from attribute import attribute
from evidence import evidence
from serve import build_record, verbalize, validate_numbers, PRETTY, SYSTEM

INTEGER_COVS = {"age", "female", "sofa", "sapsii", "charlson"}

# ---------------- deterministic recomputation ----------------
def recompute(intervals, n_draws=400, seed=0):
    """Tau distribution + gate for an arbitrary interval box."""
    rng = np.random.default_rng(seed)
    X = np.empty((n_draws, len(COVS)))
    for j, c in enumerate(COVS):
        lo, hi = intervals[c]
        col = np.full(n_draws, lo, float) if lo == hi else rng.uniform(lo, hi, n_draws)
        if c in INTEGER_COVS: col = np.round(col)
        X[:, j] = col
    taus = forest.predict(X).ravel()
    g = gate_one(intervals)
    return {
        "tau_median": float(np.percentile(taus, 50)),
        "tau_lo": float(np.percentile(taus, 5)),
        "tau_hi": float(np.percentile(taus, 95)),
        "tau_spread": float(np.percentile(taus,95) - np.percentile(taus,5)),
        "frac_harm": float((taus > 0).mean()),
        "decision": g["decision"], "reasons": g["reasons"],
    }

# ---------------- intent parsing ----------------
PARSE_SYSTEM = f"""Classify a clinician's question about a treatment-effect estimate.

Return JSON: {{"intent": <one of why_defer|what_if|what_would_help|similar_patients|explain>,
               "covariate": <exact key or null>, "value": <number or null>}}

Covariate keys: {', '.join(COVS)}
Map clinical names to keys (e.g. "SAPS-II"->sapsii, "severity score"->sapsii,
"kidney function"/"Cr"->creatinine, "blood pressure"->map, "comorbidities"->charlson).

Use "what_if" ONLY when the question supplies or hypothesises a specific value.
Use "what_would_help" for "what do you need", "what would let you answer".
Return ONLY the JSON."""

def parse_question(q, model="openai/gpt-5"):
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                 "Content-Type": "application/json"},
        json={"model": model, "temperature": 0,
              "response_format": {"type": "json_object"},
              "messages": [{"role":"system","content":PARSE_SYSTEM},
                           {"role":"user","content":q}]}, timeout=300)
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"]
    return json.loads(re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M))

# ---------------- handlers ----------------
def handle(intent, served_row, cov=None, val=None, supplied=False):
    iv = served_row["pred_interval"]["_intervals"]
    base = build_record(served_row, structured_supplied=supplied)

    if intent == "what_if":
        if cov not in COVS or val is None:
            return {"question_type": "what_if", "error":
                    "no covariate/value identified", **base}
        v = round(float(val)) if cov in INTEGER_COVS else float(val)
        iv2 = {**iv, cov: [v, v]}
        after = recompute(iv2)
        return {
            "question_type": "what_if",
            "covariate_supplied": PRETTY[cov], "value_supplied": v,
            "previous_range_for_covariate": [iv[cov][0], iv[cov][1]],
            "value_is_outside_previous_range": not (iv[cov][0] <= v <= iv[cov][1]),
            "decision_before": base["decision"],
            "decision_after": after["decision"],
            "reasons_after": after["reasons"],
            "estimate_before_pct_points": base["estimate_pct_points"],
            "estimate_after_pct_points": round(after["tau_median"]*100, 1),
            "band_after_pct_points": [round(after["tau_lo"]*100,1),
                                      round(after["tau_hi"]*100,1)],
            "uncertainty_before": round(served_row["pred_interval"]["tau_spread"], 4),
            "uncertainty_after": round(after["tau_spread"], 4),
        }

    if intent == "what_would_help":
        att = attribute(iv, n=400)
        opts = [d for d in att["drivers"][:5] if d["pct_of_instability"] > 0.01]
        if not opts:
            return {"question_type": "what_would_help", "decision": base["decision"],
                    "no_options_reason": ("all covariates were read from the note (nothing was "
                        "assumed), so there is no missing value to supply; the deferral comes "
                        "from the covariate values themselves being out of range"),
                    "options": []}
        return {"question_type": "what_would_help", "decision": base["decision"],
                "options": [{"covariate": PRETTY[d["covariate"]],
                             "current_range": [round(d["interval"][0],2),
                                               round(d["interval"][1],2)],
                             "pct_of_uncertainty": f"{round(d['pct_of_instability']*100,1)}%",
                             "would_resolve": d["gate_if_known"] == "answer",
                             "estimate_at_low_pct_points": round(d["tau_at_low"]*100,1),
                             "estimate_at_high_pct_points": round(d["tau_at_high"]*100,1)}
                            for d in opts]}

    if intent == "similar_patients":
        ev = evidence(iv)
        return {"question_type": "similar_patients",
                "contributing_patients": ev["n_contributing"],
                "effective_sample_size_overall": round(ev["effective_n"]),
                "effective_sample_size_treated_arm": round(ev["effective_n_treated"]),
                "effective_sample_size_control_arm": round(ev["effective_n_control"]),
                "note_on_effective_sizes": "variance-adjusted effective sizes, not patient counts; the arms do not sum to the overall figure",
                "observed_mortality_treated": f"{round(ev['wmort_treated']*100,1)}%",
                "observed_mortality_control": f"{round(ev['wmort_control']*100,1)}%",
                "unadjusted_contrast_pct_points": round(ev["naive_contrast"]*100,1),
                "weighted_profile": {PRETTY[k]: round(v,1)
                                     for k,v in ev["profile"].items()}}

    # why_defer / explain -> the full base record
    return {"question_type": intent, **base}

ASK_SYSTEM = SYSTEM + """

ADDITIONAL RULES FOR ANSWERING A QUESTION
- Answer the specific question asked, in 2-4 sentences. Do not restate the whole record.
- For "what_if": if decision_after is still "defer" for ANY reason (out-of-support OR
  estimator-uncertain), do NOT report any estimate, endpoint, or band. State only that
  the decision remains defer and briefly why, and that no reliable estimate can be given
  because deferring means declining to produce a number. Give estimate_after and its
  band ONLY when decision_after is "answer".
- For "what_if": do NOT quote the previously-assumed range when its low and high are
  equal (e.g. "26.0 to 26.0" or "361.0 to 361.0"). In that case the value was already
  known from the note — say the covariate was already read/known, and do not print the
  number as a range.
- For "what_would_help": name the options in order, with the share of uncertainty each
  accounts for, and say which would actually resolve the deferral. If options is empty
  and no_options_reason is present, explain using that reason — there is nothing to
  supply because the inputs were all read and are themselves out of range; do NOT say
  you "can't tell what would help".
- For "similar_patients": report the effective sample sizes and observed mortality in
  both arms, and note that the unadjusted contrast differs from the model estimate
  because of adjustment for who receives treatment. Effective sample sizes are
  variance-adjusted quantities, not patient counts — never describe them as a number of
  patients and never imply the two arms sum to the overall figure.
"""

def answer(question, served_row, supplied=False, model="openai/gpt-5"):
    p = parse_question(question, model)
    rec = handle(p.get("intent","explain"), served_row,
                 p.get("covariate"), p.get("value"), supplied)
    body = {"model": model, "temperature": 0,
            "messages": [{"role":"system","content":ASK_SYSTEM},
                         {"role":"user","content":
                          f"Question: {question}\n\nRecord:\n{json.dumps(rec, indent=2, default=str)}"}]}
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                 "Content-Type": "application/json"}, json=body, timeout=300)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"].strip()
    return {"question": question, "parsed": p, "record": rec,
            "text": text, "unverified_numbers": validate_numbers(text, rec)}

if __name__ == "__main__":
    fn = sys.argv[1] if len(sys.argv) > 1 else "served_note_only.json"
    supplied = "composites" in fn
    served = json.load(open(fn, encoding="utf-8"))

    # use one of the deferrals that a single covariate can resolve
    target = next((r for r in served if r["id"] == "easy_typical_01"), served[0])

    QUESTIONS = [
        "Why can't you give me an answer for this patient?",
        "What would you need to answer?",
        "What if the SAPS-II score is 30?",
        "What happened to similar patients in your data?",
    ]
    out = []
    for q in QUESTIONS:
        a = answer(q, target, supplied)
        out.append(a)
        print("=" * 70)
        print(f"Q: {q}")
        print(f"   [intent: {a['parsed'].get('intent')}]")
        print("-" * 70)
        print(a["text"])
        print(f"\n   unverified numbers: {a['unverified_numbers'] or 'none'}")
    json.dump(out, open("asked.json","w"), indent=2, default=str)
    print("\nwrote asked.json")