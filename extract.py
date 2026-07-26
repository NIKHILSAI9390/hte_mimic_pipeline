"""Extraction: deterministic regex + LLM categories, with post-hoc validation.
Backends: openrouter (API, synthetic phase) | ollama (local, real MIMIC phase).

Usage:
  python extract.py --backend openrouter --model google/gemini-3-pro
  python extract.py --backend openrouter --model openai/gpt-5
  python extract.py --backend ollama     --model qwen2.5:7b-instruct
"""
import os, json, re, sys, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

COVS = ["age","female","sofa","sapsii","charlson","heart_rate_mean","map",
        "resp_rate_mean","temp","spo2_mean","creatinine","wbc","platelets","bilirubin"]

# ---------------- enum vocabulary (neutral ids + definitions) ----------------
ENUM = {
 "creatinine": {"renal_normal":"kidney function normal or at baseline",
                "renal_impairment":"mild/moderate renal insufficiency or Cr above baseline",
                "renal_failure":"acute renal failure, AKI, oliguria, or dialysis"},
 "wbc": {"wbc_low":"leukopenia or neutropenia",
         "wbc_normal":"white count within normal limits",
         "wbc_high":"leukocytosis, elevated WBC, left shift, bandemia"},
 "platelets": {"plt_low":"thrombocytopenia or low platelets",
               "plt_normal":"platelets within normal limits",
               "plt_high":"thrombocytosis or elevated platelets"},
 "bilirubin": {"bili_normal":"LFTs normal, no jaundice",
               "bili_mild":"mildly elevated bilirubin or subtle icterus",
               "bili_marked":"marked jaundice or significant hyperbilirubinemia"},
 "temp": {"temp_low":"hypothermia","temp_normal":"afebrile or normal temperature",
          "temp_high":"fever, febrile, temperature spike"},
 "map": {"bp_shock":"hypotension requiring vasopressors or shock",
         "bp_stable":"hemodynamically stable or normotensive","bp_high":"hypertension"},
 "heart_rate_mean": {"hr_low":"bradycardia","hr_normal":"normal or regular rate",
                     "hr_high":"tachycardia"},
 "resp_rate_mean": {"rr_low":"low respiratory rate",
                    "rr_normal":"normal or comfortable breathing",
                    "rr_high":"tachypnea or increased work of breathing"},
 "spo2_mean": {"o2_severe":"significant hypoxemia or hypoxemic respiratory failure",
               "o2_mild":"mild hypoxemia or modest oxygen requirement",
               "o2_normal":"normal oxygenation, room air saturations"},
 "age": {"age_young":"young adult","age_middle":"middle-aged",
         "age_old":"elderly or advanced age"},
 "charlson": {"comorb_few":"few comorbidities","comorb_several":"several comorbidities",
              "comorb_many":"extensive comorbidity burden"},
}

def enum_block():
    out = []
    for k, d in ENUM.items():
        out.append(f"  {k}:")
        for lab, defn in d.items():
            out.append(f"      {lab}  = {defn}")
    return "\n".join(out)

PROMPT = f"""You are a critical care physician reading an ICU admission note.

For each variable below, report what the note actually contains:

  - If an explicit NUMBER for that variable is written in the note, return it in
    "value" and copy the literal text containing it into "evidence".
  - If no number is written but the note describes the clinical state in words,
    leave "value" null and return the ONE matching category in "category",
    copying the describing phrase into "evidence".
  - If the note contains neither, leave value, category and evidence empty.

Variables: {', '.join(c for c in COVS if c != 'female')}

Allowed categories (use the EXACT id string):
{enum_block()}

Return JSON mapping each variable to:
  {{"value": <number or null>, "category": <exact category id or null>,
    "evidence": "<verbatim snippet from the note, or empty string>"}}

RULES
- Return a number ONLY if you can copy it from the note into evidence. Never
  estimate, infer, or compute a value.
- A value written as ___ is redacted, not stated: leave "value" null.
- sofa and sapsii are composite scores essentially never written in notes; return
  them empty unless an explicit score appears.
- Do not confuse variables: a magnesium or hemoglobin value is not bilirubin.
- evidence must be copied verbatim from the note.

Return ONLY the JSON object."""

# ---------------- deterministic layer ----------------
LAB_PAT = {
 "creatinine": r"\bCreat(?:inine)?[\s\-:]+(\d+\.?\d*)",
 "wbc":        r"\bWBC[\s\-:]+(\d+\.?\d*)",
 "platelets":  r"\b(?:Plt\s*Ct|PLT\s*COUNT|Plt|Platelets?)[\s\-:]+(\d+\.?\d*)",
 "bilirubin":  r"\b(?:Tot\s*Bili|TotBili|TOT\s*BILI|Tbili|T\.?\s*bili|Total\s*Bili(?:rubin)?|Bilirubin\s*\(total\))[\s\-:]+(\d+\.?\d*)",
}
AGG = {"creatinine":max, "wbc":max, "bilirubin":max, "platelets":min}
VITAL_PAT = {
 "heart_rate_mean": r"\bHR[\s\-:]*(\d+)",
 "map":             r"\bMAP[\s\-:]*\(?(\d+)\)?",
 "resp_rate_mean":  r"\bRR[\s\-:]*(\d+)",
 "temp":            r"\b(?:T|Temp|Tm|Tmax)[\s\-:]+(\d{2,3}\.?\d*)",
 "spo2_mean":       r"(?:\bO2\s*sat|\bSpO2|\bSats?)[\s\-:]*(\d+)",
 "age":             r"(?<![\d.])(\d{2,3})\s*(?:yo\b|y/o|year[\s\-]?old|[MF]\b)",
}

def regex_vitals(text):
    out = {}
    for f, pat in VITAL_PAT.items():
        m = re.search(pat, text, re.I)
        if not m:
            out[f] = None; continue
        v = float(m.group(1))
        if f == "temp" and v > 45:            # Fahrenheit -> Celsius
            v = round((v - 32) * 5/9, 1)
        out[f] = v
    return out

def regex_labs(text):
    out = {}
    for f, pat in LAB_PAT.items():
        vals = [float(x) for x in re.findall(pat, text, re.I)]
        out[f] = AGG[f](vals) if vals else None
    return out

def regex_sex(text):
    m = re.search(r"\bSex\s*:\s*([MF])\b", text, re.I)
    return None if not m else (1.0 if m.group(1).upper()=="F" else 0.0)

# ---------------- backends ----------------
def call_openrouter(note, model, retries=3):
    key = os.environ["OPENROUTER_API_KEY"]
    body = {"model": model,
            "messages": [{"role":"system","content":PROMPT},
                         {"role":"user","content":"--- NOTE ---\n"+note}],
            "temperature": 0,
            "response_format": {"type":"json_object"}}
    for a in range(retries):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                              headers={"Authorization": f"Bearer {key}",
                                       "Content-Type": "application/json"},
                              json=body, timeout=300)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            return json.loads(re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M))
        except Exception as e:
            if a == retries-1: raise
            time.sleep(2 * (a+1))

def call_ollama(note, model, host="http://localhost:11434"):
    r = requests.post(f"{host}/api/generate", json={
        "model": model, "prompt": PROMPT + "\n\n--- NOTE ---\n" + note,
        "stream": False, "format": "json",
        "options": {"temperature": 0, "num_ctx": 8192}}, timeout=1800)
    return json.loads(r.json()["response"])

# ---------------- validation ----------------
def validate(pred, text):
    out = {}
    for c in COVS:
        if c == "female": continue
        p = pred.get(c)
        p = p if isinstance(p, dict) else {}
        val, cat, ev = p.get("value"), p.get("category"), (p.get("evidence") or "")[:200]
        if val is not None:
            try: v = float(val)
            except (TypeError, ValueError): v = None
            if v is None or not re.search(
                    rf"(?<![\d.]){re.escape(f'{v:g}')}(?![\d.])", text):
                val = None                      # claimed a number not in the text
            else:
                val = v
        if cat not in ENUM.get(c, {}):
            cat = None                          # enum violation
        mode = "stated" if val is not None else ("described" if cat else "absent")
        out[c] = {"mode":mode, "value":val, "category":cat, "evidence":ev}
    return out

def extract(text, backend, model):
    raw = call_openrouter(text, model) if backend=="openrouter" else call_ollama(text, model)
    pred = validate(raw, text)
    for f, v in regex_labs(text).items():       # deterministic layer overrides
        if v is not None:
            pred[f] = {"mode":"stated","value":v,"category":None,"evidence":"regex"}
    
    for f, v in regex_vitals(text).items():
        if v is not None:
            pred[f] = {"mode":"stated","value":v,"category":None,"evidence":"regex"}
    pred["female"] = {"mode":"stated","value":regex_sex(text),
                      "category":None,"evidence":"regex"}
    return pred

# ---------------- main ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--backend", default="openrouter", choices=["openrouter","ollama"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--notes", default="synthetic_notes_scrubbed.json")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    notes = json.load(open(a.notes, encoding="utf-8"))
    if a.limit: notes = notes[:a.limit]
    if a.backend == "ollama": a.workers = 1     # local GPU: no parallelism

    def job(n):
        return {"id":n["id"], "tier":n["tier"], "support":n["support"],
                "truth":n["truth"], "truth_mode":n["truth_mode"],
                "pred": extract(n["text"], a.backend, a.model)}

    res, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(job, n): n["id"] for n in notes}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                res.append(f.result())
            except Exception as e:
                print(f"  FAILED {futs[f]}: {e}")
            print(f"  {i}/{len(notes)}  ({time.time()-t0:.0f}s)", end="\r")

    res.sort(key=lambda r: r["id"])
    out = f"extraction_{re.sub(r'[^A-Za-z0-9]+','_',a.model)}.json"
    json.dump(res, open(out,"w"), indent=2)
    print(f"\nwrote {out}  ({len(res)}/{len(notes)} notes, {time.time()-t0:.0f}s)")