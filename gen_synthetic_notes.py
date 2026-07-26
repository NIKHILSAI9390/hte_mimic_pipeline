"""Generate synthetic MIMIC-style admission notes with exact ground truth.

Key design: the script decides each covariate's provenance BEFORE generation, and
for 'described' fields passes ONLY a qualitative descriptor - never the number.
The model cannot leak a value it was never shown.
"""
import os, json
import numpy as np, pandas as pd
import anthropic

COVS = ["age","female","sofa","sapsii","charlson","heart_rate_mean","map",
        "resp_rate_mean","temp","spo2_mean","creatinine","wbc","platelets","bilirubin"]
UNITS = {"age":"years","charlson":"points","heart_rate_mean":"bpm","map":"mmHg",
         "resp_rate_mean":"breaths/min","temp":"degC","spo2_mean":"%",
         "creatinine":"mg/dL","wbc":"K/uL","platelets":"K/uL","bilirubin":"mg/dL"}

P_STATED  = {"easy":0.95, "med":0.50, "hard":0.15}
P_DESCRIBED_GIVEN_NOT_STATED = 0.75
P_REDACT  = 0.25

# clinical display precision - values must be writable as a physician would write them
PREC = {"age":0,"sofa":0,"sapsii":0,"charlson":0,"heart_rate_mean":0,"map":0,
        "resp_rate_mean":0,"spo2_mean":0,"temp":1,"creatinine":1,"wbc":1,
        "platelets":0,"bilirubin":1}

cohort = pd.read_csv("cohort_clean.csv")

def sample_truth(support, rng):
    v = {}
    for c in COVS:
        s = cohort[c].dropna()
        if support == "typical":
            v[c] = float(s.quantile(rng.uniform(0.15, 0.85)))
        elif support == "out_of_cohort":
            v[c] = float(s.quantile(rng.choice([0.01, 0.99])))
        else:                                        # implausible
            v[c] = float(s.max() * rng.uniform(2.5, 6.0))
    v["female"] = float(rng.integers(0, 2))
    for c in COVS:                                   # round to clinical precision
        if c == "female": continue
        v[c] = round(v[c], PREC[c])
        if PREC[c] == 0: v[c] = float(int(v[c]))
    return v

BUCKETS = {
 "creatinine":[(1.3,"renal_normal",["normal renal function","preserved renal function",
                "Cr at baseline","no renal dysfunction","kidney function intact"]),
               (3.0,"renal_impairment",["mild renal insufficiency","modest renal impairment",
                "Cr above baseline","mildly worsened kidney function","borderline renal function"]),
               (1e12,"renal_failure",["acute renal failure","AKI","acute kidney injury",
                "renal failure, oliguric","severe renal dysfunction"])],
 "wbc":[(4,"wbc_low",["leukopenia","a low white count","neutropenia","depressed WBC"]),
        (11,"wbc_normal",["a normal white count","WBC within normal limits","no leukocytosis"]),
        (1e12,"wbc_high",["leukocytosis","an elevated white count","WBC up with a left shift",
         "bandemia"])],
 "platelets":[(100,"plt_low",["thrombocytopenia","a low platelet count","platelets depressed"]),
              (400,"plt_normal",["normal platelets","a normal platelet count"]),
              (1e12,"plt_high",["thrombocytosis","an elevated platelet count"])],
 "bilirubin":[(1.2,"bili_normal",["normal LFTs","no jaundice","liver panel unremarkable"]),
              (3,"bili_mild",["mild hyperbilirubinemia","mildly elevated bilirubin",
               "subtle scleral icterus"]),
              (1e12,"bili_marked",["marked jaundice","frank icterus","significant hyperbilirubinemia"])],
 "temp":[(36,"temp_low",["hypothermia","being hypothermic","a low temperature"]),
         (38,"temp_normal",["being afebrile","no fevers","a normal temperature"]),
         (1e12,"temp_high",["fever","being febrile","a temperature spike","pyrexia"])],
 "map":[(65,"bp_shock",["hypotension requiring pressors","refractory hypotension on vasopressors",
         "shock physiology needing pressor support"]),
        (90,"bp_stable",["being hemodynamically stable","normotension","stable blood pressures"]),
        (1e12,"bp_high",["hypertension","elevated blood pressures"])],
 "heart_rate_mean":[(60,"hr_low",["bradycardia","a slow heart rate"]),
                    (100,"hr_normal",["a normal heart rate","a regular rate"]),
                    (1e12,"hr_high",["tachycardia","being persistently tachycardic"])],
 "resp_rate_mean":[(12,"rr_low",["a low respiratory rate","shallow slow breathing"]),
                   (22,"rr_normal",["a normal respiratory rate","comfortable breathing"]),
                   (1e12,"rr_high",["tachypnea","increased work of breathing","rapid respirations"])],
 "spo2_mean":[(90,"o2_severe",["significant hypoxemia","profound desaturation",
               "hypoxemic respiratory failure"]),
              (95,"o2_mild",["mild hypoxemia","a modest oxygen requirement"]),
              (1e12,"o2_normal",["normal oxygenation","saturations on room air"])],
 "age":[(50,"age_young",["a young adult","a young patient"]),
        (70,"age_middle",["middle-aged","a middle-aged patient"]),
        (1e12,"age_old",["elderly","an older patient","advanced age"])],
 "charlson":[(2,"comorb_few",["few comorbidities","a limited medical history"]),
             (6,"comorb_several",["several comorbidities","a moderate comorbidity burden"]),
             (1e12,"comorb_many",["an extensive comorbidity burden","multiple significant comorbidities"])],
}

def bucket_of(cov, v):
    """Neutral bucket id for a value. Used by the SCORER."""
    if cov not in BUCKETS: return None
    for cut, bid, _ in BUCKETS[cov]:
        if v < cut: return bid
    return BUCKETS[cov][-1][1]

def descriptor(cov, v, rng):
    """Random synonym for the value's bucket. Used by the GENERATOR."""
    if cov not in BUCKETS: return None
    for cut, _, syns in BUCKETS[cov]:
        if v < cut: return str(rng.choice(syns))
    return str(rng.choice(BUCKETS[cov][-1][2]))

def assign_modes(tier, rng):
    modes = {}
    for c in COVS:
        if c == "female":
            modes[c] = "stated"; continue
        if c in ("sofa","sapsii","charlson"):
            modes[c] = "absent"; continue
        if rng.random() < P_STATED[tier]:
            modes[c] = "stated"
        elif rng.random() < P_DESCRIBED_GIVEN_NOT_STATED:
            modes[c] = "described"
        else:
            modes[c] = "absent"
    for c in ("creatinine","wbc","platelets","bilirubin"):
        if modes[c] == "stated" and rng.random() < P_REDACT:
            modes[c] = "redacted"
    return modes

SYSTEM = """You write realistic ICU admission notes in the style of MIMIC-IV \
discharge summaries. You imitate how physicians actually write under time \
pressure: telegraphic phrasing, heavy abbreviation, inconsistent spacing, \
clinically irrelevant detail alongside relevant detail. You never write in \
polished prose. You output ONLY the note."""

def fmt(c, v):
    return f"{v:.{PREC[c]}f}" if PREC[c] else f"{int(v)}"

def build_prompt(truth, modes, support,rng):
    def as_written(c):
        if c == "temp":                       # notes write Fahrenheit
            return f"temp = {truth[c]*9/5+32:.1f} degF"
        return f"{c} = {fmt(c,truth[c])} {UNITS.get(c,'')}".strip()
    stated    = [as_written(c) for c in COVS
                 if modes[c]=="stated" and c!="female"]
    described = [f"{c}: portray the patient as having {descriptor(c,truth[c],rng)}"
                 for c in COVS if modes[c]=="described" and descriptor(c,truth[c],rng)]
    redacted  = [c for c in COVS if modes[c]=="redacted"]
    sex = "F" if truth["female"]==1 else "M"
    extreme = ("\nSome values above are physiologically extreme or implausible. "
               "Write them EXACTLY as given. Do NOT correct them, soften them, round "
               "them toward normal, or add commentary questioning them. Write the note "
               "as if these were the measured values.\n") if support=="implausible" else ""

    return f"""Write ONE synthetic ICU admission note for a sepsis patient. Sex: {sex}.

Use MIMIC section headers: Name / Unit No / Admission Date / Discharge Date /
Date of Birth / Sex / Service / Allergies / Attending / Chief Complaint /
History of Present Illness / Past Medical History / Social History /
Family History / Physical Exam / Pertinent Results / Medications on Admission

WRITE THESE AS EXPLICIT NUMBERS (in the vitals line or the lab block):
{chr(10).join('  - '+s for s in stated) if stated else '  (none)'}
{extreme}
DESCRIBE THESE IN WORDS ONLY. Never write any number for them anywhere in the
note, including past medical history or baseline references:
{chr(10).join('  - '+s for s in described) if described else '  (none)'}

REDACT THESE - write the lab name followed by ___ (e.g. "Creat-___", "Plt ___"):
{chr(10).join('  - '+c for c in redacted) if redacted else '  (none)'}

Do not invent numeric values for any laboratory or vital sign not listed above.
Write narrative sections as realistic telegraphic physician prose; only the lab
block and vitals line use slot format. Include clinical noise (unrelated findings,
med lists, imaging, social details). Use ___ for names, dates, and hospitals only.
Admission state ONLY - no discharge condition, no hospital course, no outcomes.
Length 2500-4500 characters.

Output only the note text."""

def generate(n_per_cell=10, out="synthetic_notes.json", model="claude-opus-4-8"):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    rng = np.random.default_rng(0)
    notes = []
    if os.path.exists(out):
        notes = json.load(open(out, encoding="utf-8"))
        print(f"resuming: {len(notes)} notes already done")
    done = {n["id"] for n in notes}

    for tier in ["easy","med","hard"]:
        for support in ["typical","out_of_cohort","implausible"]:
            for i in range(n_per_cell):
                nid = f"{tier}_{support}_{i:02d}"
                # ALWAYS draw, even when skipping, so the rng stream stays aligned
                truth = sample_truth(support, rng)
                modes = assign_modes(tier, rng)
                prompt = build_prompt(truth, modes, support, rng)
                if nid in done:
                    continue
                msg = client.messages.create(
                    model=model, max_tokens=2000, temperature=1.0, system=SYSTEM,
                    messages=[{"role":"user","content":prompt}])
                notes.append({"id":nid,"tier":tier,"support":support,"truth":truth,
                              "truth_mode":modes,"text":msg.content[0].text})
                json.dump(notes, open(out,"w"), indent=2)     # save after EVERY note
                print(f"  {nid}  {len(msg.content[0].text)} chars  ({len(notes)}/90)")
    print(f"\nwrote {out}: {len(notes)} notes")

if __name__ == "__main__":
    generate()