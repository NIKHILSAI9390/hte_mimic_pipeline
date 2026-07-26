"""Remove invented values on described/absent fields (labs + vitals + age)."""
import json, re, sys
from collections import Counter

f = sys.argv[1] if len(sys.argv)>1 else "synthetic_notes.json"
notes = json.load(open(f, encoding="utf-8"))

# token pattern (captures the label too, so we can rewrite or delete the whole thing)
TOKEN = {
 "creatinine": r"\bCreat(?:inine)?[\s\-:]+\d+\.?\d*",
 "wbc":        r"\bWBC[\s\-:]+\d+\.?\d*",
 "platelets":  r"\b(?:Plt\s*Ct|PLT\s*COUNT|Plt|Platelets?)[\s\-:]+\d+\.?\d*",
 "bilirubin":  r"\b(?:Tot\s*Bili|TotBili|TOT\s*BILI|Tbili|T\.?\s*bili|Total\s*Bili(?:rubin)?|Bilirubin\s*\(total\))[\s\-:]+\d+\.?\d*",
 "heart_rate_mean": r"\bHR[\s\-:]+\d+",
 "map":             r"\(?\s*MAP[\s\-:]+\d+\s*\)?",
 "resp_rate_mean":  r"\bRR[\s\-:]+\d+",
 "temp":            r"\b(?:T|Temp|Tm|Tmax)[\s\-:]+\d{2,3}\.?\d*",
 "spo2_mean":       r"(?:\bO2\s*sat|\bSpO2|\bSats?)[\s\-:]+\d+%?",
}
REDACTED = {"creatinine":"Creat-___","wbc":"WBC-___","platelets":"Plt ___",
            "bilirubin":"TotBili-___","heart_rate_mean":"HR ___","map":"(MAP ___)",
            "resp_rate_mean":"RR ___","temp":"T ___","spo2_mean":"O2 sat ___"}

fixed = Counter()
for n in notes:
    t = n["text"]
    for c, pat in TOKEN.items():
        m = n["truth_mode"].get(c)
        if m not in ("described","absent"): continue
        if not re.search(pat, t, re.I): continue
        t = re.sub(pat, REDACTED[c] if m == "described" else "", t, flags=re.I)
        fixed[m] += 1
    # age: written inline like "71M" / "82 yo" -> redact the digits, keep the sentence
    if n["truth_mode"].get("age") in ("described","absent"):
        new = re.sub(r"(?<![\d.])\d{2,3}(\s*(?:yo\b|y/o|year[\s\-]?old|[MF]\b))",
                     r"___\1", t)
        if new != t: fixed[n["truth_mode"]["age"]] += 1
        t = new
    n["text"] = re.sub(r"[ \t]{2,}", " ", t)

out = f.replace(".json", "_scrubbed.json")
json.dump(notes, open(out,"w"), indent=2)
print(f"scrubbed {sum(fixed.values())} invented values "
      f"(described: {fixed['described']}, absent: {fixed['absent']})")
print(f"wrote {out}")