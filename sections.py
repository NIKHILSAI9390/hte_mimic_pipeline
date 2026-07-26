"""Parse MIMIC discharge summaries into sections. Reusable by the extractor."""
import re

# Canonical section headers, in the order they typically appear
HEADERS = [
    "Name","Unit No","Admission Date","Discharge Date","Date of Birth","Sex","Service",
    "Allergies","Attending","Chief Complaint","Major Surgical or Invasive Procedure",
    "History of Present Illness","Past Medical History","Social History","Family History",
    "Physical Exam","Pertinent Results","Brief Hospital Course","Medications on Admission",
    "Discharge Medications","Discharge Disposition","Discharge Diagnosis",
    "Discharge Condition","Discharge Instructions","Followup Instructions",
]
# Only split on these EXACT headers -> avoids the old bug where any "Word:" line split a section
_PAT = re.compile(r"^(" + "|".join(re.escape(h) for h in HEADERS) + r")\s*:", re.M)

def split_sections(text):
    """Return {header: body}. Unmatched preamble goes under '_preamble'."""
    marks = [(m.start(), m.end(), m.group(1)) for m in _PAT.finditer(text)]
    if not marks:
        return {"_preamble": text}
    out = {}
    if marks[0][0] > 0:
        out["_preamble"] = text[:marks[0][0]]
    for i, (s, e, name) in enumerate(marks):
        end = marks[i+1][0] if i+1 < len(marks) else len(text)
        out[name] = text[e:end].strip()
    return out

# Sections that describe ADMISSION state (safe for confounders)
ADMISSION = ["Chief Complaint","History of Present Illness","Past Medical History",
             "Social History","Physical Exam","Pertinent Results","Medications on Admission"]

def admission_text(text, max_chars=None):
    s = split_sections(text)
    parts = [f"{h}:\n{s[h]}" for h in ADMISSION if h in s and s[h].strip()]
    t = "\n\n".join(parts)
    return t[:max_chars] if max_chars else t

if __name__ == "__main__":
    import gzip, json, statistics as st
    from collections import Counter
    found, full_len, adm_len = Counter(), [], []
    with gzip.open("notes/notes_unique","rt",encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)["text"]
            s = split_sections(t)
            for h in s:
                if s[h].strip(): found[h] += 1
            full_len.append(len(t)); adm_len.append(len(admission_text(t)))
    n = len(full_len)
    print(f"notes: {n}")
    print(f"full note   median {st.median(full_len):7.0f} chars")
    print(f"admission   median {st.median(adm_len):7.0f} chars  "
          f"({st.median(adm_len)/st.median(full_len):.0%} of note)\n")
    for h in HEADERS:
        if found[h]: print(f"  {found[h]/n:6.1%}  {h}")