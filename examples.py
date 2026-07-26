"""Assemble the paper's worked examples: run serve.py (estimate/defer message) and
ask.py (clinician Q&A) across a curated set of patients, both regimes, into one file.
"""
import json
from serve import build_record, verbalize, validate_numbers
from ask import answer
from gate import gate_one

def decision(r):
    return gate_one(r["pred_interval"]["_intervals"])["decision"]

def pick_examples(served, supplied):
    """Curate: one clean answer, one deferral, and one single-covariate-resolvable
    deferral (the loop-closing demo)."""
    ans = next((r for r in served if decision(r)=="answer"
                and r["support"]=="typical"), None)
    dfr = next((r for r in served if decision(r)=="defer"
                and r["support"]=="implausible"), None)
    # the loop-closer: a deferred typical patient
    loop = next((r for r in served if decision(r)=="defer"
                 and r["support"]=="typical"), None)
    return [("answered", ans), ("deferred", dfr), ("resolvable", loop)]

QUESTIONS = [
    "Why can't you give me an answer for this patient?",
    "What would you need to answer?",
    "What if the SAPS-II score is 30?",
    "What happened to similar patients in your data?",
]

def run(fn):
    supplied = "composites" in fn
    served = json.load(open(fn, encoding="utf-8"))
    block = {"regime": "note+composites" if supplied else "note-only", "cases": []}

    for label, r in pick_examples(served, supplied):
        if r is None:
            continue
        rec = build_record(r, structured_supplied=supplied)
        text = verbalize(rec)
        case = {"role": label, "patient_id": r["id"], "support": r["support"],
                "decision": rec["decision"], "message": text,
                "unverified": validate_numbers(text, rec),
                "record": rec}
        # for the resolvable/deferred case, attach the Q&A transcript
        if label in ("deferred", "resolvable"):
            case["dialogue"] = []
            for q in QUESTIONS:
                a = answer(q, r, supplied)
                case["dialogue"].append({"q": q, "intent": a["parsed"].get("intent"),
                                         "a": a["text"], "unverified": a["unverified_numbers"]})
        block["cases"].append(case)
    return block

if __name__ == "__main__":
    out = [run("served_note_only.json"), run("served_with_composites.json")]
    json.dump(out, open("paper_examples.json","w"), indent=2, default=str)

    # readable console dump
    for blk in out:
        print("\n" + "#"*72)
        print(f"# REGIME: {blk['regime']}")
        print("#"*72)
        for c in blk["cases"]:
            print(f"\n--- {c['role'].upper()}  ({c['patient_id']}, {c['support']}) "
                  f"-> {c['decision']} ---")
            print(c["message"])
            flag = c["unverified"]
            if flag: print(f"  [!] unverified numbers: {flag}")
            for turn in c.get("dialogue", []):
                print(f"\n  Q: {turn['q']}  [{turn['intent']}]")
                print(f"  A: {turn['a']}")
                if turn["unverified"]:
                    print(f"     [!] unverified: {turn['unverified']}")
    print("\n\nwrote paper_examples.json")