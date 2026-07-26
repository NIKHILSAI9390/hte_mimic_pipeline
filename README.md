# Serving a Pre-Fitted Causal-Forest HTE Estimator to Text-Described Patients

Serves a causal forest — fit once on a MIMIC-IV sepsis cohort to estimate the effect of
early vasopressor initiation on 28-day mortality — to individual patients described in
free-text clinical notes. Covariates are extracted from a note into **intervals with
provenance**, propagated through the forest to yield an effect **distribution**, and a
**three-signal gate** either answers with an explained estimate or **defers**, naming the
missing measurement that would let it answer.

See `PSB2026_technical_report.md` for the full data-to-output write-up, exact result
tables, decisions log, and limitations.

## Data note (PhysioNet DUA)

Credentialed MIMIC-IV data is **not** included in this repo. `cohort_clean.csv`, real
admission-note text, and real-note extraction outputs are intentionally excluded. Real
MIMIC text must only be processed by locally-hosted models (never an external API). The
synthetic notes and their extractions **are** included, as they contain no patient data.

## Setup

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install econml scikit-learn pandas numpy requests
```

Large artifacts (`forest.pkl`, `surrogate.pkl`) are **not** tracked (size). Regenerate
them by running the two fit scripts below first.

## Pipeline and run order

The forest is fit **once** and frozen; every serving script loads the same `forest.pkl`.

**1. Fit and freeze the estimator** (requires `cohort_clean.csv` locally)
```bash
python forest_real.py     # fits the causal forest -> forest.pkl; prints ATE + HTE summary
python explain_real.py    # effect-driver ranking + surrogate rule -> surrogate.pkl
python m1_psm.py          # tabular-only PSM baseline (comparison)
```

**2. Calibration table** (category -> cohort interval band)
```bash
python calibrate.py       # -> calibration.json
```

**3. Extraction** (note text -> covariate intervals with provenance)
```bash
python gen_synthetic_notes.py     # generates synthetic notes (cross-family)
python scrub_notes.py synthetic_notes.json   # removes invented values -> *_scrubbed.json
python extract.py --backend openrouter --model openai/gpt-5 \
       --notes synthetic_notes_scrubbed.json   # -> extraction_*.json
python score_extraction.py extraction_openai_gpt_5.json
```
`extract.py` backends: `openrouter` (synthetic only), `ollama` / `vllm` (local, for real
notes under the DUA). Regex reads printed numbers; the LLM only maps prose to categories;
a validation pass strips any number not verbatim in the text (hallucination ~0 by
construction).

**4. Serving** (intervals -> effect distribution -> gate decision)
```bash
python interval_predict.py extraction_openai_gpt_5.json   # -> served_note_only.json, served_with_composites.json
python recalibrate.py     # matched-units gate thresholds -> thresholds.json
python gate.py            # prints the absolute thresholds
python gate_eval.py served_note_only.json                 # answer/defer + defer-by-support
python oracle_check.py served_note_only.json              # serving error: answered vs deferred
```

**5. Explainability, output, and interaction**
```bash
python attribute.py served_note_only.json   # which unknown covariate drives a deferral
python evidence.py served_note_only.json    # exact leaf-weight evidence base (reproduces forest.predict)
python serve.py served_note_only.json       # structured record -> LLM clinical prose (number-validated)
python ask.py  served_note_only.json        # clinician Q&A: why_defer / what_if / what_would_help / ...
python examples.py                          # assembles paper_examples.json
```

**6. Real-note validation** (local model only — DUA)
```bash
python build_real_testset.py                # -> real_testset.json (LOCAL only, credentialed)
python extract.py --backend vllm --model Qwen/Qwen2.5-32B-Instruct --notes real_testset.json --workers 8
python score_real.py extraction_Qwen_Qwen2_5_32B_Instruct.json
```

## Script index

| Stage | Script | Role |
|---|---|---|
| Data | `clean_data.py` | clean cohort export -> `cohort_clean.csv` |
| Forest | `forest_real.py` | fit + freeze forest -> `forest.pkl` |
| Forest | `explain_real.py` | effect drivers + surrogate -> `surrogate.pkl` |
| Forest | `m1_psm.py` | PSM baseline |
| Calibration | `calibrate.py` | category -> interval band -> `calibration.json` |
| Extraction | `gen_synthetic_notes.py` | synthetic note generation |
| Extraction | `scrub_notes.py` | remove invented values |
| Extraction | `sections.py` | admission-section parser |
| Extraction | `extract.py` | regex + LLM-category extraction (openrouter/ollama/vllm) |
| Extraction | `score_extraction.py` | synthetic-note scoring |
| Serving | `interval_predict.py` | intervals -> tau distribution |
| Serving | `recalibrate.py` | matched-units thresholds -> `thresholds.json` |
| Serving | `gate.py` | three-signal deferral gate |
| Serving | `gate_eval.py` | gate scoring + defer-by-support |
| Serving | `oracle_check.py` | answered-vs-deferred serving error |
| Explain | `attribute.py` | driver of a deferral (leave-one-out) |
| Explain | `evidence.py` | exact leaf-weight evidence base |
| Output | `serve.py` | structured record -> validated clinical prose |
| Interact | `ask.py` | clinician question layer |
| Interact | `examples.py` | assembles worked examples |
| Real notes | `build_real_testset.py` | real test set (local only) |
| Real notes | `score_real.py` | value accuracy + interval coverage |

## Key results

See `PSB2026_technical_report.md` for the full reportable set. Headlines: forest ATE
+0.036 (23% benefit / 77% harm); GPT-5 extraction 100% value / ~94% category on synthetic
notes; deferred patients carry ~2× the serving error of answered ones; note-only vs
note+chart deferral gradient. Real-note extraction is identical across a 7B and a 32B
model (value ~63%, interval coverage ~57%), showing the residual gap is structural
(timepoint definition + placeholder calibration), not model-limited.
