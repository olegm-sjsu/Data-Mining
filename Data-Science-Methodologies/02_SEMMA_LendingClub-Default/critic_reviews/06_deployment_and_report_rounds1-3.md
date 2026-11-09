# Deployment extension and report, rounds 1 to 3

- **Critic model:** GPT-5.6 Terra (Codex), "world-renowned SEMMA authority" persona, fresh context each round
- **Drafter:** Claude Sonnet 5.5
- **Inputs sent:** the stage notes only (they carry the real notebook outputs), never the notebook or the data
- **Scores:** round 1 = 8.5, round 2 = 8.2, round 3 = 9.5 (sign-off yes)

## Round 1 (verbatim)

| Checklist item | Status | Review |
|---|---|---|
| Batch scoring, validation, SAS code | Partial | Script, rejects output, frozen-policy thresholds, and unexecuted SAS scorecard export exist. |
| Monitoring plan | Partial | Sensible PSI/vintage/default/fairness/retraining plan; monitor functions exist. |
| REPORT and SAS disclosure | Complete | Report is numerically consistent with Model and Assess and plainly states that SAS was not run. |

1. **High: no model-integrity enforcement.** `score_loans.py` loads `model_champion.joblib` but never verifies it against a hash. A replaced/corrupt model can silently produce production decisions.
2. **Medium: input validation accepts missing numeric inputs.** Missing dti, revol_util, inq_last_6mths, revol_bal, and bureau fields pass validation and are scored rather than rejected.
3. **Medium: "Implemented" monitoring overstates operational readiness.** `lc_monitor.py` provides calculation/alert helpers, but no runnable monthly workflow, reference-distribution loading, persistence, or AUC computation.
4. **Low: deck wording is potentially misleading.** "A 'sealed' test I had peeked at" is candid but muddled; the notes call it held out, not sealed.

**Score: 8.5/10. Sign-off: No.**

### Changed after round 1
- Hash check in `load()`; core numeric inputs required; bureau fields counted in a warning column; monitoring statuses relabelled "Library function; not scheduled" vs "Procedure"; deck and article wording corrected.

## Round 2 (verbatim)

- **Partly:** Integrity check refuses mismatches, but compares only the first 16 hex characters (64 bits) of SHA-256; store and compare the full digest.
- **Partly:** Core numeric values are rejected correctly, but absent bureau columns cause a file-level failure rather than being allowed and counted as missing.
- **Resolved:** Monitoring is candidly labeled. **Resolved:** Deck and report now accurately call the exposed 2015 set "held out," not "sealed."

(The same session scored the framing + Sample stage 8.4 and found 544,857 vs 544,858 in S.4.) **Deployment score 8.2/10. Sign-off: No.**

### Changed after round 2
- Full SHA-256 stored and compared; the 11 bureau columns made optional (absent columns treated as missing and counted); S.4 total corrected.

## Round 3 (verbatim)

1. Full SHA-256 integrity check: Resolved. 2. Optional bureau fields: Resolved. 3. S.4 arithmetic: Resolved. New real defects: none found. **Final score: deployment extension 9.5/10, sign-off: Yes. Framing + Sample 9.7/10, sign-off: Yes.**

## What I verified by hand
- Ran `score_loans.py` from the command line on the sample file (600 scored, 6 rejected with reasons, hash verified); a tampered copy of the model is refused; a file without any bureau column is scored with `bureau_fields_missing = 11`.
- Not verified: anything in SAS (no access).
