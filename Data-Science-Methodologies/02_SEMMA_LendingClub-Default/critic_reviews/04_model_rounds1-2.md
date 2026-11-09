# Model, rounds 1 and 2

- **Critic model:** GPT-5.6 Terra (Codex), "world-renowned SEMMA authority" persona, fresh context each round
- **Drafter:** Claude Sonnet 5.5
- **Inputs sent:** the stage notes only (they carry the real notebook outputs), never the notebook or the data
- **Scores:** round 1 = 8.4, round 2 = 9.4 (sign-off yes)

## Round 1 (verbatim)

| Checklist item | Status |
|---|---|
| Baseline, WoE scorecard, tree, neural network or boosted model | **Complete** |
| Tune on validation only | **Complete** |
| Rebuild SAS flow if available | **Partial** (unavailability is candidly documented, but no SAS rebuild/screenshots exist) |

1. **Critical: policy is not reproducible.** Two champions are named, but the single PD <= 0.38 cutoff is not tied to Model A or Model B. Name the model used for deciles, cash-flow optimization, and the frozen deployment policy.
2. **Major: baseline KS arithmetic is wrong.** A constant majority/prior prediction has KS = 0, not 0.023. Recompute or explain a non-constant score.
3. **Major: choice rule is incomplete across variants.** It does not state whether Model A or B is the claimed/deployed model. This matters because B embeds Lending Club's own assessment.
4. **Moderate: profit claim needs tighter honesty.** The +0.07m / +0.05-point improvement is tiny and should be presented as validation-only noise until out-of-time confirmation.
5. **Moderate: fairness is deferred but not yet evidenced.** Commit to an Assess-stage proxy/error-slice check rather than treating exclusion as completion.

**Score: 8.4/10. Sign-off: No.**

### Changed after round 1
- The cutoff is tied to the Model A champion; frozen policy = Model A champion + PD <= 0.38; Model B is comparison only and never deployed.
- KS function fixed to handle tied scores through the ROC curve (majority KS now 0.000; FICO-only 0.149; tree 0.213).
- Profit gain stated as validation-only and within noise; fairness commitment added.
### Rejected
- Nothing. (SAS stays "Partial" because access was not available.)

## Round 2 (verbatim)

1. Resolved. 2. Resolved. 3. Resolved. 4. Resolved. 5. Resolved. New defect: the claimed SAS "score-code export" is not actually runnable evidence of SAS equivalence; the text does disclose this. **9.4/10. Sign-off: Yes.**

## What I verified by hand
- Choice rule applied to the validation table: LightGBM A-extended beats the best Model A scorecard by 0.0143 (> 0.005), LightGBM B beats Scorecard B by 0.0076.
- A KS bug was found by the critic (ties), not by me; it was a real code defect.
