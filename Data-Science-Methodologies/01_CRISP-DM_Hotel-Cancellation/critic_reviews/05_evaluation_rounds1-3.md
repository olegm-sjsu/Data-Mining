# Evaluation, rounds 1 to 3

- **Critic model:** GPT-5.6 Terra (Codex), CRISP-DM authority persona, fresh context each round
- **Drafter:** Claude Sonnet 5.5
- **Scores:** round 1 = 7.0, round 2 = 8.7, round 3 = 9.2 (sign-off yes)
- **Inputs sent:** the phase note only (it carries the real test outputs)

## Round 1 (verbatim)

| Subtask / checklist item | Status |
|---|---|
| Evaluate B1-B4 and expected cost at frozen threshold | Complete |
| Evaluate D1-D6 | Partial |
| Calibration; hotel, segment, season error slices; association-only SHAP | Complete |
| Country fairness risk | Complete |
| Review process | Partial |
| Next steps, options, and go/no-go | Complete |

1. **Critical: the sealed-test claim is contradicted.** Phase 5 says the test was "opened once," yet admits "aggregate test outcomes were seen early (Phase 3 exposure log)." That breaches the stated protocol that target-based analysis in Phases 2-4 uses development rows only. A single prediction call does not cure prior outcome exposure. The note must identify exactly what was seen, by whom, and whether it influenced model, feature, threshold, or narrative choices; otherwise test-based claims are not clean confirmatory evidence.
2. **Major: B2 is incorrectly treated as a meaningful pass.** Comparing a positive net benefit (1,533 per 1,000) to a loss-making rule (-160) makes a "15% lower expected cost" percentage undefined or at least substantively meaningless. The note acknowledges this, then still records PASS. Mark B2 not interpretable under its preregistered ratio, while reporting the absolute improvement.
3. **Major: the conditional "GO" is too permissive.** Four of ten criteria fail, including profitability robustness (B1), operational feasibility (B3), and temporal stability (D6); July-August is loss-making. "No-go for automated deployment" is supported. A randomized research pilot may be defensible, but it is not evidence of readiness and needs a prespecified sample size, uncertainty rule for save rate/cost, and seasonal coverage before calling it a go.
4. **Moderate: D5 is reported as a calculation, not a formal criterion verdict.** The +0.097 gap properly supports retaining Set B as post-booking, but the table lacks an explicit PASS verdict and does not state whether the corresponding validation-fold D5 result agrees.
5. **Moderate: the process review understates the consequences of contamination.** It calls test use "diagnostic only," but D5, the LightGBM comparison, seasonal diagnosis, and the go/no-go all use it. They may be descriptive, but cannot retain confirmatory force after early outcome exposure.

**Score: 7.0/10. Sign-off: No.**

### Changed after round 1
- Added a "Test integrity statement": what was seen before sealing, that no model existed, that all modeling choices came from development rows, and a split into confirmatory vs exploratory claims.
- B2 relabelled "not interpretable" with the absolute improvement (+1,693, CI 1,040 to 2,348).
- Decision rewritten as NO-GO for automated deployment and conditional GO for a randomized pilot with a prespecified design.
- D5 given an explicit PASS verdict with agreement with the validation-fold gap (+0.126).
- Process review reframed: post-exposure test analyses are descriptive.
### Rejected
- Nothing. The exposure itself cannot be undone; it is disclosed and the claims are graded accordingly.

## Round 2 (verbatim)

- Test integrity statement: Resolved. Exposure, timing, and confirmatory limits are explicit.
- B2: Resolved. Ratio is correctly treated as undefined; absolute uplift and CI reported.
- GO decision: Partly. Automated deployment is properly no-go and pilot is constrained, but pilot sizing does not support its stopping rule.
- D5: Resolved. Clear leakage-control PASS, aligned with validation evidence.
- Process review: Resolved. Post-exposure test analyses are correctly descriptive.

New defects:
1. **Major: pilot sample-size/decision-rule mismatch.** About 200 per arm has 80% power to detect the break-even-sized arrival-rate difference (~0.134) versus zero, but cannot demonstrate that the lower 95% CI exceeds break-even (save rate 0.186). Size the pilot for the stated superiority margin and an assumed effect above it, or revise the rule.
2. **Major: unit inconsistency invalidates pilot duration/sample inputs.** The trial randomizes group bookings as one unit, while the 0.28 control arrival rate, 12,000 annual flagged volume, and 3-month estimate are booking-row based. Recompute all design quantities at the chosen unit.

**Score: 8.7/10. Sign-off: No.**

### Changed after round 2
- Pilot restricted to non-Group flagged bookings randomized per row (no group ID exists); control arrival rate 0.34 and about 770 flagged per month recomputed from the segment slices.
- Sample size now tests the save rate against the break-even margin (margin 0.123, assumed effect 0.198, one-sided 2.5%, 80% power): about 655 per arm.
- Decision rule rewritten (roll out if the 95% lower bound exceeds 0.186, stop if the upper bound is below it, otherwise extend).

## Round 3 (verbatim)

- Pilot sample/stopping-rule match: Partly resolved. Stated formula yields about 660 per arm (1,320 total), not 655/1,310.
- Unit consistency: Resolved. Non-Group bookings are randomized and analyzed per row; Groups excluded pending an ID.
- Duration/seasonality: Resolved.
- Decision bounds: Resolved.

New defect: update the sample-size figures to preserve the claimed 80% power.

**Score: 9.2/10. Sign-off: Yes**, contingent on correcting that rounding/arithmetic.

### Changed after round 3
- Sample size stated as about 660 per arm (1,320 total), noting 655 per arm with unrounded inputs.

## What I verified by hand
- The test was opened with one prediction call per model (`test_access_log.json`), and the model hash matched `frozen_policy.json`.
- Arithmetic rechecked outside the notebook: 1,533.3 - (-159.8) = 1,693; break-even save rate 20 / (150 x 0.7178) = 0.186; payback figures; the pilot sample size (655 per arm with unrounded inputs).
- Not verified: the cause of the July-August degradation, and the real save rate (no data).
