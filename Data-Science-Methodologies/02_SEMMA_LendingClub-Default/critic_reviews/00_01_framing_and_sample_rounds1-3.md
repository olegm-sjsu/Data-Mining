# Problem framing and Sample, rounds 1 to 3

- **Critic model:** GPT-5.6 Terra (Codex), "world-renowned SEMMA authority" persona, fresh context each round
- **Drafter:** Claude Sonnet 5.5
- **Inputs sent:** the stage notes only (they carry the real notebook outputs), never the notebook or the data
- **Scores:** round 1 = 7.5, round 2 = 8.4, round 3 = 9.7 (sign-off yes)

## Round 1 (verbatim)

| Checklist item | Status | Evidence |
|---|---|---|
| Before Sample framing | Complete | Explicitly labeled outside SEMMA; decision, economics, pre-set criteria, and model claims stated. |
| Eligible population/filter counts | Complete | Funnel reconciles exactly to 666,785; censoring and 36/60-month maturity rules stated. |
| Stratified sample and size | Complete | Term/year/outcome strata; full training population retained after learning curve. |
| Representativeness | Partial | KS/PSI and categorical effect sizes are reported, but column coverage and stratified inclusion/allocation details are not auditable. |
| Learning curve | Complete | Held-out 2014 AUC curve, decision rule, and result clearly reported. |
| Imbalance/probability correction | Complete | 12.4% default rate; no oversampling, hence no correction required. |
| Train/validation/out-of-time partition | Partial | Chronological partitions are sound, but outcome-stratified test selection and "sealed" status are qualified by test-vintage exposure. |

1. **High: out-of-time test is not genuinely sealed.** The 2015 test default rate (14.9%) was inspected during partition planning. Calling it "sealed" elsewhere is contradictory. Aggregate outcome exposure can influence calibration expectations, thresholds, and interpretation; disclose this as a contaminated OOT test or replace it with an untouched later holdout.
2. **High: outcome-stratifying the test weakens the clean holdout claim.** Sampling test rows conditional on their realized outcome is defensible for precision only if inclusion probabilities/weights are explicit and every assessment statistic is appropriately interpreted. Here it is asserted harmless, not demonstrated. Prefer simple random sampling within time/term, or retain the full test population.
3. **Medium: representativeness evidence is incomplete.** "Numeric origination columns" and six categoricals are not enumerated; neither are missingness distributions, stratum-level results, chi-square statistics, or sample inclusion probabilities.
4. **Medium: the 60-month vintage boundary is unexplained.** Eligibility says 2010-2013, while training begins 2010-05. If January-April are absent, say so; if excluded, count and justify their removal.
5. **Low: success criteria mix forecasting and realized-return claims without a decision counterfactual.** Realized total_pymnt - funded_amnt is acceptable only in Assess, but "decline" profitability requires an explicit convention for capital redeployment and selection effects.

**Score: 7.5/10. Sign-off: No.**

### Changed after round 1
- "Sealed" replaced by "held out" everywhere, with the aggregate exposure disclosed.
- The 2015 test is now the **full population (282,787 loans), not sampled**; only the validation sample (development data) is outcome-stratified.
- Representativeness: all 59 numeric and six categorical columns enumerated, KS, PSI, missing shares, chi-square and Cramer's V, per-column CSVs, inclusion probability stated.
- First 60-month loan is May 2010 (explained). C4 counterfactual convention stated.
### Rejected
- Nothing.

## Round 2 (verbatim, Part 1 and the Sample score from the later round)

All five round-1 findings: Resolved. New defect: "S.6 says the 36-month test has 70,696 rows, contradicting S.2/S.4's full-2015 test of 282,787." (Fixed.) Later in round 2 the critic scored the stage 8.4/10: "One remaining arithmetic defect: S.4 states 544,857 sampled loans, but the listed partitions total 544,858." Sign-off: No.

### Changed after round 2
- S.6 test count corrected; S.4 total corrected to 544,858 (166,459 + 40,643 + 282,787 + 54,969).

## Round 3 (verbatim)

3. S.4 arithmetic: Resolved. Components sum to 544,858. Framing + Sample stage **9.7/10, sign-off: Yes.**

## What I verified by hand
- Funnel arithmetic (2,260,701 - 915,351 = 1,345,350 and the later steps to 666,785); partition counts re-added; learning-curve values copied from `learning_curve.csv`.
