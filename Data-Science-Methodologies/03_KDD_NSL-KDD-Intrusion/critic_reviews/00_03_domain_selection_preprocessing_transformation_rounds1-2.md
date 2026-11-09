# Domain and goals, Selection, Preprocessing, Transformation, rounds 1 and 2

- **Critic model:** GPT-5.6 Terra (Codex), "world-renowned KDD authority" persona, fresh context each round
- **Drafter:** Claude Sonnet 5.5
- **Inputs sent:** the step notes only (they carry the real notebook outputs), never the notebook or the data
- **Scores:** round 1 = 7.5, round 2 = 9.0 then 9.5 (sign-off yes)

## Round 1 (verbatim)

| Stage | Checklist item | Status |
|---|---|---:|
| Before Selection | Domain background | Complete |
| Before Selection | Goals/questions; novel, valid, useful, understandable | Partial (K1-K5 are questions with thresholds, but several operationalizations are incomplete or self-contradictory) |
| Selection | Files, records, features; provenance; attack types only in test | Complete |
| Preprocessing | Names/types; missingness, duplicates, invalids, constants, labels; outliers; five-category labels | Complete |
| Transformation | Log, scale, encode; PCA/MI reduction; rule discretization | Complete |
| Transformation | Train-only reusable pipeline | Partial (split-specific fitting for R/N validation is not specified) |

1. **High: test-label reconnaissance weakens the "held-out" claim.** Test attack-type names and counts were inspected before modeling. This is required by the assignment's novel-type analysis, but it means the test is not blind. Freeze all choices now; call it an externally evaluated, label-characterized test set, not an untouched holdout.
2. **High: K1 violates the stated validity criterion.** Its pass condition is training NMI only. A clustering pattern is not "valid" unless assessed out of sample; evaluate the fixed clustering procedure on test without refitting.
3. **High: R/N validation cannot support reproducible model selection as written.** "Hyperparameters chosen on R and N" supplies neither a selection rule nor weights/tradeoff. N has only 9 U2R and 7 R2L examples for two held types. Pre-register the metric hierarchy, confidence intervals, and what happens when R and N disagree.
4. **Medium: K2/K4 are insufficiently falsifiable.** "Best rules" is undefined and invites test-set cherry-picking; K4 does not require five-class per-class recall, especially U2R.
5. **Medium: transformation leakage boundary is underspecified.** A pipeline fit on all KDDTrain+ is appropriate only for final refit, not R/N validation. PCA, scaling, MI, discretization cutpoints, and high-prevalence filtering must be fit separately within each validation fit partition.

**Score: 7.5/10. Sign-off: No.**

### Changed after round 1
- Test described as a characterized test set held out from fitting, tuning, thresholds and rule selection. K1 now also needs NMI at least 0.40 on KDDTest+ with the fitted k-means applied unchanged.
- Model-selection rule pre-registered (macro-F1 on R, eligible within 0.02, then recall on N's held-out types with bootstrap intervals, simpler model if intervals overlap, N read in aggregate).
- K2 rule selection specified (training only, top five per category fixed before the test); K4 mandates all five classes with U2R recall, count and interval.
- Leakage boundary: every learned step refit on each validation view's fit rows; the all-train pipeline is for exploration, final refit and test only.
### Rejected
- Nothing.

## Round 2 (verbatim)

All five round-1 issues: Resolved. New defect: "K3 says anomaly detection is trained on normal only, but Transformation assigns anomaly detection to PCA fitted on all KDDTrain+. The notes must explicitly establish a separate normal-only PCA/reconstruction fit, or admit that the strict normal-only condition is not met." Score 9.0/10, sign-off: No. After the fix (clustering uses the 23 all-train principal components; each anomaly detector fits its own preprocessing and PCA on normal records only), the same reviewer scored the steps **9.5/10, sign-off: Yes**.

## What I verified by hand
- Test-21 equals exactly the KDDTest+ records with difficulty at most 20 (11,850 of 11,850 and 11,850 such rows); the mean and standard deviation of log1p(src_bytes) match the pipeline's; counts of novel types re-added (3,750).
