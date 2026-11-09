# Modeling, rounds 1 and 2

- **Critic model:** GPT-5.6 Terra (Codex), CRISP-DM authority persona, fresh context each round
- **Drafter:** Claude Sonnet 5.5
- **Scores:** round 1 = 7.5, round 2 = 9.1 (sign-off yes)
- **Inputs sent:** the phase note only

## Round 1 (verbatim)

| Item | Status |
|---|---|
| Select techniques, with assumptions | Partial |
| Generate test design: temporal holdout, rolling CV, sealed test | Complete |
| Build: majority, logistic, tree, boosted; parameters/descriptions | Partial |
| Optional clustering, visualization, use if helpful | Complete |
| Assess: comparison with fold variation and revised parameters | Partial |

1. **High: model selection is not evaluated on its declared primary metric.** The protocol says expected cost/net benefit is primary, yet the rolling-fold comparison and tuning select LightGBM by PR-AUC. Net benefit is shown only once for the final validation split. This does not establish that the final logistic choice, nor the LightGBM tuning, is robust under the actual decision objective and weekly cap.
2. **High: parameter/model specification is inadequate for reproducibility and CRISP-DM "build model."** "LightGBM, default" is not a parameterization; logistic gives only C=1; the tree gives only depth; preprocessing, class weighting, regularization/solver, objective, categorical treatment, and seed application are absent. "Six settings" is asserted but not tabulated. Artifact references are not descriptions.
3. **Medium: fold variation is selectively reported.** The table gives PR-AUC SD only, despite ROC-AUC and Brier being comparison metrics and net benefit being primary. The four LightGBM PR-AUC values do not substitute for per-fold results for every candidate or for the capped policy.
4. **Medium: technique assumptions are shallow and omit the consequential ones.** "Roughly linear log-odds" and "needs tuning" are not sufficient treatment of stationarity/covariate shift, feature availability at the declared decision point, duplicate dependence, calibration stability, or k-means' sparse/mixed encoded geometry.
5. **Low: D1 is handled honestly, but the supplementary argument should not soften the failure.** The arithmetic is sound: 0.7187-0.7096=0.0091, far below LR fold SD 0.0560; the listed paired differences also average 0.0091. They correctly retain the fail and choose logistic. Calling D1 "harsh" is interpretation, not a basis to rehabilitate it.

**Score: 7.5/10. No sign-off.**

### Changed after round 1
- Net benefit (fixed closed-form threshold 0.533, weekly cap, raw probabilities) added to every fold of every model. Result: tuned LightGBM leads logistic regression on mean fold net benefit (+212 per 1,000, ahead in 3 of 4 folds).
- Full specification table, six-setting grid tabulated, per-fold table for all five models, technique-risk table, D1 wording rewritten.
- LightGBM frozen as a secondary model so that Phase 5 can report it beside the chosen one.
### Rejected
- Nothing. The model choice was deliberately **not** changed after seeing the fold net benefits: the validation rule was fixed beforehand. The conflict is stated openly and handled in Phase 5.

## Round 2 (verbatim)

- Fix 1, Resolved: primary net benefit is now reported across folds; keeping LR under the pre-set validation rule is defensible.
- Fix 2, Partly: specifications are strong, but LightGBM's stated `subsample=0.8` "every iteration" is not reproducible as written without `subsample_freq`/`bagging_freq`.
- Fix 3, Resolved: all-model, per-fold PR-AUC and net benefit plus summary dispersion are present.
- Fix 4, Resolved: technique assumptions and shared data risks are materially improved.
- Fix 5, Resolved: D1 remains explicitly failed, without rehabilitating it.

New defects:
1. **High:** The final hyperparameter grid selects tuned LightGBM by PR-AUC, not the declared primary decision metric; distinguish tuning criterion from final policy selection or tune on net benefit.
2. **Medium:** LightGBM subsampling specification is internally inaccurate/incomplete (`subsample` alone may not activate row subsampling).
3. **Minor:** Grid Brier scores lack fold dispersion, unlike PR-AUC.

The LR-versus-LightGBM treatment is unusually honest: it preserves the predeclared LR rule while exposing the conflicting fold evidence and freezing LightGBM for a labelled Phase 5 comparison.

**Score: 9.1/10. Sign-off: yes.**

### Changed after round 2
- `subsample_freq=1` stated explicitly (the code already set it).
- Note now separates the tuning criterion (PR-AUC) from the policy choice (net benefit), and lists tuning on net benefit as a possible refinement. Grid Brier dispersion not added (minor).

## What I verified by hand
- Paired differences recomputed from the per-fold tables: PR-AUC +0.0027, +0.0019, +0.0215, +0.0105 (mean 0.0091); net benefit +295, +255, +334, -36 (mean +212).
- Threshold 20 / (0.25 x 150) = 0.533 and the ratio 1,757.0 / 1,775.5 = 0.990.
- The test period was not used: `frozen_policy.json` has `test_opened: false`.
