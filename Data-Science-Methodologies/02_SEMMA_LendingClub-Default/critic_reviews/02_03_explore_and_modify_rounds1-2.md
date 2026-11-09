# Explore and Modify, rounds 1 and 2

- **Critic model:** GPT-5.6 Terra (Codex), "world-renowned SEMMA authority" persona, fresh context each round
- **Drafter:** Claude Sonnet 5.5
- **Inputs sent:** the stage notes only (they carry the real notebook outputs), never the notebook or the data
- **Scores:** round 1 = 8.3 (both stages), round 2 = Explore 9.2, Modify 9.0 (sign-off yes for both)

## Round 1 (verbatim, Part 2)

| Checklist item | Status |
|---|---|
| Univariate distributions, missingness, outliers | Complete |
| Default by grade, purpose, DTI, income, vintage | Complete |
| Correlation/association screening and IV | Partial |
| Borrower clustering with visualization | Partial |
| Findings logged into Modify | Complete |
| Leakage audit covering every column | Complete |
| Imputation plus justified missing indicators per variable | Partial |
| Outlier treatment | Complete |
| Transforms | Complete |
| Variable selection and redundancy removal | Complete |
| Train-only reusable pipeline | Complete |

1. **High: test-population contradiction:** S.6 reports a 70,696-row test while the asserted held-out full population is 282,787.
2. **Medium: missing-data treatment is not justified per variable:** M.2 provides broad groups, not a field-level decision for the 78 origination candidates. This matters especially where absence is calendar-driven and therefore a time proxy.
3. **Medium: clustering specification is internally inconsistent:** E.4 claims twelve standardized features but lists eleven.
4. **Low: association screening is thin:** numeric redundancy examples are given, but categorical-association methodology/results are not stated; IV alone is not a complete association screen.
5. **Low: visualization is only generically referenced:** `E_*.png` is not a specific cluster visualization or cited result.

**Score: 8.3/10. Sign-off: No.**

### Changed after round 1
- S.6 count fixed. New `missing_treatment` column for every origination field plus counts in the note (24 complete; 33 bureau block; 14 expanded fields; 4 calendar-dependent not selected; 2 structural; 1 rare).
- All twelve clustering features named; the cluster figure identified (PCA of 12,000 training loans, coloured by cluster).
- Cramer's V among categoricals and against the outcome added.
### Rejected
- Nothing.

## Round 2 (verbatim, Part 1)

- S.6 test-count contradiction: Resolved. Per-variable missing treatment: Resolved. Clustering feature list: Resolved. Association screening: Resolved. Cluster figure unspecified: Resolved.

New defects:
1. **Modify:** M.2 says the "about 40" bureau fields are handled in the extended variant, while the field-count accounting calls out 33 bureau-block fields; reconcile the exact scope.
2. **Explore:** IV uses "10 quantile bins" for all candidates, but this is not a defensible stated treatment for low-cardinality categoricals; specify their categorical binning rule.

| Phase | Score | Sign-off |
|---|---:|---|
| Explore | 9.2/10 | **Yes** |
| Modify | 9.0/10 | **Yes** |

### Changed after round 2
- "About 40" replaced by 33 everywhere; the categorical binning rule (own bin per category, under 0.5% pooled, missing as MISSING) stated in Explore.

## What I verified by hand
- FICO bin edges recomputed by hand and equal to the pipeline's; the lowest-bin weight of evidence recomputed by hand (-0.4709, equal). A first by-hand attempt did not match because it used `<=` where the binner uses `<` at the bin edge; corrected, not ignored.
