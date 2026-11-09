# Extension: deployment and monitoring (after Assess, outside SEMMA)

SEMMA stops at Assess. This stage is an **extension** that makes the champion usable and
watchable. Draft v2 (after critic rounds 1 and 2). Code is in `deployment/`; checks run in notebook section "Extension".

## Batch scoring script with input validation

`deployment/score_loans.py applications.csv scored.csv [rejects.csv]` scores applications with
the frozen **Model A champion** (LightGBM A-extended, `model_champion.joblib`, hash recorded in
`policy.json`). Rows that fail validation are not scored; they go to a rejects file with the
reasons.

| Validation rule | Reason |
|---|---|
| 14 required input columns present (11 credit-bureau columns are optional) | the model needs them |
| `fico`, `annual_inc`, `loan_amnt`, `dti`, `revol_util`, `inq_last_6mths`, `revol_bal` not missing | the core application fields; a missing one is rejected, not imputed |
| `term` must be "36 months" | the 60-month model is a secondary check and is not deployed |
| FICO in [300, 850], income above 0 and at most 10 million, loan amount in [500, 40,000], DTI in [0, 100], utilization in [0, 300], inquiries in [0, 50] | implausible values |
| `issue_d` and `earliest_cr_line` in "Mon-YYYY" format; credit line not opened after the loan | the credit-history feature needs them |
| `home_ownership` in a known set | unseen categories would be silently coded |

Output columns: `pd` (probability of default), `bureau_fields_missing` (a warning count: missing credit-bureau fields, or bureau columns absent from the file altogether, are allowed because the model was trained to handle them, but the score is less reliable), `decision` (accept if PD <= 0.38, the validation
profit cutoff), and `decision_strict` (the illustrative policy that declines the riskiest 20%,
cutoff 0.190 from validation).

**Model integrity.** `score_loans.load()` computes the full SHA-256 of `model_champion.joblib` and refuses to run if it differs
from `champion_file_sha256` in `policy.json`. The notebook proves it by appending one byte to a copy of the model
and confirming the script is refused.

**Checked in the notebook:** 600 random test applications scored through the script give the same
probabilities as the notebook (tolerance 1e-10), and six deliberately bad rows are all rejected
with the right reason: negative income, 60-month term, FICO of 120, a malformed issue date, an
unknown home-ownership value, a missing DTI. On the 600 valid rows, 99.5% are accepted by the profit cutoff and 77.8% by the
strict policy. A file with no bureau columns at all is still scored (20 test rows, `bureau_fields_missing` = 11). `deployment/sample_applications.csv` (606 rows: 600 valid, six deliberately invalid) is the example input.

## SAS score code (generated, not executed in SAS)

SAS was not available, so no SAS flow or report exists. `deployment/scorecard_score_code.sas` (218 lines)
reproduces the **A-extended WoE scorecard** as a SAS DATA step (bin edges, WoE per bin, coefficients,
probability). A Python re-implementation that reads only the bin tables matches the model on 3,000
test loans to 1.7e-16, which checks the generator's logic. **It has not been run in SAS.** The input
table needs the engineered variables built as in `lc_features.add_features` (the file header lists them).

## Monitoring plan

The "Status" column separates **library functions** in `lc_monitor.py` (calculations and alert rules, run by hand or
wired to a scheduler by the lender) from **procedures** that need people, data feeds or matured outcomes. There is no
scheduled job, stored reference distribution or dashboard in this repository.

A 36-month loan's default outcome is known only after about three years, so outcome-based checks
(calibration, AUC) lag the portfolio by years. Input and score drift can be checked at once, and
early signals (delinquency in the first months) should be added when available.

| Check | Cadence | Threshold | Action | Status |
|---|---|---|---|---|
| Score PSI against the training scores | Monthly | 0.10 watch, 0.25 retrain | Investigate; retrain | Library function (`lc_monitor.psi`, `alerts`); not scheduled |
| Feature PSI (FICO, DTI, income, amount, bureau fields) | Monthly | same | Investigate the field | Library function (`lc_monitor.psi`); not scheduled |
| Predicted vs observed default rate per vintage | Quarterly, lagged | gap above 2 points | Recalibrate | Library function (`vintage_report`, `alerts`); not scheduled |
| AUC by vintage | Quarterly, lagged | drop of more than 0.03 | Retrain | **Procedure** (needs matured outcomes) |
| Approval-rate ratios by income band, state, home ownership | Monthly | below 0.8 | Fairness review | Library function (`alerts`); not scheduled |
| Scheduled retraining | Yearly, and after any retrain trigger | n/a | Retrain on the latest mature vintages | **Procedure** |
| Rollback | On a trigger | n/a | Restore the previous model file and `policy.json` | **Procedure** (manual file swap) |

**Demonstration on the 2015 test stream** (`lc_monitor.py`): score PSI against training is 0.007
(no alert). The calibration check fires: the predicted-minus-observed gap is 1.50, 2.20, 1.83 and 2.18
points for 2015Q1 to Q4, so the monitor flags 2015Q2 and 2015Q4 for **recalibration**, which is the
finding of the Assess stage. The approval-ratio check flags the lowest income quartile (0.63).

## Limits

1. Decision support only: the profit evidence for declining loans is negligible (Assess A.3); the best
   use is pricing and limit setting.
2. Calibration drifts with the default cycle and needs refreshing on recent vintages.
3. No protected-attribute data exist, so the fairness checks are proxy and slice analyses.
4. The 60-month term is not served.
5. The decision thresholds were set on 2014 data.
