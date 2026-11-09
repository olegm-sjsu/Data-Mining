# A: Assess

Draft v2 (after critic round 1). The out-of-time test is the **full 2015 population of 36-month loans, 282,787
loans**. It was scored **once per frozen candidate** (`artifacts/test_access_log.json`; the
frozen-candidate hash was checked first). The test is "held out", not untouched at the
aggregate level: its higher default rate was seen during planning (`00_problem_framing.md`).
Nothing was changed after scoring. Numbers: notebook section A, `artifacts/test_*.csv`.

Observed default rate **14.88%**. Accept-all return on funded dollars **6.55%** (9.09% on the
2014 validation loans).

## A.1 Discrimination

| Model | Test ROC-AUC | PR-AUC | KS | Gini | Validation AUC | Change |
|---|---|---|---|---|---|---|
| FICO-only logistic | 0.6008 | 0.187 | 0.152 | 0.202 | 0.5964 | +0.004 |
| Scorecard A-core | 0.6456 | 0.228 | 0.211 | 0.291 | 0.6489 | -0.003 |
| Decision tree A-extended | 0.6400 | 0.224 | 0.206 | 0.280 | 0.6426 | -0.003 |
| LightGBM A-core | 0.6516 | 0.231 | 0.220 | 0.303 | 0.6555 | -0.004 |
| Scorecard A-extended | 0.6664 | 0.246 | 0.242 | 0.333 | 0.6641 | +0.002 |
| MLP A-extended | 0.6655 | 0.244 | 0.239 | 0.331 | 0.6619 | +0.004 |
| **LightGBM A-extended (Model A champion)** | **0.6793** | 0.257 | 0.260 | 0.359 | 0.6784 | +0.001 |
| Scorecard B | 0.6871 | 0.262 | 0.273 | 0.374 | 0.6812 | +0.006 |
| LightGBM B | 0.6927 | 0.270 | 0.279 | 0.385 | 0.6888 | +0.004 |

Validation and test AUCs agree within 0.006 for every model, so validation tuning did not overfit.
Lending Club's own grade and rate add **0.0134** AUC to the champion (Model B 0.6927 vs Model A
0.6793); the bureau block adds 0.021 to the scorecard (0.6456 to 0.6664) and 0.028 to LightGBM (0.6516 to 0.6793).

**Decile lift table (Model A champion).** Deciles of predicted PD on the 282,787 test loans; "cumulative share"
is the share of all defaults in that decile and every riskier one (`test_decile_lift.csv`).

| Decile (0 = safest) | Loans | Mean PD | Default rate | Lift | Cumulative share of defaults (from the riskiest) | Return on funded |
|---|---|---|---|---|---|---|
| 0 | 28,279 | 3.1% | 3.4% | 0.23 | 100.0% | 7.6% |
| 1 | 28,279 | 5.4% | 6.4% | 0.43 | 97.7% | 8.2% |
| 2 | 28,278 | 7.1% | 8.5% | 0.57 | 93.4% | 8.1% |
| 3 | 28,279 | 8.8% | 11.0% | 0.74 | 87.7% | 7.6% |
| 4 | 28,279 | 10.5% | 12.4% | 0.83 | 80.3% | 7.9% |
| 5 | 28,278 | 12.4% | 15.1% | 1.01 | 72.0% | 7.2% |
| 6 | 28,279 | 14.6% | 17.3% | 1.16 | 61.9% | 6.5% |
| 7 | 28,278 | 17.4% | 20.3% | 1.37 | 50.2% | 5.5% |
| 8 | 28,279 | 21.1% | 23.5% | 1.58 | 36.6% | 4.5% |
| 9 | 28,279 | 28.9% | 30.9% | 2.08 | 20.8% | 1.2% |

The riskiest decile has lift 2.08 and contains 20.8% of all defaults; the riskiest 30% contain 50.2%.

## A.2 Calibration (criterion C3)

Raw: calibration slope **0.955**, intercept 0.094, mean predicted PD 12.93% against observed
14.88%, **a gap of 1.96 points**. The model **under-predicts in every decile** (for example
8.75% predicted against 11.02% observed in the fourth decile). A Platt recalibration fitted on the 2014
loans only moves the mean prediction to 13.61% (gap 1.28 points), slope 0.973, Brier 0.1204 vs
0.1206. Cause: default rates rose from 13.7% (2014) to 14.9% (2015) and the model learned
earlier vintages. Calibration therefore needs refreshing on recent vintages, even though the
ranking holds. The Platt recalibration is **diagnostic only**: it was fitted after the
test was scored (on 2014 data), is not part of the frozen policy, and does not change the
C3 verdict, which is judged on the frozen raw model.

## A.3 Profit under the stated convention (criterion C4)

Policy frozen on validation: accept if PD <= 0.38. On the test it accepts **99.5%** of loans:
return on funded dollars **6.599%** against 6.548% for accept-all, a difference of **+0.051
points** (95% month-block bootstrap CI 0.042 to 0.061; 12 issue-month blocks). Total profit 237.70
million against 237.07 million (+0.64 million, +0.27%). Statistically positive, economically
negligible: the policy declines only 0.5% of loans.

Accepting only the best-ranked share (Model A):

| Share accepted | Return on funded | Total profit |
|---|---|---|
| 100% | 6.55% | 237.1 M |
| 90% | 7.09% | 232.9 M |
| 80% | 7.37% | 218.3 M |
| 70% | 7.61% | 200.7 M |
| 50% | 7.87% | 154.7 M |

Declining loans raises the return per dollar (+1.3 points at 50%) but lowers total profit, because
**even the riskiest decile earned a positive 1.3%**. Lending Club's rates rise with risk, so a risk
score alone does not justify declining loans under realized cash flows; its value is in pricing and
limit setting. Model B performs similarly (7.68% at 50%). (The FICO-only curve is flat in places
because FICO takes few distinct values, so a share cutoff cannot split ties.)

## A.4 Stability (criterion C5)

Score PSI against training: 0.009 (validation), **0.007 (test)**. Quarterly test AUC 0.673, 0.676,
0.682, 0.683 (overall 0.679; all within 0.006, well inside the 0.03 limit). Predicted-minus-observed gaps by quarter: 1.50, 2.20,
1.83, 2.18 points. Feature PSI against training: FICO 0.051, DTI 0.075, revolving utilization 0.040,
loan amount 0.020, income 0.011, loan-to-income 0.003, total high credit limit 0.016, bank-card open
to buy 0.017. Inputs and score distributions are stable; the calibration gap is a level shift in
default behaviour, not an input shift.

## A.5 Error slices and fairness review (criterion C6)

`zip_code` and `addr_state` are not model inputs. Two policies are shown, because the profit
cutoff approves about 99% of everyone and cannot reveal a disparity: the cutoff, and an
**illustrative strict policy** that declines the riskiest 20% (cutoff at the validation 80th
percentile).

| Slice | Default | Mean PD | AUC | Approval (cutoff) | Approval (strict) |
|---|---|---|---|---|---|
| Income Q1 (lowest) | 19.3% | 18.1% | 0.639 | 98.7% | **58.8%** |
| Income Q2 | 15.7% | 13.9% | 0.668 | 99.5% | 78.4% |
| Income Q3 | 13.9% | 11.4% | 0.681 | 99.8% | 87.2% |
| Income Q4 (highest) | 11.1% | 8.8% | 0.692 | 99.9% | 93.7% |
| Rent | 17.6% | 15.1% | 0.662 | 99.2% | 71.8% |
| Mortgage | 12.1% | 10.7% | 0.684 | 99.8% | 88.4% |
| Grade A | 5.4% | 6.7% | 0.641 | 100% | 98.8% |
| Grade E | 32.8% | 22.9% | 0.572 | 95.5% | 35.1% |
| Small business | 23.0% | 20.9% | 0.624 | 96.2% | 45.1% |
| Ten largest states | 12.8% to 16.4% | 12.1% to 13.6% | 0.661 to 0.698 | 99.4% to 99.6% | 77.6% to 83.1% |

- **C6 flag:** under the strict policy, the lowest income quartile's approval rate is **0.627** of
  the highest quartile's, below the 0.8 flag. The difference follows real risk (19.3% against
  11.1% defaults) and income is a legitimate input, but a lender must still justify it
  under fair-lending rules, because low income can proxy for protected groups.
- No state group is flagged: the ratio of the lowest to the highest strict approval rate is 0.934.
- Renters against mortgage holders: ratio 0.81, just above the line.
- **Within-grade discrimination is weak** (AUC 0.57 to 0.64 inside each grade). In grades D and E
  the model under-predicts defaults (26.2% observed against 19.6% predicted; 32.8% against 22.9%),
  and it over-predicts in grade A. Its predictions are compressed relative to Lending Club's grade.

## A.6 Secondary check: 60-month loans

A scorecard trained on 2010-2011 60-month loans scores ROC-AUC 0.602 and KS 0.150 on the 34,382
loans of 2013 (validation: 0.629). Observed default 25.2%, mean PD 26.0%. It is a secondary check
and is not deployed.

## A.7 Python and SAS

SAS was not available for this project, so there is no SAS flow, screenshot or report, and no
Python versus SAS comparison. A SAS score-code export of the A-extended scorecard is provided
(`deployment/scorecard_score_code.sas`) and was checked against a Python re-implementation; it has
**not been executed in SAS**.

## A.8 Criteria scoreboard (fixed in `00_problem_framing.md`)

| ID | Result | Verdict |
|---|---|---|
| C1 | AUC **0.6793** (target 0.68), +0.0785 over FICO-only | **FAIL** by 0.0007 |
| C2 | KS 0.2598 (target 0.25) | PASS |
| C3 | slope 0.955; mean gap 1.96 points (target 1.5); 1.28 after Platt on 2014 | **FAIL** |
| C4 | +0.051 points, CI [0.042, 0.061] | PASS, but economically negligible |
| C5 | PSI 0.007; quarterly AUC within 0.006 of overall | PASS |
| C6 | lowest income quartile approval ratio 0.627 under the strict policy | **FLAG** |

**Two criteria fail (C1 by a hair, C3 clearly), one is flagged (C6), and one passes without
economic meaning (C4).** C1 missed its pre-set line by 0.0007, but a miss is a miss. C3 is a real
weakness: predictions are systematically low for 2015. C6 is a genuine concern for a lender. The
model is a modest but real risk ranker (Gini 0.36), stable over time, but not calibrated to 2015
and not a strong lever for profit through declines.

## A.9 Champion

**Champion: LightGBM A-extended (Model A).** It was chosen by the pre-registered rule (0.014 better
than the best scorecard on validation) and the test confirms it (0.6793 against 0.6664 for the
A-extended scorecard). The A-extended scorecard is kept as the transparent alternative and has the
SAS export. Recommended use: rank applicants for pricing and limit decisions and refresh calibration
on recent vintages; do not use it alone to decline loans. Model B is not deployed.

## Open items

1. A true portfolio test needs discounted cash flows and a counterfactual for declined loans.
2. The fairness review has no protected-attribute data, so it is a proxy and slice analysis only.
