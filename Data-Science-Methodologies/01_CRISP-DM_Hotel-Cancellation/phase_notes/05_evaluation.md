# Phase 5: Evaluation

Draft v2 (after critic round 1). The test period (arrivals 2017-04-22 to 2017-08-31, 24,018 rows) was
opened once: `artifacts/test_access_log.json` shows one prediction call, and the
model file hash matches the frozen policy (`8c6cd3c57dfe46d8`). Everything below
comes from notebook section 5 and `artifacts/test_*.csv`. Test non-arrival rate:
**40.99%**. Intervals are 95% block bootstraps over the 20 arrival weeks.

## Test integrity statement

The test was opened **once for model predictions** (one `predict_proba` call per
model, log and hash check above). It was **not clean of earlier outcome exposure**:
before the test was sealed in code, the analyst's exploration scripts (run
by the AI assistant, reported to the analyst) looked at whole-file outcome
summaries. Exactly what was seen: the test-period non-arrival rate (about 41%),
the rate by hotel in the test period, quarterly rates including the test
quarters, and the rate of repeated booking profiles in the test period. No model
existed at that time. The features, model families, hyperparameter grid,
threshold, calibration choice and decision rule were all chosen on development
rows afterwards. The aggregate summaries could have shaped the narrative at most
(for example, the arrival-date split, which was motivated by file structure and
recorded as Amendment A1). So:

- **Confirmatory:** the pre-registered logistic regression, the pre-registered
  criteria B1 to B4 and D1 to D4, D6. The residual risk is small and disclosed.
- **Exploratory / descriptive, not confirmatory:** D5, the LightGBM comparison,
  the monthly diagnosis of summer degradation, slices, fairness and sensitivity.
  They were computed on the test and must be confirmed on new data before anyone
  acts on them.

## 5.1 Evaluate results against the business success criteria

| Model on the test | Flagged | Precision | Recall | Net benefit / 1,000 | PR-AUC | ROC-AUC | Brier | Cal. slope |
|---|---|---|---|---|---|---|---|---|
| **Logistic regression (pre-registered)** | 22.2% | 0.718 | 0.388 | **1,533** (CI 812 to 2,186) | 0.713 | 0.767 | 0.1885 | 0.879 |
| LightGBM (secondary, frozen before the test) | 22.5% | 0.773 | 0.424 | 2,023 (CI 1,512 to 2,529) | 0.749 | 0.792 | 0.1787 | 1.035 |
| Lead-time rule (`lead_time >= 226`) | 17.0% | 0.508 | 0.211 | -160 (CI -434 to 74) | 0.487 | 0.622 | n/a | n/a |

Base-rate Brier is 0.2440. Logistic regression beats the rule by 1,693 per 1,000
(CI 1,040 to 2,348). LightGBM beats logistic regression by 489 per 1,000
(CI 241 to 773), a difference that excludes zero.

### Criteria as pre-registered in Phase 1

| Criterion | Result | Verdict |
|---|---|---|
| B1: 10th percentile of net benefit per 1,000 above 0 (Monte Carlo over save rate, cost, loss) | 10th percentile **-3,640**, median 214 | **FAIL** |
| B2: net benefit at least 15% above the lead-time rule | 1,533 vs -160 | **Not interpretable** under the pre-registered ratio (the rule loses money on the test). Absolute improvement: +1,693 per 1,000 (CI 1,040 to 2,348) |
| B3: weekly cap binds in at most 10% of test weeks | binds in **65%** of the 20 weeks | **FAIL** |
| B4: median weekly aggregate error at most 0.10 | 0.048 (no zero weeks) | PASS |
| D1 (test): LightGBM minus logistic regression PR-AUC above 0.056 | +0.036 | **FAIL** |
| D2: recall at least 0.30 | 0.388 (CI 0.333 to 0.430) | PASS |
| D3: precision at least 0.60 | 0.718 (CI 0.637 to 0.788) | PASS |
| D4: calibration slope in [0.8, 1.2] and Brier below base rate | 0.879; 0.1885 vs 0.2440 | PASS |
| D5: gap Set A+B minus Set A (LightGBM, test) | 0.846 - 0.749 = **+0.097** | **PASS: leakage control enforced.** The final model uses Set A only; the gap exceeds 0.05, so Set B stays classed as post-booking. The validation-fold gap (Phase 4, +0.126) agrees in size and direction |
| D6: quarterly PR-AUC within 0.05 of overall | 2017Q2 0.796, 2017Q3 0.538, overall 0.713 | **FAIL** |

**Four criteria fail (B1, B3, D1, D6), one is not interpretable (B2), and D2, D3, D4, D5 and B4 pass.** The honest summary is a model that has real
predictive signal (precision 0.72, recall 0.39, positive net benefit under the
base assumptions) but is **not robust** to the cost assumptions (B1), degrades
sharply in summer (D6), and relies on a capacity cap that binds most weeks (B3).

### Why B1 fails: the economics rest on two unknown numbers

Net benefit per 1,000 for the frozen flags (L = 150):

| | c = 10 | c = 20 | c = 40 |
|---|---|---|---|
| s = 0.10 | 170 | -2,046 | -6,479 |
| s = 0.25 | 3,750 | **1,533** | -2,899 |
| s = 0.40 | 7,329 | 5,113 | 680 |

The break-even save rate for the frozen policy is `c / (L x precision)` =
0.186. If outreach keeps fewer than about 19% of would-be non-arrivals (at
€20 per outreach), the policy loses money. Both the save rate and the outreach
cost are assumptions with no data behind them, so **the profitability of this
project cannot be established from this dataset**. It must be measured in a
pilot.

### Why D6 fails: summer 2017

| Arrival month (test) | Rows | Precision | Recall | Net benefit / 1,000 | PR-AUC |
|---|---|---|---|---|---|
| April (from the 22nd) | 1,841 | 0.898 | 0.488 | 3,415 | 0.845 |
| May | 6,305 | 0.830 | 0.473 | 2,781 | 0.801 |
| June | 5,639 | 0.794 | 0.453 | 2,415 | 0.773 |
| July | 5,310 | 0.535 | 0.192 | 9 | 0.566 |
| August | 4,923 | 0.519 | 0.339 | -134 | 0.502 |

Performance falls steadily through the test period and is near zero net benefit in
July and August (-60 per 1,000 combined). The cause is not identified here.
The training data holds summer 2015 and 2016, so this is not simply "unseen
season"; drift in who books and cancels in 2017 is the likely explanation, but
that is a hypothesis.

### Duplicates and profile twins (3.5, 4.6)

| Test subset | Rows | Precision | Recall | Net benefit / 1,000 | PR-AUC |
|---|---|---|---|---|---|
| All rows | 24,018 | 0.718 | 0.388 | 1,533 | 0.713 |
| No profile twin earlier | 19,131 | 0.728 | 0.450 | 1,898 | 0.743 |
| Profile twin earlier | 4,887 | 0.574 | 0.109 | 105 | 0.513 |

On the test, rows with an earlier twin are **harder**, not easier. The
validation-segment finding that duplicates in the training data help (4.6) does
not carry to the test, so the suspicion that repeats inflate performance is not
supported; the headline is not flattered by twins.

## Calibration, slices, SHAP, fairness

**Calibration.** Slope 0.879 and a Brier score of 0.1885 against 0.2440 for the
base rate. By decile, the lowest-risk tenth had mean prediction 0.036 but
actual rate 0.091, and the highest tenth 0.847 predicted against 0.890 actual
(`05_calibration_and_threshold.png`). Weekly totals of predicted probabilities
are close to actual non-arrivals (B4).

**Error slices (test).**

| Slice | Rows | Actual rate | Mean predicted | Flagged | Precision | Recall | Net benefit / 1,000 |
|---|---|---|---|---|---|---|---|
| City Hotel | 16,463 | 0.434 | 0.445 | 28.1% | 0.709 | 0.459 | 1,855 |
| Resort Hotel | 7,555 | 0.356 | 0.280 | 9.2% | 0.774 | 0.201 | 833 |
| Groups | 2,743 | 0.661 | 0.716 | 73.5% | 0.812 | 0.902 | 7,667 |
| Offline TA/TO | 3,843 | 0.354 | 0.323 | 26.5% | 0.748 | 0.559 | 2,128 |
| Online TA | 13,961 | 0.430 | 0.400 | 15.8% | 0.619 | 0.228 | 510 |
| Direct | 2,653 | 0.184 | 0.193 | 1.2% | 0.452 | 0.029 | -36 |
| Corporate | 640 | 0.242 | 0.185 | 7.2% | 0.848 | 0.252 | 848 |

The resort hotel's actual rate (0.356) is well above its mean prediction
(0.280): the model under-predicts the resort's drift. **Groups carry the
benefit** (7,667 per 1,000 rows, 11% of the rows). Each row is a booking, and a
group may be one phone call, so outreach cost per row is probably overstated
for groups and the groups' benefit uncertain in the other direction.

**SHAP (logistic regression, association only).** Mean absolute SHAP by
feature, in log-odds: `market_segment` 1.25, `agent_id` 1.21, `customer_type`
0.83, `log_lead_time` 0.61, `hotel` 0.59, `total_guests` 0.38,
`previous_cancellations` 0.34, `adults` 0.33. A longer lead time raises risk
(correlation of the log lead time with its SHAP value is +1.0). These are
associations in a retrospective dataset. For example, `agent_id` is a proxy for
how a booking was sourced, not a cause hotel staff can change.

**Fairness risk of `country`.** `country` is not a model input (Set B), but other
inputs can act as proxies. By country group on the test:

| Country | Rows | Actual rate | Mean predicted | Flagged | Recall | False-positive rate | Precision |
|---|---|---|---|---|---|---|---|
| PRT | 7,164 | 0.636 | 0.444 | 38.1% | 0.558 | 0.070 | 0.933 |
| DEU | 1,862 | 0.212 | 0.403 | 23.4% | 0.287 | **0.219** | 0.260 |
| GBR | 3,291 | 0.276 | 0.341 | 18.6% | 0.332 | 0.130 | 0.492 |
| FRA | 2,112 | 0.288 | 0.383 | 13.1% | 0.217 | 0.096 | 0.478 |
| ESP | 1,467 | 0.389 | 0.348 | 12.7% | 0.208 | 0.075 | 0.640 |
| other | 8,122 | 0.345 | 0.378 | 13.4% | 0.217 | 0.089 | 0.562 |

German guests who would have arrived are contacted wrongly 3.1 times as often as
Portuguese guests (false-positive rate 0.219 vs 0.070); the flagged-share ratio
across groups is 3.0. The harm is an unneeded call or offer, not a denial of
service, but the burden is uneven and the model over-predicts for DEU, GBR and FRA.

## Sensitivity and break-even

Weekly cap, threshold fixed (net benefit per 1,000): cap 10% = 855, 15% =
1,207, 25% = 1,533, 40% = 1,526, no cap = 1,526. Raising the cap above 25% adds
nothing, because the threshold, not capacity, becomes the limit. B3 therefore
"fails" on its wording, while the economics are indifferent between 25% and no
cap.

Bookings per year in the data are about 54,980 (whole-file count over the
span). Year-one cost is €3,240 (development 1,800 plus operating 1,440). At the
test average (peak season, assumed save rate) the annual net benefit would be about
€84,000 and payback about 0.5 months. At the B1 Monte Carlo median (214 per 1,000) it would be
about €11,800 (payback 3.3 months), and at the July-August level (-60) it never
pays back. The first number is the least reliable, since it extrapolates four
peak months to a year.

## 5.2 Review process

What went well: a pre-registered criteria set, a sealed test opened once, honest
disclosure of failures, and constructive amendments (Phases 1 to 4).
What was missed or should be repeated:

1. The cap definition: B3 was written as "cap binds in at most 10% of weeks",
   which punishes a model whose threshold flags a large share of high-risk
   groups, even when capacity is not the limiting factor.
2. The pre-registered choice rule picked logistic regression on a single
   validation segment, while the rolling folds and the test favor LightGBM
   (+212 and +489 per 1,000). A rule based on the mean over the rolling folds
   would have chosen LightGBM. This is a design weakness in the choice rule.
3. The summer degradation was found only on the test; the folds all precede it.
   A seasonally stratified validation would have shown it earlier.
4. D5, the LightGBM comparison, the monthly diagnosis and the go/no-go
   discussion all use the test. They are descriptive after the early exposure and
   cannot carry confirmatory weight (see the integrity statement). No model,
   feature or threshold was changed after the test was opened.
5. Aggregate test outcomes were seen early (Phase 3 exposure log); the
   consequences are stated above.

## 5.3 Determine next steps and decision

| Option | Assessment |
|---|---|
| Deploy the model as an automated policy | **No-go.** B1, B3 and D6 fail; July-August net benefit is negative; profitability depends on unmeasured inputs |
| Stop the project | Not justified: the model separates risk (precision 0.72 against a 0.41 base rate) |
| **Randomized pilot (research study, not a rollout)** | **Conditional go**, only with the design below |
| Switch the champion to LightGBM now | Not decided here. It leads on the folds and the test, but choosing it using the test would make its test number optimistic. Run it as a shadow challenger and confirm on newer data |

**Decision: NO-GO for deployment as an automated policy. Conditional GO for a
randomized pilot**, with the pre-registered logistic regression as champion and
LightGBM as a shadow challenger. This is a go for gathering the missing
evidence, not evidence of readiness. Pilot design (planning estimates, to be fixed by the hotel before it starts):

1. **Unit and arms:** the data has no group identifier, so the pilot covers
   **non-Group flagged bookings, randomized per booking row**, 50/50 to outreach
   or no outreach. Group bookings (38% of flagged rows in the test) are excluded
   until a group ID exists, and then they are randomized per group.
2. **Outcome:** arrival rate among flagged bookings. Applying the pilot rules
   (Groups never take a weekly slot; Phase 6 code, notebook 6.2), the test has
   4,229 eligible flagged bookings with precision 0.637, so the control arrival rate
   is about 0.36. (The first estimate, 0.66 from the segment slices, was made before
   eligibility was applied ahead of the weekly cap; the deployed rule is the one used
   here.)
3. **Hypothesis and sample size:** the rollout rule below needs the *lower* 95%
   bound of the save rate to exceed the break-even 0.186. To have 80% power
   for that when the true save rate is 0.30 (an assumed value, not evidence),
   with a one-sided 2.5% test on the difference in arrival rates (margin
   0.186 x 0.637 = 0.118, assumed effect 0.30 x 0.637 = 0.191), about
   **710 flagged bookings per arm (about 1,420 in total)** are needed. Smaller true effects need more:
   they are not detectable with this size.
4. **Duration:** the eligible flagged volume in the test was about 980 per month, so the sample takes
   about 1.4 months. The pilot should still run **at least one quarter and
   include July-August**, where the model is weakest.
5. **Decision rule:** roll out only if the 95% lower bound of the save rate is
   above 0.186 at the observed cost; stop if the upper bound is below it;
   otherwise extend the pilot.
6. **Guardrails:** monitor false-positive rates by country group (target ratio
   under 1.5), consider removing `agent_id` as a proxy, redefine B3 as a capacity
   requirement (flagged share per week), and confirm Set A availability against a
   real booking-time log.
7. **Before summer:** retrain on fresh data and investigate the July-August drop.
