# Phase 4: Modeling

Draft v2 (after critic round 1). Everything below uses the fit, calibration and validation segments
and rolling folds inside them. **The test period is still sealed**
(`artifacts/frozen_policy.json` has `test_opened: false`). Numbers come from
notebook section 4 and `artifacts/*.csv`.

## 4.1 Select modeling techniques

| Technique | Why | Assumptions and risks |
|---|---|---|
| Majority (prior) baseline | Floor for every metric | none |
| Logistic regression | Interpretable, usually well calibrated | Roughly linear log-odds in the scaled and one-hot inputs; sensitive to covariate shift in `lead_time` and segment mix |
| Decision tree, depth 8 | Readable rules, interactions | Needs the depth limit; unstable under small data changes |
| LightGBM boosted trees | Strong on mixed tabular data | Needs tuning; raw probabilities may need calibration; can memorize repeated profiles |
| k-means on the prepared matrix | Guest segments, optional | Assumes round clusters in a space made of scaled numbers and many one-hot columns, so distances are dominated by a few numeric fields |

**Risks that apply to every model** (they come from the data, not the algorithm):
(1) *non-stationarity*: the non-arrival rate moves by quarter, and the test is a
spring and summer window; (2) *feature availability*: Set A is classified by
definition, not proven, and the history counters (A2) are the least certain;
(3) *duplicate dependence*: repeated booking profiles help prediction (4.6);
(4) *calibration stability*: a calibrator fitted on one segment can hurt another
(4.8).

## 4.2 Generate test design

Four expanding-window folds by arrival date inside the first 80% of rows
(training rows 47,557, 59,584, 71,465, 83,319; about 12,000 validation rows
each). Final design: fit 0-60%, calibrate 60-70%, choose threshold 70-80%. The
test is sealed. The primary metric is **net benefit per 1,000 bookings** of the
capped policy (Phase 1 cost model). Inside the folds the threshold is not
tuned: it is fixed at the closed-form break-even 20 / (0.25 x 150) = 0.533, on
raw probabilities, with the weekly 25% rank cap. Supporting metrics are PR-AUC,
ROC-AUC and Brier score.

## 4.3 Build models: specifications

All models sit behind the same pipeline: feature construction, median
imputation, clipping at the training 99.9th percentile, standard scaling, one-hot
encoding (categories under 200 training rows grouped), then the classifier. No
class weights, no resampling. Seed 42 everywhere.

| Model | Parameters |
|---|---|
| Majority | `DummyClassifier(strategy="prior")` |
| Logistic regression | scikit-learn defaults: L2 penalty, `C=1`, lbfgs solver, `max_iter=500` |
| Decision tree | gini, `max_depth=8`, `min_samples_leaf=100` |
| LightGBM default | binary objective, `n_estimators=300`, `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=50`, `subsample=0.8` with `subsample_freq=1` (row sampling every iteration), `colsample_bytree=0.8` |
| LightGBM tuned | as default, but `num_leaves=15`, `min_child_samples=100` |

Six LightGBM settings were compared on the same folds
(`artifacts/lgbm_grid.csv`):

| num_leaves | min_child_samples | learning_rate | n_estimators | PR-AUC mean | sd | Brier |
|---|---|---|---|---|---|---|
| **15** | **100** | **0.05** | **300** | **0.7187** | 0.0590 | 0.1726 |
| 31 | 50 | 0.05 | 300 | 0.7134 | 0.0627 | 0.1768 |
| 31 | 100 | 0.03 | 500 | 0.7133 | 0.0614 | 0.1762 |
| 63 | 50 | 0.03 | 400 | 0.7058 | 0.0601 | 0.1812 |
| 15 | 50 | 0.10 | 200 | 0.7126 | 0.0621 | 0.1757 |
| 31 | 200 | 0.05 | 300 | 0.7163 | 0.0658 | 0.1752 |

Selection used mean PR-AUC as the *tuning* criterion, while the *policy* choice between models uses net benefit (4.8). Tuning directly on net benefit is a possible refinement, not done here. Differences are small next to the fold sd, so the
tuning gain is modest.

## 4.4 Assess models (Set A, rolling folds)

Per fold, as PR-AUC / net benefit per 1,000 (full detail with ROC-AUC and Brier in
`artifacts/cv_folds_all.csv`):

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|
| Majority | 0.343 / 0 | 0.402 / 0 | 0.341 / 0 | 0.363 / 0 |
| Logistic regression | 0.628 / 580 | 0.754 / 1,836 | 0.721 / 1,204 | 0.735 / 1,835 |
| Decision tree | 0.576 / 598 | 0.660 / 1,658 | 0.647 / 1,195 | 0.672 / 1,381 |
| LightGBM default | 0.620 / 526 | 0.750 / 2,085 | 0.738 / 1,551 | 0.746 / 1,711 |
| LightGBM tuned | 0.631 / 875 | 0.756 / 2,091 | 0.743 / 1,539 | 0.745 / 1,799 |

Summary over the four folds:

| Model | PR-AUC mean | PR-AUC sd | ROC-AUC | Brier | Net benefit / 1,000 mean | sd |
|---|---|---|---|---|---|---|
| Majority | 0.3623 | 0.0287 | 0.500 | 0.2312 | 0 | 0 |
| Logistic regression | 0.7096 | 0.0560 | 0.787 | 0.1763 | 1,364 | 601 |
| Decision tree | 0.6387 | 0.0430 | 0.739 | 0.1868 | 1,208 | 449 |
| LightGBM default | 0.7134 | 0.0627 | 0.788 | 0.1768 | 1,468 | 667 |
| LightGBM tuned | 0.7187 | 0.0590 | 0.792 | 0.1726 | 1,576 | 519 |

The spread across folds is large (fold 1 trains on about eleven months), so
differences between logistic regression and LightGBM must be read with the
paired view below.

**A conflict worth stating.** On the primary metric, tuned LightGBM beats
logistic regression in three of four folds (paired net benefit differences
+295, +255, +334, -36; mean +212, sd 169). Yet on the final validation segment
(fit on 0-60%, evaluated on 70-80%), logistic regression comes out slightly
ahead (1,775.5 vs 1,757.0; 4.8). The model choice rule (4.8) was fixed on the
validation segment before that result was seen, so it is kept and the decision is
**not** changed after the fact. The rolling-fold evidence says LightGBM may be
the better model. To settle this without selecting on the test, LightGBM is also
frozen now (`artifacts/model_secondary.joblib`), and Phase 5 reports it beside
the chosen model as a labelled secondary comparison. A decision to switch would
be a go or no-go item in Phase 5, not a silent change.

## 4.5 What do Set B and `deposit_type` add? (D5, validation version)

Same tuned LightGBM, rolling folds. Gap = PR-AUC minus PR-AUC of Set A.

| Feature set | PR-AUC mean | sd | Brier | Gap vs A |
|---|---|---|---|---|
| A1 (reservation request only) | 0.6921 | 0.0743 | 0.1825 | -0.0266 |
| **A (A1 + history counters)** | 0.7187 | 0.0590 | 0.1726 | 0 |
| A + `deposit_type` | 0.7480 | 0.0657 | 0.1644 | +0.0293 |
| A + B | 0.8445 | 0.0545 | 0.1398 | **+0.1258** |
| A + B + `deposit_type` | 0.8518 | 0.0464 | 0.1403 | +0.1331 |

The D5 rule (gap above 0.05 means the extra fields are post-booking) applies:
Set B adds 0.126 PR-AUC, an increase of 17%. This confirms how much of an
easy-looking model would come from information that does not exist at booking
time. The history counters (A2) add 0.027, so the least certain part of Set A
also matters, and the result for Set A carries that caveat.
`deposit_type` alone adds 0.029, which is below 0.05, so by D5 alone it would
pass, but it stays out of the model on the Phase 2 evidence (99.25%).

## 4.6 Robustness (fit to validation, tuned LightGBM, validation PR-AUC)

| Check | Fit rows | Validation PR-AUC |
|---|---|---|
| Main (duplicates kept) | 71,465 | 0.7381 |
| Full-row duplicates removed from fit | 49,673 | 0.6978 |
| Cancellation-only target (no-shows count as arrivals) | 71,465 | 0.7393 |

Removing duplicates lowers PR-AUC by 0.040. Repeated booking profiles in the
training segment help predict repeated profiles later (group bookings recur),
so part of the performance relies on repeats. Phase 5 therefore reports the test
on all rows and on rows with no profile twin. The target definition barely
matters (+0.001).

## 4.7 Guest segments (optional, tried)

k-means on the prepared Set A matrix, fit on the fit segment. Silhouette by k:
2: 0.202, **3: 0.290**, 4: 0.171, 5: 0.106, 6: 0.116, 7: 0.124. With k=3:

| Cluster | Fit rows | Fit non-arrival | Validation non-arrival | Median lead time (fit) |
|---|---|---|---|---|
| 0 | 5,044 | 31.0% | 37.7% | 49 |
| 1 | 58,223 | 39.6% | 38.6% | 82 |
| 2 | 8,198 | 17.0% | 25.1% | 4 |

Cluster 2 is last-minute bookings and is largely a restatement of lead time.
Adding the cluster as a feature gave PR-AUC 0.7168 against 0.7187 without it, so
**the clusters are not used in the model**. They are reported as exploratory
(`artifacts/figures/04_clusters.png`). The weakness of the segment signal
(silhouette 0.29 at best) means no strong guest types were found.

## 4.8 Final candidates, calibration, threshold

Two candidates were fit on the fit segment: logistic regression and tuned
LightGBM. For each, the calibration method (none, sigmoid, isotonic) was chosen
by Brier score on the validation segment and the threshold by net benefit on the
validation segment under the capped policy. The choice rule was fixed before the
logistic regression result was seen: use LightGBM only if its validation net
benefit is more than 5% higher.

| | Logistic regression | LightGBM (tuned) |
|---|---|---|
| Calibration chosen | none | none |
| Brier: none / sigmoid / isotonic | 0.1699 / 0.1739 / 0.1766 | 0.1643 / 0.1652 / 0.1687 |
| Base-rate Brier | 0.2313 | 0.2313 |
| Threshold | 0.55 | 0.50 |
| Validation PR-AUC | 0.7309 | 0.7381 |
| Calibration slope | 0.957 | 1.112 |
| Flagged share / precision / recall | 20.2% / 0.768 / 0.427 | 20.0% / 0.767 / 0.423 |
| Net benefit per 1,000 bookings | **1,775.5** | 1,757.0 |
| Weeks where the 25% cap binds | 27.3% | 18.2% |

Isotonic calibration made the Brier score worse for both models, probably
because the calibration segment (34.1% non-arrival) is small and differs from
the validation segment (36.3%). Neither model needed calibration.

**Chosen model: logistic regression**, ratio of net benefits 0.990, no
calibration, threshold 0.55 (the uncapped closed-form value from Phase 1 is
0.533). The lead-time rule from Phase 1 (flag `lead_time >= 226`) flags 6.1% of
bookings with precision 0.692, recall 0.116 and net benefit 361.6 per 1,000.

## 4.9 Gate on the validation segment (Amendment A2)

| Criterion | Result |
|---|---|
| D1 LightGBM minus logistic regression PR-AUC above the logistic regression fold sd | **FAIL**: +0.0091 against sd 0.0560 |
| D2 recall at the threshold at least 0.30 | PASS (0.427) |
| D3 precision of flagged at least 0.60 | PASS (0.768) |
| D4a calibration slope in [0.8, 1.2] | PASS (0.957) |
| D4b Brier below the base-rate predictor | PASS (0.1699 vs 0.2313) |

**D1 fails as written, and it stays a failure.** As supplementary information
only, the paired PR-AUC differences are +0.0026, +0.0019, +0.0215 and +0.0105
(mean +0.0091, sd 0.0091): LightGBM is ahead in every fold, by a small amount.
That does not change the verdict on D1, which asked for a clear gain. Its
message, that the boosted model is not clearly worth its complexity on PR-AUC,
is one reason logistic regression is the final model, together with the
validation net-benefit rule. D2 to D4 show that
Set A is informative enough to act on, so the project continues, as a
retrospective study (as Phase 1 stated).

## 4.10 What is frozen before the test

`deployment/model.joblib` (SHA-256 prefix `8c6cd3c57dfe46d8`) and
`artifacts/frozen_policy.json`: logistic regression, Set A, no calibration,
threshold 0.55, weekly cap 25%, rule threshold `lead_time >= 226`, seed 42,
`test_opened: false`.

## Caveats carried to Phase 5

0. LightGBM beats logistic regression on mean fold net benefit (+212 per 1,000), so the model choice is not settled; Phase 5 reports both.

1. The cap binds in 27% of validation weeks, far above the 10% B3 allows. B3 is
   expected to fail, and the economics say more capacity would pay.
2. Set A includes the history counters (A2), which add 0.027 PR-AUC and are the
   least certain fields.
3. Repeated profiles help (the duplicate check), so the twin-free result matters.
