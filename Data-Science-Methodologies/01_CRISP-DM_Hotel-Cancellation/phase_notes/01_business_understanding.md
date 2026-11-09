# Phase 1: Business Understanding

Methodology: CRISP-DM 1.0 (Chapman et al., 2000). **Draft v4** (revised after
critic rounds 1 to 3, see `critic_reviews/01_business_understanding_round1.md`).

Labels used throughout:

- **[D]** computed from `data/hotel_bookings.csv` (119,390 rows) and reproducible
- **[P]** stated in the source paper (Antonio, de Almeida and Nunes, *Data in
  Brief* 22, 2019, DOI 10.1016/j.dib.2018.11.126)
- **[A]** an assumption made by the author, not a fact

## 1.1 Determine business objectives

### Background

Two hotels in Portugal recorded bookings due to arrive between 1 July 2015 and
31 August 2017 **[P]**: **H1, a resort hotel in the Algarve, and H2, a city
hotel in Lisbon** **[P]**. In the CSV they appear as `Resort Hotel` and
`City Hotel`. Cancellation is a revenue-management problem: a room held for a
booking that later cancels is either resold late at a lower price or left empty.

| | Bookings | Non-arrival rate (`is_canceled = 1`) |
|---|---|---|
| All | 119,390 | 37.04% **[D]** |
| City Hotel (H2) | 79,330 | 41.73% **[D]** |
| Resort Hotel (H1) | 40,060 | 27.76% **[D]** |

About one booking in three does not turn into a stay, so planning occupancy on
bookings alone overstates demand.

### The target, defined exactly

`is_canceled = 1` covers **both cancellations and no-shows**. In the file,
`reservation_status` is `Canceled` for 43,017 rows, `No-Show` for 1,207 rows,
and `Check-Out` for 75,166 rows, and every `No-Show` row has `is_canceled = 1`
**[D]**. We therefore call the target **non-arrival**: the booked room-nights
were not used by that guest. This suits the business problem because in both cases the room is left unused
by that guest and must be resold, if it can be resold at all (the loss
parameter u below captures partial recovery). The 1,207 no-shows (about 1% of rows) are carried as a
labelled sub-group and a sensitivity check is run in Phase 4 with a
cancellation-only target.

### Business objectives

1. **Primary:** reduce revenue lost to non-arrival by flagging the bookings most
   likely not to turn into stays, and attaching one costed action to those
   bookings.
2. **Secondary:** give revenue managers a calibrated probability per booking so
   that weekly expected non-arrivals can be tracked.

The hotel's decision is not "predict cancellations". It is "which bookings do
we spend effort on". The model is judged on decision cost, not accuracy.

### The one action the model triggers **[A]**

Personal outreach by reservations staff to the guest: reconfirm the stay, and
offer either a free date change or a small discount in exchange for moving to a
non-refundable rate. One action keeps the policy and its cost model coherent. A
cheaper tier (such as an automated reminder) is listed as future work, because
it would need its own cost, effect, and threshold.

### Business success criteria (measurable, fixed before modeling)

| ID | Criterion | Pass condition **[A]** |
|---|---|---|
| B1 | Net benefit of the policy per 1,000 bookings on the untouched test period, versus "contact nobody" | The 10th percentile of net benefit over the Monte Carlo draws below is greater than 0. Draws: 10,000, independent, save rate s uniform on [0.10, 0.40], outreach cost c uniform on [10, 40] euros, loss per unrecovered non-arrival L uniform on [100, 200] euros. Threshold fixed from the validation period, not re-tuned per draw |
| B2 | Expected cost versus a simple rule, **defined now**: flag a booking if `lead_time >= tau`, with `tau` chosen on the validation period to minimize expected cost at the base assumptions under the same weekly 25% rank cap as the model (ranking by `lead_time` descending, ties broken by earlier arrival date, then by row order) | Model expected cost on the test period at least 15% lower than the rule's |
| B3 | Contact workload, evaluated on the test period only | The threshold is chosen on validation data as the value that minimizes expected cost of the complete capped policy (threshold plus weekly rank cap), then frozen. In any test arrival week where it would flag more than 25% of that week's bookings, only the top 25% by predicted probability are flagged (a rank cap; probability ties at the boundary are broken by earlier arrival date, then row order). An "arrival week" is the ISO year and ISO week of `arrival_date`. The pass condition is that the cap binds in no more than 10% of test weeks, and B1 and B2 are computed with the cap applied |
| B4 | Planning use | Median over test arrival weeks of abs(sum of predicted probabilities minus actual non-arrivals) divided by actual non-arrivals is at most 0.10. Weeks with zero actual non-arrivals are excluded and their count is reported |

## 1.2 Assess situation

### The prediction contract (the most important assumption in the project)

The source paper states that the **data point time for every observation is the
day before arrival**, with values drawn from the hotels' property management
systems **[P]**. A cancelled booking therefore carries its values as of
roughly its last change, and a booking that stayed carries values as of the day
before arrival. Consequences:

1. The table is **not** a record of what the hotel knew when the booking was
   made. Some fields are written or revised after booking and after the
   outcome is decided. Evidence in the file: `assigned_room_type` differs from
   `reserved_room_type` for 18.8% of non-cancelled bookings but only 1.8% of
   cancelled ones, and mean `booking_changes` is 0.29 versus 0.10 **[D]**.
   A booking that is cancelled cannot get a room assigned or be changed later.
2. Computing a booking date as arrival date minus `lead_time` gives a usable
   **ordering of bookings in time** for splitting the data. It does not
   reconstruct what each field said on that date. Booking dates computed this
   way run from 24 June 2013 to 31 August 2017, and 2,637 of them fall before
   2015 **[D]**. The project refers to arrival periods when describing the
   data's time span.
3. We cannot obtain true booking-time snapshots. The project is therefore a
   **retrospective benchmark with a declared decision moment**, not a claim
   that the model would be deployable at booking creation as built.

**Decision moment for this project [A]:** the model scores a booking at the
time it is made, using only **Feature Set A**, fields that are fixed when the
booking is created. Fields that the data cannot show were known then go into
**Feature Set B**, used only to measure how much apparent accuracy comes from
post-booking information (an upper bound, never deployed).

Provisional field audit (to be confirmed in Phase 2; each decision there needs
a written reason):

| Field | Provisional set | Reason |
|---|---|---|
| `hotel`, `lead_time`, arrival year, month, week, day, `stays_in_weekend_nights`, `stays_in_week_nights`, `adults`, `children`, `babies`, `meal`, `market_segment`, `distribution_channel`, `is_repeated_guest`, `previous_cancellations`, `previous_bookings_not_canceled`, `reserved_room_type`, `customer_type`, `agent`, `company` | A | Normally stated or fixed when the booking is made. Needs Phase 2 confirmation, in particular `children` and the history counters |
| `deposit_type` | **A, flagged** | Said to be defined from payment data (critic's reading of the paper, **unconfirmed**). In the file, 99.4% of `Non Refund` bookings are `is_canceled = 1` **[D]**, which is implausible for a real policy and looks like a recording artifact. Phase 2 must decide |
| `country` | B (also a fairness issue) | The paper warns that nationality can be unknown until check-in (critic's reading of the paper, **unconfirmed**) |
| `booking_changes` | B | Said to count changes up to check-in or cancellation (critic's reading, **unconfirmed**) |
| `assigned_room_type` | B | Assigned late, and the group differences above |
| `days_in_waiting_list` | B | Measured over the booking's life |
| `adr`, `required_car_parking_spaces`, `total_of_special_requests` | B | Can change after booking, and parking is only recorded for arriving guests |
| `reservation_status`, `reservation_status_date` | **Excluded** | Direct encoding of the outcome |

**Decision gate after Phase 2:** if the audit leaves Set A too weak to support a
useful model, or too many fields cannot be classified, the project continues as
an explicitly retrospective study (no deployment claim) and the report says so.
That outcome is acceptable and is reported honestly.

### Inventory of resources

- **Data:** one table, 119,390 rows, 32 columns, no booking ID **[D]**.
  Public, CC BY 4.0.
- **People:** one analyst (the student). The hotel's revenue manager is
  simulated, so all business inputs are assumptions.
- **Compute:** a laptop and Google Colab (free tier).
- **Software:** Python 3.12, pandas, scikit-learn (including gradient
  boosting), SHAP, Streamlit.
- **Knowledge:** the source paper and the CRISP-DM guide.

### Requirements

- Every feature in the final model must pass the field audit above.
- Results are reproducible: fixed seeds, pinned packages, one notebook that
  runs top to bottom.
- The deployed tool shows inputs, a probability, and the recommended action.
- **Validation protocol (fixed now):**
  1. Order bookings by **arrival date** (amended in Phase 3, see Amendment A1 at the end).
  2. The last 20% of bookings by that date is the **test period**, opened once.
  3. Inside the earlier 80%, use rolling-origin cross-validation by arrival
     date for model comparison.
  4. Probability calibration is fitted on the latest validation fold only.
  5. The decision threshold and the B2 comparison rule are chosen on
     validation data only.
  6. The cost sensitivity grid is run on predictions with the threshold
     already fixed.
  7. Duplicate rows: the split is always by date, one row at a time, so no
     row from the test period ever enters training. Because there is no
     booking ID, rows that match on every Set A field and the target may be
     true group bookings or repeated records. Phase 2 counts how many test
     rows have an identical twin in the training period. The test results are
     reported twice: on all test rows, and on test rows with no identical twin
     in training. Phase 2 also decides whether duplicates inside the
     training period are dropped.

### Assumptions

1. **[A]** Each row is one booking.
2. **[A]** The 2015 to 2017 arrival behavior is a reasonable proxy for the near
   future. A time-based split tests this partly.
3. **[A]** Outreach keeps 25% of bookings that would otherwise not arrive (the
   save rate s). This is the most uncertain input, so B1 samples it.
4. **[A]** Outreach has no effect on bookings that would have arrived, apart
   from its cost.
5. **[A]** `adr` is price per night. The currency is not stated in the data;
   we write **euros (€)** because both hotels are in Portugal.

### Constraints

- No real hotel is available to run an experiment, so the project can estimate
  the value of a policy but cannot measure a causal effect of outreach.
- Fields that the data cannot show were known at decision time are held out of
  the deployed model (see the contract above).
- Privacy and fairness: `country` is a proxy for nationality. It is in Set B
  and stays out of the deployed model unless Phase 5 shows it is both
  available at decision time and fair to use.
- Course time frame: six phases, a deployed app, an article, and a video.

### Risks and contingencies

| Risk | Likelihood | Impact | Contingency |
|---|---|---|---|
| Table is a near-arrival snapshot, so post-booking fields leak the outcome | **Certain (documented)** | High | Feature Sets A and B; B used only as an upper bound; decision gate after Phase 2; report states the retrospective limit |
| `deposit_type` artifact (99.4% of `Non Refund` do not arrive) | High | High | Phase 2 investigation; run models with and without it; exclude if unexplained |
| Random splits flatter the model | High | High | Time-based test period; a random split is shown only as a contrast |
| Duplicate rows straddling train and test | Medium | Medium | Grouping rule in the validation protocol; results with and without duplicates |
| Concept drift between 2015, 2016, and 2017 | Medium | Medium | Quarterly test performance (D6); drift trigger in Deployment |
| Cost inputs are wrong | High | Medium | B1 Monte Carlo and a Phase 5 sensitivity grid |
| Fairness harm from `country` | Medium | High | `country` kept out of the deployed model; parity check on the audit model |
| Test period reused for tuning | Medium | High | Protocol above; test opened once, and a log of test accesses kept in the notebook |
| Schedule slips | Medium | Medium | Dependencies and gates in the project plan |

### Terminology

- **Non-arrival:** `is_canceled = 1`, either a cancellation or a no-show.
- **ADR:** average daily rate, revenue per occupied room night.
- **Lead time:** days between booking creation and arrival.
- **Data point time:** the moment the field values were taken; here the day
  before arrival **[P]**.
- **Feature Set A / B:** fields assumed known at booking / fields not shown to
  be known at booking.
- **Save rate:** share of would-be non-arrivals retained by outreach.
- **PR-AUC:** area under the precision-recall curve.
- **Calibration:** whether a predicted 30% matches an observed rate near 30%.

### Costs and benefits

**Operating economics of one outreach.** Unit assumptions **[A]**: median `adr`
is 99.9 for the city hotel and 75.0 for the resort hotel **[D]** (currency
units, taken as euros). The rest are chosen for a worked example and can be
replaced with the hotel's own figures.

| Quantity | Symbol | Value |
|---|---|---|
| Revenue of a typical booking (3 nights at about €100) | V | €300 |
| Share of an unused booking's revenue that cannot be recovered by resale | u | 50% |
| Revenue lost per unrecovered non-arrival | L = u x V | €150 |
| Cost of one personal outreach, staff time plus average offer | c | €20 |
| Save rate | s | 25% |

For a booking with non-arrival probability p, the expected net benefit of
outreach is `p x s x L - c`. Outreach pays when `p > c / (s x L) = 20 / 37.5 =
0.533`. If s were 50%, the threshold would be 0.267. Phase 5 maps this
sensitivity, and B1 samples it.

**Cost matrix, relative to "do nothing" [A]:**

| | Booking does not arrive | Booking arrives |
|---|---|---|
| **We act** | True positive: cost c - s x L = 20 - 37.5 = **-€17.50** (a net gain) | False positive: cost **€20** (wasted outreach) |
| **We do not act** | False negative: **€0** relative to do nothing (the forgone gain is €37.50) | True negative: **€0** |

**Project costs [A]** (CRISP-DM requires these beside the benefit):

| Item | Estimate |
|---|---|
| Development: about 45 analyst hours at €40 per hour | €1,800 |
| Cloud and tooling: free tier | €0 |
| Operating: monitoring and retraining, about 3 hours per month | about €120 per month |
| Staff capacity for outreach | Bounded by B3 (25% of weekly bookings) |

The break-even point is reported in Phase 5: how many bookings per year the
hotel needs before net benefit covers development plus operating cost.

**Out of scope for v1: overbooking.** Selling extra rooms against predicted
non-arrivals needs capacity, the existing booking stock, the correlation of
cancellations, and a marginal rule. With the simplified assumption that a sold
extra room earns €150 if the guest does not arrive and costs €400 if a
confirmed guest is walked, the break-even probability that the guest arrives is
150 / (150 + 400) = 27.3%. A full treatment would use the distribution of
arrivals, not the sum of probabilities. This is listed as a next step only.

## 1.3 Determine data mining goals

### Technical goals

1. Build a binary classifier of non-arrival using Feature Set A.
2. Produce a **calibrated** probability for each booking.
3. Choose a decision threshold by expected cost, not by accuracy or F1.
4. Quantify how much performance Feature Set B adds on top of Set A (A+B
   versus A, as defined in D5), as evidence of leakage.
5. Explain the main associations with SHAP, read as association only.
6. **Optional:** cluster bookings into segments, used only if they give
   revenue managers a distinct, actionable group definition. If not, report
   them as exploratory and drop them.

### Data mining success criteria (fixed before modeling) [A]

| ID | Criterion | Pass condition |
|---|---|---|
| D1 | Model quality | Gradient boosting PR-AUC exceeds logistic regression PR-AUC on the test period by more than the standard deviation of logistic regression PR-AUC across the rolling-origin folds |
| D2 | Recall at the chosen threshold | At least 0.30 on the test period, with a 95% bootstrap interval reported |
| D3 | Precision of flagged bookings | At least 0.60 on the test period |
| D4 | Calibration | Calibration slope between 0.8 and 1.2 and a Brier score below that of the constant base-rate predictor, on the test period |
| D5 | Leakage control | The final model uses Set A only. The gap is defined as PR-AUC of the model trained on Set A plus Set B minus PR-AUC of the same model type trained on Set A alone, on the same validation folds and the test period. If the gap exceeds 0.05, the Set B fields stay classed as post-booking and the project states that the accuracy of any A+B model is not achievable at booking time |
| D6 | Temporal stability | PR-AUC in each test quarter within 0.05 of the overall test PR-AUC |

The primary decision metric is expected cost at the chosen threshold. PR-AUC and
calibration support it. Accuracy is reported only for completeness, because the
base rate is 37%.

## 1.4 Produce project plan

| Phase | Main work | Inputs | Output | Effort | Depends on | Review gate |
|---|---|---|---|---|---|---|
| 1. Business Understanding | This document | Brief, paper, data summary | `01_business_understanding.md` | 3 h | none | Two critic rounds, score 9 or more |
| 2. Data Understanding | Collect, describe, explore, quality audit, **field-by-field availability audit** | Raw CSV, paper | `02_data_understanding.md` | 7 h | Phase 1 | **Decision gate:** is Set A strong enough? Resolve `deposit_type` |
| 3. Data Preparation | Cleaning rules, features, time split, pipeline | Audit table | `03_data_preparation.md`, final dataset | 7 h | Phase 2 gate | Leakage unit test passes |
| 4. Modeling | Baselines, LR, tree, boosted model, tuning, optional clusters | Prepared data | `04_modeling.md`, model files | 9 h | Phase 3 | Rolling-CV table reviewed |
| 5. Evaluation | Expected cost, sensitivity, calibration, slices, SHAP, fairness, break-even, go or no-go | Test period (opened once) | `05_evaluation.md` | 7 h | Phase 4 | Go or no-go |
| 6. Deployment | Streamlit app, monitoring plan, `REPORT.md`, project review | Final model | `06_deployment.md`, app | 10 h | Phase 5 go | Final critic round |

Total about 43 hours.

**Known iteration loops:**

- Phase 2 may change this document's prediction contract (the gate above).
- Phase 4 results may send us back to Phase 3 for features.
- Phase 5 may send us back to Phase 1 if cost assumptions look wrong.

Each loop is logged in the phase note where it happens.

**Risks tied to the schedule:** the Phase 2 gate is the biggest schedule
risk. If Set A is weak, Phases 3 to 5 are re-planned as a retrospective study,
with about 4 hours of extra analysis.

### Initial assessment of tools and techniques

| Need | Tool | Why |
|---|---|---|
| Wrangling | pandas | Standard |
| Pipeline | scikit-learn `Pipeline` and `ColumnTransformer` | Fit on train only, serializable |
| Models | `DummyClassifier`, `LogisticRegression`, `DecisionTreeClassifier`, `HistGradientBoostingClassifier` | Spans interpretable to strong |
| Validation | Rolling-origin CV by derived booking date | Matches the protocol |
| Calibration | Isotonic or sigmoid fitted on the latest validation fold | Probability use |
| Explanation | SHAP | Association analysis |
| Segments (optional) | k-means or Gaussian mixture | Guest types, only if actionable |
| Deployment | Streamlit | Public URL, quick |

## References

- Chapman, P. et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide.*
- Antonio, N., de Almeida, A., Nunes, L. (2019). Hotel booking demand
  datasets. *Data in Brief*, 22, 41-49. DOI 10.1016/j.dib.2018.11.126.
  Details marked **[P]** (H1 and H2 identities, data point time) were
  confirmed through a search snippet of the paper's text; the full text should
  be read in Phase 2 for the field definitions.

## Amendment A1 (made in Phase 3, after the Phase 2 findings)

CRISP-DM is iterative, and this is a logged loop from Data Understanding back
to this document.

- **Change:** time ordering and splitting use **arrival date**, not derived
  booking date.
- **Why:** the file only contains bookings due to arrive between 2015-07-01
  and 2017-08-31. Ordering by booking date makes the latest booking dates miss
  every long-lead booking (right-censoring: the test period would be biased
  to short lead times). It also lets a training booking made before the cut
  have its outcome decided after the cut, which is label look-ahead. Ordering
  by arrival date means a model trained on stays that arrived before the cut
  uses only outcomes known by then, and it is tested on all later arrivals.
- **Effect on criteria:** none of B1 to B4 or D1 to D6 changes. "Arrival
  week" was already arrival-based. The boosted model is LightGBM (instead of
  scikit-learn's histogram boosting) so that SHAP tree explanations are exact.
- **Not changed:** Feature Set A and B, the decision moment, costs, the
  capped policy.

## Amendment A2 (made in Phase 3, after critic round 1 of Phases 2 and 3)

- **Decision gate:** the gate "is Set A strong enough?" cannot be decided from
  Phase 2 alone, since it needs a model. It is decided in Phase 4 using the
  **validation-period** versions of D1 to D4 (rolling folds and the threshold
  validation segment), never the test period. The test is opened once, in
  Phase 5, where D1 to D6 are judged. Phase 2 records only the classification.
- **Sealed test:** the test period is fixed by date (arrival from 2017-04-22)
  before any outcome is examined, and all target-based analysis in Phases 2 to 4
  uses development rows only. Phase 1's whole-file rates are descriptive.
- **Twin keys:** the target is dropped from the twin keys of validation protocol
  step 7, so the "no twin" subset is defined without test labels.
- **Set A tiers:** A1 (fixed by the reservation request) and A2 (history
  counters) are compared in Phase 4 as a robustness check on availability.
