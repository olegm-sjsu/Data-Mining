# Phase 3: Data Preparation

Draft v2 (after critic round 1). Numbers from the notebook (section 3),
`artifacts/cleaning_report.csv`, `artifacts/split_summary.csv`,
`artifacts/final_dataset_schema.csv`, `artifacts/metrics_phase3.json`.

## Test-period exposure log (honesty note)

Before the critic's first review, the analyst's exploration scripts and the first
version of this notebook looked at outcomes of the whole file, including the
test period. What was seen: the test-period non-arrival rate (about 41%), the
rate by hotel in the test period, rates by quarter including the test quarters,
and the share and rate of repeated booking profiles in the test period. **No
modeling choice was made from these numbers** (no model existed). The split
amendment (A1) came from the file's structure (arrival-window sampling), not
from outcomes. After the critic's point, the notebook was changed so that the
test period is fixed by date first, and all target-based analysis uses
development rows only. Phase 1's headline rates (37.04%, 41.73%, 27.76%) are
whole-file numbers and include test rows; they describe the dataset and are
not used for any choice. The test results of record are those of Phase 5.

## 3.1 Select data

| Group | Decision | Reason |
|---|---|---|
| `reservation_status`, `reservation_status_date` | Exclude from features | Encode the outcome. The date is used only to reason about when outcomes become known |
| Set A, 20 columns | Include | Definition-based classification (Phase 2); A1 and A2 tested in Phase 4 |
| Set B, 7 columns plus `deposit_type` | Diagnostic only | Not shown to be known at booking |
| `arrival_date_year` | Input only | Orders rows and gives the weekday; as a feature it would be an unseen test year |
| Rows | All, except invalid records | See cleaning |

## 3.2 Clean data

Rules are applied in order, on the whole table, so counts are of rows still
remaining and use no outcome.

| Rule | Rows | Action |
|---|---|---|
| Zero guests | 180 | drop |
| `adr` < 0 | 1 | drop |
| `adr` > 1000 | 1 | drop |
| `children` missing | 4 | impute 0 in the pipeline |
| `meal` Undefined | 1,169 | merge into SC |
| Segment or channel Undefined | 5 | own category |
| Zero nights | 645 | keep, flag |
| Full-row duplicates | 31,980 | keep (sensitivity in Phase 4) |

119,390 rows in, **119,208 out (182 removed, 0.15%)**.

## 3.3 Construct data and 3.4 Integrate data

Built in `deployment/hotel_features.py` (shared with the app): `total_nights`,
`zero_nights`, `total_guests`, `has_children`, `log_lead_time`,
`arrival_weekday`, `has_agent`, `has_company`, `agent_id` (rare agents grouped),
`room_changed` (Set B only). Seasonality enters through arrival month and week.
Only one table exists, so **no integration was needed**.

## 3.5 Time-based design (amendments A1 and A2 to Phase 1)

Phase 2 showed that ordering by derived booking date is unsuitable (arrival-window
sampling truncates late booking dates, and a training outcome can be decided after
the cut), so rows are ordered by **arrival date** (A1).

| Segment | Rows | Arrival dates | Non-arrival rate |
|---|---|---|---|
| Fit | 71,465 | 2015-07-01 to 2016-11-06 | 36.4% |
| Calibration | 11,854 | 2016-11-07 to 2017-02-09 | 34.1% |
| Threshold validation | 11,871 | 2017-02-10 to 2017-04-21 | 36.3% |
| **Test (sealed until Phase 5)** | 24,018 | 2017-04-22 to 2017-08-31 | sealed |

Model comparison uses four expanding-window folds inside the first three
segments, cut at 40%, 50%, 60% and 70% of rows. The test period holds only April
to August arrivals, so it is a spring and summer test, not a full year.

**Twins and duplicates.** The split is by date, row by row, so no test row enters
training. Two keys, both **without the target** so that no test label defines
the subset (Phase 1 had said Set A plus target; this is a deliberate change):
(a) every Set A field: **0** test rows have an exact twin earlier, which is
guaranteed because the key contains the arrival date, so this count is not evidence about duplicates; (b) a *profile* key
(Set A without the arrival date fields): **4,887 test rows (20.3%)** match an
earlier row. Phase 5 reports results on all test rows and on the 79.7% with no
profile twin. Duplicates *inside* the training segments are kept in the main
run, and Phase 4 reruns the models with them removed.

## 3.6 Format data: one pipeline

`build_pipeline` chains feature construction, median imputation, clipping at the
training 99.9th percentile, scaling, one-hot encoding (categories with fewer
than 200 training rows grouped), and the classifier, and is fit on training
rows only. Check by hand: after clipping, the fit-only mean lead time is 102.76,
a manual recomputation gives 102.76, and the whole-file mean is 104.11. The
design matrix has 109 columns.

**Outlier decision.** Tail values (lead time up to 737; 55 adults; 26 previous
cancellations) are real events, so rows stay and values are clipped at the
fit-segment 99.9th percentile (lead time at 482, week nights at 15, weekend
nights at 6, adults at 3, previous bookings at 19.5). Only 51 to 72 fit rows per
column are touched.

**Leakage tests (all pass):** the test period starts after every training
segment; segments are ordered; no outcome or status column is in the feature
lists; `deposit_type` is not a Set A feature; each fold trains strictly before it
validates; the twin keys contain no target.

## 3.7 Output: final dataset and description

`data/hotel_prepared.csv.gz` (not committed): 119,208 rows, 32 columns, target
rate on development rows 36.09%. Set A is 20 raw columns plus the year input (21 raw inputs for the app), giving 27 model
features. Schema (a `segment` column holds fit, cal, val or test):

| Column | Role |
|---|---|
| `hotel` | Set A |
| `lead_time` | Set A |
| `arrival_date_year` | input only (date order, weekday) |
| `arrival_date_month` | Set A |
| `arrival_date_week_number` | Set A |
| `arrival_date_day_of_month` | Set A |
| `stays_in_weekend_nights` | Set A |
| `stays_in_week_nights` | Set A |
| `adults` | Set A |
| `children` | Set A |
| `babies` | Set A |
| `meal` | Set A |
| `market_segment` | Set A |
| `distribution_channel` | Set A |
| `is_repeated_guest` | Set A |
| `previous_cancellations` | Set A |
| `previous_bookings_not_canceled` | Set A |
| `reserved_room_type` | Set A |
| `customer_type` | Set A |
| `agent` | Set A |
| `company` | Set A |
| `country` | Set B |
| `booking_changes` | Set B |
| `assigned_room_type` | Set B |
| `days_in_waiting_list` | Set B |
| `adr` | Set B |
| `required_car_parking_spaces` | Set B |
| `total_of_special_requests` | Set B |
| `deposit_type` | Set B (flagged) |
| `is_canceled` | target |
| `arrival_date` | bookkeeping |
| `segment` | bookkeeping |

## Limits and open items

1. The calibration segment (34.1% non-arrival) differs from the validation
   segment (36.3%), and the test period is a spring and summer window. Calibration
   drift is examined in Phase 5.
2. Duplicate-removal and Set A1/A2/B comparisons run in Phase 4.
