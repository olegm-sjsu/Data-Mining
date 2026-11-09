# Which hotel bookings will not arrive? A CRISP-DM project

Final report. Data: *Hotel booking demand* (Antonio, de Almeida and Nunes,
*Data in Brief*, 2019; Kaggle, CC BY 4.0). Notebook:
`notebooks/crisp_dm_hotel_bookings.ipynb`. Phase notes: `phase_notes/`. Critic
reviews: `critic_reviews/`.

## 1. Summary

About 37% of bookings at two Portuguese hotels (2015 to 2017) do not turn into a
stay. We built a model that flags bookings likely not to arrive, so the hotel can
contact those guests, and judged it by money (a cost model), not accuracy.

- **What works:** on 24,018 later bookings, flagged bookings were truly
  non-arrivals 72% of the time (base rate 41%), catching 39% of non-arrivals.
  Under the assumed costs the policy earned about €1,533 per 1,000 bookings
  against about -€160 for a lead-time rule.
- **What does not:** profit depends on two unmeasured numbers (the share of
  flagged guests outreach keeps, and its cost). Below a 19% save rate it loses
  money. The model also weakened sharply in July and August 2017 (net benefit
  near zero or negative), and it contacts German guests wrongly 3.1 times as often as
  Portuguese guests.
- **Decision:** no-go for automation; conditional go for a randomized pilot.
- **Biggest lesson:** the raw table contains information recorded *after*
  booking (for example, a parking space is recorded for 0.0% of non-arrivals).
  Using it adds 0.10 to 0.13 PR-AUC and would have made an unusable model look excellent.

## 2. Method and results by CRISP-DM phase

**Business understanding.** Objective: flag bookings for outreach. One costed
action (€20 per outreach, assumed 25% save rate, €150 lost per unrecovered
non-arrival) gives a break-even probability of 0.53. Criteria B1-B4 and D1-D6
were fixed before any modeling. The data is a day-before-arrival snapshot, so the
project is a retrospective benchmark with a declared decision moment, and
fields are split into Set A (fixed by the reservation) and Set B (not shown to
be known at booking). Four review rounds (score 4.0 to 9.2) shaped this phase.

**Data understanding.** 119,390 bookings, 32 columns. Strong signals:
non-arrival rises from 9.5% for lead times of a week or less to 80.6% above a
year; Groups 60% against Direct 15%; city 41% against resort 26% (development
rows). Quality: 26,936 development rows are full-row duplicates; 94% of `company` and 14% of `agent` are missing;
157 zero-guest rows (180 in the whole file). `deposit_type = Non Refund` is 99.25% non-arrival, an
anomaly we treat as a likely recording effect and keep out of the model.
All target-based analysis used development rows only; the test was fixed by date first.

**Data preparation.** 182 invalid rows removed (0.15%); features built in a
shared module; one scikit-learn pipeline fit on training rows only. Rows are
split by **arrival date** (a protocol amendment, because the file samples an
arrival window): fit to 2016-11-06, calibration to 2017-02-09, threshold
selection to 2017-04-21, and a test period from 2017-04-22.

**Modeling.** Five models on four expanding-window folds (PR-AUC, net benefit).
Logistic regression 0.710 and tuned LightGBM 0.719 PR-AUC; tree 0.639; majority
0.362. Set B added 0.126 PR-AUC (leakage evidence). Clusters were not useful.
The pre-registered rule chose logistic regression. The criterion D1 (clear gain for the boosted model) failed.

**Evaluation (test opened once).**

| | Logistic regression (champion) | LightGBM (challenger) | Lead-time rule |
|---|---|---|---|
| Precision / recall of flagged | 0.718 / 0.388 | 0.773 / 0.424 | 0.508 / 0.211 |
| Net benefit per 1,000 | 1,533 (CI 812 to 2,186) | 2,023 (CI 1,512 to 2,529) | -160 |
| PR-AUC | 0.713 | 0.749 | 0.487 |

Criteria: pass D2, D3, D4, D5, B4; fail B1 (robustness to cost assumptions), B3
(cap binds in 65% of weeks), D1, D6 (summer drop); B2 not interpretable. Full
detail in `phase_notes/05_evaluation.md`.

**Deployment.** A Streamlit app (single booking and CSV tabs), tested headless
and against the notebook (identical probabilities and flags on all 24,018 test
rows), pilot mode (Groups excluded, 50/50 hold-out arm), a batch cap warning, Docker
packaging, and a monitoring module with drift, aggregate-error, precision and
country false-positive checks. A slide deck is in `article/presentation.html`. The public URL still needs a GitHub push.

## 3. Limits you should know

1. **Retrospective.** Availability of Set A fields at booking time is assumed by
   definition. The history counters add 0.027 PR-AUC and are the least certain.
2. **Unmeasured economics.** The save rate and outreach cost are assumptions;
   the model's value is unproven until a pilot measures them.
3. **Seasonal drift.** July-August 2017 performance collapsed (precision
   about 0.52 to 0.54). Cause not identified.
4. **Fairness.** False-positive rate DEU 0.219 vs PRT 0.070.
5. **Early exposure.** Aggregate test outcomes were seen before sealing (disclosed
   in `03_data_preparation.md` and `05_evaluation.md`). No modeling choice used them,
   but D5, the LightGBM comparison and the monthly diagnosis are descriptive only.
6. **Model choice not settled.** LightGBM beats the chosen logistic regression on
   the rolling folds (+212 per 1,000) and the test (+489, CI 241 to 773). It runs as
   a shadow challenger.
7. **One dataset, two hotels in Portugal.** No evidence about other markets.

## 4. Experience and what I would change

**Process.** Each phase was drafted by Claude Sonnet 5.5 and reviewed by GPT-5.6
Terra in a fresh context under a "world-renowned CRISP-DM authority" persona,
with at least two rounds each (scores in the table below).

| Phase | Rounds | Score progression |
|---|---|---|
| 1 Business understanding | 4 | 4.0, 7.8, 8.9, 9.2 |
| 2 and 3 Data understanding and preparation | 3 | 6.5, 8.0, 9.1 |
| 4 Modeling | 2 | 7.5, 9.1 |
| 5 Evaluation | 3 | 7.0, 8.7, 9.2 |
| 6 Deployment and final report | 3 | 6.5, 8.7, 9.4 |

What the critic loop caught that I would have shipped: the day-before-arrival
snapshot (the "booking-time" claim was unsupported); no-shows inside the target;
test outcomes inspected before sealing; a duplicate rule that let future rows
into training; a gate that quietly required the test; a pilot sample size that
did not match its stopping rule; and an arithmetic rounding error.

What I would change: (1) choose the model by mean rolling-fold net benefit, not one
validation segment; (2) stratify validation by season; (3) define the capacity
criterion B3 as a flagged-share requirement; (4) seal the test in code before the
first look at any outcome; (5) read the source paper's field definitions first.

**What a human still has to do** (the assignment requires it): read and be able
to explain every cell of the notebook, recompute two or three numbers by hand
(the notebook shows by-hand checks for the clipped mean and the break-even
formulas), run the notebook clean in Colab, publish the Medium article, record the
walkthrough video, and deploy the app to a public URL.

## 5. Reproducing the work

```
pip install pandas==3.1.1 scikit-learn==1.9.1 lightgbm==4.7.0 shap==0.54.1 jupyter
# place hotel_bookings.csv in data/ (or set a Kaggle token in Colab secrets)
jupyter nbconvert --to notebook --execute notebooks/crisp_dm_hotel_bookings.ipynb
streamlit run deployment/app.py
```

Seed 42. Frozen model hash `8c6cd3c57dfe46d8`. Artifacts in `artifacts/`,
test access log in `artifacts/test_access_log.json`.
