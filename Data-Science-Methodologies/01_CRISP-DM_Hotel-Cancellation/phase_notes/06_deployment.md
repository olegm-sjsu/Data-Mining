# Phase 6: Deployment

Draft v2 (after critic round 1). Code is in `deployment/`; checks run in notebook section 6.

## 6.1 Plan deployment

**What is deployed:** a Streamlit decision-support app, not an automated policy
(Phase 5 decision: no-go for automation). It is meant for a randomized pilot.

| Item | Detail |
|---|---|
| Inputs | The 21 Set A raw fields: hotel, arrival date, lead time, nights, party, meal, room type, market segment, channel, customer type, history counters, agent, company |
| Outputs | Probability of non-arrival, the threshold (0.55), a recommendation (flag or no action), the challenger's probability, a warning for July and August arrivals |
| Recommended action | If flagged: personal outreach, reconfirm the stay, offer a free date change or a small discount for a non-refundable rate. Otherwise none |
| Pilot mode (default on) | Implements the Phase 5 pilot rules: **Group bookings are not eligible**, and eligible flagged bookings get a deterministic 50/50 arm (outreach or control hold-out) from a hash of the booking key (`booking_id` if supplied, else the full row). The recommendation then says contact, do not contact (control), or not in the pilot. Outcome logging must happen in the hotel's own system |
| Batch mode | Upload a CSV; the weekly 25% cap is applied (highest scores first within each arrival week); download the scored file. **The cap is local to the uploaded file**, so the app warns on the page, and again when any arrival week in the file has fewer than 200 bookings |
| Packaging | `requirements.txt` with pinned versions, a `Dockerfile`, `README.md` with run steps |
| Model versioning | SHA-256 prefix of the frozen model is stored in `policy.json` and shown in the app |

**Verification performed:**

1. *Integration test (notebook 6.1):* the deployed files are loaded from disk and
   score the 24,018 raw test rows. Probabilities equal the Phase 5 values (tolerance
   1e-10) for both models, and the capped flags are identical.
2. *Headless app test* (Streamlit `AppTest`): the app starts without exception;
   a 90-day, online, 2017-06-15 default booking gives 49.8% and no action; a
   300-day group booking in the city hotel gives 95.5% and "Flag"; a 2-day direct
   booking gives 21.8%; a July arrival shows the July-August warning.
3. A 200-row sample CSV (`deployment/sample_bookings.csv`) scores correctly
   through the batch path (35 of 200 flagged with pilot mode off; 31 with it on, of which 13 are
   outreach arm, 18 control arm, and 14 more are Group bookings above the threshold that are excluded).
4. *Pilot logic* (notebook 6.2): eligibility is applied **before** the weekly cap,
   so Group bookings never take a slot. On the test period: 4,229 eligible flagged
   bookings (precision 0.637, about 980 per month), 51.7% assigned to the outreach
   arm, 0 Group bookings flagged, 2,042 Group bookings above the threshold but
   excluded, and the assignment is stable on repeat. The headless app test shows
   a flagged online booking is put in the control arm with a "do NOT contact"
   action, and a flagged Group booking says it is not in the pilot.

**Not done (needs you):** the public URL. It needs a GitHub push and a Streamlit
Community Cloud deployment (steps in `deployment/README.md`). Until then the app
runs locally and in Docker.

## 6.2 Plan monitoring and maintenance

Outcomes are only known after the stay date, so performance checks run with a
lag; input drift checks run at once. The "Status" column separates what is
**implemented in `deployment/monitor.py`** from what is a written **procedure**
that a person must run.

| Check | Cadence | Threshold | Action | Status |
|---|---|---|---|---|
| Input drift: PSI of lead time, market segment, customer type, hotel, nights vs training | Weekly | 0.10 watch, 0.25 retrain trigger | Investigate; retrain on fresh data | Implemented (`psi`, `alerts`) |
| Aggregate error per arrival week | Weekly, lagged | Above 0.15 three weeks in a row | Retrain trigger | Implemented (`weekly_report`, `alerts`) |
| Flagged precision | Weekly, lagged | Below 0.60 three weeks in a row | Pause outreach, review | Implemented |
| False-positive rate by country group | Monthly | Max/min ratio above 1.5 | Review proxies such as `agent_id` | Implemented (`group_rates`, `fpr_ratio`) |
| Save rate (pilot treatment vs hold-out) | End of pilot, then quarterly | Below the break-even 0.186 | Stop outreach | **Procedure** (needs the hotel's outcome log) |
| Scheduled retraining | Quarterly, and before May | n/a | Retrain; re-run the Phase 5 criteria on a fresh holdout | **Procedure** |
| Champion vs challenger | After the pilot | Challenger wins on net benefit, block-bootstrap interval excludes zero | Switch | **Procedure** |
| Rollback | On a trigger | See below | Restore previous files | **Procedure** (manual file swap) |

**Demonstration on the test stream** (treated as live; `monitor.py`): PSI against
training is 0.17 for lead time, 0.14 for market segment, 0.13 for customer type
(all "watch"), 0.05 for nights and 0.00 for hotel. The country check gives a false-positive-rate ratio of 3.1, so it fires too. Weekly flagged precision
fell below 0.60 in weeks 30, 31 and 32 of 2017, so the precision rule fires. The
aggregate error did not: it was above 0.15 in only 2 of 20 weeks (weeks 30 and 35)
and never three weeks in a row, while the flagged set got worse. **Aggregate calibration (B4) alone would have missed the
summer degradation.** The three-week precision rule was set after seeing this
stream, so this is an illustration and not an independent validation of the rule.

**Rollback.** Keep the previous model file and its hash. If a trigger fires and
no fresh model passes the criteria, the fallback is "no model-driven outreach"
(the lead-time rule is not a safe fallback, since it lost money on the test).
Switching back needs only replacing `model.joblib` and `policy.json`.

**Challenger.** LightGBM runs in shadow. After the pilot, compare it with the
champion on the new data; switch only if it wins on net benefit with a block
bootstrap interval that excludes zero.

## 6.3 Produce final report

`REPORT.md` (full report), `article/medium_article.md` (draft for Medium) and
`article/presentation.html` (10-slide final presentation, open in a browser) and `article/presentation_outline.md`.

## 6.4 Review project

See "Experience and what I would change" in `REPORT.md`. In short: the biggest
gains came from the critic loop catching leakage and protocol problems early;
the biggest regrets are the model-choice rule based on one validation segment
and not stratifying validation by season.
