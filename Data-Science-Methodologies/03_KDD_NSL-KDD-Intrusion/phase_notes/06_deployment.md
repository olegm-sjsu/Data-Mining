# Step 9 (extension): using the knowledge, deployment and monitoring

Fayyad's step 9 is "using the discovered knowledge". The course's five KDD steps stop before it, so this is an
**extension**. Draft v1. Code in `deployment/`; checks in the notebook's extension section.

## FastAPI service

`deployment/main.py` (run `uvicorn main:app`; interactive docs at `/docs`):

| Endpoint | Purpose |
|---|---|
| `GET /health` | status, model hash, anomaly threshold |
| `POST /classify` | one connection (the 41 features) |
| `POST /classify_batch` | 1 to 1,000 connections |

**Input validation** (HTTP 422 on failure): all 41 fields required; counts at least 0; the 14 rate features between 0 and 1; the four binary
fields 0 or 1; `protocol_type`, `service` and `flag` must be values seen in training (the error lists examples).
**Output** per connection: the attack `category` and class probabilities (boosted model, 15 leaves, balanced weights), an `anomaly_score` and
`anomaly_alarm` (normal-only PCA reconstruction error against the frozen threshold 20.51), an `alert` flag (a non-normal category **or** an anomaly alarm), the
**matched rules** (fixed rules and the supplementary R2L rules, each with its training confidence, lift and support), the model hash and a warning that the data is 1998-derived.

**Integrity:** the service hashes `model_bundle.joblib` at start and refuses to run if it differs from `policy.json`. This detects corruption or an accidental
file swap; it is **not** protection against someone who edits both files (that would need a signed or externally stored policy).

## What was verified

| Check | Result |
|---|---|
| Service vs notebook on five test connections (normal, neptune, ipsweep, buffer_overflow, mailbomb) | categories identical to the notebook's predictions |
| Invalid inputs: rate of 1.7, unknown service, missing field | HTTP 422 for all three |
| Batch of 50 | HTTP 200, 21 alerts |
| Tampered bundle (one byte appended) | refused at start |
| **Docker** (`docker build`, `docker run`, image 963 MB) | `/health` returned status ok; `/classify` on an `ipsweep` record returned `Probe`, alert true, 3 rules matched |

Example outputs: normal connection: `normal`, no alert; a `neptune` connection: `DoS`, alert (no fixed rule matched it); an `ipsweep`
connection: `Probe`, alert, 3 rules matched; a `buffer_overflow` and a `mailbomb` connection: both `normal`, **no alert** (anomaly scores 10.8 and 1.3 against
the 20.5 threshold). The service reproduces the model, including its blind spots.

## Monitoring plan

| Check | Cadence | Threshold | Action | Status |
|---|---|---|---|---|
| Feature drift (PSI of `src_bytes`, `count`, `serror_rate`, `dst_host_*`, `duration` against training) | Daily | 0.10 watch, 0.25 act | Investigate; retrain | Library function (`kdd_monitor.feature_drift`); not scheduled |
| Alert volume against the baseline rate | Daily | more than 1.5 times the baseline | Check for an attack wave or model drift | Library function (`alert_report`); not scheduled |
| Analyst feedback: share of alerts confirmed | Weekly | below 0.60 | Tune thresholds; retrain | Library function (`analyst_precision`); needs the analysts' verdicts |
| **Missed-attack reports** (incidents found by other means that the service did not alert) | Weekly | any trend upward | Add the cases to the training data; retrain | **Procedure** (needs an incident process) |
| Retraining | Quarterly and on any trigger | n/a | Refit on fresh labelled data with a new validation design, then a fresh test | **Procedure** |
| Rollback | On a trigger | n/a | Restore the previous bundle and `policy.json` | **Procedure** (manual file swap) |

There is no scheduled job, stored reference distribution or dashboard in this repository.

**Demonstration, treating `KDDTest+` as the stream:** the largest feature PSI is `serror_rate` at 0.197, which raises a **watch** message (above 0.10, below the 0.25 retrain line); then
`dst_host_srv_count` 0.065. The alert rate is 0.412 against a training baseline of 0.465 (ratio 0.89). Alert precision is 0.964. **No retrain or alert-volume trigger fires.** Yet the
model finds only 11.8% of R2L, 2,487 of whose 2,885 records it calls normal. **Feature drift and alert volume cannot see a class-mix and novel-attack shift of this kind** (the one watch message is about `serror_rate`, not about the missed R2L).
The monitor returns NaN, with an explicit "too little data" or "no analyst verdicts" message, instead of a number when a sample is empty or too small.
Only labelled feedback (missed-attack reports, analyst verdicts) can, which is why those rows are in the plan.

## Limits

1. Trained on a 1998-derived simulation: not validated on modern traffic or encrypted traffic.
2. R2L and novel content-level attacks are largely missed (Interpretation); the service reports a normal category for them without any alert.
3. The service returns decision support for an analyst, not a blocking decision.
4. Rules shown are training rules with their training confidence, not guarantees.
