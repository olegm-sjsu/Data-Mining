# Data Understanding and Data Preparation, rounds 1 to 3 (reviewed as a pair)

- **Critic model:** GPT-5.6 Terra (Codex), CRISP-DM authority persona, fresh context each round
- **Drafter:** Claude Sonnet 5.5
- **Scores:** round 1 = 6.5, round 2 = 8.0, round 3 = 9.1 (sign-off yes)
- **Inputs sent:** the phase notes only (the notes carry the real notebook outputs), never the notebook or the data

## Round 1 (verbatim)

| Phase / CRISP-DM subtask | Status | Evidence |
|---|---|---|
| DU: Collect initial data | Complete | Source, licence, acquisition method, file size, parsing convention, and scope limits are recorded. |
| DU: Describe data | Partial | Core size/rates/ranges are stated, but the claimed column-level dictionary is external, not in the written notes under review. |
| DU: Explore data | Complete | Six explicit EDA questions each have quantified findings. |
| DU: Verify data quality | Partial | Missingness, invalid values, duplicates, tails, and anomaly analysis are quantified, but availability-at-booking remains unverified. |
| DP: Select data | Partial | A/B/excluded choices are explicit, but 21 Set-A fields are accepted as "normally fixed" without field-by-field evidence. |
| DP: Clean data | Complete | Every listed cleaning rule has a count; removals reconcile to 182 rows. |
| DP: Construct data | Complete | Derived features and Set-B-only construction are documented. |
| DP: Integrate data | Complete | Correctly states that one source table requires no integration. |
| DP: Format data | Complete | Fit-only pipeline, encoding, clipping, and leakage checks are specified. |
| DP: Outlier analysis | Complete | Tails, clipping thresholds, affected-row scale, and retention rationale are provided. |
| DP: Final dataset and description | Partial | Output count is supplied, but the claimed 32-column prepared schema is not enumerated or reconciled with constructed fields and segment label. |

1. **Critical: the "untouched test" is already contaminated.** Phase 2 reports test-period target rates, and Phase 3 uses those labels to characterize drift and the suitability of the split. That directly conflicts with the fixed protocol that test is opened once. Even absent model fitting, inspecting test outcomes before final evaluation can influence choices and invalidates the claimed untouched-test guarantee.
2. **High: the booking-time deployment claim is unsupported.** The notes explicitly acknowledge a day-before-arrival snapshot and that the source-paper definitions have not been read. Yet 21 Set-A fields are admitted on "normally fixed" intuition, not evidence. This fails Phase 1's required field-by-field availability audit.
3. **High: the decision gate is logically premature.** "Strong univariate signal" is not evidence that Set A supports the business objective. The gate promises to decide whether Set A is adequate, then defers the decisive model test to Phase 4.
4. **Medium: duplicate handling is under-specified and weakly tested.** Exact twins include arrival-date fields, making zero cross-cut twins almost guaranteed under an arrival-date split. The looser-profile sensitivity is useful, but it is not the duplicate rule promised in Phase 1, nor does it resolve train-fold duplication.
5. **Medium: final-dataset documentation is not auditable from the notes.** The prepared file is said to retain 32 columns while adding constructed features and a segment label; no final schema establishes what was retained, transformed, or removed.

**Score: 6.5/10. No sign-off.**

### Changed after round 1
- Test period fixed by date (arrival from 2017-04-22) before any outcome is inspected; all target-based Phase 2 analysis redone on the 95,349 development rows; test labels removed from Phase 3 tables; twin keys no longer contain the target; exposure log added.
- Set A split into A1 (fixed by the reservation request) and A2 (history counters) with a Phase 4 comparison; availability stated as definition-based and unprovable from this data.
- Decision gate moved to Phase 4 (Amendment A2).
- Full dictionary table (note 02) and final schema table (note 03) added.
### Rejected
- Nothing outright. Point 1 cannot be fully undone, because outcomes were already seen in aggregate. It is disclosed in the exposure log and accepted as a limitation.

## Round 2 (verbatim)

1. **Partly resolved**: Target-based Phase 2 analyses now use 95,349 development rows, but the exposure log confirms test outcomes were already inspected; a holdout cannot become genuinely "sealed" retrospectively.
2. **Partly resolved**: A1/A2 and the conditional retrospective claim are appropriate, but classing fields as "fixed by reservation request" remains definition-based without source-field documentation.
3. **Resolved**: The Phase 2 gate is correctly deferred to Phase 4 model evidence under D1-D4.
4. **Resolved**: Target-free exact/profile keys, 0/4,887 counts, dual test reporting, and the duplicate-removal sensitivity are explicitly specified.
5. **Partly resolved**: Both requested tables exist, but `arrival_date_year` is "split only" in Phase 2 and 3.1, yet "Set A" in the final schema.

New findings:
1. **Critical, gate/test contradiction.** Amendment A2 says Phase 4 decides the gate against D1-D4, but D1-D4 explicitly require test-period performance. That opens the purportedly sealed test before Phase 5.
2. **Major, Set A cardinality/schema inconsistency.** The dictionary's Set A fields total 20, while the notes claim 21; the extra field is `arrival_date_year`.
3. **Moderate, exact-twin result is tautological.** The 0 count is not duplicate evidence; the profile-key analysis is the meaningful result.

**Score: 8.0/10. Sign-off: No.**

### Changed after round 2
- Gate now uses validation-period versions of D1 to D4 only; test opened once in Phase 5.
- `arrival_date_year` described everywhere as an input-only column (20 Set A columns plus the year input, 21 raw app inputs); schema role relabelled.
- Exact-twin count stated to be non-evidence.
### Rejected
- Round 2 points 1 and 2 (partly resolved): accepted as limitations that cannot be fixed retrospectively (aggregate test exposure; paper definitions not accessible).

## Round 3 (verbatim)

1. Gate/test separation: Resolved. A2 clearly assigns the gate to validation-period D1-D4 and reserves one test opening for Phase 5.
2. `arrival_date_year`: Partly resolved. Phase 2 and 3.1 correctly call it input-only, but the final schema (3.7) labels it "Set A".
3. Exact twins: Resolved. The zero exact-twin count is explicitly non-evidence.

New defect: the final schema's year role should be corrected to "input only" (not Set A).

**Score: 9.1/10. Sign-off: Yes.**

### Changed after round 3
- Schema table in note 03 and `final_dataset_schema.csv` now label the year "input only".

## What I verified by hand
- Mean of clipped lead time recomputed outside the pipeline (102.76), matching the fitted scaler.
- Segment dates checked to be strictly ordered by assertions in the notebook (all pass).
- Every number in the notes was copied from notebook output. After the sealed-test change, the whole of section 2 was rerun and the notes rewritten from the new output.
- Not verified: Set A availability at booking time and the paper's field definitions.
