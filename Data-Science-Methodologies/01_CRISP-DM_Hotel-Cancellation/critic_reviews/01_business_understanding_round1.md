# Business Understanding, round 1

- **Date:** 2026-11-09
- **Critic model:** GPT-5.6 Terra (Codex), reasoning effort high, fresh context, "world-renowned CRISP-DM authority" persona
- **Drafter:** Claude Sonnet 5.5
- **Score:** before n/a / after 4.0 of 10

## Prompt sent

Persona and format from `CRITIC_PROMPTS.md` section 2 (CRISP-DM, phase = Business
Understanding, checklist from the project README). The critic read
`phase_notes/01_business_understanding.md` and the dataset, read only.

## Critique received (verbatim)

## 1. Required-subtask and output audit

| CRISP-DM task | Required output | Status | Evidence and judgment |
|---|---|---:|---|
| Determine business objectives | Background | **Partial** | Gives domain context, data scale, cancellation rates, and the business problem. However, it reverses the paper's hotel identifiers: H1 is the resort hotel and H2 the city hotel, not the reverse. It also omits the dataset's crucial observation time: values were extracted as of the day before arrival, not at booking creation. |
| Determine business objectives | Business objectives | **Complete** | Primary and secondary objectives are business-oriented, decision-focused, and distinguish intervention from prediction. |
| Determine business objectives | Business success criteria | **Partial** | B1-B4 are stated before modeling, but B1's "80% interval from the cost sensitivity analysis" has no input distributions/ranges or method; B2 leaves the comparator undefined; B4 is not actually established by D4's global calibration criteria. |
| Assess situation | Inventory of resources | **Partial** | Identifies data, people, compute, software, and knowledge sources. It fails to inventory the source data's defining temporal limitation: it is a near-arrival snapshot assembled from booking/change-log data, not a booking-creation dataset. |
| Assess situation | Requirements, assumptions, constraints | **Partial** | The stated requirement (features known at booking creation) is correct, and several suspect fields are named. But the assumption is not feasible with this supplied table as documented by its source, and the feature list omits important at-risk fields such as `country`, `days_in_waiting_list`, and `deposit_type`. |
| Assess situation | Risks and contingencies | **Partial** | Strong coverage of random splitting, duplication, drift, costs, and fairness. The contingency for the central data-timing risk is inadequate: a drop-column "leakage test" cannot recover booking-creation values that the dataset never supplies. |
| Assess situation | Terminology | **Partial** | ADR, lead time, save rate, PR-AUC, and calibration are adequately defined. The cancellation definition is false for this file: all 1,207 `No-Show` rows have `is_canceled = 1`. |
| Assess situation | Costs and benefits | **Partial** | Provides an intervention-level cost/benefit example and labels the main economic inputs as assumptions. CRISP-DM calls for project costs as well as projected business benefit; development, maintenance, and operating costs are absent. |
| Checklist requirement | Cost matrix for false positives and false negatives | **Partial** | The personal-outreach matrix correctly expresses incremental cost relative to doing nothing and labels figures as assumptions. The overbooking table is not a complete decision matrix, reverses the cost/gain sign convention in its heading, and does not produce an expected-value rule. |
| Determine data-mining goals | Data-mining goals | **Partial** | Classification, calibrated probabilities, cost-based decisions, and explainability are appropriate technical aims. The central "at booking creation" goal is unsupported by the supplied data; clustering is optional scope expansion without a defined decision use. |
| Determine data-mining goals | Data-mining success criteria | **Partial** | D3, D4, and D6 are measurable. D2 merely requires recall to be reported, not achieved. D5 is non-falsifiable: either performance does not collapse, or it collapses and the columns are excluded/reported. D1 does not specify the validation-level comparison used to obtain the fold standard deviation. |
| Produce project plan | Project plan | **Partial** | Covers phases, deliverables, estimated effort, and broad iteration loops. It lacks resources, inputs, dependencies, review/decision points, and schedule-to-risk dependencies for each phase, all specifically called for by CRISP-DM. |
| Produce project plan | Initial assessment of tools and techniques | **Complete** | Maps appropriate tools and techniques to concrete needs, including pipelines, time-aware validation, calibration, explanation, clustering, and deployment. |

The plan is structurally much stronger than a typical student Phase 1, but the central feasibility error prevents a higher completeness assessment.

## 2. Methodology errors

1. The assessment does not reconcile the business requirement with the actual resource. The project promises booking-creation prediction, while the source paper says the data point for each observation was defined as the day before arrival and values were taken from booking/change-log records relative to that point. This is not a minor Phase 2 detail; it determines whether the Phase 1 objective is possible at all. (Hotel booking demand datasets, repositorio.iscte-iul.pt/bitstream/10071/16929/1/Hotelbookingdemanddatasets.pdf)

2. The cost/benefit section is an intervention economics worksheet, not a full CRISP-DM cost-benefit analysis. It omits project-development, deployment, monitoring, and ongoing operating costs. The original guide explicitly requires a comparison of project costs with potential business benefit. (CRISP-DM 1.0 guide, kde.cs.uni-kassel.de/lehre/ws2012-13/kdd/files/CRISPWP-0800.pdf)

3. The project plan is incomplete under CRISP-DM. It supplies stages, durations, and outputs, but not stage-level resources, inputs, dependencies, critical steps, decision/review points, or explicit dependencies between risks and schedule. The cited CRISP-DM guide requires these elements.

4. Several "success criteria" are plans to report results rather than pass/fail criteria. D2 is descriptive, and D5 is satisfied in every outcome. CRISP-DM requires technical criteria that define a successful outcome, not merely analyses that will be performed.

5. The document correctly distinguishes business objectives from data-mining goals in form, but the underlying action design is not complete. It defines a costed high-risk outreach action, yet also deploys an uncosted, unevaluated medium-risk email action. A business policy cannot be evaluated by one decision rule while triggering another action outside that rule.

## 3. Technical errors

- **Fatal prediction-time contradiction.** The source documentation states that observations were constructed using values as of the day before arrival, including change-log values. It also warns that customers can change booking attributes at check-in or during their stay, and that nationality may not be known until check-in. The raw file therefore cannot establish what most fields were at booking creation. Deriving a booking date from arrival date minus `lead_time` does not reconstruct the historical feature state. Consequently, the stated requirement "only information available at booking creation" is currently unsupported. A time-based split by derived booking date prevents future-row training leakage; it does not remove within-row future-information leakage.

- **The proposed leakage exclusions are materially incomplete.** The paper specifically defines `booking_changes` as changes through check-in or cancellation, `country` as potentially unknown until check-in, and `deposit_type` using payment data before arrival or cancellation. `days_in_waiting_list`, `assigned_room_type`, ADR/transaction-derived values, and other change-log fields also need a field-by-field availability decision. The document names only a subset and leaves `country` as a fairness issue rather than also treating it as a timing issue.

- **The target definition is wrong.** The note says no-shows "are not cancellations here," but the local file contradicts it: all 1,207 `No-Show` reservations have `is_canceled = 1`. The project must either define the target as "cancellation or no-show/non-arrival" or construct a cancellation-only target. That choice changes the business action, loss model, and interpretation.

- **The hotel identifiers are reversed.** The project says city hotel H1 and resort hotel H2. The source paper says H1 is the resort hotel and H2 the city hotel. The raw file's `hotel` labels are correctly counted, but the H1/H2 assertion is false.

- **The "2015-2017 booking behavior" assumption is imprecise at best.** Arrival dates span July 2015-August 2017, but derived creation dates span 2013-06-24 to 2017-08-31; 2,637 implied creations are before 2015. The document should refer to arrival periods unless it explicitly defines and audits the derived booking-date period.

- **The outreach threshold arithmetic is correct.** With L=150, s=0.25, and c=20, the incremental expected benefit is p x 37.50 - 20, so action is worthwhile when p > 20/37.5 = 0.5333. At s=0.50, the threshold is 20/75 = 0.2667. The personal-outreach matrix is also arithmetically correct under its assumptions.

- **The cost matrix is incomplete as a policy model.** It assumes outreach affects only cancellers and carries no guest-experience, discount, capacity, or channel cost beyond a flat $20. That can be acceptable as an explicitly provisional assumption, but the medium-risk email action has no cost, save rate, or threshold at all. The model's recommended actions therefore do not map to a coherent multi-action decision rule.

- **The overbooking table has inconsistent signs and an incomplete decision rule.** It is headed "Cost," yet calls a benefit +$150 and a walk loss -$400; those are gains/losses, not costs. More importantly, summing individual cancellation probabilities gives expected cancellations, not the probability distribution of rooms available. Selecting extra rooms requires capacity, existing reservations, uncertainty/correlation, and a marginal expected-value rule. Under the table's own simplified gain/loss values, selling one extra room would require the probability of a walk to be below 150/(150+400)=27.27%, assuming those are the only two outcomes. No such rule is provided.

- **The currency claim is overstated.** The file supports median numeric ADR values of 99.9 for City Hotel and 75.0 for Resort Hotel. It does not establish that the unit is dollars. The sentence marking "$100" and "$75" as [D] conflates a verified numeric median with an assumed currency.

- **The verified [D] figures are otherwise correct.** The file has 119,390 rows and 32 columns; City Hotel has 79,330 bookings and a 41.73% cancellation rate; Resort Hotel has 40,060 and 27.76%; total cancellation rate is 37.04%; and no booking-ID field exists.

- **Success and validation risks remain.** Calibration must be fitted only on earlier training/validation periods, with the final temporal test untouched; the plan does not explicitly require a time-aware calibration split. If cost sensitivity, threshold choice, model selection, or simple-rule selection is tuned on the final test, the "final" test is no longer final.

- **Duplicate handling is underspecified.** "Group duplicates before splitting" is not sufficient when there is no booking ID and identical records may be legitimate independent group bookings. The grouping key must exclude outcome and post-outcome fields, and the temporal conflict between grouped duplicates and a strict chronological holdout must be resolved explicitly.

## 4. Five highest-value fixes

1. **Resolve the prediction-time feasibility problem before any modeling.** Document the paper's day-before-arrival snapshot construction; then either obtain genuine booking-creation snapshots/event logs or explicitly narrow the project to a retrospective benchmark with no claim of booking-creation deployment. Do not continue with a creation-time model using this table as-is.

2. **Define the target and decision moment exactly.** State whether the target is cancellation only or cancellation-plus-no-show, correct the H1/H2 labels, and ensure the action occurs before the target can happen. If using a near-arrival decision, establish a valid at-risk population and label window; if using booking creation, acquire data recorded then.

3. **Turn the action design into one coherent economic policy.** For every action tier, specify cost, expected effect, capacity limit, threshold/selection rule, and uncertainty range. Add project and operating costs. Replace the overbooking sketch with a capacity-aware probabilistic decision rule or remove it from this project.

4. **Replace non-falsifiable success criteria and lock validation.** Define the simple-rule baseline before modeling; give D2 a target; specify how B1 uncertainty is constructed; evaluate weekly aggregate calibration directly; and require nested/rolling time validation with calibration and threshold selection conducted before the untouched final test.

5. **Complete the CRISP-DM plan and data-availability audit.** Add a field-level "known at decision time" table, a no-feasible-data contingency, per-phase inputs/resources/dependencies, review gates, and explicit schedule risks. Treat it as a living plan with a decision gate after Phase 2.

## 5. Score

**4.0 / 10**

The document is unusually well organized, decision-oriented, and candid about many assumptions. But the unaddressed fact that the dataset is a near-arrival snapshot makes its central booking-creation claim invalid, and that flaw propagates into leakage control, evaluation, policy value, and deployment.

It would rise by one point by fully resolving the prediction-time contract: either obtain/identify booking-creation feature snapshots and a prospective label definition, or explicitly recast the work as a retrospective near-arrival benchmark with actions and targets that are feasible at that time.

## What I changed

(see `phase_notes/01_business_understanding.md`, v2)

- Replaced the "booking creation" prediction time with an explicit **prediction contract**: a retrospective benchmark with a stated decision moment, a field-level "known at decision time?" table, and a decision gate after Phase 2.
- Corrected the target: `is_canceled = 1` includes no-shows, so the target is "non-arrival (cancelled or no-show)".
- Corrected the H1/H2 labels.
- Collapsed the three-tier action table into one costed action with a capacity cap.
- Moved overbooking out of scope, with the correct decision rule noted.
- Added project and operating costs.
- Made D2 and D5 pass/fail, defined the B1 interval method, defined the B2 baseline rule, added a weekly aggregate calibration test, and fixed the calibration/threshold selection order.
- Added inputs, dependencies, and review gates per phase.
- Changed the currency symbol to a labelled assumption.

## What I rejected, and why

Nothing rejected in this round. Two points were adapted instead of taken literally:

- "Obtain genuine booking-creation snapshots": not possible, since this is a public dataset. We recast the project as a retrospective benchmark, which the critic offered as the alternative.
- The clustering goal: kept, but labelled optional with a defined decision use.

## What I verified by hand

Against `data/hotel_bookings.csv`:

- No-shows: the cross-tab of `reservation_status` by `is_canceled` shows 1,207 `No-Show` rows, all with `is_canceled = 1`. Confirmed.
- Derived booking dates: arrival dates span 2015-07-01 to 2017-08-31, derived booking dates 2013-06-24 to 2017-08-31, and 2,637 are before 2015. Confirmed.
- Post-booking leakage signal: `assigned_room_type` differs from `reserved_room_type` for 18.8% of non-cancelled bookings versus 1.8% of cancelled ones, and mean `booking_changes` is 0.29 versus 0.10. Both fields look recorded after the booking was made.
- Not yet verified: the H1/H2 labelling and the "day before arrival" snapshot claim come from the critic's reading of the paper. I have not read the paper myself, so I treat these as reported by the critic.
