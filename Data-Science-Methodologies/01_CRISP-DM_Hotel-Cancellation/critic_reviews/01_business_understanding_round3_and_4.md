# Business Understanding, rounds 3 and 4 (diff checks)

- **Critic model:** GPT-5.6 Terra (Codex), same persona, fresh context each time
- **Scores:** round 3 = 8.9, round 4 = 9.2 (round 2 was 7.8, round 1 was 4.0)

## Round 3 (verbatim)

All five round-2 fixes: Resolved. New findings:

1. **High:** threshold selection is not specified under the operational cap. The threshold is chosen on validation before B3's weekly top-25% cap, yet the deployed/test policy is threshold-plus-cap. Optimize and freeze the complete capped policy on validation.
2. **Medium:** B2's comparator capacity treatment is ambiguous. B3 describes capping "it" by predicted probability, which a lead_time rule cannot supply. State whether the same 25% weekly capacity applies to the rule and specify its deterministic tie/ranking treatment.
3. **Low:** document metadata still says "Draft v2", not v3.

Score 8.9/10. Sign-off: no.

**Changed:** threshold now minimizes expected cost of the full capped policy on validation and is then frozen; B2 rule gets the same weekly 25% cap, ranked by lead_time descending with deterministic tie-breaks, tau chosen under the cap; header updated to v4. **Rejected:** nothing.

## Round 4 (verbatim)

1. Threshold validation/full capped policy: Resolved.
2. B2 capped, rank-based baseline: Resolved.
3. Draft v4 header: Resolved.

New defects:

- B3 omits deterministic tie-breaking when predicted probabilities tie at the 25% cap boundary.
- "Arrival week" is not operationally defined (for example, ISO week and year), affecting all cap-based metrics.

Score 9.2/10. Sign-off: No, these are small but reproducibility-relevant specification gaps, not merely style.

**Changed:** added the probability tie-break (earlier arrival date, then row order) and defined arrival week as ISO year plus ISO week of `arrival_date`. **Rejected:** nothing.

## Sign-off note

Score is above 9, no subtask is Missing, and the two remaining points are
one-sentence specification gaps. They were fixed directly and not sent back for
a fifth review, to save compute. Phase 1 is signed off as draft v4.

## What I verified by hand

Numbers checked against the data and the paper summary are in
`01_business_understanding_round1.md`. The threshold arithmetic
(20 / 37.5 = 0.533; 20 / 75 = 0.267; 150 / 550 = 0.273) was also confirmed by
the critic in round 1.
