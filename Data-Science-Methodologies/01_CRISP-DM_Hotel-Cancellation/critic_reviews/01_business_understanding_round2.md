# Business Understanding, round 2 (diff check)

- **Date:** 2026-11-09
- **Critic model:** GPT-5.6 Terra (Codex), reasoning effort medium, fresh context, same CRISP-DM authority persona
- **Drafter:** Claude Sonnet 5.5
- **Score:** before 4.0 / after 7.8
- **Format:** diff check. The critic got the five round-1 findings with my fixes and read only draft v2. A first, broader round 2 was cancelled to save compute.

## Prompt sent

Persona as in `CRITIC_PROMPTS.md` section 3, trimmed: status of the five round-1
findings, at most 5 new findings, under 400 words, new score, sign-off verdict.

## Critique received (verbatim)

1. **Resolved**: The prediction contract correctly frames this as a retrospective booking-time benchmark using Set A, with Set B confined to an upper-bound leakage comparison and a Phase 2 gate.
2. **Resolved**: The target is consistently defined as non-arrival; no-shows are included and counted, and H1/H2 are correctly resort/city.
3. **Resolved**: One costed outreach action is specified; overbooking is explicitly out of scope and its break-even arithmetic is correct.
4. **Partly resolved**: B1/B2/B4/D2/D5 are substantially more testable, but the Set-B comparison remains ambiguous (see finding 2).
5. **Partly resolved**: Inputs, dependencies, gates, costs, and validation steps are present, but the duplicate rule corrupts the time split (see finding 1).

New findings:

1. **Critical, duplicate handling leaks future observations into training.** The rule sends a duplicate group that crosses the booking-date cut to training, even when some rows lie in the final 20% test period. This violates "last 20% ... is the test period" and permits post-cut observations in training. Keep all post-cut rows out of training; report cross-boundary duplicates or drop/quarantine them.
2. **Major, D5 does not define the required leakage comparison.** "PR-AUC gap between Set B and Set A" could mean B-only versus A-only, whereas the technical goal says to quantify what B *adds*, which requires comparing **A+B versus A**. Define this explicitly.
3. **Major, B3 lacks an evaluation scope and enforcement rule.** "At most 25% ... in any arrival week" does not state whether this is evaluated only on the untouched test period, nor how a globally selected threshold is constrained if a week exceeds capacity. It is not yet an operationally testable policy constraint.
4. **Moderate, B4 can be undefined.** Weekly aggregate error divides by actual non-arrivals without specifying treatment for weeks with zero non-arrivals. Define a zero-denominator rule before evaluation.
5. **Moderate, the stated business rationale overclaims.** A cancellation does not necessarily leave a room empty; it may be resold. The loss model later recognizes this through `u`, so revise the earlier assertion that the room is empty "in both cases."

**Score: 7.8/10.** **Ready to sign off: no.** The temporal-validation defect is material and must be corrected before sign-off.

## What I changed (draft v3)

1. Duplicates: the split is strictly by date, row by row, so no test-period row can reach training. Test rows with an identical twin in training are counted, and test results are reported on all test rows and on twin-free test rows.
2. D5: gap defined as PR-AUC(A+B) minus PR-AUC(A) for the same model type, folds, and test period. Technical goal 4 reworded to match.
3. B3: evaluated on the test period only, threshold fixed on validation, weekly rank cap at 25% of that week's bookings, pass if the cap binds in at most 10% of test weeks, B1 and B2 computed with the cap.
4. B4: weeks with zero actual non-arrivals excluded, count reported.
5. Rationale reworded: the room is unused by that guest and resold if possible, tied to the parameter `u`.

## What I rejected, and why

Nothing. All five findings were valid.

## What I verified by hand

Re-read the edited sections against each finding. The B3 rank cap and the twin-free reporting are design decisions, so they are only checkable once code exists in Phases 3 to 5.
