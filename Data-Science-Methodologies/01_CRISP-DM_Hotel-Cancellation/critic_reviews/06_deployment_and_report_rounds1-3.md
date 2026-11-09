# Deployment and final report, rounds 1 to 3

- **Critic model:** GPT-5.6 Terra (Codex), CRISP-DM authority persona, fresh context each round
- **Drafter:** Claude Sonnet 5.5
- **Scores:** round 1 = 6.5, round 2 = 8.7, round 3 = 9.4 (sign-off yes)
- **Inputs sent:** `06_deployment.md`, `REPORT.md`, the deployment code and the slide deck (small files)

## Round 1 (verbatim)

| Deployment task / checklist item | Status |
|---|---|
| Streamlit inputs | Complete |
| Outputs and recommended action | Partial |
| Deployment packaging / public availability | Partial |
| Monitoring, drift checks, retraining triggers | Partial |
| Rollback | Partial |
| Final report and final presentation | Partial |
| Project review / experience documentation | Complete |

1. **High: the app is not a valid implementation of the approved pilot.** Phase 5 permits only randomized, 50/50 outreach/control assignment for non-Group flagged bookings; Group bookings are excluded. The app recommends outreach for every above-threshold booking, including Groups, and provides no randomization, treatment-arm capture, or pilot guardrail. This converts a research-study conditional go into operational action.
2. **High: the weekly 25% cap is unsafe in production.** `capped_flags` caps only rows present in one uploaded CSV. Separate uploads or an incomplete weekly extract can each flag up to 25%, exceeding the real weekly capacity. The app neither requires a complete arrival-week population nor warns that the cap is batch-local.
3. **Medium: the app overstates feature availability.** It says the 27 features are "known when a booking is made." The report and Phase 5 correctly say availability of history counters is assumed, not proven, and the source is a day-before-arrival snapshot. The UI must carry that qualification.
4. **Medium: monitoring is described more strongly than implemented.** `monitor.py` implements PSI, aggregate error, and flagged precision, but not the claimed monthly country-group false-positive-rate check, pilot save-rate comparison, quarterly scheduling, challenger comparison, or rollback execution.
5. **Medium: no final presentation is evidenced.** An outline is not a final presentation. The report otherwise agrees with Phase 5's no-go for automation, conditional pilot decision, metrics, uncertainty, and test-integrity disclosure.

**Score: 6.5/10. No sign-off.**

### Changed after round 1
- Pilot mode (default on): Groups ineligible, deterministic 50/50 arm from a hash of the booking key, action text for contact / do NOT contact / not in the pilot.
- Cap warnings: page warning that the cap is local to the file, and a warning for arrival weeks with fewer than 200 bookings.
- App wording now says availability is assumed, not verified.
- Monitoring table has a Status column (implemented vs procedure); country false-positive check implemented in `monitor.py` and demonstrated (ratio 3.1).
- `article/presentation.html`, a 10-slide deck with figures.
### Rejected
- Nothing. The public URL stays a user step (needs a GitHub push and Streamlit Community Cloud).

## Round 2 (verbatim)

1. **Resolved**: Pilot defaults on; Groups are excluded and deterministic 50/50 outreach/control assignment is implemented with explicit action text and hotel-side outcome logging.
2. **Resolved**: CSV-local-cap warning and sub-200-booking-week warning are present.
3. **Resolved**: App and report now clearly say booking-time feature availability is assumed, not verified.
4. **Resolved**: Monitoring distinguishes implemented code from procedures; PSI, error, precision, and country FPR ratio are implemented and documented.
5. **Partly**: A credible 10-slide deck exists, but it makes one materially false validation claim (below). Public URL is correctly documented as an explicit user-side deployment step, not an author defect.

New defects:
1. **Major: presentation claim contradicts the report.** Slide 4 says the final test was "sealed until the end," while REPORT.md explicitly discloses aggregate test outcomes were seen before sealing. The deck must disclose this, not claim a fully sealed test.
2. **Moderate: capacity is consumed by excluded Groups.** Batch scoring applies the weekly cap before pilot eligibility. Flagged Group bookings, which cannot receive outreach, can occupy the 25% slots and suppress eligible non-Group pilot bookings; the displayed "flagged for outreach" count also overstates actionable outreach.

**Score: 8.7/10. Sign-off: No.**

### Changed after round 2
- Slide 4 and the article reworded: test fixed by date before analysis, opened once for predictions, with the early-exposure disclosure.
- Pilot eligibility now applied **before** the weekly cap (`score_frame(..., eligible=...)`); Groups count toward the week's size but never take a slot; displayed counts are actionable ones.
- Phase 5 pilot sizing recomputed with the deployed rule: eligible flagged precision 0.637, control arrival rate 0.36, about 710 per arm (1,420 total), about 980 eligible flagged per month, about 1.4 months. Article, deck and outline updated.

## Round 3 (verbatim)

1. Test-integrity correction: **Resolved.** Slide 4 accurately discloses early aggregate exposure and limits the "opened once" claim to predictions.
2. Pilot eligibility/cap correction: **Resolved.** Eligibility precedes ranking; Groups count in weekly capacity but cannot occupy a slot. App output and pilot sizing align.

New defects: none.

**Score: 9.4/10. Sign-off: Yes.**

## What I verified by hand
- Headless Streamlit tests (`AppTest`) of the default booking, a long-lead online booking (control arm, "do NOT contact") and a Group booking ("not in the pilot").
- Notebook integration test: deployed model files reproduce the Phase 5 probabilities (tolerance 1e-10) and capped flags on all 24,018 test rows.
- Pilot sample size recomputed outside the notebook (n = 712 per arm with precision 0.637).
- Not verified: the public deployment, and a real pilot (no hotel data).
