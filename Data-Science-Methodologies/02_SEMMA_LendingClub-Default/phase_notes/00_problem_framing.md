# Extension 0: Problem framing (outside SEMMA)

SEMMA has no business-understanding stage. SAS defines it as a way to organize
Enterprise Miner's tools, so this framing is an **extension** added to make the
later stages judgeable. Draft v2 (after critic round 1).

## Question and decision

Which Lending Club loans will charge off, using only what is known at
origination? The decision supported: accept or decline a 36-month personal loan
application. The 60-month term is handled separately as a secondary check
(Sample, S.2).

## Money is measured with realized cash flows

Profit per loan = `total_pymnt - funded_amnt`, the cash an investor actually
received minus the amount funded. These two columns are used **only in Assess**,
never as inputs. This needs no loss-given-default assumption.

**Counterfactual convention for "decline":** a declined loan earns 0 (the capital
is not deployed), so the value of a policy is the total profit of the loans it
accepts, and "profit per loan funded" is that profit divided by the dollars it
funds. Capital redeployment, selection effects (we only observe loans Lending Club
already accepted) and servicing fees are ignored, and cash flows are undiscounted.
The results are therefore a comparison of rankings under realized cash flows, not
a forecast of a lender's return.

## Success criteria, fixed before modeling

| ID | Criterion | Pass condition |
|---|---|---|
| C1 | Out-of-time ROC-AUC of Model A (application data only, no Lending Club grade or rate) | at least 0.68, and at least 0.02 above a FICO-only logistic regression |
| C2 | KS on the out-of-time test | at least 0.25 |
| C3 | Calibration on the out-of-time test | slope in [0.8, 1.2], and mean predicted within 1.5 points of the observed default rate |
| C4 | Profit of loans accepted at the validation-chosen cutoff versus accepting all | higher, with a 95% block-bootstrap interval (blocks = issue month) of the difference per loan funded excluding 0 |
| C5 | Stability | score PSI from train to test at most 0.25; AUC in each test quarter within 0.03 of the overall AUC |
| C6 | Fairness (report, with a flag) | zip code and state are excluded as inputs; approval rates reported by state and income band; flag any group with an approval-rate ratio below 0.8 of the highest |

Three model variants answer different claims. **A** uses application and
bureau data only: it measures independent risk assessment. **B** adds Lending
Club's own grade, sub-grade and rate: it measures how much their pricing adds and
what an investor who trusts the grade would see. Model A is the headline.

## Planning log (exposure disclosure)

While planning the partition, aggregate default rates by issue year and term
were printed for the whole eligible file, **including the vintage later used as
the out-of-time test** (2015, 36-month loans, about 14.9%, against 12.4% in the
training vintages). No model existed and no modeling choice used those
numbers, but the test is **not unseen at the aggregate level, and the notes call it
"held out", not "sealed"**. All Explore, Modify and Model analysis uses training
and validation loans only. The test partition is the **full 2015 population, with no
sampling**, so no outcome-based selection touched it.
