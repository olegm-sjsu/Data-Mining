# Which Lending Club loans will charge off? A SEMMA project

Final report. Data: *All Lending Club loan data*, Kaggle `wordsforthewise/lending-club`
(CC0), file `accepted_2007_to_2018Q4`. Notebook: `notebooks/semma_lending_club.ipynb`. Stage
notes: `phase_notes/`. Critic reviews: `critic_reviews/`. SEMMA has no business or
deployment stage, so the framing and the deployment are labelled **extensions**.

## 1. Summary

We built models that rank 36-month loans by default risk using only what is known at
origination, judged them on 282,787 later loans (all of 2015), and priced them with realized
cash flows.

- **What works:** the champion (LightGBM on application and credit-bureau data, no Lending
  Club grade or rate) has test ROC-AUC **0.679**, KS 0.260, Gini 0.359. Its riskiest decile
  defaults at 30.9% against 3.4% for the safest, and holds 20.8% of all defaults. Scores are
  stable across 2015 (PSI 0.007; quarterly AUC 0.673 to 0.683).
- **What does not:** it is **not calibrated** to 2015 (predicts 12.9% default, observed
  14.9%), it **misses the pre-set AUC target by 0.0007**, and **declining loans on its score
  does not raise total profit**: even the riskiest decile returned 1.3% on funded dollars,
  because Lending Club's rates already rise with risk. The best frozen cutoff declines 0.5% of
  loans and adds 0.05 points of return.
- **Fairness flag:** under a strict decline-20% policy, the lowest-income quartile is approved at
  63% of the highest quartile's rate (the gap follows real risk, but needs a fair-lending
  justification).
- **Biggest lesson:** about 40 of the file's 151 columns are filled in after the loan starts and
  predict the outcome almost perfectly (for example `recoveries`, single-column AUC 0.93;
  `last_fico_range_*`, 0.90). A leaky model would look brilliant and be worthless at origination.
- **SAS:** not done. SAS Viya for Learners needs your SJSU account, so there is no SAS flow,
  screenshot or report. A SAS score-code export of the scorecard is provided and was **not run in SAS**.

## 2. Method and results by SEMMA stage

**Extension 0, framing.** Question, realized-cash-flow profit convention (declined loans earn 0;
undiscounted), and six pass/fail criteria (C1 to C6) fixed before modeling.

**S: Sample.** 2,260,701 rows filtered to **666,785 eligible loans**: final status only,
individual applications, issued 2010 or later, and **matured vintages only** (36-month loans issued
by Dec 2015, 60-month by Dec 2013), which removes the censoring bias of recent loans. Partitions by
issue date: 36-month train 2010 to 2013 (166,459), validation 2014 (162,570, sampled 25% to
40,643, stratified by year and outcome), out-of-time test 2015 (**all 282,787 loans, not sampled**).
A learning curve (validation AUC 0.625 at 2% of the data to 0.684 at 100%) had not flattened, so the
whole training population is used. The validation sample matches its population (KS at most 0.008,
PSI 0.0007). Imbalance (12.4%) is mild; class weighting did not help ranking and ruined Brier
(0.246 vs 0.114), so none is used.

**E: Explore.** Default rate rises from 5.6% (grade A) to 30.3% (G), from 8% to 18% across income
quintiles, and by purpose from 9% to 22%. The credit-bureau block exists only from 2013, so its
missingness would proxy the calendar. Information value: Lending Club's grade, sub-grade and rate
0.23 to 0.26; FICO 0.14 is the strongest application-only variable. Borrower clusters are weak
(silhouette at most 0.14), with one clear low-risk group.

**M: Modify.** A leakage audit of **all 151 columns** (78 known at origination, 41 not, plus joint,
identifier, text, Lending Club assessment and fairness-proxy columns). `zip_code` and `addr_state`
are kept out of the model. Variables chosen by information value (at least 0.02) and redundancy
(|Spearman| at most 0.8); everything fit on training loans only (`WoEBinner`, `TreePrep`), with by-hand
checks of bin edges and a weight of evidence.

**M: Model.** Baselines, WoE scorecards, tree, LightGBM, MLP, with and without Lending Club's grade
and rate, tuned on validation only. By the pre-set rule, **Model A = LightGBM on application and
bureau data** (validation AUC 0.678) and Model B (adds grade and rate, AUC 0.689) is a comparison only.

**A: Assess (test opened once).**

| Criterion | Result | Verdict |
|---|---|---|
| C1 AUC at least 0.68 and 0.02 above FICO-only | 0.6793; +0.0785 | **Fail** by 0.0007 |
| C2 KS at least 0.25 | 0.2598 | Pass |
| C3 slope 0.8 to 1.2; mean gap within 1.5 points | slope 0.955; gap 1.96 | **Fail** |
| C4 profit above accept-all, month-block CI excludes 0 | +0.051 points [0.042, 0.061] | Pass, economically negligible |
| C5 PSI at most 0.25; quarterly AUC within 0.03 | 0.007; within 0.006 | Pass |
| C6 approval ratios at least 0.8 | lowest income quartile 0.627 (strict policy) | **Flag** |

Lending Club's grade and rate add 0.013 AUC to the champion. A 60-month check scores AUC 0.602 on
2013 loans.

**Extension: deployment.** `score_loans.py` validates input (rejects bad rows with reasons) and scores
with the frozen champion, checked against the notebook on 600 loans; monitoring code (PSI, per-vintage
calibration gap, approval ratios) fires on the 2015 stream exactly where Assess found the calibration
gap; SAS score code generated for the scorecard.

## 3. Limits you should know

1. **Realized-cash-flow profit** is undiscounted, ignores fees, assumes declined loans earn 0, and uses
   only loans Lending Club already accepted (selection effects).
2. **Calibration drift:** default rates rose 13.7% to 14.9% between 2014 and 2015. Refresh calibration on
   recent vintages; the Platt fix on 2014 data only halves the gap and was diagnostic.
3. **Weak within-grade discrimination** (AUC 0.57 to 0.64 inside each grade) and under-prediction in
   grades D and E.
4. **Aggregate exposure:** the 2015 vintage's default rate was seen while planning (disclosed in
   `00_problem_framing.md`); no modeling choice used it.
5. **No protected-attribute data:** the fairness review uses slices and proxies.
6. **60-month loans** have a weaker, separate scorecard and are not deployed.
7. **SAS was not available.**

## 4. Experience and what I would change

Each stage was drafted by Claude Sonnet 5.5 and reviewed by GPT-5.6 Terra in a fresh context
under a "world-renowned SEMMA authority" persona, with at least two rounds each (scores below).

| Stage | Rounds | Scores |
|---|---|---|
| Framing and Sample | 3 | 7.5, 8.4, 9.7 |
| Explore | 2 | 8.3, 9.2 |
| Modify | 2 | 8.3, 9.0 |
| Model | 2 | 8.4, 9.4 |
| Assess | 2 | 8.8, 9.4 |
| Deployment extension and report | 3 | 8.5, 8.2, 9.5 |

What the critics caught that I would have shipped: calling the 2015 test "sealed" after seeing its
aggregate default rate (it is held out, not sealed); sampling the test by outcome; a KS of 0.023 for a constant predictor (a tie-handling bug);
an unreconciled sample count; an incomplete representativeness check; and a policy cutoff not tied to
one model.

What I would change: tune and choose cutoffs on return per funded dollar (not just total profit);
use the full test population from the start; calibrate on the latest mature vintage; use monotone WoE
bins; and apply for SAS access in week one.

**What a human still has to do:** read the notebook and be able to explain each cell, recompute two or
three numbers by hand (the notebook shows by-hand checks of bin edges and WoE), run it clean in Colab,
apply for SAS Viya for Learners and rerun the SAS flow if possible, publish the Medium article, record
the walkthrough video, and add the links.

## 5. Reproducing the work

```
pip install -r requirements.txt
# put accepted_2007_to_2018Q4.csv.gz in data/ (or set a Kaggle token in Colab secrets)
jupyter nbconvert --to notebook --execute notebooks/semma_lending_club.ipynb
python deployment/score_loans.py deployment/sample_applications.csv scored.csv rejects.csv
```

Seed 42. The first run reads the 2.2-million-row file in chunks (under a minute on a laptop) and caches
the eligible loans in `data/`.
