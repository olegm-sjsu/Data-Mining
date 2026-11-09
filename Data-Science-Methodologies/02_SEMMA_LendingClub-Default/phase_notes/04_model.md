# M: Model

Draft v2 (after critic round 1). All fitting uses the 36-month **training** loans (166,459, issued 2010-2013);
all tuning uses the **2014 validation** sample (40,643). The out-of-time test is held out
and has not been scored. Numbers from notebook section M and
`artifacts/model_comparison_validation.csv`, `frozen_policy.json`.

## M.1 Candidates and the choice rule

| Family | Variants |
|---|---|
| Baselines | majority (prior); FICO-only logistic regression on WoE bins (the comparison needed by criterion C1) |
| WoE logistic **scorecard** | A-core (8 inputs), A-extended (19), B (22: A-extended plus `int_rate`, `grade`, `sub_grade`) |
| Decision tree | A-extended |
| **LightGBM** | A-core, A-extended, B |
| Neural network | small MLP (32, 16 units) on the A-extended WoE inputs |

**Which model is the claim?** **Model A** (application and credit-bureau data known at
origination, no Lending Club grade, sub-grade or rate) is the headline and the model that
is frozen and deployed. **Model B** is reported only to show what Lending Club's own grade
and rate add (and what an investor trusting the grade would have); it is never deployed.

**Choice rule, fixed before the results:** within a variant family take the best validation
ROC-AUC, but prefer the scorecard if it is within 0.005 AUC of the best (credit-risk
practice values transparency).

SAS: the same flow would be Data, Partition, Impute, Regression, Decision Tree, Neural
Network and Model Comparison nodes. SAS access was not available for this project (the
SAS Viya for Learners application needs your SJSU account), so only the Python flow was
built. A SAS score-code export is generated in the deployment extension.

## M.2 to M.4 Validation results

| Model | ROC-AUC | PR-AUC | KS | Brier | Tuned setting |
|---|---|---|---|---|---|
| Majority (prior) | 0.5000 | 0.137 | 0.000 | 0.1186 | n/a |
| FICO-only logistic (WoE) | 0.5964 | 0.171 | 0.149 | 0.1168 | n/a |
| Scorecard A-core | 0.6489 | 0.217 | 0.218 | 0.1146 | C = 0.01 |
| Scorecard A-extended | 0.6641 | 0.229 | 0.243 | 0.1136 | C = 0.01 |
| Decision tree A-extended | 0.6426 | 0.210 | 0.213 | 0.1151 | depth 8, min leaf 200 |
| LightGBM A-core | 0.6555 | 0.223 | 0.230 | 0.1142 | 15 leaves, min leaf 100, 241 trees |
| **LightGBM A-extended** | **0.6784** | 0.241 | 0.264 | 0.1128 | 31 leaves, min leaf 300, 373 trees |
| Neural net (MLP) A-extended | 0.6619 | 0.225 | 0.237 | 0.1139 | hidden (32, 16), early stopping |
| Scorecard B | 0.6812 | 0.241 | 0.269 | 0.1127 | C = 0.01 |
| **LightGBM B** | **0.6888** | 0.250 | 0.276 | 0.1122 | 31 leaves, min leaf 300, 264 trees |

Tuning: scorecard C in {0.01, 0.1, 1, 10}; tree depth in {4, 6, 8} and min leaf in {200,
500}; LightGBM four (leaves, min leaf) pairs with early stopping on the validation AUC
(learning rate 0.03, L2 5); MLP early stopping on a split inside the training loans. All
on validation only.

Reading the table:

- Credit information matters: FICO alone gives 0.596; the application variables 0.649 to
  0.678; adding Lending Club's own grade and rate lifts the same learners by about
  0.010 (scorecard 0.6641 to 0.6812, LightGBM 0.6784 to 0.6888).
- The calendar-dependent **bureau block helps**: A-core to A-extended adds 0.015 (scorecard)
  and 0.023 (LightGBM) AUC.
- LightGBM leads the scorecard by 0.014 (A-extended) and 0.008 (B). The tree and the MLP do
  not beat the scorecard.
- **Imbalance check.** Class-weighting the A-extended scorecard leaves ROC-AUC unchanged
  (0.6638 vs 0.6641) and ruins the probabilities (Brier 0.246 vs 0.1136), so no
  reweighting is used.

## M.5 Champions and the profit cutoff

By the rule: **Model A champion = LightGBM A-extended** (it beats the best scorecard by
0.014, more than 0.005); **Model B champion = LightGBM B** (0.008 more than the scorecard).

**Profit cutoff on validation, for the Model A champion (LightGBM A-extended)** (realized cash flow `total_pymnt - funded_amnt`, used here
only to tune a decision threshold): accepting every loan earns 46.65 million on 513.2
million funded (**9.09%**). The best total-profit cutoff is PD <= 0.38, which accepts
99.6% of loans and earns 46.72 million on 511.3 million funded (**9.14%**), about
+0.05 percentage points (+0.07 million on 513 million funded). That is **validation-only and
well within noise**; it is not operational evidence until the out-of-time test confirms it. Return by predicted-risk decile (0 = safest): 8.9%, 10.0%, 10.3%,
10.3%, 10.4%, 9.9%, 9.9%, 8.6%, 7.3%, **5.0%**. **Every decile is profitable.** Lending Club's
rates rise with risk, so even the riskiest tenth returned money on average, and
declining loans on predicted default alone barely improves profit. Criterion C4 is
therefore expected to be hard to pass; the Assess stage tests it.

## M.6 Secondary check: 60-month loans, modeled separately

A scorecard (A-core inputs) trained on 10,690 60-month loans (2010-2011) scores
ROC-AUC 0.629 and KS 0.181 on 2012 validation loans (9,897). It is a secondary check and is
not deployed (only 55,000 eligible 60-month loans and four mature vintages).

## M.7 Frozen before the test

The frozen policy is **Model A champion (LightGBM A-extended) with the cutoff PD <= 0.38**. All
candidates, both champions and the cutoff are saved (`artifacts/frozen/candidates.joblib`,
hash in `artifacts/frozen_policy.json`, `test_opened: false`). Assess scores each frozen
candidate once.

## Open items

1. **Fairness** is not settled by excluding `zip_code` and `addr_state`: Assess checks proxies with
   approval-rate and error slices by state and income band (criterion C6).
2. C1 requires 0.68 out-of-time for Model A; the validation AUC of the champion is 0.678, so the
   test result will be close to the line.
3. The test is held out, but the 2015 vintage's higher default rate was seen in aggregate
   during planning (see `00_problem_framing.md`).
