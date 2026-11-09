# M: Modify

Draft v2 (after critic round 1). Numbers from notebook section M and `artifacts/leakage_audit_all_columns.csv`
(all 151 columns), `model_features.json`, `information_value.csv`. Training and
validation loans only; the test is untouched.

## M.1 Leakage audit of every column

All 151 columns are classified (the notebook asserts that none is left unclassified;
the full table with reason, missing share, coverage by era, single-column AUC and IV
is the CSV above).

| Class | Columns | Decision |
|---|---|---|
| Known at origination | 78 | candidates (see coverage below) |
| **Not known at origination** | 41 | excluded |
| Joint-application fields | 16 | excluded (joint loans were removed) |
| Lending Club's own assessment (`grade`, `sub_grade`, `int_rate`, `installment`) | 4 | **only in Model B** |
| Meta (`loan_status`, `issue_d`, `policy_code`, `application_type`) | 4 | target, split date, constants |
| Identifiers | 3 | excluded |
| Free text (`emp_title`, `desc`, `title`) | 3 | excluded |
| Fairness proxies (`zip_code`, `addr_state`) | 2 | **kept out of the model**, used only in the fairness review |

**The 41 post-origination columns** are the payment history, balances, recoveries,
fees, last-payment dates, refreshed FICO, the `hardship_*`, `settlement_*` and
`debt_settlement_*` groups, and `funded_amnt_inv`. `funded_amnt`, `total_pymnt` and
`recoveries` are used only to compute realized profit in Assess. Evidence of how
strongly they leak (single-column ROC-AUC on development loans, where 0.5 is no
signal): `recoveries` **0.934**, `last_fico_range_high/low` **0.104** (equivalent
to 0.896), `collection_recovery_fee` 0.866, `total_rec_prncp` 0.170 (0.830),
`last_pymnt_amnt` 0.234 (0.766), `total_pymnt` 0.274 (0.726). Any of these would
make a model look excellent and be worthless at origination.

**Coverage of the 78 origination fields.** 14 expanded-bureau fields introduced
around 2015 (`open_acc_6m`, `il_util`, `all_util`, `inq_last_12m` and so on) are
known at origination but absent from the training vintages, so they are excluded.
The credit-bureau block has 0% coverage for 2010 and 2011, 48% for 2012 and about
100% from 2013 (Explore E.1).

## M.2 Missing values, outliers, transforms (justified per group)

The field-level decision for each of the 78 origination fields is in the column
`missing_treatment` of `artifacts/leakage_audit_all_columns.csv`. Counts: 24 complete
(none needed); 33 bureau-block fields (A-extended only; missing bin or indicator);
14 expanded-bureau fields absent in training (excluded); 4 calendar-dependent "months
since" fields not selected (`mths_since_last_major_derog`, `mths_since_recent_bc_dlq`,
`mths_since_recent_inq`, `mths_since_recent_revol_delinq`); 2 structural-missing fields
(`mths_since_last_delinq`, `mths_since_last_record`: own bin or indicator); 1 rare-missing
field (`emp_length`, 4.5% missing: own missing bin or value). Group rationale:

| Group | Treatment | Reason |
|---|---|---|
| Fields complete in all vintages (loan amount, income, DTI, FICO, accounts, inquiries) | none needed | 100% coverage |
| `revol_util` (0.07% missing) | own missing bin (scorecard) or missing value (trees) | too rare to matter |
| "Months since last ..." (`mths_since_last_delinq`, `mths_since_last_record`) | missing = "never happened": own bin or indicator | informative; neither passed the IV filter, so neither is in the final models |
| Bureau block (33 fields, calendar-dependent) | used **only** in the extended variant, missing bin (scorecard) or indicator plus native missing handling (trees) | missingness otherwise proxies the issue year |
| Outliers | clip at the training 0.1st and 99.9th percentile (trees); quantile bins (scorecard) | income to 7.1 million, utilization to 140 |
| Transforms | log income, log loan amount, credit history years, loan-to-income, revolving balance to income, FICO midpoint, employment years, WoE bins | skew and nonlinearity |

## M.3 Variable selection and redundancy removal

Rules, fixed in advance: keep a variable if its training IV is at least 0.02, and
drop the lower-IV member of any pair with |Spearman| above 0.8.
`int_rate` and `installment` are never in the application-only variants.

| Variant | Numeric | Categorical | Variables |
|---|---|---|---|
| **A-core** | 6 | 2 | `fico`, `annual_inc`, `loan_to_income`, `dti`, `revol_util`, `inq_last_6mths`; `home_ownership`, `purpose` |
| **A-extended** | 17 | 2 | A-core plus 11 bureau-block fields: `bc_open_to_buy`, `total_bc_limit`, `tot_hi_cred_lim`, `acc_open_past_24mths`, `mort_acc`, `percent_bc_gt_75`, `num_tl_op_past_12m`, `mths_since_recent_bc`, `mo_sin_rcnt_tl`, `mo_sin_rcnt_rev_tl_op`, `mo_sin_old_rev_tl_op` |
| **B** | 18 | 4 | A-extended plus `int_rate`, `grade`, `sub_grade` |

36 candidate numeric fields were dropped for IV below 0.02. `verification_status`
and `initial_list_status` were dropped for the same reason. `credit_hist_years`
(IV 0.0187) missed the line and is out; that is a rule, not a belief that history
length is irrelevant. `annual_inc` and `log_income` tie on IV (0.095), so the first
is kept.

## M.4 Everything fit on train only, in reusable pipelines

`lc_features.WoEBinner` (scorecard) and `lc_features.TreePrep` (trees, boosted) learn
bin edges, WoE values, clip bounds, category codes and missing-indicator columns from
the training partition only, and are used unchanged for validation, test and batch
scoring. Checks (all print in the notebook):

- FICO bin edges fit on train equal a by-hand quantile computation
  (667, 672, 682, 687, 692, 702, 712, 722, 747) and differ from the edges one would
  get if validation rows had been included (677 and 742 differ), so validation did
  not leak into the fit.
- WoE of the lowest FICO bin: pipeline -0.4709, by hand -0.4709.
- `TreePrep` for the extended variant produces 30 features including 11 missing
  indicators.
- Partition order assertions pass (train < validation < test in time).

Limit: WoE bins are plain quantile bins and are not forced to be monotone; the
scorecard table is therefore shown to the reader, not assumed to be monotone.

## Open items

1. Whether the bureau block helps despite its calendar dependence is decided in Model.
2. Monotone binning is a possible refinement, not done here.
