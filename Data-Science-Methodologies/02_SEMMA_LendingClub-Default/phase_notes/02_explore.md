# E: Explore

Draft v2 (after critic round 1). All analysis uses the 36-month **training and validation** loans only
(207,102 loans; train 166,459, validation 40,643). The out-of-time test is not touched.
Figures: `artifacts/figures/E_*.png`. Tables: `artifacts/information_value.csv`,
`coverage_by_vintage.csv`, `clusters_profile.csv`.

## E.1 Univariate distributions, missingness, outliers (training loans)

| Variable | Median | 1% | 99% | Max | Missing |
|---|---|---|---|---|---|
| Loan amount | 10,000 | 1,500 | 35,000 | 35,000 | 0% |
| Interest rate | 12.99 | 6.03 | 21.98 | 25.89 | 0% |
| Annual income | 60,000 | 17,509 | 226,003 | 7,141,778 | 0% |
| DTI | 15.89 | 1.35 | 33.40 | 34.99 | 0% |
| FICO (midpoint) | 692 | 662 | 802 | 847.5 | 0% |
| Revolving utilization | 58.3 | 0.9 | 97.3 | 140.4 | 0.07% |
| Credit history (years) | 13.7 | 4 | 37.7 | 65.4 | 0% |

Outliers: 146 loans with income above 500,000; 171 with revolving utilization above
100; no negative DTI or credit history; FICO starts at 662 (no one below 660). Skew is strong in income and revolving balance, which argues for logs and
bins.

**Missingness is a story about time.** Of 59 candidate origination columns, 39 have
under 95% coverage in the training loans. The pattern is almost entirely the issue year:
the credit-bureau block (33 fields such as `tot_cur_bal`, `bc_util`,
`num_tl_op_past_12m`) has 0% coverage for 2010 and 2011, 48% for 2012, and about
100% from 2013. Examples: `tot_hi_cred_lim` 0.00, 0.00, 0.48, 1.00, 1.00 for
2010 to 2014. Using these fields lets a model learn "present means later year".
The "months since last ..." fields are different: missing means "never happened".
Only `mths_since_last_delinq` (42% present) and `mths_since_last_record` (9%)
exist from 2010.

## E.2 Default rate by key variables (development loans)

| Variable | Pattern |
|---|---|
| Grade | A 5.6%, B 10.6%, C 16.1%, D 20.7%, E 23.7%, F 27.4%, G 30.3% (G has only 109 loans) |
| DTI band | up to 10: 10.0%, 10-20: 11.9%, 20-30: 15.2%, 30-40: 17.7% |
| Income quintile | lowest 17.6%, then 14.2%, 12.4%, 10.2%, highest 8.3% |
| Purpose | car 9.3% and credit card 10.8% lowest; small business 21.7%, renewable energy 18.7%, moving 16.7% highest |
| Home ownership | mortgage 10.5%, own 13.3%, rent 14.8% |
| Income verification | not verified 11.9%, source verified 13.0%, verified 13.4% (the reverse of what one expects; verification is likely applied to riskier applicants) |
| Vintage | 2010 9.9%, 2011 10.6%, 2012 13.6%, 2013 12.3%, 2014 13.7% |

The vintage drift matters: default rates rise from 10% to 14% across 2010 to 2014
even within the grades, so a model trained on early years will under-predict later.

## E.3 Association screening

Information value (IV) of each candidate on the training loans (71 candidates). Numeric
variables use 10 quantile bins plus a missing bin; categorical variables use each
category as its own bin, with categories under 0.5% of loans pooled into OTHER and missing
values as MISSING. Rules of thumb: under 0.02 useless, 0.02 to 0.1 weak,
0.1 to 0.3 medium, 0.3 to 0.5 strong, above 0.5 suspicious.

| Variable | IV | Note |
|---|---|---|
| `sub_grade` | 0.259 | Lending Club's own assessment |
| `int_rate` | 0.251 | Lending Club's own assessment |
| `grade` | 0.229 | Lending Club's own assessment |
| `fico` | 0.143 | strongest application-only variable |
| `annual_inc` / `log_income` | 0.095 | |
| `bc_open_to_buy`, `total_bc_limit`, `tot_hi_cred_lim` | 0.070, 0.060, 0.056 | bureau block, 73% to 83% training coverage |
| `loan_to_income`, `dti`, `revol_util` | 0.044, 0.041, 0.038 | |
| `home_ownership`, `purpose` | 0.032, 0.030 | |
| `credit_hist_years` | 0.019 | just under the 0.02 line |

**No variable has IV above 0.5**, so there is no sign of a leaked field among the
candidates (the post-origination columns were already excluded; see Modify). 43 of
the 71 candidates have IV below 0.02 (including `verification_status`,
`initial_list_status`). **Categorical association.** Cramer's V between the categorical candidates is small
(largest: `home_ownership` with `addr_state` 0.139, `purpose` with `grade` 0.110),
so they carry mostly separate information (`artifacts/categorical_association.csv`).
Against the outcome: `grade` 0.154, `purpose` 0.060, `home_ownership` 0.058,
`addr_state` 0.042, `verification_status` 0.013, `initial_list_status` 0.007.
Numeric redundancy: `loan_amnt` with `installment` (0.994),
`pub_rec` with `pub_rec_bankruptcies` (0.934), and the log versions of income and
amount duplicate their originals (1.000).

## E.4 Borrower clusters

k-means on twelve standardized borrower features (`log_income`, `dti`, `fico`,
`revol_util`, `log_loan_amnt`, `credit_hist_years`, `open_acc`, `total_acc`,
`inq_last_6mths`, `delinq_2yrs`, `pub_rec`, `revol_bal_to_income`; DTI, utilization and
counts capped at training quantiles, missing values set to the median), fit on the
training loans. The figure `artifacts/figures/E_clusters.png` shows 12,000 training loans on
the first two principal components of those features, coloured by cluster. **The structure is weak**: silhouette is 0.13, 0.12, 0.14, 0.11, 0.12, 0.12,
0.12 for k = 2 to 8. With k = 5:

| Cluster | Loans | Default (train) | Default (validation) | Profile |
|---|---|---|---|---|
| 4 | 28,946 | 6.8% | 7.3% | FICO 747, revolving utilization 24%, DTI 10 |
| 0 | 13,884 | 12.4% | 13.9% | income 75k, history 16 years |
| 2 | 47,268 | 12.2% | 12.8% | DTI 20, 14 open accounts, loans 15k |
| 3 | 14,644 | 13.1% | 15.0% | FICO 677 |
| 1 | 61,717 | 15.0% | 16.5% | income 45k, history 11 years, utilization 68% |

One clear group (high FICO, low utilization, 7% default) and a spread of similar
groups. Cluster default rates carry over to the validation year (ordering is the
same), so the groups are stable but add little beyond FICO and utilization.

## E.5 What each finding means for Modify

| Finding | Consequence |
|---|---|
| The bureau block exists only from 2013 | Use it only in an "extended" variant and compare it with the core variant; missing values get their own bin or indicator |
| Grade, sub-grade and rate dominate IV | Keep them out of the headline Model A; use them in Model B to measure what they add |
| Default rates drift upward by vintage | Time-ordered partitions; expect calibration drift; test stability by vintage |
| Strong right skew, a few extreme values | Clip at training quantiles (trees), bin (WoE) |
| Redundant pairs | Drop the lower-IV member of any pair with |Spearman| above 0.8 |
| 43 of the 71 candidates have IV below 0.02 (36 of the usable numeric ones) | Drop from the models |
| Income verification looks reversed | Do not trust it as a risk driver; it is dropped by IV |
| Clusters add little | Report them as exploratory; do not add them to the models |
