# S: Sample

Draft v2 (after critic round 1). Numbers from notebook section S and `artifacts/` (`sample_funnel.csv`,
`partitions_population.csv`, `learning_curve.csv`, `sample_summary.csv`,
`sample_representativeness.csv`).

## S.1 Eligible population and what each filter removes

| Step | Rows |
|---|---|
| All rows in the file (`accepted_2007_to_2018Q4`) | 2,260,701 |
| Status blank | -33 |
| Still open (Current, Late, In Grace Period) | -912,569 |
| Legacy "does not meet the credit policy" statuses | -2,749 |
| **= loans with a final status** (Fully Paid, Charged Off, Default) | 1,345,350 |
| Joint applications (different fields, few before 2015) | -25,806 |
| Issued before 2010 (small, different regime) | -6,529 |
| **Not matured** (36-month issued after Dec 2015, 60-month after Dec 2013) | -646,230 |
| **= eligible population** | **666,785** |

The three status filters are counted independently on all rows and sum to
915,351 (2,260,701 - 915,351 = 1,345,350). **Censoring:** a loan issued in 2016
may have ended early, and early endings skew toward charge-offs. Keeping only
vintages that could have run to term by December 2018 removes that bias. The
maturity rule leaves 36-month loans issued 2010 to 2015 and 60-month loans issued
2010 to 2013. "Default" (40 loans) is counted as a default.

## S.2 Partitions by issue date

| Term | Partition | Issue dates | Loans | Default rate |
|---|---|---|---|---|
| 36 months (main) | train | 2010-01 to 2013-12 | 166,459 | 12.39% |
| | validation | 2014 | 162,570 | 13.73% |
| | **out-of-time test** | 2015 | 282,787 | held out |
| 60 months (secondary) | train | 2010-05 to 2011-12 | 10,690 | 22.85% |
| | validation | 2012 | 9,897 | 27.70% |
| | test | 2013 | 34,382 | held out |

Terms are modeled separately: 60-month loans have a different risk level and fewer
mature vintages. The first 60-month loan in the file is issued in May 2010; there
are none earlier, so 2010-05 is not a filter.

## S.3 Learning curve and sample size

A quick LightGBM (application fields, no grade or rate) trained on growing
stratified fractions of the 36-month training population and scored on the 2014
validation sample:

| Fraction | Training rows | Validation ROC-AUC |
|---|---|---|
| 2% | 3,329 | 0.6246 |
| 5% | 8,322 | 0.6336 |
| 10% | 16,645 | 0.6458 |
| 25% | 41,614 | 0.6650 |
| 50% | 83,231 | 0.6762 |
| 100% | 166,459 | 0.6840 |

**The curve has not flattened** (the last doubling still adds 0.008 AUC). The
rule was "the smallest fraction within 0.002 AUC of the best", which selects
100%. So the training partition is **not subsampled**: all 166,459 loans are
used. Only the validation partition is sampled (25%). The out-of-time test is **not
sampled at all**: all 282,787 loans of 2015 are used, so no outcome-based selection
touches it.

## S.4 The sample

| Term | Partition | Sample rows | Population | Fraction |
|---|---|---|---|---|
| 36 months | train | 166,459 | 166,459 | 100% |
| 36 months | validation | 40,643 | 162,570 | 25% |
| 36 months | test | 282,787 | 282,787 | 100% (not sampled) |
| 60 months | all three | 54,969 | 54,969 | 100% |

Total 544,858 of 666,785 loans. The validation sample is **stratified by term, issue
year and outcome** with the same inclusion probability (0.25) in every stratum, so
it keeps the population default rate (13.73%) and needs no weights. Validation is
development data, so using its labels for stratification is harmless. Neither the
training nor the test partition is sampled.

**Imbalance.** The 36-month training default rate is 12.4%, mild. No oversampling
is used, so no probability correction is needed. A class-weight check is in the
Model stage.

## S.5 Is the sample representative?

Only the validation partition is a true sample, so it is the one checked against its
population (162,570 loans), on features only. **All 59 numeric origination columns**
were tested (`CORE_NUM` in the notebook; none was too sparse to test), and **six
categorical columns** (`grade`, `purpose`, `home_ownership`, `verification_status`,
`emp_length`, `addr_state`).

| Check | Result |
|---|---|
| Numeric KS | maximum 0.0082 (`mths_since_recent_bc_dlq`); all 59 below 0.01 |
| Numeric PSI | maximum 0.0007 |
| Missing share, sample vs population | for example `mths_since_recent_bc_dlq` 73.1% vs 73.3% (full per-column table in `sample_representativeness_numeric_36m_val.csv`) |
| Categorical chi-square (Cramer's V) | `grade` 0.0041, `purpose` 0.0063, `home_ownership` 0.0031, `verification_status` 0.0022, `emp_length` 0.0051, `addr_state` 0.0114; all p-values above 0.59 |
| By issue year | validation is one year (2014); the maximum KS on five key columns is 0.0051 |

The sample cannot be told apart from its population. Categorical tables with
category counts are in `sample_representativeness_categorical_36m_val.csv`.

## S.6 Output

Partitioned datasets: 36-month `train` 166,459, `val` 40,643, `test` 282,787 (full 2015 population); 60-month
10,690, 9,897, 34,382. Assertions check that train < validation < test in time.

## Open items

1. The learning curve says more training data helps, so the full training population is used.
2. The test vintage's higher default rate (about 14.9% against 12.4%) means calibration drift is expected (criterion C3).
