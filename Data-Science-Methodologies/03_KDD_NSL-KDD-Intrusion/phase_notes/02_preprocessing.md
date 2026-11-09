# Step 3: Preprocessing

Draft v2 (after critic round 1). Numbers from notebook section "Preprocessing", `artifacts/metrics_preprocessing.json`,
`attack_type_mapping.csv`, `validation_indices.npz`.

## Loading and validation against the documentation

The `.txt` files have no header, so the 41 feature names come from the ARFF documentation, plus `label` and
`difficulty`. The ARFF header declares 42 attributes (41 features and a two-valued `class`: normal or anomaly; attack types
are only in the `.txt` files). Check: every nominal feature's observed values (3 protocols, 70 services, 11 flags and four
binary columns) are inside the declared sets, across train, `KDDTest+` and `KDDTest-21`; numeric columns are numeric.
Types in training: 23 integer, 15 float, 3 string columns.

## Missing values, duplicates, invalid values, constants, labels

| Check | Result |
|---|---|
| Missing values | 0 in all three files |
| Duplicate rows (41 features and label) | 0 in all three (NSL-KDD removed duplicates) |
| Rates outside [0, 1]; negative numerics | none |
| Binary columns not 0 or 1 | none |
| Label whitespace or case problems | none |
| Constant columns | `num_outbound_cmds` (dropped from the models) |
| Near-constant (one value in at least 99.9%) | `land`, `urgent`, `num_failed_logins`, `su_attempted`, `num_shells`, `is_host_login` (kept: they are rare but meaningful for specific attacks) |
| Categories in test that never appear in train | none (for `protocol_type`, `service`, `flag`) |

## Outlier analysis and decision

Skewness in training: `src_bytes` 190.7 and `dst_bytes` 290.1 (maximum 1.4 billion), `duration` 11.9, `num_root` 236.9,
`hot` 12.6. The extreme values are **often the attacks**: 76.4% of records above the 99.9th percentile of
`src_bytes` are attacks (base rate 46.5%), 60 of the 89 extreme records are R2L. Deleting outliers would delete the signal.
**Decision:** keep every row; apply `log1p` to the skewed counts and byte fields (Transformation). Tree models are not affected by monotone transforms.

## Group the labels into the five categories

| Category | `KDDTrain+` | `KDDTest+` |
|---|---|---|
| normal | 67,343 | 9,711 |
| DoS | 45,927 | 7,460 |
| Probe | 11,656 | 2,421 |
| R2L | 995 | 2,885 |
| U2R | 52 | 67 (30 of them novel types) |

The mapping of every attack type is in `artifacts/attack_type_mapping.csv`. It follows the widely used convention; the
assignment of the test-only types is a convention, not data.

## Validation design (fixed before modeling, built from training data only)

A single random split would repeat the problem this project is about, so two views are used:

| View | Fit rows | Validation rows | Measures |
|---|---|---|---|
| **R**: stratified random 80/20 of `KDDTrain+` by attack type | 100,778 | 25,195 | known attack types, training class mix (optimistic) |
| **N**: whole attack types held out | 108,257 | 17,716 | recall on types never seen in fitting |

N holds out `back`, `teardrop` (DoS), `nmap` (Probe), `warezclient`, `multihop` (R2L) and `loadmodule` (U2R), chosen in
advance and not changed. Its validation data is those types (1,493 `nmap`, 956 `back`, 892 `teardrop`, 890 `warezclient`, 9
`loadmodule`, 7 `multihop`) plus 13,469 held-out normal records. Final models are refit on all of `KDDTrain+` with the
hyperparameters chosen by the **model-selection rule in `00_domain_and_goals.md`** (primary metric macro-F1 on R, then attack-vs-normal recall on N
among candidates within 0.02 of the best, simpler model on overlapping bootstrap intervals). N is read in aggregate with counts per type.
The test files are used for nothing before Interpretation.

**Leakage boundary.** Inside each validation view every learned step (log-scaling statistics, one-hot categories, PCA, mutual-information
ranking, discretization cut points, high-prevalence item filter) is **refit on that view's fit rows only**. The pipeline fit on all of
`KDDTrain+` (Transformation) is used for exploration and for the final refit and test, never to score a validation view.

## Open items

1. `loadmodule` and `multihop` have 9 and 7 records, so U2R and part of R2L in N are statistically thin.
