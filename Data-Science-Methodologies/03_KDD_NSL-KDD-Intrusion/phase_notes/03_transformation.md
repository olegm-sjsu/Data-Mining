# Step 4: Transformation

Draft v2 (after critic round 1). Numbers from notebook section "Transformation", `artifacts/mutual_information.csv`,
`metrics_transformation.json`, `figures/T_feature_reduction.png`.

## Log transform, scale, encode (one pipeline, fit on training rows only)

`kdd_features.KDDPrep` applies `log1p` to eleven skewed fields (`duration`, `src_bytes`, `dst_bytes`, `hot`,
`num_compromised`, `num_root`, `num_file_creations`, `count`, `srv_count`, `dst_host_count`, `dst_host_srv_count`), standardizes the 37
numeric columns (`num_outbound_cmds` dropped), and one-hot encodes `protocol_type` (3), `service` (70) and `flag` (11), giving
**121 prepared features** (37 numeric, 84 one-hot). Unseen categories at transform time become all-zero rows (none occur).

| Field | Skewness before | After `log1p` |
|---|---|---|
| `src_bytes` | 190.7 | 0.31 |
| `dst_bytes` | 290.1 | 0.49 |
| `count` | 1.51 | 0.12 |
| `duration` | 11.9 | 5.3 |
| `num_compromised` | 250.1 | 27.5 |

Fields such as `num_compromised` and `num_file_creations` stay skewed because most values are zero; they are
rare-event counts, and trees handle them. **Check by hand:** the mean and standard deviation of `log1p(src_bytes)` on
training (3.2297, 2.9822) equal the pipeline's stored values. Applied to `KDDTest+` without refitting, the scaled
`src_bytes` has mean 0.143, not 0, which shows that nothing was fit on test.

## Feature reduction: mutual information and PCA

Mutual information with the category (30,000 training records), top features: `src_bytes` 0.715, `diff_srv_rate` 0.509,
`dst_bytes` 0.461, `same_srv_rate` 0.459, `dst_host_diff_srv_rate` 0.454, `dst_host_srv_count` 0.421, `count` 0.416,
`dst_host_serror_rate` 0.401, `flag=SF` 0.373, `flag=S0` 0.363, `logged_in` 0.304. **29 of 121 features have MI below 0.001**
(mostly rare services). PCA on the prepared training matrix: **19 components hold 90% of the variance and 23 hold 95%**.

**Decisions (what is retained):** supervised models use all 121 prepared features (trees ignore weak ones, and
dropping rare services could remove the signal for a rare attack). **Clustering** (k-means, DBSCAN), which is sensitive
to noise and dimensionality, uses the **first 23 principal components** of the all-training fit. The **anomaly detectors do not use those components**: each fits its
own preprocessing and its own PCA (keeping 95% of the variance) on **normal records only** (Data Mining), so the normal-only condition of K3 holds.

## Discretization for association rules

`kdd_features.RuleDiscretizer` turns each connection into 189 items: categorical values, the four binary flags as `=1`, rate
columns as `zero`, `low` (up to 0.5) or `high`, and count columns as `zero`, `low`, `mid`, `high` using training tertiles of the
non-zero values (for example `src_bytes` cut at 187 and 306 bytes; `count` at 4 and 107). A connection has 36.4 items on average.
Nine items hold in over 99% of connections (such as `num_failed_logins=zero`) and are uninformative for rules; the mining step
excludes items that hold for more than 95% of the connections.

## Everything fit on train only, in a reusable pipeline

**Boundary (as in Preprocessing):** the numbers above come from a fit on all of `KDDTrain+`, for exploration and for the final model. When models
are compared on the validation views R and N, the same code is **refit on each view's fit rows** (scaling, one-hot categories, PCA, mutual information,
cut points), so no validation row influences any learned step.

`KDDPrep` and `RuleDiscretizer` are fit on `KDDTrain+` only, saved (`artifacts/preprocessing_objects.joblib`), reloaded and
checked to reproduce the transformed rows. The same code is imported by the API service, so training and serving cannot differ.
