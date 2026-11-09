# What patterns separate attacks from normal traffic, and do they hold for attacks never seen in training? A KDD project

Final report. Data: **NSL-KDD** (Tavallaee, Bagheri, Lu and Ghorbani, 2009), Kaggle `hassan06/nslkdd`; license listed as Unknown, so the original
authors are credited. Notebook: `notebooks/kdd_nsl_kdd.ipynb`. Step notes: `phase_notes/`. Critic reviews: `critic_reviews/`.
The course's five KDD steps are mapped onto Fayyad, Piatetsky-Shapiro and Smyth's (1996) nine; step 9 is an **extension**.

**Read this first:** the traffic is simulated and derives from a 1998 capture. Nothing here is claimed for modern networks.

## 1. Summary

We characterized connection records with clustering, association rules, anomaly detection and classification, and checked every pattern on a test
that contains 17 attack types absent from training.

- **A few patterns are real knowledge and transfer:** a burst of half-open connections (`flag=S0` with high SYN-error rates) is a SYN flood (test confidence
  0.89 to 1.00); many ICMP echo requests across many hosts (`service=eco_i`, `srv_diff_host_rate=high`) is a ping sweep (0.98 to 1.00).
- **Most of what a random split promised does not hold.** A random validation split gives accuracy 0.999 and macro-F1 0.969. On `KDDTest+` the same model gets accuracy 0.780 and
  macro-F1 0.595 (0.505 on the harder `KDDTest-21`). Two measured causes: the class mix (R2L is 0.8% of training, 12.8% of the test) and novel attack types (accuracy on them: 0.287).
- **R2L is mostly missed.** The model finds 11.8% of R2L (2,487 of 2,885 are called normal) and 20.9% of U2R (14 of 67). `guess_passwd`: 53 training records, 1,231 test records, 0 found.
- **All four pre-registered knowledge goals fail** their conditions: K1 (clusters line up with categories: NMI 0.477 on training, 0.304 on test), K2 (R2L has no
  rule at the required support), K3 (anomaly novel-type recall 0.100 at the 1% threshold, although ROC-AUC is 0.958), K4 (novel-type recall 0.386, 0.014 short of 0.40, with macro-F1 0.595 passing).
- **Unsupervised detection ranks well but its 1% threshold is too strict.** Post hoc, 5% alarms on normal traffic would catch 67% of novel and 76% of known attack records.
- **Biggest lesson:** the test is built to differ from training, and a validation that ignores that (a random split) overstates performance by about 22 accuracy points.

## 2. Method and results by step

**Step 1, domain and goals.** Attack categories, feature groups, goals K1 to K5 with pass conditions, and the model-selection rule, all written before modeling.

**Step 2, Selection.** `KDDTrain+` (125,973), `KDDTest+` (22,544) and `KDDTest-21` (11,850, exactly the records of `KDDTest+` with difficulty at most 20). The
`difficulty` column is dropped. **17 attack types occur only in test** (3,750 records, 29.2% of its attacks). The test class mix differs strongly from training.

**Step 3, Preprocessing.** No missing values or duplicates; types validated against the ARFF documentation; `num_outbound_cmds` is constant and dropped; extreme `src_bytes`
values are mostly attacks, so no row is removed; labels grouped into five categories. Validation design: view **R** (random 80/20) and view **N** (six whole attack types held out).

**Step 4, Transformation.** `log1p` on eleven skewed fields (skewness of `src_bytes` 191 to 0.31), scaling, one-hot encoding: 121 features; mutual information and PCA (23 components
hold 95%); 189 discretized items for rules; every learned step refit inside each validation view.

**Steps 5 to 7, Data Mining.** k-means and DBSCAN; FP-Growth rules per category; three normal-only anomaly detectors; baseline, tree and boosted classifiers chosen by a pre-registered rule
on both views. Chosen: depth-6 tree, boosted model (15 leaves, balanced weights), PCA-reconstruction anomaly detector.

**Step 8, Interpretation (test scored once).**

| `KDDTest+` | Accuracy | Macro-F1 | R2L recall | U2R recall |
|---|---|---|---|---|
| baseline (logistic) | 0.741 | 0.519 | 0.028 | 0.119 |
| tree (depth 6) | 0.738 | 0.509 | 0.049 | 0.075 |
| **boosted** | **0.780** | **0.595** | 0.118 | 0.209 |

Attack-vs-normal recall on novel types: boosted 0.386, tree 0.179, baseline 0.311, anomaly detector (frozen threshold) 0.100, boosted-or-anomaly 0.415.

**Step 9 (extension), deployment.** A FastAPI service (validated input, category, anomaly score, matched rules, integrity check), tested through its HTTP interface and in a
**Docker** container, plus monitoring code. The monitoring demonstration shows the weakness: no retrain or alert-volume trigger fires on the test stream (one feature-drift watch message does) although R2L recall is 0.118.

## 3. Limits you should know

1. **1998-derived simulated traffic.** Fixed payload sizes of old attack tools (for example `src_bytes`) are artifacts and would not appear in modern attacks.
2. **Categories of the 17 test-only types are a convention**, not data.
3. **Small counts:** U2R 67 test records (recall interval 0.10 to 0.31); view N holds 9 and 7 records of two types; the U2R rules match 11 to 18 training records.
4. **The test was characterized in advance:** attack-type names and counts were inspected while planning (disclosed). It was held out from fitting, tuning, thresholds and rule selection.
5. **A selection-code slip** (the simplicity tie-break was not applied, which chose the depth-8 tree) was found and fixed; the test had been scored once under the wrong tree.
6. **The rules are training patterns**; the R2L ones describe `warezclient`, absent from the test.
7. Optional LSH was not done.

## 4. Experience and what I would change

Each step was drafted by Claude Sonnet 5.5 and reviewed by GPT-5.6 Terra in a fresh context under a "world-renowned KDD authority" persona, with at least two rounds each.

| Step | Rounds | Scores |
|---|---|---|
| Domain, Selection, Preprocessing, Transformation | 2 | 7.5, 9.0, 9.5 (after one fix) |
| Data Mining and Interpretation | 2 | 8.7, 9.0 |
| Deployment extension and report | 3 | 8.4, 9.4, sign-off after one text fix |

What the critics caught that I would have shipped: a validation design that leaked (a pipeline fit on all training data used inside validation); K1 judged on training only; a model-selection
rule without a written tie-break; a pruning step in the rule code that was missing from the written protocol; and an overclaim that payload attacks are "invisible".

What I would change: set the alarm threshold at a higher false-alarm budget chosen with analysts; reweight or oversample R2L; add payload-aware features; build a
validation set that mimics the test's class mix; and add a real labelled-feedback loop to monitoring.

**What a human still has to do:** read the notebook and explain each cell, recompute two or three numbers by hand (the notebook shows by-hand checks of the scaler and bin edges),
run it clean in Colab, publish the Medium article, record the walkthrough video, and add the links.

## 5. Reproducing the work

```
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute notebooks/kdd_nsl_kdd.ipynb      # data downloads from Kaggle if data/ is empty
cd deployment && uvicorn main:app --port 8000                              # or: docker build -t kdd-nsl-service . && docker run -p 8000:8000 kdd-nsl-service
```
Seed 42. Frozen bundle hash in `artifacts/frozen_policy.json`; test access log in `artifacts/test_access_log.json`.
