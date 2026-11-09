# Project 3: KDD, Network Intrusion Knowledge Discovery

| | |
|---|---|
| **Question** | What patterns separate attacks from normal traffic, and do they hold on attack types never seen in training? |
| **Dataset** | [NSL-KDD](https://www.kaggle.com/datasets/hassan06/nslkdd). Use `KDDTrain+`, `KDDTest+`, and `KDDTest-21`. About 126k training and 22.5k test records |
| **Source to cite** | Tavallaee et al., "A detailed analysis of the KDD CUP 99 data set", 2009. The Kaggle page lists the license as Unknown, so credit the original authors |
| **Target** | `label`, grouped into normal, DoS, Probe, R2L, U2R |
| **Deployment** | FastAPI service in Docker |
| **Colab** | _add badge_ |
| **Medium** | _add link_ |
| **YouTube** | _add link_ |

## Pitfalls to hand the LLM and check yourself

- **The last column is `difficulty`,** which was derived from how classifiers performed on each record. Drop it from the features.
- **The test set contains attack types that are not in train.** That is the point of this project, so report novel-attack results separately. Never fit anything on test.
- **Train and test have different class mixes.** Random cross-validation on train will look far better than the test result. Explain the gap.
- **Rare classes:** U2R is tiny. Accuracy hides it, so report per-class recall and macro metrics.
- **Heavy skew:** `src_bytes` and `dst_bytes` need log transforms. `num_outbound_cmds` is constant.
- **Age of the data:** the traffic is simulated and derives from a 1998 capture. Do not claim the patterns apply to modern networks.

## Checklist

KDD here uses the course's five steps, with Fayyad et al.'s (1996) fuller nine
step process mapped in. Write each output in `phase_notes/`, and save at least
two critic rounds per phase.

### Before Selection, `00_domain_and_goals.md` (Fayyad step 1)

- [ ] Domain background: attack categories and what each feature group measures
- [ ] Knowledge goals, stated as questions, and what would count as novel, valid, useful, and understandable

### 1. Selection, `01_selection.md` (Fayyad step 2)

- [ ] Which files, records, and features form the target data, and why
- [ ] Data provenance, including how NSL-KDD differs from KDD Cup 99
- [ ] Which attack types appear only in test

### 2. Preprocessing, `02_preprocessing.md` (Fayyad step 3)

- [ ] Load with the correct column names and types, and validate against the documentation
- [ ] Check missing values, duplicates, invalid values, constant columns, and label cleanup
- [ ] Outlier analysis with a decision
- [ ] Group labels into the attack categories

### 3. Transformation, `03_transformation.md` (Fayyad step 4)

- [ ] Log transform skewed values, scale, and encode `protocol_type`, `service`, and `flag`
- [ ] Feature reduction using PCA or mutual information, and show what was retained
- [ ] Discretize features for association rule mining
- [ ] Everything fit on train only, in a reusable pipeline

### 4. Data Mining, `04_data_mining.md` (Fayyad steps 5 to 7)

- [ ] Match each knowledge goal to a method, and explain why
- [ ] Clustering (k-means plus one density-based method): do clusters line up with attack categories? Visualize them
- [ ] Association rules (FP-Growth): rules that characterize each attack category
- [ ] Anomaly detection trained on normal traffic only (Isolation Forest or similar)
- [ ] Classification with a baseline, interpretable tree, and boosted model
- [ ] Optional: LSH for finding similar connections, as in the course deck's prompt

### 5. Interpretation/Evaluation, `05_interpretation.md` (Fayyad step 8)

- [ ] Evaluate per class on `KDDTest+` and `KDDTest-21`, with macro-F1 and confusion matrices
- [ ] Results on known versus novel attack types, supervised versus anomaly detection
- [ ] Turn the patterns into plain-language knowledge, for example the top rules with support, confidence, and lift
- [ ] Check each pattern against security domain knowledge. Mark what is real and what is a dataset artifact
- [ ] Return to earlier steps where results demand it, and document the loop

### Using the knowledge (extension), `06_deployment.md` (Fayyad step 9)

- [ ] FastAPI service in `deployment/` returning attack category, anomaly score, and matched rules, packaged with Docker
- [ ] Monitoring plan: drift, alert volume, feedback from analysts, retraining
- [ ] `REPORT.md`, including limits of this 1998-derived data
