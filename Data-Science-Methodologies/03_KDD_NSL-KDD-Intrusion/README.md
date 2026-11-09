# Project 3: KDD, Network Intrusion Knowledge Discovery

| | |
|---|---|
| **Question** | What patterns separate attacks from normal traffic, and do they hold on attack types never seen in training? |
| **Dataset** | [NSL-KDD](https://www.kaggle.com/datasets/hassan06/nslkdd). Use `KDDTrain+`, `KDDTest+`, and `KDDTest-21`. About 126k training and 22.5k test records |
| **Source to cite** | Tavallaee et al., "A detailed analysis of the KDD CUP 99 data set", 2009. The Kaggle page lists the license as Unknown, so credit the original authors |
| **Target** | `label`, grouped into normal, DoS, Probe, R2L, U2R |
| **Deployment** | FastAPI service in Docker |
| **Colab** | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/olegm-sjsu/Data-Mining/blob/main/Data-Science-Methodologies/03_KDD_NSL-KDD-Intrusion/notebooks/kdd_nsl_kdd.ipynb) (works after the push) |
| **Medium** | _add link_ |
| **YouTube** | _add link_ |

## Status

All KDD steps plus the extension are drafted, run and reviewed (see `critic_reviews/`). Read [REPORT.md](REPORT.md) first. **All four pre-registered knowledge goals failed their
conditions** and the report says so. Optional LSH was not done.

**Still to do by you:**

- [ ] Read the notebook and be able to explain each cell; recompute two or three numbers by hand
- [ ] Push to GitHub, then run the notebook clean in Colab (needs a `KAGGLE_API_TOKEN` secret)
- [ ] Publish `article/medium_article.md` on Medium (fill the links) and add the URL here
- [ ] Record the YouTube walkthrough (`article/presentation.html` is the slide deck) and add the URL here

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

- [x] Domain background: attack categories and what each feature group measures
- [x] Knowledge goals, stated as questions, and what would count as novel, valid, useful, and understandable

### 1. Selection, `01_selection.md` (Fayyad step 2)

- [x] Which files, records, and features form the target data, and why
- [x] Data provenance, including how NSL-KDD differs from KDD Cup 99
- [x] Which attack types appear only in test

### 2. Preprocessing, `02_preprocessing.md` (Fayyad step 3)

- [x] Load with the correct column names and types, and validate against the documentation
- [x] Check missing values, duplicates, invalid values, constant columns, and label cleanup
- [x] Outlier analysis with a decision
- [x] Group labels into the attack categories

### 3. Transformation, `03_transformation.md` (Fayyad step 4)

- [x] Log transform skewed values, scale, and encode `protocol_type`, `service`, and `flag`
- [x] Feature reduction using PCA or mutual information, and show what was retained
- [x] Discretize features for association rule mining
- [x] Everything fit on train only, in a reusable pipeline

### 4. Data Mining, `04_data_mining.md` (Fayyad steps 5 to 7)

- [x] Match each knowledge goal to a method, and explain why
- [x] Clustering (k-means plus one density-based method): do clusters line up with attack categories? Visualize them
- [x] Association rules (FP-Growth): rules that characterize each attack category
- [x] Anomaly detection trained on normal traffic only (Isolation Forest or similar)
- [x] Classification with a baseline, interpretable tree, and boosted model
- [ ] Optional: LSH for finding similar connections, as in the course deck's prompt (not done)

### 5. Interpretation/Evaluation, `05_interpretation.md` (Fayyad step 8)

- [x] Evaluate per class on `KDDTest+` and `KDDTest-21`, with macro-F1 and confusion matrices
- [x] Results on known versus novel attack types, supervised versus anomaly detection
- [x] Turn the patterns into plain-language knowledge, for example the top rules with support, confidence, and lift
- [x] Check each pattern against security domain knowledge. Mark what is real and what is a dataset artifact
- [x] Return to earlier steps where results demand it, and document the loop

### Using the knowledge (extension), `06_deployment.md` (Fayyad step 9)

- [x] FastAPI service in `deployment/` returning attack category, anomaly score, and matched rules, packaged with Docker
- [x] Monitoring plan: drift, alert volume, feedback from analysts, retraining
- [x] `REPORT.md`, including limits of this 1998-derived data
