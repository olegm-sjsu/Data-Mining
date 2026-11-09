# Project 2: SEMMA, Lending Club Loan Default

| | |
|---|---|
| **Question** | Which loans will charge off, using only what is known at origination? |
| **Dataset** | [All Lending Club loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club), CC0, about 650 MB compressed, about 2.2M accepted loans. Use `accepted_2007_to_2018Q4` |
| **Target** | From `loan_status`: Charged Off or Default is 1, Fully Paid is 0. Exclude loans still open |
| **Tools** | Python in Colab, plus a SAS demo if available |
| **Deployment** | Batch scoring script, plus SAS score code if available |
| **Colab** | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/olegm-sjsu/Data-Mining/blob/main/Data-Science-Methodologies/02_SEMMA_LendingClub-Default/notebooks/semma_lending_club.ipynb) (works after the push) |
| **Medium** | _add link_ |
| **YouTube** | _add link_ |

## Status

All SEMMA stages plus the two extensions are drafted, run and reviewed (see `critic_reviews/`).
Read [REPORT.md](REPORT.md) first. **SAS was not available**: no SAS flow, screenshots or report;
a SAS score-code export is generated and has not been run.

**Still to do by you:**

- [ ] Read the notebook and be able to explain each cell; recompute two or three numbers by hand
- [ ] Apply for SAS Viya for Learners with your SJSU email; if access arrives, rebuild the flow and add screenshots to `artifacts/`
- [ ] Push to GitHub, then run the notebook clean in Colab (needs a `KAGGLE_API_TOKEN` secret)
- [ ] Publish `article/medium_article.md` on Medium (fill the links) and add the URL here
- [ ] Record the YouTube walkthrough (`article/presentation.html` is the slide deck) and add the URL here

## Pitfalls to hand the LLM and check yourself

- **Post-origination leakage:** the file has many columns filled in after the loan starts, including `total_pymnt*`, `recoveries`, `collection_recovery_fee`, `out_prncp*`, `last_pymnt_*`, `last_fico_*`, and the `hardship_*` and `settlement_*` groups. Go through all columns and classify each as known or not known at origination.
- **Censoring:** recent loans have not had time to finish, and early endings skew toward charge-offs. Keep only vintages that could have matured by the end of the data, and handle 36-month and 60-month terms separately.
- **Time drift:** train on earlier vintages and test on later ones.
- **Fairness:** `zip_code` and `addr_state` can act as proxies for protected groups. Lending also has fair-lending rules, so justify whether these stay in.
- **`grade` and `int_rate`** are Lending Club's own risk assessment. Compare models with and without them, and be clear about which you are claiming.

## Checklist

SEMMA comes from SAS and is defined as a way to organize Enterprise Miner's
tools. It has no business or deployment phase, so those are labeled as
extensions below. Write each output in `phase_notes/`, and save at least two
critic rounds per phase.

### Before Sample (extension), `00_problem_framing.md`

- [x] Short problem statement, loss assumptions, and success criteria, marked as outside SEMMA itself

### S: Sample, `01_sample.md`

- [x] Define the eligible population, and count what each filter removes
- [x] Draw a stratified sample by vintage, term, and outcome, and justify the size
- [x] Prove it is representative against the full eligible population (KS or PSI on numeric columns, chi-square on categorical ones)
- [x] Learning curve showing the sample is large enough
- [x] Decide how to treat imbalance. If you oversample, correct the probabilities back
- [x] Partition into train, validation, and an out-of-time test

### E: Explore, `02_explore.md`

- [x] Univariate distributions, missingness patterns, outliers
- [x] Default rate by grade, purpose, DTI, income, and vintage
- [x] Correlation and association screening, with information value per variable
- [x] Clustering of borrowers to look for structure, with a visualization
- [x] Log what each finding means for Modify

### M: Modify, `03_modify.md`

- [x] Leakage audit table covering every column
- [x] Impute with missing indicators, justified per variable
- [x] Outlier treatment by capping or binning
- [x] Transforms such as log income, credit history length, and WoE binning
- [x] Variable selection and redundancy removal
- [x] Everything fit on train only, in a reusable pipeline

### M: Model, `04_model.md`

- [x] Baseline, WoE logistic scorecard, decision tree, neural network or boosted model
- [x] Tune on validation only
- [x] In SAS: rebuild the flow if available (data, partition, impute, regression, tree, neural network, model comparison) and export screenshots to `artifacts/`

### A: Assess, `05_assess.md`

- [x] ROC-AUC, PR-AUC, KS, Gini, decile lift table, calibration
- [x] Profit or loss at the chosen cutoff under your stated assumptions
- [x] Out-of-time test performance and stability (PSI over vintages)
- [x] Error slices and fairness review
- [x] Compare Python and SAS results if both exist, and explain any gap
- [x] Choose the champion model and justify it

### After Assess (extension), `06_deployment.md`

- [x] Batch scoring script in `deployment/` with input validation and SAS score code if available
- [x] Monitoring plan: PSI, default rate by vintage, retraining triggers
- [x] `REPORT.md`, including the SAS demo and report or the reason it was not possible
