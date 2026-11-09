# Project 2: SEMMA, Lending Club Loan Default

| | |
|---|---|
| **Question** | Which loans will charge off, using only what is known at origination? |
| **Dataset** | [All Lending Club loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club), CC0, about 650 MB compressed, about 2.2M accepted loans. Use `accepted_2007_to_2018Q4` |
| **Target** | From `loan_status`: Charged Off or Default is 1, Fully Paid is 0. Exclude loans still open |
| **Tools** | Python in Colab, plus a SAS demo if available |
| **Deployment** | Batch scoring script, plus SAS score code if available |
| **Colab** | _add badge_ |
| **Medium** | _add link_ |
| **YouTube** | _add link_ |

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

- [ ] Short problem statement, loss assumptions, and success criteria, marked as outside SEMMA itself

### S: Sample, `01_sample.md`

- [ ] Define the eligible population, and count what each filter removes
- [ ] Draw a stratified sample by vintage, term, and outcome, and justify the size
- [ ] Prove it is representative against the full eligible population (KS or PSI on numeric columns, chi-square on categorical ones)
- [ ] Learning curve showing the sample is large enough
- [ ] Decide how to treat imbalance. If you oversample, correct the probabilities back
- [ ] Partition into train, validation, and an out-of-time test

### E: Explore, `02_explore.md`

- [ ] Univariate distributions, missingness patterns, outliers
- [ ] Default rate by grade, purpose, DTI, income, and vintage
- [ ] Correlation and association screening, with information value per variable
- [ ] Clustering of borrowers to look for structure, with a visualization
- [ ] Log what each finding means for Modify

### M: Modify, `03_modify.md`

- [ ] Leakage audit table covering every column
- [ ] Impute with missing indicators, justified per variable
- [ ] Outlier treatment by capping or binning
- [ ] Transforms such as log income, credit history length, and WoE binning
- [ ] Variable selection and redundancy removal
- [ ] Everything fit on train only, in a reusable pipeline

### M: Model, `04_model.md`

- [ ] Baseline, WoE logistic scorecard, decision tree, neural network or boosted model
- [ ] Tune on validation only
- [ ] In SAS: rebuild the flow if available (data, partition, impute, regression, tree, neural network, model comparison) and export screenshots to `artifacts/`

### A: Assess, `05_assess.md`

- [ ] ROC-AUC, PR-AUC, KS, Gini, decile lift table, calibration
- [ ] Profit or loss at the chosen cutoff under your stated assumptions
- [ ] Out-of-time test performance and stability (PSI over vintages)
- [ ] Error slices and fairness review
- [ ] Compare Python and SAS results if both exist, and explain any gap
- [ ] Choose the champion model and justify it

### After Assess (extension), `06_deployment.md`

- [ ] Batch scoring script in `deployment/` with input validation and SAS score code if available
- [ ] Monitoring plan: PSI, default rate by vintage, retraining triggers
- [ ] `REPORT.md`, including the SAS demo and report or the reason it was not possible
