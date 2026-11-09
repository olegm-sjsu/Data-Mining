# Step 2: Selection

Draft v1. Numbers from notebook section "Selection" and `artifacts/selection_files.csv`,
`novel_attack_types.csv`, `class_mix.csv`.

## Which files, records and features form the target data

| File | Records | Role |
|---|---|---|
| `KDDTrain+` | 125,973 | training data |
| `KDDTest+` | 22,544 | main test; contains attack types absent from training |
| `KDDTest-21` | 11,850 | harder subset of `KDDTest+`: exactly the records with difficulty at most 20 (all 11,850 rows also appear in `KDDTest+`, and `KDDTest+` has 11,850 such rows) |
| `KDDTrain+_20Percent` | 25,192 | **not used**: every row is already in `KDDTrain+` |

Features: the first 41 columns. `label` is the attack type, grouped into five categories (Preprocessing).
The last column, `difficulty`, is dropped from modeling (it counts how many of 21 classifiers labelled the
record correctly, so it is derived from classifier behaviour). `KDDTest+` and `KDDTrain+` share 610
identical records (41 features and label), 2.7% of the test.

## Provenance, and how NSL-KDD differs from KDD Cup 99

NSL-KDD (Tavallaee, Bagheri, Lu, Ghorbani, 2009) is built from KDD Cup 99, which derives from the 1998 DARPA
evaluation: simulated traffic captured at MIT Lincoln Laboratory. As the paper describes it (these figures
were not re-verified here): the original training file had about 78% redundant records and the test file about
75%, which biased classifiers toward frequent records, so NSL-KDD **removes duplicates**, and **samples from each
difficulty level inversely to its share in the original data**. The test set is deliberately **not from the same
distribution as training**. The Kaggle page lists the license as Unknown, so the original authors are credited.

## Which attack types appear only in test

**17 attack types occur only in `KDDTest+`** (3,750 records, 29.2% of the attacks in `KDDTest+`; 3,740 of them are in
`KDDTest-21`). Two types occur only in training (`spy`, `warezclient`).

| Category | Test-only types (records in `KDDTest+`) |
|---|---|
| Probe | `mscan` 996, `saint` 319 |
| DoS | `apache2` 737, `processtable` 685, `mailbomb` 293, `udpstorm` 2, `worm` 2 |
| R2L | `snmpguess` 331, `snmpgetattack` 178, `httptunnel` 133, `named` 17, `sendmail` 14, `xlock` 9, `xsnoop` 4 |
| U2R | `ps` 15, `xterm` 13, `sqlattack` 2 |

**The class mix differs strongly.**

| Category | `KDDTrain+` | `KDDTest+` | `KDDTest-21` |
|---|---|---|---|
| normal | 67,343 (53.5%) | 9,711 (43.1%) | 2,152 (18.2%) |
| DoS | 45,927 (36.5%) | 7,460 (33.1%) | 4,344 (36.7%) |
| Probe | 11,656 (9.3%) | 2,421 (10.7%) | 2,402 (20.3%) |
| R2L | 995 (0.8%) | 2,885 (12.8%) | 2,885 (24.4%) |
| U2R | 52 (0.04%) | 67 (0.30%) | 67 (0.57%) |

R2L is 0.8% of training and 12.8% of the test. Even for attack types that exist in training, test counts
differ sharply: `warezmaster` has 20 training and 944 test records (47 times), `guess_passwd` 53 and 1,231 (23 times).
So the test mixes **novel types** with **known types in very different proportions**. Both matter for Interpretation.

## Notes

1. The category of each test-only type is a **convention** (the files have no category column), listed in
   `artifacts/attack_type_mapping.csv`.
2. The attack-type names and counts of the test were inspected here (see the disclosure in `00_domain_and_goals.md`).
