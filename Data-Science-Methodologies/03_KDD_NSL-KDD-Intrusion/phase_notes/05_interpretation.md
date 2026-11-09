# Step 8: Interpretation / Evaluation

Draft v1. The two test files were scored **once per model** (`artifacts/test_access_log.json`; the bundle hash was checked
first). They are a **characterized test, held out from fitting, tuning, threshold setting and rule selection**; attack-type names and counts were inspected while
planning. Numbers: notebook section "Interpretation", `artifacts/test_*.csv`. Intervals are 95% bootstrap over records.

## 5.1 Per-class results on KDDTest+ and KDDTest-21

| File | Model | Accuracy | Macro-F1 (95% CI) | Recall: normal / DoS / Probe / R2L / U2R |
|---|---|---|---|---|
| KDDTest+ | baseline (logistic) | 0.741 | 0.519 (0.497 to 0.545) | 0.923 / 0.796 / 0.704 / 0.028 / 0.119 |
| KDDTest+ | tree (depth 6) | 0.738 | 0.509 (0.488 to 0.532) | 0.972 / 0.749 / 0.607 / 0.049 / 0.075 |
| KDDTest+ | **boosted** | **0.780** | **0.595 (0.567 to 0.619)** | 0.971 / 0.811 / 0.726 / 0.118 / 0.209 |
| KDDTest-21 | baseline | 0.507 | 0.412 (0.388 to 0.438) | 0.656 / 0.650 / 0.702 / 0.028 / 0.119 |
| KDDTest-21 | tree | 0.504 | 0.407 (0.386 to 0.430) | 0.875 / 0.574 / 0.604 / 0.049 / 0.075 |
| KDDTest-21 | boosted | 0.583 | 0.505 (0.481 to 0.533) | 0.870 / 0.677 / 0.724 / 0.118 / 0.209 |

Boosted model on `KDDTest+`, per class:

| Class | Precision | Recall | F1 | Records |
|---|---|---|---|---|
| normal | 0.690 | 0.971 | 0.806 | 9,711 |
| DoS | 0.962 | 0.811 | 0.880 | 7,460 |
| Probe | 0.792 | 0.726 | 0.757 | 2,421 |
| **R2L** | 0.980 | **0.118** | 0.210 | 2,885 |
| U2R | 0.700 | 0.209 | 0.322 | 67 |

Confusion matrix (rows true, columns predicted: normal, DoS, Probe, R2L, U2R): normal 9,428 / 72 / 207 / 2 / 2; DoS 1,209 / 6,051 / 200 / 0 / 0; Probe
498 / 166 / 1,757 / 0 / 0; **R2L 2,487 / 0 / 54 / 340 / 4**; U2R 48 / 0 / 0 / 5 / 14 (`I_confusion.png`). **Most R2L (2,487 of 2,885) is classified as
normal.** U2R recall is 14 of 67 (0.209, 95% CI 0.104 to 0.306); of the 30 novel-type U2R records 6 are found. The tree and the baseline are worse on every
rare class. Accuracy hides this: 0.78 overall.

## 5.2 Known versus novel attack types; supervised versus anomaly detection (`KDDTest+`)

| Detector | Recall, known-type attacks | Recall, novel-type attacks | False alarms on normal |
|---|---|---|---|
| supervised baseline | 0.759 | 0.311 | 0.077 |
| supervised tree | 0.760 | 0.179 | 0.028 |
| **supervised boosted** | 0.786 | **0.386** | 0.029 |
| anomaly detector at its frozen 1% threshold | 0.051 | 0.100 | 0.006 |
| union: boosted OR anomaly alarm | 0.815 | 0.415 | 0.034 |

(The baseline's novel recall is "predicted any attack category"; the boosted model also gets the *category* right for 78% of known-type and only
**28.7% of novel-type** attacks.) The 3,750 novel-type records are 29.2% of the test attacks.

**The anomaly detector ranks well but alarms rarely.** ROC-AUC (attack vs normal) on `KDDTest+` is **0.958** (known-type 0.963, novel-type 0.947). The median anomaly score is
0.24 for normal, 12.66 for known and 9.30 for novel attacks, but the frozen threshold, set at 1% false alarms on training normals, is 20.51, above most attacks. *Diagnostic,
post hoc (thresholds taken from the test normals, so not a deployable setting):* recall at 1% / 5% / 10% / 20% alarms on normal traffic is 0.27 / 0.76 / 0.86 / 1.00 for known types and
0.31 / 0.67 / 0.78 / 0.98 for novel types. So unsupervised detection does see novel attacks as well as known ones, if a higher false-alarm rate is accepted.

**By novel type (boosted flags an attack / anomaly alarm):** `saint` 99.7% / 7.8%; `apache2` 56.7% / 22.8%; `mscan` 50.1% / 10.4%; `httptunnel` 39.1% / 10.5%;
`processtable` 21.2% / 5.8%; and **0% for `mailbomb` (293 records), `snmpguess` (331) and `snmpgetattack` (178)** from both detectors. Attacks whose evidence lives in the content (mail floods, SNMP queries) were not separated from ordinary connections by these features and models.

## 5.3 Why random validation looked so much better (K5)

| View | Accuracy | Macro-F1 |
|---|---|---|
| Random split of `KDDTrain+` (view R) | 0.999 | 0.969 |
| `KDDTest+` | 0.780 | 0.595 |
| `KDDTest+`, known attack types and normal only | 0.879 | 0.674 |
| `KDDTest+`, novel attack types only (category correct) | 0.287 | n/a |
| `KDDTest+` per-class recall weighted by the **training** class mix | 0.883 | n/a |
| `KDDTest-21` | 0.583 | 0.505 |

The 22-point accuracy gap has two measurable parts: (1) **class mix**: R2L is 12.8% of the test but 0.8% of training and the model finds 11.8% of it; weighting test
recalls by the training mix gives accuracy 0.883. (2) **novel types**: removing them lifts accuracy from 0.780 to 0.879. Even **known** types behave differently: `guess_passwd` has 53
training records and 1,231 test records, and the model finds **0 of them**; `warezmaster` 20 and 944, recall 0.358. The random split never meets that shift.

## 5.4 Knowledge goals checked on the test

| Goal | Result | Verdict |
|---|---|---|
| K1 k-means NMI at least 0.50 (train) and 0.40 (`KDDTest+`) | train 0.477; test **0.304** (ARI 0.289); `KDDTest-21` 0.223 | **Fail** |
| K2 each category has a fixed rule that keeps confidence at least 0.80 | normal, DoS, Probe yes (140 to 4,628 matched records); U2R passes only literally (one rule has confidence 1.00 on 1 record; the best-supported has 0.60 on 10); **R2L has no rule** at the required support | **Fail** (R2L; U2R not credible) |
| K3 anomaly AUC at least 0.80, novel recall at least 0.50 at the 1% threshold | AUC **0.958**; novel recall **0.100** | **Fail** (recall) |
| K4 macro-F1 at least 0.55 and novel attack-vs-normal recall at least 0.40 | macro-F1 **0.595**; novel recall **0.386** | **Fail** (recall, by 0.014) |

**Fixed rules on `KDDTest+`** (pre-registered; no re-ranking): normal rules keep confidence 1.00 (2,912 to 4,628 records matched); DoS rules 0.89 to 1.00 (1,283 to 1,511
matched); Probe rules 0.98 to 1.00 (140 to 165 matched); U2R rules match only 0 to 10 test records (one has confidence 0.60 on 10). The supplementary R2L rules match 0 to 3
test records: **they describe `warezclient`, which does not occur in the test.** The literal K2 "pass" for U2R rests on 1 to 10 records and is not credible, and one retained U2R rule has test confidence 0.60, below the 0.80 line.

## 5.5 Plain-language knowledge, checked against security domain knowledge

| Pattern (support, confidence, lift from training; test confidence) | In plain language | Real or artifact? |
|---|---|---|
| `flag=S0` with high server-error rates, `count=high`, `rerror_rate=zero` (support 0.58 to 0.70, 1.00, 2.74; test 0.89 to 1.00) | a burst of half-open connections to one host: a **SYN flood** | **Real** behaviour. But `neptune` is 41,214 of 45,927 training DoS, so this pattern is mostly one attack |
| `service=eco_i` with `srv_diff_host_rate=high`, `srv_count=high` (0.22 to 0.30, 1.00, 10.8; test 0.98 to 1.00) | many ICMP echo requests spread over many hosts: a **ping sweep** | **Real** (`ipsweep`). Port scans (`portsweep`, `satan`) are not covered by it |
| `service=http` with `src_bytes=mid` (0.36, 1.00, 1.87; test 1.00) | ordinary web requests of typical size look normal | **Partly artifact**: the request size of the simulated normal traffic; an attacker could copy it |
| `root_shell=positive` or `service=telnet` with `hot=mid` (0.21 to 0.33, 0.94 to 1.00, about 2,300; test 0.60 on 10 records) | an interactive session that obtains a root shell | **Real** in kind, **fragile** (11 to 18 training records) |
| `is_guest_login=1`, short `duration`, `service=ftp`, large `src_bytes` (R2L supplementary; support 0.19) | anonymous FTP transfers of large files: **warezclient** | **Real**, but it **does not transfer**: `warezclient` is absent from the test, 3 matches |
| `src_bytes` has the highest mutual information (0.715) and the highest boosted-model gain (17%) | | **Largely artifact**: 1998 attack tools send fixed payload sizes (for example `smurf`), which a modern attack would not copy |

**What these features did not detect:** `snmpguess`, `snmpgetattack`, `mailbomb` (0% by both detectors) and `guess_passwd` (0 of 1,231). Their evidence is mostly in the content of a
session (queries, mail volume, password attempts). The data does include content features (`hot`, `num_failed_logins` and others), so this is a statement about these models and
features on this data, not a proof that such attacks are undetectable from connection summaries.

## 5.6 Loops back to earlier steps (documented)

1. **Interpretation to Selection and Preprocessing (before the test):** the "novel attack types" question made a single random split unusable, so the
   validation design was changed to views R and N (critic round 1 added the refit-within-view rule).
2. **Data Mining to Preprocessing:** an implementation slip in the tie-break of the model-selection rule was found and corrected (see `04_data_mining.md`).
3. **After the test (not acted on, to keep the test clean):** the findings suggest (a) a calibrated anomaly threshold at a higher false-alarm budget, (b) re-weighting
   R2L, (c) payload-aware features. These would need a new validation design and a fresh test; none was applied.

## Limits

1. 1998-derived simulated traffic: nothing here is claimed for modern networks.
2. The category of the 17 test-only types is a convention. 3. N is small (9 and 7 records for two types). 4. R2L and U2R conclusions rest on small counts.
