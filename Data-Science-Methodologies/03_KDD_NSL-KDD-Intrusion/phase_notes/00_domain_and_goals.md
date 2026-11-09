# Step 1 (Fayyad): domain and goals

Draft v2 (after critic round 1). This precedes the course's five KDD steps; Fayyad, Piatetsky-Shapiro and Smyth
(1996) call it developing an understanding of the application domain and setting goals.

## Domain background

**Warning first:** NSL-KDD traffic is simulated and derives from a 1998 capture (the DARPA
evaluation, via KDD Cup 99). No pattern here is claimed to hold on modern networks.

**Attack categories** (KDD Cup 99 task description):

| Category | Meaning | Examples in this data |
|---|---|---|
| Normal | ordinary connections | |
| DoS | flood a host so it cannot serve | `neptune` (SYN flood), `smurf`, `back`, `pod`, `teardrop` |
| Probe | scan or survey the network | `satan`, `ipsweep`, `portsweep`, `nmap` |
| R2L | remote user gains local access without an account | `guess_passwd`, `warezclient`, `ftp_write` |
| U2R | local user gains root | `buffer_overflow`, `rootkit`, `loadmodule` |

**What each feature group measures** (41 features per connection):

1. *Basic TCP*: `duration`, `protocol_type`, `service`, `flag` (connection state, for example `S0` =
   SYN seen, no reply; `SF` = normal finish), `src_bytes`, `dst_bytes`, `land`, `wrong_fragment`, `urgent`.
2. *Content* (domain-knowledge features read from the payload): `hot`, `num_failed_logins`, `logged_in`,
   `num_compromised`, `root_shell`, `su_attempted`, `num_root`, `num_file_creations`, `num_shells`,
   `num_access_files`, `num_outbound_cmds`, `is_host_login`, `is_guest_login`. Meaningful for R2L and U2R, which
   happen inside a session.
3. *Time-based traffic* over a two-second window: `count`, `srv_count`, `serror_rate`, `rerror_rate`,
   `same_srv_rate`, `diff_srv_rate`, `srv_diff_host_rate` (and their `srv_` versions). Meaningful for DoS and Probe, which
   show up as many similar connections.
4. *Host-based traffic* over the last 100 connections to the same host: the `dst_host_*` features, which catch slow scans.

## Knowledge goals, fixed before any modeling

In Fayyad's sense, knowledge must be **valid** (holds on the held-out test), **novel**,
**potentially useful** and **understandable** (a person can read it and check it against security knowledge).
A pattern that only reflects how the data was simulated is marked an **artifact**.

| ID | Question | Method | Pass condition |
|---|---|---|---|
| K1 | Do natural groups of connections line up with the attack categories? | k-means and DBSCAN | k-means NMI against the five categories at least 0.50 on training **and at least 0.40 on `KDDTest+`** with the fitted k-means applied unchanged (validity needs an out-of-sample check). DBSCAN cannot score new points and is descriptive |
| K2 | Which compact, readable rules characterize each category? | FP-Growth association rules | **Rule selection (training data only):** per category mine itemsets with support at least 0.20 inside the category, keep rules with at most four items and training confidence at least 0.90, rank by lift, then shorter antecedent, then support, **drop a rule whose antecedent is a more specific version of an already kept rule with confidence no more than 0.005 lower** (added in the notebook code before the test was opened; this note was updated to match it), and fix the top five per category before the test is opened. **Pass:** every category has such a rule, and the fixed rules keep confidence at least 0.80 on `KDDTest+` (reported per rule, no re-ranking) |
| K3 | Can a detector trained on normal traffic only find attacks, including types never seen? | Isolation Forest and a reconstruction-error detector | ROC-AUC (normal vs attack) at least 0.80 on KDDTest+, and recall on **novel** attack types at least 0.50 at the threshold set for 1% false alarms on held-out normal training traffic |
| K4 | How well does supervised classification do per class, on known and novel types? | decision tree, boosted trees | macro-F1 at least 0.55 on `KDDTest+` (also reported on `KDDTest-21`); attack-vs-normal recall on novel types at least 0.40. **Mandatory reporting:** precision, recall and F1 for all five classes with confusion matrices, and U2R recall with its record count (67) and a bootstrap interval (no pass condition: the class is tiny) |
| K5 | Why do random cross-validation results look so much better than the test results? | validation designs compared with the test | explain the gap with numbers |

## Model-selection rule (fixed now)

Supervised candidates are compared on the two validation views (Preprocessing). **Primary metric: macro-F1 on view R.** A
candidate is eligible if its R macro-F1 is within 0.02 of the best. Among the eligible, the choice is the highest
**attack-vs-normal recall on the held-out attack types of view N**; if the 95% bootstrap intervals of two candidates overlap, the
simpler model wins. N is too small to interpret per class (9 `loadmodule` and 7 `multihop` records), so N is read in aggregate with
counts per type. Anomaly detectors are chosen by ROC-AUC on N, with the alarm threshold set for 1% false alarms on held-out normal
training records. Nothing is chosen on the test.

## Disclosure

The project question needs to know which attack types occur only in test, so attack-type **names and counts** of the test files
were inspected while planning (Selection). No model existed and no modeling choice used the counts. The test is therefore a
**characterized test set that is held out from fitting, tuning, threshold setting and rule selection**; it is not blind to its own
label names, and the notes never call it "unseen".
