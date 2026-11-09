# Deployment extension: batch scoring of Lending Club applications

Decision support only. See `../REPORT.md` for the limits.

```bash
cd deployment
pip install -r ../requirements.txt
python score_loans.py sample_applications.csv scored.csv rejects.csv
```

| File | Purpose |
|---|---|
| `score_loans.py` | Validation and scoring (accept if PD <= cutoff; strict illustrative policy) |
| `lc_features.py` | Feature engineering, WoE binner, SAS generator (shared with training) |
| `lc_monitor.py` | PSI, per-vintage calibration gap, approval-ratio alerts |
| `model_champion.joblib`, `policy.json` | Frozen Model A champion and thresholds (hash in `policy.json`) |
| `model_scorecard.joblib`, `scorecard_score_code.sas` | Transparent scorecard and its SAS export (not run in SAS) |
| `sample_applications.csv` | 606 example rows, six deliberately invalid |
