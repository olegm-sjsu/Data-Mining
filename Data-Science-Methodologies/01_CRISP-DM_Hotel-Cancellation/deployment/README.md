# Deployment: booking non-arrival risk app

Decision-support tool for a controlled pilot. It scores bookings with the frozen
logistic regression (champion) and shows a LightGBM challenger probability (shadow only).

## Run locally

```bash
cd deployment
pip install -r requirements.txt
streamlit run app.py
```

Tabs: score one booking; score a CSV (`sample_bookings.csv` is an example, with
the weekly 25% cap applied); about this model.

## Run in Docker

```bash
docker build -t hotel-risk deployment && docker run -p 8501:8501 hotel-risk
```

## Public URL (Streamlit Community Cloud) — your step

1. Push the repository to GitHub.
2. On share.streamlit.io choose "New app", pick the repo, set the main file to
   `Data-Science-Methodologies/01_CRISP-DM_Hotel-Cancellation/deployment/app.py`.
3. Put the resulting URL in the project README and the Medium article.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `scoring.py` | Loading, scoring, weekly capped flags (also used by the notebook check) |
| `hotel_features.py` | Feature construction and pipeline (shared with training) |
| `monitor.py` | PSI and weekly performance checks |
| `model.joblib`, `policy.json` | Frozen champion and threshold (hash in `policy.json`) |
| `model_challenger.joblib` | Shadow LightGBM |

Package versions are pinned to those used for the reported results.
