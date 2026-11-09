"""FastAPI service: classify one network connection (or a batch) and return the attack category, an anomaly score and the matched rules.

Run:  uvicorn main:app --port 8000      Docs: http://localhost:8000/docs
Decision support only. The model was trained on 1998-derived simulated traffic (NSL-KDD); see REPORT.md for what it cannot do."""
import hashlib
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import Field, create_model

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kdd_features as kf  # noqa: E402  (needed so the pickled bundle can be loaded)

STATE = {}


def build_connection_model():
    fields = {}
    for c in kf.COLS:
        if c in kf.CAT_COLS:
            fields[c] = (str, ...)
        elif c in kf.RATE_COLS:
            fields[c] = (float, Field(ge=0, le=1))
        elif c in kf.BINARY_COLS:
            fields[c] = (int, Field(ge=0, le=1))
        else:
            fields[c] = (float, Field(ge=0))
    return create_model("Connection", **fields)


Connection = build_connection_model()


def load_bundle():
    policy = json.loads((HERE / "policy.json").read_text())
    path = HERE / "model_bundle.joblib"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != policy["bundle_sha256"]:
        raise RuntimeError(f"model bundle hash {digest} does not match policy.json: refusing to start")
    return joblib.load(path), policy


@asynccontextmanager
async def lifespan(app):
    STATE["bundle"], STATE["policy"] = load_bundle()
    yield
    STATE.clear()


app = FastAPI(title="NSL-KDD intrusion knowledge service", version="1.0", lifespan=lifespan)


def check_categories(df: pd.DataFrame):
    prep = STATE["bundle"]["prep"]
    for c in kf.CAT_COLS:
        bad = sorted(set(df[c].astype(str)) - set(prep.cats_[c]))
        if bad:
            raise HTTPException(422, f"unknown {c} value(s) {bad[:3]}; known values: {prep.cats_[c][:8]}...")


def score(df: pd.DataFrame):
    B, pol = STATE["bundle"], STATE["policy"]
    check_categories(df)
    X = B["prep"].transform(df[kf.COLS])
    model = B["models"]["boosted"]
    proba = model.predict_proba(X)
    anomaly = B["anomaly"].score(df)
    items = B["disc"].items(df[kf.COLS])
    out = []
    for i in range(len(df)):
        cls = model.classes_[int(proba[i].argmax())]
        matched = [{"group": g, "category": r["category"], "rule": " AND ".join(r["antecedent"]), "train_confidence": round(r["confidence"], 3),
                    "train_lift": round(r["lift"], 2), "train_support_in_category": round(r["support_in_category"], 3)}
                   for g, rules in (("fixed", B["rules"]), ("supplementary", B["rules_supplementary"])) for r in kf.match_rules(items.iloc[i], rules)]
        out.append({"category": cls, "probabilities": {c: round(float(p), 4) for c, p in zip(model.classes_, proba[i])},
                    "anomaly_score": round(float(anomaly[i]), 3), "anomaly_alarm": bool(anomaly[i] > B["anomaly_threshold"]),
                    "alert": bool(cls != "normal" or anomaly[i] > B["anomaly_threshold"]), "matched_rules": matched,
                    "model_hash": pol["bundle_sha256"][:16], "warning": "1998-derived simulated data; decision support only."})
    return out


@app.get("/health")
def health():
    return {"status": "ok", "model_hash": STATE["policy"]["bundle_sha256"][:16], "anomaly_threshold": STATE["bundle"]["anomaly_threshold"]}


@app.post("/classify")
def classify(conn: Connection):
    return score(pd.DataFrame([conn.model_dump()]))[0]


@app.post("/classify_batch")
def classify_batch(conns: list[Connection]):
    if not 1 <= len(conns) <= 1000:
        raise HTTPException(422, "send between 1 and 1000 connections")
    return score(pd.DataFrame([c.model_dump() for c in conns]))
