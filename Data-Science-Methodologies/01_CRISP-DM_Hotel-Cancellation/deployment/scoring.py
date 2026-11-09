"""Scoring helpers shared by the Streamlit app and the notebook's deployment check."""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hotel_features as hf  # noqa: E402  (needed so the pickled pipelines can be loaded)

OPTIONAL_COLS = ["agent", "company", "children"]


def load_policy():
    return json.loads((HERE / "policy.json").read_text())


def load_models():
    champion = joblib.load(HERE / "model.joblib")
    challenger_path = HERE / "model_challenger.joblib"
    challenger = joblib.load(challenger_path) if challenger_path.exists() else None
    return champion, challenger


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Keep exactly the Set A input columns; optional columns may be missing."""
    x = df.copy()
    for c in OPTIONAL_COLS:
        if c not in x.columns:
            x[c] = np.nan
    missing = [c for c in hf.RAW_A_COLS if c not in x.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    return x[hf.RAW_A_COLS]


def arrival_dates(x: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(dict(year=x["arrival_date_year"], month=x["arrival_date_month"].map(hf.MONTH_NUM),
                               day=x["arrival_date_day_of_month"]))


def capped_flags(score, week, arr, thr, cap):
    """Flag bookings with score >= thr; within each arrival week keep at most
    floor(cap x bookings that week), highest score first (ties: earlier arrival, then row order)."""
    score, week, arr = np.asarray(score, float), np.asarray(week), np.asarray(arr)
    n = len(score)
    order = np.lexsort((np.arange(n), arr, -score, week))
    ws = week[order]
    starts = np.r_[True, ws[1:] != ws[:-1]]
    grp = np.cumsum(starts) - 1
    rank = np.arange(n) - np.flatnonzero(starts)[grp]
    size = np.bincount(grp)[grp]
    flags = np.zeros(n, bool)
    flags[order] = (score[order] >= thr) & (rank < np.floor(cap * size))
    return flags


def score_frame(df: pd.DataFrame, champion, challenger=None, policy=None, apply_cap=True, eligible=None) -> pd.DataFrame:
    """`eligible` (optional boolean array): bookings that may receive outreach, for example non-Group bookings in the pilot.
    Ineligible bookings never take a weekly slot, but still count towards the week's size."""
    policy = policy or load_policy()
    x = prepare(df)
    out = pd.DataFrame(index=df.index)
    out["p_champion"] = champion.predict_proba(x)[:, 1]
    out["above_threshold"] = out["p_champion"] >= policy["threshold"]
    if challenger is not None:
        out["p_challenger"] = challenger.predict_proba(x)[:, 1]
    arr = arrival_dates(x)
    iso = arr.dt.isocalendar()
    week = (iso.year.astype(int) * 100 + iso.week.astype(int)).values
    used = out["p_champion"].values if eligible is None else np.where(np.asarray(eligible), out["p_champion"].values, -np.inf)
    out["flag_outreach"] = (capped_flags(used, week, arr.values.astype("datetime64[D]").astype(int),
                                         policy["threshold"], policy["cap"]) if apply_cap else (out["above_threshold"].values & (True if eligible is None else np.asarray(eligible))))
    return out


def pilot_arm(key: str, seed: str = "hotel-pilot-1") -> str:
    """Deterministic 50/50 assignment from a booking key (the same key always gets the same arm)."""
    h = int(hashlib.sha256(f"{seed}|{key}".encode()).hexdigest(), 16)
    return "outreach" if h % 2 == 0 else "control (hold-out)"


def pilot_columns(df: pd.DataFrame, flagged, above=None) -> pd.DataFrame:
    """Pilot rules from the Phase 5 decision: Group bookings are not eligible; eligible flagged bookings are
    randomized 50/50. If the file has no `booking_id`, the key is the full row content (documented limit)."""
    keys = df["booking_id"].astype(str) if "booking_id" in df.columns else df.apply(lambda r: "|".join(map(str, r.values)), axis=1)
    eligible = df["market_segment"].ne("Groups")
    out = pd.DataFrame(index=df.index)
    out["pilot_eligible"] = eligible
    out["pilot_arm"] = np.where(eligible & np.asarray(flagged), keys.map(pilot_arm), "")
    above = np.asarray(flagged) if above is None else np.asarray(above)
    out["action"] = np.select(
        [above & ~eligible.values, ~np.asarray(flagged), out["pilot_arm"].eq("outreach"), out["pilot_arm"].eq("control (hold-out)")],
        ["above threshold but Group booking: not in the pilot, do not contact on the model's say-so", "none",
         "contact the guest (pilot: outreach arm)", "do NOT contact (pilot: control arm); log the outcome"], default="none")
    return out


def week_sizes(df: pd.DataFrame) -> pd.Series:
    arr = arrival_dates(prepare(df))
    iso = arr.dt.isocalendar()
    return pd.Series((iso.year.astype(int) * 100 + iso.week.astype(int)).values).value_counts().sort_index()


def action_text(p: float, thr: float) -> str:
    if p >= thr:
        return ("Flag for personal outreach: contact the guest, reconfirm the stay, and offer a free date change "
                "or a small discount for a non-refundable rate.")
    return "No action: probability is below the outreach threshold."
