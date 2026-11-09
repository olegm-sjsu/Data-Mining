"""Monitoring checks for the intrusion service. Run on a day or week of scored connections."""
import numpy as np
import pandas as pd

THRESHOLDS = {"psi_watch": 0.10, "psi_act": 0.25, "alert_rate_ratio": 1.5, "analyst_precision_min": 0.60}


def psi(reference, current, bins=10, min_rows=30):
    """Population stability index; NaN (not 0) when either sample is too small to say anything."""
    ref, cur = pd.Series(reference).dropna().astype(float), pd.Series(current).dropna().astype(float)
    if len(ref) < min_rows or len(cur) < min_rows:
        return float("nan")
    e = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(e) < 3:
        return 0.0     # a (nearly) constant reference column cannot drift in a measurable way
    e[0], e[-1] = -np.inf, np.inf
    pr, pc = np.histogram(ref, e)[0] / len(ref), np.histogram(cur, e)[0] / len(cur)
    pr, pc = np.clip(pr, 1e-4, None), np.clip(pc, 1e-4, None)
    return float(((pc - pr) * np.log(pc / pr)).sum())


def feature_drift(train_df, new_df, cols):
    return pd.Series({c: psi(train_df[c], new_df[c]) for c in cols}).sort_values(ascending=False)


def alert_report(alerts, baseline_rate):
    rate = float(np.mean(alerts))
    return {"alert_rate": rate, "ratio_to_baseline": rate / baseline_rate if baseline_rate else np.nan}


def analyst_precision(verdicts):
    """verdicts: iterable of 1 (analyst confirmed the alert) or 0 (false alarm)."""
    v = np.asarray(list(verdicts), float)
    return float(v.mean()) if len(v) else np.nan


def alerts(drift, alert_info=None, precision=None):
    msgs = [f"feature drift {c}: PSI {v:.2f} (> {THRESHOLDS['psi_act']}): investigate or retrain" for c, v in drift.items() if v > THRESHOLDS["psi_act"]]
    msgs += [f"feature drift {c}: PSI {v:.2f} (> {THRESHOLDS['psi_watch']}): WATCH" for c, v in drift.items() if THRESHOLDS["psi_watch"] < v <= THRESHOLDS["psi_act"]]
    msgs += [f"feature {c}: too little data to compute drift" for c, v in drift.items() if pd.isna(v)]
    if alert_info and alert_info["ratio_to_baseline"] > THRESHOLDS["alert_rate_ratio"]:
        msgs.append(f"alert volume {alert_info['ratio_to_baseline']:.1f} times the baseline: check for an attack wave or a drifted model")
    if precision is not None and pd.isna(precision):
        msgs.append("no analyst verdicts available: alert precision cannot be assessed")
    elif precision is not None and precision < THRESHOLDS["analyst_precision_min"]:
        msgs.append(f"analyst-confirmed precision {precision:.2f} < {THRESHOLDS['analyst_precision_min']}: tune thresholds or retrain")
    return msgs
