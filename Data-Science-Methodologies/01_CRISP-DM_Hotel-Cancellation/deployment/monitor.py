"""Drift and performance checks for the deployed model (run weekly on scored bookings)."""
import numpy as np
import pandas as pd


def psi(expected, actual, bins=10):
    """Population stability index. Numeric inputs are binned by the expected sample's quantiles;
    categorical inputs use their categories. Rule of thumb: <0.1 stable, 0.1-0.25 watch, >0.25 act."""
    e, a = pd.Series(expected).dropna(), pd.Series(actual).dropna()
    if e.dtype.kind in "if" and e.nunique() > 12:
        edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        pe = pd.cut(e, edges).value_counts(normalize=True, sort=False)
        pa = pd.cut(a, edges).value_counts(normalize=True, sort=False)
    else:
        cats = sorted(set(e.unique()) | set(a.unique()), key=str)
        pe = e.value_counts(normalize=True).reindex(cats).fillna(0)
        pa = a.value_counts(normalize=True).reindex(cats).fillna(0)
    pe, pa = pe.clip(lower=1e-4), pa.clip(lower=1e-4)
    return float(((pa - pe) * np.log(pa / pe)).sum())


def weekly_report(df, week_col="week", p_col="p", y_col="y", flag_col="flag"):
    """Weekly aggregate error, flagged precision and flagged share (needs outcomes, so run with a lag)."""
    g = df.groupby(week_col)
    out = pd.DataFrame({"bookings": g.size(), "expected_non_arrivals": g[p_col].sum(), "actual_non_arrivals": g[y_col].sum(),
                        "flagged": g[flag_col].sum(),
                        "flagged_precision": g.apply(lambda d: d.loc[d[flag_col], y_col].mean() if d[flag_col].any() else np.nan)})
    out["aggregate_error"] = (out.expected_non_arrivals - out.actual_non_arrivals).abs() / out.actual_non_arrivals.replace(0, np.nan)
    out["flagged_share"] = out.flagged / out.bookings
    return out


def group_rates(df, group_col, y_col="y", flag_col="flag", min_rows=300):
    """Flagged share, recall and false-positive rate per group (fairness check)."""
    rows = []
    for g, d in df.groupby(group_col):
        if len(d) < min_rows:
            continue
        neg, pos = d[~d[y_col].astype(bool)], d[d[y_col].astype(bool)]
        rows.append({group_col: g, "rows": len(d), "flagged_share": d[flag_col].mean(),
                     "recall": pos[flag_col].mean() if len(pos) else np.nan, "false_positive_rate": neg[flag_col].mean() if len(neg) else np.nan})
    return pd.DataFrame(rows).set_index(group_col)


def fpr_ratio(rates):
    r = rates["false_positive_rate"].dropna()
    return float(r.max() / r.min()) if len(r) > 1 and r.min() > 0 else float("nan")


THRESHOLDS = {"psi_watch": 0.10, "psi_act": 0.25, "aggregate_error_retrain": 0.15, "flagged_precision_min": 0.60,
              "false_positive_rate_ratio_max": 1.5}


def alerts(psis: dict, weekly: pd.DataFrame, consecutive=3, country_rates=None):
    msgs = [f"PSI {k} = {v:.2f} (> {THRESHOLDS['psi_act']}): retrain trigger" for k, v in psis.items() if v > THRESHOLDS["psi_act"]]
    msgs += [f"PSI {k} = {v:.2f} (> {THRESHOLDS['psi_watch']}): watch" for k, v in psis.items() if THRESHOLDS["psi_watch"] < v <= THRESHOLDS["psi_act"]]
    bad = (weekly.aggregate_error > THRESHOLDS["aggregate_error_retrain"]).astype(int).rolling(consecutive).sum().eq(consecutive).any()
    if bad:
        msgs.append(f"weekly aggregate error above {THRESHOLDS['aggregate_error_retrain']} for {consecutive} weeks in a row: retrain trigger")
    low = (weekly.flagged_precision < THRESHOLDS["flagged_precision_min"]).astype(int).rolling(3).sum().eq(3).any()
    if low:
        msgs.append("flagged precision below 0.60 for 3 weeks in a row: pause outreach and review "
                    "(aggregate error alone would not catch this: totals can stay right while the flagged set gets worse)")
    if country_rates is not None and fpr_ratio(country_rates) > THRESHOLDS["false_positive_rate_ratio_max"]:
        msgs.append(f"false-positive-rate ratio across country groups is {fpr_ratio(country_rates):.1f} (> {THRESHOLDS['false_positive_rate_ratio_max']}): review proxies such as agent_id")
    return msgs
