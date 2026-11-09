"""Monitoring checks for the deployed loan model (run monthly on scored loans)."""
import numpy as np
import pandas as pd

THRESHOLDS = {"psi_watch": 0.10, "psi_act": 0.25, "calibration_gap_points": 2.0, "auc_drop": 0.03, "approval_ratio_min": 0.8}


def psi(reference, current, bins=10):
    e = np.unique(np.quantile(np.asarray(reference, float), np.linspace(0, 1, bins + 1)))
    e[0], e[-1] = -np.inf, np.inf
    pr, pc = np.histogram(reference, e)[0] / len(reference), np.histogram(current, e)[0] / len(current)
    pr, pc = np.clip(pr, 1e-4, None), np.clip(pc, 1e-4, None)
    return float(((pc - pr) * np.log(pc / pr)).sum())


def vintage_report(df, vintage_col="vintage", pd_col="pd", y_col=None):
    """Per vintage: volume, mean predicted PD and, once outcomes are known, the observed default rate."""
    g = df.groupby(vintage_col)
    out = pd.DataFrame({"loans": g.size(), "mean_pd": g[pd_col].mean()})
    if y_col is not None:
        out["observed"] = g[y_col].mean()
        out["gap_points"] = 100 * (out.observed - out.mean_pd)
    return out


def alerts(score_psi, report=None, approval_ratios=None):
    msgs = []
    if score_psi > THRESHOLDS["psi_act"]:
        msgs.append(f"score PSI {score_psi:.2f} > {THRESHOLDS['psi_act']}: retrain")
    elif score_psi > THRESHOLDS["psi_watch"]:
        msgs.append(f"score PSI {score_psi:.2f} > {THRESHOLDS['psi_watch']}: watch")
    if report is not None and "gap_points" in report:
        bad = report[report.gap_points.abs() > THRESHOLDS["calibration_gap_points"]]
        if len(bad):
            msgs.append(f"calibration gap above {THRESHOLDS['calibration_gap_points']} points in {list(bad.index)}: recalibrate")
    if approval_ratios is not None:
        low = {k: round(v, 2) for k, v in approval_ratios.items() if v < THRESHOLDS["approval_ratio_min"]}
        if low:
            msgs.append(f"approval-rate ratio below {THRESHOLDS['approval_ratio_min']} for {low}: fairness review")
    return msgs
