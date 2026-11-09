"""Batch scoring of Lending Club loan applications with the frozen Model A champion.

Usage:  python score_loans.py applications.csv scored.csv [rejects.csv]
Rows that fail validation are written to the rejects file with the reason and are not scored.
Decisions are decision support only: see REPORT.md for the limits."""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lc_features as lf  # noqa: E402  (needed so the pickled model can be loaded)

BASE_COLS = ["issue_d", "earliest_cr_line", "fico_range_low", "fico_range_high", "annual_inc", "loan_amnt", "emp_length", "revol_bal", "dti",
             "revol_util", "inq_last_6mths", "home_ownership", "purpose", "term"]
REQUIRED_NON_MISSING = ["fico_range_low", "fico_range_high", "annual_inc", "loan_amnt", "dti", "revol_util", "inq_last_6mths", "revol_bal"]
RULES = {"fico_range_low": (300, 850), "fico_range_high": (300, 850), "annual_inc": (1, 10_000_000), "loan_amnt": (500, 40_000), "dti": (0, 100),
         "revol_util": (0, 300), "inq_last_6mths": (0, 50), "revol_bal": (0, 10_000_000)}
HOME = {"RENT", "MORTGAGE", "OWN", "OTHER", "NONE", "ANY"}


def load():
    """Load the frozen champion, refusing to run if the model file no longer matches the hash recorded in policy.json."""
    policy = json.loads((HERE / "policy.json").read_text())
    path = HERE / "model_champion.joblib"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != policy["champion_file_sha256"]:
        raise RuntimeError(f"model file hash {digest} does not match policy.json ({policy['champion_file_sha256']}): refusing to score")
    return joblib.load(path), policy


ENGINEERED = {"fico", "log_income", "log_loan_amnt", "loan_to_income", "emp_length_yrs", "revol_bal_to_income", "credit_hist_years"}   # built by add_features from BASE_COLS


def bureau_columns(policy):
    """Credit-bureau fields: allowed to be absent or empty (the model was trained to handle missing values)."""
    core = {"annual_inc", "dti", "revol_util", "inq_last_6mths"}
    return [c for c in policy["features"]["A_ext_num"] if c not in ENGINEERED and c not in core]


def required_columns(policy):
    raw = [c for c in policy["features"]["A_ext_num"] if c not in ENGINEERED and c not in bureau_columns(policy)]
    return sorted(set(BASE_COLS) | set(raw) | set(policy["features"]["cat"]))


def validate(df: pd.DataFrame, policy) -> pd.Series:
    """Return a string per row: empty if valid, otherwise the reasons the row is rejected."""
    missing = [c for c in required_columns(policy) if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    why = pd.Series("", index=df.index, dtype="object")
    df = df.copy()
    for c in bureau_columns(policy):          # an absent bureau column is treated as all missing, not as an error
        if c not in df.columns:
            df[c] = np.nan

    def add(mask, text):
        why[mask] = why[mask] + text + "; "

    add(df["term"].astype(str).str.strip().ne("36 months"), "term must be 36 months (the 60-month model is not deployed)")
    for c, (lo, hi) in RULES.items():
        v = pd.to_numeric(df[c], errors="coerce")
        add(v.isna() & (c in REQUIRED_NON_MISSING), f"{c} missing")
        add(v.notna() & ((v < lo) | (v > hi)), f"{c} outside [{lo}, {hi}]")
    for c in ("issue_d", "earliest_cr_line"):
        add(pd.to_datetime(df[c], format="%b-%Y", errors="coerce").isna(), f"{c} not in 'Mon-YYYY' format")
    add(~df["home_ownership"].astype(str).str.upper().isin(HOME), "unknown home_ownership")
    ok_dates = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce").notna() & pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce").notna()
    gap = (pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce") - pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")).dt.days
    add(ok_dates & (gap < 0), "credit line opened after the loan")
    return why


def score(df: pd.DataFrame, model, policy):
    why = validate(df, policy)
    good = why.eq("")
    out = df.loc[good].copy()
    for c in bureau_columns(policy):
        if c not in out.columns:
            out[c] = np.nan
    x = lf.add_features(out)
    out["pd"] = model.predict_proba(x)[:, 1]
    out["bureau_fields_missing"] = out[bureau_columns(policy)].isna().sum(axis=1)   # a warning, not a rejection: the model handles missing bureau fields, but the score is less reliable
    out["decision"] = np.where(out["pd"] <= policy["cutoff_pd"], "accept", "decline")
    out["decision_strict"] = np.where(out["pd"] <= policy["strict_cutoff_pd"], "accept", "decline")
    rejects = df.loc[~good].copy()
    rejects["reject_reason"] = why[~good]
    return out, rejects


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    model, policy = load()
    df = pd.read_csv(argv[1], low_memory=False)
    out, rejects = score(df, model, policy)
    out.to_csv(argv[2], index=False)
    rej_path = argv[3] if len(argv) > 3 else str(Path(argv[2]).with_suffix(".rejects.csv"))
    rejects.to_csv(rej_path, index=False)
    print(f"scored {len(out):,} of {len(df):,} rows ({len(rejects):,} rejected, see {rej_path}); accept share {(out.decision == 'accept').mean():.1%}")
    if (out.bureau_fields_missing > 0).any():
        print(f"warning: {(out.bureau_fields_missing > 0).sum():,} scored rows have missing credit-bureau fields (column bureau_fields_missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
