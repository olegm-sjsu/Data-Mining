"""Feature engineering and weight-of-evidence (WoE) helpers for the Lending Club SEMMA project.
Used by the notebook (training) and the batch scoring script, so both build features the same way.
Everything fitted (bin edges, WoE values) is learned from training rows only."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

EMP_MAP = {"< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4, "5 years": 5, "6 years": 6,
           "7 years": 7, "8 years": 8, "9 years": 9, "10+ years": 10}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features known at origination. Needs `issue_d` (application month, e.g. 'Jan-2013')."""
    x = df.copy()
    issue = pd.to_datetime(x["issue_d"], format="%b-%Y")
    earliest = pd.to_datetime(x["earliest_cr_line"], format="%b-%Y")
    x["credit_hist_years"] = ((issue - earliest).dt.days / 365.25).astype(float)
    x["fico"] = (pd.to_numeric(x["fico_range_low"]) + pd.to_numeric(x["fico_range_high"])) / 2
    inc = pd.to_numeric(x["annual_inc"], errors="coerce").clip(lower=1)
    x["log_income"] = np.log(inc)
    x["log_loan_amnt"] = np.log(pd.to_numeric(x["loan_amnt"], errors="coerce").clip(lower=1))
    x["loan_to_income"] = pd.to_numeric(x["loan_amnt"], errors="coerce") / inc
    x["emp_length_yrs"] = x["emp_length"].map(EMP_MAP).astype(float)
    x["revol_bal_to_income"] = pd.to_numeric(x["revol_bal"], errors="coerce") / inc
    x["dti"] = pd.to_numeric(x["dti"], errors="coerce")
    x["revol_util"] = pd.to_numeric(x["revol_util"], errors="coerce")
    return x


class WoEBinner(BaseEstimator, TransformerMixin):
    """Quantile-bin numeric columns (plus a MISSING bin) and group rare categories, then replace each bin by
    its weight of evidence ln(share of good / share of bad). Fit on training rows only."""

    def __init__(self, num_cols, cat_cols, n_bins=10, min_cat=0.005, smooth=0.5):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.n_bins = n_bins
        self.min_cat = min_cat
        self.smooth = smooth

    def _num_bin(self, v, edges):
        v = pd.to_numeric(v, errors="coerce")
        return np.where(v.isna(), -1, np.searchsorted(edges, v.fillna(0).values, side="right"))

    def _cat_bin(self, v, keep):
        v = v.astype("object").where(v.notna(), "MISSING")
        return v.where(v.isin(keep), "OTHER").astype(str).values

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        self.bad_, self.good_ = float(y.sum()), float((1 - y).sum())
        self.edges_, self.keep_, self.woe_, self.iv_, rows = {}, {}, {}, {}, []
        s = self.smooth
        for c in self.num_cols:
            v = pd.to_numeric(X[c], errors="coerce")
            self.edges_[c] = np.unique(np.nanquantile(v, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
            b = self._num_bin(X[c], self.edges_[c])
            self.woe_[c], self.iv_[c] = self._woe(b, y, c, rows, kind="num")
        for c in self.cat_cols:
            share = X[c].astype("object").where(X[c].notna(), "MISSING").value_counts(normalize=True)
            self.keep_[c] = set(share[share >= self.min_cat].index)
            b = self._cat_bin(X[c], self.keep_[c])
            self.woe_[c], self.iv_[c] = self._woe(b, y, c, rows, kind="cat")
        self.bin_table_ = pd.DataFrame(rows)
        return self

    def _woe(self, b, y, c, rows, kind):
        s = self.smooth
        out, iv = {}, 0.0
        for k in pd.unique(b):
            m = b == k
            bad, good = y[m].sum(), (1 - y[m]).sum()
            w = float(np.log(((good + s) / (self.good_ + s)) / ((bad + s) / (self.bad_ + s))))
            out[k] = w
            part = ((good + s) / (self.good_ + s) - (bad + s) / (self.bad_ + s)) * w
            iv += part
            rows.append({"variable": c, "kind": kind, "bin": k, "loans": int(m.sum()), "bad_rate": float(y[m].mean()), "woe": w, "iv_part": float(part)})
        return out, float(iv)

    def transform(self, X):
        out = {}
        for c in self.num_cols:
            b = self._num_bin(X[c], self.edges_[c])
            out[c + "_woe"] = pd.Series(b, index=X.index).map(self.woe_[c]).fillna(0.0).values
        for c in self.cat_cols:
            b = self._cat_bin(X[c], self.keep_[c])
            out[c + "_woe"] = pd.Series(b, index=X.index).map(self.woe_[c]).fillna(0.0).values
        return pd.DataFrame(out, index=X.index)

    def set_output(self, *, transform=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return np.array([c + "_woe" for c in list(self.num_cols) + list(self.cat_cols)], dtype=object)


def iv_table(X, y, num_cols, cat_cols, n_bins=10):
    """Information value of each variable on the given rows."""
    wb = WoEBinner(num_cols, cat_cols, n_bins=n_bins).fit(X, y)
    t = pd.DataFrame({"variable": list(wb.iv_), "iv": list(wb.iv_.values())}).sort_values("iv", ascending=False).reset_index(drop=True)
    t["strength"] = pd.cut(t.iv, [-1, 0.02, 0.1, 0.3, 0.5, 100], labels=["useless", "weak", "medium", "strong", "suspiciously strong"])
    return t


def ks_stat(y, p):
    """Kolmogorov-Smirnov statistic between the score distributions of bad and good loans (ties handled by the ROC curve)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(np.asarray(y), np.asarray(p))
    return float(np.max(tpr - fpr))


class TreePrep(BaseEstimator, TransformerMixin):
    """Preparation for tree and boosted models: numeric columns clipped at training quantiles (NaN kept),
    categorical columns ordinal-coded (unseen -> -1), and a missing-indicator for numeric columns that are
    missing in at least `miss_min` of the training rows. Fit on training rows only."""

    def __init__(self, num_cols, cat_cols, clip=(0.001, 0.999), miss_min=0.01):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.clip = clip
        self.miss_min = miss_min

    def fit(self, X, y=None):
        num = X[self.num_cols].apply(pd.to_numeric, errors="coerce")
        self.lo_ = num.quantile(self.clip[0])
        self.hi_ = num.quantile(self.clip[1])
        self.miss_cols_ = [c for c in self.num_cols if num[c].isna().mean() >= self.miss_min]
        self.cat_map_ = {c: {v: i for i, v in enumerate(sorted(X[c].dropna().astype(str).unique()))} for c in self.cat_cols}
        return self

    def transform(self, X):
        num = X[self.num_cols].apply(pd.to_numeric, errors="coerce")
        out = num.clip(self.lo_, self.hi_, axis=1).astype("float32")
        for c in self.miss_cols_:
            out[c + "_missing"] = num[c].isna().astype("float32")
        for c in self.cat_cols:
            out[c + "_code"] = X[c].astype("object").where(X[c].notna(), None).map(lambda v: self.cat_map_[c].get(str(v), -1) if v is not None else -1).astype("float32")
        return out

    def set_output(self, *, transform=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return np.array(list(self.num_cols) + [c + "_missing" for c in self.miss_cols_] + [c + "_code" for c in self.cat_cols], dtype=object)


class Pipe:
    """Feature preparation (fit on training rows) followed by a classifier, with a predict_proba on raw rows."""

    def __init__(self, prep, clf):
        self.prep, self.clf = prep, clf

    def fit(self, X, y):
        self.prep.fit(X, y)
        self.clf.fit(self.prep.transform(X), y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(self.prep.transform(X))


class TreeModel:
    """TreePrep plus an already fitted tree or boosted classifier."""

    def __init__(self, tp, clf):
        self.tp, self.clf = tp, clf

    def predict_proba(self, X):
        return self.clf.predict_proba(self.tp.transform(X))


def scorecard_to_sas(pipe, dataset_in="work.loans", dataset_out="work.scored", var_prefix="") -> str:
    """Generate a SAS DATA step that reproduces a WoE logistic scorecard (Pipe(WoEBinner, LogisticRegression)).
    The code is derived from the fitted bin tables; it has not been executed in SAS."""
    wb, lr = pipe.prep, pipe.clf
    names = list(wb.num_cols) + list(wb.cat_cols)
    lines = [f"/* Lending Club scorecard, generated from the Python model. Not executed in SAS. */",
             "/* The input table needs the engineered variables built as in lc_features.add_features (for example fico, loan_to_income): */",
             "/* " + ", ".join(names) + " */", f"data {dataset_out};", f"  set {dataset_in};"]
    for c in wb.num_cols:
        e, w = wb.edges_[c], wb.woe_[c]
        lines.append(f"  /* {c} */")
        lines.append(f"  if missing({c}) then {c}_woe = {w.get(-1, 0.0):.6f};")
        for k in range(len(e)):
            lines.append(f"  else if {c} < {e[k]:.6f} then {c}_woe = {w.get(k, 0.0):.6f};")
        lines.append(f"  else {c}_woe = {w.get(len(e), 0.0):.6f};")
    for c in wb.cat_cols:
        lines.append(f"  /* {c} */")
        lines.append(f"  select (strip(put({c}, $32.)));")
        for k, v in wb.woe_[c].items():
            if k in ("OTHER", "MISSING"):
                continue
            lines.append(f"    when ('{str(k).replace(chr(39), '')}') {c}_woe = {v:.6f};")
        lines.append(f"    otherwise {c}_woe = {wb.woe_[c].get('OTHER', 0.0):.6f};")
        lines.append("  end;")
        if "MISSING" in wb.woe_[c]:
            lines.append(f"  if missing({c}) then {c}_woe = {wb.woe_[c]['MISSING']:.6f};")
    terms = " + ".join(f"({b:.6f} * {n}_woe)" for n, b in zip(names, lr.coef_[0]))
    lines.append(f"  logit = {float(lr.intercept_[0]):.6f} + {terms};")
    lines.append("  pd = 1 / (1 + exp(-logit));")
    lines.append("run;")
    return "\n".join(lines)


def scorecard_apply(pipe, df: pd.DataFrame) -> np.ndarray:
    """Independent re-implementation of the scorecard from its bin tables (used to cross-check the generated SAS logic)."""
    wb, lr = pipe.prep, pipe.clf
    names = list(wb.num_cols) + list(wb.cat_cols)
    logit = np.full(len(df), float(lr.intercept_[0]))
    for n, b in zip(names, lr.coef_[0]):
        if n in wb.num_cols:
            v = pd.to_numeric(df[n], errors="coerce").values
            e, w = wb.edges_[n], wb.woe_[n]
            woe = np.array([w.get(-1, 0.0) if np.isnan(x) else w.get(int(sum(x >= ee for ee in e)), 0.0) for x in v])
        else:
            col = df[n].astype("object").where(df[n].notna(), "MISSING")
            woe = np.array([wb.woe_[n].get(x if x in wb.keep_[n] or x == "MISSING" else "OTHER", 0.0) for x in col])
        logit += b * woe
    return 1 / (1 + np.exp(-logit))
