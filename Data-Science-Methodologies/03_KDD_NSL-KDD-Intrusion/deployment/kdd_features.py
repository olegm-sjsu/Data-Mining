"""Shared code for the NSL-KDD knowledge-discovery project: loading, label grouping, feature preparation,
discretization for association rules, and rule matching. Used by the notebook and by the FastAPI service.
Everything fitted (scalers, categories, bin edges) is learned from training rows only."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

COLS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
        "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files",
        "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
        "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]
ALL_COLS = COLS + ["label", "difficulty"]
CAT_COLS = ["protocol_type", "service", "flag"]
BINARY_COLS = ["land", "logged_in", "is_host_login", "is_guest_login"]
RATE_COLS = [c for c in COLS if c.endswith("_rate")]
NUM_COLS = [c for c in COLS if c not in CAT_COLS]
CATEGORIES = ["normal", "DoS", "Probe", "R2L", "U2R"]

# Attack type to category. The widely used convention, from the KDD Cup 99 task description plus the 17 test-only types;
# the data files do not contain categories, so the assignment of the test-only types is a documented convention.
ATTACK_MAP = {
    "normal": "normal",
    **{a: "DoS" for a in ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2", "udpstorm", "processtable", "mailbomb", "worm"]},
    **{a: "Probe" for a in ["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"]},
    **{a: "R2L" for a in ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster", "xlock", "xsnoop",
                          "snmpguess", "snmpgetattack", "httptunnel", "named", "sendmail"]},
    **{a: "U2R" for a in ["buffer_overflow", "loadmodule", "perl", "rootkit", "ps", "sqlattack", "xterm"]},
}


def load_nsl(path):
    df = pd.read_csv(path, header=None, names=ALL_COLS)
    return df


def group_labels(label: pd.Series) -> pd.Series:
    return label.map(ATTACK_MAP)


class KDDPrep(BaseEstimator, TransformerMixin):
    """Log-transform skewed numeric columns, standardize numeric columns, one-hot encode protocol_type, service and flag.
    Unseen categories become all zeros. Fit on training rows only."""

    LOG_COLS = ["duration", "src_bytes", "dst_bytes", "hot", "num_compromised", "num_root", "num_file_creations", "count", "srv_count",
                "dst_host_count", "dst_host_srv_count"]

    def __init__(self, drop=("num_outbound_cmds",)):
        self.drop = drop

    def _numeric(self, X):
        num = X[[c for c in NUM_COLS if c not in self.drop]].apply(pd.to_numeric, errors="coerce").astype(float).copy()
        for c in self.LOG_COLS:
            if c in num.columns:
                num[c] = np.log1p(num[c].clip(lower=0))
        return num

    def fit(self, X, y=None):
        num = self._numeric(X)
        self.num_cols_ = list(num.columns)
        self.mean_ = num.mean()
        self.std_ = num.std().replace(0, 1.0)
        self.cats_ = {c: sorted(X[c].astype(str).unique()) for c in CAT_COLS}
        self.feature_names_ = self.num_cols_ + [f"{c}={v}" for c in CAT_COLS for v in self.cats_[c]]
        return self

    def transform(self, X):
        num = (self._numeric(X)[self.num_cols_] - self.mean_) / self.std_
        parts = [num.values]
        for c in CAT_COLS:
            col = X[c].astype(str).values
            parts.append(np.column_stack([(col == v).astype(float) for v in self.cats_[c]]))
        return np.hstack(parts)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_, dtype=object)


class RuleDiscretizer(BaseEstimator, TransformerMixin):
    """Turn each connection into a set of items for association rules: categorical values, binary flags, rate columns in
    {zero, low, high} (fixed cut points 0 and 0.5) and counts in {zero, low, mid, high} (training quantile cut points)."""

    COUNT_COLS = [c for c in NUM_COLS if c not in RATE_COLS + BINARY_COLS + ["num_outbound_cmds"]]

    def fit(self, X, y=None):
        self.edges_ = {}
        for c in self.COUNT_COLS:
            v = pd.to_numeric(X[c], errors="coerce")
            nz = v[v > 0]
            self.edges_[c] = np.unique(np.quantile(nz, [1 / 3, 2 / 3])) if len(nz) > 20 else np.array([])
        return self

    def items(self, X) -> pd.DataFrame:
        out = {}
        for c in CAT_COLS:
            for v in sorted(X[c].astype(str).unique()):
                out[f"{c}={v}"] = (X[c].astype(str) == v).values
        for c in BINARY_COLS:
            out[f"{c}=1"] = (pd.to_numeric(X[c]) == 1).values
        for c in RATE_COLS:
            v = pd.to_numeric(X[c]).values
            out[f"{c}=zero"] = v == 0
            out[f"{c}=low"] = (v > 0) & (v <= 0.5)
            out[f"{c}=high"] = v > 0.5
        for c in self.COUNT_COLS:
            v = pd.to_numeric(X[c]).values
            e = self.edges_[c]
            out[f"{c}=zero"] = v == 0
            if len(e) == 2:
                out[f"{c}=low"] = (v > 0) & (v <= e[0])
                out[f"{c}=mid"] = (v > e[0]) & (v <= e[1])
                out[f"{c}=high"] = v > e[1]
            else:
                out[f"{c}=positive"] = v > 0
        return pd.DataFrame(out, index=X.index)

    def transform(self, X):
        return self.items(X)


def match_rules(item_row: pd.Series, rules):
    """Rules (dicts with `antecedent` list, `category`, `confidence`, `lift`, `support`) whose antecedent holds for one item row."""
    held = set(item_row.index[item_row.values])
    return [r for r in rules if set(r["antecedent"]) <= held]


class AnomalyDetector:
    """Detector trained on normal connections only. kind: 'iforest', 'pca_recon' (squared reconstruction error after PCA keeping 95% of
    the variance) or 'mahalanobis' (distance in PCA space). A higher score means more unusual."""

    def __init__(self, kind, seed=42):
        self.kind, self.seed = kind, seed

    def fit(self, df_normal):
        self.prep = KDDPrep().fit(df_normal)
        X = self.prep.transform(df_normal)
        if self.kind == "iforest":
            self.m = IsolationForest(n_estimators=200, random_state=self.seed, n_jobs=4).fit(X)
        else:
            self.pca = PCA(0.95, random_state=self.seed).fit(X)
            self.ev = self.pca.explained_variance_
        return self

    def score(self, df):
        X = self.prep.transform(df)
        if self.kind == "iforest":
            return -self.m.score_samples(X)
        Z = self.pca.transform(X)
        if self.kind == "pca_recon":
            return ((X - self.pca.inverse_transform(Z)) ** 2).sum(axis=1)
        return ((Z ** 2) / self.ev).sum(axis=1)
