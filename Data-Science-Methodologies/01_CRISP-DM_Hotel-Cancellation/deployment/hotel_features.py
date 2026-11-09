"""Shared feature engineering for the hotel non-arrival project.

Used by the notebook (training) and the Streamlit app (serving), so that both
build features in exactly the same way. All functions are row-wise or fit on
training data only, with no use of the target.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
MONTH_NUM = {m: i for i, m in enumerate(MONTHS, 1)}

# Raw columns the app must supply (Feature Set A inputs)
RAW_A_COLS = [
    "hotel", "lead_time", "arrival_date_year", "arrival_date_month",
    "arrival_date_week_number", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
    "babies", "meal", "market_segment", "distribution_channel",
    "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "reserved_room_type", "customer_type",
    "agent", "company",
]
# Extra raw columns for the diagnostic (not deployed) Set B models
RAW_B_COLS = ["country", "booking_changes", "assigned_room_type",
              "days_in_waiting_list", "adr", "required_car_parking_spaces",
              "total_of_special_requests"]

NUM_A = ["lead_time", "log_lead_time", "arrival_date_week_number",
         "arrival_date_day_of_month", "stays_in_weekend_nights",
         "stays_in_week_nights", "total_nights", "adults", "children",
         "babies", "total_guests", "previous_cancellations",
         "previous_bookings_not_canceled", "arrival_weekday",
         "is_repeated_guest", "has_children", "has_agent", "has_company",
         "zero_nights"]
CAT_A = ["hotel", "arrival_month", "meal", "market_segment",
         "distribution_channel", "reserved_room_type", "customer_type",
         "agent_id"]
NUM_B = ["booking_changes", "days_in_waiting_list", "adr",
         "required_car_parking_spaces", "total_of_special_requests",
         "room_changed"]
CAT_B = ["country", "assigned_room_type"]
CAT_DEP = ["deposit_type"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature construction. No target, no fitted state."""
    x = df.copy()
    x["children"] = x["children"].fillna(0)
    x["total_nights"] = x["stays_in_weekend_nights"] + x["stays_in_week_nights"]
    x["zero_nights"] = (x["total_nights"] == 0).astype(int)
    x["total_guests"] = x["adults"] + x["children"] + x["babies"]
    x["has_children"] = ((x["children"] + x["babies"]) > 0).astype(int)
    x["log_lead_time"] = np.log1p(x["lead_time"])
    month_num = x["arrival_date_month"].map(MONTH_NUM)
    arr = pd.to_datetime(dict(year=x["arrival_date_year"], month=month_num,
                              day=x["arrival_date_day_of_month"]))
    x["arrival_weekday"] = arr.dt.weekday
    x["arrival_month"] = x["arrival_date_month"]
    x["meal"] = x["meal"].replace({"Undefined": "SC"})
    x["has_agent"] = x["agent"].notna().astype(int)
    x["has_company"] = x["company"].notna().astype(int)
    x["agent_id"] = x["agent"].apply(lambda v: "none" if pd.isna(v) else str(int(v)))
    if "assigned_room_type" in x.columns:
        x["room_changed"] = (x["assigned_room_type"] != x["reserved_room_type"]).astype(int)
    return x


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip numeric columns at training-set quantiles (fit on train only)."""

    def __init__(self, lower=0.0, upper=0.999):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lo_ = X.quantile(self.lower).values
        self.hi_ = X.quantile(self.upper).values
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_pipeline(clf, num_cols, cat_cols, min_freq=200):
    """Feature engineering + preprocessing + classifier, one fitted object."""
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper(0.0, 0.999)),
                    ("sc", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist",
                        min_frequency=min_freq, sparse_output=False)
    pre = ColumnTransformer([("num", num, list(num_cols)),
                             ("cat", cat, list(cat_cols))])
    return Pipeline([("fe", FunctionTransformer(add_features, validate=False)),
                     ("pre", pre), ("clf", clf)])
