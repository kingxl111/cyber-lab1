from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
import scipy.sparse as sp


TARGET = "booking_complete"

BASE_NUMERIC = [
    "num_passengers",
    "purchase_lead",
    "length_of_stay",
    "flight_hour",
    "flight_duration",
    "wants_extra_baggage",
    "wants_preferred_seat",
    "wants_in_flight_meals",
]

BASE_CATEGORICAL = [
    "sales_channel",
    "trip_type",
    "flight_day",
    "route",
    "booking_origin",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    x["flight_hour_sin"] = np.sin(2 * np.pi * x["flight_hour"] / 24.0)
    x["flight_hour_cos"] = np.cos(2 * np.pi * x["flight_hour"] / 24.0)
    x["is_weekend"] = x["flight_day"].isin(["Sat", "Sun"]).astype(int)
    x["lead_log"] = np.log1p(x["purchase_lead"])
    x["stay_log"] = np.log1p(x["length_of_stay"])
    x["passengers_per_day"] = x["num_passengers"] / (x["length_of_stay"] + 1.0)
    x["lead_per_stay"] = x["purchase_lead"] / (x["length_of_stay"] + 1.0)
    x["has_addons"] = (
        (x["wants_extra_baggage"] == 1)
        | (x["wants_preferred_seat"] == 1)
        | (x["wants_in_flight_meals"] == 1)
    ).astype(int)

    return x


def _one_hot_encoder(min_frequency=None):
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            min_frequency=min_frequency,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
            min_frequency=min_frequency,
        )


def _to_dense(x):
    if sp.issparse(x):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


@dataclass
class SplitData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer
    feature_names: list[str]


def make_splits(
    csv_path: str,
    use_engineered_features: bool = False,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    min_frequency: int | None = None,
) -> SplitData:
    df = pd.read_csv(csv_path, encoding="cp1252").dropna(subset=[TARGET]).copy()

    # df = pd.read_csv(csv_path).dropna(subset=[TARGET]).copy()

    if use_engineered_features:
        df = add_engineered_features(df)
        numeric_cols = BASE_NUMERIC + [
            "flight_hour_sin",
            "flight_hour_cos",
            "is_weekend",
            "lead_log",
            "stay_log",
            "passengers_per_day",
            "lead_per_stay",
            "has_addons",
        ]
    else:
        numeric_cols = BASE_NUMERIC

    categorical_cols = BASE_CATEGORICAL

    X = df[numeric_cols + categorical_cols].copy()
    y = df[TARGET].astype(int).values

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    val_rel_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_rel_size,
        random_state=random_state,
        stratify=y_train_full,
    )

    num_scaler = RobustScaler() if use_engineered_features else StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", num_scaler),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", _one_hot_encoder(min_frequency=min_frequency)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    X_train_p = _to_dense(preprocessor.fit_transform(X_train))
    X_val_p = _to_dense(preprocessor.transform(X_val))
    X_test_p = _to_dense(preprocessor.transform(X_test))

    feature_names = []
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        pass

    return SplitData(
        X_train=X_train_p,
        X_val=X_val_p,
        X_test=X_test_p,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_names=feature_names,
    )
