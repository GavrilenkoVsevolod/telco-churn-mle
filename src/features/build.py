"""Сборка sklearn-пайплайна: препроцессинг + модель."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def build_pipeline() -> Pipeline:
    """Возвращает необученный Pipeline: препроцессинг + LogisticRegression."""
    # Определяем колонки по типу
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_cols = [
        c
        for c in [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ]
    ]

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
