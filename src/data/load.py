"""Загрузка и базовая валидация датасета Telco Churn."""

from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

TARGET = "Churn"

# Колонки, которые точно есть в датасете — защита от «не тот файл»
REQUIRED_COLUMNS = {
    "customerID",
    "gender",
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
    TARGET,
}


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Читает CSV, проверяет схему, чистит типы. Возвращает DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Датасет не найден: {path}. Скачай с Kaggle и положи в data/raw/.")

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"В датасете нет колонок: {sorted(missing)}")

    # TotalCharges приходит строкой с пробелами (у новых клиентов tenure=0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Целевая в Yes/No -> 0/1
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype("int8")

    # customerID не нужен как фича
    df = df.drop(columns=["customerID"])

    return df


def basic_report(df: pd.DataFrame) -> dict:
    """Короткая сводка — пригодится для логов."""
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "target_rate": float(df[TARGET].mean()),
        "n_missing_total_charges": int(df["TotalCharges"].isna().sum()),
    }


if __name__ == "__main__":
    df = load_raw()
    print(basic_report(df))
    print(df.head())
    print(df.dtypes)
