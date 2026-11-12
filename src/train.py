"""Обучение baseline-модели Telco Churn с логированием в MLflow."""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data.load import basic_report, load_raw
from src.features.build import build_pipeline, split_xy

load_dotenv()

RANDOM_STATE = 42
TEST_SIZE = 0.2
ARTIFACTS_DIR = Path("artifacts")
EXPERIMENT_NAME = "telco-churn"


def main() -> None:
    # 1. Данные
    df = load_raw()
    report = basic_report(df)
    print("Data report:", report)

    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 2. MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="logreg-baseline") as run:
        # Логируем параметры данных и модели
        mlflow.log_params(
            {
                "model": "LogisticRegression",
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "n_rows": report["n_rows"],
                "target_rate": round(report["target_rate"], 4),
            }
        )

        # 3. Обучение
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)

        # 4. Метрики
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }
        mlflow.log_metrics({k: round(v, 4) for k, v in metrics.items()})
        print("Metrics:", {k: round(v, 4) for k, v in metrics.items()})

        # 5. Сохраняем модель и в MLflow, и локально
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            skops_trusted_types=["numpy.dtype"],
        )

        ARTIFACTS_DIR.mkdir(exist_ok=True)
        local_path = ARTIFACTS_DIR / "model.joblib"
        joblib.dump(pipeline, local_path)
        print(f"Model saved to {local_path}")

        print(f"MLflow run_id: {run.info.run_id}")


if __name__ == "__main__":
    main()
