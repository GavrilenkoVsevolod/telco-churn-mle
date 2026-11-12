"""FastAPI-сервис для предсказания оттока клиентов Telco."""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from src.serve.schemas import CustomerFeatures, PredictionResponse

load_dotenv()

MODEL_PATH = Path("artifacts/model.joblib")
MODEL_VERSION = "local-joblib-v1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружаем модель один раз при старте, держим в app.state."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Модель не найдена: {MODEL_PATH}. Сначала запусти `uv run python -m src.train`."
        )
    app.state.model = joblib.load(MODEL_PATH)
    app.state.model_version = MODEL_VERSION
    yield
    # shutdown: ничего чистить не нужно


app = FastAPI(
    title="Telco Churn API",
    version="0.1.0",
    description="Предсказание вероятности оттока клиента телеком-оператора.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Liveness-проба. Возвращает ok, если модель загружена."""
    return {
        "status": "ok",
        "model_version": getattr(app.state, "model_version", "not-loaded"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures) -> PredictionResponse:
    """Предсказание вероятности оттока по признакам клиента."""
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    # pydantic -> pandas DataFrame (ровно те колонки, что модель видела на обучении)
    df = pd.DataFrame([features.model_dump()])

    try:
        proba = float(model.predict_proba(df)[0, 1])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Ошибка предсказания: {e}") from e

    return PredictionResponse(
        churn=proba >= 0.5,
        probability=round(proba, 4),
        model_version=app.state.model_version,
    )


@app.get("/model-info")
def model_info() -> dict:
    """Показывает, откуда взялась модель и какие у неё метрики (из MLflow)."""
    info = {"model_path": str(MODEL_PATH), "exists": MODEL_PATH.exists()}

    try:
        client = mlflow.MlflowClient()
        exp = client.get_experiment_by_name("telco-churn")
        if exp is not None:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs:
                info["mlflow_run_id"] = runs[0].info.run_id
                info["mlflow_metrics"] = runs[0].data.metrics
    except Exception as e:  # noqa: BLE001
        info["mlflow_error"] = str(e)

    return info
