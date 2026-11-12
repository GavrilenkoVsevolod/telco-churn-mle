"""Pydantic-схемы для Telco Churn API."""

from typing import Literal

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Признаки одного клиента — ровно те, что ждёт Pipeline."""

    # Числовые
    tenure: int = Field(..., ge=0, le=100, description="Месяцев с компанией")
    MonthlyCharges: float = Field(..., ge=0, description="Ежемесячный платёж")
    TotalCharges: float | None = Field(None, ge=0, description="Всего оплачено (может быть null)")

    # Категориальные
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tenure": 12,
                    "MonthlyCharges": 70.5,
                    "TotalCharges": 846.0,
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "Yes",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Ответ сервиса."""

    churn: bool = Field(..., description="Предсказанный класс")
    probability: float = Field(..., ge=0, le=1, description="Вероятность оттока")
    model_version: str = Field(..., description="Версия/run_id модели")
