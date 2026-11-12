"""Тесты FastAPI-сервиса Telco Churn."""

import pytest
from fastapi.testclient import TestClient

from src.serve.app import app


@pytest.fixture(scope="module")
def client():
    """TestClient с контекстом — триггерит lifespan (загрузку модели)."""
    with TestClient(app) as c:
        yield c


VALID_PAYLOAD = {
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


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_ok(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["churn"] in (True, False)
    assert 0.0 <= body["probability"] <= 1.0
    assert "model_version" in body


def test_predict_with_null_total_charges(client):
    """TotalCharges=null должен обрабатываться (новый клиент)."""
    payload = {**VALID_PAYLOAD, "tenure": 0, "TotalCharges": None}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text


def test_predict_invalid_contract(client):
    """Невалидная категория -> 422 от pydantic."""
    payload = {**VALID_PAYLOAD, "Contract": "Lifetime"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_missing_field(client):
    """Пропущенное поле -> 422."""
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "tenure"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_out_of_range(client):
    """tenure=-5 -> 422 (валидация Field(ge=0))."""
    payload = {**VALID_PAYLOAD, "tenure": -5}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
