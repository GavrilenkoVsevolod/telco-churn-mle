# Telco Churn MLE Project

End-to-end ML service для предсказания оттока клиентов телеком-оператора.

## Стек
- **ML:** scikit-learn (Pipeline + LogisticRegression), MLflow (трекинг)
- **API:** FastAPI + Pydantic
- **Инфраструктура:** Docker, docker-compose, uv
- **Качество:** pytest, ruff

## Структура
- `src/data/load.py` — загрузка и валидация
- `src/features/build.py` — sklearn-пайплайн
- `src/train.py` — обучение + логирование в MLflow
- `src/serve/` — FastAPI-сервис
- `tests/` — тесты API

## Быстрый старт

```bash
# Установка
uv sync

# Обучение (создаст artifacts/model.joblib и mlflow.db)
uv run python -m src.train

# Тесты
uv run pytest -v

# Запуск API
uv run uvicorn src.serve.app:app --reload --port 8000
# Swagger: http://localhost:8000/docs

# Или через Docker
docker compose up --build