"""Метрики Prometheus и структурное логирование для churn API."""

import json
import logging
import sys
import time
from contextlib import contextmanager

from prometheus_client import Counter, Histogram

# ---------- Метрики Prometheus ----------

# Счётчик предсказаний с разбивкой по предсказанному классу
PREDICTIONS_TOTAL = Counter(
    "churn_predictions_total",
    "Total number of churn predictions",
    ["churn"],  # "true" / "false"
)

# Гистограмма latency (в секундах)
PREDICTION_LATENCY = Histogram(
    "churn_prediction_latency_seconds",
    "Latency of /predict endpoint in seconds",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

# Гистограмма предсказанных вероятностей — для детекции дрифта
PREDICTION_PROBABILITY = Histogram(
    "churn_probability",
    "Distribution of predicted churn probabilities",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
)


# ---------- Структурный логгер ----------


def get_logger(name: str = "churn-api") -> logging.Logger:
    """JSON-логгер в stdout — формат, который понимают все коллекторы."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


class JsonFormatter(logging.Formatter):
    """Форматтер: каждое сообщение — одна JSON-строка."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Если передали extra={"event": {...}} — добавляем
        if hasattr(record, "event"):
            payload["event"] = record.event
        return json.dumps(payload, ensure_ascii=False)


@contextmanager
def timed_latency():
    """Замеряет время блока и пишет в гистограмму PREDICTION_LATENCY."""
    start = time.perf_counter()
    try:
        yield
    finally:
        PREDICTION_LATENCY.observe(time.perf_counter() - start)
