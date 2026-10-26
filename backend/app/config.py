from __future__ import annotations
import os


def allowed_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    return [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]


def max_requests_per_minute() -> int:
    try:
        return max(5, min(600, int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))))
    except ValueError:
        return 30
