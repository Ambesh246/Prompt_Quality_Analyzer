from pathlib import Path
import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.ml.model import PreferenceModel

client = TestClient(create_app())


def test_health_and_model_info():
    health = client.get("/health")
    info = client.get("/model-info")
    assert health.status_code == 200
    assert health.json()["model_available"] is True
    assert info.status_code == 200
    assert info.json()["version"].startswith("pqa-")
    assert len(info.json()["rubric_weights"]) == 10


def test_analyze_prompt_returns_ten_dimensions_and_bounded_score():
    response = client.post("/analyze", json={"prompt": "Explain photosynthesis in simple terms with one example."})
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["score"] <= 100
    assert len(body["dimensions"]) == 10
    assert body["model_score"] is not None
    assert body["model"]["scoring_mode"] == "colab_regression"
    assert body["model_preference_score"] is None
    assert body["model"]["version"]
    assert body["scoring_method"]


def test_whitespace_input_rejected():
    response = client.post("/analyze", json={"prompt": "   \n "})
    assert response.status_code == 422


def test_empty_or_missing_input_rejected():
    assert client.post("/analyze", json={"prompt": ""}).status_code == 422
    assert client.post("/analyze", json={}).status_code == 422


def test_long_prompt_rejected_by_schema():
    response = client.post("/analyze", json={"prompt": "x" * 8001})
    assert response.status_code == 422


def test_extra_fields_rejected():
    response = client.post("/analyze", json={"prompt": "Explain gravity", "unknown": True})
    assert response.status_code == 422


def test_rule_based_improvement_and_rescore():
    prompt = "help me with my Python code"
    response = client.post("/improve", json={"prompt": prompt, "use_provider": False})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["provider"] == "rule_based_fallback"
    assert body["rewritten_prompt"]
    assert body["improved_analysis"]["score"] >= 0
    assert body["score_delta"] == round(body["improved_analysis"]["score"] - body["original_analysis"]["score"], 1)
    assert "not an LLM-generated rewrite" in body["notice"]


def test_health_degraded_and_analysis_fails_if_artifact_missing(tmp_path: Path):
    service = PreferenceModel(tmp_path)
    app = create_app(service)
    test_client = TestClient(app)
    assert test_client.get("/health").json()["status"] == "degraded"
    response = test_client.post(
    "/analyze",
    json={"prompt": "Explain gravity"},
    )

    assert response.status_code == 200
    assert response.json()["scoring_method"] == "explainable_rubric_only"


def test_request_body_size_middleware():
    response = client.post("/analyze", content="x" * 100_001, headers={"content-type": "application/json", "content-length": "100001"})
    assert response.status_code == 413


def test_cors_allows_local_frontend_origin():
    app = create_app()
    test_client = TestClient(app)
    response = test_client.options(
        "/analyze",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_provider_failure_preserves_original_analysis(monkeypatch):
    import importlib
    main_module = importlib.import_module("app.main")

    async def fake_provider_failure(prompt, rubric, use_provider=True):
        return {
            "status": "provider_error",
            "provider": "openai",
            "error": "The rewrite provider could not complete the request.",
            "diagnostic_type": "TimeoutException",
        }

    monkeypatch.setattr(main_module, "improve_prompt", fake_provider_failure)
    test_client = TestClient(create_app())
    response = test_client.post("/improve", json={"prompt": "Analyze my data", "use_provider": True})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "provider_error"
    assert body["original_analysis"]["score"] is not None
    assert body["rewritten_prompt"] is None
    assert "could not complete" in body["error"]


def test_rate_limiter_returns_429_after_configured_limit():
    from app.main import InMemoryRateLimiter
    app = create_app()
    app.state.rate_limiter = InMemoryRateLimiter(5)
    test_client = TestClient(app)
    statuses = [test_client.post("/analyze", json={"prompt": "Explain gravity"}).status_code for _ in range(6)]
    assert statuses[:5] == [200] * 5
    assert statuses[5] == 429
