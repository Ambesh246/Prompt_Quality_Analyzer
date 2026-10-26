from fastapi.testclient import TestClient
from app.main import create_app


def test_end_to_end_without_provider_key():
    client = TestClient(create_app())
    original = "help me plan a study schedule"
    analyzed = client.post("/analyze", json={"prompt": original})
    assert analyzed.status_code == 200
    analysis = analyzed.json()
    assert 0 <= analysis["score"] <= 100

    improved = client.post("/improve", json={"prompt": original, "use_provider": False})
    assert improved.status_code == 200
    data = improved.json()
    assert data["status"] == "success"
    assert data["original_analysis"]["score"] == analysis["score"]
    assert data["rewritten_prompt"].strip()
    assert 0 <= data["improved_analysis"]["score"] <= 100
    assert data["provider"] == "rule_based_fallback"
