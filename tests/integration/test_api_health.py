"""Integration test: FastAPI health endpoint via TestClient."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["app"] == "InsureFlow AI"
    assert isinstance(body["demo_mode"], bool)


def test_openapi_docs_available():
    assert client.get("/openapi.json").status_code == 200
