"""
Minimal proof that the Jarvis foundation works: the server starts and the
/health endpoint responds correctly.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "Rhasspy"
