from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "optic-pixel"


def test_index_page():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Optic Pixel" in resp.text
