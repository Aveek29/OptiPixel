import io
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_unsupported_mime_type():
    resp = client.post("/api/enhance", files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")})
    assert resp.status_code == 415


def test_unsupported_extension():
    resp = client.post("/api/enhance", files={"file": ("test.bmp", io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100), "image/bmp")})
    assert resp.status_code == 415


def test_oversized_upload():
    resp = client.post("/api/enhance", files={"file": ("large.jpg", io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * (11 * 1024 * 1024)), "image/jpeg")})
    assert resp.status_code == 413
