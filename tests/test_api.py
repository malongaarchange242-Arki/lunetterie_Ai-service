import os
import sys
from io import BytesIO

from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.api.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_analyze_endpoint_accepts_image() -> None:
    image = Image.new("RGB", (256, 256), color="blue")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post(
        "/analyze",
        files={"file": ("test.png", buffer.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["shape"]
    assert payload["color"]
