"""
Phase 1, Step 1 test: confirm the app boots and /health responds correctly.

This is intentionally the simplest possible test — it exists to catch
"the app doesn't even start" failures (bad import, bad config, missing
dependency) before we build anything on top of it.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["app_name"] == "VidSense AI"
    # These fields exist because config.py should always report which
    # Whisper setup this deployment is using -- useful for debugging
    # "why is transcription slow/fast on this machine" later.
    assert "whisper_model_size" in body
    assert "whisper_device" in body
