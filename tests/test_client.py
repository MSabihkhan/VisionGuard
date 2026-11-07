import io

import numpy as np
import pytest
import requests
import tensorflow as tf

import client


class FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self.body = body

    def json(self):
        return self.body


def png_bytes():
    image = np.random.randint(0, 255, (40, 40, 3), dtype="uint8")
    return tf.io.encode_png(image).numpy()


def test_predict_parses_response(monkeypatch):
    body = {"label": "Real", "confidence": 0.9, "ai_score": 0.1, "model": "VisionGuard v2"}
    monkeypatch.setattr(client.requests, "post", lambda *a, **k: FakeResponse(200, body))
    assert client.predict(b"x", "a.png") == ("Real", 0.9, 0.1, "VisionGuard v2")


def test_bad_image_becomes_value_error(monkeypatch):
    monkeypatch.setattr(client.requests, "post", lambda *a, **k: FakeResponse(400, {"error": "bad"}))
    with pytest.raises(ValueError):
        client.predict(b"x", "a.png")


def test_server_error_becomes_api_error(monkeypatch):
    monkeypatch.setattr(client.requests, "post", lambda *a, **k: FakeResponse(502, {}))
    with pytest.raises(client.ApiError):
        client.predict(b"x", "a.png")


def test_unreachable_becomes_api_error(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError()

    monkeypatch.setattr(client.requests, "post", boom)
    with pytest.raises(client.ApiError):
        client.predict(b"x", "a.png")


def test_flask_uses_remote_api(monkeypatch):
    import app

    calls = []

    def fake(raw, filename):
        calls.append(filename)
        return "AI-generated", 0.8, 0.8, "VisionGuard v9"

    monkeypatch.setattr(app, "API_URL", "http://fake")
    monkeypatch.setattr(app.client, "predict", fake)
    data = {"image": [(io.BytesIO(png_bytes()), "one.png"), (io.BytesIO(png_bytes()), "two.png")]}
    response = app.app.test_client().post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert calls == ["one.png", "two.png"]
    assert b"VisionGuard v9" in response.data


def test_flask_reports_unreachable_api(monkeypatch):
    import app

    def down(raw, filename):
        raise client.ApiError("down")

    monkeypatch.setattr(app, "API_URL", "http://fake")
    monkeypatch.setattr(app.client, "predict", down)
    data = {"image": (io.BytesIO(png_bytes()), "one.png")}
    response = app.app.test_client().post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 502
