import io

import numpy as np
import tensorflow as tf

KEY = {"X-API-Key": "test-key"}


def png_bytes():
    image = np.random.randint(0, 255, (40, 40, 3), dtype="uint8")
    return tf.io.encode_png(image).numpy()


def send(client, name, data, headers=KEY):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)}, headers=headers, content_type="multipart/form-data")


def test_health_needs_no_key(api_client):
    assert api_client.get("/health").get_json()["status"] == "ok"


def test_predict_returns_json(api_client):
    response = send(api_client, "photo.png", png_bytes())
    body = response.get_json()
    assert response.status_code == 200
    assert body["label"] in ("AI-generated", "Real")
    assert 0 <= body["confidence"] <= 1
    assert body["filename"] == "photo.png"


def test_missing_key_rejected(api_client):
    assert send(api_client, "photo.png", png_bytes(), headers={}).status_code == 401


def test_wrong_key_rejected(api_client):
    assert send(api_client, "photo.png", png_bytes(), headers={"X-API-Key": "nope"}).status_code == 401


def test_no_file(api_client):
    response = api_client.post("/predict", headers=KEY)
    assert response.status_code == 400


def test_wrong_type(api_client):
    assert send(api_client, "notes.txt", b"hello").status_code == 400


def test_undecodable(api_client):
    assert send(api_client, "fake.jpg", b"not an image").status_code == 400


def test_too_large(api_client):
    assert send(api_client, "big.png", b"x" * (2 * 1024 * 1024)).status_code == 413
