import io

import numpy as np
import tensorflow as tf


def png_bytes():
    image = np.random.randint(0, 255, (40, 40, 3), dtype="uint8")
    return tf.io.encode_png(image).numpy()


def upload(client, name, data):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)}, content_type="multipart/form-data")


def test_index_page(client):
    assert client.get("/").status_code == 200


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_valid_upload(client):
    response = upload(client, "photo.png", png_bytes())
    assert response.status_code == 200
    assert b"Confidence" in response.data


def test_wrong_file_type(client):
    response = upload(client, "notes.txt", b"hello")
    assert response.status_code == 400
    assert b"Only PNG and JPG" in response.data


def test_undecodable_image(client):
    response = upload(client, "fake.jpg", b"not an image")
    assert response.status_code == 400
    assert b"could not be decoded" in response.data


def test_file_too_large(client):
    response = upload(client, "big.png", b"x" * (2 * 1024 * 1024))
    assert response.status_code == 413


def test_history_records_upload(client):
    upload(client, "../history check.png", png_bytes())
    response = client.get("/history")
    assert b"history_check.png" in response.data


def test_unknown_page(client):
    assert client.get("/missing").status_code == 404


def test_multiple_uploads(client):
    data = {"image": [(io.BytesIO(png_bytes()), "a.png"), (io.BytesIO(png_bytes()), "b.png")]}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert response.data.count(b"Confidence") == 2
