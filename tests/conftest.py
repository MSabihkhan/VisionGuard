import os
import tempfile
from pathlib import Path

import pytest
import tensorflow as tf

folder = Path(tempfile.mkdtemp())

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(160, 160, 3)),
    tf.keras.layers.Rescaling(1 / 255),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.save(folder / "model.keras")

os.environ["MODEL_PATH"] = str(folder / "model.keras")
os.environ["DATABASE_URL"] = "sqlite:///" + (folder / "test.db").as_posix()
os.environ["THRESHOLD"] = "0.5"
os.environ["MAX_UPLOAD_MB"] = "1"
os.environ["API_KEY"] = "test-key"
os.environ["API_URL"] = ""


@pytest.fixture
def client():
    import app

    return app.app.test_client()


@pytest.fixture
def api_client():
    import api

    return api.app.test_client()
