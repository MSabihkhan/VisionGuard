import json
import os
from pathlib import Path

import tensorflow as tf
from dotenv import load_dotenv

from preprocess import prepare

load_dotenv()

MODEL_PATH = os.environ.get("MODEL_PATH", "models/champion/model.keras")
info_path = Path(MODEL_PATH).with_name("info.json")
info = json.loads(info_path.read_text()) if info_path.exists() else {}

THRESHOLD = float(os.environ.get("THRESHOLD") or info.get("threshold", 0.5))
MODEL_NAME = f"{info.get('name', 'VisionGuard')} v{info.get('version', '?')}"

model = tf.keras.models.load_model(MODEL_PATH)
IMG = model.input_shape[1]


def predict(raw_bytes):
    try:
        image = prepare(raw_bytes, IMG)
    except tf.errors.InvalidArgumentError:
        raise ValueError("image could not be decoded")
    prob = float(model(image[None, ...], training=False)[0][0])
    if prob >= THRESHOLD:
        return "AI-generated", prob, prob
    return "Real", 1 - prob, prob
