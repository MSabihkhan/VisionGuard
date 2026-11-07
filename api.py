import hmac
import logging
import os

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

import inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ALLOWED = {"png", "jpg", "jpeg"}
MAX_MB = int(os.environ.get("MAX_UPLOAD_MB", 5))
API_KEY = os.environ.get("API_KEY", "")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024


def key_ok():
    sent = request.headers.get("X-API-Key", "")
    return API_KEY != "" and hmac.compare_digest(sent, API_KEY)


@app.get("/health")
def health():
    return jsonify(status="ok", model=inference.MODEL_NAME, threshold=inference.THRESHOLD)


@app.post("/predict")
def predict():
    if not key_ok():
        app.logger.warning("rejected request: bad or missing API key from %s", request.remote_addr)
        return jsonify(error="invalid or missing API key"), 401

    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="send an image file in the 'image' field"), 400

    filename = secure_filename(file.filename)
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED:
        return jsonify(error="only PNG and JPG images are allowed"), 400

    try:
        label, confidence, ai_score = inference.predict(file.read())
    except ValueError:
        return jsonify(error="the file could not be decoded as an image"), 400

    app.logger.info("predict file=%s label=%s confidence=%.4f ai_score=%.4f", filename, label, confidence, ai_score)
    return jsonify(
        filename=filename,
        label=label,
        confidence=round(confidence, 4),
        ai_score=round(ai_score, 4),
        threshold=inference.THRESHOLD,
        model=inference.MODEL_NAME,
    )


@app.errorhandler(404)
def not_found(error):
    return jsonify(error="not found"), 404


@app.errorhandler(405)
def wrong_method(error):
    return jsonify(error="method not allowed"), 405


@app.errorhandler(413)
def too_large(error):
    return jsonify(error=f"file too large, the limit is {MAX_MB} MB"), 413


@app.errorhandler(500)
def server_error(error):
    return jsonify(error="internal server error"), 500


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 8000)), debug=False)
