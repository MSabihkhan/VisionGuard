import base64
import os

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

import client
import db

API_URL = os.environ.get("API_URL", "").rstrip("/")

if not API_URL:
    import inference

ALLOWED = {"png", "jpg", "jpeg"}
MAX_MB = int(os.environ.get("MAX_UPLOAD_MB", 5))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
db.init_db()


def classify(raw, filename):
    if API_URL:
        return client.predict(raw, filename)
    label, confidence, ai_score = inference.predict(raw)
    return label, confidence, ai_score, inference.MODEL_NAME


def fail(message, status=400):
    return render_template("index.html", error=message, max_mb=MAX_MB), status


@app.get("/")
def index():
    return render_template("index.html", max_mb=MAX_MB)


@app.post("/predict")
def predict():
    files = [f for f in request.files.getlist("image") if f.filename]
    if not files:
        return fail("Please choose an image file.")

    results = []
    errors = []
    for file in files:
        filename = secure_filename(file.filename)
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if extension not in ALLOWED:
            errors.append(f"{filename}: Only PNG and JPG images are allowed.")
            continue

        raw = file.read()
        try:
            label, confidence, ai_score, model_name = classify(raw, filename)
        except ValueError:
            errors.append(f"{filename}: This file looks like an image but could not be decoded.")
            continue
        except client.ApiError:
            return fail("The prediction service is not reachable right now. Please try again.", 502)

        db.add_prediction(filename, label, confidence, ai_score, model_name)
        results.append({
            "filename": filename,
            "label": label,
            "confidence": confidence,
            "ai_score": ai_score,
            "model_name": model_name,
            "preview": f"data:image/{extension};base64," + base64.b64encode(raw).decode(),
        })

    if not results:
        return fail(" ".join(errors))
    return render_template("result.html", results=results, errors=errors)


@app.get("/history")
def history():
    return render_template("history.html", rows=db.recent())


@app.get("/health")
def health():
    try:
        db.ping()
    except Exception:
        return jsonify(status="unhealthy", database="down"), 503
    if API_URL:
        return jsonify(status="ok", database="ok", backend="api", api_url=API_URL)
    return jsonify(status="ok", database="ok", backend="local", model=inference.MODEL_NAME, threshold=inference.THRESHOLD)


@app.errorhandler(404)
def not_found(error):
    return render_template("error.html", code=404, message="That page does not exist."), 404


@app.errorhandler(413)
def too_large(error):
    return render_template("error.html", code=413, message=f"The upload is too large. The limit is {MAX_MB} MB."), 413


@app.errorhandler(500)
def server_error(error):
    return render_template("error.html", code=500, message="Something went wrong on our side. Please try again."), 500


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 5000)), debug=False)
