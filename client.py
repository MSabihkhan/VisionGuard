import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = os.environ.get("API_URL", "").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
TIMEOUT = float(os.environ.get("API_TIMEOUT", 60))


class ApiError(Exception):
    pass


def predict(raw, filename):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            headers={"X-API-Key": API_KEY},
            files={"image": (filename, raw)},
            timeout=TIMEOUT,
        )
    except requests.RequestException:
        raise ApiError("could not reach the prediction service")
    if response.status_code == 400:
        raise ValueError(response.json().get("error", "bad image"))
    if response.status_code != 200:
        raise ApiError(f"prediction service returned {response.status_code}")
    data = response.json()
    return data["label"], data["confidence"], data["ai_score"], data["model"]
