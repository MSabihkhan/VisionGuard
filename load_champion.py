import os
import sys

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient

from preprocess import prepare

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

NAME = "VisionGuard"
version = MlflowClient().get_model_version_by_alias(NAME, "champion")
run = MlflowClient().get_run(version.run_id)
threshold = float(run.data.params["threshold"])
img = int(run.data.params["img_size"])

model = mlflow.keras.load_model(f"models:/{NAME}@champion")
print("loaded", NAME, "version", version.version, "by alias champion")

raw = open(sys.argv[1], "rb").read()
x = prepare(raw, img)[None, ...]
prob = float(model.predict(x, verbose=0)[0][0])
label = "AI-generated" if prob >= threshold else "Real"
confidence = prob if prob >= threshold else 1 - prob
print("image:", sys.argv[1])
print("label:", label, "| P(AI):", round(prob, 4), "| confidence:", round(confidence, 4))
