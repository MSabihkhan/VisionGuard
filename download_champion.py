import json
import os
import shutil
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

NAME = "VisionGuard"
client = MlflowClient()
version = client.get_model_version_by_alias(NAME, "champion")
run = client.get_run(version.run_id)

out = Path("models/champion")
out.mkdir(parents=True, exist_ok=True)
folder = mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{NAME}@champion", dst_path="models/champion_download")
model_file = next(Path(folder).rglob("*.keras"))
shutil.copy(model_file, out / "model.keras")
shutil.rmtree("models/champion_download")

info = {
    "name": NAME,
    "version": version.version,
    "run_id": version.run_id,
    "threshold": float(run.data.params["threshold"]),
    "img_size": int(run.data.params["img_size"]),
}
(out / "info.json").write_text(json.dumps(info, indent=2))
print(info)
