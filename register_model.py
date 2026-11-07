import os

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

NAME = "VisionGuard"
client = MlflowClient()

runs = mlflow.search_runs(
    experiment_names=["visionguard"],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["metrics.val_f1 DESC", "metrics.best_val_loss ASC"],
    output_format="list",
)
print("top runs by validation F1:")
for r in runs[:5]:
    print(r.data.tags.get("mlflow.runName"), round(r.data.metrics["val_f1"], 4), round(r.data.metrics["best_val_loss"], 4))

experiment = mlflow.get_experiment_by_name("visionguard")
logged = mlflow.search_logged_models(experiment_ids=[experiment.experiment_id], output_format="list")
model_of_run = {m.source_run_id: m.model_id for m in logged}

second, best = runs[1], runs[0]
v_second = mlflow.register_model(f"models:/{model_of_run[second.info.run_id]}", NAME)
v_best = mlflow.register_model(f"models:/{model_of_run[best.info.run_id]}", NAME)

client.set_registered_model_alias(NAME, "champion", v_best.version)
print("version", v_second.version, "from run", second.data.tags.get("mlflow.runName"))
print("version", v_best.version, "from run", best.data.tags.get("mlflow.runName"), "-> alias champion")
