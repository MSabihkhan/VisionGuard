import os
import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv()  # reads .env; the values never appear in code
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("toy_mnist")

(x, y), _ = tf.keras.datasets.mnist.load_data()
x = x[:10000] / 255.0
y = y[:10000]

lr = 0.001  # change this to 0.01 for your second run

with mlflow.start_run(run_name=f"toy_lr_{lr}"):
    mlflow.log_params({"lr": lr, "epochs": 5, "units": 64})
    mlflow.set_tag("purpose", "3.2 toy run")

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    h = model.fit(x, y, epochs=5, validation_split=0.2, verbose=2)

    for epoch in range(5):
        for k, v in h.history.items():
            mlflow.log_metric(k, v[epoch], step=epoch)

    plt.plot(h.history["loss"], label="train")
    plt.plot(h.history["val_loss"], label="val")
    plt.legend()
    plt.savefig("loss.png")
    mlflow.log_artifact("loss.png")

    mlflow.keras.log_model(model, "model")