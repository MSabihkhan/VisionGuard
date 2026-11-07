import sys

import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("toy_mnist_local")

lr = float(sys.argv[1])

(x, y), _ = tf.keras.datasets.mnist.load_data()
x = x[:10000] / 255.0
y = y[:10000]

with mlflow.start_run(run_name=f"toy_lr_{lr}"):
    mlflow.log_params({"lr": lr, "epochs": 5, "units": 64})
    mlflow.set_tag("purpose", "3.1 local toy run")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h = model.fit(x, y, epochs=5, validation_split=0.2, verbose=2)

    for epoch in range(5):
        for k, v in h.history.items():
            mlflow.log_metric(k, v[epoch], step=epoch)

    plt.plot(h.history["loss"], label="train")
    plt.plot(h.history["val_loss"], label="val")
    plt.legend()
    plt.savefig("toyrun/loss_local.png")
    mlflow.log_artifact("toyrun/loss_local.png")

    mlflow.keras.log_model(model, name="model")
