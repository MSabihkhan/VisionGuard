"""Task 2.7 (toy part) - Keras Tuner on Fashion-MNIST.

Searches learning rate and number of units with random search (8 trials),
then prints the results table and the best settings.

Run from the project root:  python toyrun/keras_tuner_toy.py
"""
import keras_tuner as kt
import tensorflow as tf

tf.keras.utils.set_random_seed(42)  # fixes Python, NumPy and TF seeds together

(x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
x_train = (x_train / 255.0).astype("float32")

# Subset keeps the toy fast; validation_split below carves out the validation data
x_train, y_train = x_train[:20000], y_train[:20000]


def build_model(hp):
    """Called once per trial with a fresh set of hyperparameter values."""
    units = hp.Int("units", min_value=32, max_value=256, step=32)
    lr = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",   # what the tuner tries to maximise
    max_trials=8,
    executions_per_trial=1,
    directory="toy_tuner",
    project_name="fashion_mnist",
    overwrite=True,
    seed=42,
)

tuner.search(x_train, y_train, epochs=3, validation_split=0.2, verbose=2)

print("\n===== Results summary (top trials) =====")
tuner.results_summary()

best_hp = tuner.get_best_hyperparameters(1)[0]
print("\n===== Best settings =====")
print("units         :", best_hp.get("units"))
print("learning_rate :", best_hp.get("learning_rate"))

# Compact table of every trial, handy for the report
print("\n===== All trials =====")
print(f"{'trial':<8}{'units':<8}{'learning_rate':<16}{'val_accuracy':<12}")
for t in tuner.oracle.get_best_trials(num_trials=8):
    print(
        f"{t.trial_id:<8}{t.hyperparameters.get('units'):<8}"
        f"{t.hyperparameters.get('learning_rate'):<16.5f}{t.score:<12.4f}"
    )