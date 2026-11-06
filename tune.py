import keras_tuner as kt
import pandas as pd
import tensorflow as tf

from data import IMG, SEED, make_ds

tf.keras.utils.set_random_seed(SEED)

train_ds = make_ds("splits/train.csv", train=True)
val_ds = make_ds("splits/val.csv")


def build_model(hp):
    lr = hp.Float("learning_rate", 1e-5, 1e-3, sampling="log")
    dropout = hp.Float("dropout", 0.2, 0.6, step=0.1)
    unfrozen = hp.Int("unfrozen_layers", 0, 60, step=20)

    base = tf.keras.applications.MobileNetV2(input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet")
    base.trainable = False
    for layer in base.layers[len(base.layers) - unfrozen:]:
        layer.trainable = True

    inputs = tf.keras.Input(shape=(IMG, IMG, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


tuner = kt.RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=10,
    executions_per_trial=1,
    directory="tuner",
    project_name="visionguard",
    overwrite=True,
    seed=SEED,
)

stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
tuner.search(train_ds, validation_data=val_ds, epochs=10, callbacks=[stop], verbose=2)

tuner.results_summary()

rows = []
for t in tuner.oracle.get_best_trials(num_trials=10):
    rows.append({
        "trial": t.trial_id,
        "learning_rate": t.hyperparameters.get("learning_rate"),
        "dropout": t.hyperparameters.get("dropout"),
        "unfrozen_layers": t.hyperparameters.get("unfrozen_layers"),
        "val_loss": t.score,
    })
table = pd.DataFrame(rows).round(5)
table.to_csv("reports/tuner_results.csv", index=False)
print(table.to_string(index=False))

best = tuner.get_best_hyperparameters(1)[0]
print("best learning_rate:", best.get("learning_rate"))
print("best dropout:", best.get("dropout"))
print("best unfrozen_layers:", best.get("unfrozen_layers"))
