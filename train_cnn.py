import tensorflow as tf

from data import IMG, SEED, make_ds
from helpers import get_callbacks, plot_curves

tf.keras.utils.set_random_seed(SEED)

train_ds = make_ds("splits/train.csv", train=True)
val_ds = make_ds("splits/val.csv")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG, IMG, 3)),
    tf.keras.layers.Rescaling(1 / 255),
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(momentum=0.9),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(momentum=0.9),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(momentum=0.9),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()

history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=get_callbacks("models/cnn.keras"), verbose=2)
plot_curves(history, "cnn")
