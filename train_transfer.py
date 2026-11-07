import tensorflow as tf

from data import IMG, SEED, make_ds
from helpers import get_callbacks, plot_curves

tf.keras.utils.set_random_seed(SEED)

train_ds = make_ds("splits/train.csv", train=True)
val_ds = make_ds("splits/val.csv")

base = tf.keras.applications.MobileNetV2(input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet")
base.trainable = False

inputs = tf.keras.Input(shape=(IMG, IMG, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)


def save_trainable(stage):
    names = [layer.name for layer in base.layers if layer.trainable and layer.weights]
    names += [layer.name for layer in model.layers if layer is not base and layer.trainable and layer.weights]
    print(stage, "trainable layers:", len(names))
    print(names)
    with open(f"reports/trainable_{stage}.txt", "w") as f:
        f.write("\n".join(names))


model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
save_trainable("stage1")
history1 = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=get_callbacks("models/mobilenet_stage1.keras"), verbose=2)
plot_curves(history1, "mobilenet_stage1")

base.trainable = True
for layer in base.layers:
    layer.trainable = layer.name.startswith("block_16") or layer.name.startswith("Conv_1")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
save_trainable("stage2")
history2 = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=get_callbacks("models/mobilenet.keras"), verbose=2)
plot_curves(history2, "mobilenet_stage2")
