from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

IMG = 160
BATCH = 32
SEED = 42

augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])


def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG, IMG))
    return img, label


def make_ds(csv, train=False):
    df = pd.read_csv(csv)
    ds = tf.data.Dataset.from_tensor_slices((df.path.tolist(), df.label.values.astype("float32")))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    if train:
        ds = ds.shuffle(len(df), seed=SEED)
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(SEED)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    ds = make_ds("splits/train.csv", train=True)
    images, labels = next(iter(ds))
    names = {0: "Real", 1: "AI"}
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))
    for ax, img, lab in zip(axes.ravel(), images, labels):
        ax.imshow(img.numpy().astype("uint8"))
        ax.set_title(names[int(lab)])
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("reports/figures/fig3_augmented_batch.png", dpi=150)
    print("batch shape:", images.shape, "pixel range:", float(images.numpy().min()), float(images.numpy().max()))
