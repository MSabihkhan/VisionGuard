from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

Path("models").mkdir(exist_ok=True)
Path("reports/figures").mkdir(parents=True, exist_ok=True)


def get_callbacks(path):
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]


def plot_curves(history, name):
    h = pd.DataFrame(history.history)
    h.to_csv(f"reports/history_{name}.csv", index=False)
    best = int(np.argmin(h.val_loss))
    print(name, "best epoch:", best + 1, "val_loss:", round(h.val_loss[best], 4), "val_accuracy:", round(h.val_accuracy[best], 4))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        ax.plot(h.index + 1, h[metric], label="train")
        ax.plot(h.index + 1, h["val_" + metric], label="val")
        ax.axvline(best + 1, color="red", linestyle="--", label=f"best epoch {best + 1}")
        ax.set_xlabel("epoch")
        ax.set_title(f"{name} {metric}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"reports/figures/curves_{name}.png", dpi=150)
    plt.close()
