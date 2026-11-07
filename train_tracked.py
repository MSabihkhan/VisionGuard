import argparse
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from data import SEED, make_ds
from helpers import get_callbacks

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--approach", choices=["manual", "autolog"], default="manual")
parser.add_argument("--backbone", choices=["mobilenetv2", "efficientnetb0"], default="mobilenetv2")
parser.add_argument("--img", type=int, default=160)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--unfreeze", type=int, default=0)
parser.add_argument("--no-augment", action="store_true")
parser.add_argument("--epochs", type=int, default=25)
args = parser.parse_args()

tf.keras.utils.set_random_seed(SEED)

train_ds = make_ds("splits/train.csv", train=True, img=args.img, augment_on=not args.no_augment)
val_ds = make_ds("splits/val.csv", img=args.img)
test_ds = make_ds("splits/test.csv", img=args.img)
val = pd.read_csv("splits/val.csv")
test = pd.read_csv("splits/test.csv")

if args.backbone == "mobilenetv2":
    base = tf.keras.applications.MobileNetV2(input_shape=(args.img, args.img, 3), include_top=False, weights="imagenet")
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
else:
    base = tf.keras.applications.EfficientNetB0(input_shape=(args.img, args.img, 3), include_top=False, weights="imagenet")
    preprocess = tf.keras.applications.efficientnet.preprocess_input

base.trainable = False
if args.unfreeze > 0:
    for layer in base.layers[-args.unfreeze:]:
        layer.trainable = True

inputs = tf.keras.Input(shape=(args.img, args.img, 3))
x = preprocess(inputs)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(args.dropout)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss="binary_crossentropy", metrics=["accuracy"])


def git_commit():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def dvc_version():
    for line in open("data/raw.dvc"):
        if "md5:" in line:
            return line.split("md5:")[1].strip()


out_dir = Path("reports/tracked") / args.name
out_dir.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("visionguard")

if args.approach == "autolog":
    mlflow.tensorflow.autolog(log_models=False)

callbacks = get_callbacks(f"models/tracked_{args.name}.keras")

with mlflow.start_run(run_name=args.name):
    mlflow.set_tags({
        "git_commit": git_commit(),
        "dvc_data_version": dvc_version(),
        "seed": str(SEED),
        "approach": args.approach,
    })
    mlflow.log_params({
        "backbone": args.backbone,
        "img_size": args.img,
        "learning_rate": args.lr,
        "dropout": args.dropout,
        "unfrozen_layers": args.unfreeze,
        "augmentation": not args.no_augment,
        "max_epochs": args.epochs,
        "batch": 32,
        "seed": SEED,
    })

    if args.approach == "manual":
        callbacks.append(tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: mlflow.log_metrics(logs, step=epoch)
        ))

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=2)
    h = pd.DataFrame(history.history)
    best = int(np.argmin(h.val_loss))
    mlflow.log_metrics({"best_epoch": best + 1, "best_val_loss": float(h.val_loss[best])})

    val_prob = model.predict(val_ds, verbose=0).ravel()
    test_prob = model.predict(test_ds, verbose=0).ravel()
    thresholds = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(val.label, val_prob >= t) for t in thresholds]
    thr = float(thresholds[int(np.argmax(f1s))])
    pred = (test_prob >= thr).astype(int)

    mlflow.log_param("threshold", round(thr, 2))
    mlflow.log_metrics({
        "val_f1": float(max(f1s)),
        "val_roc_auc": float(roc_auc_score(val.label, val_prob)),
        "test_accuracy": accuracy_score(test.label, pred),
        "test_precision": precision_score(test.label, pred),
        "test_recall": recall_score(test.label, pred),
        "test_f1": f1_score(test.label, pred),
        "test_roc_auc": roc_auc_score(test.label, test_prob),
    })

    ConfusionMatrixDisplay(confusion_matrix(test.label, pred), display_labels=["Real", "AI"]).plot(cmap="Blues")
    plt.title(f"{args.name} confusion matrix")
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(test.label, test_prob)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title(f"{args.name} ROC curve")
    plt.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close()

    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (out_dir / "pip_freeze.txt").write_text(freeze)

    mlflow.log_artifact(str(out_dir / "confusion_matrix.png"))
    mlflow.log_artifact(str(out_dir / "roc_curve.png"))
    mlflow.log_artifact(str(out_dir / "pip_freeze.txt"))
    mlflow.log_artifact("splits/train.csv", "splits")
    mlflow.log_artifact("splits/val.csv", "splits")
    mlflow.log_artifact("splits/test.csv", "splits")
    mlflow.keras.log_model(model, name="model")

    print(args.name, "best epoch:", best + 1, "threshold:", round(thr, 2), "test f1:", round(f1_score(test.label, pred), 4), "test auc:", round(roc_auc_score(test.label, test_prob), 4))
