import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
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

from data import make_ds

models = {"cnn": "models/cnn.keras", "mobilenet": "models/mobilenet.keras"}

val = pd.read_csv("splits/val.csv")
test = pd.read_csv("splits/test.csv")
val_ds = make_ds("splits/val.csv")
test_ds = make_ds("splits/test.csv")

rows = []
plt.figure(figsize=(6, 5))
for name, path in models.items():
    model = tf.keras.models.load_model(path)
    val_prob = model.predict(val_ds, verbose=0).ravel()
    test_prob = model.predict(test_ds, verbose=0).ravel()

    thresholds = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(val.label, val_prob >= t) for t in thresholds]
    thr = float(thresholds[int(np.argmax(f1s))])

    pred = (test_prob >= thr).astype(int)
    cm = confusion_matrix(test.label, pred)
    rows.append({
        "model": name,
        "threshold": round(thr, 2),
        "accuracy": accuracy_score(test.label, pred),
        "precision": precision_score(test.label, pred),
        "recall": recall_score(test.label, pred),
        "f1": f1_score(test.label, pred),
        "roc_auc": roc_auc_score(test.label, test_prob),
        "real_called_fake": int(cm[0, 1]),
        "fake_called_real": int(cm[1, 0]),
    })

    ConfusionMatrixDisplay(cm, display_labels=["Real", "AI"]).plot(cmap="Blues")
    plt.title(f"{name} confusion matrix (threshold {thr:.2f})")
    plt.savefig(f"reports/figures/confusion_{name}.png", dpi=150)
    plt.close()

    plt.figure(1)
    fpr, tpr, _ = roc_curve(test.label, test_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC {rows[-1]['roc_auc']:.3f})")

    out = test[["path", "label"]].copy()
    out["prob"] = test_prob
    out["pred"] = pred
    out.to_csv(f"reports/test_preds_{name}.csv", index=False)

plt.figure(1)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title("ROC curve on test set")
plt.legend()
plt.savefig("reports/figures/roc_curves.png", dpi=150)
plt.close()

table = pd.DataFrame(rows).round(4)
table.to_csv("reports/metrics.csv", index=False)
print(table.to_string(index=False))
