import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

metrics = pd.read_csv("reports/metrics.csv")
best = metrics.sort_values("f1").iloc[-1].model
print("analysing model:", best)

preds = pd.read_csv(f"reports/test_preds_{best}.csv")
audit = pd.read_csv("reports/audit.csv")[["path", "width", "height", "file_size", "format", "mode"]]

wrong = preds[preds.pred != preds.label].copy()
wrong["confidence"] = (wrong.prob - wrong.label).abs()
wrong = wrong.sort_values("confidence", ascending=False).head(10)
wrong = wrong.merge(audit, on="path", how="left")
names = {0: "Real", 1: "AI"}

fig, axes = plt.subplots(2, 5, figsize=(16, 7))
for ax, (_, r) in zip(axes.ravel(), wrong.iterrows()):
    ax.imshow(Image.open(r.path).convert("RGB"))
    ax.set_title(f"true {names[r.label]} / pred {names[r.pred]}\nprob AI {r.prob:.2f}", fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("reports/figures/fig_misclassified.png", dpi=150)

print(wrong[["path", "label", "pred", "prob", "width", "height", "file_size", "format", "mode"]].to_string())
wrong.to_csv("reports/misclassified.csv", index=False)
