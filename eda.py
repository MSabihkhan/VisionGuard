"""EDA figures: sample grid and histograms per class."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image

Path("reports/figures").mkdir(parents=True, exist_ok=True)
df = pd.read_csv("reports/audit.csv")
df = df[~df.corrupt]
classes = sorted(df.cls.unique())

# Figure 1: random sample grid, one row per class
N = 6
fig, axes = plt.subplots(len(classes), N, figsize=(2.6 * N, 2.8 * len(classes)))
for r, c in enumerate(classes):
    sample = df[df.cls == c].sample(N, random_state=1)
    for k, path in enumerate(sample.path):
        axes[r, k].imshow(Image.open(path).convert("RGB"))
        axes[r, k].axis("off")
    axes[r, 0].set_title(c, loc="left", fontsize=11)
plt.tight_layout()
plt.savefig("reports/figures/fig1_samples.png", dpi=150)
plt.close()

# Figure 2: histograms of width, height, file size per class
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, col in zip(ax, ["width", "height", "file_size"]):
    sns.histplot(data=df, x=col, hue="cls", bins=40, ax=a, log_scale=(col == "file_size"))
    a.set_title(col)
plt.tight_layout()
plt.savefig("reports/figures/fig2_histograms.png", dpi=150)
plt.close()
